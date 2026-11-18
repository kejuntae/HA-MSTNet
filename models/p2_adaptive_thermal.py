"""Adaptive physics mixing, separate physical context, and late fusion across scales."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.full_future_common import positions
from models.physical_patch_thermal import (
    PatchThermalEstimator, PatchThermalForecaster, PatchThermalOptions,
)


@dataclass(frozen=True)
class AdaptiveP2Options(PatchThermalOptions):
    """Configure temperature mixing, context pooling, and scale fusion independently."""

    loss_profile: str = "data"
    ramp_epochs: int = 0
    learn_physics_mix: bool = True
    physics_mix_initial: float = 0.5
    physics_context_pooling: str = "concat"
    history_fusion: str = "late_gated"
    scale_gate_width: int = 8

    def __post_init__(self):
        super().__post_init__()
        if type(self.learn_physics_mix) is not bool:
            raise ValueError('learn_physics_mix must be Boolean.')
        mix = self.physics_mix_initial
        if (isinstance(mix, bool) or not isinstance(mix, (int, float))
                or not math.isfinite(mix) or not 0 <= mix <= 1):
            raise ValueError('physics_mix_initial must be finite and in [0, 1].')
        if self.learn_physics_mix and not 0 < mix < 1:
            raise ValueError('A learnable mixture must start strictly within (0, 1).')
        if self.physics_context_pooling not in ("mean", "concat"):
            raise ValueError('physics_context_pooling must be mean or concat.')
        if self.history_fusion not in ("early", "late_mean", "late_gated"):
            raise ValueError('history_fusion must be early, late_mean, or late_gated.')
        if type(self.scale_gate_width) is not int or self.scale_gate_width < 1:
            raise ValueError('scale_gate_width must be a positive integer.')


class AdaptiveP2Forecaster(PatchThermalForecaster):
    """Control branch and scale contributions while retaining the three thermal modes."""

    def __init__(self, context, options):
        super().__init__(context, options)
        width = options.d_model
        scales = len(options.patch_lengths)
        # Initialize the shared layers first to preserve same-seed initialization.
        if options.physics_context_pooling == "concat" and scales > 1:
            self.thermal_context = nn.Linear(scales * width, options.physics_width)
        if options.history_fusion == "late_gated" and scales > 1:
            self.scale_gate = nn.Sequential(
                nn.Linear(scales * width, options.scale_gate_width), nn.GELU(),
                nn.Linear(options.scale_gate_width, scales),
            )
            # Start with equal scale weights and learn them without a balancing loss.
            nn.init.zeros_(self.scale_gate[-1].weight)
            nn.init.zeros_(self.scale_gate[-1].bias)
        else:
            self.scale_gate = None
        if options.learn_physics_mix:
            initial = math.log(options.physics_mix_initial / (1 - options.physics_mix_initial))
            self.physics_mix_logit = nn.Parameter(torch.tensor(initial))
        else:
            self.register_parameter("physics_mix_logit", None)

    def _patch_tokens(self, history):
        """Left-pad and unfold patches, using their centers as positions."""
        result = []
        for index, (length, projection) in enumerate(zip(
                self.options.patch_lengths, self.projections)):
            values = history
            padding = (-values.shape[1]) % length
            if padding:
                values = F.pad(values.transpose(1, 2), (padding, 0),
                               mode="replicate").transpose(1, 2)
            patches = values.unfold(1, length, length).permute(0, 1, 3, 2).flatten(2)
            tokens = projection(patches) + self.scale_embedding[index]
            centers = (torch.arange(tokens.shape[1], device=history.device,
                                    dtype=history.dtype) + .5) * length - values.shape[1]
            result.append(positions(tokens, centers))
        return result

    def encode_history(self, history):
        """Encode scales with shared weights and build a separate physics context."""
        if self.options.history_fusion == "early":
            memory, _, summaries = super().encode_history(history)
        else:
            memories = [self.history_encoder(tokens)
                        for tokens in self._patch_tokens(history)]
            summaries = torch.stack([memory.mean(1) for memory in memories], 1)
            memory = torch.cat(memories, 1)
        context = (summaries.flatten(1)
                   if self.options.physics_context_pooling == "concat"
                   else summaries.mean(1))
        # The returned context uses history only, including during holdout calibration.
        return memory, context, summaries

    def physics_mix(self, reference):
        """Use one global mixture weight independent of future power or targets."""
        if self.physics_mix_logit is not None:
            return self.physics_mix_logit.sigmoid()
        return reference.new_tensor(self.options.physics_mix_initial)

    def forward_components(self, inputs):
        env = self.environment(inputs)
        memory, context, summaries = self.encode_history(env["history"])
        future = self.future_encoder(positions(
            self.future_projection(env["future"]), env["lead"],
        ))
        result = {}
        if self.options.history_fusion == "early":
            features = self.cross(future, memory)
        else:
            counts = [math.ceil(env["history"].shape[1] / length)
                      for length in self.options.patch_lengths]
            # Each scale reads the same future queries with shared cross-attention weights.
            features_by_scale = torch.stack(
                [self.cross(future, part) for part in memory.split(counts, 1)], 2)
            if self.scale_gate is not None:
                weights = self.scale_gate(summaries.flatten(1)).softmax(-1)
            else:
                weights = inputs.new_full((len(inputs), len(counts)), 1 / len(counts))
            features = (features_by_scale * weights[:, None, :, None]).sum(2)
            result.update(scale_weights=weights,
                          gates=weights[:, None].expand(-1, future.shape[1], -1))
        neural = env["baseline"] + self.head(features).squeeze(-1)
        physical = self.thermal(
            env["history_raw"], env["future_raw"], self.thermal_context(context),
            self.interval_minutes / 60, self.interval_start_inputs,
        )
        alpha = self.physics_mix(inputs)
        prediction = (1 - alpha) * neural + alpha * self.to_scaled(physical["prediction_c"])
        prediction_c = self.to_c(prediction)
        result.update(
            prediction=prediction, prediction_c=prediction_c,
            neural_temperature_c=self.to_c(neural), prior_c=physical["prediction_c"],
            correction_c=prediction_c - physical["prediction_c"],
            physics_mix_weight=alpha.expand_as(prediction),
            physical_states_c=physical["states_c"], states_c=physical["states_c"],
            equilibrium_c=physical["equilibrium_c"],
            thermal_hours=physical["hours"], thermal_retention=physical["retention"],
            thermal_gains=physical["gains"], thermal_lag_hours=physical["lag_hours"],
            mode_weights=physical["weights"], initial_states_c=physical["initial"],
            physics_confidence=physical["confidence"], scale_summaries=summaries,
        )
        return result


class AdaptiveP2Estimator(PatchThermalEstimator):
    """Describe adaptive fusion and preserve physical loss and diagnostic settings."""

    option_type = AdaptiveP2Options
    network_type = AdaptiveP2Forecaster

    def describe_architecture(self):
        result = super().describe_architecture()
        opt = self.options
        late = opt.history_fusion != "early"
        result.update(
            fusion="learned_temperature_mix" if opt.learn_physics_mix else "temperature_mix",
            fixed_physics_mix=None if opt.learn_physics_mix else opt.physics_mix_initial,
            physics_mix_policy="global_learned_scalar" if opt.learn_physics_mix else "fixed",
            physics_mix_initial=opt.physics_mix_initial,
            history_pooling=opt.physics_context_pooling,
            physics_context_input_width=(len(opt.patch_lengths) * opt.d_model
                                         if opt.physics_context_pooling == "concat" else opt.d_model),
            history_fusion=opt.history_fusion,
            history_encoder_shared_across_scales=True,
            history_encoder_calls=len(opt.patch_lengths) if late else 1,
            cross_readout_shared_across_scales=True,
            cross_readout_calls=len(opt.patch_lengths) if late else 1,
            scale_gate_input="history_summaries_only" if opt.history_fusion == "late_gated" else None,
            scale_weights_constant_within_forecast=True if late else None,
            physics_confidence_used_for_fusion=False,
            initialization_policy="original_P2_common_modules_initialized_first",
        )
        return result
