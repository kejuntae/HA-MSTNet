"""Shared multiscale encoding and thermal forecasting layers for HA-MSTNet."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.full_future_common import (
    CrossReadout, FullFutureNetwork, positions, readout,
)
from models.multiresolution_patch_transformer import PatchFutureOptions
from models.nest_enhanced_gated_common import EnhancementOptions, ThermalContinuation
from models.physics_future_common import (
    PhysicsFutureEstimator, PhysicsLossOptions, PhysicsObjectiveMixin,
)


@dataclass(frozen=True)
class PatchThermalOptions(PatchFutureOptions, PhysicsLossOptions):
    """Configure patch capacity and auxiliary losses."""

    d_model: int = 32
    feedforward_ratio: int = 3
    physics_width: int = 16
    disturbance_bound_c: float = 2.0

    def __post_init__(self):
        PatchFutureOptions.__post_init__(self)
        self.validate_physics(("data", "history", "control", "full"))
        for name in ("feedforward_ratio", "physics_width"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.physics_width % self.heads:
            raise ValueError('physics_width must be divisible by heads.')
        bound = self.disturbance_bound_c
        if (isinstance(bound, bool) or not isinstance(bound, (int, float))
                or not math.isfinite(bound) or bound <= 0):
            raise ValueError('disturbance_bound_c must be a finite positive number.')


def patch_encoder(options):
    """Use a configurable feedforward width, three times the embedding width by default."""
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            options.d_model, options.heads,
            dim_feedforward=options.feedforward_ratio * options.d_model,
            dropout=options.dropout, activation="gelu", batch_first=True,
        ), options.layers, enable_nested_tensor=False,
    )


class PatchThermalForecaster(PhysicsObjectiveMixin, FullFutureNetwork):
    """Shared multiscale patch encoder and physics calibration."""

    def __init__(self, context, options):
        super().__init__(context, options)
        if context.history_steps < 2:
            raise ValueError('Physics calibration requires at least two history points.')
        width = options.d_model
        self.projections = nn.ModuleList([
            nn.Linear(length * 7, width) for length in options.patch_lengths
        ])
        self.scale_embedding = nn.Parameter(
            torch.randn(len(options.patch_lengths), width) * 0.02,
        )
        self.history_encoder = patch_encoder(options)
        self.future_projection = nn.Linear(len(self.external_indices), width)
        self.future_encoder = patch_encoder(options)
        self.cross = CrossReadout(options)
        self.head = readout(width, width)
        self.thermal_context = (
            nn.Identity() if width == options.physics_width
            else nn.Linear(width, options.physics_width)
        )
        self.thermal = ThermalContinuation(options.physics_width, EnhancementOptions(
            width=options.physics_width, attention_heads=options.heads,
        ))
        self.auxiliary_losses = {
            "data": (), "history": ("history",), "control": ("control",),
            "full": ("history", "control"),
        }[options.loss_profile]

    def encode_history(self, history):
        """Encode all tokens together, then weight each scale equally."""
        tokens_by_scale, counts = [], []
        for index, (length, projection) in enumerate(zip(
            self.options.patch_lengths, self.projections,
        )):
            values = history
            padding = (-values.shape[1]) % length
            if padding:
                values = F.pad(values.transpose(1, 2), (padding, 0),
                               mode="replicate").transpose(1, 2)
            patches = values.unfold(1, length, length).permute(0, 1, 3, 2).flatten(2)
            tokens = projection(patches) + self.scale_embedding[index]
            centers = (torch.arange(tokens.shape[1], device=history.device,
                                    dtype=history.dtype) + 0.5) * length - values.shape[1]
            tokens_by_scale.append(positions(tokens, centers))
            counts.append(tokens.shape[1])
        memory = self.history_encoder(torch.cat(tokens_by_scale, 1))
        summaries = torch.stack([part.mean(1) for part in memory.split(counts, 1)], 1)
        return memory, summaries.mean(1), summaries

    def history_calibration(self, inputs):
        """Encode only the prefix; use held-out room temperatures as targets."""
        raw = self.environment(inputs)["history_raw"]
        holdout = min(self.options.holdout_steps, raw.shape[1] - 1)
        prefix = raw[:, :-holdout]
        features = (prefix - self.minimum) / self.span
        features = features.clone()
        features[:, :, 0] = (prefix[:, :, 0] - prefix[:, -1:, 0]) / self.span[0]
        features[:, :, 1] = (prefix[:, :, 1] - prefix[:, -1:, 0]) / self.span[0]
        _, context, _ = self.encode_history(features)
        start = prefix.shape[1] - int(self.interval_start_inputs)
        drives = raw[:, start:start + holdout].clone()
        drives[:, :, 0] = 0
        physical = self.thermal(
            prefix, drives, self.thermal_context(context),
            self.interval_minutes / 60, self.interval_start_inputs,
        )
        return physical["prediction_c"], raw[:, -holdout:, 0]

    def history_loss(self, inputs, prediction=None):
        selected = inputs[:self.options.physical_batch_size]
        values, targets = self.history_calibration(selected)
        return ((values - targets) / self.span[0]).square().mean()


class PatchThermalEstimator(PhysicsFutureEstimator):
    """Share physics checkpoint identity, scaling, and alignment."""

    def describe_architecture(self):
        description = super().describe_architecture()
        description.update(
            encoder_feedforward_width=self.options.feedforward_ratio * self.options.d_model,
            patch_lengths=list(self.options.patch_lengths),
            history_pooling="equal_mean_of_scale_means",
            physical_module_count=1,
            thermal_modes=3,
            physics_width=self.options.physics_width,
            future_power_in_neural_path=True,
            neural_future_feature_count=6,
            future_temporal_policy="full_known_future",
            power_path_policy="neural_and_physics",
            physical_state_recurrence=True,
            disturbance_bound_c=None,
        )
        return description
