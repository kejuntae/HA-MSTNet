"""Shared thermal dynamics in physical units, with confidence weighting."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.gated_common import GatedAttentionOptions
from models.thermal_utils import (
    OrderedTimeScales,
    parameter_head,
    stable_filter,
)


@dataclass(frozen=True)
class EnhancementOptions(GatedAttentionOptions):
    """Configure physical losses and Celsius temperature differences."""

    warmup_epochs: int = 0
    heat_weight: float = 0.0
    physical_batch_size: int = 8
    direction_weight: float = 0.05
    balance_weight: float = 0.03
    reconstruction_weight: float = 0.02
    state_consistency_weight: float = 0.02
    disturbance_weight: float = 0.01
    confidence_floor: float = 0.25
    confidence_scale_c: float = 0.5
    off_power_w: float = 50.0
    thermal_hours: tuple[float, ...] = (1.0, 6.0, 24.0)
    minimum_hours: float = 0.125
    maximum_hours: float = 168.0
    lag_hours: float = 0.5
    minimum_lag_hours: float = 0.05
    maximum_lag_hours: float = 12.0
    initial_gains: tuple[float, ...] = (8.0, 8.0, 5.0, 0.5)
    maximum_gain: float = 40.0
    offset_bound_c: float = 8.0
    disturbance_bound_c: float = 0.5
    prior_mix: float = 0.15
    routing_temperature_c2: float = 0.5
    segment_steps: int = 4

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "direction_weight",
            "balance_weight",
            "reconstruction_weight",
            "state_consistency_weight",
            "disturbance_weight",
        ):
            self.validate_real(name, nonnegative=True)
        for name in (
            "confidence_scale_c",
            "off_power_w",
            "minimum_hours",
            "maximum_hours",
            "minimum_lag_hours",
            "maximum_lag_hours",
            "maximum_gain",
            "offset_bound_c",
            "disturbance_bound_c",
            "routing_temperature_c2",
        ):
            self.validate_real(name, positive=True)
        for name in ("confidence_floor", "prior_mix"):
            self.validate_real(name, positive=True)
            if getattr(self, name) >= 1:
                raise ValueError(f"{name} must be less than 1.")
        if self.warmup_epochs != 0:
            raise ValueError('Physical penalties must apply from the first epoch.')
        if self.heat_weight or self.shift_weight or self.equilibrium_weight:
            raise ValueError(
                'Use the new physics losses; legacy heating and static balance losses must be zero.'
            )
        if type(self.segment_steps) is not int or self.segment_steps < 1:
            raise ValueError('segment_steps must be a positive integer.')
        if len(self.history_scales) != 3:
            raise ValueError('Exactly three history scales are required.')
        hours = tuple(self.thermal_hours)
        if (
            len(hours) != 3
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in hours
            )
            or not self.minimum_hours
            < hours[0]
            < hours[1]
            < hours[2]
            < self.maximum_hours
        ):
            raise ValueError('thermal_hours must be strictly increasing within the bounds.')
        object.__setattr__(self, "thermal_hours", hours)
        self.validate_real("lag_hours", positive=True)
        if (
            not self.minimum_lag_hours
            < self.lag_hours
            < self.maximum_lag_hours
        ):
            raise ValueError('lag_hours must be within the bounds.')
        gains = tuple(self.initial_gains)
        if len(gains) != 4 or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 < v < self.maximum_gain
            for v in gains
        ):
            raise ValueError('initial_gains must contain four positive gains within the bounds.')
        object.__setattr__(self, "initial_gains", gains)


class ThermalContinuation(nn.Module):
    """Use the same stable decay and signed heat sources for history and forecasting."""

    def __init__(self, width, options):
        super().__init__()
        self.options = options
        self.hours = OrderedTimeScales(
            options.thermal_hours,
            options.minimum_hours,
            options.maximum_hours,
        )
        self.time_head = parameter_head(width, width, 4)
        self.gain_head = parameter_head(width, width, 12)
        gain = torch.tensor(options.initial_gains) / options.maximum_gain
        self.gain_logits = nn.Parameter(gain.logit().repeat(3, 1))
        self.lag_head = parameter_head(width, width, 3)
        fraction = (options.lag_hours - options.minimum_lag_hours) / (
            options.maximum_lag_hours - options.minimum_lag_hours
        )
        self.lag_logits = nn.Parameter(
            torch.full((3,), math.log(fraction / (1 - fraction)))
        )
        self.weights_head = parameter_head(width, width, 3)
        self.offset_head = parameter_head(width, width, 3)
        self.disturbance_head = parameter_head(width, width, 3)
        self.solar_head = parameter_head(width + 2, width, 1)

    def parameters_from_history(self, context):
        opt = self.options
        hours = self.hours(self.time_head(context))
        weights = self.weights_head(context).softmax(-1)
        offsets = opt.offset_bound_c * self.offset_head(context).tanh()
        offsets = offsets - (offsets * weights).sum(-1, keepdim=True)
        gains = (
            opt.maximum_gain
            * (
                self.gain_logits[None]
                + self.gain_head(context).reshape(-1, 3, 4)
            ).sigmoid()
        )
        lag_hours = (
            opt.minimum_lag_hours
            + (opt.maximum_lag_hours - opt.minimum_lag_hours)
            * (self.lag_logits[None] + self.lag_head(context)).sigmoid()
        )
        disturbance = (
            opt.disturbance_bound_c * self.disturbance_head(context).tanh()
        )
        return dict(
            hours=hours,
            weights=weights,
            offsets=offsets,
            gains=gains,
            lag_hours=lag_hours,
            disturbance=disturbance,
            context=context,
        )

    def sources(self, raw, params):
        """Convert to kW and kW/m²; the absorption gate uses history and time only."""
        batch, steps = raw.shape[:2]
        clock = raw[:, :, 4:6]
        transmission = self.solar_head(
            torch.cat(
                (
                    params["context"][:, None].expand(-1, steps, -1),
                    clock,
                ),
                -1,
            )
        ).sigmoid()
        q = raw[:, :, 3] / 1000
        sources = torch.stack(
            (
                q.clamp_min(0),
                (-q).clamp_min(0),
                raw[:, :, 2].clamp_min(0) / 1000 * transmission.squeeze(-1),
                raw[:, :, 6].clamp_min(0),
            ),
            -1,
        )
        return sources[:, :, None].expand(batch, steps, 3, 4), transmission

    @staticmethod
    def equilibrium(raw, filtered_sources, params):
        signs = raw.new_tensor((1, -1, 1, 1))
        forcing = (filtered_sources * params["gains"][:, None] * signs).sum(-1)
        return raw[:, :, 1:2] + forcing + params["disturbance"][:, None]

    def forward(self, history_raw, future_raw, context, dt_hours, alignment):
        params = self.parameters_from_history(context)
        retention = torch.exp(-dt_hours / params["hours"])
        lag_retention = torch.exp(-dt_hours / params["lag_hours"])
        history_sources, _ = self.sources(history_raw, params)
        initial = history_raw[:, 0, :1] + params["offsets"]
        if history_raw.shape[1] > 1:
            # Align BOPTEST interval-start inputs and NEST target-row inputs separately.
            selected = history_raw[:, :-1] if alignment else history_raw[:, 1:]
            drives = (
                history_sources[:, :-1]
                if alignment
                else history_sources[:, 1:]
            )
            filtered = stable_filter(
                drives, lag_retention, history_sources[:, 0]
            )
            equilibrium = self.equilibrium(selected, filtered, params)
            states = stable_filter(
                equilibrium[..., None],
                retention,
                initial[..., None],
            ).squeeze(-1)
            reconstruction = (states * params["weights"][:, None]).sum(-1)
            history_error = reconstruction - history_raw[:, 1:, 0]
            lag_initial = filtered[:, -1]
            initial = states[:, -1]
        else:
            history_error = history_raw[:, :, 0] * 0
            lag_initial = history_sources[:, 0]
        baseline = history_raw[:, -1, :1]
        # Anchor to the last observed temperature while preserving differences between thermal modes.
        initial = (
            initial
            + baseline
            - (initial * params["weights"]).sum(-1, keepdim=True)
        )
        sources, transmission = self.sources(future_raw, params)
        filtered = stable_filter(sources, lag_retention, lag_initial)
        equilibrium = self.equilibrium(future_raw, filtered, params)
        states = stable_filter(
            equilibrium[..., None],
            retention,
            initial[..., None],
        ).squeeze(-1)
        prediction_c = (states * params["weights"][:, None]).sum(-1)
        floor = self.options.confidence_floor
        confidence = floor + (1 - floor) * torch.exp(
            -history_error.detach().abs().mean(1)
            / self.options.confidence_scale_c
        )
        return {
            **params,
            "retention": retention,
            "initial": initial,
            "states_c": states,
            "equilibrium_c": equilibrium,
            "prediction_c": prediction_c,
            "history_error_c": history_error,
            "confidence": confidence,
            "solar_transmission": transmission,
            "power_w": future_raw[:, :, 3],
        }
