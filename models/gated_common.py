"""Gated-model settings retained by the thermal module."""

from dataclasses import dataclass
import math

from models.causal_encoder import AttentionEncoderOptions


@dataclass(frozen=True)
class GatedAttentionOptions(AttentionEncoderOptions):
    """Validate loss and heat-response settings used with training-based scaling."""

    heat_weight: float = 0.1
    heat_fraction: float = 0.05
    physical_batch_size: int = 16
    heat_hours: tuple[float, ...] = (0.25, 1.0, 4.0, 12.0)
    warmup_epochs: int = 5
    ramp_epochs: int = 5
    shift_weight: float = 0.0
    shift_delta_c: float = 0.5
    equilibrium_weight: float = 0.0
    event_scale: float = 0.1
    event_lags: int = 4

    def __post_init__(self):
        super().__post_init__()
        for name in ("physical_batch_size", "ramp_epochs", "event_lags"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.warmup_epochs) is not int or self.warmup_epochs < 0:
            raise ValueError('warmup_epochs must be a nonnegative integer.')
        for name in ("heat_weight", "shift_weight", "equilibrium_weight"):
            self.validate_real(name, nonnegative=True)
        for name in ("heat_fraction", "shift_delta_c", "event_scale"):
            self.validate_real(name, positive=True)
        if self.heat_fraction > 1:
            raise ValueError('heat_fraction must not exceed the training power span.')
        hours = tuple(self.heat_hours)
        if not hours or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0 for value in hours):
            raise ValueError('heat_hours must contain finite positive numbers.')
        object.__setattr__(self, "heat_hours", hours)

    def validate_real(self, name, *, nonnegative=False, positive=False):
        """Reject booleans, nonfinite values, and invalid weight ranges."""
        value = getattr(self, name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (nonnegative and value < 0) or (positive and value <= 0)):
            raise ValueError(f"{name} is out of range.")
