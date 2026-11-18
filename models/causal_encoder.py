"""Encoder settings and validation used by the thermal module."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CausalEncoderOptions:
    """Configure fixed history scales and a horizon-independent readout."""

    width: int = 32
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    history_scales: tuple[int, ...] = (8, 24, 48)
    history_bins: int = 8
    dropout: float = 0.05
    trend_decay_steps: float = 6.0
    near_weight: float = 0.5
    near_steps: int = 4

    def __post_init__(self):
        for name in ('width', 'history_bins', 'near_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer.')
        for name in ('dilations', 'history_scales'):
            values = tuple(getattr(self, name))
            if not values or any(type(v) is not int or v < 1 for v in values):
                raise ValueError(f'{name} must be a nonempty sequence of positive integers.')
            object.__setattr__(self, name, values)
        for name in ('dropout', 'trend_decay_steps', 'near_weight'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value)):
                raise ValueError(f'{name} must be a finite real number.')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1).')
        if self.trend_decay_steps <= 0 or self.near_weight < 0:
            raise ValueError('Trend decay must be positive and near-term weight must be nonnegative.')


@dataclass(frozen=True)
class AttentionEncoderOptions(CausalEncoderOptions):
    """Extend the encoder settings with attention options."""

    width: int = 64
    attention_heads: int = 4
    attention_layers: int = 1
    attention_residual_scale: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        for name in ('attention_heads', 'attention_layers'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.width % self.attention_heads:
            raise ValueError('width must be divisible by attention_heads.')
        scale = self.attention_residual_scale
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or scale < 0):
            raise ValueError('attention_residual_scale must be finite and nonnegative.')
