"""Ordered time constants and numerically stable thermal filters."""

import torch
from torch import nn


def parameter_head(input_width, width, output_width):
    """Initialize a small head for physical parameters, not temperature outputs."""
    head = nn.Sequential(
        nn.Linear(input_width, width), nn.GELU(),
        nn.Linear(width, output_width),
    )
    nn.init.normal_(head[-1].weight, std=0.001)
    nn.init.zeros_(head[-1].bias)
    return head


class OrderedTimeScales(nn.Module):
    """Use positive spacings and slack to enforce ordering and the upper bound."""

    def __init__(self, hours, minimum, maximum):
        super().__init__()
        gaps = torch.diff(torch.tensor((minimum, *hours, maximum)))
        self.log_gaps = nn.Parameter(gaps.log())
        self.minimum = minimum
        self.maximum = maximum

    def forward(self, adjustments):
        fractions = 1e-5 + (1 - 4e-5) * (
            self.log_gaps + adjustments
        ).softmax(-1)
        return self.minimum + (self.maximum - self.minimum) * (
            fractions.cumsum(-1)[..., :3]
        )


def stable_filter(values, retention, initial):
    """Unroll a first-order recurrence using nonnegative kernels and no negative powers."""
    steps = values.shape[1]
    positions = torch.arange(steps, device=values.device)
    lag = positions[:, None] - positions[None, :]
    kernel = torch.exp(
        retention.log()[:, None, None, :] * lag.clamp_min(0)[None, :, :, None]
    ) * (lag >= 0)[None, :, :, None] * (1 - retention[:, None, None, :])
    filtered = torch.einsum("bjks,bksc->bjsc", kernel, values)
    decay = retention[:, None, :, None].pow(
        (positions + 1)[None, :, None, None]
    )
    return filtered + decay * initial[:, None]
