"""Shared physics losses, power perturbations, and checkpoint handling."""

from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import nn

from models.full_future_common import FullFutureEstimator
from models.registry import canonical_model_name


@dataclass(frozen=True)
class PhysicsLossOptions:
    """Normalize temperature errors by the training room-temperature span."""

    loss_profile: str = "full"
    history_weight: float = 0.1
    control_weight: float = 0.1
    balance_weight: float = 0.1
    identification_weight: float = 0.1
    ramp_epochs: int = 5
    holdout_steps: int = 4
    physical_batch_size: int = 8
    control_fraction: float = 0.05

    def validate_physics(self, profiles):
        if self.loss_profile not in profiles:
            raise ValueError(f"loss_profile must be one of {profiles}.")
        for name in ("history_weight", "control_weight", "balance_weight",
                     "identification_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                    not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("holdout_steps", "physical_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.ramp_epochs) is not int or self.ramp_epochs < 0:
            raise ValueError('ramp_epochs must be a nonnegative integer.')
        if not 0 < self.control_fraction < 1:
            raise ValueError('control_fraction must be in (0, 1).')


@contextmanager
def deterministic_pair(network):
    """Disable dropout for paired forecasts without disabling gradients."""
    modes = [(module, module.training) for module in network.modules()]
    try:
        network.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def control_plan(network, inputs, sign, *, randomize=False):
    """Apply signed segment perturbations within the training power range and mask from the first change."""
    batch = len(inputs)
    horizon = network.horizon_steps
    device = inputs.device
    start = network.history_steps - int(network.interval_start_inputs)
    original = inputs[:, start:start + horizon, 3]
    if randomize:
        onset = torch.randint(0, horizon, (batch,), device=device)
        duration = torch.randint(1, max(2, horizon // 2 + 1), (batch,), device=device)
    else:
        onset = (torch.arange(batch, device=device) * 7 + horizon // 3) % horizon
        duration = torch.full_like(onset, max(1, horizon // 4))
    lead = torch.arange(horizon, device=device)[None]
    requested = (lead >= onset[:, None]) & (lead < (onset + duration)[:, None])
    bounded = (original >= 0) & (original <= 1)
    fraction = getattr(network.options, "control_fraction", 0.05)
    candidate = (original + sign * fraction).clamp(0, 1)
    changed_q = torch.where(requested & bounded, candidate, original)
    actual = (changed_q - original).abs() > 1e-7
    first = torch.where(actual, lead.expand(batch, -1), horizon).min(1).values
    eligible = first < horizon
    before = (lead < first[:, None]) & eligible[:, None]
    after = (lead >= first[:, None]) & eligible[:, None]
    changed = inputs.clone()
    changed[:, start:start + horizon, 3] = changed_q
    return changed, before, after, actual


def masked_mean(value, mask):
    """Return a differentiable zero for an empty mask."""
    return (value * mask).sum() / mask.sum().clamp_min(1)


class PhysicsObjectiveMixin:
    """Compute only enabled losses and report their raw and weighted values."""

    auxiliary_losses = ()

    def set_training_epoch(self, epoch):
        self.training_epoch = epoch

    def regularization_factor(self):
        ramp = self.options.ramp_epochs
        return min(1.0, getattr(self, "training_epoch", 1) / ramp) if ramp else 1.0

    def control_loss(self, inputs, prediction=None):
        selected = inputs[:self.options.physical_batch_size]
        terms = []
        randomize = self.training
        with deterministic_pair(self):
            baseline = self(selected)
            for sign in (1, -1):
                changed, before, after, _ = control_plan(
                    self, selected, sign, randomize=randomize,
                )
                delta = self(changed) - baseline
                terms.append(masked_mean(delta.square(), before)
                             + masked_mean(torch.relu(-sign * delta).square(), after))
        return torch.stack(terms).mean()

    def batch_loss(self, inputs, targets):
        prediction = self(inputs)
        data = (prediction - targets).square().mean()
        factor = self.regularization_factor()
        terms = {"data": data, "data_weight": data.new_tensor(1),
                 "data_weighted": data}
        total = data
        for name in self.auxiliary_losses:
            value = getattr(self, f"{name}_loss")(inputs, prediction=prediction)
            weight = factor * getattr(self.options, f"{name}_weight")
            terms[name] = value
            terms[f"{name}_weight"] = data.new_tensor(weight)
            terms[f"{name}_weighted"] = value * weight
            total = total + value * weight
        terms.update(total=total, regularization_factor=data.new_tensor(factor))
        return prediction, terms


class PhysicsFutureEstimator(FullFutureEstimator):
    """Check model identity, training scale, and alignment before loading weights."""

    def state_payload(self):
        payload = super().state_payload()
        payload.update(
            model_identity=self.name, physics_version=1,
            interval_minutes=self.network.interval_minutes,
            minimum=self.network.minimum.detach().cpu().tolist(),
            span=self.network.span.detach().cpu().tolist(),
            calibration=getattr(self, "calibration", {}),
        )
        return payload

    def load_payload(self, payload, context):
        identity = canonical_model_name(payload.get("model_identity", ""))
        if identity != self.name or payload.get("physics_version") != 1:
            raise ValueError('The physics checkpoint has an incompatible model identity or version.')
        if payload.get("interval_minutes") != context.interval_minutes:
            raise ValueError('The physics checkpoint has a different sampling interval.')
        for key in ("minimum", "span"):
            expected = torch.tensor([getattr(context.scaler, key)[name]
                                     for name in context.feature_names])
            if not torch.equal(torch.tensor(payload[key]), expected):
                raise ValueError(f"The physics checkpoint training scaler has a different {key}.")
        super().load_payload(payload, context)
        self.calibration = payload.get("calibration", {})

    def describe_architecture(self):
        description = super().describe_architecture()
        description.update(
            loss_profile=self.options.loss_profile,
            physical_normalization="training_temperature_span",
            physics_version=1, loss="data_mse_plus_enabled_physics",
        )
        return description
