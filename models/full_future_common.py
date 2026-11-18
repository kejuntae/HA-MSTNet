"""Shared layers, input alignment, and estimation for known future drivers."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from exp.training import fit_torch_model, initialize_seed, predict_torch_model
from models.base import BaseEstimator, FitResult
from util.model_features import THERMAL_FEATURES


@dataclass(frozen=True)
class FullFutureOptions:
    """Configure encoder width, heads, depth, and dropout."""

    d_model: int = 64
    heads: int = 4
    layers: int = 1
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("d_model", "heads", "layers"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.d_model % self.heads or self.d_model % 2:
            raise ValueError('d_model must be even and divisible by heads.')
        if (isinstance(self.dropout, bool)
                or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError('dropout must be in [0, 1).')


def positions(tokens, coordinates):
    """Add sinusoidal positions independent of sequence length."""
    width = tokens.shape[-1]
    frequency = torch.exp(torch.arange(
        0, width, 2, device=tokens.device, dtype=tokens.dtype,
    ) * (-math.log(10000.0) / width))
    angles = coordinates[:, None] * frequency
    embedding = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
    return tokens + embedding[None]


def readout(input_width, width, output_width=1):
    """Initialize the readout near zero to preserve the initial temperature baseline."""
    result = nn.Sequential(nn.Linear(input_width, width), nn.GELU(),
                           nn.Linear(width, output_width))
    nn.init.normal_(result[-1].weight, std=0.005)
    nn.init.zeros_(result[-1].bias)
    return result


class CrossReadout(nn.Module):
    """Read historical memory with future queries and a residual connection."""

    def __init__(self, options):
        super().__init__()
        width = options.d_model
        self.attention = nn.MultiheadAttention(
            width, options.heads, dropout=options.dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(),
            nn.Dropout(options.dropout), nn.Linear(2 * width, width),
        )

    def forward(self, query, memory):
        attended, _ = self.attention(query, memory, memory, need_weights=False)
        updated = self.norm1(query + attended)
        return self.norm2(updated + self.feedforward(updated))


class FullFutureNetwork(nn.Module):
    """Separate history from six future drivers and exclude future room temperature."""

    def __init__(self, context, options):
        super().__init__()
        self.options = options
        self.history_steps = context.history_steps
        self.horizon_steps = context.horizon_steps
        self.interval_minutes = context.interval_minutes
        self.interval_start_inputs = context.interval_start_inputs
        self.room_index = context.feature_index("room_temperature")
        self.power_index = context.feature_index("hvac_power")
        self.external_indices = tuple(i for i in range(7) if i != self.room_index)
        for key in ("minimum", "span"):
            values = getattr(context.scaler, key)
            self.register_buffer(key, torch.tensor([
                values[name] for name in context.feature_names
            ], dtype=torch.float32))

    def to_c(self, value):
        return value * self.span[self.room_index] + self.minimum[self.room_index]

    def to_scaled(self, value):
        return (value - self.minimum[self.room_index]) / self.span[self.room_index]

    def environment(self, inputs):
        """Align interval-start BOPTEST inputs and target-row NEST inputs."""
        if inputs.ndim != 3 or inputs.shape[1:] != (
            self.history_steps + self.horizon_steps, 7,
        ):
            raise ValueError('Input shape does not match the full-future model context.')
        history = inputs[:, :self.history_steps].clone()
        baseline = history[:, -1, self.room_index:self.room_index + 1]
        if self.interval_start_inputs:
            history[:, -1, self.external_indices] = (
                history[:, -2, self.external_indices]
                if self.history_steps > 1 else 0
            )
        start = self.history_steps - int(self.interval_start_inputs)
        future = inputs[:, start:start + self.horizon_steps,
                        self.external_indices].clone()
        history_raw = history * self.span + self.minimum
        external_raw = future * self.span[list(self.external_indices)]
        external_raw = external_raw + self.minimum[list(self.external_indices)]
        future_raw = torch.cat((torch.zeros_like(external_raw[:, :, :1]),
                                external_raw), -1)
        features = history.clone()
        features[:, :, self.room_index] -= baseline
        features[:, :, 1] = (
            history_raw[:, :, 1] - self.to_c(baseline)
        ) / self.span[self.room_index]
        drivers = future.clone()
        drivers[:, :, 0] = (
            external_raw[:, :, 0] - self.to_c(baseline)
        ) / self.span[self.room_index]
        lead = torch.arange(1, self.horizon_steps + 1,
                            device=inputs.device, dtype=inputs.dtype)
        hours = lead * self.interval_minutes / 60
        coordinates = torch.stack((
            hours / 24, torch.log1p(hours) / math.log(25),
            torch.sin(2 * math.pi * hours / 24),
            torch.cos(2 * math.pi * hours / 24), torch.exp(-hours / 2),
        ), -1)
        return dict(history=features, history_raw=history_raw,
                    future=drivers, future_raw=future_raw, baseline=baseline,
                    lead=lead, coordinates=coordinates)

    def batch_loss(self, inputs, targets):
        """Compute forecast MSE on the scaled temperature."""
        prediction = self(inputs)
        data = (prediction - targets).square().mean()
        return prediction, {"total": data, "data": data}

    def forward(self, inputs):
        return self.forward_components(inputs)["prediction"]


class FullFutureEstimator(BaseEstimator):
    """Share the fit interface and restore checkpoints with the model's option type."""

    required_features = THERMAL_FEATURES

    def __init__(self, **parameters):
        self.options = self.option_type(**parameters)
        self.network = None
        self.loss_components = ()

    def _build(self, context, seed):
        initialize_seed(seed)
        return self.network_type(context, self.options)

    def fit(self, train, validation, context, training_config, seed):
        """Fit using the common trainer and select weights by validation data MSE."""
        self.network = self._build(context, seed)
        outcome = fit_torch_model(
            self.network, train.scaled_inputs,
            context.scaler.transform_target(train.targets_c),
            validation.scaled_inputs,
            context.scaler.transform_target(validation.targets_c),
            training_config, seed,
            batch_loss_function=lambda model, x, y: model.batch_loss(x, y),
            validation_component='data',
        )
        self.loss_components = outcome.loss_components
        return FitResult(outcome.train_losses, outcome.validation_losses,
                         outcome.best_epoch, outcome.completed_epochs,
                         outcome.elapsed_seconds)

    def predict(self, windows, context, training_config):
        """Return room-temperature predictions in Celsius."""
        if self.network is None:
            raise ValueError('Fit the model or load weights first.')
        prediction = predict_torch_model(self.network, windows.scaled_inputs,
                                        training_config.batch_size, training_config.device)
        return context.scaler.inverse_target(prediction)

    def state_payload(self):
        """Save weights, architecture options, and alignment metadata."""
        if self.network is None:
            raise ValueError('The model has not been fitted.')
        return {'state_dict': self.network.state_dict(), 'options': asdict(self.options),
                'interval_start_inputs': self.network.interval_start_inputs,
                'history_steps': self.network.history_steps,
                'horizon_steps': self.network.horizon_steps}

    def load_payload(self, payload, context):
        for name in ("history_steps", "horizon_steps", "interval_start_inputs"):
            if payload[name] != getattr(context, name):
                raise ValueError(f"Checkpoint and context disagree on {name}.")
        self.options = self.option_type(**payload["options"])
        self.network = self._build(context, 0)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()

    def parameter_count(self):
        return 0 if self.network is None else sum(
            p.numel() for p in self.network.parameters() if p.requires_grad)

    def describe_architecture(self):
        return dict(name=self.name, inputs=list(self.required_features),
                    direction=self.direction, options=asdict(self.options),
                    loss="scaled_temperature_mse", validation_loss="data_mse",
                    future_temporal_policy="full_known_future",
                    future_temperature_visible=False, hard_monotonicity=False,
                    temperature_autoregression=False)
