"""Multi-step MLP baseline using history and known future drivers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from exp.training import fit_torch_model, initialize_seed, predict_torch_model
from models.base import BaseEstimator, FitResult, ModelContext
from models.registry import CANONICAL_FEATURE_ORDER
from util.config import TrainingConfig
from util.features import ThermalWindows


def _validate_architecture(
    hidden_sizes: Sequence[int],
    dropout: float,
) -> tuple[tuple[int, ...], float]:
    """Validate the two or three hidden layers and dropout rate."""
    if isinstance(hidden_sizes, (str, bytes)):
        raise ValueError('hidden_sizes must contain two or three positive integers.')
    try:
        sizes = tuple(hidden_sizes)
    except TypeError as error:
        raise ValueError('hidden_sizes must contain two or three positive integers.') from error
    if len(sizes) not in (2, 3) or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 1
        for size in sizes
    ):
        raise ValueError('hidden_sizes must contain two or three positive integers.')
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not math.isfinite(float(dropout))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise ValueError('dropout must be in [0, 1).')
    return sizes, float(dropout)


class MLPForecaster(nn.Module):
    """Flatten history and future drivers to predict the whole horizon at once."""

    def __init__(
        self,
        feature_count: int,
        history_steps: int,
        horizon_steps: int,
        room_index: int,
        hidden_sizes: Sequence[int] = (128, 128),
        dropout: float = 0.1,
        interval_start_inputs: bool = False,
    ) -> None:
        super().__init__()
        for value, name in (
            (feature_count, "feature_count"),
            (history_steps, "history_steps"),
            (horizon_steps, "horizon_steps"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            not isinstance(room_index, int)
            or isinstance(room_index, bool)
            or not 0 <= room_index < feature_count
        ):
            raise ValueError('room_index is outside the feature range.')
        if not isinstance(interval_start_inputs, bool):
            raise ValueError('interval_start_inputs must be Boolean.')
        sizes, dropout = _validate_architecture(hidden_sizes, dropout)

        self.feature_count = feature_count
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.room_index = room_index
        self.hidden_sizes = sizes
        self.dropout = dropout
        self.interval_start_inputs = interval_start_inputs

        # Keep all history features and known future drivers only.
        input_width = history_steps * feature_count + horizon_steps * (
            feature_count - 1
        )
        widths = (input_width, *sizes, horizon_steps)
        layers: list[nn.Module] = []
        for input_size, output_size in zip(widths[:-2], widths[1:-1]):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(widths[-2], widths[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return multi-step room temperatures in scaled space."""
        expected_steps = self.history_steps + self.horizon_steps
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 3
            or inputs.shape[1] != expected_steps
            or inputs.shape[2] != self.feature_count
        ):
            raise ValueError(
                'MLP input shape must be '
                f"[batch, {expected_steps}, {self.feature_count}]."
            )

        history = inputs[:, :self.history_steps, :].reshape(inputs.shape[0], -1)
        start = self.history_steps - int(self.interval_start_inputs)
        future = inputs[:, start:start + self.horizon_steps, :]
        # Never use future room temperature as an input.
        future_exogenous = torch.cat(
            (future[:, :, :self.room_index],
             future[:, :, self.room_index + 1:]),
            dim=2,
        ).reshape(inputs.shape[0], -1)
        features = torch.cat((history, future_exogenous), dim=1)
        # Predict a residual at each lead time relative to the last measured temperature.
        baseline = inputs[:, self.history_steps - 1, self.room_index].unsqueeze(1)
        return baseline + self.network(features)


class MLPEstimator(BaseEstimator):
    """Adapt the MLP to the shared estimator interface."""

    name = "mlp"
    required_features = CANONICAL_FEATURE_ORDER

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        self.hidden_sizes, self.dropout = _validate_architecture(
            hidden_sizes, dropout
        )
        self.network: MLPForecaster | None = None

    def _build(self, context: ModelContext, seed: int) -> MLPForecaster:
        """Build a network matching the forecast horizon in the context."""
        initialize_seed(seed)
        return MLPForecaster(
            feature_count=len(context.feature_names),
            history_steps=context.history_steps,
            horizon_steps=context.horizon_steps,
            room_index=context.feature_index("room_temperature"),
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            interval_start_inputs=context.interval_start_inputs,
        )

    def fit(
        self,
        train: ThermalWindows,
        validation: ThermalWindows,
        context: ModelContext,
        training_config: TrainingConfig,
        seed: int,
    ) -> FitResult:
        """Train and restore the lowest-validation-loss weights."""
        self.network = self._build(context, seed)
        outcome = fit_torch_model(
            self.network,
            train.scaled_inputs,
            context.scaler.transform_target(train.targets_c),
            validation.scaled_inputs,
            context.scaler.transform_target(validation.targets_c),
            training_config,
            seed,
        )
        return FitResult(
            outcome.train_losses,
            outcome.validation_losses,
            outcome.best_epoch,
            outcome.completed_epochs,
            outcome.elapsed_seconds,
        )

    def predict(
        self,
        windows: ThermalWindows,
        context: ModelContext,
        training_config: TrainingConfig,
    ) -> np.ndarray:
        """Predict in batches and return Celsius."""
        if self.network is None:
            raise ValueError('Fit the MLP first.')
        scaled = predict_torch_model(
            self.network,
            windows.scaled_inputs,
            training_config.batch_size,
            training_config.device,
        )
        return context.scaler.inverse_target(scaled)

    def state_payload(self) -> Mapping[str, object]:
        """Return the architecture settings and trained weights."""
        if self.network is None:
            raise ValueError('Fit the MLP before saving state.')
        return {
            "state_dict": self.network.state_dict(),
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "interval_start_inputs": self.network.interval_start_inputs,
        }

    def parameter_count(self) -> int:
        """Count trainable parameters."""
        if self.network is None:
            return 0
        return sum(parameter.numel() for parameter in self.network.parameters())

    def describe_architecture(self) -> Mapping[str, object]:
        """Return architecture metadata for output artifacts."""
        return {
            "name": "MLP",
            "inputs": list(self.required_features),
            "states": ['Last observed room temperature'],
            "modules": ['Flattened history and known future drivers', 'Two- or three-layer ReLU MLP', 'Multi-step residual output'],
            "constraints": ['No future room temperature as input'],
            "output": 'Multi-step room temperature (°C)',
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
        }
