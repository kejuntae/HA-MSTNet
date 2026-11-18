"""Data-driven residual LSTM for closed-loop room-temperature forecasting."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import nn

from exp.training import (
    fit_torch_model,
    initialize_seed,
    predict_torch_model,
)
from models.base import BaseEstimator, FitResult, ModelContext
from models.registry import CANONICAL_FEATURE_ORDER
from util.config import TrainingConfig
from util.features import ThermalWindows


class LSTMForecaster(nn.Module):
    """Encode history and autoregressively feed back predicted room temperature."""

    def __init__(
        self,
        feature_count: int,
        history_steps: int,
        horizon_steps: int,
        room_index: int,
        hidden_size: int,
        layers: int,
        interval_start_inputs: bool = False,
    ) -> None:
        super().__init__()
        # Require positive integer layer sizes.
        for value, name in (
            (feature_count, "feature_count"),
            (history_steps, "history_steps"),
            (horizon_steps, "horizon_steps"),
            (hidden_size, "hidden_size"),
            (layers, "layers"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 <= room_index < feature_count:
            raise ValueError('room_index is outside the feature range.')

        # Store input dimensions for window validation.
        self.feature_count = feature_count
        self.history_steps = history_steps
        self.interval_start_inputs = interval_start_inputs
        self.horizon_steps = horizon_steps
        self.room_index = room_index

        # Project features to the hidden width and apply Tanh.
        self.input_projection = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.Tanh(),
        )
        # Keep the layout [batch, time, features].
        self.recurrent = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
        )
        # Normalize hidden outputs and read out the next temperature increment.
        self.normalization = nn.LayerNorm(hidden_size)
        self.delta_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a multi-step closed-loop forecast in scaled space."""
        # The input contains observations and known forecast drivers.
        expected_steps = self.history_steps + self.horizon_steps
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 3
            or inputs.shape[1] != expected_steps
            or inputs.shape[2] != self.feature_count
        ):
            raise ValueError(
                'LSTM input shape must be '
                f"[batch, {expected_steps}, {self.feature_count}]."
            )

        # Encode all history and keep the final recurrent state.
        history_embedding = self.input_projection(
            inputs[:, :self.history_steps, :]
        )
        _, state = self.recurrent(history_embedding)

        # Initialize the rollout at the last observed temperature.
        temperature = inputs[:, self.history_steps - 1, self.room_index]
        predictions: list[torch.Tensor] = []
        for step in range(self.horizon_steps):
            # Use known weather, HVAC, and time inputs for the current step.
            source = self.history_steps + step - int(self.interval_start_inputs)
            current = inputs[:, source, :].clone()
            # Feed back the previous prediction, never a future measurement.
            current[:, self.room_index] = temperature

            # Run a one-step input with the previous recurrent state.
            embedding = self.input_projection(current).unsqueeze(1)
            recurrent_output, state = self.recurrent(embedding, state)
            hidden = self.normalization(recurrent_output[:, 0, :])

            # Predict the change in temperature rather than its absolute value.
            temperature = temperature + self.delta_head(hidden).squeeze(-1)
            predictions.append(temperature)
        # Stack predictions as [batch, forecast steps].
        return torch.stack(predictions, dim=1)


class LSTMEstimator(BaseEstimator):
    """Adapt the LSTM to shared training and Celsius predictions."""

    name = "lstm"
    required_features = CANONICAL_FEATURE_ORDER

    def __init__(self, hidden_size: int = 64, layers: int = 1) -> None:
        # Store configurable width and depth plus the fitted network.
        for value, name in ((hidden_size, "hidden_size"), (layers, "layers")):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.hidden_size = hidden_size
        self.layers = layers
        self.network: LSTMForecaster | None = None

    def _build(self, context: ModelContext, seed: int) -> LSTMForecaster:
        """Seed and build the network."""
        # Set the seed before layer initialization.
        initialize_seed(seed)
        # Get dimensions and column indices from the data context.
        return LSTMForecaster(
            feature_count=len(context.feature_names),
            history_steps=context.history_steps,
            horizon_steps=context.horizon_steps,
            room_index=context.feature_index("room_temperature"),
            hidden_size=self.hidden_size,
            layers=self.layers,
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
        """Fit in scaled space and restore the best validation weights."""
        # Rebuild the network on each fit to avoid reusing old weights.
        self.network = self._build(context, seed)

        # Scale Celsius targets to match the input windows.
        outcome = fit_torch_model(
            self.network,
            train.scaled_inputs,
            context.scaler.transform_target(train.targets_c),
            validation.scaled_inputs,
            context.scaler.transform_target(validation.targets_c),
            training_config,
            seed,
        )
        # Convert training output to the shared FitResult.
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
        """Predict room temperatures in Celsius."""
        if self.network is None:
            raise ValueError('Fit the LSTM first.')
        # Use the shared batch and device-aware prediction helper.
        scaled = predict_torch_model(
            self.network,
            windows.scaled_inputs,
            training_config.batch_size,
            training_config.device,
        )
        # Return Celsius rather than normalized output.
        return context.scaler.inverse_target(scaled)

    def state_payload(self) -> Mapping[str, object]:
        """Return weights and architecture settings."""
        if self.network is None:
            raise ValueError('Fit the LSTM before saving state.')
        # Restoring the weights also requires width and depth.
        return {
            "state_dict": self.network.state_dict(),
            "interval_start_inputs": self.network.interval_start_inputs,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
        }

    def parameter_count(self) -> int:
        """Count trainable parameters."""
        if self.network is None:
            return 0
        # Count scalar elements in each parameter tensor.
        return sum(parameter.numel() for parameter in self.network.parameters())

    def describe_architecture(self) -> Mapping[str, object]:
        """Return LSTM architecture metadata."""
        return {
            "name": "LSTM",
            "inputs": list(self.required_features),
            "states": ['Hidden state', 'Cell state', 'Previous room-temperature prediction'],
            "modules": ['Input projection', "LSTM", 'Layer normalization', 'Residual temperature head'],
            "constraints": ['Closed-loop temperature feedback during forecasting'],
            "output": 'Multi-step room temperature (°C)',
        }
