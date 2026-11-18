"""Causal PCICNN with nonnegative weights shared by time lag."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from exp.training import fit_torch_model, initialize_seed, predict_torch_model
from models.base import BaseEstimator, FitResult, ModelContext
from util.config import TrainingConfig
from util.features import ThermalWindows
from util.model_features import THERMAL_FEATURES


REFERENCE_EXOGENOUS_FEATURES = (
    "outdoor_temperature", "hvac_power", "solar_irradiance", "occupancy",
)
LAYER_COUNT_CONVENTION = "all_linear_layers"
INITIALIZATION = "reference_glorot_div10"


class CausalSharedLinear(nn.Module):
    """Lower-triangular linear layer with nonnegative lag-shared weights."""

    def __init__(
        self,
        steps: int,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        *,
        fan_in: int | None = None,
        fan_out: int | None = None,
    ) -> None:
        super().__init__()
        # Time steps and channel counts determine the weight shape.
        for name, value in (
            ("steps", steps),
            ("in_channels", in_channels),
            ("out_channels", out_channels),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(bias, bool):
            raise ValueError('bias must be Boolean.')

        self.steps = steps
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Store one weight block per lag: [lag, output channels, input channels].
        self.raw_weight = nn.Parameter(
            torch.empty(steps, out_channels, in_channels)
        )
        # Use the full reference matrix's Glorot range, not the shared tensor's fan-in.
        dense_fan_in = steps * in_channels if fan_in is None else fan_in
        dense_fan_out = steps * out_channels if fan_out is None else fan_out
        if dense_fan_in < 1 or dense_fan_out < 1:
            raise ValueError('The full matrix fan-in and fan-out must be positive.')
        bound = (6.0 / (dense_fan_in + dense_fan_out))**0.5 / 10.0
        nn.init.uniform_(self.raw_weight, -bound, bound)
        if bias:
            # The reference uses a separate bias for each output node and time step.
            self.bias = nn.Parameter(torch.empty(steps, out_channels))
            bound = 1.0 / dense_fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            # Register None explicitly to keep the module and state dict layout consistent.
            self.register_parameter("bias", None)

    def materialized_weight(self) -> torch.Tensor:
        """Expand weights over output steps, input steps, and both channel dimensions."""
        # Use the difference between output and input steps as the lag.
        output_steps = torch.arange(self.steps, device=self.raw_weight.device)
        input_steps = torch.arange(self.steps, device=self.raw_weight.device)
        lags = output_steps[:, None] - input_steps[None, :]
        # Allow connections only from input steps at or before the output step.
        valid_connections = lags >= 0

        # Square weights for nonnegativity and reuse the same slice for each lag.
        # Clamp for safe indexing; the mask zeros future connections.
        shared_weight = self.raw_weight[lags.clamp_min(0)].square()
        # Layout: [output step, input step, output channel, input channel].
        return torch.where(
            valid_connections[..., None, None],
            shared_weight,
            torch.zeros_like(shared_weight),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Use only current and past inputs for each output step."""
        # Input layout: [batch, time, input channel].
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3:
            raise ValueError('inputs must be a tensor with shape [batch, steps, channels].')
        if inputs.shape[1:] != (self.steps, self.in_channels):
            raise ValueError(
                'inputs must have shape '
                f"[batch, {self.steps}, {self.in_channels}]."
            )
        if not torch.is_floating_point(inputs):
            raise ValueError('inputs must be floating-point tensors.')
        if inputs.dtype != self.raw_weight.dtype:
            raise ValueError('inputs must have the same dtype as the model parameters.')
        if inputs.device != self.raw_weight.device:
            raise ValueError('inputs and model parameters must be on the same device.')
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError('inputs must not contain nonfinite values.')

        # Sum over input steps and channels, retaining output steps and channels.
        outputs = torch.einsum(
            "bsi,tsoi->bto",
            inputs,
            self.materialized_weight(),
        )
        if self.bias is not None:
            # Broadcast the step-specific bias only across the batch.
            outputs = outputs + self.bias
        return outputs


class PCICNNForecaster(nn.Module):
    """Forecast from the current room temperature and signed exogenous inputs."""

    def __init__(
        self,
        history_steps: int,
        horizon_steps: int,
        room_index: int,
        exogenous_indices: tuple[int, ...],
        amplification: int,
        layers: int,
        bias: bool,
        hvac_index: int | None = None,
        zero_power_scaled: float = 0.0,
        interval_start_inputs: bool = False,
    ) -> None:
        super().__init__()
        # Require positive horizon, channel expansion, and layer counts.
        for name, value in (
            ("history_steps", history_steps),
            ("horizon_steps", horizon_steps),
            ("amplification", amplification),
            ("layers", layers),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(room_index, int) or isinstance(room_index, bool):
            raise ValueError('room_index must be a nonnegative integer.')
        if room_index < 0:
            raise ValueError('room_index must be a nonnegative integer.')
        if (
            not isinstance(exogenous_indices, tuple)
            or not exogenous_indices
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                for index in exogenous_indices
            )
        ):
            raise ValueError('exogenous_indices must be a nonempty tuple of nonnegative integers.')
        if len(set(exogenous_indices)) != len(exogenous_indices):
            raise ValueError('exogenous_indices must not contain duplicates.')
        if room_index in exogenous_indices:
            raise ValueError('Future exogenous inputs must not include room temperature.')
        if not isinstance(bias, bool):
            raise ValueError('bias must be Boolean.')
        if hvac_index is not None and (
            not isinstance(hvac_index, int)
            or isinstance(hvac_index, bool)
            or hvac_index not in exogenous_indices
        ):
            raise ValueError('hvac_index must be in exogenous_indices.')
        if not np.isfinite(zero_power_scaled):
            raise ValueError('zero_power_scaled must be finite.')
        if layers < 2:
            raise ValueError('layers must be at least 2, including the input and output layers.')
        if not isinstance(interval_start_inputs, bool):
            raise ValueError('interval_start_inputs must be Boolean.')

        # Forecast inputs are the last observed temperature and future drivers.
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.interval_start_inputs = interval_start_inputs
        self.room_index = room_index
        self.exogenous_indices = exogenous_indices
        self.hvac_exogenous_position = (
            exogenous_indices.index(hvac_index)
            if hvac_index is not None
            else None
        )
        self.register_buffer(
            "zero_power_scaled",
            torch.tensor(float(zero_power_scaled)),
        )
        self.feature_count = max((room_index, *exogenous_indices)) + 1
        self.amplification = amplification
        self.layers = layers
        # amplification is the number of hidden nodes per time step.
        hidden_channels = amplification

        # Project exogenous inputs into a wider causal representation.
        self.input_layer = CausalSharedLinear(
            horizon_steps,
            len(exogenous_indices),
            hidden_channels,
            bias,
            fan_in=1 + horizon_steps * len(exogenous_indices),
        )
        # The current temperature uses a separate nonnegative weight per step and channel.
        # This is the reference input_state_weight path.
        self.raw_state_weight = nn.Parameter(
            torch.empty(horizon_steps, hidden_channels)
        )
        input_width = 1 + horizon_steps * len(exogenous_indices)
        hidden_width = horizon_steps * hidden_channels
        state_bound = (6.0 / (input_width + hidden_width))**0.5 / 10.0
        nn.init.uniform_(self.raw_state_weight, -state_bound, state_bound)
        # Hidden layers preserve width, causality, lag sharing, and nonnegativity.
        self.hidden_layers = nn.ModuleList(
            CausalSharedLinear(
                horizon_steps,
                hidden_channels,
                hidden_channels,
                bias,
            )
            for _ in range(layers - 2)
        )
        # Read out one absolute scaled temperature at each forecast step.
        self.output_layer = CausalSharedLinear(
            horizon_steps,
            hidden_channels,
            1,
            bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict absolute future temperatures in scaled space."""
        # Require the full history and forecast window.
        expected_steps = self.history_steps + self.horizon_steps
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 3
            or inputs.shape[1] != expected_steps
            or inputs.shape[2] != self.feature_count
        ):
            raise ValueError(
                'PCICNN inputs must have shape '
                f"[batch, {expected_steps}, {self.feature_count}]."
            )
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError('PCICNN inputs must all be finite.')

        # Feed the last observed temperature through the learned state path.
        current_room = inputs[:, self.history_steps - 1, self.room_index]
        # Interval-start alignment uses u(t) for T(t+1); otherwise use the target row.
        source_start = self.history_steps - int(self.interval_start_inputs)
        source_stop = source_start + self.horizon_steps
        future = inputs[
            :, source_start:source_stop, self.exogenous_indices
        ].clone()
        if self.hvac_exogenous_position is not None:
            # Subtract scaled physical zero to recover the sign of HVAC power.
            future[:, :, self.hvac_exogenous_position] -= (
                self.zero_power_scaled
            )
        state_effect = (
            current_room[:, None, None]
            * self.raw_state_weight.square()[None, :, :]
        )
        # LeakyReLU preserves a negative hidden response to cooling.
        hidden = functional.leaky_relu(self.input_layer(future) + state_effect)
        for layer in self.hidden_layers:
            hidden = functional.leaky_relu(layer(hidden))
        # Predict absolute scaled temperature as in the reference implementation.
        return self.output_layer(hidden).squeeze(-1)

    def lag_penalty(self) -> torch.Tensor:
        """Penalize increases in raw shared weight norms with lag."""
        # Take channel norms per physical input in the first layer and matrix norms elsewhere.
        penalty = self.raw_state_weight.new_zeros(())
        layers = (self.input_layer, *self.hidden_layers, self.output_layer)
        for index, layer in enumerate(layers):
            dimensions = 1 if index == 0 else (1, 2)
            norms = torch.linalg.vector_norm(layer.raw_weight, dim=dimensions)
            increases = functional.relu(norms.diff(dim=0))
            penalty = penalty + increases.sum().square()
        return penalty

    def bias_penalty(self) -> torch.Tensor:
        """Penalize input and hidden biases, excluding output biases and state weights."""
        penalty = self.raw_state_weight.new_zeros(())
        for layer in (self.input_layer, *self.hidden_layers):
            if layer.bias is not None:
                penalty = penalty + layer.bias.square().sum()
        return penalty


def _regularization_coefficient(value: float, name: str) -> float:
    """Validate the regularization coefficient; zero disables the term."""
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not np.isfinite(value) or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return float(value)


def build_pcicnn_objective(
    lag_regularization: float, bias_regularization: float,
):
    """Build a logged training objective while retaining validation MSE for selection."""
    lag_coefficient = _regularization_coefficient(
        lag_regularization, "lag_regularization",
    )
    bias_coefficient = _regularization_coefficient(
        bias_regularization, "bias_regularization",
    )

    def objective(model, inputs, targets):
        """Return predictions and raw and weighted regularization terms."""
        predictions = model(inputs)
        mse = functional.mse_loss(predictions, targets)
        lag = model.lag_penalty()
        bias = model.bias_penalty()
        weighted_lag = lag_coefficient * lag
        weighted_bias = bias_coefficient * bias
        return predictions, {
            "mse": mse, "lag_penalty": lag, "bias_penalty": bias,
            "weighted_lag": weighted_lag, "weighted_bias": weighted_bias,
            "total": mse + weighted_lag + weighted_bias,
        }

    return objective


def _context_metadata(context: ModelContext) -> dict[str, object]:
    """Record feature order, alignment, and training scaling for weight restoration."""
    return {
        "feature_names": list(context.feature_names),
        "history_steps": context.history_steps,
        "horizon_steps": context.horizon_steps,
        "interval_minutes": context.interval_minutes,
        "interval_start_inputs": context.interval_start_inputs,
        "scaler": context.scaler.to_dict(),
    }


class PCICNNEstimator(BaseEstimator):
    """Adapt PCICNN to the shared estimator interface."""

    name = "pcicnn"
    implementation_version = 2
    required_features = THERMAL_FEATURES

    def __init__(
        self,
        amplification: int = 15,
        layers: int = 3,
        bias: bool = False,
        lag_regularization: float = 0.001,
        bias_regularization: float = 0.001,
    ) -> None:
        # Count input and output layers toward the reference depth of three.
        for name, value in (
            ("amplification", amplification), ("layers", layers),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(bias, bool):
            raise ValueError('bias must be Boolean.')
        if layers < 2:
            raise ValueError('layers must be at least 2, including the input and output layers.')
        self.amplification = amplification
        self.layers = layers
        self.bias = bias
        self.lag_regularization = _regularization_coefficient(
            lag_regularization, "lag_regularization",
        )
        self.bias_regularization = _regularization_coefficient(
            bias_regularization, "bias_regularization",
        )
        self.network: PCICNNForecaster | None = None
        self.loss_components: tuple[dict[str, float], ...] = ()
        self.context_metadata: dict[str, object] | None = None

    def _build(self, context: ModelContext, seed: int) -> PCICNNForecaster:
        """Build the network from the common feature map."""
        # Seed the random generators before initializing weights.
        initialize_seed(seed)
        self.context_metadata = _context_metadata(context)
        room_index = context.feature_index("room_temperature")
        # The core selects four physical drivers from the seven public inputs.
        exogenous_indices = tuple(
            context.feature_index(name)
            for name in REFERENCE_EXOGENOUS_FEATURES
        )
        hvac_index = context.feature_index("hvac_power")
        # Locate physical zero power in the common [0, 1] scale.
        zero_power_scaled = (
            0.0 - context.scaler.minimum["hvac_power"]
        ) / context.scaler.span["hvac_power"]
        return PCICNNForecaster(
            context.history_steps,
            context.horizon_steps,
            room_index,
            exogenous_indices,
            self.amplification,
            self.layers,
            self.bias,
            hvac_index=hvac_index,
            zero_power_scaled=zero_power_scaled,
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
        # Rebuild the network for every fit.
        self.network = self._build(context, seed)
        self.loss_components = ()
        # Scale targets to match the network output.
        outcome = fit_torch_model(
            self.network,
            train.scaled_inputs,
            context.scaler.transform_target(train.targets_c),
            validation.scaled_inputs,
            context.scaler.transform_target(validation.targets_c),
            training_config,
            seed,
            batch_loss_function=build_pcicnn_objective(
                self.lag_regularization, self.bias_regularization,
            ),
            validation_component="mse",
        )
        self.loss_components = outcome.loss_components
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
            raise ValueError('Fit PCICNN first.')
        # Use the shared prediction helper for batches, devices, and model mode.
        scaled = predict_torch_model(
            self.network,
            windows.scaled_inputs,
            training_config.batch_size,
            training_config.device,
        )
        # Return a Celsius array with shape [samples, forecast steps].
        return context.scaler.inverse_target(scaled)

    def state_payload(self) -> Mapping[str, object]:
        """Return network weights and architecture settings."""
        if self.network is None:
            raise ValueError('Fit PCICNN before saving state.')
        # Rebuild all causal layers with matching options before loading weights.
        return {
            "state_dict": self.network.state_dict(),
            "model_name": self.name,
            "implementation_version": self.implementation_version,
            "exogenous_features": list(REFERENCE_EXOGENOUS_FEATURES),
            "layer_count_convention": LAYER_COUNT_CONVENTION,
            "initialization": INITIALIZATION,
            "interval_start_inputs": self.network.interval_start_inputs,
            "context": self.context_metadata,
            "amplification": self.amplification,
            "layers": self.layers,
            "bias": self.bias,
            "lag_regularization": self.lag_regularization,
            "bias_regularization": self.bias_regularization,
        }

    def load_payload(
        self, payload: Mapping[str, object], context: ModelContext,
    ) -> None:
        """Reject incompatible checkpoint versions and data contexts before restoring."""
        expected = {
            "model_name": self.name,
            "implementation_version": self.implementation_version,
            "exogenous_features": list(REFERENCE_EXOGENOUS_FEATURES),
            "layer_count_convention": LAYER_COUNT_CONVENTION,
            "initialization": INITIALIZATION,
            "interval_start_inputs": context.interval_start_inputs,
            "context": _context_metadata(context),
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"PCICNN checkpoint {name} is incompatible; retraining is required.")
        try:
            restored = PCICNNEstimator(**{
                key: payload[key] for key in (
                    "amplification", "layers", "bias",
                    "lag_regularization", "bias_regularization",
                )
            })
            restored.network = restored._build(context, seed=0)
            restored.network.load_state_dict(
                payload["state_dict"], strict=True,
            )
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError('PCICNN checkpoint parameters are missing or have incompatible shapes.') from error
        # Replace the existing model only after restoration succeeds.
        self.__dict__.update(restored.__dict__)

    def parameter_count(self) -> int:
        """Count PCICNN trainable parameters."""
        if self.network is None:
            return 0
        # Count shared lag weights once rather than counting every connection.
        return sum(
            parameter.numel() for parameter in self.network.parameters()
        )

    def describe_architecture(self) -> Mapping[str, object]:
        """Return metadata for the causal weight-sharing architecture."""
        return {
            "name": "PCICNN",
            "implementation_version": self.implementation_version,
            "inputs": list(self.required_features),
            "exogenous_features": list(REFERENCE_EXOGENOUS_FEATURES),
            "layers": self.layers,
            "layer_count_convention": LAYER_COUNT_CONVENTION,
            "initialization": INITIALIZATION,
            "lag_regularization": self.lag_regularization,
            "bias_regularization": self.bias_regularization,
            "selection_metric": "validation_mse",
            "states": ['Learnable nonnegative path from the current room temperature', 'Signed exogenous inputs'],
            "modules": ['Causal input layer', 'Causal hidden layers', 'Causal output layer'],
            "constraints": [
                'Lower-triangular causal connections',
                'Nonnegative weights',
                'Weights shared by time lag',
                'HVAC input centered at physical zero power',
            ],
            "output": 'Absolute future room temperatures in scaled space (returned in °C)',
        }
