"""Adaptive physics MLP with full temperature feedback and physical increments."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from exp.training import (
    fit_torch_model,
    initialize_seed,
    predict_torch_model,
)
from models.base import BaseEstimator, FitResult, ModelContext
from models.registry import CANONICAL_FEATURE_ORDER
from util.config import TrainingConfig
from util.features import ThermalWindows


class APCNNMLPForecaster(nn.Module):
    """Forecast room temperature with stepwise residual updates."""

    # The neural branch predicts an increment from the previous full temperature prediction.
    DATA_FEATURES = (
        "room_temperature",
        "solar_irradiance",
        "time_sin",
        "time_cos",
        "occupancy",
    )
    # The E branch estimates HVAC and exchange coefficients from current conditions.
    PHYSICAL_FEATURES = CANONICAL_FEATURE_ORDER

    def __init__(
        self,
        feature_count: int,
        history_steps: int,
        horizon_steps: int,
        feature_indices: Mapping[str, int],
        hidden_size: int,
        layers: int,
        coefficient_hidden_size: int,
        zero_power_scaled: float,
        temperature_scaling: Mapping[str, tuple[float, float]],
        power_span: float,
        interval_minutes: int,
        physical_scale: float = 0.01,
        division_factor: float = 10.0,
        interval_start_inputs: bool = False,
    ) -> None:
        super().__init__()
        # Validate dimensions before constructing the layers.
        for value, name in (
            (feature_count, "feature_count"),
            (history_steps, "history_steps"),
            (horizon_steps, "horizon_steps"),
            (hidden_size, "hidden_size"),
            (layers, "layers"),
            (coefficient_hidden_size, "coefficient_hidden_size"),
            (interval_minutes, "interval_minutes"),
        ):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(zero_power_scaled):
            raise ValueError('zero_power_scaled must be finite.')
        if not np.isfinite(physical_scale) or physical_scale <= 0.0:
            raise ValueError('physical_scale must be a finite positive number.')
        for name, value in (("power_span", power_span),
                            ("division_factor", division_factor)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number.")

        # Report missing inputs from both branches at once.
        required = set(self.DATA_FEATURES) | set(self.PHYSICAL_FEATURES)
        missing = sorted(required.difference(feature_indices))
        if missing:
            raise ValueError(f"APCNN is missing feature indices: {', '.join(missing)}.")
        expected = {name: i for i, name in enumerate(CANONICAL_FEATURE_ORDER)}
        if feature_count != 7 or dict(feature_indices) != expected:
            raise ValueError('APCNN must use the canonical seven features in order.')

        # Store the window and architecture settings for validation and rollout.
        self.feature_count = feature_count
        self.history_steps = history_steps
        self.interval_start_inputs = interval_start_inputs
        self.horizon_steps = horizon_steps
        self.feature_indices = dict(feature_indices)
        self.hidden_size = hidden_size
        self.layers = layers
        self.zero_power_scaled = float(zero_power_scaled)
        self.physical_scale = float(physical_scale)
        self.division_factor = float(division_factor)
        self.interval_minutes = interval_minutes
        # Resolve D-branch feature indices once.
        self.data_indices = [
            self.feature_indices[name] for name in self.DATA_FEATURES
        ]

        # Keep physical scales distinct from normalized inputs.
        for name in ("room_temperature", "outdoor_temperature"):
            values = temperature_scaling.get(name)
            if (
                not isinstance(values, (tuple, list))
                or len(values) != 2
                or not np.isfinite(values).all()
                or values[1] <= 0.0
            ):
                raise ValueError(f"{name} has invalid temperature scaling.")
            # Store scaling as buffers for checkpoints and device transfers.
            self.register_buffer(
                f"_{name}_minimum",
                torch.tensor(float(values[0])),
            )
            self.register_buffer(
                f"_{name}_span",
                torch.tensor(float(values[1])),
            )

        self.register_buffer("_power_span", torch.tensor(float(power_span)))
        self.register_buffer("_zero_power_scaled",
                             torch.tensor(float(zero_power_scaled)))
        lower = -float(zero_power_scaled) * float(power_span)
        reference = max(abs(lower), abs(lower + float(power_span)))
        self.register_buffer("reference_power_w",
                             torch.tensor(reference if reference > 0 else 1.))
        self.register_buffer("_step_hours",
                             torch.tensor(float(interval_minutes) / 60))

        modules: list[nn.Module] = []
        input_size = len(self.data_indices)
        for _ in range(layers):
            modules.extend((nn.Linear(input_size, hidden_size), nn.Sigmoid()))
            input_size = hidden_size
        modules.append(nn.Linear(hidden_size, 1))
        self.data_mlp = nn.Sequential(*modules)

        # Generate three positive coefficients from current conditions, as in Adapt-PCNN.
        self.coefficient_hidden = nn.Linear(
            len(self.PHYSICAL_FEATURES),
            coefficient_hidden_size,
        )
        self.coefficient_head = nn.Linear(coefficient_hidden_size, 3)
        # Use Xavier weights and zero biases for all linear layers.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """Validate the shape and finiteness of the complete window."""
        # Require a full history and forecast window with all input features.
        expected_steps = self.history_steps + self.horizon_steps
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 3
            or inputs.shape[1] != expected_steps
            or inputs.shape[2] != self.feature_count
        ):
            raise ValueError(
                'APCNN input shape must be '
                f"[batch, {expected_steps}, {self.feature_count}]."
            )
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError('APCNN inputs must all be finite.')

    def _temperature_on_room_scale(
        self,
        features: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Convert independently scaled outdoor temperature to the indoor scale."""
        # Recover Celsius first, then apply the room-temperature scale.
        values = features[:, self.feature_indices[name]]
        minimum = getattr(self, f"_{name}_minimum")
        span = getattr(self, f"_{name}_span")
        return (values * span + minimum - self._room_temperature_minimum
                ) / self._room_temperature_span

    def _physical_operating_input(
        self,
        features: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Build the operating inputs for the dynamic coefficient network."""
        # Use the seven common features, predicted room temperature, and signed power.
        operating = features[:, [
            self.feature_indices[name] for name in self.PHYSICAL_FEATURES
        ]].clone()
        operating[:, 0] = temperature
        operating[:, 1] = self._temperature_on_room_scale(
            features, "outdoor_temperature",
        )
        channels = self.hvac_channels(features)
        operating[:, 3] = (
            channels["heating_power_w"] + channels["cooling_power_w"]
        ) / self.reference_power_w
        return operating

    def hvac_channels(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Restore watts before splitting signed heating and cooling."""
        scaled = features[..., self.feature_indices["hvac_power"]]
        power = (scaled - self._zero_power_scaled) * self._power_span
        epsilon = torch.maximum(
            power.new_tensor(1e-6),
            4 * torch.finfo(power.dtype).eps * self._power_span,
        )
        heating = torch.where(power > epsilon, power, torch.zeros_like(power))
        cooling = torch.where(power < -epsilon, power, torch.zeros_like(power))
        return {
            "heating_power_w": heating,
            "cooling_power_w": cooling,
            "hvac_active": (heating > 0) | (cooling < 0),
        }

    def physical_coefficients(
        self,
        features: torch.Tensor,
        temperature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Estimate HVAC temperature rates (°C/h) and outdoor exchange rates (1/h)."""
        # Recompute coefficients per sample and forecast step.
        operating_input = self._physical_operating_input(
            features,
            temperature,
        )
        # Softplus is differentiable and keeps the coefficients nonnegative.
        hidden = functional.softplus(
            self.coefficient_hidden(operating_input)
        )
        values = functional.softplus(self.coefficient_head(hidden))
        # Epsilon prevents exact zeros from numerical underflow.
        epsilon = torch.finfo(values.dtype).eps
        values = (values + epsilon) * self.physical_scale
        return {
            "heating_gain": values[:, 0],
            "cooling_gain": values[:, 1],
            "outdoor_exchange": values[:, 2],
        }

    def _physical_increment(
        self,
        features: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a Celsius rate to a scaled temperature increment over one step."""
        coefficients = self.physical_coefficients(features, temperature)
        channels = self.hvac_channels(features)
        outdoor = features[:, self.feature_indices["outdoor_temperature"]]
        outdoor_c = (outdoor * self._outdoor_temperature_span
                     + self._outdoor_temperature_minimum)
        room_c = (temperature * self._room_temperature_span
                  + self._room_temperature_minimum)
        # HVAC rates are in °C/h at reference power; exchange is in 1/h.
        rate_c = (
            coefficients["heating_gain"] * channels["heating_power_w"]
            / self.reference_power_w
            + coefficients["cooling_gain"] * channels["cooling_power_w"]
            / self.reference_power_w
            + coefficients["outdoor_exchange"] * (outdoor_c - room_c)
        )
        return rate_c * self._step_hours / self._room_temperature_span

    def physical_rollout(self, inputs: torch.Tensor) -> torch.Tensor:
        """Roll out the physical branch alone for diagnostics and consistency checks."""
        self._validate_inputs(inputs)
        room_index = self.feature_indices["room_temperature"]
        # Start at the last measured temperature with zero accumulated effect.
        baseline = inputs[:, self.history_steps - 1, room_index]
        temperature = baseline.clone()
        effect = torch.zeros_like(baseline)
        effects: list[torch.Tensor] = []
        for step in range(self.horizon_steps):
            # Read only future exogenous inputs and feed back the previous physical state.
            source = (self.history_steps + step
                      - int(self.interval_start_inputs))
            features = inputs[:, source, :]
            effect = effect + self._physical_increment(features, temperature)
            temperature = baseline + effect
            effects.append(effect)
        # Return the cumulative effect relative to baseline, not absolute temperature.
        return torch.stack(effects, dim=1)

    def forward_components(
        self, inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forecast with full temperature feedback and return increments and power diagnostics."""
        self._validate_inputs(inputs)
        room_index = self.feature_indices["room_temperature"]
        temperature = inputs[:, self.history_steps - 1, room_index]
        records: dict[str, list[torch.Tensor]] = {
            key: [] for key in (
                "temperature", "neural_increment", "physical_increment",
                "heating_power_w", "cooling_power_w", "hvac_active",
            )
        }

        for step in range(self.horizon_steps):
            # Both branches use the previous full prediction, never future measured temperature.
            source = (self.history_steps + step
                      - int(self.interval_start_inputs))
            features = inputs[:, source, :].clone()
            features[:, room_index] = temperature
            neural = self.data_mlp(
                features[:, self.data_indices]
            ).squeeze(-1) / self.division_factor
            physical = self._physical_increment(features, temperature)
            temperature = temperature + neural + physical
            records["temperature"].append(temperature)
            records["neural_increment"].append(neural)
            records["physical_increment"].append(physical)
            for key, value in self.hvac_channels(features).items():
                records[key].append(value)
        return {key: torch.stack(values, dim=1)
                for key, values in records.items()}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the scaled room temperature expected by the trainer."""
        return self.forward_components(inputs)["temperature"]


class APCNNMLPEstimator(BaseEstimator):
    """Adapt APCNN-MLP to the shared estimator interface."""

    network_class = APCNNMLPForecaster
    name = "apcnn_mlp"
    display_name = "APCNN-MLP"
    implementation_version = 2
    data_module_name = 'Feedforward neural increment MLP'
    # Use the common seven inputs.
    required_features = CANONICAL_FEATURE_ORDER

    def __init__(
        self,
        hidden_size: int = 128,
        layers: int = 3,
        coefficient_hidden_size: int = 64,
        physical_scale: float = 0.01,
        division_factor: float = 10.0,
    ) -> None:
        # Store the MLP architecture and physics adaptation parameters.
        for value, name in (
            (hidden_size, "hidden_size"),
            (layers, "layers"),
            (coefficient_hidden_size, "coefficient_hidden_size"),
        ):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(physical_scale) or physical_scale <= 0.0:
            raise ValueError('physical_scale must be a finite positive number.')
        if (isinstance(division_factor, bool)
                or not isinstance(division_factor, (int, float))
                or not np.isfinite(division_factor) or division_factor <= 0):
            raise ValueError('division_factor must be a finite positive number.')
        self.hidden_size = hidden_size
        self.layers = layers
        self.coefficient_hidden_size = coefficient_hidden_size
        self.physical_scale = float(physical_scale)
        self.division_factor = float(division_factor)
        self.network: APCNNMLPForecaster | None = None

    def _build(
        self,
        context: ModelContext,
        seed: int,
    ) -> APCNNMLPForecaster:
        """Build APCNN from the training scaler and canonical features."""
        # Seed initialization before constructing the network.
        initialize_seed(seed)
        # Resolve feature indices once for both branches.
        feature_indices = {
            name: context.feature_index(name)
            for name in self.required_features
        }
        # Express physical zero power on the normalized HVAC scale.
        zero_power_scaled = (
            0.0 - context.scaler.minimum["hvac_power"]
        ) / context.scaler.span["hvac_power"]
        temperature_scaling = {
            name: (
                context.scaler.minimum[name], context.scaler.span[name],
            )
            for name in ("room_temperature", "outdoor_temperature")
        }
        return self.network_class(
            feature_count=len(context.feature_names),
            history_steps=context.history_steps,
            horizon_steps=context.horizon_steps,
            feature_indices=feature_indices,
            hidden_size=self.hidden_size,
            layers=self.layers,
            coefficient_hidden_size=self.coefficient_hidden_size,
            zero_power_scaled=zero_power_scaled,
            temperature_scaling=temperature_scaling,
            power_span=context.scaler.span["hvac_power"],
            interval_minutes=context.interval_minutes,
            physical_scale=self.physical_scale,
            division_factor=self.division_factor,
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
        # Use the shared trainer; forecast state is local to each call.
        self.network = self._build(context, seed)
        # Scale target temperatures to match the network output.
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
        """Predict room temperatures in Celsius."""
        if self.network is None:
            raise ValueError(f"{self.display_name} must be fitted first.")
        # The shared helper manages batches and CPU/CUDA placement.
        scaled = predict_torch_model(
            self.network,
            windows.scaled_inputs,
            training_config.batch_size,
            training_config.device,
        )
        # Expose only Celsius predictions to the experiment layer.
        return context.scaler.inverse_target(scaled)

    def state_payload(self) -> Mapping[str, object]:
        """Return network weights and all architecture parameters."""
        if self.network is None:
            raise ValueError(f"{self.display_name} must be fitted before saving state.")
        # Include all hyperparameters needed to rebuild the network.
        return {
            "state_dict": self.network.state_dict(),
            "model_name": self.name,
            "implementation_version": self.implementation_version,
            "division_factor": self.division_factor,
            "interval_minutes": self.network.interval_minutes,
            "history_steps": self.network.history_steps,
            "horizon_steps": self.network.horizon_steps,
            "interval_start_inputs": self.network.interval_start_inputs,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "coefficient_hidden_size": self.coefficient_hidden_size,
            "physical_scale": self.physical_scale,
        }

    def parameter_count(self) -> int:
        """Count APCNN trainable parameters."""
        if self.network is None:
            return 0
        # Count both the neural increment and coefficient network parameters.
        return sum(p.numel() for p in self.network.parameters())

    def describe_architecture(self) -> Mapping[str, object]:
        """Describe the feedback, neural increment, and physical rates."""
        return {
            "name": self.display_name,
            "implementation_version": self.implementation_version,
            "inputs": list(self.required_features),
            "data_features": list(APCNNMLPForecaster.DATA_FEATURES),
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "coefficient_hidden_size": self.coefficient_hidden_size,
            "division_factor": self.division_factor,
            "physical_scale": self.physical_scale,
            "activation": {"hidden": "Sigmoid", "output": "Linear"},
            "initialization": "Xavier normal weights, zero biases",
            "states": ['Previous full room-temperature prediction'],
            "modules": [
                self.data_module_name,
                'Operating condition encoder',
                'Dynamic positive coefficient head',
                'Building physics increment discretized in hours',
            ],
            "constraints": [
                'Three strictly positive, condition-dependent physical coefficients',
                'Heating contributes positively and cooling negatively',
                'Compute physical temperature changes from watts and Celsius differences over the sampling interval',
                'Divide the neural increment by division_factor before applying it to the full temperature',
                'Closed-loop forecast without measured future room temperature',
            ],
            "coefficient_units": {
                "heating_gain": "°C/h at reference power",
                "cooling_gain": "°C/h at reference power",
                "outdoor_exchange": "1/h",
            },
            "output": "T_next = T + neural/division_factor + physical",
        }

