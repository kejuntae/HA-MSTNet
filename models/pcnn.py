"""PCNN with seven inputs, positive physical layers, and reference D/E recurrence."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import nn

from exp.training import fit_torch_model, initialize_seed, predict_torch_model
from models.base import BaseEstimator, FitResult, ModelContext
from util.config import TrainingConfig
from util.features import ThermalWindows
from util.model_features import THERMAL_FEATURES


class DiagonalPositiveLinear(nn.Module):
    """Use exponential parameterization to keep scalar physical weights positive."""

    def __init__(self) -> None:
        super().__init__()
        self.log_weight = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Scale physical inputs with positive weights and no bias."""
        return inputs * self.log_weight.exp()


class PCNNForecaster(nn.Module):
    """Forecast with the reference PCNN recurrence T = D + E."""

    DATA_FEATURES = (
        "solar_irradiance",
        "time_sin",
        "time_cos",
        "occupancy",
    )

    def __init__(
        self,
        feature_count: int,
        history_steps: int,
        horizon_steps: int,
        feature_indices: Mapping[str, int],
        hidden_size: int,
        layers: int,
        zero_power_scaled: float,
        temperature_scaling: Mapping[str, tuple[float, float]],
        interval_minutes: int,
        power_span: float,
        input_hidden_sizes: tuple[int, ...] | list[int] = (128, 128),
        output_hidden_sizes: tuple[int, ...] | list[int] = (128, 128),
        learn_initial_hidden_states: bool = True,
        base_division_factor: float = 10.0,
        interval_start_inputs: bool = False,
    ) -> None:
        super().__init__()

        # Architecture dimensions must be positive integers, not booleans.
        for value, name in (
            (feature_count, "feature_count"),
            (history_steps, "history_steps"),
            (horizon_steps, "horizon_steps"),
            (hidden_size, "hidden_size"),
            (layers, "layers"),
            (interval_minutes, "interval_minutes"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        for sizes, name in (
            (input_hidden_sizes, "input_hidden_sizes"),
            (output_hidden_sizes, "output_hidden_sizes"),
        ):
            if (
                not isinstance(sizes, (list, tuple))
                or not sizes
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 1
                    for value in sizes
                )
            ):
                raise ValueError(f"{name} must be a nonempty sequence of positive integers.")
        if not isinstance(learn_initial_hidden_states, bool):
            raise ValueError('learn_initial_hidden_states must be Boolean.')
        if (
            not isinstance(base_division_factor, (int, float))
            or isinstance(base_division_factor, bool)
            or not math.isfinite(float(base_division_factor))
            or float(base_division_factor) <= 0.0
        ):
            raise ValueError('base_division_factor must be a finite positive number.')
        if not isinstance(feature_indices, Mapping):
            raise ValueError('feature_indices must map features to indices.')
        if (
            isinstance(power_span, bool)
            or not isinstance(power_span, (int, float))
            or not math.isfinite(float(power_span))
            or power_span <= 0
        ):
            raise ValueError('power_span must be a finite positive number.')
        if not math.isfinite(float(zero_power_scaled)):
            raise ValueError('zero_power_scaled must be finite.')
        required = (
            *self.DATA_FEATURES,
            "room_temperature",
            "outdoor_temperature",
            "hvac_power",
        )
        missing = sorted(set(required).difference(feature_indices))
        if missing:
            raise ValueError(f"PCNN is missing feature indices: {', '.join(missing)}.")
        used_indices: dict[int, str] = {}
        for name in required:
            index = feature_indices[name]
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(f"{name} feature index must be an integer, not a Boolean.")
            if not 0 <= index < feature_count:
                raise ValueError(
                    f"{name} feature index must satisfy "
                    f"0 <= index < {feature_count}."
                )
            previous_name = used_indices.get(index)
            if previous_name is not None:
                raise ValueError(
                    'PCNN feature indices must not be duplicated: '
                    f"{previous_name} and {name} share index {index}."
                )
            used_indices[index] = name
        if not isinstance(temperature_scaling, Mapping):
            raise ValueError('temperature_scaling must be a mapping.')
        for name in (
            "room_temperature",
            "outdoor_temperature",
        ):
            values = temperature_scaling.get(name)
            if (
                not isinstance(values, (tuple, list))
                or len(values) != 2
                or not np.isfinite(values).all()
                or float(values[1]) <= 0.0
            ):
                raise ValueError(f"{name} has invalid temperature scaling.")

        self.feature_count = feature_count
        self.history_steps = history_steps
        self.interval_start_inputs = interval_start_inputs
        self.horizon_steps = horizon_steps
        self.feature_indices = dict(feature_indices)
        self.zero_power_scaled = float(zero_power_scaled)
        self.register_buffer(
            "_hvac_power_span", torch.tensor(float(power_span)),
        )
        self.temperature_scaling = {
            name: tuple(float(value) for value in values)
            for name, values in temperature_scaling.items()
        }
        # Save each temperature scale as buffers for device transfers and checkpoints.
        for name in (
            "room_temperature",
            "outdoor_temperature",
        ):
            minimum, span = self.temperature_scaling[name]
            self.register_buffer(
                f"_{name}_minimum", torch.tensor(minimum)
            )
            self.register_buffer(f"_{name}_span", torch.tensor(span))
        self.interval_minutes = interval_minutes
        self.input_hidden_sizes = tuple(input_hidden_sizes)
        self.output_hidden_sizes = tuple(output_hidden_sizes)
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.base_division_factor = float(base_division_factor)
        self.data_indices = [
            self.feature_indices[name] for name in self.DATA_FEATURES
        ]

        # Room temperature anchors the residual but is not a neural increment input.
        # Use ReLU in the input MLP and Tanh in the output MLP.
        input_sizes = (len(self.data_indices), *self.input_hidden_sizes)
        input_layers: list[nn.Module] = []
        for input_size, output_size in zip(input_sizes, input_sizes[1:]):
            input_layers.extend(
                (nn.Linear(input_size, output_size), nn.ReLU())
            )
        self.input_projection = nn.Sequential(*input_layers)

        # Match the LSTM input width and keep the reference learnable initial state.
        self.base_lstm = nn.LSTM(
            self.input_hidden_sizes[-1],
            hidden_size,
            num_layers=layers,
            batch_first=True,
        )
        self.base_normalization = nn.LayerNorm(hidden_size)
        if self.learn_initial_hidden_states:
            self.initial_base_h = nn.Parameter(
                torch.zeros(layers, hidden_size)
            )
            self.initial_base_c = nn.Parameter(
                torch.zeros(layers, hidden_size)
            )
        else:
            self.register_parameter("initial_base_h", None)
            self.register_parameter("initial_base_c", None)

        # Keep Tanh after the final scalar output layer.
        output_sizes = (hidden_size, *self.output_hidden_sizes, 1)
        output_layers: list[nn.Module] = []
        for input_size, output_size in zip(output_sizes, output_sizes[1:]):
            output_layers.extend(
                (nn.Linear(input_size, output_size), nn.Tanh())
            )
        self.base_delta_head = nn.Sequential(*output_layers)

        # Initialize positive weights to one, then apply physical and time-step priors.
        self.a = DiagonalPositiveLinear()
        self.b = DiagonalPositiveLinear()
        self.d = DiagonalPositiveLinear()

        # Derive reference power from training bounds only.
        # Use maximum absolute power for either sign; use 1 W for an all-zero feature.
        power_minimum = -self.zero_power_scaled * float(power_span)
        reference_power = max(
            abs(power_minimum), abs(power_minimum + float(power_span)),
        )
        reference_power = reference_power if reference_power > 0 else 1.0
        self.register_buffer(
            "reference_power_w", torch.tensor(reference_power),
        )

        # Priors: 2°C in 4 h at reference power; 1.5°C loss in 6 h at a 25°C difference.
        # The [0.1, 0.9] E scale contributes a factor of 0.8 / room-temperature span.
        room_span = self.temperature_scaling["room_temperature"][1]
        step_hours = float(self.interval_minutes) / 60.0
        initial_hvac = (
            0.8 / room_span * 2.0 / reference_power / 4.0 * step_hours
        )
        initial_outdoor = 0.8 / room_span * 1.5 / 25.0 / 6.0 * step_hours
        self.register_buffer("initial_value_a", torch.tensor(initial_hvac))
        self.register_buffer("initial_value_b", torch.tensor(initial_outdoor))
        self.register_buffer("initial_value_d", torch.tensor(initial_hvac))

        self._initialize_original_parameters()

    def _split_hvac_power(
        self, scaled_power: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Restore watts and preserve the signs of heating and cooling."""
        power = (scaled_power - self.zero_power_scaled) * self._hvac_power_span
        # Ignore round-trip scaling noise around physical zero power.
        epsilon = torch.maximum(
            power.new_tensor(1e-6),
            4 * torch.finfo(power.dtype).eps * self._hvac_power_span,
        )
        heating = torch.where(power > epsilon, power, torch.zeros_like(power))
        cooling = torch.where(power < -epsilon, power, torch.zeros_like(power))
        return {
            "heating_power_w": heating,
            "cooling_power_w": cooling,
            "hvac_active": (heating > 0) | (cooling < 0),
        }

    def hvac_channels(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Derive heating, cooling, and activity from the seven public inputs."""
        return self._split_hvac_power(
            features[..., self.feature_indices["hvac_power"]],
        )

    def _initialize_original_parameters(self) -> None:
        """Apply reference Xavier, zero-bias, and physical parameter initialization."""
        for name, parameter in self.named_parameters():
            if "norm" not in name:
                if "bias" in name or "log_weight" in name:
                    nn.init.constant_(parameter, 0.0)
                elif "weight" in name:
                    nn.init.xavier_normal_(parameter)

    @staticmethod
    def _to_legacy(value: torch.Tensor) -> torch.Tensor:
        """Map public [0, 1] inputs to the reference [0.1, 0.9] scale."""
        return value * 0.8 + 0.1

    @staticmethod
    def _to_legacy_value(value: float) -> float:
        """Map one public scaled value to its reference scale."""
        return value * 0.8 + 0.1

    @staticmethod
    def _from_legacy_value(value: torch.Tensor) -> torch.Tensor:
        """Convert a reference scaled value to the public [0, 1] scale."""
        return (value - 0.1) / 0.8

    @staticmethod
    def _effect_from_legacy(value: torch.Tensor) -> torch.Tensor:
        """Map the accumulated E increment without an offset."""
        return value / 0.8

    def _inverse_legacy_temperature(
        self,
        value: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Convert reference temperatures to Celsius using the feature's own scaler."""
        minimum = getattr(self, f"_{name}_minimum")
        span = getattr(self, f"_{name}_span")
        return (value - 0.1) / 0.8 * span + minimum

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """Validate the history and forecast window shape and finiteness."""
        expected_steps = self.history_steps + self.horizon_steps
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 3
            or inputs.shape[1] != expected_steps
            or inputs.shape[2] != self.feature_count
        ):
            raise ValueError(
                'PCNN inputs must have shape '
                f"[batch, {expected_steps}, {self.feature_count}]."
            )
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError('PCNN inputs must all be finite.')

    def _initial_lstm_state(
        self,
        batch_size: int,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create independent initial D states for the current batch."""
        if not self.learn_initial_hidden_states:
            shape = (
                self.base_lstm.num_layers,
                batch_size,
                self.base_lstm.hidden_size,
            )
            return inputs.new_zeros(shape), inputs.new_zeros(shape)
        return (
            self.initial_base_h.unsqueeze(1).expand(
                -1, batch_size, -1
            ).contiguous(),
            self.initial_base_c.unsqueeze(1).expand(
                -1, batch_size, -1
            ).contiguous(),
        )

    def _base_step(
        self,
        features: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute one base-temperature step with the reference D branch."""
        embedding = self.input_projection(
            features[:, self.data_indices]
        ).unsqueeze(1)
        recurrent, next_state = self.base_lstm(embedding, state)
        hidden = self.base_normalization(recurrent[:, 0, :])
        room_index = self.feature_indices["room_temperature"]
        base = features[:, room_index] + (
            self.base_delta_head(hidden).squeeze(-1)
            / self.base_division_factor
        )
        return base, next_state

    def _validate_physical_step_inputs(
        self,
        features: torch.Tensor,
        base_temperature: torch.Tensor,
        last_effect: torch.Tensor,
    ) -> None:
        """Check the shapes of per-step physical inputs."""
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 2
            or features.shape[1] != self.feature_count
        ):
            raise ValueError(
                'features must have shape '
                f"[batch, {self.feature_count}]."
            )
        batch_size = features.shape[0]
        for value, name in (
            (base_temperature, "base_temperature"),
            (last_effect, "last_effect"),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or value.shape[0] != batch_size
            ):
                raise ValueError(
                    f"{name} must have shape [batch], matching the batch size of features."
                )

    def physical_step(
        self,
        features: torch.Tensor,
        base_temperature: torch.Tensor,
        last_effect: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update E in the reference scale and convert the result back."""
        self._validate_physical_step_inputs(
            features, base_temperature, last_effect
        )
        legacy_effect, legacy_power = self._physical_step_legacy(
            self._to_legacy(features),
            self._to_legacy(base_temperature),
            last_effect * 0.8,
        )
        return (
            self._effect_from_legacy(legacy_effect),
            self._from_legacy_value(legacy_power),
        )

    def _physical_step_legacy(
        self,
        features: torch.Tensor,
        base_temperature: torch.Tensor,
        last_effect: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the reference single-step E formula in the [0.1, 0.9] scale."""
        total_scaled = base_temperature + last_effect
        room_c = self._inverse_legacy_temperature(
            total_scaled, "room_temperature"
        )
        outdoor_c = self._inverse_legacy_temperature(
            features[:, self.feature_indices["outdoor_temperature"]],
            "outdoor_temperature",
        )
        effect = last_effect - self.b(
            (room_c - outdoor_c).unsqueeze(-1)
        ).squeeze(-1) * self.initial_value_b
        # Invert scaling before splitting HVAC; cooling remains negative.
        # Net thermal power already reflects equipment and valve action.
        power = features[:, self.feature_indices["hvac_power"]]
        channels = self._split_hvac_power(self._from_legacy_value(power))
        effect = effect + self.a(
            channels["heating_power_w"].unsqueeze(-1)
        ).squeeze(-1) * self.initial_value_a
        effect = effect + self.d(
            channels["cooling_power_w"].unsqueeze(-1)
        ).squeeze(-1) * self.initial_value_d
        return effect, power

    def rollout_components(
        self,
        inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Warm up on history, forecast in closed loop, and return D/E diagnostics."""
        self._validate_inputs(inputs)
        legacy_inputs = self._to_legacy(inputs)
        batch_size = legacy_inputs.shape[0]
        state = self._initial_lstm_state(batch_size, legacy_inputs)
        last_base = legacy_inputs.new_zeros(batch_size)
        last_effect = legacy_inputs.new_zeros(batch_size)

        bases: list[torch.Tensor] = []
        effects: list[torch.Tensor] = []
        temperatures: list[torch.Tensor] = []
        powers: list[torch.Tensor] = []
        heating_powers: list[torch.Tensor] = []
        cooling_powers: list[torch.Tensor] = []
        hvac_active: list[torch.Tensor] = []
        room_index = self.feature_indices["room_temperature"]

        for step in range(self.history_steps + self.horizon_steps):
            features = legacy_inputs[:, step, :].clone()
            if self.interval_start_inputs and step == self.history_steps - 1:
                # Anchor the state at the origin and apply the interval driver on the first forecast step.
                last_base = features[:, room_index] - last_effect
                continue
            if self.interval_start_inputs and step >= self.history_steps:
                # The current BOPTEST interval drives the next temperature label.
                features = legacy_inputs[:, step - 1, :].clone()
            warm_start = step < self.history_steps
            if warm_start:
                base_input = features[:, room_index] - last_effect
            else:
                base_input = last_base
            features[:, room_index] = base_input

            last_effect, power = self._physical_step_legacy(
                features,
                base_input,
                last_effect,
            )
            last_base, state = self._base_step(features, state)
            temperature = last_base + last_effect

            if not warm_start:
                bases.append(self._from_legacy_value(last_base))
                effects.append(self._effect_from_legacy(last_effect))
                temperatures.append(self._from_legacy_value(temperature))
                powers.append(self._from_legacy_value(power))
                channels = self._split_hvac_power(powers[-1])
                heating_powers.append(channels["heating_power_w"])
                cooling_powers.append(channels["cooling_power_w"])
                hvac_active.append(channels["hvac_active"])

        return {
            "base": torch.stack(bases, dim=1),
            "effect": torch.stack(effects, dim=1),
            "temperature": torch.stack(temperatures, dim=1),
            "power": torch.stack(powers, dim=1),
            "heating_power_w": torch.stack(heating_powers, dim=1),
            "cooling_power_w": torch.stack(cooling_powers, dim=1),
            "hvac_active": torch.stack(hvac_active, dim=1),
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the scaled temperature forecast expected by the shared trainer."""
        return self.rollout_components(inputs)["temperature"]


class PCNNEstimator(BaseEstimator):
    """Adapt reference-compatible PCNN to common fitting and prediction."""

    name = "pcnn"
    implementation_version = 2
    required_features = THERMAL_FEATURES

    def __init__(
        self,
        hidden_size: int = 512,
        layers: int = 2,
        input_hidden_sizes: tuple[int, ...] | list[int] = (128, 128),
        output_hidden_sizes: tuple[int, ...] | list[int] = (128, 128),
        learn_initial_hidden_states: bool = True,
        base_division_factor: float = 10.0,
    ) -> None:
        # Store all architecture settings needed to rebuild the network.
        for value, name in ((hidden_size, "hidden_size"), (layers, "layers")):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        for sizes, name in (
            (input_hidden_sizes, "input_hidden_sizes"),
            (output_hidden_sizes, "output_hidden_sizes"),
        ):
            if (
                not isinstance(sizes, (list, tuple))
                or not sizes
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 1
                    for value in sizes
                )
            ):
                raise ValueError(f"{name} must be a nonempty sequence of positive integers.")
        if not isinstance(learn_initial_hidden_states, bool):
            raise ValueError('learn_initial_hidden_states must be Boolean.')
        if (
            not isinstance(base_division_factor, (int, float))
            or isinstance(base_division_factor, bool)
            or not math.isfinite(float(base_division_factor))
            or float(base_division_factor) <= 0.0
        ):
            raise ValueError('base_division_factor must be a finite positive number.')
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_hidden_sizes = tuple(input_hidden_sizes)
        self.output_hidden_sizes = tuple(output_hidden_sizes)
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.base_division_factor = float(base_division_factor)
        self.network: PCNNForecaster | None = None

    def _build(self, context: ModelContext, seed: int) -> PCNNForecaster:
        """Build PCNN using scaling bounds from the model context."""
        # Seed before creating network parameters.
        initialize_seed(seed)
        feature_indices = {
            name: context.feature_index(name)
            for name in self.required_features
        }
        # Map physical zero power to the public HVAC scale.
        zero_power_scaled = (
            0.0 - context.scaler.minimum["hvac_power"]
        ) / context.scaler.span["hvac_power"]
        # Keep separate indoor and outdoor scales to calculate exchange in Celsius.
        temperature_scaling = {
            name: (
                context.scaler.minimum[name],
                context.scaler.span[name],
            )
            for name in (
                "room_temperature",
                "outdoor_temperature",
            )
        }
        return PCNNForecaster(
            feature_count=len(context.feature_names),
            history_steps=context.history_steps,
            horizon_steps=context.horizon_steps,
            feature_indices=feature_indices,
            hidden_size=self.hidden_size,
            layers=self.layers,
            zero_power_scaled=zero_power_scaled,
            temperature_scaling=temperature_scaling,
            interval_minutes=context.interval_minutes,
            power_span=context.scaler.span["hvac_power"],
            input_hidden_sizes=self.input_hidden_sizes,
            output_hidden_sizes=self.output_hidden_sizes,
            learn_initial_hidden_states=self.learn_initial_hidden_states,
            base_division_factor=self.base_division_factor,
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
        """Fit PCNN in the common scaled space."""
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
        """Predict room temperature in Celsius."""
        if self.network is None:
            raise ValueError('Fit PCNN first.')
        scaled = predict_torch_model(
            self.network,
            windows.scaled_inputs,
            training_config.batch_size,
            training_config.device,
        )
        return context.scaler.inverse_target(scaled)

    def state_payload(self) -> Mapping[str, object]:
        """Return the architecture and weights needed to rebuild PCNN."""
        if self.network is None:
            raise ValueError('Fit PCNN before saving state.')
        return {
            "state_dict": self.network.state_dict(),
            "implementation_version": self.implementation_version,
            "interval_start_inputs": self.network.interval_start_inputs,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "input_hidden_sizes": list(self.input_hidden_sizes),
            "output_hidden_sizes": list(self.output_hidden_sizes),
            "learn_initial_hidden_states": self.learn_initial_hidden_states,
            "base_division_factor": self.base_division_factor,
        }

    def parameter_count(self) -> int:
        """Count PCNN trainable parameters."""
        if self.network is None:
            return 0
        return sum(
            parameter.numel()
            for parameter in self.network.parameters()
        )

    def describe_architecture(self) -> Mapping[str, object]:
        """Describe the reference-compatible D/E architecture."""
        data_features = list(PCNNForecaster.DATA_FEATURES)
        input_sizes = [len(data_features), *self.input_hidden_sizes]
        output_sizes = [self.hidden_size, *self.output_hidden_sizes, 1]
        states = [
            'Local D-branch LSTM state initialized for each forward pass',
            'Accumulated E effect',
            'Previous base temperature prediction',
        ]
        if self.learn_initial_hidden_states:
            states.insert(0, 'Learnable initial states for the D branch')
        return {
            "name": "PCNN (Seven-feature adaptation)",
            "inputs": list(self.required_features),
            "data_features": data_features,
            "input_hidden_sizes": list(self.input_hidden_sizes),
            "lstm_hidden_size": self.hidden_size,
            "lstm_layers": self.layers,
            "output_hidden_sizes": list(self.output_hidden_sizes),
            "learn_initial_hidden_states": self.learn_initial_hidden_states,
            "base_division_factor": self.base_division_factor,
            "implementation_version": self.implementation_version,
            "activation": {"input": "ReLU", "output": "Tanh"},
            "normalization": "LayerNorm",
            "states": states,
            "modules": [
                'Input MLP: ' + " → ".join(map(str, input_sizes)),
                f"D-LSTM:{self.hidden_size}×{self.layers}",
                'Base temperature head: ' + " → ".join(map(str, output_sizes)),
                'Physics E accumulator in the reference scale',
            ],
            "constraints": [
                'a/b/d use exp(log_weight) to remain positive',
                'Convert net thermal power to watts before splitting nonnegative heating and nonpositive cooling',
                'Exclude indoor temperature, outdoor temperature, and HVAC power from the neural increment',
                'No extra valve feature or second scaling of net thermal power',
                'Closed-loop recurrence in the reference [0.1, 0.9] scale',
                f"Divide each D increment by {self.base_division_factor:g}",
            ],
            "physical_initialization": {
                "reference_power": 'Maximum absolute power from the training scaling bounds (W)',
                "hvac_prior": '2°C change in four hours at reference power',
                "outdoor_prior": '1.5°C loss in six hours at a 25°C temperature difference',
                "interval": 'Discretize using the actual sampling interval',
            },
            "output": 'Multi-step room temperature T = D + E (°C)',
        }
