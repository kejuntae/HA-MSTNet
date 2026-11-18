"""Reference four-module ModNN and its seven-feature adapter.

The core keeps the reference feature order and returns temperatures and heat flows. The
adapter maps project inputs and scaling to the reference implementation.
"""

from __future__ import annotations


import math
from numbers import Real
from typing import Callable, Mapping

import numpy as np

import torch
from torch import nn

from exp.training import fit_torch_model, initialize_seed, predict_torch_model
from util.model_features import THERMAL_FEATURES

from models.base import BaseEstimator, FitResult, ModelContext
from util.config import TrainingConfig
from util.features import ThermalWindows


# ---------------------------------------------------------------------------
# Reference thermal process modules.
# ---------------------------------------------------------------------------


def _validate_positive_int(value: int, name: str) -> None:
    """Require a positive integer architecture dimension."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _normalize_constraints(
    constraints: tuple[str, ...] | list[str], external_model: str,
) -> tuple[str, ...]:
    """Validate and order constraint names as in the reference implementation."""
    order = ("hvac", "internal", "ambient", "solar")
    if (not isinstance(constraints, (tuple, list))
            or any(not isinstance(name, str) for name in constraints)):
        raise ValueError('constraints must be a list or tuple of constraint names.')
    unknown = set(constraints).difference(order)
    if unknown:
        raise ValueError(f"Unknown ModNN constraints: {sorted(unknown)}.")
    if {"ambient", "solar"}.intersection(constraints) and external_model != "RNN":
        raise ValueError("The ambient and solar constraints require external_model='RNN'.")
    return tuple(name for name in order if name in constraints)


class OriginalZone(nn.Module):
    """Convert total heat flow into a room-temperature increment."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        _validate_positive_int(input_size, "input_size")
        _validate_positive_int(output_size, "output_size")
        # No bias: zero heat flow must yield zero temperature change.
        self.scale = nn.Linear(input_size, output_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the discrete temperature increment from total heat flow."""
        return self.scale(inputs)


class OriginalInternal(nn.Module):
    """Estimate internal gains from time of day and occupancy."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        _validate_positive_int(input_size, "input_size")
        _validate_positive_int(hidden_size, "hidden_size")
        _validate_positive_int(output_size, "output_size")
        if input_size < 2:
            raise ValueError('input_size must be at least 2.')
        # FC1 reads the two time features, not occupancy.
        self.FC1 = nn.Linear(input_size - 1, hidden_size)
        # FC2 maps the schedule embedding to one internal-gain channel.
        self.FC2 = nn.Linear(hidden_size, output_size)
        # Keep layer names and biases aligned with ModNN_data.py.
        self.scale = nn.Linear(1, 1, bias=False)
        self.relu = nn.ReLU()
        # The reference defines Sigmoid but does not call it in this forward pass.
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Combine the schedule embedding and occupancy, then scale the heat flow."""
        # FC1/ReLU/FC2 maps the time features from [B, T, 2] to [B, T, 1].
        embedding = self.relu(self.FC1(inputs[:, :, :-1]))
        # Add occupancy directly to the learned schedule.
        embedding = self.FC2(embedding) + inputs[:, :, -1:]
        # A bias-free scale maps the combined schedule to heat flow.
        return self.scale(embedding)


class OriginalHVAC(nn.Module):
    """Map signed HVAC power to room heat flow."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        _validate_positive_int(input_size, "input_size")
        _validate_positive_int(output_size, "output_size")
        self.scale = nn.Linear(input_size, output_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a linear mapping to signed HVAC power."""
        return self.scale(inputs)


class OriginalExternal(nn.Module):
    """Model envelope inertia and outdoor exchange with a recurrent state."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, model_type: str
    ) -> None:
        super().__init__()
        _validate_positive_int(input_size, "input_size")
        _validate_positive_int(hidden_size, "hidden_size")
        _validate_positive_int(output_size, "output_size")
        # All three recurrent cell types share the external heat-flow interface.
        recurrent_types = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
        if model_type not in recurrent_types:
            raise ValueError('external_model must be RNN, GRU, or LSTM.')
        self.ext_mdl = recurrent_types[model_type](
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        # Map the last recurrent output to the current external heat flow.
        # Keep the reference attribute names for state_dict compatibility.
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Solar layers produce a hidden-state update, not a direct heat flow.
        self.solar1 = nn.Linear(1, hidden_size)
        self.solar2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
        # Higher room temperature reduces gains; higher outdoor temperature increases them.
        self.register_buffer("in_sign", torch.tensor([-1.0, 1.0]))

    def forward(
        self,
        inputs: torch.Tensor,
        hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Compute external heat flow and inject solar gains into the next hidden state."""
        # Recurrent inputs are indoor and outdoor temperatures: [B, window, 2].
        recurrent_output, hidden = self.ext_mdl(inputs[:, :, [0, 1]], hidden_state)
        # Use the last output to get the current heat flow with shape [B, 1, 1].
        output = self.fc2(self.relu(self.fc1(recurrent_output[:, -1:, :])))
        # Inject solar gains at the final step only; the state is [1, B, hidden].
        solar_gain = torch.tanh(
            self.solar2(self.relu(self.solar1(inputs[:, -1:, [2]])))
        ).permute(1, 0, 2)
        if isinstance(hidden, tuple):
            # For LSTM, update only the hidden state and leave the cell state unchanged.
            recurrent_hidden, cell = hidden
            return output, (recurrent_hidden + solar_gain, cell)
        # For RNN/GRU, add the solar gains to the single hidden state.
        return output, hidden + solar_gain

    @torch.no_grad()
    def apply_constraints(self, ambient: bool, solar: bool) -> None:
        """Apply the selected sign constraints to the environment and solar paths."""
        if not isinstance(self.ext_mdl, nn.RNN):
            raise ValueError('Ambient and solar constraints are supported only by a plain RNN.')
        if ambient:
            self.ext_mdl.weight_ih_l0.copy_(
                self.ext_mdl.weight_ih_l0.abs() * self.in_sign,
            )
        if ambient or solar:
            self.ext_mdl.weight_hh_l0.clamp_(min=0)
            self.fc1.weight.clamp_(min=0)
            self.fc2.weight.clamp_(min=0)
        if solar:
            self.solar1.weight.clamp_(min=0)
            self.solar2.weight.clamp_(min=0)


class OriginalMODNNCore(nn.Module):
    """Roll out the reference ModNN with interval-start or target-row inputs.

    Use measured temperatures as residual anchors in history and predicted temperatures
    during forecasting.
    """

    def __init__(
        self,
        history_steps: int,
        internal_hidden: int = 8,
        external_hidden: int = 16,
        external_model: str = "RNN",
        window: int = 1,
        constraints: tuple[str, ...] | list[str] = ("hvac", "internal"),
        interval_start_inputs: bool = True,
    ) -> None:
        super().__init__()
        # history_steps sets the encoding length; window sets the external lookback.
        _validate_positive_int(history_steps, "history_steps")
        _validate_positive_int(internal_hidden, "internal_hidden")
        _validate_positive_int(external_hidden, "external_hidden")
        _validate_positive_int(window, "window")
        if window >= history_steps:
            raise ValueError('window must be smaller than history_steps.')
        self.history_steps = history_steps
        self.window = window
        self.external_model = external_model
        self.constraints = _normalize_constraints(constraints, external_model)
        if not isinstance(interval_start_inputs, bool):
            raise ValueError('interval_start_inputs must be Boolean.')
        self.interval_start_inputs = interval_start_inputs
        # Preserve the reference top-level names for state_dict compatibility.
        self.Ext = OriginalExternal(2, external_hidden, 1, external_model)
        self.Zone = OriginalZone(1, 1)
        self.HVAC = OriginalHVAC(1, 1)
        self.Int = OriginalInternal(3, internal_hidden, 1)

    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Keep Zone nonnegative and apply configured sign constraints elsewhere."""
        self.Zone.scale.weight.clamp_(min=0)
        if "internal" in self.constraints:
            self.Int.scale.weight.clamp_(min=0)
        if "hvac" in self.constraints:
            self.HVAC.scale.weight.clamp_(min=0)
        if {"ambient", "solar"}.intersection(self.constraints):
            self.Ext.apply_constraints(
                ambient="ambient" in self.constraints,
                solar="solar" in self.constraints,
            )

    def _validate_inputs(self, original_inputs: torch.Tensor) -> None:
        """Validate the seven reference input features before rollout."""
        # Require at least one forecast step after history.
        if not isinstance(original_inputs, torch.Tensor):
            raise ValueError('ModNN inputs must be PyTorch tensors.')
        if (
            original_inputs.ndim != 3 or original_inputs.shape[2] != 7
            or original_inputs.shape[1] <= self.history_steps
        ):
            raise ValueError('ModNN inputs must have shape [batch, steps, 7], with steps greater than history_steps.')
        if original_inputs.dtype != torch.float32:
            raise ValueError('ModNN input dtype must be torch.float32.')
        if not bool(torch.isfinite(original_inputs).all()):
            raise ValueError('ModNN inputs must all be finite.')

    def _initial_external_state(
        self, original_inputs: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Initialize the reference external hidden state to ones on the input device."""
        # The reference initializes to ones on the input device and dtype.
        shape = (1, original_inputs.shape[0], self.Ext.hidden_size)
        hidden = original_inputs.new_ones(shape)
        if self.external_model == "LSTM":
            return hidden, original_inputs.new_ones(shape)
        return hidden

    def forward(
        self, original_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return full trajectories of temperature, HVAC, and the other heat-flow terms."""
        self._validate_inputs(original_inputs)
        # Slice feature views without copying the inputs.
        # Start with outdoor temperature and solar radiation; append room temperature later.
        external_features = original_inputs[:, :, [1, 2]]
        # Internal gains use time sine, time cosine, and occupancy.
        internal_inputs = original_inputs[:, :, [3, 4, 5]]
        # Keep HVAC as a single-channel input for its linear layer.
        hvac_inputs = original_inputs[:, :, [6]]

        # Preallocate full trajectories on the input device and dtype.
        temperature = torch.zeros_like(original_inputs[:, :, [0]])
        hvac_fluxes = torch.zeros_like(hvac_inputs)
        external_fluxes = torch.zeros_like(hvac_inputs)
        internal_fluxes = torch.zeros_like(hvac_inputs)
        total_fluxes = torch.zeros_like(hvac_inputs)
        hidden = self._initial_external_state(original_inputs)

        # Use measured temperature and HVAC during lookback, before heat-flow rollout.
        for index in range(self.window):
            temperature[:, index, :] = original_inputs[:, index, [0]]
            hvac_fluxes[:, index, :] = hvac_inputs[:, index, :]

        # Start from the first measurement after lookback and keep state shape [B, 1, 1].
        estimated_temperature = original_inputs[:, [[self.window]], [0]]
        # Interval-start data use the current row to drive the next step; target-row data use the next row.
        driver_shift = int(not self.interval_start_inputs)

        # History encoding
        # Use estimated temperature for external heat flow, but anchor residuals to observed history.
        for index in range(self.window, self.history_steps):
            # Write the previous estimate at the current step.
            temperature[:, index, :] = estimated_temperature.squeeze(1)
            # External inputs are estimated room temperature, outdoor temperature, and solar radiation.
            # Use the lookback window ending at the current step.
            start = index - self.window + 1
            driver = index + driver_shift
            driver_start = driver - self.window + 1
            external_input = torch.cat(
                [temperature[:, start:index + 1, :],
                 external_features[:, driver_start:driver + 1, :]],
                dim=2,
            )
            # External heat flow also updates the envelope's recurrent state.
            external_flux, hidden = self.Ext(external_input, hidden)
            # Read HVAC and internal gains from the aligned driver row.
            hvac_flux = self.HVAC(hvac_inputs[:, driver:driver + 1, :])
            internal_flux = self.Int(internal_inputs[:, driver:driver + 1, :])
            # Sum the three heat flows before applying the Zone module.
            total_flux = external_flux + internal_flux + hvac_flux

            # Retain individual heat flows for diagnostics.
            hvac_fluxes[:, driver, :] = hvac_flux.squeeze(1)
            external_fluxes[:, driver, :] = external_flux.squeeze(1)
            internal_fluxes[:, driver, :] = internal_flux.squeeze(1)
            total_fluxes[:, driver, :] = total_flux.squeeze(1)
            # Anchor the history residual to the current measured temperature.
            # T_est(index+1) = T_measured(index) + Zone(Q_total(driver)).
            estimated_temperature = (
                original_inputs[:, index:index + 1, [0]]
                + self.Zone(total_flux)
            )

        # Forecast decoding
        # Decode using model state only; future temperature placeholders are ignored.
        for index in range(self.history_steps, original_inputs.shape[1]):
            # Write the current prediction before computing its external heat flow.
            temperature[:, index, :] = estimated_temperature.squeeze(1)
            driver = index + driver_shift
            if driver >= original_inputs.shape[1]:
                # Stop when the final temperature is written and no next driver is available.
                break
            start = index - self.window + 1
            driver_start = driver - self.window + 1
            external_input = torch.cat(
                [temperature[:, start:index + 1, :],
                 external_features[:, driver_start:driver + 1, :]],
                dim=2,
            )
            external_flux, hidden = self.Ext(external_input, hidden)
            hvac_flux = self.HVAC(hvac_inputs[:, driver:driver + 1, :])
            internal_flux = self.Int(internal_inputs[:, driver:driver + 1, :])
            total_flux = external_flux + internal_flux + hvac_flux

            hvac_fluxes[:, driver, :] = hvac_flux.squeeze(1)
            external_fluxes[:, driver, :] = external_flux.squeeze(1)
            internal_fluxes[:, driver, :] = internal_flux.squeeze(1)
            total_fluxes[:, driver, :] = total_flux.squeeze(1)
            # Update the forecast in closed loop.
            # T_est(index+1) = T_est(index) + Zone(Q_total(driver)).
            estimated_temperature = estimated_temperature + self.Zone(total_flux)

        # Keep the full trajectory for the difference loss, including the history-to-forecast boundary.
        return temperature, hvac_fluxes, (external_fluxes, internal_fluxes, total_fluxes)


def build_modnn_loss(
    history_steps: int, diff_alpha: float
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Combine forecast temperature MSE with MAE of adjacent temperature changes.

    Use the full trajectory for the difference term, including the history-to-forecast
    boundary.
    """
    # Use history_steps to locate the first forecast target.
    _validate_positive_int(history_steps, "history_steps")
    if (
        not isinstance(diff_alpha, Real) or isinstance(diff_alpha, bool)
        or not math.isfinite(float(diff_alpha)) or float(diff_alpha) < 0.0
    ):
        raise ValueError('diff_alpha must be a finite nonnegative real number.')
    # Capture a plain scalar weight without adding model parameters.
    weight = float(diff_alpha)

    def loss_function(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute forecast MSE and full-window temperature-difference MAE."""
        # Require matching shapes and at least one forecast step.
        if (
            predictions.ndim != 2 or targets.ndim != 2 or predictions.shape != targets.shape
            or predictions.shape[1] <= history_steps
        ):
            raise ValueError('ModNN loss requires matching two-dimensional tensors that include forecast steps.')
        # Measure absolute temperature error on forecast steps only.
        temperature_loss = nn.functional.mse_loss(
            predictions[:, history_steps:], targets[:, history_steps:]
        )
        # Compare neighboring temperature changes across the full trajectory.
        difference_loss = torch.mean(
            torch.abs(torch.diff(predictions, dim=1) - torch.diff(targets, dim=1))
        )
        # Weight the change error with diff_alpha.
        return temperature_loss + weight * difference_loss

    return loss_function


class MODNNForecaster(nn.Module):
    """Adapt the common seven-feature input to the reference ModNN scales."""

    # The project's canonical input order.
    REQUIRED_FEATURES = THERMAL_FEATURES
    # The reference model expects a fixed seven-feature order.
    ORIGINAL_FEATURE_ORDER = (
        "room_temperature",
        "outdoor_temperature",
        "solar_irradiance",
        "time_sin",
        "time_cos",
        "occupancy",
        "hvac_power",
    )

    def __init__(
        self,
        feature_count: int, history_steps: int, horizon_steps: int,
        feature_indices: Mapping[str, int],
        internal_hidden: int, external_hidden: int, external_model: str, window: int,
        legacy_scales: tuple[float, ...],
        legacy_offsets: tuple[float, ...],
        constraints: tuple[str, ...] | list[str] = ("hvac", "internal"),
        interval_start_inputs: bool = False,
    ) -> None:
        super().__init__()
        for value, name in (
            (feature_count, "feature_count"),
            (history_steps, "history_steps"),
            (horizon_steps, "horizon_steps"),
        ):
            _validate_positive_int(value, name)
        # Resolve the actual position of each named feature.
        missing = [name for name in self.REQUIRED_FEATURES if name not in feature_indices]
        if missing:
            raise ValueError(f"ModNN is missing feature indices: {', '.join(missing)}.")
        # Save indices for a single reorder operation in forward.
        indices = tuple(int(feature_indices[name]) for name in self.ORIGINAL_FEATURE_ORDER)
        if any(index < 0 or index >= feature_count for index in indices):
            raise ValueError('ModNN feature indices are outside the input feature range.')
        # Require seven distinct input columns.
        if len(set(indices)) != len(indices):
            raise ValueError('The seven required ModNN features must map to distinct input columns.')
        if len(legacy_scales) != 7 or len(legacy_offsets) != 7:
            raise ValueError('legacy_scales and legacy_offsets must each contain seven values.')
        # Reject nonfinite affine coefficients before recurrent processing.
        affine_values = np.asarray((*legacy_scales, *legacy_offsets), dtype=float)
        if not np.isfinite(affine_values).all():
            raise ValueError('ModNN affine coefficients must all be finite.')

        self.feature_count = feature_count
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.feature_indices = dict(feature_indices)
        self.original_feature_indices = indices
        self.external_model = external_model
        # Buffers follow device transfers and checkpoints but are not trainable.
        self.register_buffer("legacy_scales", torch.tensor(legacy_scales, dtype=torch.float32))
        self.register_buffer("legacy_offsets", torch.tensor(legacy_offsets, dtype=torch.float32))
        # Preserve the reference module names and connectivity.
        self.core = OriginalMODNNCore(
            history_steps=history_steps,
            internal_hidden=internal_hidden,
            external_hidden=external_hidden,
            external_model=external_model,
            window=window,
            constraints=constraints,
            interval_start_inputs=interval_start_inputs,
        )
        self.interval_start_inputs = interval_start_inputs

    def _validate_public_inputs(
        self, inputs: torch.Tensor, require_full_window: bool
    ) -> None:
        """Validate the public input shape, numeric type, and finiteness."""
        if (
            not isinstance(inputs, torch.Tensor) or inputs.ndim != 3
            or inputs.shape[2] != self.feature_count
        ):
            raise ValueError(f"Public ModNN inputs must have shape [batch, steps, {self.feature_count}].")
        expected_steps = self.history_steps + self.horizon_steps
        if require_full_window and inputs.shape[1] != expected_steps:
            raise ValueError(f"Public ModNN input sequence length must be {expected_steps}.")
        if inputs.dtype != torch.float32:
            raise ValueError('Public ModNN input dtype must be torch.float32.')
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError('Public ModNN inputs must all be finite.')

    def _to_original_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reorder and rescale inputs for the reference ModNN."""
        self._validate_public_inputs(inputs, require_full_window=False)
        # Reorder features, then apply the fixed affine coefficients.
        ordered = inputs[:, :, list(self.original_feature_indices)]
        return ordered * self.legacy_scales + self.legacy_offsets

    def forward_components(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return temperature and four full heat-flow trajectories."""
        self._validate_public_inputs(inputs, require_full_window=True)
        return self.core(self._to_original_inputs(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the full scaled temperature trajectory with shape [B, H + P]."""
        full_temperature, _, _ = self.forward_components(inputs)  # Process one input batch.

        # Drop only the singleton output channel to return a 2D tensor.
        return full_temperature.squeeze(-1)

    def enforce_physical_constraints(self) -> None:
        """Apply the selected reference constraints after initialization and optimizer steps."""
        self.core.apply_constraints()


class MODNNEstimator(BaseEstimator):
    """Adapt reference ModNN to the common training and prediction interface."""

    name = "modnn"
    implementation_version = 2
    required_features = MODNNForecaster.REQUIRED_FEATURES

    def __init__(
        self,
        internal_hidden: int = 8, external_hidden: int = 16,
        external_model: str = "RNN", window: int = 1, diff_alpha: float = 0.3,
        enforce_constraints: bool = True,
        constraints: tuple[str, ...] | list[str] = ("hvac", "internal"),
    ) -> None:
        # Internal-gain schedule hidden width.
        _validate_positive_int(internal_hidden, "internal_hidden")
        # Envelope recurrent hidden width.
        _validate_positive_int(external_hidden, "external_hidden")
        # Recent temperature and weather steps used by the external module.
        _validate_positive_int(window, "window")
        if external_model not in {"RNN", "GRU", "LSTM"}:
            raise ValueError('external_model must be RNN, GRU, or LSTM.')
        if (
            not isinstance(diff_alpha, Real) or isinstance(diff_alpha, bool)
            or not math.isfinite(float(diff_alpha)) or float(diff_alpha) < 0.0
        ):
            raise ValueError('diff_alpha must be a finite nonnegative real number.')
        if not isinstance(enforce_constraints, bool):
            raise ValueError('enforce_constraints must be Boolean.')

        # diff_alpha weights difference MAE; enforce_constraints enables post-update projection.
        self.internal_hidden = internal_hidden
        self.external_hidden = external_hidden
        self.external_model = external_model
        self.window = window
        self.diff_alpha = float(diff_alpha)
        self.enforce_constraints = enforce_constraints
        self.constraints = _normalize_constraints(constraints, external_model)
        # Leave fitted state unset until the network is built.
        self.shared_temperature_minimum: float | None = None
        self.shared_temperature_span: float | None = None
        self.network: MODNNForecaster | None = None

    def _legacy_affine(
        self, context: ModelContext
    ) -> tuple[tuple[float, ...], tuple[float, ...], float, float]:
        """Build the affine mapping from project input scales to reference feature scales."""
        # Ensure all reference features exist in the shared context.
        for feature_name in self.required_features:
            context.feature_index(feature_name)

        # Scale indoor and outdoor temperatures using their combined training range.
        minimum = context.scaler.minimum
        span = context.scaler.span
        room_minimum = float(minimum["room_temperature"])
        room_maximum = room_minimum + float(span["room_temperature"])
        outdoor_minimum = float(minimum["outdoor_temperature"])
        outdoor_maximum = outdoor_minimum + float(span["outdoor_temperature"])
        # Include the full training range of both temperature features.
        shared_minimum = min(room_minimum, outdoor_minimum)
        shared_maximum = max(room_maximum, outdoor_maximum)
        shared_span = shared_maximum - shared_minimum
        if not math.isfinite(shared_span) or shared_span <= 0.0:
            raise ValueError('The shared indoor and outdoor temperature span must be finite and positive.')

        # Raw time sine and cosine values must be within [-1, 1].
        for feature_name in ("time_sin", "time_cos"):
            lower = float(minimum[feature_name])
            upper = lower + float(span[feature_name])
            if lower < -1.0 - 1e-6 or upper > 1.0 + 1e-6:
                raise ValueError(f"{feature_name} must have training bounds within [-1, 1].")

        def temperature_affine(feature_name: str) -> tuple[float, float]:
            """Map a temperature column to the shared indoor/outdoor [-1, 1] range."""
            current_minimum = float(minimum[feature_name])
            current_span = float(span[feature_name])
            # Compose the inverse public scaling with the shared reference [-1, 1] transform.
            scale = 2.0 * current_span / shared_span
            offset = 2.0 * (current_minimum - shared_minimum) / shared_span - 1.0
            return scale, offset

        room_scale, room_offset = temperature_affine("room_temperature")
        outdoor_scale, outdoor_offset = temperature_affine("outdoor_temperature")

        # Compose the inverse public time scaling with the Internal module's [0, 1] mapping.
        time_sin_scale = float(span["time_sin"]) / 2.0
        time_sin_offset = float(minimum["time_sin"]) / 2.0 + 0.5
        time_cos_scale = float(span["time_cos"]) / 2.0
        time_cos_offset = float(minimum["time_cos"]) / 2.0 + 0.5

        # Keep the reference solar and occupancy mappings; anchor HVAC at physical zero.
        power_minimum = float(minimum["hvac_power"])
        power_span = float(span["hvac_power"])
        reference_power = max(abs(power_minimum),
                              abs(power_minimum + power_span))
        if not math.isfinite(reference_power) or reference_power <= 0:
            raise ValueError('HVAC reference power must be finite and positive.')
        power_scale = power_span / reference_power
        power_offset = power_minimum / reference_power
        scales = (room_scale, outdoor_scale, 2.0, time_sin_scale,
                  time_cos_scale, 2.0, power_scale)
        offsets = (
            room_offset, outdoor_offset, -1.0, time_sin_offset,
            time_cos_offset, -1.0, power_offset,
        )
        return scales, offsets, shared_minimum, shared_span

    def _build(self, context: ModelContext, seed: int) -> MODNNForecaster:
        """Build the reference core and its fixed scaling adapter."""
        # Seed the random generators before building the layers.
        initialize_seed(seed)
        legacy_scales, legacy_offsets, shared_minimum, shared_span = self._legacy_affine(context)

        # Share the temperature range for target scaling and prediction inversion.
        self.shared_temperature_minimum = shared_minimum
        self.shared_temperature_span = shared_span

        # Record only the required feature indices.
        indices = {name: context.feature_index(name) for name in self.required_features}

        network = MODNNForecaster(
            feature_count=len(context.feature_names), history_steps=context.history_steps,
            horizon_steps=context.horizon_steps, feature_indices=indices,
            internal_hidden=self.internal_hidden, external_hidden=self.external_hidden,
            external_model=self.external_model, window=self.window,
            legacy_scales=legacy_scales, legacy_offsets=legacy_offsets,
            constraints=self.constraints,
            interval_start_inputs=context.interval_start_inputs,
        )
        if self.enforce_constraints:
            network.enforce_physical_constraints()
        return network

    def _full_legacy_targets(
        self,
        windows: ThermalWindows,
        context: ModelContext,
    ) -> np.ndarray:
        """Join historical and future Celsius targets and apply the shared reference scale."""
        if self.shared_temperature_minimum is None or self.shared_temperature_span is None:
            _, _, shared_minimum, shared_span = self._legacy_affine(context)
        else:
            shared_minimum = self.shared_temperature_minimum
            shared_span = self.shared_temperature_span
        # Use unscaled historical temperature in Celsius.
        room_index = context.feature_index("room_temperature")
        history_temperature = windows.raw_inputs[:, :context.history_steps, room_index]
        # Concatenate H history steps and P forecast targets.
        full_temperature = np.concatenate((history_temperature, windows.targets_c), axis=1)
        # Map the full target trajectory to the shared reference [-1, 1] range.
        legacy = 2.0 * (full_temperature - shared_minimum) / shared_span - 1.0
        return np.asarray(legacy, dtype=np.float32)

    def fit(
        self,
        train: ThermalWindows,
        validation: ThermalWindows,
        context: ModelContext,
        training_config: TrainingConfig,
        seed: int,
    ) -> FitResult:
        """Fit ModNN with the reference full-trajectory objective."""
        # Reinitialize the network at each fit.
        self.network = self._build(context, seed)

        # The shared trainer applies constraints after each optimizer update.
        constraint_callback = (
            self.network.enforce_physical_constraints if self.enforce_constraints else None
        )

        # The adapter remaps public scaled windows during forward.
        # The reference loss requires targets for all H + P steps.
        outcome = fit_torch_model(
            self.network, train.scaled_inputs, self._full_legacy_targets(train, context),
            validation.scaled_inputs, self._full_legacy_targets(validation, context),
            training_config, seed,
            constraint_callback=constraint_callback,
            loss_function=build_modnn_loss(context.history_steps, self.diff_alpha),
        )
        return FitResult(
            outcome.train_losses, outcome.validation_losses, outcome.best_epoch,
            outcome.completed_epochs, outcome.elapsed_seconds,
        )

    def predict(
        self,
        windows: ThermalWindows,
        context: ModelContext,
        training_config: TrainingConfig,
    ) -> np.ndarray:
        """Return forecast room temperatures in Celsius."""
        if (
            self.network is None or self.shared_temperature_minimum is None
            or self.shared_temperature_span is None
        ):
            raise ValueError('Fit ModNN first.')
        # The network returns all H + P steps in the reference [-1, 1] space.
        full_legacy = predict_torch_model(
            self.network, windows.scaled_inputs, training_config.batch_size, training_config.device
        )
        # Remove the historical prefix for the public forecast.
        future_legacy = full_legacy[:, context.history_steps:]
        # Invert the reference temperature scale.
        # T = (x + 1) / 2 * shared_span + shared_minimum.
        return (
            (future_legacy + 1.0) * 0.5 * self.shared_temperature_span
            + self.shared_temperature_minimum
        )

    def state_payload(self) -> Mapping[str, object]:
        """Return weights, scaling, architecture, and constraints needed for restoration."""
        if (
            self.network is None or self.shared_temperature_minimum is None
            or self.shared_temperature_span is None
        ):
            raise ValueError('Fit ModNN before saving state.')
        return {
            "state_dict": self.network.state_dict(),
            "model_name": self.name,
            "implementation_version": self.implementation_version,
            "constraints": list(self.constraints),
            "interval_start_inputs": self.network.interval_start_inputs,
            "hvac_scaling": "zero_anchored_maxabs",
            "internal_hidden": self.internal_hidden,
            "external_hidden": self.external_hidden,
            "external_model": self.external_model,
            "window": self.window,
            "diff_alpha": self.diff_alpha,
            "enforce_constraints": self.enforce_constraints,
            "shared_temperature_minimum": self.shared_temperature_minimum,
            "shared_temperature_span": self.shared_temperature_span,
        }

    def parameter_count(self) -> int:
        """Count trainable parameters in the reference four-module core."""
        if self.network is None:
            return 0
        # Fixed affine buffers are not counted as trainable parameters.
        return sum(parameter.numel() for parameter in self.network.parameters())

    def describe_architecture(self) -> Mapping[str, object]:
        """Describe the reference architecture and the scaling, constraints, and alignment adaptations."""
        # Closed-loop feedback is always active; constraints are optional.
        constraints = ['Closed-loop feedback during forecasting']
        if self.enforce_constraints:
            constraints.append('Nonnegative Zone scale')
            if "internal" in self.constraints:
                constraints.append('Nonnegative internal gains scale')
            if "hvac" in self.constraints:
                constraints.append('Nonnegative HVAC scale')
            if "ambient" in self.constraints:
                constraints.append('Nonpositive indoor input; nonnegative outdoor input and external output path')
            if "solar" in self.constraints:
                constraints.append('Nonnegative solar gains and shared external output path')
        return {
            "name": "MODNN",
            "variant": 'Seven-feature adapter for the reference data-driven ModNN',
            "implementation_version": self.implementation_version,
            "selected_constraints": list(self.constraints),
            "enforce_constraints": self.enforce_constraints,
            "hvac_scaling": "zero_anchored_maxabs",
            "inputs": list(self.required_features),
            "states": [
                'Recurrent envelope thermal state',
                'Solar irradiance updates the external recurrent state',
                'Previous predicted room temperature',
            ],
            "modules": [
                "OriginalExternal",
                "OriginalInternal",
                "OriginalHVAC",
                "OriginalZone",
            ],
            "recurrent_type": self.external_model,
            "window": self.window,
            "solar_window_aggregation": "last",
            "constraints": constraints,
            "loss": f"Forecast MSE + {self.diff_alpha:g} × full-trajectory temperature-difference MAE",
            "output": 'Multi-step forecast of room temperature (°C)',
        }
