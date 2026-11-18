"""Gray-box 3R2C model of indoor air and the envelope.

Three resistances connect air, envelope, and outdoors; both thermal masses can absorb
solar gains. Units are °C, W, J/K, and K/W.
"""

from __future__ import annotations

from datetime import timedelta
import time
from typing import Mapping

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

from models.base import BaseEstimator, FitResult, ModelContext
from util.config import TrainingConfig
from util.features import ThermalWindows
from util.model_features import THERMAL_FEATURES
from util.solar import window_irradiance_from_spans


PARAMETER_NAMES = (
    "air_capacity", "mass_capacity", "outdoor_resistance",
    "mass_resistance", "envelope_resistance", "solar_gain", "mass_solar_gain",
)
# Use J/K for capacitance, K/W for resistance, and m² for solar gain area.
DEFAULT_PARAMETERS = np.array([2.e6, 1.e7, .04, .02, .08, 2., 1.])
LOWER_BOUNDS = np.array([1.e5, 1.e6, 1.e-4, 1.e-4, 1.e-4, 1.e-8, 1.e-8])
UPPER_BOUNDS = np.array([1.e9, 1.e11, 10., 10., 10., 100., 100.])


def discrete_matrices(
    parameters: np.ndarray,
    time_step_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize exactly with a matrix exponential and zero-order-held inputs."""
    values = np.asarray(parameters, dtype=float)
    if (values.shape != (7,) or not np.isfinite(values).all()
            or np.any(values <= 0.)):
        raise ValueError('3R2C requires seven finite positive parameters.')
    if not np.isfinite(time_step_seconds) or time_step_seconds <= 0.:
        raise ValueError('time_step_seconds must be a finite positive number.')
    ca, cm, ro, rm, re, ga, gm = values
    # States are air and envelope temperatures; fluxes conserve energy.
    a = np.array([
        [-(1. / ro + 1. / rm) / ca, 1. / (rm * ca)],
        [1. / (rm * cm), -(1. / rm + 1. / re) / cm],
    ])
    # Inputs are signed HVAC power, irradiance, and outdoor temperature.
    b = np.array([
        [1. / ca, ga / ca, 1. / (ro * ca)],
        [0., gm / cm, 1. / (re * cm)],
    ])
    augmented = np.zeros((5, 5))
    augmented[:2, :2] = a
    augmented[:2, 2:] = b
    discrete = expm(augmented * time_step_seconds)
    return discrete[:2, :2], discrete[:2, 2:]


class ThreeR2CEstimator(BaseEstimator):
    """Identify positive physical parameters and estimate wall temperature from history."""

    name = "3r2c"
    required_features = THERMAL_FEATURES

    def __init__(
        self,
        parameters: Mapping[str, float] | None = None,
        max_fit_windows: int = 1024,
        max_nfev: int = 200,
        rc_window_irr: bool = False,
    ) -> None:
        for value, label in ((max_fit_windows, "max_fit_windows"),
                             (max_nfev, "max_nfev")):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(f"{label} must be a positive integer.")
        if not isinstance(rc_window_irr, bool):
            raise ValueError('rc_window_irr must be Boolean.')
        self.max_fit_windows = max_fit_windows
        self.max_nfev = max_nfev
        self.rc_window_irr = rc_window_irr
        self.parameters = None
        self.fit_diagnostics: dict[str, object] = {}
        if parameters is not None:
            missing = [key for key in PARAMETER_NAMES if key not in parameters]
            if missing:
                raise ValueError(f"3R2C is missing parameters: {', '.join(missing)}.")
            values = np.array(
                [parameters[key] for key in PARAMETER_NAMES], float,
            )
            discrete_matrices(values, 1.)
            self.parameters = dict(zip(PARAMETER_NAMES, map(float, values)))

    @staticmethod
    def _check_context(context: ModelContext) -> None:
        """Validate the horizon and sampling interval."""
        for label in ("history_steps", "horizon_steps", "interval_minutes"):
            value = getattr(context, label)
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(f"{label} must be a positive integer.")

    def _prepare_inputs(self, windows, context, selection):
        """Read historical room temperature and the drivers used for each transition."""
        self._check_context(context)
        indices = {name: context.feature_index(name)
                   for name in self.required_features}
        raw = windows.raw_inputs[selection]
        total = context.history_steps + context.horizon_steps
        if (raw.ndim != 3 or raw.shape[1] != total
                or raw.shape[2] != len(context.feature_names)):
            raise ValueError('3R2C inputs must have shape [windows, history + horizon, features].')
        room = np.asarray(
            raw[:, :context.history_steps, indices["room_temperature"]], float,
        )
        # Forecast T(t+1) through T(t+H) using u(t) through u(t+H-1).
        power = raw[:, :-1, indices["hvac_power"]]
        solar = raw[:, :-1, indices["solar_irradiance"]]
        outdoor = raw[:, :-1, indices["outdoor_temperature"]]
        if not all(np.isfinite(value).all()
                   for value in (room, power, solar, outdoor)):
            raise ValueError('Historical room temperatures and the drivers used by 3R2C must be finite.')
        if self.rc_window_irr:
            sample_indices = np.arange(len(windows.raw_inputs))[selection]
            # Exclude the last unused driver row from irradiance conversion.
            spans = tuple(
                (windows.spans[i][0], windows.spans[i][1]
                 - timedelta(minutes=context.interval_minutes))
                for i in sample_indices
            )
            solar = window_irradiance_from_spans(
                solar, spans, context.interval_minutes,
            )
        return room, np.stack((power, solar, outdoor), axis=-1).astype(float)

    @staticmethod
    def _rollout(parameters, room, inputs, context):
        """Warm up and roll out in batches with the same rules for fitting and inference."""
        ad, bd = discrete_matrices(parameters, context.interval_minutes * 60.)
        history = context.history_steps
        rm, re = parameters[3:5]
        # Initialize wall temperature at passive equilibrium using mean historical boundaries.
        mass = (
            room.mean(axis=1) / rm
            + inputs[:, :history, 2].mean(axis=1) / re
        ) / (1. / rm + 1. / re)
        for step in range(history - 1):
            mass = (ad[1, 0] * room[:, step] + ad[1, 1] * mass
                    + inputs[:, step] @ bd[1])
        state = np.column_stack((room[:, -1], mass))
        predictions = np.empty((len(room), context.horizon_steps))
        for step in range(context.horizon_steps):
            state = state @ ad.T + inputs[:, history - 1 + step] @ bd.T
            predictions[:, step] = state[:, 0]
        return predictions

    def fit(
        self,
        train: ThermalWindows,
        validation: ThermalWindows,
        context: ModelContext,
        training_config: TrainingConfig,
        seed: int,
    ) -> FitResult:
        """Minimize multi-step rollout residuals with log-space parameters."""
        del seed
        self._check_context(context)
        if len(train.raw_inputs) == 0 or len(validation.raw_inputs) == 0:
            raise ValueError('3R2C training and validation windows must not be empty.')
        for windows in (train, validation):
            if (windows.targets_c.shape != (len(windows.raw_inputs),
                                            context.horizon_steps)
                    or not np.isfinite(windows.targets_c).all()):
                raise ValueError('3R2C fitting targets must be a finite [windows, horizon] array.')
        started = time.perf_counter()
        # Sample windows evenly across the training interval; never use the test set.
        count = min(len(train.raw_inputs), self.max_fit_windows)
        selected = np.linspace(0, len(train.raw_inputs) - 1, count, dtype=int)
        room, inputs = self._prepare_inputs(train, context, selected)
        targets = train.targets_c[selected].astype(float)
        initial = (
            DEFAULT_PARAMETERS.copy() if self.parameters is None
            else np.array([self.parameters[key] for key in PARAMETER_NAMES])
        )
        if np.any(initial < LOWER_BOUNDS) or np.any(initial > UPPER_BOUNDS):
            raise ValueError('Initial 3R2C parameters must be within the optimization bounds.')

        def residuals(log_parameters):
            prediction = self._rollout(
                np.exp(log_parameters), room, inputs, context,
            )
            return (prediction - targets).ravel()

        result = least_squares(
            residuals, np.log(initial),
            bounds=(np.log(LOWER_BOUNDS), np.log(UPPER_BOUNDS)),
            loss="soft_l1", f_scale=.1, x_scale="jac", max_nfev=self.max_nfev,
        )
        values = np.exp(result.x)
        self.parameters = dict(zip(PARAMETER_NAMES, map(float, values)))
        self.fit_diagnostics = {
            "success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "nfev": int(result.nfev),
            "cost": float(result.cost), "fit_windows": count,
            "selection": "uniform_training_windows",
            "objective": "multistep_free_run_soft_l1",
            "at_bounds": [name for name, active in
                          zip(PARAMETER_NAMES, result.active_mask) if active],
        }
        train_prediction = self.predict(train, context, training_config)
        validation_prediction = self.predict(
            validation, context, training_config,
        )
        return FitResult(
            train_losses=(float(np.mean(
                (train_prediction - train.targets_c) ** 2,
            )),),
            validation_losses=(float(np.mean(
                (validation_prediction - validation.targets_c) ** 2,
            )),),
            best_epoch=1, completed_epochs=1,
            elapsed_seconds=time.perf_counter() - started,
        )

    def predict(
        self,
        windows: ThermalWindows,
        context: ModelContext,
        training_config: TrainingConfig,
    ) -> np.ndarray:
        """Predict Celsius temperatures in chunks without reading future labels."""
        del training_config
        self._check_context(context)
        if self.parameters is None:
            raise ValueError('Fit 3R2C or supply its parameters first.')
        values = np.array([self.parameters[key] for key in PARAMETER_NAMES])
        count = len(windows.raw_inputs)
        result = np.empty((count, context.horizon_steps))
        # Chunk size limits peak prediction memory without affecting identification.
        for start in range(0, count, 2048):
            selection = slice(start, min(start + 2048, count))
            room, inputs = self._prepare_inputs(windows, context, selection)
            result[selection] = self._rollout(values, room, inputs, context)
        if not np.isfinite(result).all():
            raise ValueError('3R2C produced nonfinite predictions; check parameters and input units.')
        return result

    def state_payload(self) -> Mapping[str, object]:
        """Return the state needed to restore predictions."""
        if self.parameters is None:
            raise ValueError('Fit 3R2C before saving state.')
        return {
            "parameters": dict(self.parameters),
            "max_fit_windows": self.max_fit_windows, "max_nfev": self.max_nfev,
            "rc_window_irr": self.rc_window_irr,
        }

    def physical_metadata(self) -> Mapping[str, object]:
        """Report fitted parameters, units, and convergence information."""
        state = self.state_payload()
        return {
            "model": self.name, **state,
            "parameter_units": dict(zip(PARAMETER_NAMES,
                                        ("J/K", "J/K", "K/W", "K/W", "K/W",
                                         "m^2", "m^2"))),
            "fit_diagnostics": dict(self.fit_diagnostics),
        }

    def parameter_count(self) -> int:
        """Count the five RC and two solar gain parameters."""
        return len(PARAMETER_NAMES)

    def describe_architecture(self) -> Mapping[str, object]:
        """Describe the air and envelope nodes and their three heat transfer paths."""
        return {
            "name": "3R2C", "inputs": list(self.required_features),
            "states": ['Indoor air temperature', 'Envelope temperature'],
            "modules": ['Outdoor-to-indoor air resistance', 'Air-to-envelope resistance',
                        'Envelope-to-outdoor resistance', 'Two thermal capacitances', 'Solar gains to both nodes', "HVAC"],
            "constraints": ['Positive parameters', 'Exact matrix-exponential discretization with zero-order hold'],
            "rc_window_irr": self.rc_window_irr,
            "solar_input": 'Vertical-window irradiance' if self.rc_window_irr else 'Global horizontal irradiance',
            "output": 'Multi-step room temperature (°C)',
        }
