"""Shared estimator contract for the eight models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Mapping

import numpy as np

from util.config import TrainingConfig
from util.features import FeatureScaler, ThermalWindows
from util.model_features import THERMAL_FEATURES


@dataclass(frozen=True)
class ModelContext:
    """Common context for interpreting model windows."""

    # Match feature order to the final input tensor dimension.
    feature_names: tuple[str, ...]
    # History counts observed steps; horizon counts forecast steps.
    history_steps: int
    horizon_steps: int
    # Use the sampling interval to convert steps to physical time.
    interval_minutes: int
    # Share the scaler for inputs and forward/inverse target transforms.
    scaler: FeatureScaler
    # Interval-start inputs drive the next label; the default keeps legacy alignment.
    interval_start_inputs: bool = False

    def __post_init__(self) -> None:
        """Enforce the canonical seven-feature input layout."""
        if self.feature_names != THERMAL_FEATURES:
            raise ValueError('Model inputs must follow the canonical seven-feature order.')
        if self.scaler.feature_names != THERMAL_FEATURES:
            raise ValueError('The scaler must use the canonical seven features.')

    def feature_index(self, name: str) -> int:
        """Return the canonical index or raise a clear error for a missing feature."""
        # Resolve indices through the context instead of hard-coding columns.
        try:
            return self.feature_names.index(name)
        except ValueError as error:
            raise ValueError(f"Model context is missing feature {name}.") from error


@dataclass(frozen=True)
class FitResult:
    """Common result of model fitting or system identification."""

    # Neural models record per-epoch losses; gray-box fits record one loss.
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    # Epoch numbering starts at one to match the training logs.
    best_epoch: int
    completed_epochs: int
    # Measure duration in seconds for comparison.
    elapsed_seconds: float


class BaseEstimator(ABC):
    """Interface for model fitting, prediction, and serialization."""

    # Each implementation declares its registry name and required features.
    name: ClassVar[str]
    required_features: ClassVar[tuple[str, ...]]

    # Share one interface across gray-box and neural models.
    @abstractmethod
    def fit(
        self,
        train: ThermalWindows,
        validation: ThermalWindows,
        context: ModelContext,
        training_config: TrainingConfig,
        seed: int,
    ) -> FitResult:
        """Fit the estimator on training and validation windows."""
        # Concrete models return the common fit result.
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        windows: ThermalWindows,
        context: ModelContext,
        training_config: TrainingConfig,
    ) -> np.ndarray:
        """Return a two-dimensional array of room temperatures in Celsius."""
        # Return [samples, forecast steps] in Celsius.
        raise NotImplementedError

    @abstractmethod
    def state_payload(self) -> Mapping[str, object]:
        """Return serializable model state."""
        # Include weights or physical parameters and the configuration needed to rebuild.
        raise NotImplementedError

    @abstractmethod
    def parameter_count(self) -> int:
        """Count trainable or identified parameters."""
        # Use the count to compare model complexity.
        raise NotImplementedError

    @abstractmethod
    def describe_architecture(self) -> Mapping[str, object]:
        """Return architecture metadata for documentation and diagrams."""
        # Keep metadata serializable without live model objects.
        raise NotImplementedError
