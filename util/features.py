"""Time splits, training-only scaling, and leakage-free model windows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplits:
    """Store training, validation, and test tables."""

    # Preserve row order within each split; pre-split dates need not be ordered across splits.
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class FeatureScaler:
    """Fit per-feature min-max scaling on the training split only."""

    # Use one feature order for scaling and tensor inputs.
    feature_names: tuple[str, ...]
    # Store each training minimum and the difference between maximum and minimum.
    minimum: dict[str, float]
    span: dict[str, float]

    @classmethod
    def fit(
        cls,
        training_frame: pd.DataFrame,
        feature_names: Iterable[str],
        direct_thermal_power: bool = False,
    ) -> "FeatureScaler":
        """Fit finite scaling bounds for each feature, skipping missing values."""
        # Materialize feature order once for one-shot iterators.
        names = tuple(feature_names)
        _require_columns(training_frame, names)
        minimum: dict[str, float] = {}
        span: dict[str, float] = {}
        for name in names:
            # Convert nonnumeric values to NaN and fit only on finite values.
            values = pd.to_numeric(
                training_frame[name],
                errors="coerce",
            ).to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"Training feature {name} has no finite values.")
            # Min-max scaling is (value - minimum) / span.
            lower = float(finite.min())
            width = float(finite.max()) - lower
            minimum[name] = lower
            # Use a span of one for constant features to avoid division by zero.
            span[name] = width if width > 0.0 else 1.0
            if direct_thermal_power and name in {"valve_position", "hvac_mode"}:
                # Preserve fixed physical encodings even when a feature is constant.
                minimum[name] = 0.0 if name == "valve_position" else -1.0
                span[name] = 1.0 if name == "valve_position" else 2.0
        # Reuse this scaler for validation and test data.
        return cls(names, minimum, span)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Scale features in canonical order using the training bounds."""
        _require_columns(frame, self.feature_names)
        # Preserve the time index for window generation.
        transformed = pd.DataFrame(index=frame.index)
        for name in self.feature_names:
            # Do not refit scaling on validation or test data.
            transformed[name] = (
                pd.to_numeric(frame[name], errors="coerce")
                - self.minimum[name]
            ) / self.span[name]
        # Reorder columns explicitly to match model inputs.
        return transformed.loc[:, self.feature_names]

    def transform_target(self, values_c: np.ndarray) -> np.ndarray:
        """Scale Celsius targets using the training room-temperature bounds."""
        # Use the same scale for room-temperature targets and inputs.
        return (
            np.asarray(values_c, dtype=float)
            - self.minimum["room_temperature"]
        ) / self.span["room_temperature"]

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        """Convert scaled room temperatures back to Celsius."""
        # Invert the target transform to return Celsius.
        return (
            np.asarray(values, dtype=float)
            * self.span["room_temperature"]
            + self.minimum["room_temperature"]
        )

    def to_dict(self) -> dict[str, object]:
        """Return checkpoint metadata as plain Python containers."""
        # Copy the containers so callers cannot mutate references in the scaler.
        return {
            "feature_names": list(self.feature_names),
            "minimum": dict(self.minimum),
            "span": dict(self.span),
        }


@dataclass(frozen=True)
class ThermalWindows:
    """Store raw and scaled inputs and Celsius targets for the same windows."""

    # Both input arrays have shape [windows, history + horizon, features].
    raw_inputs: np.ndarray
    scaled_inputs: np.ndarray
    # Targets have shape [windows, horizon] and remain in Celsius.
    targets_c: np.ndarray
    # Each target value has a corresponding real timestamp.
    target_times: np.ndarray
    # Store each full window's start and end for auditing.
    spans: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        """Return the number of valid windows."""
        # Each target row corresponds to one valid window.
        return int(self.targets_c.shape[0])


@dataclass(frozen=True)
class PreparedData:
    """Store the three window sets and their shared training scaler."""

    # Build windows independently for each split with one training scaler.
    train: ThermalWindows
    validation: ThermalWindows
    test: ThermalWindows
    scaler: FeatureScaler
    # Record row counts and candidate and valid window counts.
    summary: dict[str, object]


def _require_columns(frame: pd.DataFrame, names: Iterable[str]) -> None:
    """Reject missing features before indexing by position."""
    # List all missing features in a single error.
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Data is missing features: {', '.join(missing)}.")


def split_by_time(
    frame: pd.DataFrame,
    ratios: tuple[float, float, float],
) -> TimeSplits:
    """Split rows in time order using the given train, validation, and test ratios."""
    # Require three finite positive split ratios.
    if len(ratios) != 3 or not all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
        for value in ratios
    ):
        raise ValueError('Time splitting requires three finite positive ratios.')
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError('Time split ratios must sum to 1.')

    # Round down the first two split sizes and assign remaining rows to test.
    train_end = int(len(frame) * ratios[0])
    validation_end = train_end + int(len(frame) * ratios[1])
    # Require each split to be nonempty.
    if train_end < 1 or validation_end <= train_end or validation_end >= len(frame):
        raise ValueError(f"Too few rows for three nonempty splits; row count: {len(frame)}.")
    # Slice in time order and copy to protect the source data.
    return TimeSplits(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:validation_end].copy(),
        test=frame.iloc[validation_end:].copy(),
    )


def _empty_windows(
    total_steps: int,
    horizon_steps: int,
    feature_names: tuple[str, ...],
) -> ThermalWindows:
    """Return an empty window set with consistent array shapes."""
    # Preserve the expected dimensions even when no window is valid.
    return ThermalWindows(
        raw_inputs=np.empty((0, total_steps, len(feature_names)), dtype=np.float32),
        scaled_inputs=np.empty((0, total_steps, len(feature_names)), dtype=np.float32),
        targets_c=np.empty((0, horizon_steps), dtype=np.float32),
        target_times=np.empty((0, horizon_steps), dtype="datetime64[ns]"),
        spans=(),
        feature_names=feature_names,
    )


def _valid_window_starts(
    frame: pd.DataFrame,
    names: tuple[str, ...],
    total_steps: int,
    interval_minutes: int,
    history_steps: int,
    window_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid starts, numeric values, and nanosecond timestamps."""
    # Extract model features into a NumPy matrix.
    values = frame.loc[:, names].to_numpy(dtype=float)
    # A row is valid only if all required values are finite.
    finite_rows = np.isfinite(values).all(axis=1)

    # Convolve with ones to count valid rows per candidate window.
    finite_counts = np.convolve(
        finite_rows.astype(np.int32),
        np.ones(total_steps, dtype=np.int32),
        mode="valid",
    )
    index_values = frame.index.to_numpy(dtype="datetime64[ns]")
    # A step not matching the configured sampling interval breaks continuity.
    gap_ok = np.diff(index_values) == np.timedelta64(interval_minutes, "m")
    # Never cross episode boundaries, even when their timestamps are adjacent.
    episode_starts = np.zeros(len(frame), dtype=np.int64)
    if "episode_id" in frame:
        episodes = frame["episode_id"].to_numpy()
        same_episode = episodes[1:] == episodes[:-1]
        gap_ok &= same_episode
        boundaries = np.r_[True, ~same_episode]
        episode_starts = np.maximum.accumulate(
            np.where(boundaries, np.arange(len(frame)), 0)
        )
    # A window of total_steps rows has total_steps - 1 intervals.
    gap_counts = np.convolve(
        gap_ok.astype(np.int32),
        np.ones(total_steps - 1, dtype=np.int32),
        mode="valid",
    )
    # Require finite inputs and continuous timestamps for each window.
    starts = np.flatnonzero(
        (finite_counts == total_steps) & (gap_counts == total_steps - 1)
    )
    if "target_origin_allowed" in frame:
        allowed = frame["target_origin_allowed"]
        if allowed.isna().any() or not pd.api.types.is_bool_dtype(allowed):
            raise ValueError('target_origin_allowed must contain Boolean values with no missing entries.')
        # The forecast origin is the final history row.
        origins = starts + history_steps - 1
        starts = starts[allowed.to_numpy(dtype=bool)[origins]]
    # Apply the stride to each episode's original row grid before invalid rows are discarded.
    starts = starts[(starts - episode_starts[starts]) % window_stride == 0]
    return starts, values, index_values


def count_valid_windows(
    frame: pd.DataFrame,
    feature_names: Iterable[str],
    history_steps: int,
    horizon_steps: int,
    interval_minutes: int,
    window_stride: int = 1,
) -> int:
    """Count windows at the given stride without crossing gaps or missing values."""
    # Convert feature names to a reusable tuple.
    names = tuple(feature_names)
    _require_columns(frame, names)
    for value, label in (
        (history_steps, "history_steps"),
        (horizon_steps, "horizon_steps"),
        (interval_minutes, "interval_minutes"),
        (window_stride, "window_stride"),
    ):
        # Reject bool explicitly because it is a subclass of int.
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{label} must be a positive integer.")
    total_steps = history_steps + horizon_steps
    # Too few rows means no candidate windows.
    if len(frame) < total_steps:
        return 0
    starts, _, _ = _valid_window_starts(
        frame,
        names,
        total_steps,
        interval_minutes,
        history_steps,
        window_stride,
    )
    return int(len(starts))


def make_windows(
    frame: pd.DataFrame,
    feature_names: Iterable[str],
    scaler: FeatureScaler,
    history_steps: int,
    horizon_steps: int,
    interval_minutes: int,
    window_stride: int = 1,
) -> ThermalWindows:
    """Build all valid contiguous windows at the given stride in one pass."""
    # Validate parameters before building the window index matrix.
    names = tuple(feature_names)
    _require_columns(frame, names)  # Reject missing feature columns.
    for value, label in (
        (history_steps, "history_steps"),
        (horizon_steps, "horizon_steps"),
        (interval_minutes, "interval_minutes"),
        (window_stride, "window_stride"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{label} must be a positive integer.")
    if "room_temperature" not in names:
        raise ValueError('Window features must include room_temperature.')

    total_steps = history_steps + horizon_steps
    # Return correctly shaped empty arrays when there are too few rows.
    if len(frame) < total_steps:
        return _empty_windows(total_steps, horizon_steps, names)
    # Find the room-temperature column to extract and mask future targets.
    room_index = names.index("room_temperature")
    valid_starts, values, index_values = _valid_window_starts(
        frame,
        names,
        total_steps,
        interval_minutes,
        history_steps,
        window_stride,
    )
    if len(valid_starts) == 0:
        return _empty_windows(total_steps, horizon_steps, names)

    # Broadcast start positions and offsets into a matrix of row indices.
    offsets = np.arange(total_steps)
    row_indices = valid_starts[:, None] + offsets[None, :]
    # Raw inputs have shape [valid windows, total steps, features].
    raw = values[row_indices]

    # Copy future room temperatures as labels before masking them in inputs.
    targets = raw[:, history_steps:, room_index].copy()
    masked = raw.copy()
    # Fill future input temperatures with the final observed temperature.
    masked[:, history_steps:, room_index] = masked[
        :, history_steps - 1:history_steps, room_index
    ]

    # Collect scaling values in feature order.
    minimum = np.array([scaler.minimum[name] for name in names], dtype=float)
    span = np.array([scaler.span[name] for name in names], dtype=float)
    # Broadcast scaling across windows and time steps.
    scaled = (masked - minimum[None, None, :]) / span[None, None, :]
    if not np.isfinite(scaled).all():
        raise ValueError('Valid windows contain nonfinite values after scaling.')
    # Target timestamps start after the history segment.
    target_indices = row_indices[:, history_steps:]
    return ThermalWindows(
        # Mask future temperatures in raw inputs too, so gray-box models cannot read them.
        raw_inputs=masked.astype(np.float32),
        scaled_inputs=scaled.astype(np.float32),
        targets_c=targets.astype(np.float32),
        target_times=index_values[target_indices],
        # Track the source time span of each window.
        spans=tuple(
            (frame.index[start], frame.index[start + total_steps - 1])
            for start in valid_starts
        ),
        feature_names=names,
    )


def prepare_windows(
    frame: pd.DataFrame | TimeSplits,
    feature_names: Iterable[str],
    history_steps: int,
    horizon_steps: int,
    interval_minutes: int,
    split_ratios: tuple[float, float, float] | None = (0.7, 0.1, 0.2),
    window_stride: int = 1,
    direct_thermal_power: bool = False,
    interval_start_inputs: bool = False,
) -> PreparedData:
    """Apply training-only scaling and windowing to one table or pre-split data."""
    names = tuple(feature_names)
    # Use pre-split tables as provided; do not merge and re-split them.
    if isinstance(frame, TimeSplits):
        splits = frame
    else:
        if split_ratios is None:
            raise ValueError('Building windows from one table requires split_ratios.')
        splits = split_by_time(frame, split_ratios)
    # Fit scaling bounds on training only.
    scaler = FeatureScaler.fit(splits.train, names, direct_thermal_power)
    # Build windows within each split, never across boundaries.
    train = make_windows(
        splits.train,
        names,
        scaler,
        history_steps,
        horizon_steps,
        interval_minutes,
        window_stride,
    )
    validation = make_windows(
        splits.validation,
        names,
        scaler,
        history_steps,
        horizon_steps,
        interval_minutes,
        window_stride,
    )
    test = make_windows(
        splits.test,
        names,
        scaler,
        history_steps,
        horizon_steps,
        interval_minutes,
        window_stride,
    )
    # Reject splits with no valid windows.
    for label, windows in (
        ('Training', train),
        ('Validation', validation),
        ('Test', test),
    ):
        if len(windows) == 0:
            raise ValueError(f"{label} split has no valid contiguous windows.")
    total_steps = history_steps + horizon_steps
    # Record candidate and valid counts to show the effect of gaps and missing data.
    return PreparedData(
        train=train,
        validation=validation,
        test=test,
        scaler=scaler,
        summary={
            "feature_names": list(names),
            "window_stride": window_stride,
            "train_rows": int(len(splits.train)),
            "validation_rows": int(len(splits.validation)),
            "test_rows": int(len(splits.test)),
            "train_windows": len(train),
            "validation_windows": len(validation),
            "test_windows": len(test),
            "split_dataset": not isinstance(frame, TimeSplits),
            "direct_thermal_power": direct_thermal_power,
            "interval_start_inputs": interval_start_inputs,
            **{
                f"{label}_candidate_windows": sum(
                    max(0, len(group) - total_steps + 1)
                    for group in (
                        [group for _, group in part.groupby("episode_id", sort=False)]
                        if "episode_id" in part else [part]
                    )
                )
                for label, part in (
                    ("train", splits.train), ("validation", splits.validation),
                    ("test", splits.test),
                )
            },
        },
    )
