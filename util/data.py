"""Load preprocessed CSVs and map their columns to canonical thermal features."""

from __future__ import annotations

from dataclasses import dataclass, field
from os.path import commonpath
from pathlib import Path

import numpy as np
import pandas as pd

from util.config import DatasetConfig
from util.features import TimeSplits
from util.model_features import THERMAL_FEATURES


@dataclass(frozen=True)
class CanonicalDataset:
    """Store canonical measurements and their provenance."""

    # Columns use the shared canonical feature names.
    frame: pd.DataFrame
    # Keep the resolved path for provenance.
    dataset_path: Path
    # Record that occupancy comes from a CSV column.
    occupancy_source: str
    # Window validation and physical models use the sampling interval.
    interval_minutes: int
    # Keep pre-split tables separate; the combined view is for summaries only.
    splits: TimeSplits | None = None
    dataset_paths: dict[str, Path] = field(default_factory=dict)


def _binary_occupancy(values: np.ndarray) -> np.ndarray:
    """Binarize finite occupancy values and leave nonfinite values missing."""
    values = np.asarray(values, dtype=float)
    # Leave nonfinite values missing so the window filter can reject them.
    return np.where(np.isfinite(values), (values > 0).astype(float), np.nan)


def _read_timestamp(frame: pd.DataFrame, path: Path) -> pd.DatetimeIndex:
    """Parse the first CSV column as timestamps."""
    # The CSV needs at least a timestamp column.
    if frame.shape[1] == 0:
        raise ValueError(f"CSV has no columns; row count: {len(frame)}:{path}")
    # Parse and remove the first column as the time index.
    timestamp_name = frame.columns[0]
    try:
        # Reject invalid dates rather than treating them as missing.
        timestamps = pd.to_datetime(frame.pop(timestamp_name), errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Timestamp column {timestamp_name} could not be parsed; row count: {len(frame)}."
        ) from error
    # Use one index name throughout loading and reporting.
    return pd.DatetimeIndex(timestamps, name="timestamp")


def _numeric_column(
    values: pd.Series,
    source_name: str,
    row_count: int,
) -> pd.Series:
    """Convert numeric values while distinguishing missing data from invalid text."""
    # Convert missing values and invalid text to NaN for validation.
    converted = pd.to_numeric(values, errors="coerce")
    # A nonempty value that became NaN is invalid text.
    invalid = values.notna() & converted.isna()
    if invalid.any():
        raise ValueError(
            f"Column {source_name} contains nonnumeric values; row count: {row_count}."
        )
    # Use floats for numeric features and missing values.
    return converted.astype(float)


def _validate_time_index(
    index: pd.DatetimeIndex,
    interval_minutes: int,
    row_count: int,
) -> None:
    """Require strictly increasing timestamps at the configured interval."""
    # Reject duplicate or out-of-order timestamps before positional slicing.
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"Time index must be strictly increasing and unique; row count: {row_count}.")
    if len(index) > 1:
        # Measure the elapsed time between consecutive rows.
        differences = index.to_series().diff().iloc[1:]
        expected = pd.Timedelta(minutes=interval_minutes)
        # Reject gaps within the time series being checked.
        if not bool((differences == expected).all()):
            raise ValueError(
                f"Sampling interval must be {interval_minutes} minutes; row count: {row_count}."
            )


def _load_canonical_csv(config: DatasetConfig, path: Path) -> CanonicalDataset:
    """Load a CSV, validate each episode's time axis, and map its numeric columns."""
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Preprocessed CSV not found: {path}")

    # Read prepared data without filling gaps or resampling.
    dtype = {config.episode_column: "string"} if config.episode_column else None
    frame = pd.read_csv(path, dtype=dtype)
    if frame.empty:
        raise ValueError(f"Preprocessed CSV is empty: {path}")
    # The timestamp column has already been removed.
    timestamps = _read_timestamp(frame, path)
    metadata = _read_window_metadata(frame, config)
    if "episode_id" in metadata:
        episodes = metadata["episode_id"]
        boundaries = np.flatnonzero(np.r_[True, episodes[1:] != episodes[:-1]])
        if len(set(episodes[boundaries])) != len(boundaries):
            raise ValueError(f"Each episode_id must occupy a single contiguous block: {path}")
        for start, end in zip(boundaries, np.r_[boundaries[1:], len(frame)]):
            try:
                _validate_time_index(
                    timestamps[start:end], config.interval_minutes, end - start,
                )
            except ValueError as error:
                raise ValueError(f"{path.name} episode={episodes[start]}:{error}") from error
    else:
        _validate_time_index(timestamps, config.interval_minutes, len(frame))

    # Ignore legacy mappings for features outside the canonical model inputs.
    model_columns = {
        name: source for name, source in config.columns.items()
        if name in THERMAL_FEATURES
    }
    missing_sources = [
        source for source in model_columns.values()
        if source not in frame.columns
    ]
    if missing_sources:
        raise ValueError(
            'CSV is missing columns: '
            f"{', '.join(missing_sources)}; row count: {len(frame)}."
        )

    # Build the canonical table on the shared time index.
    canonical = pd.DataFrame(index=timestamps)
    for target_name, source_name in model_columns.items():
        # Each config entry maps a canonical feature to its source CSV column.
        canonical[target_name] = _numeric_column(
            frame[source_name],
            source_name,
            len(frame),
        ).to_numpy()

    # Require recorded occupancy; do not generate a schedule during loading.
    occupancy_column = config.occupancy.column
    if occupancy_column not in frame.columns:
        raise ValueError(
            f"CSV is missing the occupancy column {occupancy_column},"
            f"row count: {len(frame)}."
        )
    canonical["occupancy"] = _binary_occupancy(_numeric_column(
        frame[occupancy_column],
        occupancy_column,
        len(frame),
    ).to_numpy())

    # Keep window metadata separate from numeric model features.
    for name, values in metadata.items():
        canonical[name] = values

    # Preserve data and occupancy provenance for reporting.
    return CanonicalDataset(
        frame=canonical,
        dataset_path=path,
        occupancy_source="column",
        interval_minutes=config.interval_minutes,
    )


def _read_window_metadata(
    frame: pd.DataFrame, config: DatasetConfig,
) -> dict[str, np.ndarray]:
    """Read episode IDs and explicit Boolean forecast-origin flags."""
    metadata = {}
    for name, source in (
        ("episode_id", config.episode_column),
        ("target_origin_allowed", config.target_origin_column),
    ):
        if source is None:
            continue
        if source not in frame:
            raise ValueError(f"CSV is missing window metadata column {source}.")
        values = frame[source]
        if values.isna().any():
            raise ValueError(f"Window metadata column {source} cannot contain missing values.")
        if name == "episode_id":
            if values.astype(str).str.strip().eq("").any():
                raise ValueError(f"Window metadata column {source} cannot contain empty identifiers.")
            metadata[name] = values.astype(str).to_numpy()
        else:
            if not pd.api.types.is_bool_dtype(values):
                raise ValueError(f"Window metadata column {source} must contain Boolean values (True/False).")
            metadata[name] = values.to_numpy(dtype=bool)
    return metadata


def load_canonical_dataset(config: DatasetConfig) -> CanonicalDataset:
    """Read a single CSV or pre-split files without reassigning rows."""
    if config.split_dataset:
        if config.csv_path is None:
            raise ValueError('Ratio-based splitting requires csv_path.')
        return _load_canonical_csv(config, config.csv_path)

    if not config.split_csv_paths or set(config.split_csv_paths) != {
        "train", "validation", "test",
    }:
        raise ValueError('Pre-split mode requires three split_csv_paths entries.')
    frames = {}
    seen_episodes = set()
    for label in ("train", "validation", "test"):
        dataset = _load_canonical_csv(config, config.split_csv_paths[label])
        frame = dataset.frame
        if "episode_id" in frame:
            episodes = set(frame["episode_id"])
            shared = seen_episodes & episodes
            if shared:
                raise ValueError(f"episode_id values are shared across splits: {sorted(shared)}.")
            seen_episodes.update(episodes)
        frames[label] = frame
    paths = {name: path.resolve() for name, path in config.split_csv_paths.items()}
    return CanonicalDataset(
        frame=pd.concat(list(frames.values())),
        dataset_path=Path(commonpath([str(path.parent) for path in paths.values()])),
        occupancy_source=config.occupancy.mode,
        interval_minutes=config.interval_minutes,
        splits=TimeSplits(**frames),
        dataset_paths=paths,
    )


def summarize_dataset(dataset: CanonicalDataset) -> dict[str, object]:
    """Summarize file provenance and episode counts per split."""
    summary = summarize_frame(
        dataset.frame, dataset.dataset_path, dataset.occupancy_source,
        dataset.interval_minutes,
    )
    summary["split_dataset"] = dataset.splits is None
    if dataset.splits is not None:
        summary["dataset_paths"] = {
            name: str(path) for name, path in dataset.dataset_paths.items()
        }
        summary["splits"] = {
            name: summarize_frame(
                getattr(dataset.splits, name), path, dataset.occupancy_source,
                dataset.interval_minutes,
            )
            for name, path in dataset.dataset_paths.items()
        }
    return summary


def summarize_frame(
    frame: pd.DataFrame,
    dataset_path: Path,
    occupancy_source: str,
    interval_minutes: int,
) -> dict[str, object]:
    """Summarize missing values and time coverage without modifying the data."""
    # Count rows whose required values are all finite.
    values = frame.drop(columns=["episode_id", "target_origin_allowed"], errors="ignore")
    numeric = values.to_numpy(dtype=float)
    finite_rows = np.isfinite(numeric).all(axis=1)
    # Keep the summary directly JSON-serializable.
    return {
        "dataset_path": str(dataset_path),
        "start": frame.index.min().isoformat() if len(frame) else None,
        "end": frame.index.max().isoformat() if len(frame) else None,
        "rows": int(len(frame)),
        "interval_minutes": int(interval_minutes),
        "occupancy_source": occupancy_source,
        # Report missing values per feature.
        "missing_by_field": {
            column: int(frame[column].isna().sum())
            for column in values.columns
        },
        # Count rows with no missing or nonfinite model features.
        "finite_complete_rows": int(finite_rows.sum()),
        **({"episodes": int(frame["episode_id"].nunique())}
           if "episode_id" in frame else {}),
        **({"allowed_origins": int(frame["target_origin_allowed"].sum())}
           if "target_origin_allowed" in frame else {}),
    }
