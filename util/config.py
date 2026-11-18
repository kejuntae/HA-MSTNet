"""Load and validate dataset and experiment JSON configurations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


REQUIRED_CANONICAL_COLUMNS = (
    "outdoor_temperature",
    "solar_irradiance",
    "valve_position",
    "neighbor_temperature",
    "room_temperature",
    "hvac_power",
    "hvac_mode",
    "month_sin",
    "month_cos",
    "weekday",
    "time_sin",
    "time_cos",
)


def _positive_integer(value: object, name: str) -> int:
    """Return a positive integer or raise a field-specific error."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    """Return a nonnegative integer or raise a field-specific error."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _finite_positive(value: object, name: str) -> float:
    """Return a finite positive number or raise a field-specific error."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a finite positive number.")
    return float(value)


@dataclass(frozen=True)
class OccupancyConfig:
    """Define an existing occupancy column; positive values indicate occupancy."""

    mode: str
    column: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "OccupancyConfig":
        """Build and validate an occupancy configuration from JSON."""
        mode = payload.get("mode")
        if mode != "column":
            raise ValueError(
                'occupancy.mode must be column; training does not generate occupancy schedules. '
                'For NEST, run datasets/nest2024/generate_occupancy.py first.'
            )
        column = payload.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ValueError('occupancy.column is required in column mode.')
        return cls(mode=mode, column=column)


@dataclass(frozen=True)
class DatasetConfig:
    """Configure a single CSV or pre-split files and optional episode metadata."""

    name: str
    csv_path: Path | None
    interval_minutes: int
    columns: dict[str, str]
    occupancy: OccupancyConfig
    split_ratios: tuple[float, float, float] | None
    split_dataset: bool = True
    split_csv_paths: dict[str, Path] | None = None
    episode_column: str | None = None
    target_origin_column: str | None = None
    direct_thermal_power: bool = False
    interval_start_inputs: bool = False

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        project_root: Path,
    ) -> "DatasetConfig":
        """Validate paths for the chosen split mode and keep declared metadata columns."""
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Dataset name must be a nonempty string.')
        split_dataset = payload.get("split_dataset", True)
        if not isinstance(split_dataset, bool):
            raise ValueError('split_dataset must be Boolean.')

        def resolve_csv(value: object, field_name: str) -> Path:
            """Resolve relative CSV paths against the project root."""
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a nonempty string.")
            path = Path(value)
            return (path if path.is_absolute() else project_root / path).resolve()

        csv_path = None
        split_csv_paths = None
        ratios = None
        if split_dataset:
            csv_path = resolve_csv(payload.get("csv_path"), "csv_path")
            if payload.get("split_csv_paths") is not None:
                raise ValueError('Ratio-based splitting cannot also specify split_csv_paths.')
            raw_ratios = payload.get("split_ratios")
            if not isinstance(raw_ratios, list) or len(raw_ratios) != 3:
                raise ValueError('split_ratios must contain three ratios.')
            ratios = tuple(float(value) for value in raw_ratios)
            if not all(math.isfinite(value) and value > 0.0 for value in ratios):
                raise ValueError('split_ratios must contain finite positive values.')
            if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
                raise ValueError('split_ratios must sum to 1.')
        else:
            paths = payload.get("split_csv_paths")
            if not isinstance(paths, Mapping) or set(paths) != {
                "train", "validation", "test",
            }:
                raise ValueError('split_csv_paths must specify train, validation, and test.')
            if (payload.get("csv_path") is not None
                    or payload.get("split_ratios") is not None):
                raise ValueError('Pre-split mode cannot specify csv_path or split_ratios.')
            split_csv_paths = {
                label: resolve_csv(paths[label], f"split_csv_paths.{label}")
                for label in ("train", "validation", "test")
            }
            if len(set(split_csv_paths.values())) != 3:
                raise ValueError('The three splits must use different CSV files.')

        raw_columns = payload.get("columns")
        if not isinstance(raw_columns, Mapping):
            raise ValueError('columns must be a mapping.')
        if any(
            key not in REQUIRED_CANONICAL_COLUMNS
            or not isinstance(value, str) or not value.strip()
            for key, value in raw_columns.items()
        ):
            raise ValueError('columns must map canonical features to nonempty CSV column names.')
        columns = dict(raw_columns)
        # Check model-specific input requirements when building windows.
        if "room_temperature" not in columns:
            raise ValueError('columns is missing the canonical feature room_temperature.')

        metadata = {}
        for field_name in ("episode_column", "target_origin_column"):
            value = payload.get(field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be a nonempty string.")
            metadata[field_name] = value
        if any(
            value in columns.values()
            for value in metadata.values() if value is not None
        ):
            raise ValueError('Episode and forecast-origin metadata cannot be mapped to numeric model features.')

        occupancy_payload = payload.get("occupancy")
        for option in ("direct_thermal_power", "interval_start_inputs"):
            if not isinstance(payload.get(option, False), bool):
                raise ValueError(f"{option} must be Boolean.")
        if payload.get("direct_thermal_power", False):
            if "hvac_power" not in columns:
                raise ValueError('direct_thermal_power requires hvac_power.')
            if {"valve_position", "hvac_mode"} & columns.keys():
                raise ValueError('Direct thermal power mode cannot also map valve position or HVAC mode.')
        if not isinstance(occupancy_payload, Mapping):
            raise ValueError('occupancy must be a mapping.')
        return cls(
            name=name.strip(),
            csv_path=csv_path,
            interval_minutes=_positive_integer(
                payload.get("interval_minutes"), "interval_minutes",
            ),
            columns=columns,
            occupancy=OccupancyConfig.from_mapping(occupancy_payload),
            split_ratios=ratios,
            split_dataset=split_dataset,
            split_csv_paths=split_csv_paths,
            direct_thermal_power=payload.get("direct_thermal_power", False),
            interval_start_inputs=payload.get("interval_start_inputs", False),
            **metadata,
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Shared training parameters for neural models."""

    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    device: str = "auto"
    gradient_clip: float | None = None
    # Zero disables live plots; a positive value sets the refresh interval in epochs.
    plot_every: int = 5

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TrainingConfig":
        """Build the training configuration from a mapping."""
        device = payload.get("device", "auto")
        if not isinstance(device, str) or not device.strip():
            raise ValueError('device must be a nonempty string.')
        clip = payload.get("gradient_clip")
        gradient_clip = None if clip is None else _finite_positive(
            clip,
            "gradient_clip",
        )
        return cls(
            epochs=_positive_integer(payload.get("epochs"), "epochs"),
            batch_size=_positive_integer(
                payload.get("batch_size"),
                "batch_size",
            ),
            learning_rate=_finite_positive(
                payload.get("learning_rate"),
                "learning_rate",
            ),
            patience=_positive_integer(payload.get("patience"), "patience"),
            device=device.strip(),
            gradient_clip=gradient_clip,
            plot_every=_nonnegative_integer(payload.get("plot_every", 5), "plot_every"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared window sizes, models, seeds, and hyperparameters."""

    name: str
    dataset_config: Path
    history_steps: int
    horizon_steps: int
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    training: TrainingConfig
    model_params: dict[str, dict[str, Any]]
    window_stride: int = 1
    # Only the RC model can project horizontal irradiance onto a vertical window.
    rc_window_irr: bool = False

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        project_root: Path,
    ) -> "ExperimentConfig":
        """Build the experiment configuration from JSON."""
        name = payload.get("name")
        dataset_value = payload.get("dataset_config")
        raw_seeds = payload.get("seeds")
        raw_models = payload.get("models")
        raw_training = payload.get("training")
        raw_params = payload.get("model_params", {})
        raw_rc_window_irr = payload.get("rc_window_irr", False)
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Experiment name must be a nonempty string.')
        if not isinstance(dataset_value, str) or not dataset_value.strip():
            raise ValueError('dataset_config must be a nonempty string.')
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise ValueError('seeds must be a nonempty list.')
        if not isinstance(raw_models, list) or not raw_models or not all(
            isinstance(value, str) and value.strip()
            for value in raw_models
        ):
            raise ValueError('models must be a nonempty list of model names.')
        if not isinstance(raw_training, Mapping):
            raise ValueError('training must be a mapping.')
        if not isinstance(raw_params, Mapping):
            raise ValueError('model_params must be a mapping.')
        if not isinstance(raw_rc_window_irr, bool):
            raise ValueError('rc_window_irr must be Boolean.')

        dataset_path = Path(dataset_value)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
        seeds = tuple(
            _nonnegative_integer(value, "seed") for value in raw_seeds
        )
        params = {
            str(model): dict(values)
            for model, values in raw_params.items()
            if isinstance(values, Mapping)
        }
        return cls(
            name=name.strip(),
            dataset_config=dataset_path.resolve(),
            history_steps=_positive_integer(
                payload.get("history_steps"),
                "history_steps",
            ),
            horizon_steps=_positive_integer(
                payload.get("horizon_steps"),
                "horizon_steps",
            ),
            seeds=seeds,
            models=tuple(value.strip() for value in raw_models),
            training=TrainingConfig.from_mapping(raw_training),
            model_params=params,
            window_stride=_positive_integer(
                payload.get("window_stride", 1),
                "window_stride",
            ),
            rc_window_irr=raw_rc_window_irr,
        )


def _load_json(path: Path) -> Mapping[str, object]:
    """Load JSON and require an object at the top level."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Config file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Config file is not valid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config file must contain a top-level object: {path}")
    return payload


def _project_root_from_config(path: Path) -> Path:
    """Find the project root from a config file, including nested model configs."""
    resolved = path.resolve()
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent
    if len(resolved.parents) < 3:
        raise ValueError(f"Config path is not deep enough to identify the project root: {resolved}")
    return resolved.parents[2]


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load a dataset configuration."""
    resolved = Path(path).resolve()
    return DatasetConfig.from_mapping(
        _load_json(resolved),
        _project_root_from_config(resolved),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration."""
    resolved = Path(path).resolve()
    return ExperimentConfig.from_mapping(
        _load_json(resolved),
        _project_root_from_config(resolved),
    )
