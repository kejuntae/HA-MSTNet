"""Prepare single-zone BOPTEST inputs while preserving splits and episode boundaries."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CASE_DIR / "datasets" / "complete_15min_2019"
OUTPUT_DIR = CASE_DIR.parent / "data_preprocessed" / "Boptest_single_zone"
SPLITS = {"train": "train", "validation": "validation", "test": "prediction_test"}
TIME_COLUMNS = ("datetime", "datetime_end", "time_start_s", "time_end_s")
EPISODE_COLUMNS = (
    "split", "episode_id", "episode_row", "is_episode_start",
    "is_episode_end", "data_role", "target_origin_allowed",
)
THERMAL_COLUMNS = (
    "operative_temperature_c", "outdoor_temperature_c", "solar_ghi_w_m2",
    "q_floor_mean_w", "occupancy_people",
)
LABEL_COLUMN = "next_operative_temperature_c"


def prepare_split(path: Path, expected_split: str) -> pd.DataFrame:
    """Validate transitions, select thermal inputs, and add NEST2024-compatible time features."""
    source = pd.read_csv(path)
    columns = (*TIME_COLUMNS, *EPISODE_COLUMNS, *THERMAL_COLUMNS, LABEL_COLUMN)
    missing = set((*columns, "datetime_start")) - set(source.columns)
    if missing:
        raise ValueError(f"{path.name} is missing fields: {sorted(missing)}")
    if source.empty or not source["split"].eq(expected_split).all():
        raise ValueError(f"{path.name} is empty or has the wrong split label.")
    result = source.loc[:, columns].copy()
    times = pd.DatetimeIndex(pd.to_datetime(result["datetime"], errors="raise"))
    ends = pd.DatetimeIndex(pd.to_datetime(result["datetime_end"], errors="raise"))
    starts = pd.DatetimeIndex(pd.to_datetime(source["datetime_start"]))
    if times.tz is not None or not times.equals(starts):
        raise ValueError(f"{path.name} has an invalid start time or model timezone.")
    if not ((ends - times) == pd.Timedelta(minutes=15)).all():
        raise ValueError(f"{path.name} contains intervals other than 15 minutes.")
    epoch = pd.Timestamp("2019-01-01")
    for timestamps, field in ((times, "time_start_s"), (ends, "time_end_s")):
        if not np.array_equal((timestamps - epoch).total_seconds(), source[field]):
            raise ValueError(f"{path.name} has dates that do not match simulation seconds.")
    if result.loc[:, EPISODE_COLUMNS].isna().any().any():
        raise ValueError(f"{path.name} has missing window metadata.")
    for field in ("is_episode_start", "is_episode_end", "target_origin_allowed"):
        if not pd.api.types.is_bool_dtype(result[field]):
            raise ValueError(f"{path.name}: {field} must be Boolean.")
    if not result["data_role"].isin(
        ["history_support", "target", "forecast_support"]
    ).all() or not result["target_origin_allowed"].eq(
        result["data_role"].eq("target")
    ).all():
        raise ValueError(f"{path.name} has inconsistent support rows and forecast-origin flags.")
    numeric = result.loc[:, (*THERMAL_COLUMNS, LABEL_COLUMN)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path.name} contains missing or infinite thermal features or targets.")

    # Reject out-of-order episodes rather than sorting or joining them.
    for episode, group in result.groupby("episode_id", sort=False):
        group_times = pd.DatetimeIndex(pd.to_datetime(group["datetime"]))
        if not np.array_equal(group["episode_row"], np.arange(len(group))):
            raise ValueError(f"{episode} must have consecutive row numbers starting at zero.")
        if not (np.diff(group_times.values) == np.timedelta64(15, "m")).all():
            raise ValueError(f"{episode} has discontinuous timestamps.")
        positions = group.index.to_numpy()
        if not (np.diff(positions) == 1).all():
            raise ValueError(f"{episode} is not a single contiguous block of rows.")
        for field, boundary in (("is_episode_start", 0), ("is_episode_end", -1)):
            if group[field].sum() != 1 or not group[field].iloc[boundary]:
                raise ValueError(f"{episode} has invalid episode boundary flags.")
        if not np.allclose(
            group[LABEL_COLUMN].iloc[:-1],
            group["operative_temperature_c"].iloc[1:],
            rtol=0.0, atol=1e-8,
        ):
            raise ValueError(f"{episode} has end labels that differ from the next row temperature.")

    # Keep the existing units (°C, W, and W/m²).
    # Preserve operative temperature and scheduled headcount without binarizing occupancy.
    result = result.rename(columns={"datetime": "time"})
    result["time"] = times.strftime("%Y-%m-%d %H:%M:%S")
    result["datetime_end"] = ends.strftime("%Y-%m-%d %H:%M:%S")
    month_angle = 2.0 * np.pi * times.month / 12.0
    day_seconds = times.hour * 3600 + times.minute * 60 + times.second
    day_angle = 2.0 * np.pi * day_seconds / 86400.0
    result["Time month sin"] = np.sin(month_angle)
    result["Time month cos"] = np.cos(month_angle)
    result["Time weekday"] = times.weekday
    result["Time interval sin"] = np.sin(day_angle)
    result["Time interval cos"] = np.cos(day_angle)
    return result


def main() -> None:
    """Validate all splits before writing the three thermal-model CSV files."""
    parser = argparse.ArgumentParser(description="Export BOPTEST thermal-model inputs.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    prepared = {
        name: prepare_split(args.source_dir / f"{name}.csv", split)
        for name, split in SPLITS.items()
    }
    episode_sets = [set(frame["episode_id"]) for frame in prepared.values()]
    if any(episode_sets[i] & episode_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('An episode_id appears in more than one split.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in prepared.items():
        output = args.output_dir / f"{name}.csv"
        frame.to_csv(output, index=False)
        print(
            f"{output}: {len(frame)} rows, {len(frame.columns)} columns, "
            f"{frame['episode_id'].nunique()} episodes, "
            f"{int(frame['target_origin_allowed'].sum())} eligible forecast origins"
        )


if __name__ == "__main__":
    main()
