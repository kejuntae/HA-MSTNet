"""Prepare four years of NEST UMAR measurements for room-temperature modeling.

Clean sensor data, estimate room-level HVAC power, resample to 15 minutes, and produce a
wide table and room subsets.
"""

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Four annual wide-table CSVs from the UMAR release.
UMAR_DATA_FILES: list[str] = [
    "umar_2019-07-01_2020-07-01_wide_descId.csv",
    "umar_2020-07-01_2021-07-01_wide_descId.csv",
    "umar_2021-07-01_2022-07-01_wide_descId.csv",
    "umar_2022-07-01_2023-07-01_wide_descId.csv",
]

# Assumed relative flow weights, not measured flows or verified room areas.
DEFAULT_UMAR_FLOW_WEIGHTS: dict[str, float] = {
    "272": 31.0,  # Bedroom 1: one radiant circuit.
    "273": 93.0,  # Living room: three parallel circuits, each weighted 31.0.
    "274": 31.0,  # Bedroom 2: one radiant circuit.
    "275": 10.0,  # Bathroom 1: one small circuit.
    "276": 10.0,  # Bathroom 2: one small circuit.
}

# Auxiliary or incomplete features omitted by default.
DEFAULT_DROP_COLUMNS: list[str] = [
    "dhw_volume",
    "dhw_power",
    "total_active_power",
    "heating_power",
    "cooling_power",
    "temp_276",
    "setp_276",
    "y1_276",
    "temp_275",
    "setp_275",
    "y1_275",
]

# Default output location for the processed wide table.
DEFAULT_OUTPUT_DIR: Path = Path(__file__).resolve().parents[1] / "data_preprocessed" / "NEST2024"
DEFAULT_OUTPUT_FILE: Path = DEFAULT_OUTPUT_DIR / "UMAR_2024.csv"

# Use aggregate power for allocation only; exclude it from exported tables.
EXCLUDED_EXPORT_COLUMNS = ("total_active_power", "heating_power", "cooling_power")


def _get_default_umar_dir() -> Path:
    """Return the absolute path to the default UMAR input directory."""
    return Path(__file__).resolve().parent / "UMAR"


def _normalize_valve_states(data: pd.DataFrame) -> None:
    """Map valve state 2 to open (1) in place, preserving other values and missing data."""
    valve_columns = {"y2_273", "y3_273"}
    for room in DEFAULT_UMAR_FLOW_WEIGHTS:
        valve_columns.update({
            f"y1_{room}", f"Thermal valve {room}",
            f"valve_{room}", f"valve room {room}",
        })
    for column in data.columns:
        if column in valve_columns:
            data[column] = data[column].replace(2, 1)


def load_umar_raw_data(
    file_name: str | None = None,
    data_dir: Path | str | None = None,
    concatenate: bool = False,
    parse_dates: bool = True,
    set_time_index: bool = True,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Load one annual UMAR CSV or all four.

    Select a file by name or start year. When loading all years, return a dictionary
    unless concatenation is requested; concatenated timestamps are sorted and
    deduplicated.
    """
    target_dir = Path(data_dir) if data_dir is not None else _get_default_umar_dir()
    if not target_dir.exists():
        raise FileNotFoundError(f"UMAR data directory not found: {target_dir}")

    # Select the input files.
    if file_name is not None:
        # Accept an exact filename, path, start year, or substring.
        if file_name in UMAR_DATA_FILES:
            matched_files = [file_name]
        elif (target_dir / file_name).exists():
            matched_files = [(target_dir / file_name).name]
        else:
            # Prefer a start-year match.
            prefix_matched = [f for f in UMAR_DATA_FILES if f.startswith(f"umar_{file_name}")]
            if len(prefix_matched) == 1:
                matched_files = prefix_matched
            else:
                # Fall back to a substring match.
                matched = [f for f in UMAR_DATA_FILES if file_name in f]
                # Accept a year span such as '2020-2021'.
                if not matched and "-" in file_name:
                    parts = file_name.replace(" ", "").split("-")
                    if len(parts) == 2:
                        matched = [f for f in UMAR_DATA_FILES if parts[0] in f and parts[1] in f]
                if not matched:
                    raise FileNotFoundError(f"No match in {target_dir} for file: {file_name}")
                matched_files = matched
    else:
        # Use all four annual files.
        matched_files = UMAR_DATA_FILES

    loaded_dfs: dict[str, pd.DataFrame] = {}

    # Read the selected files.
    for fname in matched_files:
        fpath = target_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Data file not found: {fpath}")

        print(f"Reading: {fname} ...")
        # The exported CSV may include an unnamed index column.
        df = pd.read_csv(fpath)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Treat valve state 2 as open before allocation and branch aggregation.
        _normalize_valve_states(df)

        if parse_dates and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            if set_time_index:
                df = df.set_index("time").sort_index()

        loaded_dfs[fname] = df

    # Return a single frame, concatenate years, or return a dictionary.
    if file_name is not None and len(matched_files) == 1:
        return loaded_dfs[matched_files[0]]

    if concatenate:
        print('Merging four datasets, sorting by time, and removing duplicates...')
        combined_df = pd.concat(list(loaded_dfs.values()))
        if set_time_index and isinstance(combined_df.index, pd.DatetimeIndex):
            combined_df = combined_df[~combined_df.index.duplicated(keep="first")].sort_index()
        elif "time" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["time"]).sort_values("time")
        return combined_df

    return loaded_dfs


def check_and_clean_anomalies(
    df: pd.DataFrame,
    solar_col: str = "irrad",
    temp_amb_col: str = "temp_amb",
    heating_col: str = "heating_power",
    cooling_col: str = "cooling_power",
    solar_max_hours: float = 20.0,
    temp_max_minutes: float = 30.0,
    power_max_days: float = 1.0,
    clip_solar_zero: bool = True,
    inplace: bool = False,
    power_zero_threshold_kw: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mask stuck sensor readings and report the affected rows and columns.

    By default, clip negative irradiance and mask flat runs longer than 20 h for
    irradiance, 30 min for outdoor temperature, or 24 h for nonzero HVAC power. Preserve
    power readings at or below 0.01 kW in magnitude.
    """
    data = df if inplace else df.copy()

    if not np.isfinite(power_zero_threshold_kw) or power_zero_threshold_kw < 0:
        raise ValueError('The near-zero power threshold must be finite and nonnegative in kW.')

    # Prefer the timestamp index, falling back to the time column.
    has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
    if has_datetime_index:
        time_series = data.index.to_series()
    elif "time" in data.columns:
        time_series = pd.to_datetime(data["time"])
    else:
        time_series = None

    # Find indices of constant runs longer than the duration limit.
    def _find_constant_sequence_indices(
        series: pd.Series, max_duration: pd.Timedelta, step_count_fallback: int
    ) -> pd.Index:
        valid_mask = series.notna()
        if not valid_mask.any():
            return series.index[:0]

        # Start a new run when either the value or its validity changes.
        change = series.ne(series.shift()) | valid_mask.ne(valid_mask.shift())
        group_id = change.cumsum()

        v_series = series[valid_mask]
        v_group = group_id[valid_mask]

        if time_series is not None:
            # Compare elapsed time from the first to the last sample in the run.
            v_time = time_series[valid_mask]
            starts = v_time.groupby(v_group).transform("min")
            ends = v_time.groupby(v_group).transform("max")
            durations = ends - starts
            exceeded_mask = durations > max_duration
        else:
            # Fall back to sample counts when timestamps are unavailable.
            group_sizes = v_group.groupby(v_group).transform("size")
            exceeded_mask = group_sizes > step_count_fallback

        return v_series.index[exceeded_mask]

    deleted_report: dict[str, dict[str, Any]] = {}
    all_affected_indices: set = set()
    total_deleted = 0

    # Clean solar irradiance.
    if solar_col in data.columns:
        if clip_solar_zero:
            # Clip negative nighttime sensor readings to zero.
            data[solar_col] = data[solar_col].clip(lower=0.0)

        solar_duration = pd.Timedelta(hours=solar_max_hours)
        fallback_steps = int(solar_max_hours * 60)
        bad_idx = _find_constant_sequence_indices(data[solar_col], solar_duration, fallback_steps)

        count = len(bad_idx)
        if count > 0:
            data.loc[bad_idx, solar_col] = np.nan
            all_affected_indices.update(bad_idx.tolist())
            total_deleted += count

        deleted_report[solar_col] = {
            "deleted_count": count,
            "deleted_indices": bad_idx,
            "threshold_used": f"> {solar_max_hours} hours of constant readings (negative readings clipped to zero)",
        }

    # Clean outdoor temperature.
    if temp_amb_col in data.columns:
        temp_duration = pd.Timedelta(minutes=temp_max_minutes)
        fallback_steps = int(temp_max_minutes)
        bad_idx = _find_constant_sequence_indices(data[temp_amb_col], temp_duration, fallback_steps)

        count = len(bad_idx)
        if count > 0:
            data.loc[bad_idx, temp_amb_col] = np.nan
            all_affected_indices.update(bad_idx.tolist())
            total_deleted += count

        deleted_report[temp_amb_col] = {
            "deleted_count": count,
            "deleted_indices": bad_idx,
            "threshold_used": f"> {temp_max_minutes} minutes of constant readings",
        }

    # Clean heating and cooling power.
    power_cols = [c for c in [heating_col, cooling_col] if c in data.columns]
    power_duration = pd.Timedelta(days=power_max_days)
    fallback_steps = int(power_max_days * 1440)

    for p_col in power_cols:
        bad_idx = _find_constant_sequence_indices(data[p_col], power_duration, fallback_steps)
        # Preserve sustained zero and near-zero power during shutdowns.
        near_zero = data.loc[bad_idx, p_col].abs().le(power_zero_threshold_kw)
        retained_count = int(near_zero.sum())
        bad_idx = bad_idx[~near_zero.to_numpy()]
        count = len(bad_idx)
        if count > 0:
            data.loc[bad_idx, p_col] = np.nan
            all_affected_indices.update(bad_idx.tolist())
            total_deleted += count

        deleted_report[p_col] = {
            "deleted_count": count,
            "deleted_indices": bad_idx,
            "threshold_used": (
                f"> {power_max_days} days of constant readings with absolute value > {power_zero_threshold_kw} kW"
            ),
            "retained_near_zero_count": retained_count,
        }

    # Summarize the affected readings.
    report: dict[str, Any] = {
        "total_deleted_points": total_deleted,
        "affected_rows_count": len(all_affected_indices),
        "deleted_indices_by_column": deleted_report,
    }

    return data, report


def calculate_room_thermal_powers(
    df: pd.DataFrame,
    heating_col: str = "heating_power",
    cooling_col: str = "cooling_power",
    weights: dict[str, float] | None = None,
    add_net_power: bool = True,
    add_split_power: bool = False,
    rooms: list[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Estimate room-level power from the UMAR unit's aggregate heat meter.

    Allocate power by valve opening and assumed flow weights (31:93:31:10:10 by
    default). Net power is positive for heating and negative for cooling, in kW;
    separate heating and cooling columns are optional.
    """
    data = df if inplace else df.copy()
    # Normalize valve state 2 here as well to avoid doubling its flow weight.
    _normalize_valve_states(data)

    if weights is None:
        weights = DEFAULT_UMAR_FLOW_WEIGHTS

    if rooms is None:
        rooms = ["272", "273", "274", "275", "276"]

    n_samples = len(data)

    # Resolve each room's valve openings in the range [0, 1].
    valve_dict: dict[str, np.ndarray] = {}
    for r in rooms:
        candidate_cols = [
            f"y1_{r}",
            f"Thermal valve {r}",
            f"valve_{r}",
            f"valve room {r}",
        ]
        found_col = None
        for c in candidate_cols:
            if c in data.columns:
                found_col = c
                break

        if r == "273":
            # Room 273 may expose three parallel valve circuits.
            y_273_cols = [c for c in ["y1_273", "y2_273", "y3_273"] if c in data.columns]
            if len(y_273_cols) > 1:
                valve_dict[r] = data[y_273_cols].mean(axis=1).fillna(0.0).to_numpy()
            elif found_col is not None:
                valve_dict[r] = data[found_col].fillna(0.0).to_numpy()
            else:
                valve_dict[r] = np.zeros(n_samples, dtype=float)
        else:
            if found_col is not None:
                valve_dict[r] = data[found_col].fillna(0.0).to_numpy()
            else:
                valve_dict[r] = np.zeros(n_samples, dtype=float)

    # Estimate each room's flow as its assumed weight times its valve opening.
    weight_vec = np.array([weights.get(r, 0.0) for r in rooms], dtype=float)
    valve_matrix = np.column_stack([valve_dict[r] for r in rooms])  # (N, len(rooms))
    flow_matrix = valve_matrix * weight_vec  # (N, len(rooms))

    # Sum weighted openings across rooms.
    total_flow = flow_matrix.sum(axis=1, keepdims=True)  # (N, 1)

    # Allocate no power when all valves are closed.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_matrix = np.where(total_flow > 0, flow_matrix / total_flow, 0.0)

    # Read aggregate heating and cooling power.
    has_heating = heating_col in data.columns
    has_cooling = cooling_col in data.columns

    heat_total = (
        data[heating_col].fillna(0.0).to_numpy()[:, np.newaxis]
        if has_heating
        else np.zeros((n_samples, 1))
    )
    cool_total = (
        data[cooling_col].fillna(0.0).to_numpy()[:, np.newaxis]
        if has_cooling
        else np.zeros((n_samples, 1))
    )

    # Allocate power to individual rooms.
    heat_rooms = heat_total * ratio_matrix  # (N, len(rooms))
    cool_rooms = cool_total * ratio_matrix  # (N, len(rooms))

    # Add the allocated power columns.
    for idx, r in enumerate(rooms):
        if add_split_power:
            if has_heating:
                data[f"heat_power_{r}"] = heat_rooms[:, idx]
            if has_cooling:
                data[f"cool_power_{r}"] = cool_rooms[:, idx]

        if add_net_power:
            # Signed net thermal power in kW: heating is positive and cooling is negative.
            data[f"power_room_{r}"] = heat_rooms[:, idx] - cool_rooms[:, idx]

    return data


def drop_and_merge_features(
    df: pd.DataFrame,
    drop_cols: list[str] | None = None,
    merge_valves: bool = True,
    merge_windows: bool = True,
    drop_auxiliary_room_powers: bool = True,
    drop_split_powers: bool = True,
    valve_prefix: str = "valve",
    inplace: bool = False,
) -> pd.DataFrame:
    """Combine valve and window signals and drop unused features.

    Average parallel valves and windows for room 273. By default, remove domestic hot
    water and auxiliary-room signals, and keep signed net power rather than separate
    heating and cooling columns.
    """
    data = df if inplace else df.copy()

    # Combine branch valve signals and remove the original columns.
    if merge_valves:
        # Room 272.
        if "y1_272" in data.columns:
            data[f"{valve_prefix}_272"] = (data["y1_272"] > 0.5).astype(float)
        elif f"{valve_prefix}_272" not in data.columns and "valve_272" in data.columns:
            data[f"{valve_prefix}_272"] = (data["valve_272"] > 0.5).astype(float)

        # Average the three parallel valves in room 273.
        y_273_cols = [c for c in ["y1_273", "y2_273", "y3_273"] if c in data.columns]
        if y_273_cols:
            data[f"{valve_prefix}_273"] = data[y_273_cols].apply(
                lambda s: (s > 0.5).astype(float)
            ).mean(axis=1)

        # Room 274.
        if "y1_274" in data.columns:
            data[f"{valve_prefix}_274"] = (data["y1_274"] > 0.5).astype(float)
        elif f"{valve_prefix}_274" not in data.columns and "valve_274" in data.columns:
            data[f"{valve_prefix}_274"] = (data["valve_274"] > 0.5).astype(float)

        # Drop the original valve columns.
        old_valves = [
            "y1_272", "y1_273", "y2_273", "y3_273", "y1_274", "y1_275", "y1_276"
        ]
        data = data.drop(columns=[c for c in old_valves if c in data.columns])

    # Combine window signals and remove the original columns.
    if merge_windows:
        # Average the two exterior window signals in room 273.
        win_273_cols = [c for c in ["window_273_1", "window_273_2"] if c in data.columns]
        if win_273_cols:
            data["window_273"] = data[win_273_cols].apply(
                lambda s: (s > 0.5).astype(float)
            ).mean(axis=1)
            data = data.drop(columns=win_273_cols)

        # Express openings in rooms 272 and 274 as fractions in [0, 1].
        for r in ["272", "274"]:
            w_col = f"window_{r}"
            if w_col in data.columns:
                data[w_col] = (data[w_col] > 0.5).astype(float)

    # Drop the selected unused features.
    target_drops = list(drop_cols) if drop_cols is not None else list(DEFAULT_DROP_COLUMNS)

    if drop_auxiliary_room_powers:
        aux_power_cols = [
            f"{prefix}_{r}"
            for r in ["275", "276"]
            for prefix in ["power_room", "heat_power", "cool_power"]
        ]
        for c in aux_power_cols:
            if c not in target_drops:
                target_drops.append(c)

    if drop_split_powers:
        # Keep net room power and remove the separate heating and cooling columns.
        split_power_cols = [
            f"{prefix}_{r}"
            for r in ["272", "273", "274"]
            for prefix in ["heat_power", "cool_power"]
        ]
        for c in split_power_cols:
            if c not in target_drops:
                target_drops.append(c)

    existing_drops = [c for c in target_drops if c in data.columns]
    if existing_drops:
        data = data.drop(columns=existing_drops)

    return data


# Keep the legacy alias.
drop_unnecessary_features = drop_and_merge_features


def adjust_data_resolution(
    df: pd.DataFrame,
    rule: str = "15min",
    interp_limit_hours: float = 2.0,
    fill_power_zero: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """Resample each signal type to 15 minutes by default.

    Average temperatures and weather with limited time interpolation. Use the last
    setpoint per interval, convert valve/window states to duty cycles, and average
    signed room power without filling missing power by default.
    """
    data = df if inplace else df.copy()

    # Ensure a timestamp index.
    if not isinstance(data.index, pd.DatetimeIndex):
        if "time" in data.columns:
            data["time"] = pd.to_datetime(data["time"])
            data = data.set_index("time").sort_index()
        else:
            raise ValueError("Resampling requires a DatetimeIndex or a 'time' column.")

    rule_delta = pd.Timedelta(rule)
    if interp_limit_hours is not None and interp_limit_hours > 0:
        limit_steps = max(1, int(pd.Timedelta(hours=interp_limit_hours) / rule_delta))
    else:
        limit_steps = 0

    target_index = data.resample(rule).mean().index
    resampled = pd.DataFrame(index=target_index)
    processed_cols: set[str] = set()

    # Indoor temperatures (excluding outdoor temperature).
    cols_temp = [c for c in data.columns if c.startswith("temp_") and c != "temp_amb"]
    for c in cols_temp:
        s = data[c].resample(rule).mean()
        if limit_steps > 0:
            s = s.interpolate(method="time", limit=limit_steps)
        resampled[c] = s
        processed_cols.add(c)

    # Temperature setpoints.
    cols_setp = [c for c in data.columns if c.startswith("setp_")]
    for c in cols_setp:
        resampled[c] = data[c].resample(rule).last().ffill().bfill()
        processed_cols.add(c)

    # Outdoor temperature and solar irradiance.
    if "temp_amb" in data.columns:
        s = data["temp_amb"].resample(rule).mean()
        if limit_steps > 0:
            s = s.interpolate(method="time", limit=limit_steps)
        resampled["temp_amb"] = s
        processed_cols.add("temp_amb")

    if "irrad" in data.columns:
        # Averaging irradiance preserves W/m²; do not convert it from accumulated energy.
        s = data["irrad"].clip(lower=0.0).resample(rule).mean()
        if limit_steps > 0:
            s = s.interpolate(method="time", limit=limit_steps)
        resampled["irrad"] = s
        processed_cols.add("irrad")

    # Valve and window states.
    cols_actuator = [
        c for c in data.columns
        if c not in processed_cols and (
            c.startswith(("y", "window_")) or "valve" in c.lower()
        )
    ]
    for c in cols_actuator:
        s_raw = data[c].dropna()
        if (s_raw > 1.0).any() or (len(s_raw) > 0 and (s_raw % 1 == 0).all()):
            act_duty = (data[c] > 0.5).astype(float)
        else:
            act_duty = data[c].clip(lower=0.0, upper=1.0)
        resampled[c] = act_duty.resample(rule).mean().fillna(0.0)
        processed_cols.add(c)

    # Aggregate and room-level power.
    cols_power = [
        c for c in data.columns
        if c not in processed_cols and "power" in c.lower()
    ]
    for c in cols_power:
        if c.startswith("power_room"):
            s = data[c].resample(rule).mean()
        else:
            s = data[c].clip(lower=0.0).resample(rule).mean()

        if fill_power_zero:
            s = s.fillna(0.0)
        resampled[c] = s
        processed_cols.add(c)

    # Domestic hot water flow, when present.
    if "dhw_volume" in data.columns and "dhw_volume" not in processed_cols:
        resampled["dhw_volume"] = data["dhw_volume"].resample(rule).mean().fillna(0.0)
        processed_cols.add("dhw_volume")

    # Average the remaining numeric features.
    remaining = [c for c in data.columns if c not in processed_cols]
    for c in remaining:
        resampled[c] = data[c].resample(rule).mean()

    # Preserve the input column order.
    ordered_cols = [c for c in data.columns if c in resampled.columns]
    extra_cols = [c for c in resampled.columns if c not in ordered_cols]
    return resampled[ordered_cols + extra_cols]


# Keep the legacy alias.
resample_umar_data = adjust_data_resolution


def generate_time_features(
    df: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add the five calendar features used by the UMAR benchmark.

    Encode month (1–12) and seconds since midnight as sine/cosine pairs. Encode weekdays
    from 0 (Monday) to 6 (Sunday).
    """
    data = df if inplace else df.copy()

    # Read timestamps.
    if isinstance(data.index, pd.DatetimeIndex):
        time_index = data.index
    elif "time" in data.columns:
        time_index = pd.to_datetime(data["time"])
    else:
        raise ValueError("Calendar features require a DatetimeIndex or a 'time' column.")

    # Encode months 1 through 12.
    months = time_index.month
    theta_month = 2.0 * np.pi * months / 12.0
    data["Time month sin"] = np.sin(theta_month)
    data["Time month cos"] = np.cos(theta_month)

    # Encode weekday indices (Monday = 0).
    data["Time weekday"] = time_index.weekday

    # Encode time of day using elapsed seconds, independent of sampling interval.
    seconds_of_day = time_index.hour * 3600 + time_index.minute * 60 + time_index.second
    theta_day = 2.0 * np.pi * seconds_of_day / 86400.0
    data["Time interval sin"] = np.sin(theta_day)
    data["Time interval cos"] = np.cos(theta_day)

    return data


# Keep the legacy alias.
add_time_features = generate_time_features


def save_processed_data(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    index_label: str = "time",
    float_format: str = "%.6f",
) -> Path:
    """Write the processed CSV without aggregate electrical, heating, or cooling power."""
    target_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed data to: {target_path} ...")
    # Drop aggregate power even if the caller skipped feature selection.
    export_df = df.drop(columns=list(EXCLUDED_EXPORT_COLUMNS), errors="ignore")
    export_df.to_csv(target_path, index=True, index_label=index_label, float_format=float_format)
    file_size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {len(export_df):,} rows x {export_df.shape[1]} columns | file size: {file_size_mb:.2f} MB")
    return target_path


# Room 273 is the living room between bedrooms 272 and 274.
UMAR_ROOM_NEIGHBORS: dict[str, list[str]] = {
    "272": ["273"],
    "273": ["272", "274"],
    "274": ["273"],
}


def generate_room_subsets(
    df: pd.DataFrame,
    output_dir: Path | str | None = None,
    neighbor_mapping: dict[str, list[str]] | None = None,
    include_building_powers: bool = False,
    save_csv: bool = True,
    float_format: str = "%.6f",
) -> dict[str, pd.DataFrame]:
    """Build room subsets with power in watts and a single neighboring temperature feature.

    Take the mean across adjacent rooms without skipping missing values. HVAC mode is 1
    for heating and -1 for zero or negative power; missing power leaves the mode
    undefined.
    """
    target_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    neighbors_dict = neighbor_mapping if neighbor_mapping is not None else UMAR_ROOM_NEIGHBORS
    subsets: dict[str, pd.DataFrame] = {}

    # Keep shared features, excluding other rooms' fields and aggregate power.
    room_pattern = re.compile(r"_(?:272|273|274|275|276)(?:_\d+)?$")
    common_cols = [
        column for column in df.columns
        if not room_pattern.search(column)
        and column not in (*EXCLUDED_EXPORT_COLUMNS, "neighbor_temperature", "hvac_mode")
    ]
    for room, neighbors in neighbors_dict.items():
        core_cols = [
            f"temp_{room}", f"setp_{room}", f"valve_{room}",
            f"window_{room}", f"power_room_{room}",
        ]
        neighbor_cols = [f"temp_{neighbor}" for neighbor in neighbors]
        missing = [column for column in core_cols + neighbor_cols if column not in df]
        if missing:
            raise ValueError(f"Room {room} subset is missing required fields: {missing}")
        if not neighbor_cols:
            raise ValueError(f"Room {room} must have at least one neighboring room")
        selected_cols = list(dict.fromkeys(core_cols + common_cols))
        room_df = df[selected_cols].copy()
        # Average neighboring temperatures without skipping missing readings.
        room_df.insert(
            len(core_cols), "neighbor_temperature",
            df[neighbor_cols].mean(axis=1, skipna=False),
        )
        # Keep allocation and resampling in kW; convert room exports to watts.
        power_col = f"power_room_{room}"
        room_df[power_col] = room_df[power_col] * 1000.0
        power = room_df[power_col]
        # The legacy binary mode groups zero power with non-heating.
        mode = pd.Series(np.nan, index=room_df.index, dtype=float)
        mode.loc[power.gt(0)] = 1.0
        mode.loc[power.le(0)] = -1.0
        room_df.insert(len(core_cols) + 1, "hvac_mode", mode)
        room_df.attrs["room_power_unit"] = "W"
        room_df.attrs["neighbor_rooms"] = list(neighbors)
        subsets[room] = room_df

    # Validate all room fields before writing any subset.
    if save_csv:
        for room, room_df in subsets.items():
            save_processed_data(
                room_df, target_dir / f"UMAR_2024_room{room}.csv",
                float_format=float_format,
            )
    return subsets


def run_umar_preprocessing_pipeline(
    file_name: str | None = "2020",
    data_dir: Path | str | None = None,
    concatenate: bool = False,
    rule: str = "15min",
    clean_anomalies: bool = True,
    drop_unnecessary: bool = True,
    add_time_features: bool = True,
    fill_power_zero: bool = False,
    save_csv: bool = False,
    output_path: Path | str | None = None,
    generate_subsets: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the UMAR pipeline and return the processed frame and a stage report.

    Clean and allocate power before resampling. Optionally add calendar features, create
    room subsets, and save CSVs.
    """
    pipeline_info: dict[str, Any] = {}

    # Load the raw data.
    df_raw = load_umar_raw_data(
        file_name=file_name, data_dir=data_dir, concatenate=concatenate
    )
    if isinstance(df_raw, dict):
        first_key = next(iter(df_raw.keys()))
        df_current = df_raw[first_key]
        pipeline_info["loaded_file"] = first_key
    else:
        df_current = df_raw
        pipeline_info["loaded_file"] = str(file_name) if file_name else "all_4_years_combined"
    pipeline_info["raw_shape"] = df_current.shape

    # Mask anomalous readings.
    if clean_anomalies:
        df_cleaned, anomaly_report = check_and_clean_anomalies(df_current)
        pipeline_info["anomaly_report"] = anomaly_report
        df_current = df_cleaned

    # Allocate heating and cooling power to rooms.
    df_powers = calculate_room_thermal_powers(df_current, add_split_power=False)
    pipeline_info["powers_shape"] = df_powers.shape
    df_current = df_powers

    # Merge signals and drop unused features.
    if drop_unnecessary:
        df_dropped = drop_and_merge_features(df_current, drop_split_powers=True)
        pipeline_info["dropped_features_shape"] = df_dropped.shape
        df_current = df_dropped

    # Resample by signal type.
    df_resampled = adjust_data_resolution(
        df_current, rule=rule, fill_power_zero=fill_power_zero
    )

    # Add calendar features.
    if add_time_features:
        df_final = generate_time_features(df_resampled)
    else:
        df_final = df_resampled

    pipeline_info["final_shape"] = df_final.shape
    pipeline_info["columns_final"] = list(df_final.columns)

    # Save the wide CSV.
    if save_csv:
        saved_path = save_processed_data(df_final, output_path=output_path)
        pipeline_info["saved_csv_path"] = str(saved_path)

    # Generate the room subsets.
    if generate_subsets:
        subset_dir = Path(output_path).parent if output_path is not None else DEFAULT_OUTPUT_DIR
        subsets = generate_room_subsets(
            df_final, output_dir=subset_dir, save_csv=save_csv,
        )
        pipeline_info["subsets"] = subsets
        if save_csv:
            pipeline_info["subset_csv_paths"] = {
                room: str(subset_dir / f"UMAR_2024_room{room}.csv")
                for room in subsets
            }

    return df_final, pipeline_info


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Verify the end-to-end pipeline on one annual file.
    print("=" * 70)
    print('Checking the UMAR preprocessing pipeline end to end...')
    print("=" * 70)

    # Load a single annual file.
    print('\\n[Step 1] Read the 2020-2021 dataset:')
    df_raw = load_umar_raw_data(file_name="2020")
    print(f"Loaded one dataset: shape {df_raw.shape}; time range: {df_raw.index.min()} to {df_raw.index.max()}")

    # Check anomaly detection and cleaning.
    print('\\n[Step 2] Detect and clean anomalies:')
    df_cleaned, anomaly_report = check_and_clean_anomalies(df_raw)
    print(f"Cleaning complete. Affected values: {anomaly_report['total_deleted_points']} values")
    for col, detail in anomaly_report["deleted_indices_by_column"].items():
        print(f" - [{col}] Rule: {detail['threshold_used']}")
        print(f"   Values set to NaN: {detail['deleted_count']} values")

    print('\\n[Step 3] Allocate heating and cooling power to rooms (keep net power by default):')
    df_powers = calculate_room_thermal_powers(df_cleaned, add_split_power=False)
    new_cols = [c for c in df_powers.columns if c not in df_raw.columns]
    print(f"Added {len(new_cols)} room power features; data shape: {df_powers.shape}")
    print(f"New net room power columns: {new_cols}")

    # Check signal merging and column selection.
    print('\\n[Step 4] Merge valve/window signals and drop unused features (DHW/275/276):')
    df_filtered = drop_and_merge_features(df_powers, drop_split_powers=True)
    dropped_cols = [c for c in df_powers.columns if c not in df_filtered.columns]
    print(f"Removed or replaced {len(dropped_cols)} columns; remaining: {len(df_filtered.columns)}")
    print(f"Retained columns ({len(df_filtered.columns)} total): {list(df_filtered.columns)}")

    # Check 15-minute resampling.
    print('\\n[Step 5] Resample from 1 to 15 minutes by signal type:')
    df_15min = adjust_data_resolution(df_filtered, rule="15min")
    print(f"Resampling complete. Original rows: {len(df_filtered):,} -> 15-minute rows: {len(df_15min):,}")
    print(f"Retained row percentage: {len(df_15min) / len(df_filtered) * 100:.2f}%")

    # Check calendar feature generation.
    print('\\n[Step 6] Add calendar features (matching Data_274__1.csv):')
    df_with_time = generate_time_features(df_15min)
    time_feature_cols = [
        "Time month sin", "Time month cos", "Time weekday",
        "Time interval sin", "Time interval cos"
    ]
    print(f"Added five calendar features: {time_feature_cols}")
    print(f"Final modeling data shape: {df_with_time.shape}")

    print('\\nFirst five processed 15-minute rows, including calendar features:')
    sample_preview_cols = [
        "temp_amb",
        "irrad",
        "temp_274",
        "setp_274",
        "valve_274",
        "window_273",
        "cooling_power",
        "power_room_274",
        "Time month sin",
        "Time weekday",
        "Time interval sin",
    ]
    preview_cols = [c for c in sample_preview_cols if c in df_with_time.columns]
    print(df_with_time[preview_cols].head().to_string())

    # Check CSV output.
    print('\\n[Step 7] Check CSV output:')
    print(f"Default output path: {DEFAULT_OUTPUT_FILE}")
    if DEFAULT_OUTPUT_FILE.exists():
        size_mb = DEFAULT_OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"Output already exists: {DEFAULT_OUTPUT_FILE.name} | size: {size_mb:.2f} MB")
    else:
        print('No output exists yet. Use --save-all to process and save all four years.')

    if "--save-all" in sys.argv or "--all" in sys.argv:
        print('\\n--save-all requested: merging and saving all four years...')
        df_all, info_all = run_umar_preprocessing_pipeline(
            file_name=None, concatenate=True, save_csv=True
        )
        print(f"Saved all four years. Shape: {df_all.shape}")

    print("\n" + "=" * 70)
    print('All seven preprocessing checks passed.')
    print("=" * 70)

