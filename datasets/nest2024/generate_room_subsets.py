"""Export separate UMAR modeling tables for rooms 272, 273, and 274."""

import sys
from pathlib import Path

import pandas as pd

from preprocessing_UMAR import generate_room_subsets as _generate_room_subsets

# Default input and output paths.
BASE_DIR: Path = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CSV: Path = (
    BASE_DIR / "datasets" / "data_preprocessed" / "NEST2024" / "UMAR_2024.csv"
)
DEFAULT_OUTPUT_DIR: Path = DEFAULT_SOURCE_CSV.parent

# UMAR room adjacency: bedrooms 272 and 274 each share a wall with living room 273.
UMAR_ROOM_NEIGHBORS: dict[str, list[str]] = {
    "272": ["273"],
    "273": ["272", "274"],
    "274": ["273"],
}

# Weather covariates.
WEATHER_COLUMNS: list[str] = ["temp_amb", "irrad"]

# Five calendar features aligned with the benchmark.
TIME_FEATURE_COLUMNS: list[str] = [
    "Time month sin",
    "Time month cos",
    "Time weekday",
    "Time interval sin",
    "Time interval cos",
]

# Four physical consistency flags.
QUALITY_FLAG_COLUMNS: list[str] = [
    "thermal_valves_valid",
    "thermal_heating_valid",
    "thermal_cooling_valid",
    "thermal_allocation_valid",
]


def generate_room_subsets(
    source_df_or_path: pd.DataFrame | Path | str = DEFAULT_SOURCE_CSV,
    output_dir: Path | str | None = None,
    neighbor_mapping: dict[str, list[str]] | None = None,
    include_building_powers: bool = False,
    include_quality_flags: bool = True,
    save_csv: bool = True,
    float_format: str = "%.6f",
) -> dict[str, pd.DataFrame]:
    """Export room-specific features and aggregate neighboring temperatures.

    Pass a wide table or its CSV path. Building-level power is always excluded; quality
    flags and CSV export are optional.
    """
    # Load the wide table.
    if isinstance(source_df_or_path, (str, Path)):
        src_path = Path(source_df_or_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Source dataset not found: {src_path}")
        print(f"Reading source dataset: {src_path} ...")
        df = pd.read_csv(src_path, parse_dates=["time"], index_col="time")
    else:
        df = source_df_or_path.copy()
        if "time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")

    print(f"Source dataset: {len(df):,} rows x {df.shape[1]} columns | time range: {df.index.min()} to {df.index.max()}")

    # Reuse the main pipeline's adjacency and column-selection rules.
    if not include_quality_flags:
        df = df.drop(columns=QUALITY_FLAG_COLUMNS, errors="ignore")
    return _generate_room_subsets(
        df,
        output_dir=output_dir,
        neighbor_mapping=neighbor_mapping,
        include_building_powers=include_building_powers,
        save_csv=save_csv,
        float_format=float_format,
    )


def main() -> None:
    """Load the default wide table and write the room CSVs."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    generate_room_subsets(
        source_df_or_path=DEFAULT_SOURCE_CSV,
        output_dir=DEFAULT_OUTPUT_DIR,
        include_building_powers=False,
        include_quality_flags=True,
        save_csv=True,
    )


if __name__ == "__main__":
    main()
