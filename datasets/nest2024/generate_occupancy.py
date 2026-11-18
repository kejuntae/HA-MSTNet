"""Add a schedule-based occupancy proxy to the NEST UMAR training tables."""

import argparse
from pathlib import Path

import pandas as pd

from preprocessing_UMAR import DEFAULT_OUTPUT_DIR


OUTPUT_TABLES = (
    "UMAR_2024.csv",
    "UMAR_2024_room272.csv",
    "UMAR_2024_room273.csv",
    "UMAR_2024_room274.csv",
)


def add_occupancy(path: Path) -> None:
    """Update occupancy from the CSV timestamps without changing the text of other values."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    timestamps = pd.to_datetime(frame["time"])
    if timestamps.isna().any():
        raise ValueError(f"{path}: missing timestamps in the time column")
    away = (timestamps.dt.dayofweek < 5) & timestamps.dt.hour.between(8, 17)
    frame["occupancy"] = (~away).astype(int)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a schedule-based occupancy proxy.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = [args.output_dir / name for name in OUTPUT_TABLES]
    for path in paths:
        if not path.is_file():
            parser.error(f"Missing {path}; run preprocessing_UMAR.py --save-all first.")
    for path in paths:
        add_occupancy(path)


if __name__ == "__main__":
    main()
