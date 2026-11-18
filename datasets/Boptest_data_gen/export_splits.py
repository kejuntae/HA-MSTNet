"""Assemble a completed BOPTEST batch into train, validation, and test CSVs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_thermal_data import CASE_DIR, SOURCE_DIR, SPLITS, prepare_split


CALENDAR_ORIGIN = "2019-01-01"
STEP_SECONDS = 900
DAY_SECONDS = 86400


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble_episode(folder: Path, processed_dir: Path, planned: dict) -> pd.DataFrame:
    manifest = read_json(folder / "run_manifest.json")
    if (manifest["status"] != "completed" or manifest["pilot"] or
            manifest["profile"] != "full" or manifest["episode"] != planned):
        raise ValueError(f"{folder}: expected a completed full-length run for {planned['id']}.")
    split, episode_id, run_id = planned["split"], planned["id"], manifest["run_id"]
    directory = processed_dir / run_id / split / episode_id
    transitions = pd.read_csv(directory / "transitions_15min.csv")
    states = pd.read_csv(directory / "states_15min.csv")
    start = (planned["start_day"] + 7) * DAY_SECONDS
    end = planned["end_day"] * DAY_SECONDS
    expected_times = np.arange(start, end + STEP_SECONDS, STEP_SECONDS)
    if (not np.array_equal(states["time_s"], expected_times) or
            not np.array_equal(transitions["time_start_s"], expected_times[:-1]) or
            not np.array_equal(transitions["time_end_s"], expected_times[1:]) or
            not transitions["episode_id"].eq(episode_id).all() or
            not transitions["split"].eq(split).all()):
        raise ValueError(f"{folder}: inconsistent 15-minute state and transition records.")

    starts = pd.to_datetime(transitions["time_start_s"], unit="s", origin=CALENDAR_ORIGIN)
    ends = pd.to_datetime(transitions["time_end_s"], unit="s", origin=CALENDAR_ORIGIN)
    history = transitions["time_start_s"] < start + DAY_SECONDS
    forecast = transitions["time_start_s"] >= end - DAY_SECONDS
    if not np.array_equal(transitions["target_origin_allowed"], ~(history | forecast)):
        raise ValueError(f"{folder}: prediction origin flags do not match the episode plan.")

    frame = pd.DataFrame({
        "datetime": starts, "datetime_start": starts, "datetime_end": ends,
        "year": starts.dt.year, "month": starts.dt.month, "day": starts.dt.day,
        "hour": starts.dt.hour, "minute": starts.dt.minute, "split": split,
        "episode_id": episode_id, "run_id": run_id, "episode_row": transitions.index,
        "is_episode_start": transitions.index == 0,
        "is_episode_end": transitions.index == len(transitions) - 1,
        "data_role": np.select([history, forecast], ["history_support", "forecast_support"],
                               default="target"),
    })
    frame = frame.join(transitions[["target_origin_allowed"] + [
        name for name in transitions if name not in (*frame.columns, "target_origin_allowed")
    ]])
    for name in states.drop(columns="time_s"):
        if name not in frame:
            frame[name] = states[name].to_numpy()[:-1]
        next_name = f"next_{name}"
        if next_name not in frame:
            frame[next_name] = states[name].to_numpy()[1:]
    return frame


def export_batch(batch_path: Path, output_dir: Path, processed_dir: Path) -> None:
    batch = read_json(batch_path)
    if batch["status"] != "completed":
        raise ValueError("The collection batch did not finish.")
    folders = {}
    for entry in batch["runs"]:
        folder = CASE_DIR / entry
        manifest = read_json(folder / "run_manifest.json")
        episode_id = manifest["episode"]["id"]
        if episode_id in folders:
            raise ValueError(f"Duplicate episode: {episode_id}")
        folders[episode_id] = folder
    if not folders:
        raise ValueError("The batch has no completed runs.")
    config = read_json(next(iter(folders.values())) / "config_snapshot.json")
    planned = config["episodes"]
    if folders.keys() != {episode["id"] for episode in planned}:
        raise ValueError("Export requires all episodes in the collection plan.")

    frames = {split: [] for split in SPLITS.values()}
    episodes = []
    for episode in planned:
        episode_id, split = episode["id"], episode["split"]
        folder = folders[episode_id]
        if read_json(folder / "config_snapshot.json") != config:
            raise ValueError(f"{folder}: batch collection configurations differ.")
        frame = assemble_episode(folder, processed_dir, episode)
        frames[split].append(frame)
        episodes.append({
            "episode_id": episode_id, "split": split, "run_id": frame["run_id"].iloc[0],
            "rows": len(frame), "target_rows": int(frame["target_origin_allowed"].sum()),
            "forecast_file": (folder / "forecasts.jsonl").relative_to(CASE_DIR).as_posix(),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in SPLITS.items():
        path = output_dir / f"{name}.csv"
        pd.concat(frames[split], ignore_index=True).to_csv(path, index=False)
        prepared = prepare_split(path, split)
        print(f"{path}: {len(prepared)} rows")
    (output_dir / "export_manifest.json").write_text(
        json.dumps({"calendar_origin": CALENDAR_ORIGIN, "step_s": STEP_SECONDS,
                    "episodes": episodes}, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--processed-dir", type=Path, default=CASE_DIR / "processed")
    args = parser.parse_args()
    export_batch(args.batch, args.output_dir, args.processed_dir)


if __name__ == "__main__":
    main()
