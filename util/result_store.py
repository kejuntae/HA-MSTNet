"""Manage experiment directories, per-run artifacts, and combined metrics."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import csv
import io
import re
import uuid

import numpy as np

from util.artifacts import save_results, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = (
    "experiment", "model", "seed", "status", "mae_c", "rmse_c",
    "mape_percent", "r2", "history_steps", "horizon_steps",
    "sample_count", "best_epoch",
    "training_seconds", "inference_seconds", "parameter_count",
    "run_id", "window_fingerprint", "output_directory", "error_message",
)


def resolve_output_root(value):
    """Expand home paths and resolve relative output paths from the project root."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def safe_component(value):
    """Make a run label safe to use as one path component."""
    name = re.sub(r'[^\w.-]+', "_", str(value)).strip("._")
    if not name:
        raise ValueError('The results directory name cannot be empty.')
    return name


def new_run_id():
    """Combine a timestamp with a random suffix for unique run IDs."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]


def save_history(path, train_losses, validation_losses, best_epoch):
    """Save per-epoch history atomically, even without live plots."""
    save_table(path, [
        {
            "epoch": index + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "is_best_epoch": index + 1 == best_epoch,
        }
        for index, (train_loss, validation_loss) in enumerate(
            zip(train_losses, validation_losses)
        )
    ])


def save_prediction_data(path, windows, predictions, interval_minutes):
    """Save Celsius predictions and timestamps in window order without object arrays."""
    target_times = np.asarray(windows.target_times, dtype="datetime64[ns]")
    forecast_origins = target_times[:, 0] - np.timedelta64(interval_minutes, "m")
    # Forecast origins are last observed timestamps, not the first target timestamps.
    np.savez_compressed(
        path, prediction_c=predictions, truth_c=windows.targets_c,
        target_times=target_times, forecast_origins=forecast_origins,
        lead_steps=np.arange(1, predictions.shape[1] + 1),
    )


def save_snapshot(path, config, options, **extra):
    """Save resolved settings and output flags."""
    save_results(path, {**asdict(config), "output": options, **extra})


def append_metrics(path, row):
    """Append CSV metrics with a UTF-8 BOM for Excel compatibility."""
    import os

    path = Path(path)
    # Use a fixed-size lock file for Windows and POSIX locking.
    with path.with_suffix(".lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(
                buffer, fieldnames=METRIC_COLUMNS,
                extrasaction="ignore", lineterminator="\n",
            )
            if not path.exists() or path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
            with path.open("a", encoding="utf-8-sig", newline="") as stream:
                stream.write(buffer.getvalue())
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
