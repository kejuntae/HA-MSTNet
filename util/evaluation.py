"""Forecast accuracy in Celsius and summary statistics across runs."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Iterable, Mapping

import numpy as np


def temperature_metrics(
    targets_c: np.ndarray,
    predictions_c: np.ndarray,
) -> dict[str, float | int]:
    """Compute MAE, RMSE, MAPE, and R² for matching Celsius arrays."""
    targets = np.asarray(targets_c, dtype=float)
    predictions = np.asarray(predictions_c, dtype=float)
    if targets.shape != predictions.shape or targets.ndim != 2 or targets.size == 0:
        raise ValueError('Targets and predictions must be nonempty two-dimensional arrays of the same shape.')
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError('Targets and predictions must contain only finite values.')
    errors = predictions - targets
    absolute = np.abs(errors)
    squared = errors ** 2
    denominator_mask = np.abs(targets) > 1e-6
    if denominator_mask.any():
        mape = float(
            np.mean(absolute[denominator_mask] / np.abs(targets[denominator_mask]))
            * 100.0
        )
    else:
        mape = 0.0
    centered = targets - float(targets.mean())
    total_variation = float(np.sum(centered ** 2))
    residual = float(np.sum(squared))
    r2 = 1.0 - residual / total_variation if total_variation > 0.0 else 0.0
    return {
        "mae_c": float(absolute.mean()),
        "rmse_c": float(np.sqrt(squared.mean())),
        "mape_percent": mape,
        "r2": float(r2),
        "sample_count": int(targets.size),
    }


def summarize_runs(
    runs: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Report means and sample standard deviations by model."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for run in runs:
        model = run.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError('Each run result must include a model name.')
        grouped[model].append(run)

    summaries: list[dict[str, object]] = []
    for model, model_runs in grouped.items():
        numeric_keys = sorted(
            set.intersection(
                *(
                    {
                        key for key, value in run.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                    }
                    for run in model_runs
                )
            )
        )
        summary: dict[str, object] = {
            "model": model,
            "run_count": len(model_runs),
        }
        for key in numeric_keys:
            values = [float(run[key]) for run in model_runs]
            summary[f"{key}_mean"] = statistics.mean(values)
            summary[f"{key}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries
