"""Load shared data, train the eight registered models, and save comparable results."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import time
from typing import Iterable, Mapping

import numpy as np
import torch

from exp.progress import print_progress
from util.live_plot import LiveTrainingPlot, observe_training
from models.base import ModelContext
from models.registry import (
    canonical_model_name,
    create_estimator,
    get_registration,
    required_feature_union,
)
from util.artifacts import save_checkpoint, save_results, save_table
from util.result_store import (
    append_metrics, new_run_id, resolve_output_root, safe_component,
    save_history, save_prediction_data, save_snapshot,
)
from util.test_plot import save_loss_plot, save_test_plot
from util.config import (
    ExperimentConfig,
    load_dataset_config,
)
from util.data import load_canonical_dataset, summarize_dataset
from util.evaluation import summarize_runs, temperature_metrics
from util.features import (
    PreparedData,
    ThermalWindows,
    count_valid_windows,
    prepare_windows,
    split_by_time,
)


# ---------------------------------------------------------------------------
# Data preparation and audit
# ---------------------------------------------------------------------------


def window_fingerprint(windows: ThermalWindows) -> str:
    """Fingerprint window timestamps to verify that models use the same samples."""
    # Hash only the window start and end times.
    digest = hashlib.sha256()
    for start, end in windows.spans:
        digest.update(start.isoformat().encode("utf-8"))
        digest.update(b"|")
        digest.update(end.isoformat().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_experiment_data(
    config: ExperimentConfig,
    model_names: tuple[str, ...],
) -> tuple[PreparedData, dict[str, object]]:
    """Load preprocessed CSVs and build shared windows with all required features."""
    print_progress('Data', f"Reading dataset config: {config.dataset_config}")
    # The config selects one CSV to split or three pre-split CSVs.
    dataset_config = load_dataset_config(config.dataset_config)
    dataset = load_canonical_dataset(dataset_config)

    # Use the union of required features so all models see the same windows and splits.
    feature_names = required_feature_union(model_names)

    print_progress(
        'Data',
        (
            f"Loaded data: file={dataset.dataset_path}, "
            f"rows={len(dataset.frame)}, model features={len(feature_names)}"
        ),
    )
    prepared = prepare_windows(
        dataset.splits if dataset.splits is not None else dataset.frame,
        feature_names,
        config.history_steps,
        config.horizon_steps,
        dataset.interval_minutes,
        dataset_config.split_ratios,
        config.window_stride,
        direct_thermal_power=dataset_config.direct_thermal_power,
        interval_start_inputs=dataset_config.interval_start_inputs,
    )
    print_progress(
        'Data',
        (
            f"Built windows: train={len(prepared.train)}, "
            f"validation={len(prepared.validation)}, test={len(prepared.test)}"
        ),
    )

    # Record data and occupancy sources and window counts for reproducibility.
    data_summary = summarize_dataset(dataset)
    data_summary["windows"] = prepared.summary
    return prepared, data_summary


def inspect_data(
    config: ExperimentConfig,
    model_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Summarize the data source, splits, and valid windows using the training rules."""
    names = model_names or config.models
    print_progress(
        'Data',
        f"Checking data for models: {', '.join(names)}",
    )
    dataset_config = load_dataset_config(config.dataset_config)
    dataset = load_canonical_dataset(dataset_config)
    feature_names = required_feature_union(names)

    # Count windows using the same split, episode, and forecast-origin rules as training.
    splits = dataset.splits
    if splits is None:
        splits = split_by_time(dataset.frame, dataset_config.split_ratios)
    counts = {
        "train": count_valid_windows(
            splits.train,
            feature_names,
            config.history_steps,
            config.horizon_steps,
            dataset.interval_minutes,
            config.window_stride,
        ),
        "validation": count_valid_windows(
            splits.validation,
            feature_names,
            config.history_steps,
            config.horizon_steps,
            dataset.interval_minutes,
            config.window_stride,
        ),
        "test": count_valid_windows(
            splits.test,
            feature_names,
            config.history_steps,
            config.horizon_steps,
            dataset.interval_minutes,
            config.window_stride,
        ),
    }

    # Count windows without building training tensors.
    summary = summarize_dataset(dataset)
    summary["windows"] = {
        "feature_names": list(feature_names),
        "split_dataset": dataset_config.split_dataset,
        "direct_thermal_power": dataset_config.direct_thermal_power,
        "interval_start_inputs": dataset_config.interval_start_inputs,
        "split_ratios": (
            list(dataset_config.split_ratios)
            if dataset_config.split_ratios is not None else None
        ),
        "window_stride": config.window_stride,
        "train_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        **{
            f"{label}_windows": count
            for label, count in counts.items()
        },
    }
    print_progress(
        'Data',
        (
            f"Data check complete: train windows={counts['train']}, "
            f"validation windows={counts['validation']}, test windows={counts['test']}"
        ),
    )
    return summary


# ---------------------------------------------------------------------------
# Single model run
# ---------------------------------------------------------------------------


def _run_seeds(model_name: str, seeds: tuple[int, ...]) -> tuple[int, ...]:
    """Run all configured seeds, even for deterministic estimators."""
    return seeds


def _estimator_parameters(
    config: ExperimentConfig,
    model_name: str,
) -> dict[str, object]:
    """Resolve estimator parameters and apply the window irradiance option only to the RC model."""
    configured = config.model_params.get(model_name)
    if configured is None:
        canonical_name = canonical_model_name(model_name)
        configured = next(
            (values for name, values in config.model_params.items()
             if canonical_model_name(name) == canonical_name),
            {},
        )
    parameters = dict(configured)
    if model_name in {"3r2c"}:
        parameters["rc_window_irr"] = config.rc_window_irr
    return parameters


def _checkpoint_state(
    payload: Mapping[str, object],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Separate tensor weights from JSON metadata."""
    # Save weights with PyTorch and keep architecture and settings as readable metadata.
    metadata = dict(payload)
    raw_state = metadata.pop("state_dict", {})
    if not isinstance(raw_state, Mapping):
        raise ValueError('Model state_dict must be a mapping.')

    # Validate tensors before saving so malformed weights are rejected early.
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"state_dict entry {name} is not a tensor.")
        state[str(name)] = value
    return state, metadata


def _run_one(
    model_name: str,
    seed: int,
    config: ExperimentConfig,
    prepared: PreparedData,
    output_directory: Path,
    save_weights: bool = True,
    save_test_plots: bool = True,
) -> dict[str, object]:
    """Fit and evaluate one model run and save its artifacts."""
    # Check that the shared windows include all inputs required by the model.
    registration = get_registration(model_name)
    missing = sorted(
        set(registration.required_features).difference(prepared.train.feature_names)
    )
    if missing:
        raise ValueError(f"Shared windows are missing model features: {', '.join(missing)}.")

    # Build the registered estimator from its name and parameters.
    estimator = create_estimator(
        model_name,
        _estimator_parameters(config, model_name),
    )

    # Share feature order, time scale, and normalization across models.
    context = ModelContext(
        feature_names=prepared.train.feature_names,
        history_steps=config.history_steps,
        horizon_steps=config.horizon_steps,
        interval_minutes=_interval_from_spans(prepared.train),
        scaler=prepared.scaler,
        interval_start_inputs=prepared.summary.get("interval_start_inputs", False),
    )

    # Use validation for early stopping and model selection; hold out test data until evaluation.
    monitor = LiveTrainingPlot(
        estimator, prepared.validation, context, config.training,
        f"{model_name} | seed {seed}",
        history_path=output_directory / "history.csv",
    )
    with observe_training(monitor):
        fit_result = estimator.fit(
            prepared.train,
            prepared.validation,
            context,
            config.training,
            seed,
        )
    print_progress(
        'Model',
        (
            f"Training complete: model={model_name}, seed={seed}, "
            f"best epoch={fit_result.best_epoch}, "
            f"elapsed={fit_result.elapsed_seconds:.2f} seconds"
        ),
    )

    # Record loss components and physical parameters even when weights are not saved.
    components = getattr(estimator, "loss_components", ())
    if components:
        save_table(output_directory / "loss_components.csv", components)
    training_metadata = getattr(estimator, "training_metadata", None)
    if callable(training_metadata):
        save_results(output_directory / "training_stages.json",
                     training_metadata())
    physics = getattr(estimator, "physical_metadata", None)
    if callable(physics):
        save_results(output_directory / "rc_parameters.json", physics())

    # Save the loss plot after training, before any test failure can prevent it.
    loss_plot = save_loss_plot(
        output_directory / "figures" / "loss_curve.png",
        fit_result.train_losses, fit_result.validation_losses,
        fit_result.best_epoch, f"{config.name}_{model_name} | seed {seed}",
    )

    # Time prediction only, excluding training.
    print_progress(
        'Prediction',
        f"Starting: model={model_name}, test windows={len(prepared.test)}",
    )
    prediction_started = time.perf_counter()
    predictions = estimator.predict(prepared.test, context, config.training)
    inference_seconds = time.perf_counter() - prediction_started
    monitor.finish(prepared.test, predictions, fit_result)

    # Targets and predictions are both in Celsius.
    metrics = temperature_metrics(prepared.test.targets_c, predictions)
    print_progress(
        'Prediction',
        f"Finished: model={model_name}, elapsed={inference_seconds:.2f} seconds",
    )
    save_history(
        output_directory / "history.csv", fit_result.train_losses,
        fit_result.validation_losses, fit_result.best_epoch,
    )
    save_prediction_data(
        output_directory / "predictions.npz", prepared.test, predictions,
        context.interval_minutes,
    )
    checkpoint = None
    if save_weights:
        state_dict, model_metadata = _checkpoint_state(estimator.state_payload())
        checkpoint = output_directory / "checkpoint.pt"
        save_checkpoint(checkpoint, state_dict, {
            "model": model_name, "seed": seed, "model_state": model_metadata,
            "architecture": estimator.describe_architecture(),
            "scaler": context.scaler.to_dict(), "history_steps": context.history_steps,
            "horizon_steps": context.horizon_steps,
            "interval_minutes": context.interval_minutes,
            "interval_start_inputs": context.interval_start_inputs,
            "feature_names": list(context.feature_names),
        })
        print_progress('Artifacts', f"Saved checkpoint: {checkpoint}")
    test_plot = None
    if save_test_plots:
        test_plot = save_test_plot(
            output_directory / "figures" / "test_predictions.png",
            prepared.test, predictions, f"{config.name}_{model_name} | seed {seed}",
        )
    return {
        "model": model_name,
        "seed": seed,
        "status": "completed",
        "deterministic": model_name in {"3r2c"},
        "best_epoch": fit_result.best_epoch,
        "completed_epochs": fit_result.completed_epochs,
        "final_train_loss": fit_result.train_losses[-1],
        "final_validation_loss": fit_result.validation_losses[-1],
        "training_seconds": fit_result.elapsed_seconds,
        "inference_seconds": inference_seconds,
        "parameter_count": estimator.parameter_count(),
        "window_fingerprint": window_fingerprint(prepared.test),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "test_plot": str(test_plot) if test_plot is not None else None,
        "loss_plot": str(loss_plot),
        "output_directory": str(output_directory),
        **metrics,
    }


def _interval_from_spans(windows: ThermalWindows) -> int:
    """Infer the sampling interval in minutes from the window timestamps."""
    # For multi-step forecasts, use adjacent target timestamps.
    if windows.target_times.shape[1] >= 2:
        delta = windows.target_times[0, 1] - windows.target_times[0, 0]
    else:
        # For a one-step forecast, infer the interval from the full window span.
        start, end = windows.spans[0]
        total_steps = windows.raw_inputs.shape[1]
        delta = (end - start).to_timedelta64() / (total_steps - 1)
    minutes = int(delta / np.timedelta64(1, "m"))
    if minutes < 1:
        raise ValueError('Could not infer a positive sampling interval from the windows.')
    return minutes


# ---------------------------------------------------------------------------
# Experiment orchestration and configuration overrides
# ---------------------------------------------------------------------------


def run_comparison(
    config: ExperimentConfig,
    output_root: str | Path = "results",
    model_names: Iterable[str] | None = None,
    prepared_data: PreparedData | None = None,
    save_weights: bool = True,
    save_test_plots: bool = True,
) -> dict[str, object]:
    """Run models on shared windows and keep completed results if a run fails."""
    # Use explicit model names when supplied; otherwise follow the experiment config.
    names = tuple(model_names) if model_names is not None else config.models
    if not names:
        raise ValueError('At least one model is required.')
    if len(set(names)) != len(names) or len(set(config.seeds)) != len(config.seeds):
        raise ValueError('Model and seed lists must not contain duplicates.')

    # Reject unknown models before loading data or starting training.
    for name in names:
        get_registration(name)
    if not isinstance(save_weights, bool) or not isinstance(save_test_plots, bool):
        raise ValueError('The save-weights and save-test-plots flags must be Boolean.')

    print_progress(
        'Experiment',
        f"Starting: name={config.name}, models={', '.join(names)}",
    )
    if prepared_data is None:
        # Prepare the data once for all models.
        prepared, data_summary = prepare_experiment_data(config, names,)
    else:
        # Tests and callers can supply already prepared windows.
        prepared = prepared_data
        data_summary = {"windows": prepared.summary, "source": "provided"}
        print_progress(
            'Data',
            (
                f"Using prepared windows: train={len(prepared.train)}, "
                f"validation={len(prepared.validation)}, test={len(prepared.test)}"
            ),
        )

    root = resolve_output_root(output_root)
    run_id = new_run_id()
    # Share one batch ID but save each model separately by experiment and model name.
    output_directory = root / "comparisons" / run_id
    output_directory.mkdir(parents=True, exist_ok=False)
    output_options = {
        "output_root": str(root), "save_weights": save_weights,
        "save_test_plots": save_test_plots,
    }
    save_snapshot(output_directory / "config_snapshot.json", config, output_options,
                  run_id=run_id, selected_models=list(names))
    print_progress('Experiment', f"Results directory: {output_directory.resolve()}")
    runs: list[dict[str, object]] = []

    def persist_comparison():
        """Refresh the summary after each run so partial results survive interruption."""
        completed = [run for run in runs if run["status"] == "completed"]
        summaries = summarize_runs(completed) if completed else []
        payload = {
            "experiment": config.name, "models": list(names),
            "run_id": run_id, "comparison_directory": str(output_directory),
            "metrics_file": str(root / "metrics.csv"),
            "all_successful": len(completed) == len(runs),
            "data": data_summary, "runs": runs, "summary": summaries,
        }
        save_table(output_directory / "runs.csv", runs)
        save_table(output_directory / "summary_boptest.csv", summaries)
        save_results(output_directory / "results.json", payload)
        return payload

    # Keep a stable ordering by model, then by seed.
    for model_name in names:
        for seed in _run_seeds(model_name, config.seeds): # Run each model and seed with the same single-run entry point.
            model_directory = (
                root / f"{safe_component(config.name)}_{safe_component(model_name)}"
                / run_id / f"seed_{seed}"
            )
            model_directory.mkdir(parents=True, exist_ok=False)
            save_snapshot(model_directory / "config_snapshot.json", config, output_options,
                          run_id=run_id, selected_model=model_name, seed=seed)
            print_progress(
                'Model',
                f"Starting: model={model_name}, seed={seed}",
            )
            try:
                run_result = _run_one(
                    model_name,
                    seed,
                    config,
                    prepared,
                    model_directory,
                    save_weights=save_weights,
                    save_test_plots=save_test_plots,
                )
                runs.append(run_result)
                print_progress(
                    'Model',
                    f"Finished: model={model_name}, seed={seed}",
                )
            except Exception as error:  # noqa: BLE001 - Log the failure and continue the batch.
                # A failed run must not discard results from the rest of the batch.
                print_progress(
                    'Model',
                    (
                        f"Failed: model={model_name}, seed={seed},"
                        f"{type(error).__name__}: {error}"
                    ),
                )
                runs.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "window_fingerprint": window_fingerprint(prepared.test),
                        "output_directory": str(model_directory),
                    }
                )

            runs[-1].update({
                "run_id": run_id, "experiment": config.name,
                "history_steps": config.history_steps,
                "horizon_steps": config.horizon_steps,
            })
            save_results(model_directory / "metrics.json", runs[-1])
            append_metrics(root / "metrics.csv", runs[-1])
            persist_comparison()

    payload = persist_comparison()
    completed = [run for run in runs if run["status"] == "completed"]
    print_progress(
        'Experiment',
        (
            f"Finished: successful={len(completed)}/{len(runs)}, "
            f"results directory={output_directory.resolve()}"
        ),
    )
    return payload


def run_model(
    config: ExperimentConfig,
    model_name: str,
    **kwargs: object,
) -> dict[str, object]:
    """Run a single registered model through the shared experiment workflow."""
    # Reuse the comparison workflow for consistent output and error handling.
    return run_comparison(config, model_names=(model_name,), **kwargs)


def with_overrides(
    config: ExperimentConfig,
    history_steps: int | None = None,
    horizon_steps: int | None = None,
    window_stride: int | None = None,
    seeds: tuple[int, ...] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    patience: int | None = None,
    device: str | None = None,
    rc_window_irr: bool | None = None,
    plot_every: int | None = None,
) -> ExperimentConfig:
    """Apply explicit overrides to a new, validated experiment configuration."""
    # Keep the original config and preserve settings where overrides are None.
    training = replace(
        config.training,
        epochs=config.training.epochs if epochs is None else epochs,
        batch_size=(
            config.training.batch_size if batch_size is None else batch_size
        ),
        learning_rate=(
            config.training.learning_rate
            if learning_rate is None
            else learning_rate
        ),
        patience=config.training.patience if patience is None else patience,
        device=config.training.device if device is None else device,
        plot_every=config.training.plot_every if plot_every is None else plot_every,
    )

    # Apply window and seed overrides after updating the training settings.
    return replace(
        config,
        history_steps=(
            config.history_steps if history_steps is None else history_steps
        ),
        horizon_steps=(
            config.horizon_steps if horizon_steps is None else horizon_steps
        ),
        window_stride=(
            config.window_stride if window_stride is None else window_stride
        ),
        seeds=config.seeds if seeds is None else seeds,
        rc_window_irr=(
            config.rc_window_irr
            if rc_window_irr is None
            else rc_window_irr
        ),
        training=training,
    )
