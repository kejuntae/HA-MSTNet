"""Command-line entry point for the building temperature forecasting benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from exp.experiment import inspect_data, run_comparison, run_model, with_overrides
from util.live_plot import wait_for_training_plots
from models.registry import canonical_model_name, list_models
from util.config import load_dataset_config, load_experiment_config


# Resolve default paths relative to the project root, not the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = PROJECT_ROOT / "configs" / "experiments" / "boptest_single_zone.json"


def _positive_integer_argument(value: str) -> int:
    """Parse a strictly positive integer override."""
    # Validate the CLI value before passing it to the experiment configuration.
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer_argument(value: str) -> int:
    """Parse a nonnegative random seed."""
    # Seeds may be zero but cannot be negative.
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_float_argument(value: str) -> float:
    """Parse a finite positive float."""
    # Require a finite learning rate greater than zero.
    parsed = float(value)
    if not 0.0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _nonnegative_float_argument(value):
    """Parse a nonnegative loss weight; zero disables the term."""
    parsed = float(value)
    if not 0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return parsed


def _resolved_config(arguments: argparse.Namespace):
    """Load the JSON configuration and apply explicit CLI overrides."""
    # Keep values from JSON unless the user supplies a CLI override.
    config = load_experiment_config(arguments.config)
    seeds = tuple(arguments.seeds) if arguments.seeds else None
    config = with_overrides(
        config,
        history_steps=arguments.history_steps,
        horizon_steps=arguments.horizon_steps,
        window_stride=arguments.window_stride,
        seeds=seeds,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        patience=arguments.patience,
        device=arguments.device,
        rc_window_irr=arguments.rc_window_irr,
        plot_every=arguments.plot_every,
    )
    return config


def _print_json(payload: object) -> None:
    """Print results as standard UTF-8 JSON."""
    # Reject NaN and infinity to keep the output valid JSON.
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return a script-friendly exit code."""
    parser = argparse.ArgumentParser(description="HA-MSTNet: building temperature forecasting benchmark")

    # Load defaults from the experiment configuration.
    parser.add_argument("command", nargs="?", choices=("list-models", "data", "train", "compare"), default="train",
                        help="Command to run")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT), help="Path to the experiment JSON config")
    parser.add_argument("--model", type=canonical_model_name, choices=list_models(), default=None,
                        help="Model for train or data. By default, train uses the first configured model "
                             "and data checks all configured models.")
    parser.add_argument("--output-root", default=None,
                        help="Results directory; relative paths are resolved from the project root")
    parser.add_argument("--save-weights", action=argparse.BooleanOptionalAction,
                        default=True, help="Save the best model weights")
    parser.add_argument("--save-test-plots", action=argparse.BooleanOptionalAction,
                        default=True, help="Save test forecast plots independently of live plotting")
    parser.add_argument("--history-steps", type=_positive_integer_argument, default=None, help="History window length in steps")
    parser.add_argument("--horizon-steps", type=_positive_integer_argument, default=None, help="Forecast horizon in steps")
    parser.add_argument("--window-stride", type=_positive_integer_argument, default=None, help="Steps between successive window starts")
    parser.add_argument("--seeds", type=_nonnegative_integer_argument, nargs="+", default=None, help="Random seeds")
    parser.add_argument("--epochs", type=_positive_integer_argument, default=None, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=_positive_integer_argument, default=None, help="Training batch size")
    parser.add_argument("--learning-rate", type=_positive_float_argument, default=None,
                        help="Initial learning rate; cosine decay to 1%% of this value")
    parser.add_argument("--patience", type=_positive_integer_argument, default=None, help="Early-stopping patience in epochs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="Compute device")
    parser.add_argument("--plot-every", type=_nonnegative_integer_argument, default=None,
                        help="Refresh live plots every N epochs; use 0 to disable. Defaults to the config.")

    # RC model options.
    parser.add_argument("--rc-window-irr", action=argparse.BooleanOptionalAction,
                        default=None, help="Project global horizontal irradiance onto a vertical window (3R2C only)")

    # Explicit CLI arguments override the defaults for batch runs.
    arguments = parser.parse_args(argv)
    if arguments.command == "list-models":
        # Use the registry as the only source of model names.
        _print_json(list(list_models()))
        return 0

    try:
        # Use the same configuration and window stride for all experiment commands.
        config = _resolved_config(arguments)
        if arguments.output_root is None:
            dataset = load_dataset_config(config.dataset_config)
            folder = ("boptest_single_zone"
                      if "boptest" in dataset.name.lower()
                      else "NEST2024_three_zone")
            arguments.output_root = str(PROJECT_ROOT / "results" / folder)
        if arguments.command == "data":
            # Report fields, missing values, splits, and windows without changing data.
            # Check the same input requirements that training uses for the selected model.
            names = (arguments.model,) if arguments.model is not None else None
            _print_json(inspect_data(config, names))
            return 0
        if arguments.command == "train":
            # Train only the selected model.
            result = run_model(
                config, arguments.model or config.models[0],
                output_root=arguments.output_root,
                save_weights=arguments.save_weights,
                save_test_plots=arguments.save_test_plots,
            )
        else:
            # Compare all models in the configuration.
            result = run_comparison(
                config, output_root=arguments.output_root,
                save_weights=arguments.save_weights,
                save_test_plots=arguments.save_test_plots,
            )

        # Write results as JSON and return 1 if any run fails.
        _print_json(result)
        wait_for_training_plots()
        return 0 if result["all_successful"] else 1
    except (OSError, ValueError, RuntimeError) as error:
        # Write expected errors to stderr and return 2.
        payload = {"status": "failed", "error_type": type(error).__name__, "error_message": str(error)}
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    # Pass the exit code to the operating system.
    raise SystemExit(main())
