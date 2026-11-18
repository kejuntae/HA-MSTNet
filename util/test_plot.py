"""Save forecast and loss PNGs with Agg, without a graphical display."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np


def _save_png(figure, path):
    """Save as PNG and release the canvas without changing the live backend."""
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError('Plot output paths must end in .png.')
    target.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure)
    try:
        figure.savefig(target, format="png", dpi=180, facecolor="white")
    finally:
        figure.clear()
    return target


def save_test_plot(path, windows, predictions, label):
    """Plot the first, middle, and last test windows against real time."""
    actual = np.asarray(windows.targets_c, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if (actual.ndim != 2 or actual.size == 0
            or predictions.shape != actual.shape
            or not np.isfinite(predictions).all()
            or not np.isfinite(actual).all()):
        raise ValueError('Test predictions and targets must be finite, nonempty two-dimensional arrays of the same shape.')
    times = np.asarray(windows.target_times, dtype="datetime64[ns]")
    if times.shape != actual.shape or np.isnat(times).any():
        raise ValueError('Target timestamps must match the prediction shape and contain no missing values.')
    indices = np.unique(
        np.linspace(0, len(actual) - 1, min(3, len(actual))).astype(int)
    )
    figure = Figure(figsize=(12, 3.1 * len(indices) + 0.7), layout="constrained")
    axes = figure.subplots(len(indices), 1, squeeze=False)[:, 0]
    figure.suptitle(f"{label} | Test forecasts", fontsize=15)
    for axis, index in zip(axes, indices):
        axis.plot(times[index], actual[index], color="#2474B5",
                  linewidth=1.8, marker=".", label="Observed")
        axis.plot(times[index], predictions[index], color="#C07A22",
                  linewidth=1.8, marker=".", linestyle="--", label="Predicted")
        error = float(np.mean(np.abs(predictions[index] - actual[index])))
        axis.set(title=f"Test window {index} | MAE = {error:.4f} °C",
                 ylabel="Temperature (°C)", xlabel="Target time")
        locator = AutoDateLocator(minticks=3, maxticks=6)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        axis.grid(alpha=0.22)
        axis.legend(loc="best")
    return _save_png(figure, path)


def save_loss_plot(path, train_losses, validation_losses, best_epoch, label):
    """Plot the full loss history and a close-up after epoch one; mark the best epoch."""
    train = np.asarray(train_losses, dtype=float)
    validation = np.asarray(validation_losses, dtype=float)
    if (train.ndim != 1 or train.size == 0 or train.shape != validation.shape
            or not np.isfinite(train).all() or not np.isfinite(validation).all()):
        raise ValueError('Training and validation losses must be finite, nonempty one-dimensional arrays of the same length.')
    if not 1 <= best_epoch <= len(train):
        raise ValueError('The best epoch must be within the completed epochs.')
    # Keep all epochs in the left panel and show later epochs in more detail.
    panels = 2 if len(train) > 2 else 1
    figure = Figure(figsize=(12, 4.8), layout="constrained")
    axes = figure.subplots(1, panels, squeeze=False)[0]
    figure.suptitle(f"{label} | Training and validation loss", fontsize=14)
    epochs = np.arange(1, len(train) + 1)
    for index, axis in enumerate(axes):
        start = 0 if index == 0 else 1
        axis.plot(epochs[start:], train[start:], color="#2474B5",
                  marker=".", label="Training")
        axis.plot(epochs[start:], validation[start:], color="#C07A22",
                  marker=".", linestyle="--", label="Validation")
        if best_epoch >= start + 1:
            axis.axvline(best_epoch, color="#666666", linestyle=":",
                         label=f"Best epoch: {best_epoch}")
        axis.set(title="All epochs" if index == 0 else "Detail: epoch 2 onward",
                 xlabel="Epoch", ylabel="Model objective (training scale)")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.22)
        axis.legend(loc="best", fontsize=9)
    return _save_png(figure, path)
