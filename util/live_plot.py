"""Display training loss and Celsius forecasts in a local window."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import time

import numpy as np

from exp.progress import print_progress
from util.result_store import save_history


# Scope each observer to one training run.
_observer = ContextVar("training_plot_observer", default=None)
_figures = []


@contextmanager
def observe_training(observer):
    """Register an observer for one run and restore the previous observer on exit."""
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


def report_training(train_losses, validation_losses, best_epoch, final=False):
    """Send training statistics to the current observer."""
    observer = _observer.get()
    if observer is not None:
        observer.update(train_losses, validation_losses, best_epoch, final)


def pump_plot_events():
    """Process window events between batches to keep plots responsive."""
    observer = _observer.get()
    if observer is not None:
        observer.pump()


def wait_for_training_plots():
    """Keep result windows open after training until the user closes them."""
    if _figures:
        import matplotlib.pyplot as plt

        if any(plt.fignum_exists(figure.number) for figure in _figures):
            plt.show(block=True)
        _figures.clear()


def preview_windows(windows):
    """Select first, middle, and last windows without merging overlapping forecasts."""
    indices = np.unique(np.linspace(0, len(windows) - 1, min(3, len(windows))).astype(int))
    return replace(
        windows,
        raw_inputs=windows.raw_inputs[indices],
        scaled_inputs=windows.scaled_inputs[indices],
        targets_c=windows.targets_c[indices],
        target_times=windows.target_times[indices],
        spans=tuple(windows.spans[i] for i in indices),
    ), indices


class LiveTrainingPlot:
    """Reuse one window to display losses and example forecasts."""

    def __init__(self, estimator, validation, context, config, label,
                 history_path=None):
        self.estimator = estimator
        self.context = context
        self.config = config
        self.label = label
        self.history_path = history_path
        self.figure = None
        self.last_pump = 0.0
        if config.plot_every == 0:
            return
        import matplotlib.pyplot as plt

        self.plt = plt
        # Continue training without plots if no graphical backend is available.
        try:
            self.figure = plt.figure(figsize=(13, 8), layout="constrained")
            if self.figure.canvas.required_interactive_framework is None:
                print_progress('Plotting', 'The current backend does not support interactive windows; skipping live plots.')
                plt.close(self.figure)
                self.figure = None
                return
        except (ImportError, RuntimeError) as error:
            print_progress('Plotting', f"Could not open a plot window; skipping live plots: {error}")
            return
        self.preview, _ = preview_windows(validation)
        grid = self.figure.add_gridspec(2, 3)
        self.loss_axis = self.figure.add_subplot(grid[0, :])
        self.prediction_axes = [self.figure.add_subplot(grid[1, i]) for i in range(3)]
        self.figure.canvas.manager.set_window_title(f"Training monitor | {label}")
        _figures.append(self.figure)
        plt.show(block=False)
        self.pump()

    def active(self):
        """Check whether the user has left the plot window open."""
        return self.figure is not None and self.plt.fignum_exists(self.figure.number)

    def pump(self):
        """Throttle window events to avoid overhead on every batch."""
        if self.active() and time.monotonic() - self.last_pump >= 0.2:
            self.figure.canvas.flush_events()
            self.last_pump = time.monotonic()

    def draw_predictions(self, windows, predictions, split):
        """Plot a separate Celsius forecast against real timestamps for each origin."""
        from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

        for i, axis in enumerate(self.prediction_axes):
            axis.clear()
            axis.set_visible(i < len(windows))
            if i >= len(windows):
                continue
            axis.plot(windows.target_times[i], windows.targets_c[i],
                      label="Observed", color="#333333", marker=".")
            axis.plot(windows.target_times[i], predictions[i],
                      label="Predicted", color="#2474B5", linestyle="--", marker=".")
            axis.set(title=f"{split} forecast {i + 1}", ylabel="Temperature (°C)")
            locator = AutoDateLocator(minticks=3, maxticks=4)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
            axis.tick_params(axis="x", labelrotation=25)
            axis.grid(alpha=0.2)
            axis.legend()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def update(self, train_losses, validation_losses, best_epoch, final=False):
        """Refresh validation plots at the first epoch, configured intervals, and after restoring best weights."""
        if self.history_path is not None:
            save_history(self.history_path, train_losses, validation_losses, best_epoch)
        from matplotlib.ticker import MaxNLocator

        epoch = len(train_losses)
        if not self.active() or not (
            final or epoch == 1 or epoch % self.config.plot_every == 0
        ):
            return
        axis = self.loss_axis
        axis.clear()
        epochs = np.arange(1, epoch + 1)
        axis.plot(epochs, train_losses, label="Training", color="#2474B5")
        axis.plot(epochs, validation_losses, label="Validation",
                  color="#C07A22", linestyle="--")
        axis.axvline(best_epoch, color="#555555", linestyle=":", label="Best epoch")
        axis.set(xlabel="Epoch", ylabel="Model objective (training scale)",
                 title="Training and validation loss")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        axis.legend()
        state = f"best weights: epoch {best_epoch}" if final else f"current weights: epoch {epoch}"
        self.figure.suptitle(f"{self.label} | {state}")
        # Use the estimator interface for consistent Celsius output across models.
        predictions = self.estimator.predict(self.preview, self.context, self.config)
        self.draw_predictions(self.preview, predictions, "Validation")

    def finish(self, windows, predictions, fit_result):
        """Show best-weight test predictions only after training."""
        if not self.active():
            return
        self.update(fit_result.train_losses, fit_result.validation_losses,
                    fit_result.best_epoch, final=True)
        sample, indices = preview_windows(windows)
        self.figure.suptitle(f"{self.label} | Finished | best epoch {fit_result.best_epoch}")
        self.draw_predictions(sample, predictions[indices], "Test")
