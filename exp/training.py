"""Shared PyTorch training and prediction loops for neural estimators."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
import sys
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from exp.progress import print_progress
from util.live_plot import pump_plot_events, report_training
from util.config import TrainingConfig


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TorchTrainingOutcome:
    """Store the best model and per-epoch losses."""

    # Keep training statistics immutable.
    model: nn.Module
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    completed_epochs: int
    elapsed_seconds: float
    learning_rates: tuple[float, ...] = ()
    loss_components: tuple[dict[str, float], ...] = ()


def resolve_device(requested: str) -> torch.device:
    """Resolve auto, CPU, or CUDA and reject an explicit request for unavailable CUDA."""
    # Normalize case and strip surrounding whitespace.
    normalized = requested.strip().lower()
    if normalized == "auto":
        # Prefer CUDA in auto mode and fall back to CPU when needed.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        # Do not silently fall back if CUDA was requested explicitly.
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested explicitly but is unavailable.')
        try:
            return torch.device(normalized)
        except RuntimeError as error:
            raise ValueError(f"Invalid CUDA device: {requested}.") from error
    raise ValueError(f"Device must be auto, cpu, or cuda; got {requested}.")


def initialize_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA."""
    # Reject booleans explicitly because bool subclasses int.
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')

    # Seed each random number generator for reproducible runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        # Trade some speed for more reproducible cuDNN behavior.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validated_arrays(
    inputs: np.ndarray,
    targets: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate array shapes, sample counts, and finite values."""
    # Convert array-like inputs to NumPy for validation.
    input_array = np.asarray(inputs)
    target_array = np.asarray(targets)

    # Inputs have shape (samples, sequence length, features).
    if input_array.ndim != 3 or input_array.shape[1] < 1:
        raise ValueError(f"{label} inputs must be a nonempty three-dimensional array.")
    # Targets have shape (samples, forecast horizon).
    if target_array.ndim != 2 or target_array.shape[1] < 1:
        raise ValueError(f"{label} targets must be a nonempty two-dimensional array.")
    # Inputs and targets must contain the same nonzero number of samples.
    if input_array.shape[0] != target_array.shape[0] or input_array.shape[0] < 1:
        raise ValueError(f"{label} inputs and targets must have the same nonzero sample count.")
    if np.iscomplexobj(input_array) or np.iscomplexobj(target_array):
        raise ValueError(f"{label} inputs and targets must be real-valued.")
    # Reject NaN and infinity before training.
    if not np.isfinite(input_array).all() or not np.isfinite(target_array).all():
        raise ValueError(f"{label} inputs and targets must contain only finite values.")

    # Use float32 for PyTorch without copying if the dtype already matches.
    return (
        input_array.astype(np.float32, copy=False),
        target_array.astype(np.float32, copy=False),
    )


def _batch_objective(model, inputs, targets, loss_function, batch_loss_function):
    """Pass the current batch explicitly so physical losses do not depend on cached forward state."""
    if batch_loss_function is None:
        predictions = model(inputs)
        terms = None
    else:
        predictions, terms = batch_loss_function(model, inputs, targets)
    if predictions.shape != targets.shape:
        raise ValueError('Model output shape must match the target shape.')
    if terms is None:
        loss = loss_function(predictions, targets)
        terms = {"total": loss}
    else:
        loss = terms["total"]
    for name, value in terms.items():
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError(f"Loss component {name} must be a finite scalar.")
    return loss, terms


def _accumulate_components(totals, terms, count):
    """Accumulate sample-weighted loss components as floats, without retaining the computation graph."""
    for key, value in terms.items():
        totals[key] = totals.get(key, 0.0) + float(value.detach()) * count


def _mean_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: LossFunction,
    device: torch.device,
    batch_loss_function=None,
    component_totals=None,
    selection_component="total",
) -> float:
    """Compute a sample-weighted mean loss over the full dataset."""
    total = 0.0
    samples = 0
    # Evaluate without gradients to save memory.
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            targets = batch_targets.to(device)
            loss, terms = _batch_objective(
                model, batch_inputs.to(device), targets,
                loss_function, batch_loss_function,
            )
            if component_totals is not None:
                _accumulate_components(component_totals, terms, len(batch_inputs))
            # Use a separate selection metric while retaining every validation loss component.
            loss = terms[selection_component]
            # Weight batches by their actual size, including a smaller final batch.
            total += float(loss) * len(batch_inputs)
            samples += len(batch_inputs)
            pump_plot_events()
    return total / samples


def fit_torch_model(
    model: nn.Module,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    config: TrainingConfig,
    seed: int,
    constraint_callback: Callable[[], None] | None = None,
    loss_function: LossFunction | None = None,
    batch_loss_function: Callable | None = None,
    validation_component: str = "total",
) -> TorchTrainingOutcome:
    """Train with early stopping and restore the weights with the lowest validation loss."""
    # Validate both splits separately so errors identify the affected dataset.
    train_x, train_y = _validated_arrays(train_inputs, train_targets, 'Training')
    validation_x, validation_y = _validated_arrays(
        validation_inputs,
        validation_targets,
        'Validation',
    )
    if train_y.shape[1] != validation_y.shape[1]:
        raise ValueError('Training and validation target horizons must match.')

    # Seed the run before building data loaders or starting optimization.
    initialize_seed(seed)
    device = resolve_device(config.device)  # Resolve the requested device.
    model.to(device)

    # Use a dedicated generator for reproducible training shuffle order.
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(  # Build the training data loader.
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    # Keep validation samples in their original order.
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_x),
            torch.from_numpy(validation_y),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Use Adam for all models; allow a model-specific loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Decay the learning rate to 1% of its initial value by the final epoch.
    # A single epoch uses the initial rate without a zero-length schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs - 1),
        eta_min=config.learning_rate * 0.01,
    )
    active_loss: LossFunction = loss_function or nn.MSELoss()

    # Track per-epoch losses for analysis and plotting.
    train_losses: list[float] = []
    validation_losses: list[float] = []
    learning_rates: list[float] = []
    component_history: list[dict[str, float]] = []

    # Keep the best validation weights, not the final epoch's weights.
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    # Count consecutive epochs without improvement for early stopping.
    non_improvements = 0
    started = time.perf_counter()
    print_progress(
        'Training',
        (
            f"Starting: device={device}, training samples={len(train_x)}, "
            f"validation samples={len(validation_x)}, batch size={config.batch_size}, "
            f"max epochs={config.epochs}"
        ),
    )

    # Show per-epoch training progress with tqdm.
    epoch_progress = tqdm(
        total=config.epochs,
        desc='Training progress',
        unit='epoch',
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
    )
    for epoch in range(config.epochs):
        # Let models warm up regularization when they provide an epoch hook.
        epoch_callback = getattr(model, "set_training_epoch", None)
        if callable(epoch_callback):
            epoch_callback(epoch + 1)
        # Record the learning rate actually used for this epoch.
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)
        # Enable dropout and BatchNorm updates during training.
        model.train()
        total = 0.0
        samples = 0
        train_components = {}
        for batch_inputs, batch_targets in train_loader:
            # Move the batch to the same device as the model.
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Clear the previous batch's gradients before the next optimization step.
            optimizer.zero_grad()
            loss, terms = _batch_objective(
                model, batch_inputs, batch_targets, active_loss, batch_loss_function,
            )
            _accumulate_components(train_components, terms, len(batch_inputs))
            loss.backward()

            # Optionally clip gradients to limit exploding updates.
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            pump_plot_events()

            # Apply any model-specific constraints after the optimizer step.
            if constraint_callback is not None:
                constraint_callback()

            # Detach the loss before logging it to release the graph.
            total += float(loss.detach()) * len(batch_inputs)
            samples += len(batch_inputs)
        train_losses.append(total / samples)

        # Disable dropout and BatchNorm updates during validation.
        model.eval()
        validation_components = {}
        validation_loss = _mean_loss(
            model,
            validation_loader,
            active_loss,
            device,
            batch_loss_function,
            validation_components,
            validation_component,
        )
        validation_losses.append(validation_loss)
        if batch_loss_function is not None:
            component_history.append({
                "epoch": epoch + 1,
                **{f"train_{key}": value / samples
                   for key, value in train_components.items()},
                **{f"validation_{key}": value / len(validation_x)
                   for key, value in validation_components.items()},
            })

        should_stop = False
        # Copy the weights and reset patience when validation loss improves.
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            non_improvements = 0
        else:
            non_improvements += 1
            # Stop once validation loss has failed to improve for the patience period.
            if non_improvements >= config.patience:
                should_stop = True

        completed_epoch = epoch + 1
        report_training(train_losses, validation_losses, best_epoch)
        # Update progress after both current-epoch losses are available.
        epoch_progress.update(1)
        epoch_progress.set_postfix(
            train_loss=f"{train_losses[-1]:.6f}",
            val_loss=f"{validation_loss:.6f}",
            best_epoch=best_epoch,
            lr=f"{current_lr:.3e}",
            refresh=True,
        )
        should_report = (
            completed_epoch % 5 == 0
            or completed_epoch == config.epochs
            or should_stop
        )
        if should_report:
            print_progress(
                'Training',
                (
                    f"Epoch {completed_epoch}/{config.epochs}: "
                    f"training loss={train_losses[-1]:.6f}, "
                    f"validation loss={validation_loss:.6f}, "
                    f"best epoch={best_epoch}, learning rate={current_lr:.3e}"
                ),
            )
        if should_stop:
            print_progress(
                'Training',
                (
                    f"Early stopping: validation loss did not improve for {config.patience} epochs; "
                    f"restoring weights from epoch {best_epoch}"
                ),
            )
            break
        # Skip the scheduler update if there will be no next epoch.
        if epoch + 1 < config.epochs:
            scheduler.step()
    epoch_progress.close()

    if best_state is None:
        raise RuntimeError('Training produced no best weights to restore.')

    # Restore the best validation weights regardless of how training ended.
    model.load_state_dict(best_state)
    report_training(train_losses, validation_losses, best_epoch, final=True)
    elapsed = time.perf_counter() - started
    print_progress(
        'Training',
        (
            f"Finished: trained for {len(train_losses)} epochs; best epoch={best_epoch}, "
            f"best validation loss={best_loss:.6f}, elapsed={elapsed:.2f} seconds"
        ),
    )
    return TorchTrainingOutcome(
        model=model,
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        best_epoch=best_epoch,
        completed_epochs=len(train_losses),
        elapsed_seconds=elapsed,
        learning_rates=tuple(learning_rates),
        loss_components=tuple(component_history),
    )


def predict_torch_model(
    model: nn.Module,
    inputs: np.ndarray,
    batch_size: int,
    device_name: str,
) -> np.ndarray:
    """Predict in batches in input order and restore the caller's model mode."""
    # Validate the same three-dimensional input shape used in training.
    array = np.asarray(inputs)
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError('Prediction inputs must be a nonempty three-dimensional array.')
    if np.iscomplexobj(array) or not np.isfinite(array).all():
        raise ValueError('Prediction inputs must contain only finite real values.')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')

    device = resolve_device(device_name)
    model.to(device)
    # Save the caller's model mode for restoration after prediction.
    was_training = model.training
    predictions: list[np.ndarray] = []
    try:
        model.eval()
        # Disable autograd during prediction to save memory.
        with torch.no_grad():
            for start in range(0, len(array), batch_size):
                # Convert and move one float32 batch at a time to limit device memory usage.
                batch = torch.from_numpy(
                    array[start:start + batch_size].astype(np.float32, copy=False)
                ).to(device)
                output = model(batch)
                if output.ndim != 2:
                    raise ValueError('Model predictions must be two-dimensional.')
                # Move each batch to CPU and concatenate results in input order.
                predictions.append(output.detach().cpu().numpy())
    finally:
        # Restore the original model mode even if prediction fails.
        model.train(was_training)
    return np.concatenate(predictions, axis=0)
