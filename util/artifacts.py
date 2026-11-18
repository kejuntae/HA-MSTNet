"""Atomic writes for checkpoints and standard JSON results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
import torch


def _json_ready(value: object) -> object:
    """Convert NumPy values recursively and reject nonfinite numbers."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Result values must be finite.')
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    raise ValueError(f"Results contain an unserializable type: {type(value).__name__}.")


def save_results(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Write UTF-8 JSON with an atomic replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ready = _json_ready(payload)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
        suffix=".tmp",
    ) as stream:
        json.dump(ready, stream, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = Path(stream.name)
    temporary.replace(target)
    return target


def save_table(
    path: str | Path,
    rows: list[Mapping[str, object]],
) -> Path:
    """Write UTF-8 CSV with all observed fields and encode nested values as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    field_names: list[str] = []
    for row in rows:
        for key in row:
            name = str(key)
            if name not in field_names:
                field_names.append(name)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8-sig",
        newline="",
        dir=target.parent,
        delete=False,
        suffix=".tmp",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            encoded: dict[str, object] = {}
            for key, value in row.items():
                ready = _json_ready(value)
                encoded[str(key)] = (
                    json.dumps(ready, ensure_ascii=False, allow_nan=False)
                    if isinstance(ready, (dict, list))
                    else ready
                )
            writer.writerow(encoded)
        temporary = Path(stream.name)
    temporary.replace(target)
    return target


def save_checkpoint(
    path: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object],
) -> Path:
    """Copy tensors to CPU and save the checkpoint atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
    }
    ready_metadata = _json_ready(metadata)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=target.parent,
        delete=False,
        suffix=".tmp",
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(
            {"state_dict": cpu_state, "metadata": ready_metadata},
            temporary,
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
