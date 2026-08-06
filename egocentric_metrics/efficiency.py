"""Caller-supplied runtime, parameter, and memory summaries."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _parameter_count(parameters: Iterable[object] | None) -> int:
    if parameters is None:
        return 0
    count = 0
    for parameter in parameters:
        count += int(parameter.numel()) if hasattr(parameter, "numel") else int(np.asarray(parameter).size)
    return count


def efficiency_metrics(*, elapsed_seconds: float, processed_frames: int, parameters: Iterable[object] | None = None, peak_memory_bytes: int | None = None) -> dict[str, float | int]:
    if elapsed_seconds <= 0.0 or processed_frames < 0:
        raise ValueError("elapsed_seconds must be positive and processed_frames must be non-negative")
    return {
        "runtime_seconds": float(elapsed_seconds),
        "runtime_ms": float(elapsed_seconds * 1000.0),
        "fps": float(processed_frames / elapsed_seconds),
        "parameter_count": _parameter_count(parameters),
        "peak_memory_gib": float(peak_memory_bytes / 1024**3) if peak_memory_bytes is not None else float("nan"),
    }
