"""Depth-estimation metrics."""

from __future__ import annotations

import numpy as np

from .common import as_numpy


def depth_metrics(prediction, target, mask=None, *, unit_scale: float = 1.0) -> dict[str, float | int]:
    """Compute standard depth errors over finite pixels."""
    pred = as_numpy(prediction, dtype=float)
    target = as_numpy(target, dtype=float)
    if pred.shape != target.shape:
        raise ValueError(f"prediction and target must have the same shape, got {pred.shape} and {target.shape}")
    finite = np.isfinite(pred) & np.isfinite(target)
    if mask is not None:
        explicit = np.broadcast_to(as_numpy(mask, dtype=bool), pred.shape)
        finite &= explicit
    absolute_error = np.abs(pred - target)
    squared_error = np.square(pred - target)
    valid_count = int(np.count_nonzero(finite))
    result: dict[str, float | int] = {"valid_count": valid_count}
    if valid_count == 0:
        for name in ("mae", "rmse", "abs_rel", "sq_rel", "log_rmse", "delta1", "delta2", "delta3"):
            result[name] = float("nan")
        return result
    result["mae"] = float(np.mean(absolute_error[finite]) * unit_scale)
    result["rmse"] = float(np.sqrt(np.mean(squared_error[finite])) * unit_scale)

    positive = finite & (pred > 0) & (target > 0)
    if not np.any(positive):
        for name in ("abs_rel", "sq_rel", "log_rmse", "delta1", "delta2", "delta3"):
            result[name] = float("nan")
        return result
    relative_error = absolute_error[positive] / target[positive]
    result["abs_rel"] = float(np.mean(relative_error))
    result["sq_rel"] = float(np.mean(squared_error[positive] / target[positive]))
    log_error = np.log(pred[positive]) - np.log(target[positive])
    result["log_rmse"] = float(np.sqrt(np.mean(np.square(log_error))))
    ratio = np.maximum(pred[positive] / target[positive], target[positive] / pred[positive])
    result["delta1"] = float(np.mean(ratio < 1.25))
    result["delta2"] = float(np.mean(ratio < 1.25**2))
    result["delta3"] = float(np.mean(ratio < 1.25**3))
    return result
