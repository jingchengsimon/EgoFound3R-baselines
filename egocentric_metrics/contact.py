"""Continuous contact-distance metrics."""

from __future__ import annotations

import numpy as np

from .common import as_numpy


def contact_distance_metrics(prediction, target, mask=None, *, unit_scale: float = 1000.0) -> dict[str, float | int]:
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction and target must have the same shape, got {pred.shape} and {gt.shape}")
    valid = np.isfinite(pred) & np.isfinite(gt)
    if mask is not None:
        valid &= np.broadcast_to(as_numpy(mask, dtype=bool), pred.shape)
    error = np.abs(pred[valid] - gt[valid]) * unit_scale
    if error.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "median_absolute_error": float("nan"), "valid_count": 0}
    return {
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_absolute_error": float(np.median(error)),
        "valid_count": int(error.size),
    }
