"""Point-tracking metrics following the TAP-Vid visibility convention."""

from __future__ import annotations

import numpy as np

from .common import as_numpy


def tracking_metrics(prediction_xy, target_xy, prediction_visible, target_visible, *, thresholds: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)) -> dict[str, float | int]:
    pred = as_numpy(prediction_xy, dtype=float)
    gt = as_numpy(target_xy, dtype=float)
    pred_visible = as_numpy(prediction_visible, dtype=bool)
    gt_visible = as_numpy(target_visible, dtype=bool)
    if pred.shape != gt.shape or pred.shape[-1:] != (2,) or pred_visible.shape != pred.shape[:-1] or gt_visible.shape != pred.shape[:-1]:
        raise ValueError("coordinates must match (..., 2), with matching visibility arrays")
    if not thresholds or any(threshold <= 0.0 for threshold in thresholds):
        raise ValueError("thresholds must be non-empty and positive")
    errors = np.linalg.norm(pred - gt, axis=-1)
    finite = np.isfinite(errors)
    visible_gt = gt_visible & finite
    visible_count = int(np.count_nonzero(visible_gt))
    deltas: list[float] = []
    jaccards: list[float] = []
    for threshold in thresholds:
        correct = errors <= threshold
        deltas.append(float(np.mean(correct[visible_gt])) if visible_count else float("nan"))
        intersection = np.count_nonzero(pred_visible & visible_gt & correct)
        union = np.count_nonzero(pred_visible | gt_visible)
        jaccards.append(float(intersection / union) if union else float("nan"))
    occlusion = pred_visible == gt_visible
    return {
        "average_jaccard": float(np.nanmean(jaccards)) if np.any(np.isfinite(jaccards)) else float("nan"),
        "delta_avg_visible": float(np.nanmean(deltas)) if np.any(np.isfinite(deltas)) else float("nan"),
        "occlusion_accuracy": float(np.mean(occlusion)),
        "visible_count": visible_count,
    }
