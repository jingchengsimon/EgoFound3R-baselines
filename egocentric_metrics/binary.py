"""Metrics for binary visibility and contact predictions."""

from __future__ import annotations

import numpy as np

from .common import as_numpy


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(
    prediction,
    target,
    mask=None,
    *,
    prediction_type: str = "label",
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute confusion-derived metrics for binary labels, scores, or logits."""
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction and target must have the same shape, got {pred.shape} and {gt.shape}")
    valid = np.isfinite(pred) & np.isfinite(gt)
    if mask is not None:
        valid &= np.broadcast_to(as_numpy(mask, dtype=bool), pred.shape)
    if prediction_type == "label":
        pred_label = pred > 0.5
    elif prediction_type == "score":
        pred_label = pred >= threshold
    elif prediction_type == "logit":
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1) for logits")
        pred_label = pred >= np.log(threshold / (1.0 - threshold))
    else:
        raise ValueError("prediction_type must be 'label', 'score', or 'logit'")
    gt_label = gt > 0.5
    pred_label = pred_label[valid]
    gt_label = gt_label[valid]
    tp = int(np.count_nonzero(pred_label & gt_label))
    tn = int(np.count_nonzero(~pred_label & ~gt_label))
    fp = int(np.count_nonzero(pred_label & ~gt_label))
    fn = int(np.count_nonzero(~pred_label & gt_label))
    count = tp + tn + fp + fn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    result: dict[str, float | int] = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "valid_count": count,
        "accuracy": _ratio(tp + tn, count),
        "balanced_accuracy": float(np.nanmean([recall, specificity])) if np.any(np.isfinite([recall, specificity])) else float("nan"),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _ratio(2.0 * tp, 2.0 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
    }
    if prediction_type == "logit":
        logits = pred[valid]
        bce = np.maximum(logits, 0.0) - logits * gt_label + np.log1p(np.exp(-np.abs(logits)))
        result["bce"] = float(np.mean(bce)) if count else float("nan")
    elif prediction_type == "score":
        scores = np.clip(pred[valid], 1e-7, 1.0 - 1e-7)
        labels = gt_label.astype(float)
        result["bce"] = float(np.mean(-labels * np.log(scores) - (1.0 - labels) * np.log1p(-scores))) if count else float("nan")
    else:
        result["bce"] = float("nan")
    if count == 0:
        for name in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "iou"):
            result[name] = float("nan")
    return result
