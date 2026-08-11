"""Point-cloud geometry metrics."""

from __future__ import annotations

import numpy as np

from .common import as_numpy


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("point clouds must have shape (N, 3)")
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.empty(0), np.empty(0, dtype=int)
    squared = np.sum(np.square(source[:, None, :] - target[None, :, :]), axis=-1)
    indices = np.argmin(squared, axis=1)
    return np.sqrt(np.min(squared, axis=1)), indices


def pointcloud_metrics(
    prediction,
    target,
    *,
    prediction_normals=None,
    target_normals=None,
    thresholds: tuple[float, ...] = (0.005, 0.01, 0.02),
) -> dict[str, float | int]:
    """Compute symmetric nearest-neighbour point-cloud errors."""
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1:] != (3,) or gt.shape[1:] != (3,):
        raise ValueError("prediction and target clouds must have shape (N, 3)")
    pred = pred[np.isfinite(pred).all(axis=1)]
    gt = gt[np.isfinite(gt).all(axis=1)]
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        result: dict[str, float | int] = {
            "chamfer_l1": float("nan"),
            "chamfer_l2": float("nan"),
            "p2point": float("nan"),
            "pred_to_target": float("nan"),
            "target_to_pred": float("nan"),
            "valid_prediction_count": int(pred.shape[0]),
            "valid_target_count": int(gt.shape[0]),
        }
        result.update({f"fscore@{threshold:g}": float("nan") for threshold in thresholds})
        if prediction_normals is not None and target_normals is not None:
            result["normal_consistency"] = float("nan")
        return result

    pred_distances, pred_indices = _nearest_distances(pred, gt)
    gt_distances, gt_indices = _nearest_distances(gt, pred)
    result = {
        "chamfer_l1": float((pred_distances.mean() + gt_distances.mean()) / 2.0),
        "chamfer_l2": float((np.square(pred_distances).mean() + np.square(gt_distances).mean()) / 2.0),
        "p2point": float(pred_distances.mean()),
        "pred_to_target": float(pred_distances.mean()),
        "target_to_pred": float(gt_distances.mean()),
        "valid_prediction_count": int(pred.shape[0]),
        "valid_target_count": int(gt.shape[0]),
    }
    for threshold in thresholds:
        precision = float(np.mean(pred_distances <= threshold))
        recall = float(np.mean(gt_distances <= threshold))
        denominator = precision + recall
        result[f"fscore@{threshold:g}"] = 2.0 * precision * recall / denominator if denominator else float("nan")

    if prediction_normals is not None or target_normals is not None:
        if prediction_normals is None or target_normals is None:
            raise ValueError("prediction_normals and target_normals must be supplied together")
        pred_normals = as_numpy(prediction_normals, dtype=float)
        gt_normals = as_numpy(target_normals, dtype=float)
        if pred_normals.shape != (prediction.shape[0], 3) or gt_normals.shape != (target.shape[0], 3):
            raise ValueError("normal arrays must match the original cloud shapes")
        pred_normals = pred_normals[np.isfinite(as_numpy(prediction, dtype=float)).all(axis=1)]
        gt_normals = gt_normals[np.isfinite(as_numpy(target, dtype=float)).all(axis=1)]
        pred_normals /= np.linalg.norm(pred_normals, axis=1, keepdims=True).clip(min=1e-12)
        gt_normals /= np.linalg.norm(gt_normals, axis=1, keepdims=True).clip(min=1e-12)
        pred_consistency = np.abs(np.sum(pred_normals * gt_normals[pred_indices], axis=1))
        gt_consistency = np.abs(np.sum(gt_normals * pred_normals[gt_indices], axis=1))
        result["normal_consistency"] = float((pred_consistency.mean() + gt_consistency.mean()) / 2.0)
    return result
