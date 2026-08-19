"""Protocol-explicit geometry, trajectory, and object-pose metrics."""

from __future__ import annotations

import numpy as np

from .alignment import apply_transform, umeyama
from .common import as_numpy


def _finite_errors(errors) -> np.ndarray:
    value = as_numpy(errors, dtype=float).reshape(-1)
    return value[np.isfinite(value)]


def pck_auc(errors, *, max_threshold: float, num_thresholds: int = 100) -> dict[str, object]:
    """Integrate PCK over uniformly sampled thresholds in the input unit."""
    if max_threshold <= 0.0 or num_thresholds < 2:
        raise ValueError("max_threshold must be positive and num_thresholds must be at least 2")
    valid = _finite_errors(errors)
    thresholds = np.linspace(0.0, max_threshold, num_thresholds)
    if valid.size == 0:
        pck = np.full_like(thresholds, np.nan)
        return {"auc": float("nan"), "pck": pck, "thresholds": thresholds, "valid_count": 0}
    pck = np.array([np.mean(valid <= threshold) for threshold in thresholds], dtype=float)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {
        "auc": float(integrate(pck, thresholds) / max_threshold),
        "pck": pck,
        "thresholds": thresholds,
        "valid_count": int(valid.size),
    }


def point_pck_auc(prediction, target, mask=None, *, max_threshold: float, num_thresholds: int = 100, unit_scale: float = 1.0) -> dict[str, object]:
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape or pred.shape[-1:] != (3,):
        raise ValueError("point arrays must have matching shape (..., 3)")
    errors = np.linalg.norm(pred - gt, axis=-1) * unit_scale
    valid = np.isfinite(errors)
    if mask is not None:
        valid &= np.broadcast_to(as_numpy(mask, dtype=bool), errors.shape)
    return pck_auc(errors[valid], max_threshold=max_threshold, num_thresholds=num_thresholds)


def ate(prediction, target, *, alignment: str = "none") -> float:
    """Absolute trajectory RMSE, optionally after one rigid or Sim(3) alignment."""
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError("trajectories must have matching shape (T, 3)")
    valid = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    if not np.any(valid):
        return float("nan")
    if alignment == "none":
        aligned = pred
    elif alignment in {"rigid", "similarity"}:
        transform = umeyama(pred[valid][None], gt[valid][None], fix_scale=alignment == "rigid")
        aligned = apply_transform(pred[None], transform)[0]
    else:
        raise ValueError("alignment must be 'none', 'rigid', or 'similarity'")
    return float(np.sqrt(np.mean(np.sum(np.square(aligned[valid] - gt[valid]), axis=-1))))


def relative_rotation_angle(prediction_rotation, target_rotation) -> np.ndarray:
    pred = as_numpy(prediction_rotation, dtype=float)
    gt = as_numpy(target_rotation, dtype=float)
    if pred.shape[-2:] != (3, 3) or gt.shape[-2:] != (3, 3):
        raise ValueError("rotation arrays must end in shape (3, 3)")
    try:
        pred, gt = np.broadcast_arrays(pred, gt)
    except ValueError as exc:
        raise ValueError("rotation arrays must have broadcast-compatible leading dimensions") from exc
    relative = np.matmul(pred, np.swapaxes(gt, -1, -2))
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def relative_translation_angle(prediction_translation, target_translation) -> np.ndarray:
    pred = as_numpy(prediction_translation, dtype=float)
    gt = as_numpy(target_translation, dtype=float)
    if pred.shape != gt.shape or pred.shape[-1:] != (3,):
        raise ValueError("translation arrays must have matching shape (..., 3)")
    denom = np.linalg.norm(pred, axis=-1) * np.linalg.norm(gt, axis=-1)
    dot = np.sum(pred * gt, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = dot / denom
    return np.where(denom > 0.0, np.arccos(np.clip(cosine, -1.0, 1.0)), np.nan)


def rpe(prediction_poses, target_poses, *, delta: int = 1) -> dict[str, float | int]:
    """Mean adjacent-relative-pose translation and rotation error."""
    pred = as_numpy(prediction_poses, dtype=float)
    gt = as_numpy(target_poses, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-2:] != (4, 4):
        raise ValueError("pose arrays must have matching shape (T, 4, 4)")
    if delta < 1 or pred.shape[0] <= delta:
        raise ValueError("delta must be positive and smaller than sequence length")
    pred_rel = np.matmul(np.linalg.inv(pred[:-delta]), pred[delta:])
    gt_rel = np.matmul(np.linalg.inv(gt[:-delta]), gt[delta:])
    error = np.matmul(np.linalg.inv(gt_rel), pred_rel)
    translation = np.linalg.norm(error[:, :3, 3], axis=-1)
    rotation = relative_rotation_angle(error[:, :3, :3], np.eye(3))
    return {
        "translation_l2": float(np.mean(translation)),
        "rotation_rad": float(np.mean(rotation)),
        "valid_count": int(translation.size),
    }


def rra(rotation_errors_rad, *, threshold_deg: float = 30.0) -> float:
    errors = _finite_errors(rotation_errors_rad)
    return float(np.mean(errors <= np.deg2rad(threshold_deg))) if errors.size else float("nan")


def rta(translation_angle_errors_rad, *, threshold_deg: float = 30.0) -> float:
    errors = _finite_errors(translation_angle_errors_rad)
    return float(np.mean(errors <= np.deg2rad(threshold_deg))) if errors.size else float("nan")


def pose_auc(rotation_errors_rad, translation_errors_rad, *, max_threshold_deg: float = 30.0, num_thresholds: int = 100) -> dict[str, object]:
    rotation = as_numpy(rotation_errors_rad, dtype=float).reshape(-1)
    translation = as_numpy(translation_errors_rad, dtype=float).reshape(-1)
    if rotation.shape != translation.shape:
        raise ValueError("rotation and translation error arrays must have the same shape")
    combined = np.maximum(rotation, translation) * (180.0 / np.pi)
    return pck_auc(combined, max_threshold=max_threshold_deg, num_thresholds=num_thresholds)


def _transform_object_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    if pose.shape != (4, 4):
        raise ValueError("object poses must have shape (4, 4)")
    return np.einsum("ij,nj->ni", pose[:3, :3], points) + pose[:3, 3]


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.empty(0, dtype=float)
    squared = np.sum(np.square(source[:, None] - target[None]), axis=-1)
    return np.sqrt(np.min(squared, axis=1))


def add_metrics(model_points, prediction_pose, target_pose, *, object_diameter: float) -> dict[str, float]:
    """Compute ADD, symmetric ADD-S, and ADD-0.1D success for an object pose."""
    points = as_numpy(model_points, dtype=float)
    if points.ndim != 2 or points.shape[-1] != 3 or object_diameter <= 0.0:
        raise ValueError("model_points must have shape (N, 3) and object_diameter must be positive")
    pred = _transform_object_points(points, as_numpy(prediction_pose, dtype=float))
    gt = _transform_object_points(points, as_numpy(target_pose, dtype=float))
    add = float(np.mean(np.linalg.norm(pred - gt, axis=-1)))
    adds = float(np.mean(_nearest_distances(pred, gt)))
    return {"add": add, "adds": adds, "add_0_1d": float(add < 0.1 * object_diameter)}


def contact_coverage(hand_points, object_points, *, threshold: float) -> float:
    """Fraction of hand points within threshold of the object surface samples."""
    hand = as_numpy(hand_points, dtype=float)
    obj = as_numpy(object_points, dtype=float)
    if hand.ndim != 2 or obj.ndim != 2 or hand.shape[-1:] != (3,) or obj.shape[-1:] != (3,) or threshold < 0.0:
        raise ValueError("point arrays must have shape (N, 3) and threshold must be non-negative")
    valid_hand = hand[np.isfinite(hand).all(axis=-1)]
    valid_object = obj[np.isfinite(obj).all(axis=-1)]
    distances = _nearest_distances(valid_hand, valid_object)
    return float(np.mean(distances <= threshold)) if distances.size else float("nan")
