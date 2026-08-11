"""Camera intrinsic and extrinsic metrics."""

from __future__ import annotations

import numpy as np

from .common import as_numpy


def _valid_batch(prediction: np.ndarray, target: np.ndarray, mask, trailing_dims: int) -> np.ndarray:
    valid = np.isfinite(prediction).all(axis=tuple(range(prediction.ndim - trailing_dims, prediction.ndim)))
    valid &= np.isfinite(target).all(axis=tuple(range(target.ndim - trailing_dims, target.ndim)))
    if mask is not None:
        valid &= np.broadcast_to(as_numpy(mask, dtype=bool), valid.shape)
    return valid


def intrinsic_metrics(prediction, target, mask=None, *, unit_scale: float = 1.0) -> dict[str, float | int]:
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape or pred.ndim < 2 or pred.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must have matching (..., 3, 3) shapes, got {pred.shape} and {gt.shape}")
    valid = _valid_batch(pred, gt, mask, 2)
    if not np.any(valid):
        return {name: float("nan") for name in ("l1", "rmse", "fx_abs", "fy_abs", "cx_abs", "cy_abs")} | {"valid_count": 0}
    diff = pred - gt
    selected = diff[valid]
    return {
        "l1": float(np.mean(np.abs(selected)) * unit_scale),
        "rmse": float(np.sqrt(np.mean(np.square(selected))) * unit_scale),
        "fx_abs": float(np.mean(np.abs(diff[..., 0, 0][valid])) * unit_scale),
        "fy_abs": float(np.mean(np.abs(diff[..., 1, 1][valid])) * unit_scale),
        "cx_abs": float(np.mean(np.abs(diff[..., 0, 2][valid])) * unit_scale),
        "cy_abs": float(np.mean(np.abs(diff[..., 1, 2][valid])) * unit_scale),
        "valid_count": int(np.count_nonzero(valid)),
    }


def _rotation_angle(pred_rotation: np.ndarray, gt_rotation: np.ndarray) -> np.ndarray:
    relative = np.matmul(pred_rotation, np.swapaxes(gt_rotation, -1, -2))
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def extrinsic_metrics(prediction, target, mask=None, *, translation_unit_scale: float = 1.0) -> dict[str, float | int]:
    """Evaluate homogeneous camera poses ``(..., 4, 4)``."""
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape or pred.ndim < 2 or pred.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsics must have matching (..., 4, 4) shapes, got {pred.shape} and {gt.shape}")
    valid = _valid_batch(pred, gt, mask, 2)
    if not np.any(valid):
        return {name: float("nan") for name in ("translation_l2", "rotation_geodesic_rad", "rotation_fro")} | {"valid_count": 0}
    translation_diff = pred[..., :3, 3] - gt[..., :3, 3]
    rotation_diff = pred[..., :3, :3] - gt[..., :3, :3]
    angles = _rotation_angle(pred[..., :3, :3], gt[..., :3, :3])
    return {
        "translation_l2": float(np.mean(np.linalg.norm(translation_diff[valid], axis=-1)) * translation_unit_scale),
        "rotation_geodesic_rad": float(np.mean(angles[valid])),
        "rotation_fro": float(np.mean(np.linalg.norm(rotation_diff[valid], axis=(-2, -1)))),
        "valid_count": int(np.count_nonzero(valid)),
    }


def relative_pose_metrics(prediction, target, mask=None, *, translation_unit_scale: float = 1.0) -> dict[str, float | int]:
    """Compare adjacent relative camera motions for a ``(T, 4, 4)`` sequence."""
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-2:] != (4, 4) or pred.shape[0] < 2:
        raise ValueError("relative poses require matching (T, 4, 4) arrays with T >= 2")
    pred_rel = np.matmul(np.linalg.inv(pred[:-1]), pred[1:])
    gt_rel = np.matmul(np.linalg.inv(gt[:-1]), gt[1:])
    return extrinsic_metrics(pred_rel, gt_rel, mask=mask, translation_unit_scale=translation_unit_scale)
