"""Temporal motion and contact-sliding metrics."""

from __future__ import annotations

import numpy as np

from .alignment import apply_transform, umeyama
from .common import as_numpy


def _joint_sequence(joints) -> np.ndarray:
    value = as_numpy(joints, dtype=float)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"joints must have shape (T, J, 3), got {value.shape}")
    return value


def acceleration(joints, *, fps: float = 30.0, unit_scale: float = 1.0) -> np.ndarray:
    """Return joint-averaged acceleration norm for each second-difference step."""
    value = _joint_sequence(joints)
    if value.shape[0] < 3:
        return np.empty(0, dtype=float)
    difference = (value[2:] - 2.0 * value[1:-1] + value[:-2]) * (fps**2)
    return np.linalg.norm(difference, axis=-1).mean(axis=-1) * unit_scale


def _matching_sequences(prediction, target, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    pred = _joint_sequence(prediction)
    gt = _joint_sequence(target)
    if pred.shape != gt.shape:
        raise ValueError(f"{name} inputs must have matching shape (T, N, 3)")
    return pred, gt


def frame_difference_error(prediction, target, *, unit_scale: float = 1.0) -> float:
    """Mean error between predicted and GT frame-to-frame 3D displacements."""
    pred, gt = _matching_sequences(prediction, target, name="frame difference")
    if pred.shape[0] < 2:
        return float("nan")
    error = np.linalg.norm(np.diff(pred, axis=0) - np.diff(gt, axis=0), axis=-1)
    return float(np.mean(error) * unit_scale)


def mpfje(prediction_joints, target_joints, *, unit_scale: float = 1.0) -> float:
    """Mean per-frame joint displacement error."""
    return frame_difference_error(prediction_joints, target_joints, unit_scale=unit_scale)


def mpfve(prediction_vertices, target_vertices, *, unit_scale: float = 1.0) -> float:
    """Mean per-frame vertex displacement error."""
    return frame_difference_error(prediction_vertices, target_vertices, unit_scale=unit_scale)


def acceleration_error(prediction, target, *, fps: float = 30.0, unit_scale: float = 1.0) -> float:
    """Mean norm of predicted-minus-GT second finite differences."""
    pred, gt = _matching_sequences(prediction, target, name="acceleration")
    if pred.shape[0] < 3:
        return float("nan")
    difference = (np.diff(pred, n=2, axis=0) - np.diff(gt, n=2, axis=0)) * (fps**2)
    return float(np.mean(np.linalg.norm(difference, axis=-1)) * unit_scale)


def temporal_point_errors(
    prediction,
    target,
    frame_valid,
    *,
    fps: float = 30.0,
    unit_scale: float = 1.0,
) -> dict[str, float | int]:
    """Return mask-aware velocity and acceleration errors for one contiguous point track.

    ``frame_valid`` must mark frames whose *whole hand* is valid.  Invalid or
    non-finite neighbours never bridge a temporal finite difference.
    """
    pred, gt = _matching_sequences(prediction, target, name="temporal point")
    valid = as_numpy(frame_valid, dtype=bool)
    if valid.shape != (pred.shape[0],):
        raise ValueError(f"frame_valid must have shape ({pred.shape[0]},), got {valid.shape}")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    valid &= np.isfinite(pred).all(axis=(1, 2)) & np.isfinite(gt).all(axis=(1, 2))
    result: dict[str, float | int] = {
        "velocity_error": float("nan"),
        "velocity_pair_count": 0,
        "acceleration_error": float("nan"),
        "acceleration_triplet_count": 0,
    }
    if pred.shape[0] >= 2:
        pair_valid = valid[:-1] & valid[1:]
        result["velocity_pair_count"] = int(np.count_nonzero(pair_valid))
        if np.any(pair_valid):
            difference = ((pred[1:] - pred[:-1]) - (gt[1:] - gt[:-1])) * fps
            result["velocity_error"] = float(np.mean(np.linalg.norm(difference[pair_valid], axis=-1)) * unit_scale)
    if pred.shape[0] >= 3:
        triplet_valid = valid[:-2] & valid[1:-1] & valid[2:]
        result["acceleration_triplet_count"] = int(np.count_nonzero(triplet_valid))
        if np.any(triplet_valid):
            difference = (
                (pred[2:] - 2.0 * pred[1:-1] + pred[:-2])
                - (gt[2:] - 2.0 * gt[1:-1] + gt[:-2])
            ) * (fps**2)
            result["acceleration_error"] = float(np.mean(np.linalg.norm(difference[triplet_valid], axis=-1)) * unit_scale)
    return result


def jitter(joints, *, fps: float = 30.0, divisor: float = 10.0, unit_scale: float = 1.0) -> np.ndarray:
    """Exact third finite-difference jitter used by ``compute_metric.py``."""
    value = _joint_sequence(joints)
    if value.shape[0] < 4:
        return np.empty(0, dtype=float)
    difference = (value[3:] - 3.0 * value[2:-1] + 3.0 * value[1:-2] - value[:-3]) * (fps**3)
    return np.linalg.norm(difference, axis=-1).mean(axis=-1) * unit_scale / divisor


def rte(target_trajectory, prediction_trajectory, *, percent: bool = False) -> np.ndarray:
    """Compute fixed-scale aligned root translation error normalized by GT travel."""
    target = as_numpy(target_trajectory, dtype=float)
    prediction = as_numpy(prediction_trajectory, dtype=float)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[-1] != 3:
        raise ValueError("trajectories must have matching shape (T, 3)")
    if target.shape[0] == 0:
        return np.empty(0, dtype=float)
    transform = umeyama(prediction[None], target[None], fix_scale=True)
    aligned = apply_transform(prediction[None], transform)[0]
    travel = float(np.sum(np.linalg.norm(np.diff(target, axis=0), axis=-1)))
    if travel == 0.0 or not np.isfinite(travel):
        return np.full(target.shape[0], np.nan, dtype=float)
    values = np.linalg.norm(target - aligned, axis=-1) / travel
    return values * (100.0 if percent else 1.0)


def contact_sliding(
    target_vertices,
    prediction_vertices,
    *,
    contact_vertex_indices,
    threshold: float = 1e-2,
    contact_mask=None,
    unit_scale: float = 1000.0,
) -> np.ndarray:
    """Return predicted displacement at GT-stationary/contact vertex locations."""
    target = as_numpy(target_vertices, dtype=float)
    prediction = as_numpy(prediction_vertices, dtype=float)
    if target.shape != prediction.shape or target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("vertices must have matching shape (T, V, 3)")
    indices = np.asarray(contact_vertex_indices, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= target.shape[1]):
        raise ValueError("contact_vertex_indices contains an invalid vertex")
    target_selected = target[:, indices]
    prediction_selected = prediction[:, indices]
    gt_displacement = np.linalg.norm(np.diff(target_selected, axis=0), axis=-1)
    predicted_displacement = np.linalg.norm(np.diff(prediction_selected, axis=0), axis=-1)
    if contact_mask is None:
        contacts = gt_displacement < threshold
    else:
        contacts = np.broadcast_to(as_numpy(contact_mask, dtype=bool), gt_displacement.shape)
    finite = np.isfinite(gt_displacement) & np.isfinite(predicted_displacement)
    return (predicted_displacement[contacts & finite] * unit_scale).astype(float)
