"""MANO/body keypoint and mesh metrics."""

from __future__ import annotations

import numpy as np

from .alignment import apply_transform, umeyama
from .common import as_numpy


def _validate_points(prediction, target) -> tuple[np.ndarray, np.ndarray]:
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"points must have matching shape (T, N, 3), got {pred.shape} and {gt.shape}")
    return pred, gt


def _valid_joint_mask(pred: np.ndarray, gt: np.ndarray, mask) -> np.ndarray:
    valid = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    if mask is not None:
        valid &= np.broadcast_to(as_numpy(mask, dtype=bool), valid.shape)
    return valid


def _mean_point_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, unit_scale: float) -> tuple[float, int, np.ndarray]:
    errors = np.linalg.norm(pred - gt, axis=-1)
    per_frame = np.full(pred.shape[0], np.nan, dtype=float)
    for frame in range(pred.shape[0]):
        if np.any(mask[frame]):
            per_frame[frame] = np.mean(errors[frame, mask[frame]]) * unit_scale
    return (float(np.nanmean(per_frame)) if np.any(np.isfinite(per_frame)) else float("nan"), int(mask.sum()), per_frame)


def _procrustes_per_frame(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    aligned = np.full_like(pred, np.nan)
    for frame in range(pred.shape[0]):
        valid = mask[frame]
        if np.count_nonzero(valid) < 3:
            continue
        transform = umeyama(pred[frame, valid][None], gt[frame, valid][None])
        aligned[frame] = apply_transform(pred[frame][None], transform)[0]
    return aligned


def mano_metrics(
    prediction,
    target,
    *,
    joint_mask=None,
    root_index: int = 0,
    vertices_prediction=None,
    vertices_target=None,
    vertex_mask=None,
    unit_scale: float = 1000.0,
) -> dict[str, float | int | np.ndarray]:
    """Compute joint errors and, optionally, MANO vertex errors.

    Inputs are ``(T, J, 3)`` for joints and ``(T, V, 3)`` for vertices.
    ``unit_scale=1000`` reports metre inputs in millimetres.
    """
    pred, gt = _validate_points(prediction, target)
    if not 0 <= root_index < pred.shape[1]:
        raise ValueError(f"root_index {root_index} is outside J={pred.shape[1]}")
    mask = _valid_joint_mask(pred, gt, joint_mask)
    raw, count, raw_per_frame = _mean_point_error(pred, gt, mask, unit_scale)

    pred_root = pred - pred[:, root_index : root_index + 1]
    gt_root = gt - gt[:, root_index : root_index + 1]
    root_relative, _, root_per_frame = _mean_point_error(pred_root, gt_root, mask, unit_scale)

    aligned = _procrustes_per_frame(pred, gt, mask)
    pa_mask = mask & np.isfinite(aligned).all(axis=-1)
    pa, _, pa_per_frame = _mean_point_error(aligned, gt, pa_mask, unit_scale)
    result: dict[str, float | int | np.ndarray] = {
        "mpjpe": raw,
        "root_relative_mpjpe": root_relative,
        "pa_mpjpe": pa,
        "mpjpe_per_frame": raw_per_frame,
        "root_relative_mpjpe_per_frame": root_per_frame,
        "pa_mpjpe_per_frame": pa_per_frame,
        "valid_joint_count": count,
    }
    if vertices_prediction is not None or vertices_target is not None:
        if vertices_prediction is None or vertices_target is None:
            raise ValueError("vertices_prediction and vertices_target must be supplied together")
        vpred, vgt = _validate_points(vertices_prediction, vertices_target)
        vmask = _valid_joint_mask(vpred, vgt, vertex_mask)
        pve, vcount, pve_per_frame = _mean_point_error(vpred, vgt, vmask, unit_scale)
        vpred_root = vpred - vpred[:, :1]
        vgt_root = vgt - vgt[:, :1]
        root_pve, _, root_pve_per_frame = _mean_point_error(vpred_root, vgt_root, vmask, unit_scale)
        valigned = _procrustes_per_frame(vpred, vgt, vmask)
        vpa_mask = vmask & np.isfinite(valigned).all(axis=-1)
        pa_pve, _, pa_pve_per_frame = _mean_point_error(valigned, vgt, vpa_mask, unit_scale)
        result.update({
            "pve": pve,
            "root_relative_pve": root_pve,
            "pa_pve": pa_pve,
            "pve_per_frame": pve_per_frame,
            "root_relative_pve_per_frame": root_pve_per_frame,
            "pa_pve_per_frame": pa_pve_per_frame,
            "valid_vertex_count": vcount,
        })
    return result


def sequence_chunk_mpjpe(prediction, target, *, mode: str, chunk_length: int = 100, joint_mask=None, unit_scale: float = 1000.0) -> np.ndarray:
    """Return per-frame MPJPE after script-compatible chunk alignment."""
    pred, gt = _validate_points(prediction, target)
    if mode not in {"first2", "all"}:
        raise ValueError("mode must be 'first2' or 'all'")
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    mask = _valid_joint_mask(pred, gt, joint_mask)
    output = np.full(pred.shape[0], np.nan, dtype=float)
    for start in range(0, pred.shape[0], chunk_length):
        end = min(pred.shape[0], start + chunk_length)
        align_end = min(end, start + 2) if mode == "first2" else end
        align_mask = mask[start:align_end].reshape(-1)
        source = pred[start:align_end].reshape(-1, 3)
        destination = gt[start:align_end].reshape(-1, 3)
        if np.count_nonzero(align_mask) < 3:
            continue
        transform = umeyama(source[align_mask][None], destination[align_mask][None])
        aligned = apply_transform(pred[start:end][None], transform)[0]
        errors = np.linalg.norm(aligned - gt[start:end], axis=-1)
        for local_frame in range(end - start):
            valid = mask[start + local_frame]
            if np.any(valid):
                output[start + local_frame] = np.mean(errors[local_frame, valid]) * unit_scale
    return output


def world_mpjpe(prediction, target, joint_mask=None, *, unit_scale: float = 1000.0) -> np.ndarray:
    """Per-frame MPJPE in the supplied world coordinate system, without alignment."""
    pred, gt = _validate_points(prediction, target)
    mask = _valid_joint_mask(pred, gt, joint_mask)
    _, _, per_frame = _mean_point_error(pred, gt, mask, unit_scale)
    return per_frame


def world_aligned_mpjpe(
    prediction,
    target,
    joint_mask=None,
    *,
    mode: str = "all",
    chunk_length: int = 100,
    unit_scale: float = 1000.0,
) -> np.ndarray:
    """Per-frame world MPJPE after one Sim(3) alignment per temporal chunk.

    ``mode='all'`` is the usual WA-MPJPE interpretation and is identical to
    the original script's ``waa_mpjpe``. ``mode='first2'`` is the script's
    ``wa2_mpjpe`` protocol, useful for exposing drift after initialization.
    """
    return sequence_chunk_mpjpe(
        prediction,
        target,
        mode=mode,
        chunk_length=chunk_length,
        joint_mask=joint_mask,
        unit_scale=unit_scale,
    )


def keypoint_mpjpe(prediction, target, mask=None, *, alignment: str = "none", unit_scale: float = 1000.0) -> float:
    """Compute generic keypoint MPJPE with script-compatible alignment modes."""
    pred, gt = _validate_points(prediction, target)
    valid = _valid_joint_mask(pred, gt, mask)
    if alignment == "none":
        aligned = pred
    elif alignment == "procrustes":
        aligned = _procrustes_per_frame(pred, gt, valid)
        valid &= np.isfinite(aligned).all(axis=-1)
    elif alignment == "scale":
        aligned = pred.copy()
        for frame in range(pred.shape[0]):
            visible = valid[frame]
            denominator = np.sum(np.square(pred[frame, visible]))
            if np.any(visible) and denominator > 0.0:
                scale = np.sum(pred[frame, visible] * gt[frame, visible]) / denominator
                aligned[frame] *= scale
            else:
                valid[frame] = False
    else:
        raise ValueError("alignment must be 'none', 'scale', or 'procrustes'")
    value, _, _ = _mean_point_error(aligned, gt, valid, unit_scale)
    return value


def vertex_metrics(
    prediction,
    target,
    *,
    vertex_mask=None,
    root_index: int = 0,
    unit_scale: float = 1000.0,
) -> dict[str, float | int | np.ndarray]:
    """Compute MANO vertex PVE variants without requiring joint arrays."""
    pred, gt = _validate_points(prediction, target)
    if not 0 <= root_index < pred.shape[1]:
        raise ValueError(f"root_index {root_index} is outside V={pred.shape[1]}")
    mask = _valid_joint_mask(pred, gt, vertex_mask)
    pve, count, pve_per_frame = _mean_point_error(pred, gt, mask, unit_scale)
    pred_root = pred - pred[:, root_index : root_index + 1]
    gt_root = gt - gt[:, root_index : root_index + 1]
    root_pve, _, root_pve_per_frame = _mean_point_error(pred_root, gt_root, mask, unit_scale)
    aligned = _procrustes_per_frame(pred, gt, mask)
    pa_mask = mask & np.isfinite(aligned).all(axis=-1)
    pa_pve, _, pa_pve_per_frame = _mean_point_error(aligned, gt, pa_mask, unit_scale)
    return {
        "pve": pve,
        "root_relative_pve": root_pve,
        "pa_pve": pa_pve,
        "pve_per_frame": pve_per_frame,
        "root_relative_pve_per_frame": root_pve_per_frame,
        "pa_pve_per_frame": pa_pve_per_frame,
        "valid_vertex_count": count,
    }


def hand_scale_error(prediction, target, *, joint_pair: tuple[int, int] = (0, 9), unit_scale: float = 1000.0) -> float:
    """Mean error of a protocol-selected hand-scale joint-pair distance."""
    pred, gt = _validate_points(prediction, target)
    first, second = joint_pair
    if not (0 <= first < pred.shape[1] and 0 <= second < pred.shape[1] and first != second):
        raise ValueError(f"joint_pair {joint_pair} must contain distinct indices in [0, {pred.shape[1]})")
    pred_scale = np.linalg.norm(pred[:, first] - pred[:, second], axis=-1)
    gt_scale = np.linalg.norm(gt[:, first] - gt[:, second], axis=-1)
    valid = np.isfinite(pred_scale) & np.isfinite(gt_scale)
    return float(np.mean(np.abs(pred_scale[valid] - gt_scale[valid])) * unit_scale) if np.any(valid) else float("nan")
