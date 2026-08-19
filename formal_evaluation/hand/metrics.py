"""Protocol-explicit hand geometry metrics for joints, markers, and MANO vertices."""

from __future__ import annotations

import numpy as np

from egocentric_metrics import binary_metrics, keypoint_mpjpe, temporal_point_errors, world_aligned_mpjpe, world_mpjpe


_GRANULARITIES = {
    "joint": (21, "mpjpe", "mpjve", "mpjae"),
    "marker": (195, "mpmpe", "mpmve", "mpmae"),
    "vertex": (778, "mpvpe", "mpvve", "mpvae"),
}


def _mean_per_frame(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _check_points(name: str, value: np.ndarray, *, point_count: int, frame_count: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (frame_count, 2, point_count, 3):
        raise ValueError(f"{name} must have shape ({frame_count}, 2, {point_count}, 3), got {result.shape}")
    return result


def _side_prefix(side: str, granularity: str) -> str:
    return f"hand_{side}_" if granularity == "joint" else f"hand_{side}_{granularity}_"


def compute_hand_metrics(
    prediction,
    target,
    prediction_valid,
    target_valid,
    *,
    granularity: str = "joint",
    root_prediction=None,
    root_target=None,
    world_prediction=None,
    world_target=None,
    temporal_fps: float = 30.0,
) -> dict[str, float | int]:
    """Evaluate one hand geometry granularity.

    Native metrics use the supplied coordinate frame. W/WA are emitted
    only when the caller supplies explicit world-coordinate tensors.
    """
    if granularity not in _GRANULARITIES:
        raise ValueError(f"unknown hand granularity: {granularity}")
    point_count, position_name, velocity_name, acceleration_name = _GRANULARITIES[granularity]
    pred = np.asarray(prediction, dtype=float)
    gt = np.asarray(target, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 4 or pred.shape[1:] != (2, point_count, 3):
        raise ValueError(f"{granularity} points must have matching (T, 2, {point_count}, 3) shapes")
    pred_valid = np.asarray(prediction_valid, dtype=bool)
    gt_valid = np.asarray(target_valid, dtype=bool)
    if pred_valid.shape != gt_valid.shape or pred_valid.shape != pred.shape[:2]:
        raise ValueError("hand validity must have shape (T, 2)")
    roots_pred = roots_gt = None
    if root_prediction is not None or root_target is not None:
        if root_prediction is None or root_target is None:
            raise ValueError("root prediction and target must be provided together")
        roots_pred = np.asarray(root_prediction, dtype=float)
        roots_gt = np.asarray(root_target, dtype=float)
        if roots_pred.shape != roots_gt.shape or roots_pred.shape != pred.shape[:2] + (3,):
            raise ValueError(f"roots must have shape {pred.shape[:2] + (3,)}")
    world_pred = world_gt = None
    if world_prediction is not None or world_target is not None:
        if world_prediction is None or world_target is None:
            raise ValueError("world prediction and target must be provided together")
        world_pred = _check_points("world prediction", world_prediction, point_count=point_count, frame_count=pred.shape[0])
        world_gt = _check_points("world target", world_target, point_count=point_count, frame_count=pred.shape[0])

    result: dict[str, float | int] = {"hand_coverage": float(np.mean(np.any(pred_valid, axis=1)))} if granularity == "joint" else {}
    for hand_index, side in enumerate(("left", "right")):
        prefix = _side_prefix(side, granularity)
        valid_frames = pred_valid[:, hand_index] & gt_valid[:, hand_index]
        point_mask = np.broadcast_to(valid_frames[:, None], (pred.shape[0], point_count))
        result[f"{prefix}valid_frame_count"] = int(np.count_nonzero(valid_frames))
        result[f"{prefix}{position_name}"] = float(keypoint_mpjpe(pred[:, hand_index], gt[:, hand_index], point_mask, alignment="none", unit_scale=1000.0))
        result[f"{prefix}pa_{position_name}"] = float(keypoint_mpjpe(pred[:, hand_index], gt[:, hand_index], point_mask, alignment="procrustes", unit_scale=1000.0))
        result[f"{prefix}global_sim3_{position_name}"] = _mean_per_frame(world_aligned_mpjpe(
            pred[:, hand_index], gt[:, hand_index], joint_mask=point_mask,
            mode="all", chunk_length=pred.shape[0], unit_scale=1000.0,
        ))
        if roots_pred is not None:
            relative_pred = pred[:, hand_index] - roots_pred[:, hand_index, None, :]
            relative_gt = gt[:, hand_index] - roots_gt[:, hand_index, None, :]
            result[f"{prefix}rr_{position_name}"] = float(keypoint_mpjpe(relative_pred, relative_gt, point_mask, alignment="none", unit_scale=1000.0))
        temporal = temporal_point_errors(pred[:, hand_index], gt[:, hand_index], valid_frames, fps=temporal_fps, unit_scale=1000.0)
        result[f"{prefix}{velocity_name}"] = float(temporal["velocity_error"])
        result[f"{prefix}{velocity_name}_pair_count"] = int(temporal["velocity_pair_count"])
        result[f"{prefix}{acceleration_name}"] = float(temporal["acceleration_error"])
        result[f"{prefix}{acceleration_name}_triplet_count"] = int(temporal["acceleration_triplet_count"])
        if world_pred is not None:
            world_mask = point_mask & np.isfinite(world_pred[:, hand_index]).all(axis=-1) & np.isfinite(world_gt[:, hand_index]).all(axis=-1)
            result[f"{prefix}w_{position_name}"] = _mean_per_frame(world_mpjpe(world_pred[:, hand_index], world_gt[:, hand_index], world_mask, unit_scale=1000.0))
            result[f"{prefix}wa_{position_name}"] = _mean_per_frame(world_aligned_mpjpe(
                world_pred[:, hand_index], world_gt[:, hand_index], joint_mask=world_mask,
                mode="all", chunk_length=pred.shape[0], unit_scale=1000.0,
            ))
        if granularity == "joint":
            presence = binary_metrics(pred_valid[:, hand_index], gt_valid[:, hand_index])
            result.update({
                f"hand_{side}_presence_precision": float(presence["precision"]),
                f"hand_{side}_presence_recall": float(presence["recall"]),
                f"hand_{side}_presence_f1": float(presence["f1"]),
                f"hand_{side}_sim3_mpjpe": result[f"{prefix}global_sim3_mpjpe"],
            })
    return result
