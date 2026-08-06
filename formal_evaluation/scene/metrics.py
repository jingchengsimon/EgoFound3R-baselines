"""Scene formal metrics backed by ``egocentric_metrics``."""

from __future__ import annotations

import numpy as np

from egocentric_metrics import ate, depth_metrics, extrinsic_metrics


def window_depth_scale(depth_pairs: list[tuple[np.ndarray, np.ndarray]], *, scale_type: str) -> float:
    """Protocol-level scale alignment; the metrics library intentionally does not infer it."""
    if scale_type not in {"relative", "up_to_scale"}:
        return 1.0
    ratios = []
    for prediction, target in depth_pairs:
        valid = np.isfinite(prediction) & np.isfinite(target) & (prediction > 0.0) & (target > 0.0)
        if np.any(valid):
            ratios.append((target[valid] / prediction[valid]).reshape(-1))
    return float(np.median(np.concatenate(ratios))) if ratios else float("nan")


def compute_scene_metrics(prediction_pose, target_pose, depth_pairs, *, scale_type: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if prediction_pose is not None and target_pose is not None:
        pred = np.asarray(prediction_pose, dtype=float)
        gt = np.asarray(target_pose, dtype=float)
        valid = np.isfinite(pred).all(axis=(1, 2)) & np.isfinite(gt).all(axis=(1, 2))
        if np.any(valid):
            result["camera_ate_aligned"] = ate(pred[valid, :3, 3], gt[valid, :3, 3], alignment="similarity")
            pose_values = extrinsic_metrics(pred, gt, mask=valid)
            result["camera_rot_error_deg"] = float(np.rad2deg(pose_values["rotation_geodesic_rad"]))
    if depth_pairs:
        scale = window_depth_scale(depth_pairs, scale_type=scale_type)
        result["depth_window_scale"] = scale
        frame_metrics = [depth_metrics(prediction * scale, target) for prediction, target in depth_pairs]
        for key in ("abs_rel", "sq_rel", "rmse", "log_rmse", "delta1", "delta2", "delta3"):
            values = np.asarray([metric[key] for metric in frame_metrics], dtype=float)
            if np.isfinite(values).any():
                result[f"depth_{key}"] = float(np.nanmean(values))
    return result
