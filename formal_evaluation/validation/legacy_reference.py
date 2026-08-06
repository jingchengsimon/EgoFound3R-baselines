"""Read-only reproduction of the former formal metric definitions.

This module is only a comparison reference.  Formal reporting must use the
metrics in ``formal_evaluation.hand/scene/contact`` instead.
"""

from __future__ import annotations

import numpy as np


def _umeyama(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_mean, target_mean = prediction.mean(axis=0), target.mean(axis=0)
    pred_centered, target_centered = prediction - pred_mean, target - target_mean
    variance = np.sum(pred_centered**2) / prediction.shape[0]
    covariance = target_centered.T @ pred_centered / prediction.shape[0]
    left, singular, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        correction[2, 2] = -1.0
    rotation = left @ correction @ right
    scale = np.trace(np.diag(singular) @ correction) / variance if variance > 1e-10 else 1.0
    return (scale * (rotation @ prediction.T)).T + target_mean - scale * rotation @ pred_mean


def _legacy_mpjpe(prediction, target, frame_mask) -> float:
    errors = np.linalg.norm(prediction - target, axis=-1)
    return float(np.mean(errors[frame_mask]) * 1000.0) if np.any(frame_mask) else float("nan")


def _legacy_pa_mpjpe(prediction, target, frame_mask) -> float:
    values = []
    for index in np.flatnonzero(frame_mask):
        aligned = _umeyama(prediction[index], target[index])
        values.append(np.linalg.norm(aligned - target[index], axis=-1).mean() * 1000.0)
    return float(np.mean(values)) if values else float("nan")


def hand_metrics(prediction, target, prediction_valid, target_valid) -> dict[str, float]:
    result: dict[str, float] = {"hand_coverage": float(np.mean(np.any(prediction_valid, axis=1)))}
    for index, side in enumerate(("left", "right")):
        mask = prediction_valid[:, index].astype(bool) & target_valid[:, index].astype(bool)
        pred, gt = prediction[:, index], target[:, index]
        result[f"hand_{side}_mpjpe"] = _legacy_mpjpe(pred, gt, mask)
        result[f"hand_{side}_rr_mpjpe"] = _legacy_mpjpe(pred - pred[:, :1], gt - gt[:, :1], mask)
        result[f"hand_{side}_pa_mpjpe"] = _legacy_pa_mpjpe(pred, gt, mask)
        if np.any(mask):
            aligned = _umeyama(pred[mask].reshape(-1, 3), gt[mask].reshape(-1, 3))
            result[f"hand_{side}_sim3_mpjpe"] = float(np.linalg.norm(aligned - gt[mask].reshape(-1, 3), axis=-1).mean() * 1000.0)
        else:
            result[f"hand_{side}_sim3_mpjpe"] = float("nan")
        pred_present, gt_present = prediction_valid[:, index].astype(bool), target_valid[:, index].astype(bool)
        true_positive = int(np.count_nonzero(pred_present & gt_present))
        false_positive = int(np.count_nonzero(pred_present & ~gt_present))
        false_negative = int(np.count_nonzero(~pred_present & gt_present))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else float("nan")
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
        result[f"hand_{side}_presence_precision"] = precision
        result[f"hand_{side}_presence_recall"] = recall
        result[f"hand_{side}_presence_f1"] = 2.0 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return result


def scene_metrics(prediction_pose, target_pose, depth_pairs, *, scale_type: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if prediction_pose is not None and target_pose is not None:
        pred_pos, gt_pos = prediction_pose[:, :3, 3], target_pose[:, :3, 3]
        valid = np.isfinite(pred_pos).all(axis=1) & np.isfinite(gt_pos).all(axis=1)
        if np.any(valid):
            aligned = _umeyama(pred_pos[valid], gt_pos[valid])
            result["camera_ate_aligned"] = float(np.linalg.norm(aligned - gt_pos[valid], axis=-1).mean())
            relative = prediction_pose[valid, :3, :3] @ np.swapaxes(target_pose[valid, :3, :3], -1, -2)
            cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
            result["camera_rot_error_deg"] = float(np.mean(np.rad2deg(np.arccos(cosine))))
    if depth_pairs:
        if scale_type in {"relative", "up_to_scale"}:
            ratios = [gt[np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-8) & (gt > 0.01)] / pred[np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-8) & (gt > 0.01)] for pred, gt in depth_pairs]
            ratios = [ratio for ratio in ratios if ratio.size]
            scale = float(np.median(np.concatenate(ratios))) if ratios else float("nan")
        else:
            scale = 1.0
        result["depth_window_scale"] = scale
        per_frame = []
        for prediction, target in depth_pairs:
            mask = np.isfinite(prediction) & np.isfinite(target) & (target > 0.01)
            if np.count_nonzero(mask) < 100:
                continue
            pred, gt = prediction[mask] * scale, target[mask]
            ratio = np.maximum(pred / gt, gt / pred)
            per_frame.append({
                "abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
                "sq_rel": float(np.mean((pred - gt) ** 2 / gt)),
                "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
                "log_rmse": float(np.sqrt(np.mean((np.log(pred + 1e-8) - np.log(gt + 1e-8)) ** 2))),
                "delta1": float(np.mean(ratio < 1.25)),
                "delta2": float(np.mean(ratio < 1.25**2)),
                "delta3": float(np.mean(ratio < 1.25**3)),
            })
        for key in ("abs_rel", "sq_rel", "rmse", "log_rmse", "delta1", "delta2", "delta3"):
            values = [entry[key] for entry in per_frame]
            if values:
                result[f"depth_{key}"] = float(np.mean(values))
    return result


def contact_metrics(probability, target, mask, *, threshold: float = 0.5) -> dict[str, float]:
    selected_probability = probability[mask]
    selected_target = target[mask].astype(bool)
    predicted = selected_probability >= threshold
    true_positive = int(np.count_nonzero(predicted & selected_target))
    false_positive = int(np.count_nonzero(predicted & ~selected_target))
    false_negative = int(np.count_nonzero(~predicted & selected_target))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else float("nan")
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else float("nan"),
    }
