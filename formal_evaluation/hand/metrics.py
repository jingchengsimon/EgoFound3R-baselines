"""Hand formal metrics backed exclusively by ``egocentric_metrics``."""

from __future__ import annotations

import numpy as np

from egocentric_metrics import binary_metrics, mano_metrics, world_aligned_mpjpe


def compute_hand_metrics(prediction, target, prediction_valid, target_valid) -> dict[str, float | int]:
    """Evaluate left/right camera-space 21-joint hand predictions per window."""
    pred = np.asarray(prediction, dtype=float)
    gt = np.asarray(target, dtype=float)
    pred_valid = np.asarray(prediction_valid, dtype=bool)
    gt_valid = np.asarray(target_valid, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 4 or pred.shape[1:] != (2, 21, 3):
        raise ValueError("hand joints must have matching (T, 2, 21, 3) shapes")
    if pred_valid.shape != gt_valid.shape or pred_valid.shape != pred.shape[:2]:
        raise ValueError("hand validity must have shape (T, 2)")

    result: dict[str, float | int] = {
        "hand_coverage": float(np.mean(np.any(pred_valid, axis=1))),
    }
    for hand_index, side in enumerate(("left", "right")):
        valid_frames = pred_valid[:, hand_index] & gt_valid[:, hand_index]
        joint_mask = np.broadcast_to(valid_frames[:, None], pred.shape[:1] + (21,))
        values = mano_metrics(pred[:, hand_index], gt[:, hand_index], joint_mask=joint_mask, root_index=0)
        aligned = world_aligned_mpjpe(
            pred[:, hand_index],
            gt[:, hand_index],
            joint_mask=joint_mask,
            mode="all",
            chunk_length=pred.shape[0],
            unit_scale=1000.0,
        )
        presence = binary_metrics(pred_valid[:, hand_index], gt_valid[:, hand_index])
        result.update({
            f"hand_{side}_mpjpe": float(values["mpjpe"]),
            f"hand_{side}_rr_mpjpe": float(values["root_relative_mpjpe"]),
            f"hand_{side}_pa_mpjpe": float(values["pa_mpjpe"]),
            f"hand_{side}_sim3_mpjpe": float(np.nanmean(aligned)) if np.isfinite(aligned).any() else float("nan"),
            f"hand_{side}_presence_precision": float(presence["precision"]),
            f"hand_{side}_presence_recall": float(presence["recall"]),
            f"hand_{side}_presence_f1": float(presence["f1"]),
            f"hand_{side}_valid_frame_count": int(np.count_nonzero(valid_frames)),
        })
    return result
