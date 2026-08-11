"""Detection, heatmap, and binary ranking metrics."""

from __future__ import annotations

import numpy as np

from .common import as_numpy


def average_precision(scores, labels, mask=None) -> dict[str, object]:
    score = as_numpy(scores, dtype=float).reshape(-1)
    label = as_numpy(labels, dtype=bool).reshape(-1)
    if score.shape != label.shape:
        raise ValueError("scores and labels must have the same shape")
    valid = np.isfinite(score)
    if mask is not None:
        valid &= np.broadcast_to(as_numpy(mask, dtype=bool).reshape(-1), valid.shape)
    score = score[valid]
    label = label[valid]
    order = np.argsort(-score, kind="mergesort")
    label = label[order]
    score = score[order]
    positives = int(np.count_nonzero(label))
    if positives == 0:
        return {"ap": float("nan"), "precision": np.empty(0), "recall": np.empty(0), "valid_count": int(label.size)}
    true_positive = np.cumsum(label, dtype=float)
    false_positive = np.cumsum(~label, dtype=float)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / positives
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    ap = float(np.sum((recall - previous_recall) * precision))
    return {"ap": ap, "precision": precision, "recall": recall, "valid_count": int(label.size)}


def roc_auc(scores, labels, mask=None) -> float:
    score = as_numpy(scores, dtype=float).reshape(-1)
    label = as_numpy(labels, dtype=bool).reshape(-1)
    if score.shape != label.shape:
        raise ValueError("scores and labels must have the same shape")
    valid = np.isfinite(score)
    if mask is not None:
        valid &= np.broadcast_to(as_numpy(mask, dtype=bool).reshape(-1), valid.shape)
    score = score[valid]
    label = label[valid]
    positive = int(np.count_nonzero(label))
    negative = int(label.size - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = float(np.sum(original_ranks[label]))
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def soft_similarity(prediction, target) -> float:
    pred = np.maximum(as_numpy(prediction, dtype=float).reshape(-1), 0.0)
    gt = np.maximum(as_numpy(target, dtype=float).reshape(-1), 0.0)
    if pred.shape != gt.shape:
        raise ValueError("soft maps must have the same shape")
    pred_sum = float(np.sum(pred))
    gt_sum = float(np.sum(gt))
    if pred_sum <= 0.0 or gt_sum <= 0.0:
        return float("nan")
    pred /= pred_sum
    gt /= gt_sum
    return float(np.sum(np.minimum(pred, gt)))


def box_iou(prediction_boxes, target_boxes) -> np.ndarray:
    pred = as_numpy(prediction_boxes, dtype=float)
    gt = as_numpy(target_boxes, dtype=float)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[-1] != 4 or gt.shape[-1] != 4:
        raise ValueError("boxes must have shape (N, 4) in xyxy format")
    pred_area = np.maximum(pred[:, 2] - pred[:, 0], 0.0) * np.maximum(pred[:, 3] - pred[:, 1], 0.0)
    gt_area = np.maximum(gt[:, 2] - gt[:, 0], 0.0) * np.maximum(gt[:, 3] - gt[:, 1], 0.0)
    left_top = np.maximum(pred[:, None, :2], gt[None, :, :2])
    right_bottom = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    intersection = np.prod(np.maximum(right_bottom - left_top, 0.0), axis=-1)
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def detection_average_precision(prediction_boxes, scores, target_boxes, *, iou_threshold: float = 0.5) -> dict[str, object]:
    pred = as_numpy(prediction_boxes, dtype=float)
    score = as_numpy(scores, dtype=float).reshape(-1)
    gt = as_numpy(target_boxes, dtype=float)
    if pred.shape[0] != score.size or iou_threshold <= 0.0 or iou_threshold > 1.0:
        raise ValueError("prediction boxes/scores mismatch or invalid IoU threshold")
    overlaps = box_iou(pred, gt)
    order = np.argsort(-score, kind="mergesort")
    matched = np.zeros(gt.shape[0], dtype=bool)
    labels = np.zeros(pred.shape[0], dtype=bool)
    for index in order:
        if gt.shape[0] == 0:
            continue
        target_index = int(np.argmax(overlaps[index]))
        if overlaps[index, target_index] >= iou_threshold and not matched[target_index]:
            labels[index] = True
            matched[target_index] = True
    if gt.shape[0] == 0:
        return {"ap": float("nan"), "precision": np.empty(0), "recall": np.empty(0), "valid_count": int(pred.shape[0])}
    ordered_labels = labels[order]
    true_positive = np.cumsum(ordered_labels, dtype=float)
    false_positive = np.cumsum(~ordered_labels, dtype=float)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall = true_positive / gt.shape[0]
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    return {
        "ap": float(np.sum((recall - previous_recall) * precision)),
        "precision": precision,
        "recall": recall,
        "valid_count": int(pred.shape[0]),
    }


def mean_average_precision(prediction_boxes_by_class, scores_by_class, target_boxes_by_class, *, iou_threshold: float = 0.5) -> dict[str, object]:
    """Macro-average single-image detection AP over caller-provided classes."""
    if not (len(prediction_boxes_by_class) == len(scores_by_class) == len(target_boxes_by_class)):
        raise ValueError("per-class prediction boxes, scores, and target boxes must have equal lengths")
    ap_values = np.array([
        detection_average_precision(pred_boxes, scores, gt_boxes, iou_threshold=iou_threshold)["ap"]
        for pred_boxes, scores, gt_boxes in zip(prediction_boxes_by_class, scores_by_class, target_boxes_by_class)
    ], dtype=float)
    finite = ap_values[np.isfinite(ap_values)]
    return {"map": float(np.mean(finite)) if finite.size else float("nan"), "ap_per_class": ap_values, "class_count": int(ap_values.size)}
