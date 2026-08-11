"""Contact aggregation that keeps the metrics-library empty-set semantics."""

from __future__ import annotations

import numpy as np

from formal_evaluation.contact.metrics import compute_contact_metrics


def aggregate_contact_windows(windows: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]], *, threshold: float = 0.5) -> dict[str, object]:
    """Report global micro and finite-only window-macro P/R/F1."""
    all_prediction, all_target, all_mask = [], [], []
    per_window = []
    for records in windows.values():
        prediction = np.concatenate([record[0].reshape(-1) for record in records])
        target = np.concatenate([record[1].reshape(-1) for record in records])
        mask = np.concatenate([record[2].reshape(-1) for record in records])
        metrics = compute_contact_metrics(prediction, target, mask, threshold=threshold)
        per_window.append(metrics)
        all_prediction.append(prediction)
        all_target.append(target)
        all_mask.append(mask)
    global_metrics = compute_contact_metrics(
        np.concatenate(all_prediction), np.concatenate(all_target), np.concatenate(all_mask), threshold=threshold
    ) if all_prediction else {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "valid_count": 0}
    output: dict[str, object] = {"global_micro": global_metrics, "evaluated_windows": len(per_window)}
    for key in ("precision", "recall", "f1"):
        values = np.asarray([metric[key] for metric in per_window], dtype=float)
        finite = values[np.isfinite(values)]
        output[f"window_macro_{key}"] = float(np.mean(finite)) if finite.size else float("nan")
        output[f"window_macro_{key}_undefined_window_count"] = int(values.size - finite.size)
    return output
