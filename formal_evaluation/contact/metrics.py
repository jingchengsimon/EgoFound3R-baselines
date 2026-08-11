"""Contact formal metrics backed by ``egocentric_metrics``."""

from __future__ import annotations

import numpy as np

from egocentric_metrics import average_precision, binary_metrics


def compute_contact_metrics(probability, target, supervision_mask, *, threshold: float = 0.5) -> dict[str, float | int]:
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(supervision_mask, dtype=bool)
    if probability.shape != target.shape or mask.shape != probability.shape:
        raise ValueError("contact probability, target, and supervision mask must share a shape")
    binary = binary_metrics(probability, target, mask=mask, prediction_type="score", threshold=threshold)
    result = {
        key: binary[key]
        for key in ("precision", "recall", "f1", "tp", "tn", "fp", "fn", "valid_count")
    }
    result["average_precision"] = average_precision(probability, target, mask=mask)
    return result
