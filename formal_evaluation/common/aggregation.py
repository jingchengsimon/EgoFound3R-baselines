"""Explicit window-level aggregation for formal evaluation."""

from __future__ import annotations

import numpy as np


def aggregate_windows(window_results: list[dict[str, object]], *, method: str) -> dict[str, object]:
    """Summarize scalar window metrics, excluding undefined (`NaN`) windows."""
    summary: dict[str, object] = {"method": method, "n_windows": len(window_results)}
    keys = {
        key
        for result in window_results
        for key, value in result.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    for key in sorted(keys):
        raw = [result[key] for result in window_results if key in result]
        values = np.asarray(raw, dtype=float)
        finite = values[np.isfinite(values)]
        summary[f"{key}_count"] = int(finite.size)
        summary[f"{key}_undefined_window_count"] = int(values.size - finite.size)
        if finite.size:
            summary[f"{key}_mean"] = float(np.mean(finite))
            summary[f"{key}_median"] = float(np.median(finite))
            summary[f"{key}_std"] = float(np.std(finite))
        else:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_median"] = float("nan")
            summary[f"{key}_std"] = float("nan")
    return summary
