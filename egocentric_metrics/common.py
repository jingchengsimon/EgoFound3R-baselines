"""Shared, NumPy-first input and reduction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Convert NumPy-like values and detached PyTorch tensors to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def prepare_mask(values: Any, mask: Any | None = None) -> np.ndarray:
    """Return an explicit boolean mask intersected with finite values."""
    array = as_numpy(values)
    valid = np.isfinite(array)
    if mask is not None:
        explicit = as_numpy(mask, dtype=bool)
        try:
            explicit = np.broadcast_to(explicit, array.shape)
        except ValueError as exc:
            raise ValueError(
                f"mask shape {explicit.shape} cannot broadcast to {array.shape}"
            ) from exc
        valid &= explicit
    return valid


def safe_mean(values: Any, mask: Any | None = None) -> float:
    array = as_numpy(values, dtype=float)
    valid = prepare_mask(array, mask)
    return float(np.mean(array[valid])) if np.any(valid) else float("nan")


def safe_rmse(values: Any, mask: Any | None = None) -> float:
    array = as_numpy(values, dtype=float)
    valid = prepare_mask(array, mask)
    return float(np.sqrt(np.mean(np.square(array[valid])))) if np.any(valid) else float("nan")


def require_shape(array: Any, *, ndim: int | None = None, last_dim: int | None = None, name: str = "array") -> np.ndarray:
    result = as_numpy(array)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {result.shape}")
    if last_dim is not None and (result.ndim == 0 or result.shape[-1] != last_dim):
        raise ValueError(f"{name} must end in dimension {last_dim}, got shape {result.shape}")
    return result
