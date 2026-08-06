"""Rigid and similarity alignment primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import as_numpy


@dataclass(frozen=True)
class SimilarityTransform:
    """Transform satisfying ``target ~= scale * rotation @ source + translation``."""

    scale: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray


def umeyama(source: np.ndarray, target: np.ndarray, *, fix_scale: bool = False) -> SimilarityTransform:
    """Estimate a (batched) proper similarity transform with Umeyama SVD."""
    x = as_numpy(source, dtype=float)
    y = as_numpy(target, dtype=float)
    if x.shape != y.shape or x.ndim < 2 or x.shape[-1] != 3:
        raise ValueError(f"source and target must have the same (..., N, 3) shape, got {x.shape} and {y.shape}")
    if x.shape[-2] < 1:
        raise ValueError("at least one 3D correspondence is required")

    x_mean = np.mean(x, axis=-2, keepdims=True)
    y_mean = np.mean(y, axis=-2, keepdims=True)
    x0 = x - x_mean
    y0 = y - y_mean
    n_points = x.shape[-2]
    covariance = np.einsum("...ni,...nj->...ij", x0, y0) / n_points
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.broadcast_to(np.eye(3), u.shape).copy()
    reflection = np.linalg.det(u) * np.linalg.det(vt) < 0
    correction[..., 2, 2] = np.where(reflection, -1.0, 1.0)
    rotation = np.matmul(np.matmul(vt.swapaxes(-1, -2), correction), u.swapaxes(-1, -2))

    if fix_scale:
        scale = np.ones(x.shape[:-2], dtype=float)
    else:
        variance = np.sum(np.square(x0), axis=(-2, -1)) / n_points
        numerator = np.sum(singular_values * np.diagonal(correction, axis1=-2, axis2=-1), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = numerator / variance
    translation = (y_mean[..., 0, :] - scale[..., None] * np.einsum("...ij,...j->...i", rotation, x_mean[..., 0, :]))
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def apply_transform(points: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    points_array = as_numpy(points, dtype=float)
    if points_array.shape[-1] != 3:
        raise ValueError(f"points must end in dimension 3, got {points_array.shape}")
    rotated = np.einsum("...ij,...nj->...ni", transform.rotation, points_array)
    return transform.scale[..., None, None] * rotated + transform.translation[..., None, :]
