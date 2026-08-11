"""Compatibility aggregate for the original global motion metric script."""

from __future__ import annotations

import numpy as np

from .common import as_numpy
from .mano import world_aligned_mpjpe, world_mpjpe
from .temporal import contact_sliding, jitter, rte


def compute_global_metrics(
    prediction_joints,
    target_joints,
    *,
    prediction_vertices=None,
    target_vertices=None,
    contact_vertex_indices=None,
    mask=None,
    chunk_length: int = 100,
    fps: float = 30.0,
    unit_scale: float = 1000.0,
) -> dict[str, np.ndarray]:
    """Compute the script's ``wa2_mpjpe``, ``waa_mpjpe``, ``rte``, ``jitter`` and ``fs``.

    Joint/vertex coordinates are expected in metres. ``wa2`` and ``waa`` and
    contact sliding are returned in millimetres; RTE is returned in percent;
    jitter retains the original script unit and divisor.
    """
    prediction = as_numpy(prediction_joints, dtype=float)
    target = as_numpy(target_joints, dtype=float)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("joint arrays must have matching shape (T, J, 3)")
    if mask is not None:
        frame_mask = as_numpy(mask, dtype=bool)
        if frame_mask.shape != (prediction.shape[0],):
            raise ValueError("global metric mask must have shape (T,)")
        prediction = prediction[frame_mask]
        target = target[frame_mask]
    result: dict[str, np.ndarray] = {
        "w_mpjpe": world_mpjpe(prediction, target, unit_scale=unit_scale),
        "wa2_mpjpe": world_aligned_mpjpe(prediction, target, mode="first2", chunk_length=chunk_length, unit_scale=unit_scale),
        "wa_mpjpe": world_aligned_mpjpe(prediction, target, mode="all", chunk_length=chunk_length, unit_scale=unit_scale),
        "rte": rte(target[:, 0], prediction[:, 0], percent=True),
        "jitter": jitter(prediction, fps=fps),
        "fs": np.empty(0, dtype=float),
    }
    result["waa_mpjpe"] = result["wa_mpjpe"]
    if prediction_vertices is not None or target_vertices is not None:
        if prediction_vertices is None or target_vertices is None or contact_vertex_indices is None:
            raise ValueError("prediction_vertices, target_vertices, and contact_vertex_indices are required together")
        pred_vertices = as_numpy(prediction_vertices, dtype=float)
        target_vertices_array = as_numpy(target_vertices, dtype=float)
        if mask is not None:
            pred_vertices = pred_vertices[frame_mask]
            target_vertices_array = target_vertices_array[frame_mask]
        result["fs"] = contact_sliding(
            target_vertices_array,
            pred_vertices,
            contact_vertex_indices=contact_vertex_indices,
            unit_scale=unit_scale,
        )
    return result
