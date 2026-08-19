"""Fixed MeshGraphormer level-0/level-1 MANO resampling for evaluation only."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


MANO_VERTEX_COUNT = 778
MANO_MARKER_COUNT = 195
_ASSET = Path(__file__).with_name("data") / "mano_downsampling.npz"


@lru_cache(maxsize=1)
def _level1_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Return MeshGraphormer's fixed level-0<->level-1 linear maps."""
    with np.load(_ASSET, allow_pickle=True, encoding="latin1") as data:
        downsample = data["D"][0].toarray()
        upsample = data["U"][0].toarray()
    if downsample.shape != (MANO_MARKER_COUNT, MANO_VERTEX_COUNT):
        raise ValueError(f"unexpected MANO level-1 downsample shape: {downsample.shape}")
    if upsample.shape != (MANO_VERTEX_COUNT, MANO_MARKER_COUNT):
        raise ValueError(f"unexpected MANO level-1 upsample shape: {upsample.shape}")
    return np.asarray(downsample), np.asarray(upsample)


@lru_cache(maxsize=1)
def marker_vertex_ids_195() -> np.ndarray:
    """Return the level-1 vertices as their original 778-vertex MANO IDs."""
    downsample, _ = _level1_matrices()
    nonzero = np.count_nonzero(downsample, axis=1)
    if not np.all(nonzero == 1):
        raise ValueError("MANO level-1 downsample is not a vertex subset")
    ids = downsample.argmax(axis=1)
    if not np.allclose(downsample[np.arange(MANO_MARKER_COUNT), ids], 1.0):
        raise ValueError("MANO level-1 downsample has non-unit weights")
    return ids.astype(np.int64, copy=False)


def downsample_mano_vertices(vertices: np.ndarray) -> np.ndarray:
    """Select the canonical 195 MeshGraphormer marker vertices from MANO-778."""
    value = np.asarray(vertices)
    if value.ndim < 2 or value.shape[-2:] != (MANO_VERTEX_COUNT, 3):
        raise ValueError(f"expected (..., {MANO_VERTEX_COUNT}, 3), got {value.shape}")
    return value[..., marker_vertex_ids_195(), :]


def upsample_mano_markers(markers: np.ndarray) -> np.ndarray:
    """Linearly reconstruct the derived 778-vertex mesh from 195 level-1 markers."""
    value = np.asarray(markers)
    if value.ndim < 2 or value.shape[-2:] != (MANO_MARKER_COUNT, 3):
        raise ValueError(f"expected (..., {MANO_MARKER_COUNT}, 3), got {value.shape}")
    _, upsample = _level1_matrices()
    return np.einsum("vm,...mc->...vc", upsample.astype(value.dtype, copy=False), value)
