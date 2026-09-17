"""MANO 195-to-778 upsampling contract (centroid-local signed weights, exact anchors)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ManoAsset:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as archive:
            self.neighbors = archive["neighbor_indices"]
            self.weights = archive["geometry_weights"]
            self.anchors = archive["source_vertex_ids"]
            self.faces = archive["faces"]

    def upsample(self, markers: np.ndarray) -> np.ndarray:
        """markers: (..., 195, 3) -> (..., 778, 3) with exact anchor restoration."""
        center = markers.mean(-2, keepdims=True)
        vertices = center + ((markers - center)[..., self.neighbors, :] * self.weights[..., None]).sum(-2)
        vertices[..., self.anchors, :] = markers
        return vertices

    def interpolate_scalar(self, values: np.ndarray) -> np.ndarray:
        """values: (..., 195) -> (..., 778) non-negative normalized weights."""
        weights = np.maximum(self.weights, 0.0)
        weights = weights / np.maximum(weights.sum(-1, keepdims=True), 1e-12)
        result = (values[..., self.neighbors] * weights).sum(-1)
        result[..., self.anchors] = values
        return result
