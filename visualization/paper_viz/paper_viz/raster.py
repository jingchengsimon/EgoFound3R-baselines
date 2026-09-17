"""Camera-space z-buffer rasterization and derived visibility/contact/distance signals."""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.spatial import cKDTree


@njit(cache=True)
def rasterize(vertices, faces, K, scale, height, width, owner_code, zbuf, owner, face_id):
    for fi in range(faces.shape[0]):
        i0, i1, i2 = faces[fi]
        x0, y0, z0 = vertices[i0]
        x1, y1, z1 = vertices[i1]
        x2, y2, z2 = vertices[i2]
        if z0 <= 1e-6 or z1 <= 1e-6 or z2 <= 1e-6:
            continue
        u0 = (K[0, 0] * x0 / z0 + K[0, 2]) * scale
        v0 = (K[1, 1] * y0 / z0 + K[1, 2]) * scale
        u1 = (K[0, 0] * x1 / z1 + K[0, 2]) * scale
        v1 = (K[1, 1] * y1 / z1 + K[1, 2]) * scale
        u2 = (K[0, 0] * x2 / z2 + K[0, 2]) * scale
        v2 = (K[1, 1] * y2 / z2 + K[1, 2]) * scale
        xmin = max(0, int(np.floor(min(u0, u1, u2))))
        xmax = min(width - 1, int(np.ceil(max(u0, u1, u2))))
        ymin = max(0, int(np.floor(min(v0, v1, v2))))
        ymax = min(height - 1, int(np.ceil(max(v0, v1, v2))))
        if xmin > xmax or ymin > ymax:
            continue
        den = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(den) < 1e-8:
            continue
        for py in range(ymin, ymax + 1):
            yy = py + 0.5
            for px in range(xmin, xmax + 1):
                xx = px + 0.5
                w0 = ((v1 - v2) * (xx - u2) + (u2 - u1) * (yy - v2)) / den
                w1 = ((v2 - v0) * (xx - u2) + (u0 - u2) * (yy - v2)) / den
                w2 = 1.0 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                invz = w0 / z0 + w1 / z1 + w2 / z2
                if invz <= 0:
                    continue
                depth = 1.0 / invz
                if depth < zbuf[py, px]:
                    zbuf[py, px] = depth
                    owner[py, px] = owner_code
                    face_id[py, px] = fi


def make_scene(object_vertices, object_faces, hand_vertices, hand_valid, faces, K, height, width, scale):
    zbuf = np.full((height, width), np.inf, np.float32)
    owner = np.full((height, width), -1, np.int16)
    face_id = np.full((height, width), -1, np.int32)
    if object_vertices is not None and len(object_vertices):
        rasterize(object_vertices.astype(np.float32), object_faces.astype(np.int64), K, scale,
                  height, width, 0, zbuf, owner, face_id)
    for side in range(2):
        if hand_valid[side]:
            rasterize(hand_vertices[side].astype(np.float32), faces, K, scale,
                      height, width, side + 1, zbuf, owner, face_id)
    return zbuf, owner, face_id


def derived_visibility(vertices, K, scale, zbuf, owner):
    """Per-vertex (2, 778) float in [0, 1]: 1 visible, 0 occluded by object or other hand."""
    height, width = zbuf.shape
    output = np.zeros((2, 778), np.float32)
    for side in range(2):
        projected = vertices[side] @ K.T
        depth = vertices[side, :, 2]
        uv = projected[:, :2] / np.where(depth[:, None] > 1e-6, projected[:, 2:3], np.nan)
        px = np.rint(uv[:, 0] * scale).astype(np.int64)
        py = np.rint(uv[:, 1] * scale).astype(np.int64)
        inside = (depth > 1e-6) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        ids = np.flatnonzero(inside)
        tolerance = np.maximum(0.002, 0.003 * depth[ids])
        output[side, ids] = ((owner[py[ids], px[ids]] == side + 1)
                             & (np.abs(zbuf[py[ids], px[ids]] - depth[ids]) <= tolerance))
    return output


def face_visibility(vertices, faces, K, scale, zbuf, owner):
    """Per-face (2, F) float in [0, 1] via centroid depth/owner test; robust at silhouettes."""
    output = np.zeros((2, faces.shape[0]), np.float32)
    centroids = vertices[:, faces].mean(2)
    for side in range(2):
        projected = centroids[side] @ K.T
        depth = centroids[side, :, 2]
        uv = projected[:, :2] / np.where(depth[:, None] > 1e-6, projected[:, 2:3], np.nan)
        px = np.rint(uv[:, 0] * scale).astype(np.int64)
        py = np.rint(uv[:, 1] * scale).astype(np.int64)
        height, width = zbuf.shape
        inside = (depth > 1e-6) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        ids = np.flatnonzero(inside)
        tolerance = np.maximum(0.002, 0.004 * depth[ids])
        output[side, ids] = ((owner[py[ids], px[ids]] == side + 1)
                             & (np.abs(zbuf[py[ids], px[ids]] - depth[ids]) <= tolerance))
    return output


def derived_distance(vertices, object_vertices):
    """Per-vertex (2, 778) distance in meters to nearest object vertex; inf without object."""
    if object_vertices is None or not len(object_vertices):
        return np.full((2, 778), np.inf, np.float32)
    tree = cKDTree(object_vertices)
    out = np.empty((2, 778), np.float32)
    for side in range(2):
        out[side] = tree.query(vertices[side], k=1)[0]
    return out


def derived_contact(distance_m: np.ndarray, cutoff: float = 0.02) -> np.ndarray:
    """Smooth contact probability from distance: 1 at 0 m, 0 beyond cutoff."""
    finite = np.isfinite(distance_m)
    values = np.clip(1.0 - np.where(finite, distance_m, cutoff) / cutoff, 0.0, 1.0)
    return np.where(finite, values, 0.0).astype(np.float32)
