"""Opt-in CPU BVH preserving the legacy contact kernel's degenerate faces."""
from contextlib import contextmanager
import numpy as np
import open3d as o3d
import torch


class LegacyCompatibleBVH:
    def __init__(self, original, kernel):
        self.original = original
        self.scale = kernel.DISTANCE_COORDINATE_SCALE
        self.min_area = kernel.MIN_TRIANGLE_AREA_SCALED
        self.cache = {}
        self.stats = dict(builds=0, small_faces=0, threshold_fallback_points=0)

    def clear(self):
        self.cache.clear()

    def __call__(self, points, mask, mesh, *, computation_device=None):
        if points is None or mesh is None or not bool(mask.any()):
            return self.original(points, mask, mesh, computation_device=computation_device)
        if points.device.type != 'cpu' or mask.device.type != 'cpu':
            raise ValueError('BVH compatibility path is CPU-only')
        vertices, faces = mesh
        key = (vertices.data_ptr(), faces.data_ptr())
        if key not in self.cache:
            scaled = vertices.float().contiguous() * self.scale
            triangles = scaled[faces]
            doubled_area = torch.linalg.vector_norm(torch.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0], dim=-1), dim=-1)
            # PyTorch3D uses triangle AREA, while the upstream mesh filter uses
            # DOUBLED area. A conservative boundary goes through the old kernel.
            small = doubled_area <= 2 * self.min_area * (1 + 1e-4)
            scene = None
            if bool((~small).any()):
                scene = o3d.t.geometry.RaycastingScene(nthreads=1)
                scene.add_triangles(o3d.core.Tensor(scaled.numpy()),
                    o3d.core.Tensor(faces[~small].numpy().astype(np.uint32)))
            # Retain both tensors: pointer identities cannot be recycled in cache.
            self.cache[key] = (vertices, faces, scene, faces[small])
            self.stats['builds'] += 1
            self.stats['small_faces'] += int(small.sum())
        _, _, scene, small_faces = self.cache[key]
        idx = torch.nonzero(mask).flatten()
        selected = points[idx].float().contiguous()
        if scene is None:
            d = torch.full((len(idx),), float('inf'))
        else:
            query = (selected * self.scale).contiguous().numpy()
            d = torch.from_numpy(scene.compute_distance(
                o3d.core.Tensor(query), nthreads=1).numpy().copy()) / self.scale
        if len(small_faces):
            exact, _ = self.original(selected, torch.ones(len(idx), dtype=torch.bool),
                                     (vertices, small_faces))
            d = torch.minimum(d, exact)
        # Float32 BVH and PyTorch3D rounding can disagree at the decision boundary.
        # Re-evaluate these points against the ENTIRE original mesh.
        near = (torch.abs(d - .014) <= 1e-6) | (torch.abs(d - .018) <= 1e-6)
        if bool(near.any()):
            exact, _ = self.original(selected, near, mesh)
            d[near] = exact[near]
            self.stats['threshold_fallback_points'] += int(near.sum())
        out = torch.zeros(mask.shape, dtype=torch.float32)
        valid = torch.zeros_like(mask)
        out[idx] = d
        valid[idx] = torch.isfinite(d) & (d >= 0)
        return out, valid


@contextmanager
def install(kernel):
    original = kernel._point_to_mesh_distances
    fast = LegacyCompatibleBVH(original, kernel)
    kernel._point_to_mesh_distances = fast
    try:
        yield fast
    finally:
        kernel._point_to_mesh_distances = original
