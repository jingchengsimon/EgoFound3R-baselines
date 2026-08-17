from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
import torch


DISTANCE_COORDINATE_SCALE = 1000.0
MIN_TRIANGLE_AREA_SCALED = 5e-3


def _valid_local_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    mesh_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_vertices = torch.as_tensor(vertices, dtype=torch.float32, device="cpu").contiguous()
    local_faces = torch.as_tensor(faces, dtype=torch.long, device="cpu").contiguous()
    if local_vertices.ndim != 2 or local_vertices.shape[1] != 3 or local_vertices.shape[0] < 3:
        raise ValueError("object mesh vertices must have shape (V, 3), V >= 3")
    if local_faces.ndim != 2 or local_faces.shape[1] != 3 or local_faces.numel() == 0:
        raise ValueError("object mesh faces must have shape (F, 3)")
    if int(local_faces.min().item()) < 0 or int(local_faces.max().item()) >= int(local_vertices.shape[0]):
        raise ValueError("object mesh face index is out of range")
    if not np.isfinite(mesh_scale) or mesh_scale <= 0.0:
        raise ValueError("mesh_scale must be finite and > 0")
    triangles = local_vertices[local_faces]
    doubled_area = torch.linalg.vector_norm(
        torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1),
        dim=-1,
    )
    valid_faces = torch.isfinite(triangles).all(dim=(-2, -1))
    valid_faces &= doubled_area * (float(mesh_scale) * DISTANCE_COORDINATE_SCALE) ** 2 >= MIN_TRIANGLE_AREA_SCALED
    if not bool(valid_faces.any().item()):
        raise ValueError("object mesh contains no finite non-degenerate triangles")
    return local_vertices, local_faces[valid_faces].contiguous()


def _object_to_camera_tensor(
    object_to_camera: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    transform = torch.as_tensor(object_to_camera, dtype=torch.float32, device=device)
    if transform.shape != (4, 4) or not bool(torch.isfinite(transform).all().item()):
        raise ValueError("object_to_camera must be a finite (4, 4) transform")
    return transform


def camera_points_to_object_local(
    points_camera: torch.Tensor,
    object_to_camera: torch.Tensor,
    *,
    mesh_scale: float,
) -> torch.Tensor:
    points = torch.as_tensor(points_camera, dtype=torch.float32)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points_camera must have shape (N, 3)")
    transform = _object_to_camera_tensor(object_to_camera, device=points.device)
    return (points - transform[:3, 3]) @ transform[:3, :3] / float(mesh_scale)


def camera_origin_to_object_local(
    object_to_camera: torch.Tensor,
    *,
    mesh_scale: float,
    device: torch.device,
) -> torch.Tensor:
    transform = _object_to_camera_tensor(object_to_camera, device=device)
    return (-transform[:3, 3]) @ transform[:3, :3] / float(mesh_scale)


@dataclass(frozen=True)
class ObjectMeshAccel:
    """Exact CPU BVH queries over one immutable object-local triangle mesh."""

    local_vertices: torch.Tensor
    local_faces: torch.Tensor
    mesh_scale: float
    _scene: o3d.t.geometry.RaycastingScene

    @classmethod
    def from_local_mesh(
        cls,
        local_vertices: torch.Tensor,
        local_faces: torch.Tensor,
        *,
        mesh_scale: float = 1.0,
    ) -> "ObjectMeshAccel":
        vertices, faces = _valid_local_mesh(
            local_vertices,
            local_faces,
            mesh_scale=mesh_scale,
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.core.Tensor(vertices.numpy(), dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(faces.to(dtype=torch.uint32).numpy(), dtype=o3d.core.Dtype.UInt32),
        )
        return cls(vertices, faces, float(mesh_scale), scene)

    def distances_camera(
        self,
        points_camera: torch.Tensor,
        object_to_camera: torch.Tensor,
    ) -> torch.Tensor:
        points = torch.as_tensor(points_camera, dtype=torch.float32)
        local_points = camera_points_to_object_local(
            points,
            object_to_camera,
            mesh_scale=self.mesh_scale,
        )
        if not bool(torch.isfinite(local_points).all().item()):
            raise ValueError("object distance query points must be finite")
        distances = self._scene.compute_distance(
            o3d.core.Tensor(
                local_points.detach().cpu().contiguous().numpy(),
                dtype=o3d.core.Dtype.Float32,
            )
        ).numpy()
        return torch.from_numpy(np.ascontiguousarray(distances)).to(
            device=points.device,
            dtype=torch.float32,
        ) * self.mesh_scale

    def segment_occluded_camera(
        self,
        endpoints_camera: torch.Tensor,
        object_to_camera: torch.Tensor,
        *,
        endpoint_epsilon: float,
    ) -> torch.Tensor:
        endpoints = torch.as_tensor(endpoints_camera, dtype=torch.float32)
        if endpoints.ndim != 2 or endpoints.shape[-1] != 3:
            raise ValueError("endpoints_camera must have shape (N, 3)")
        output = torch.zeros(endpoints.shape[0], dtype=torch.bool, device=endpoints.device)
        valid = torch.isfinite(endpoints).all(dim=-1)
        if not bool(valid.any().item()):
            return output
        valid_endpoints = endpoints[valid]
        local_endpoints = camera_points_to_object_local(
            valid_endpoints,
            object_to_camera,
            mesh_scale=self.mesh_scale,
        )
        origin = camera_origin_to_object_local(
            object_to_camera,
            mesh_scale=self.mesh_scale,
            device=endpoints.device,
        ).expand_as(local_endpoints)
        rays = torch.cat((origin, local_endpoints - origin), dim=-1)
        t_hit = self._scene.cast_rays(
            o3d.core.Tensor(
                rays.detach().cpu().contiguous().numpy(),
                dtype=o3d.core.Dtype.Float32,
            )
        )["t_hit"].numpy()
        hit = np.isfinite(t_hit)
        hit &= t_hit > float(endpoint_epsilon)
        hit &= t_hit < 1.0 - float(endpoint_epsilon)
        output[valid] = torch.from_numpy(hit).to(device=output.device, dtype=torch.bool)
        return output


class ObjectMeshAccelCache:
    """PID-safe LRU with payload-byte and entry budgets for local object BVHs.

    The payload accounting covers retained PyTorch local vertices/faces. Open3D
    owns additional opaque BVH memory, so this budget is intentionally not
    advertised as a process-RSS ceiling.
    """

    def __init__(self, *, max_entries: int = 64, max_bytes: int = 1 << 30) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._pid = os.getpid()
        self._entries: OrderedDict[
            tuple[str, int, int, float], tuple[ObjectMeshAccel, int]
        ] = OrderedDict()
        self.bytes_used = 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _reset_if_pid_changed(self) -> None:
        if self._pid != os.getpid():
            self._pid = os.getpid()
            self._entries.clear()
            self.bytes_used = 0

    @staticmethod
    def _mesh_nbytes(accel: ObjectMeshAccel) -> int:
        return int(
            accel.local_vertices.numel() * accel.local_vertices.element_size()
            + accel.local_faces.numel() * accel.local_faces.element_size()
        )

    def get_or_create(
        self,
        mesh_path: str,
        local_vertices: torch.Tensor,
        local_faces: torch.Tensor,
        *,
        mesh_scale: float,
    ) -> ObjectMeshAccel:
        self._reset_if_pid_changed()
        path = Path(mesh_path).resolve()
        stat = path.stat()
        key = (str(path), int(stat.st_mtime_ns), int(stat.st_size), float(mesh_scale))
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached
            return cached[0]
        accel = ObjectMeshAccel.from_local_mesh(
            local_vertices,
            local_faces,
            mesh_scale=mesh_scale,
        )
        mesh_nbytes = self._mesh_nbytes(accel)
        if mesh_nbytes > self.max_bytes:
            return accel
        self._entries[key] = (accel, mesh_nbytes)
        self.bytes_used += mesh_nbytes
        while len(self._entries) > self.max_entries or self.bytes_used > self.max_bytes:
            _, (_, evicted_bytes) = self._entries.popitem(last=False)
            self.bytes_used -= evicted_bytes
        return accel


class ObjectMeshRawCache:
    """PID-safe, byte- and entry-bounded LRU for immutable CPU object meshes."""

    def __init__(self, *, max_entries: int = 64, max_bytes: int = 1 << 30) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._pid = os.getpid()
        self._entries: OrderedDict[
            tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, int]
        ] = OrderedDict()
        self.bytes_used = 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def _reset_if_pid_changed(self) -> None:
        if self._pid != os.getpid():
            self._pid = os.getpid()
            self._entries.clear()
            self.bytes_used = 0

    @staticmethod
    def _mesh_nbytes(vertices: torch.Tensor, faces: torch.Tensor) -> int:
        return int(vertices.numel() * vertices.element_size() + faces.numel() * faces.element_size())

    def get_or_load(
        self,
        mesh_path: str,
        loader: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._reset_if_pid_changed()
        path = Path(mesh_path).resolve()
        stat = path.stat()
        key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached
            return cached[0], cached[1]

        loaded_vertices, loaded_faces = loader(str(path))
        vertices = torch.as_tensor(loaded_vertices, dtype=torch.float32, device="cpu").contiguous()
        faces = torch.as_tensor(loaded_faces, dtype=torch.long, device="cpu").contiguous()
        mesh_nbytes = self._mesh_nbytes(vertices, faces)
        if mesh_nbytes > self.max_bytes:
            return vertices, faces
        self._entries[key] = (vertices, faces, mesh_nbytes)
        self.bytes_used += mesh_nbytes
        while len(self._entries) > self.max_entries or self.bytes_used > self.max_bytes:
            _, (_, _, evicted_bytes) = self._entries.popitem(last=False)
            self.bytes_used -= evicted_bytes
        return vertices, faces


@dataclass(frozen=True)
class SceneObjectPart:
    """One immutable local object mesh and its current object-to-camera transform."""

    mesh_path: str
    local_vertices: torch.Tensor
    local_faces: torch.Tensor
    object_to_camera: torch.Tensor
    mesh_scale: float = 1.0

    def camera_vertices(self) -> torch.Tensor:
        vertices = torch.as_tensor(self.local_vertices, dtype=torch.float32)
        transform = _object_to_camera_tensor(
            self.object_to_camera,
            device=vertices.device,
        )
        if not np.isfinite(self.mesh_scale) or self.mesh_scale <= 0.0:
            raise ValueError("mesh_scale must be finite and > 0")
        return (
            vertices * float(self.mesh_scale)
        ) @ transform[:3, :3].T + transform[:3, 3]


@dataclass(frozen=True)
class SceneObjectMesh:
    """Frame scene object list that preserves local meshes for BVH reuse."""

    parts: tuple[SceneObjectPart, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("SceneObjectMesh requires at least one part")

    @property
    def face_count(self) -> int:
        return sum(int(part.local_faces.shape[0]) for part in self.parts)

    def camera_tri_mesh(self) -> tuple[torch.Tensor, torch.Tensor]:
        vertices_by_part: list[torch.Tensor] = []
        faces_by_part: list[torch.Tensor] = []
        vertex_offset = 0
        for part in self.parts:
            vertices = part.camera_vertices()
            faces = torch.as_tensor(part.local_faces, dtype=torch.long)
            vertices_by_part.append(vertices)
            faces_by_part.append(faces + vertex_offset)
            vertex_offset += int(vertices.shape[0])
        return torch.cat(vertices_by_part, dim=0), torch.cat(faces_by_part, dim=0)


@dataclass(frozen=True)
class SceneObjectAccel:
    """Union query accelerator for a frame's original object meshes."""

    parts: tuple[tuple[ObjectMeshAccel, torch.Tensor], ...]

    @classmethod
    def from_scene_object(
        cls,
        scene_object: SceneObjectMesh,
        *,
        cache: ObjectMeshAccelCache | None = None,
    ) -> "SceneObjectAccel":
        parts: list[tuple[ObjectMeshAccel, torch.Tensor]] = []
        for object_part in scene_object.parts:
            if cache is None:
                accel = ObjectMeshAccel.from_local_mesh(
                    object_part.local_vertices,
                    object_part.local_faces,
                    mesh_scale=object_part.mesh_scale,
                )
            else:
                accel = cache.get_or_create(
                    object_part.mesh_path,
                    object_part.local_vertices,
                    object_part.local_faces,
                    mesh_scale=object_part.mesh_scale,
                )
            parts.append((accel, torch.as_tensor(object_part.object_to_camera, dtype=torch.float32)))
        return cls(tuple(parts))

    def distances_camera(self, points_camera: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(points_camera, dtype=torch.float32)
        if not self.parts:
            raise ValueError("SceneObjectAccel has no object parts")
        distances = [
            accel.distances_camera(points, object_to_camera)
            for accel, object_to_camera in self.parts
        ]
        return torch.stack(distances, dim=0).amin(dim=0)

    def segment_occluded_camera(
        self,
        endpoints_camera: torch.Tensor,
        *,
        endpoint_epsilon: float,
    ) -> torch.Tensor:
        endpoints = torch.as_tensor(endpoints_camera, dtype=torch.float32)
        if not self.parts:
            raise ValueError("SceneObjectAccel has no object parts")
        return torch.stack(
            [
                accel.segment_occluded_camera(
                    endpoints,
                    object_to_camera,
                    endpoint_epsilon=endpoint_epsilon,
                )
                for accel, object_to_camera in self.parts
            ],
            dim=0,
        ).any(dim=0)
