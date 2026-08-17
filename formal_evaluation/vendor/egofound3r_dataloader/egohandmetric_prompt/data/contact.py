from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import torch
from pytorch3d.loss.point_mesh_distance import point_face_distance


FINGERTIP_OPENPOSE_INDICES = (4, 8, 12, 16, 20)
JOINT_CONTACT_THRESHOLD_METERS = 0.018
FINGERTIP_CONTACT_THRESHOLD_METERS = 0.014
MARKER_CONTACT_THRESHOLD_METERS = 0.014
DISTANCE_COORDINATE_SCALE = 1000.0
MIN_TRIANGLE_AREA_SCALED = 5e-3


@dataclass(frozen=True)
class LazyContactTargets:
    joint_targets: torch.Tensor
    joint_supervision_mask: torch.Tensor
    joint_distances: torch.Tensor
    joint_distance_supervision_mask: torch.Tensor
    marker_targets: torch.Tensor
    marker_supervision_mask: torch.Tensor
    marker_distances: torch.Tensor
    marker_distance_supervision_mask: torch.Tensor


class ObjectDistanceAccelerator(Protocol):
    def distances_camera(self, points_camera: torch.Tensor) -> torch.Tensor:
        """Return one finite metric point-to-surface distance per camera-frame point."""


def _empty_targets(*, marker_count: int, device: torch.device) -> LazyContactTargets:
    return LazyContactTargets(
        joint_targets=torch.zeros((2, 21), dtype=torch.float32, device=device),
        joint_supervision_mask=torch.zeros((2, 21), dtype=torch.bool, device=device),
        joint_distances=torch.zeros((2, 21), dtype=torch.float32, device=device),
        joint_distance_supervision_mask=torch.zeros((2, 21), dtype=torch.bool, device=device),
        marker_targets=torch.zeros((2, marker_count), dtype=torch.float32, device=device),
        marker_supervision_mask=torch.zeros((2, marker_count), dtype=torch.bool, device=device),
        marker_distances=torch.zeros((2, marker_count), dtype=torch.float32, device=device),
        marker_distance_supervision_mask=torch.zeros((2, marker_count), dtype=torch.bool, device=device),
    )


def _result_device(
    side_vertices: Sequence[torch.Tensor | None], side_joints: Sequence[torch.Tensor | None]
) -> torch.device:
    for value in (*side_vertices, *side_joints):
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _valid_vertices(value: torch.Tensor | None, *, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    vertices = torch.as_tensor(value, dtype=torch.float32, device=device)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        return None
    return vertices if bool(torch.isfinite(vertices).all().item()) else None


def _valid_points(
    value: torch.Tensor | None,
    *,
    count: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    point_mask = torch.zeros(count, dtype=torch.bool, device=device)
    if value is None:
        return None, point_mask
    points = torch.as_tensor(value, dtype=torch.float32, device=device)
    if points.shape != (count, 3):
        return None, point_mask
    return points, torch.isfinite(points).all(dim=-1)


def _valid_mesh(
    mesh: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if mesh is None:
        return None
    vertices = _valid_vertices(mesh[0], device=device)
    if vertices is None:
        return None
    faces = torch.as_tensor(mesh[1], dtype=torch.long, device=device)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.numel() == 0:
        return None
    if int(faces.min().item()) < 0 or int(faces.max().item()) >= int(vertices.shape[0]):
        return None
    triangles = vertices[faces]
    doubled_area = torch.linalg.vector_norm(
        torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1),
        dim=-1,
    )
    valid_faces = torch.isfinite(triangles).all(dim=(-2, -1)) & (doubled_area * DISTANCE_COORDINATE_SCALE**2 >= MIN_TRIANGLE_AREA_SCALED)
    if not bool(valid_faces.any().item()):
        return None
    return vertices, faces[valid_faces]


def _point_to_mesh_distances(
    points: torch.Tensor | None,
    point_mask: torch.Tensor,
    mesh: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    computation_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.zeros(point_mask.shape, dtype=torch.float32, device=point_mask.device)
    valid = torch.zeros_like(point_mask)
    if points is None or mesh is None or not bool(point_mask.any().item()):
        return distances, valid
    vertices, faces = mesh
    source_device = point_mask.device
    work_device = source_device if computation_device is None else torch.device(computation_device)
    try:
        point_indices = torch.nonzero(point_mask, as_tuple=False).flatten()
        selected_points = points[point_indices].to(device=work_device, dtype=torch.float32).contiguous()
        triangles = vertices[faces].to(device=work_device, dtype=torch.float32).contiguous()
        point_first_idx = torch.zeros(1, dtype=torch.int64, device=work_device)
        triangle_first_idx = torch.zeros(1, dtype=torch.int64, device=work_device)
        with torch.no_grad():
            squared_distances = point_face_distance(
                (selected_points * DISTANCE_COORDINATE_SCALE).contiguous(),
                point_first_idx,
                (triangles * DISTANCE_COORDINATE_SCALE).contiguous(),
                triangle_first_idx,
                int(selected_points.shape[0]),
                MIN_TRIANGLE_AREA_SCALED,
            )
            selected_distances = (
                squared_distances.clamp_min(0.0).sqrt() / DISTANCE_COORDINATE_SCALE
            ).to(device=source_device, dtype=torch.float32)
    except (OSError, RuntimeError):
        if work_device != source_device:
            return _point_to_mesh_distances(
                points,
                point_mask,
                mesh,
                computation_device=source_device,
            )
        raise
    selected_valid = torch.isfinite(selected_distances) & (selected_distances >= 0.0)
    if bool(selected_valid.any().item()):
        valid_indices = point_indices[selected_valid]
        distances[valid_indices] = selected_distances[selected_valid]
        valid[valid_indices] = True
    return distances, valid


def _point_to_object_accelerated_distances(
    points: torch.Tensor | None,
    point_mask: torch.Tensor,
    accelerator: ObjectDistanceAccelerator,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.zeros(point_mask.shape, dtype=torch.float32, device=point_mask.device)
    valid = torch.zeros_like(point_mask)
    if points is None or not bool(point_mask.any().item()):
        return distances, valid
    point_indices = torch.nonzero(point_mask, as_tuple=False).flatten()
    selected_points = points[point_indices].contiguous()
    selected_distances = torch.as_tensor(
        accelerator.distances_camera(selected_points),
        dtype=torch.float32,
        device=points.device,
    )
    if selected_distances.shape != (int(point_indices.shape[0]),):
        raise ValueError("object distance accelerator returned an unexpected shape")
    if not bool(torch.isfinite(selected_distances).all().item()) or bool((selected_distances < 0.0).any().item()):
        raise ValueError("object distance accelerator returned an invalid distance")
    distances[point_indices] = selected_distances
    valid[point_indices] = True
    return distances, valid


def _source_targets(
    points: torch.Tensor | None,
    point_mask: torch.Tensor,
    mesh: tuple[torch.Tensor, torch.Tensor] | None,
    thresholds: torch.Tensor,
    *,
    object_distance_accelerator: ObjectDistanceAccelerator | None = None,
    object_mesh_factory: Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    point_to_mesh_compute_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if object_distance_accelerator is None:
        distances, valid = _point_to_mesh_distances(
            points, point_mask, mesh, computation_device=point_to_mesh_compute_device
        )
    else:
        try:
            distances, valid = _point_to_object_accelerated_distances(
                points,
                point_mask,
                object_distance_accelerator,
            )
        except Exception:
            fallback_mesh = mesh if mesh is not None else (
                None if object_mesh_factory is None else object_mesh_factory()
            )
            distances, valid = _point_to_mesh_distances(points, point_mask, fallback_mesh)
    targets = torch.zeros(point_mask.shape, dtype=torch.float32, device=point_mask.device)
    targets[valid] = (distances[valid] <= thresholds[valid]).to(dtype=torch.float32)
    return targets, valid, distances


def _known_absent_source(point_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(point_mask.shape, dtype=torch.float32, device=point_mask.device),
        point_mask.clone(),
        torch.full(point_mask.shape, float("inf"), dtype=torch.float32, device=point_mask.device),
    )


def _union_sources(
    first_targets: torch.Tensor,
    first_mask: torch.Tensor,
    second_targets: torch.Tensor,
    second_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = (first_mask & (first_targets > 0.5)) | (second_mask & (second_targets > 0.5))
    supervision_mask = positive | (first_mask & second_mask)
    return positive.to(dtype=torch.float32), supervision_mask


def _combine_distances(
    first_distances: torch.Tensor,
    first_mask: torch.Tensor,
    second_distances: torch.Tensor,
    second_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    supervision_mask = first_mask & second_mask
    distances = torch.minimum(first_distances, second_distances)
    return torch.where(supervision_mask, distances, torch.zeros_like(distances)), supervision_mask


def _joint_thresholds(device: torch.device) -> torch.Tensor:
    thresholds = torch.full((21,), JOINT_CONTACT_THRESHOLD_METERS, dtype=torch.float32, device=device)
    thresholds[list(FINGERTIP_OPENPOSE_INDICES)] = FINGERTIP_CONTACT_THRESHOLD_METERS
    return thresholds


def lazy_contact_targets_from_scene(
    *,
    side_vertices: Sequence[torch.Tensor | None],
    side_joints: Sequence[torch.Tensor | None],
    mano_faces: torch.Tensor,
    marker_vertex_ids: Sequence[int],
    object_mesh: tuple[torch.Tensor, torch.Tensor] | None,
    mode: str,
    object_distance_accelerator: ObjectDistanceAccelerator | None = None,
    object_mesh_factory: Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    interhand_compute_device: torch.device | str | None = None,
) -> LazyContactTargets:
    if mode not in {"object_and_interhand", "interhand_only", "disabled"}:
        raise ValueError(f"非法 contact supervision mode: {mode}")
    if len(side_vertices) != 2 or len(side_joints) != 2:
        raise ValueError("lazy contact 需要固定 left/right 两侧输入。")
    device = _result_device(side_vertices, side_joints)
    marker_count = len(marker_vertex_ids)
    result = _empty_targets(marker_count=marker_count, device=device)
    if mode == "disabled":
        return result

    faces = torch.as_tensor(mano_faces, dtype=torch.long, device=device)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.numel() == 0 or int(faces.min().item()) < 0:
        return result
    object_tri_mesh = _valid_mesh(object_mesh, device=device)

    def resolve_object_tri_mesh() -> tuple[torch.Tensor, torch.Tensor] | None:
        nonlocal object_tri_mesh
        if object_tri_mesh is None and object_mesh_factory is not None:
            object_tri_mesh = _valid_mesh(object_mesh_factory(), device=device)
        if object_tri_mesh is None:
            raise ValueError("object mesh fallback is unavailable or invalid")
        return object_tri_mesh

    # For HOI supervision an unknown object source invalidates the complete
    # contact label.  Retaining inter-hand positives would turn an incomplete
    # object annotation into partial, semantically different supervision.
    if (
        mode == "object_and_interhand"
        and object_tri_mesh is None
        and (object_distance_accelerator is None or object_mesh_factory is None)
    ):
        return result
    marker_ids = torch.as_tensor(marker_vertex_ids, dtype=torch.long, device=device)
    joint_thresholds = _joint_thresholds(device)
    marker_thresholds = torch.full(
        (marker_count,), MARKER_CONTACT_THRESHOLD_METERS, dtype=torch.float32, device=device
    )

    joint_targets = result.joint_targets.clone()
    joint_masks = result.joint_supervision_mask.clone()
    joint_distances = result.joint_distances.clone()
    joint_distance_masks = result.joint_distance_supervision_mask.clone()
    marker_targets = result.marker_targets.clone()
    marker_masks = result.marker_supervision_mask.clone()
    marker_distances = result.marker_distances.clone()
    marker_distance_masks = result.marker_distance_supervision_mask.clone()

    vertices_by_side = [_valid_vertices(vertices, device=device) for vertices in side_vertices]
    for side_index, vertices in enumerate(vertices_by_side):
        if vertices is None or marker_ids.numel() and int(marker_ids.max().item()) >= int(vertices.shape[0]):
            continue
        joints, joint_point_mask = _valid_points(side_joints[side_index], count=21, device=device)
        markers, marker_point_mask = _valid_points(
            vertices[marker_ids] if marker_count else torch.empty((0, 3), device=device),
            count=marker_count,
            device=device,
        )
        if not bool(joint_point_mask.any().item()) and not bool(marker_point_mask.any().item()):
            continue
        other_vertices = vertices_by_side[1 - side_index]
        other_mesh = None if other_vertices is None else _valid_mesh((other_vertices, faces), device=device)

        if mode == "interhand_only":
            if other_mesh is None:
                continue
            joint_target, joint_mask, joint_distance = _source_targets(
                joints,
                joint_point_mask,
                other_mesh,
                joint_thresholds,
                point_to_mesh_compute_device=interhand_compute_device,
            )
            marker_target, marker_mask, marker_distance = _source_targets(
                markers,
                marker_point_mask,
                other_mesh,
                marker_thresholds,
                point_to_mesh_compute_device=interhand_compute_device,
            )
            joint_targets[side_index] = joint_target
            joint_masks[side_index] = joint_mask
            joint_distances[side_index] = joint_distance
            joint_distance_masks[side_index] = joint_mask
            marker_targets[side_index] = marker_target
            marker_masks[side_index] = marker_mask
            marker_distances[side_index] = marker_distance
            marker_distance_masks[side_index] = marker_mask
            continue

        try:
            object_joint_target, object_joint_mask, object_joint_distance = _source_targets(
                joints,
                joint_point_mask,
                object_tri_mesh,
                joint_thresholds,
                object_distance_accelerator=object_distance_accelerator,
                object_mesh_factory=resolve_object_tri_mesh,
            )
            object_marker_target, object_marker_mask, object_marker_distance = _source_targets(
                markers,
                marker_point_mask,
                object_tri_mesh,
                marker_thresholds,
                object_distance_accelerator=object_distance_accelerator,
                object_mesh_factory=resolve_object_tri_mesh,
            )
        except ValueError:
            # An HOI object annotation that cannot be queried by either path is
            # semantically unknown, so do not retain partial inter-hand labels.
            return result
        if other_mesh is None:
            inter_joint_target, inter_joint_mask, inter_joint_distance = _known_absent_source(joint_point_mask)
            inter_marker_target, inter_marker_mask, inter_marker_distance = _known_absent_source(marker_point_mask)
        else:
            inter_joint_target, inter_joint_mask, inter_joint_distance = _source_targets(
                joints,
                joint_point_mask,
                other_mesh,
                joint_thresholds,
                point_to_mesh_compute_device=interhand_compute_device,
            )
            inter_marker_target, inter_marker_mask, inter_marker_distance = _source_targets(
                markers,
                marker_point_mask,
                other_mesh,
                marker_thresholds,
                point_to_mesh_compute_device=interhand_compute_device,
            )
        joint_targets[side_index], joint_masks[side_index] = _union_sources(
            object_joint_target, object_joint_mask, inter_joint_target, inter_joint_mask
        )
        marker_targets[side_index], marker_masks[side_index] = _union_sources(
            object_marker_target, object_marker_mask, inter_marker_target, inter_marker_mask
        )
        joint_distances[side_index], joint_distance_masks[side_index] = _combine_distances(
            object_joint_distance, object_joint_mask, inter_joint_distance, inter_joint_mask
        )
        marker_distances[side_index], marker_distance_masks[side_index] = _combine_distances(
            object_marker_distance, object_marker_mask, inter_marker_distance, inter_marker_mask
        )

    return LazyContactTargets(
        joint_targets=joint_targets,
        joint_supervision_mask=joint_masks,
        joint_distances=joint_distances,
        joint_distance_supervision_mask=joint_distance_masks,
        marker_targets=marker_targets,
        marker_supervision_mask=marker_masks,
        marker_distances=marker_distances,
        marker_distance_supervision_mask=marker_distance_masks,
    )
