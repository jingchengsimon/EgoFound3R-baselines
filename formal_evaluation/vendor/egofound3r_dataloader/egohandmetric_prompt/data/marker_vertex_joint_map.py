from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from egohandmetric_prompt.data.marker_vertices import marker_vertex_ids_for_count


OPENPOSE_HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

DEFAULT_MARKER_VERTEX_JOINT_WEIGHT_COUNT = 195
SMOKE_MARKER_VERTEX_JOINT_WEIGHT_MAX_COUNT = 20


def marker_vertex_joint_weight_path(marker_count: int) -> Path:
    if int(marker_count) != DEFAULT_MARKER_VERTEX_JOINT_WEIGHT_COUNT:
        raise ValueError(f"没有 marker_count={marker_count} 的 vertex-joint mapping 权重文件。")
    return Path(__file__).with_name("marker_vertex_joint_weights_195.pt")


def closest_bone_vertex_joint_weights(
    marker_vertices: torch.Tensor,
    joints: torch.Tensor,
    *,
    bones: Sequence[tuple[int, int]] = OPENPOSE_HAND_BONES,
) -> torch.Tensor:
    if marker_vertices.ndim != 2 or marker_vertices.shape[-1] != 3:
        raise ValueError("marker_vertices must have shape [V, 3]")
    if joints.ndim != 2 or joints.shape != (21, 3):
        raise ValueError("joints must have shape [21, 3]")
    marker_vertices = marker_vertices.detach().to(dtype=torch.float32, device="cpu")
    joints = joints.detach().to(dtype=torch.float32, device="cpu")
    bone_index = torch.as_tensor(bones, dtype=torch.long)
    starts = joints[bone_index[:, 0]]
    ends = joints[bone_index[:, 1]]
    segments = ends - starts
    segment_length_sq = segments.square().sum(dim=-1).clamp_min(1e-8)
    relative_vertices = marker_vertices[:, None, :] - starts[None, :, :]
    projection_t = (relative_vertices * segments[None, :, :]).sum(dim=-1) / segment_length_sq[None, :]
    projection_t = projection_t.clamp(0.0, 1.0)
    closest_points = starts[None, :, :] + projection_t[..., None] * segments[None, :, :]
    distances = (marker_vertices[:, None, :] - closest_points).square().sum(dim=-1)
    closest_bones = distances.argmin(dim=-1)
    weights = torch.zeros(marker_vertices.shape[0], joints.shape[0], dtype=torch.float32)
    for vertex_index, bone_id in enumerate(closest_bones.tolist()):
        joint_a, joint_b = bones[bone_id]
        t_value = float(projection_t[vertex_index, bone_id].item())
        weights[vertex_index, joint_a] = 1.0 - t_value
        weights[vertex_index, joint_b] = t_value
    validate_vertex_joint_weights(weights, marker_count=marker_vertices.shape[0], joint_count=joints.shape[0])
    return weights


def validate_vertex_joint_weights(
    weights: torch.Tensor,
    *,
    marker_count: int,
    joint_count: int = 21,
    bones: Sequence[tuple[int, int]] = OPENPOSE_HAND_BONES,
) -> None:
    if weights.ndim != 2 or tuple(weights.shape) != (int(marker_count), int(joint_count)):
        raise ValueError(
            f"vertex-joint mapping shape must be [{int(marker_count)}, {int(joint_count)}], "
            f"got {tuple(weights.shape)}"
        )
    if not torch.is_floating_point(weights):
        raise ValueError("vertex-joint mapping must be floating point")
    weights_cpu = weights.detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(weights_cpu).all().item():
        raise ValueError("vertex-joint mapping contains non-finite values")
    if (weights_cpu < -1e-6).any().item():
        raise ValueError("vertex-joint mapping contains negative weights")
    row_sums = weights_cpu.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        raise ValueError("vertex-joint mapping rows must sum to 1")
    nonzero = weights_cpu > 1e-6
    nonzero_counts = nonzero.sum(dim=1)
    if (nonzero_counts == 0).any().item() or (nonzero_counts > 2).any().item():
        raise ValueError("vertex-joint mapping rows must contain one or two non-zero joints")
    valid_edges = {tuple(sorted(edge)) for edge in bones}
    valid_joint_ids = {joint for edge in bones for joint in edge}
    for row_index, row_nonzero in enumerate(nonzero):
        joint_ids = torch.nonzero(row_nonzero, as_tuple=False).flatten().tolist()
        if len(joint_ids) == 1:
            if joint_ids[0] not in valid_joint_ids:
                raise ValueError(f"vertex-joint mapping row {row_index} uses invalid joint {joint_ids[0]}")
            continue
        if tuple(sorted(joint_ids)) not in valid_edges:
            raise ValueError(f"vertex-joint mapping row {row_index} does not match one skeleton bone: {joint_ids}")


def smoke_vertex_joint_weights(marker_count: int, *, joint_count: int = 21) -> torch.Tensor:
    marker_count = int(marker_count)
    if marker_count <= 0 or marker_count > SMOKE_MARKER_VERTEX_JOINT_WEIGHT_MAX_COUNT:
        raise ValueError(f"没有 marker_count={marker_count} 的 smoke vertex-joint mapping。")
    weights = torch.zeros(marker_count, joint_count, dtype=torch.float32)
    for vertex_index, (joint_a, joint_b) in enumerate(OPENPOSE_HAND_BONES[:marker_count]):
        weights[vertex_index, joint_a] = 0.5
        weights[vertex_index, joint_b] = 0.5
    validate_vertex_joint_weights(weights, marker_count=marker_count, joint_count=joint_count)
    return weights


def marker_vertex_joint_weights_for_count(marker_count: int) -> torch.Tensor:
    marker_count = int(marker_count)
    if marker_count <= SMOKE_MARKER_VERTEX_JOINT_WEIGHT_MAX_COUNT:
        return smoke_vertex_joint_weights(marker_count)
    if marker_count != DEFAULT_MARKER_VERTEX_JOINT_WEIGHT_COUNT:
        raise ValueError(f"缺少 marker_count={marker_count} 的 vertex-joint mapping。")
    path = marker_vertex_joint_weight_path(marker_count)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    marker_vertex_ids = payload["marker_vertex_ids"].tolist()
    if marker_vertex_ids != marker_vertex_ids_for_count(marker_count):
        raise ValueError("vertex-joint mapping marker_vertex_ids 与当前 marker 顶点配置不一致。")
    bones = [tuple(edge) for edge in payload["bones"].tolist()]
    if bones != list(OPENPOSE_HAND_BONES):
        raise ValueError("vertex-joint mapping bones 与当前 hand skeleton 不一致。")
    weights = payload["weights"].to(dtype=torch.float32, device="cpu")
    validate_vertex_joint_weights(weights, marker_count=marker_count, joint_count=21)
    return weights


__all__ = [
    "OPENPOSE_HAND_BONES",
    "closest_bone_vertex_joint_weights",
    "marker_vertex_joint_weight_path",
    "marker_vertex_joint_weights_for_count",
    "smoke_vertex_joint_weights",
    "validate_vertex_joint_weights",
]
