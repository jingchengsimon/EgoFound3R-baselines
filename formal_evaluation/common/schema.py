from __future__ import annotations

from collections.abc import Mapping

import numpy as np


SCHEMA_VERSION = "egofound3r_comparison_output_v1"


def _require_shape(name: str, array: np.ndarray, expected: tuple[int | None, ...]) -> None:
    if array.ndim != len(expected):
        raise ValueError(f"{name} 维度错误: expected={expected}, actual={array.shape}")
    for actual, wanted in zip(array.shape, expected, strict=True):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name} 形状错误: expected={expected}, actual={array.shape}")


def _require_finite_where(name: str, array: np.ndarray, mask: np.ndarray) -> None:
    expanded_mask = mask
    while expanded_mask.ndim < array.ndim:
        expanded_mask = expanded_mask[..., None]
    expanded_mask = np.broadcast_to(expanded_mask, array.shape)
    if not np.isfinite(array[expanded_mask]).all():
        raise ValueError(f"{name} 在有效 mask 内含 NaN/Inf")


def validate_comparison_output(metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> None:
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version 错误: {metadata.get('schema_version')}")
    frame_ids = metadata.get("frame_ids")
    if not isinstance(frame_ids, list) or not frame_ids or not all(isinstance(item, str) for item in frame_ids):
        raise ValueError("metadata.frame_ids 必须是非空字符串列表")
    frame_count = len(frame_ids)
    capabilities = metadata.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise ValueError("metadata.capabilities 必须是字段到 bool 的映射")
    for field_name, enabled in capabilities.items():
        if not isinstance(field_name, str) or not isinstance(enabled, bool):
            raise ValueError("metadata.capabilities 只能包含字符串字段名和 bool 值")
        if enabled != (field_name in arrays):
            raise ValueError(f"capability 与输出不一致: {field_name}={enabled}")
    undeclared = sorted(set(arrays) - set(capabilities))
    if undeclared:
        raise ValueError(f"输出字段未在 capabilities 声明: {undeclared}")

    shapes: dict[str, tuple[int | None, ...]] = {
        "camera_c2w": (frame_count, 4, 4),
        "camera_valid": (frame_count,),
        "intrinsics": (frame_count, 3, 3),
        "intrinsics_valid": (frame_count,),
        "depth": (frame_count, None, None),
        "depth_valid": (frame_count, None, None),
        "depth_confidence": (frame_count, None, None),
        "world_points": (frame_count, None, None, 3),
        "world_points_valid": (frame_count, None, None),
        "world_points_confidence": (frame_count, None, None),
        "camera_points": (frame_count, None, None, 3),
        "camera_points_valid": (frame_count, None, None),
        "camera_points_confidence": (frame_count, None, None),
        "hand_joints_camera": (frame_count, 2, 21, 3),
        "hand_joints_world": (frame_count, 2, 21, 3),
        "hand_vertices_camera": (frame_count, 2, 778, 3),
        "hand_vertices_world": (frame_count, 2, 778, 3),
        "hand_markers_camera": (frame_count, 2, 195, 3),
        "hand_markers_world": (frame_count, 2, 195, 3),
        "hand_valid": (frame_count, 2),
        "hand_presence_probability": (frame_count, 2),
        "hand_visibility": (frame_count, 2, 21),
        "marker_visibility": (frame_count, 2, 195),
        "joint_contact_probability": (frame_count, 2, 21),
        "marker_contact_probability": (frame_count, 2, 195),
    }
    unknown = sorted(set(arrays) - set(shapes))
    if unknown:
        raise ValueError(f"未知 canonical 输出字段: {unknown}")
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} 必须是 numpy.ndarray")
        _require_shape(name, array, shapes[name])

    boolean_fields = {
        "camera_valid",
        "intrinsics_valid",
        "depth_valid",
        "world_points_valid",
        "camera_points_valid",
        "hand_valid",
    }
    for name in boolean_fields & arrays.keys():
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"{name} 必须为 bool")

    masked_fields = {
        "camera_c2w": "camera_valid",
        "intrinsics": "intrinsics_valid",
        "depth": "depth_valid",
        "depth_confidence": "depth_valid",
        "world_points": "world_points_valid",
        "world_points_confidence": "world_points_valid",
        "camera_points": "camera_points_valid",
        "camera_points_confidence": "camera_points_valid",
        "hand_joints_camera": "hand_valid",
        "hand_joints_world": "hand_valid",
        "hand_vertices_camera": "hand_valid",
        "hand_vertices_world": "hand_valid",
        "hand_markers_camera": "hand_valid",
        "hand_markers_world": "hand_valid",
        "hand_presence_probability": "hand_valid",
        "hand_visibility": "hand_valid",
        "marker_visibility": "hand_valid",
        "joint_contact_probability": "hand_valid",
        "marker_contact_probability": "hand_valid",
    }
    for field_name, mask_name in masked_fields.items():
        if field_name in arrays:
            if mask_name not in arrays:
                raise ValueError(f"{field_name} 缺少有效性字段 {mask_name}")
            _require_finite_where(field_name, arrays[field_name], arrays[mask_name])

    for probability_name in (
        "hand_presence_probability",
        "hand_visibility",
        "marker_visibility",
        "joint_contact_probability",
        "marker_contact_probability",
    ):
        if probability_name in arrays:
            values = arrays[probability_name]
            valid = arrays["hand_valid"]
            while valid.ndim < values.ndim:
                valid = valid[..., None]
            selected = values[np.broadcast_to(valid, values.shape)]
            if np.any((selected < 0.0) | (selected > 1.0)):
                raise ValueError(f"{probability_name} 有效值必须位于 [0, 1]")
