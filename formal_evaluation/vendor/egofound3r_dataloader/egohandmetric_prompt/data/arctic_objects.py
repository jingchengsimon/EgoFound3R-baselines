"""Lazy official ARCTIC articulated-object descriptors for EgoForce frames."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


_SEQUENCE_CACHE_LIMIT = 4


@dataclass(frozen=True)
class ArcticFrameIdentity:
    subject: str
    sequence_name: str
    frame_id: int


def parse_egoforce_arctic_identity(sequence_id: str, frame_id: str | int) -> ArcticFrameIdentity:
    parts = sequence_id.split("@")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid EgoForce ARCTIC sequence_id: {sequence_id!r}")
    subject, sequence_name, camera_name = parts
    if camera_name != "cam0":
        raise ValueError(f"Official ARCTIC egocamera geometry only supports cam0, got {camera_name!r}")
    try:
        image_frame_id = int(frame_id)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid EgoForce ARCTIC frame_id: {frame_id!r}") from error
    if image_frame_id < 0:
        raise ValueError(f"EgoForce ARCTIC frame_id must be non-negative, got {image_frame_id}")
    return ArcticFrameIdentity(subject=subject, sequence_name=sequence_name, frame_id=image_frame_id)


def axis_angle_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Return a 3x3 Rodrigues rotation matrix for a finite axis-angle vector."""
    vector = np.asarray(axis_angle, dtype=np.float32).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("axis-angle rotation must be a finite vector of shape (3,)")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = vector / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float32,
    )
    identity = np.eye(3, dtype=np.float32)
    return (identity + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)).astype(np.float32)


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return transform


def articulated_part_transforms(
    *,
    articulation_radians: float,
    object_axis_angle: np.ndarray,
    object_translation_mm: np.ndarray,
    world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return object bottom/top transforms in the EgoForce cam0 coordinate system."""
    if not np.isfinite(articulation_radians):
        raise ValueError("ARCTIC articulation must be finite")
    camera_transform = np.asarray(world_to_camera, dtype=np.float32)
    if camera_transform.shape != (4, 4) or not np.isfinite(camera_transform).all():
        raise ValueError("ARCTIC world_to_camera must be a finite (4, 4) matrix")
    global_transform = _homogeneous(axis_angle_matrix(object_axis_angle), np.asarray(object_translation_mm) / 1000.0)
    top_articulation = _homogeneous(axis_angle_matrix(np.array([0.0, 0.0, -articulation_radians])), np.zeros(3))
    return (
        (camera_transform @ global_transform).astype(np.float32),
        (camera_transform @ global_transform @ top_articulation).astype(np.float32),
    )


class ArcticObjectResolver:
    """Resolve real ARCTIC object parts at sample materialization time only."""

    def __init__(self, sidecar_root: str | Path) -> None:
        self.sidecar_root = Path(sidecar_root)
        self.data_root = self.sidecar_root / "arctic_data" / "data"
        self._misc: dict[str, Any] | None = None
        self._sequence_cache: OrderedDict[str, tuple[np.ndarray, dict[str, Any]]] = OrderedDict()

    def _published_layout_exists(self) -> bool:
        return self.data_root.is_dir()

    def _load_misc(self) -> dict[str, Any]:
        if self._misc is not None:
            return self._misc
        path = self.data_root / "meta" / "misc.json"
        if not path.is_file():
            raise FileNotFoundError(f"Published ARCTIC sidecar is missing misc.json: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"ARCTIC misc.json must contain a JSON object: {path}")
        self._misc = payload
        return payload

    def _sequence_arrays(self, identity: ArcticFrameIdentity) -> tuple[np.ndarray, dict[str, Any]] | None:
        cache_key = f"{identity.subject}/{identity.sequence_name}"
        cached = self._sequence_cache.get(cache_key)
        if cached is not None:
            self._sequence_cache.move_to_end(cache_key)
            return cached
        raw_root = self.data_root / "raw_seqs" / identity.subject
        object_path = raw_root / f"{identity.sequence_name}.object.npy"
        camera_path = raw_root / f"{identity.sequence_name}.egocam.dist.npy"
        if not object_path.is_file() or not camera_path.is_file():
            return None
        object_pose = np.asarray(np.load(object_path, allow_pickle=False), dtype=np.float32)
        camera_payload = np.load(camera_path, allow_pickle=True).item()
        if object_pose.ndim != 2 or object_pose.shape[1] != 7 or not np.isfinite(object_pose).all():
            raise ValueError(f"Malformed ARCTIC object pose array: {object_path}")
        if not isinstance(camera_payload, dict):
            raise ValueError(f"Malformed ARCTIC egocamera payload: {camera_path}")
        self._sequence_cache[cache_key] = (object_pose, camera_payload)
        self._sequence_cache.move_to_end(cache_key)
        while len(self._sequence_cache) > _SEQUENCE_CACHE_LIMIT:
            self._sequence_cache.popitem(last=False)
        return self._sequence_cache[cache_key]

    @staticmethod
    def _world_to_camera(camera_payload: dict[str, Any], frame_index: int) -> np.ndarray | None:
        try:
            rotation = np.asarray(camera_payload["R_k_cam_np"], dtype=np.float32)[frame_index]
            translation = np.asarray(camera_payload["T_k_cam_np"], dtype=np.float32)[frame_index].reshape(3)
        except (IndexError, KeyError, ValueError):
            return None
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Malformed ARCTIC egocamera transform at requested frame")
        return _homogeneous(rotation, translation)

    def scene_objects(self, sequence_id: str, frame_id: str | int) -> list[dict[str, Any]]:
        """Return [] for unavailable data, otherwise two top/bottom mesh descriptors."""
        if not self._published_layout_exists():
            return []
        identity = parse_egoforce_arctic_identity(sequence_id, frame_id)
        misc = self._load_misc()
        subject_misc = misc.get(identity.subject)
        if not isinstance(subject_misc, dict) or "ioi_offset" not in subject_misc:
            return []
        raw_frame_index = identity.frame_id - int(subject_misc["ioi_offset"])
        if raw_frame_index < 0:
            return []
        sequence = self._sequence_arrays(identity)
        if sequence is None:
            return []
        object_pose, camera_payload = sequence
        if raw_frame_index >= len(object_pose):
            return []
        world_to_camera = self._world_to_camera(camera_payload, raw_frame_index)
        if world_to_camera is None:
            return []
        object_name = identity.sequence_name.split("_", 1)[0]
        templates = self.data_root / "meta" / "object_vtemplates" / object_name
        bottom = templates / "bottom.obj"
        top = templates / "top.obj"
        if not bottom.is_file() or not top.is_file():
            return []
        pose = object_pose[raw_frame_index]
        bottom_to_camera, top_to_camera = articulated_part_transforms(
            articulation_radians=float(pose[0]),
            object_axis_angle=pose[1:4],
            object_translation_mm=pose[4:7],
            world_to_camera=world_to_camera,
        )
        return [
            {
                "object_id": f"{object_name}:bottom",
                "mesh_path": str(bottom),
                "mesh_scale": 1e-3,
                "object_to_camera": bottom_to_camera,
            },
            {
                "object_id": f"{object_name}:top",
                "mesh_path": str(top),
                "mesh_scale": 1e-3,
                "object_to_camera": top_to_camera,
            },
        ]
