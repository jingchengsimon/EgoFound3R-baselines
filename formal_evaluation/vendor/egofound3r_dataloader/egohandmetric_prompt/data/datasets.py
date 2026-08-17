from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import pickle
import tarfile
from bisect import bisect_left, bisect_right
from collections import OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import get_worker_info

from egohandmetric_prompt.configs import default_data_root_path
from egohandmetric_prompt.data.base import BaseFrameDataset
from egohandmetric_prompt.data.arctic_objects import ArcticObjectResolver
from egohandmetric_prompt.data.media import prewarm_media_refs
from egohandmetric_prompt.data.schema import FrameRecord, HandAnnotation, MediaRef


_HOT3D_SEQUENCE_CACHE_LIMIT = 4
_HOT3D_RGB_CACHE_PRUNE_INTERVAL = 32
_FOREHOI_METADATA_CACHE_LIMIT = 8
_FOREHOI_FRAME_METADATA_CACHE_LIMIT = 256
_OAKINK_PREVIEW_CACHE_LIMIT = 4
_OAKINK_SEQUENCE_IMAGE_CACHE_LIMIT = 4
_OAKINK_SEQUENCE_CACHE_LIMIT = 4
_OAKINK_RUNTIME_PREVIEW_VERSION = 2
_OAKINK_EGOCENTRIC_CAMERA_ID = "104422070969"
_STERA_SESSION_CACHE_LIMIT = 4
_STERA_ROW_LOCATOR_LIMIT = 256
_STERA_SEQUENCE_ENTRY_CACHE_VERSION = 6
_REINTERHAND_JSON_CACHE_LIMIT = 512
_STERA_H5_CACHE_LIMIT = 4
_ARCTIC_H5_CACHE_LIMIT = 4
_TACO_SEQUENCE_SOURCE_CACHE_LIMIT = 4
_H2O_SEQUENCE_RUNTIME_CACHE_LIMIT = 8
_H2O_FRAME_ANNOTATION_CACHE_LIMIT = 512
_HOI4D_SEQUENCE_RUNTIME_CACHE_LIMIT = 8
_HOI4D_FRAME_ANNOTATION_CACHE_LIMIT = 512
_STERA_LEFT_ROTATION_CONVENTION = np.diag(np.array([1.0, -1.0, -1.0], dtype=np.float32))
_RECTIFICATION_MAP_CACHE_LIMIT = 32
_RECTIFICATION_MAP_CACHE: OrderedDict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = OrderedDict()
_FOREHOI_OPENGL_TO_CV = np.diag(np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32))
_FOREHOI_METADATA_TO_WORLD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_HOT3D_CW_CAMERA_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_HOI4D_OBJECT_CATEGORY_BY_CODE = {
    1: "ToyCar",
    2: "Mug",
    3: "Laptop",
    4: "StorageFurniture",
    5: "Bottle",
    6: "Safe",
    7: "Bowl",
    8: "Bucket",
    9: "Scissors",
    11: "Pliers",
    12: "Kettle",
    13: "Knife",
    14: "TrashCan",
    17: "Lamp",
    18: "Stapler",
    20: "Chair",
}
_HOI4D_RIGID_MESH_EXTENTS_CACHE: dict[str, np.ndarray | None] = {}
_HOI4D_OBJECT_PAYLOAD_UNSET = object()


def default_data_root() -> Path:
    return default_data_root_path()


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


# PSI / OpenXR order: Palm, Wrist, then each finger's metacarpal-to-tip chain.
# The project uses wrist followed by four joints for thumb, index, middle, ring, pinky.
HOLOASSIST_TO_OPENPOSE_JOINT_INDICES: tuple[int, ...] = (
    1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25,
)


def parse_holoassist_hand_row(row: Iterable[Any]) -> np.ndarray | None:
    values = list(row)
    expected_fields = 3 + (26 * 16) + 26 + 26
    if len(values) != expected_fields:
        raise ValueError(f"HoloAssist hand row must contain {expected_fields} fields, got {len(values)}")
    active = int(float(values[2])) != 0
    matrix_values = np.asarray(values[3 : 3 + (26 * 16)], dtype=np.float32).reshape(26, 4, 4)
    valid_offset = 3 + (26 * 16)
    valid = np.asarray(values[valid_offset : valid_offset + 26], dtype=np.int64).astype(bool)
    tracked = np.asarray(values[valid_offset + 26 :], dtype=np.int64).astype(bool)
    selected_indices = np.asarray(HOLOASSIST_TO_OPENPOSE_JOINT_INDICES, dtype=np.int64)
    if not active or not np.all(valid[selected_indices] & tracked[selected_indices]):
        return None
    joints = matrix_values[selected_indices, :3, 3]
    if not np.isfinite(joints).all():
        return None
    return joints.astype(np.float32)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _lru_get(cache: OrderedDict[str, Any], key: str) -> Any | None:
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
    return cached


def _lru_put(cache: OrderedDict[str, Any], key: str, value: Any, *, limit: int) -> Any:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_jsonl_dict(path: Path, key_name: str) -> dict[Any, dict[str, Any]]:
    result = {}
    for row in _load_jsonl(path):
        result[row[key_name]] = row
    return result


def _light_index_cache_path(root: Path, cache_name: str) -> Path:
    return root / ".light_index_cache" / f"{cache_name}.pkl"


@contextmanager
def _cache_file_lock(cache_path: Path) -> Iterable[None]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return 0
        return max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
    finally:
        capture.release()


def _light_index_source_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=False)
    signature: dict[str, Any] = {
        "path": str(resolved),
        "exists": path.exists(),
    }
    if signature["exists"]:
        stat = path.stat()
        signature["mtime_ns"] = stat.st_mtime_ns
        signature["size"] = stat.st_size
    return signature


def _light_index_sources_match(
    cached_source_signatures: list[dict[str, Any]],
    current_source_signatures: list[dict[str, Any]] | None = None,
) -> bool:
    if current_source_signatures is None:
        current_source_signatures = [
            _light_index_source_signature(Path(str(source_signature["path"])))
            for source_signature in cached_source_signatures
        ]
    return cached_source_signatures == current_source_signatures


def _read_light_index_cache(
    cache_path: Path,
    source_signatures: list[dict[str, Any]],
    *,
    version: int = 2,
    compatible_versions: tuple[int, ...] = (),
) -> list[dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    try:
        payload = _load_pickle(cache_path)
        record_index = payload.get("record_index")
        cached_sources = payload.get("sources")
        accepted_versions = (version, *compatible_versions)
        if (
            payload.get("version") in accepted_versions
            and isinstance(cached_sources, list)
            and _light_index_sources_match(cached_sources, source_signatures if source_signatures else None)
            and isinstance(record_index, list)
        ):
            return record_index
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, ModuleNotFoundError):
        return None
    return None


def _load_or_build_light_index_cache(
    cache_path: Path,
    source_paths: Iterable[Path],
    build_index: Callable[[], list[dict[str, Any]] | tuple[list[dict[str, Any]], list[Path]]],
    *,
    version: int = 2,
    compatible_versions: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    source_signatures = [_light_index_source_signature(path) for path in source_paths]
    cached = _read_light_index_cache(
        cache_path,
        source_signatures,
        version=version,
        compatible_versions=compatible_versions,
    )
    if cached is not None:
        return cached
    with _cache_file_lock(cache_path):
        cached = _read_light_index_cache(
            cache_path,
            source_signatures,
            version=version,
            compatible_versions=compatible_versions,
        )
        if cached is not None:
            return cached
        build_output = build_index()
        if isinstance(build_output, tuple):
            record_index, built_source_paths = build_output
            source_signatures = [_light_index_source_signature(path) for path in built_source_paths]
        else:
            record_index = build_output
        with cache_path.open("wb") as handle:
            pickle.dump(
                {"version": version, "sources": source_signatures, "record_index": record_index},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return record_index


def _read_sequence_entry_cache(
    cache_path: Path,
    source_signatures: list[dict[str, Any]],
    *,
    cached_source_paths_builder: Callable[[list[dict[str, Any]]], Iterable[Path]] | None,
    version: int,
) -> list[dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    try:
        payload = _load_pickle(cache_path)
        sequence_entries = payload.get("sequence_entries")
        cached_sources = payload.get("sources")
        expected_source_signatures = source_signatures
        if cached_source_paths_builder is not None and isinstance(sequence_entries, list):
            expected_source_signatures = [
                _light_index_source_signature(path)
                for path in cached_source_paths_builder(sequence_entries)
            ]
        if (
            payload.get("version") == version
            and isinstance(cached_sources, list)
            and cached_sources == expected_source_signatures
            and isinstance(sequence_entries, list)
        ):
            return sequence_entries
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, ModuleNotFoundError):
        return None
    return None


def _load_or_build_sequence_entry_cache(
    cache_path: Path,
    source_paths: Iterable[Path],
    build_entries: Callable[[], list[dict[str, Any]]],
    *,
    cached_source_paths_builder: Callable[[list[dict[str, Any]]], Iterable[Path]] | None = None,
    version: int = 3,
) -> list[dict[str, Any]]:
    source_signatures = [_light_index_source_signature(path) for path in source_paths]
    cached = _read_sequence_entry_cache(
        cache_path,
        source_signatures,
        cached_source_paths_builder=cached_source_paths_builder,
        version=version,
    )
    if cached is not None:
        return cached
    with _cache_file_lock(cache_path):
        cached = _read_sequence_entry_cache(
            cache_path,
            source_signatures,
            cached_source_paths_builder=cached_source_paths_builder,
            version=version,
        )
        if cached is not None:
            return cached
        sequence_entries = build_entries()
        if cached_source_paths_builder is not None:
            source_signatures = [
                _light_index_source_signature(path)
                for path in cached_source_paths_builder(sequence_entries)
            ]
        with cache_path.open("wb") as handle:
            pickle.dump(
                {"version": version, "sources": source_signatures, "sequence_entries": sequence_entries},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return sequence_entries


def _load_video_frame_count_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        payload = _load_pickle(cache_path)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, ModuleNotFoundError):
        return {}
    entries = payload.get("entries")
    if payload.get("version") != 1 or not isinstance(entries, dict):
        return {}
    return entries


def _load_or_build_video_frame_counts(cache_path: Path, video_paths: list[Path]) -> dict[Path, int]:
    with _cache_file_lock(cache_path):
        entries = _load_video_frame_count_cache(cache_path)
        frame_counts: dict[Path, int] = {}
        missing: list[tuple[Path, dict[str, Any]]] = []
        for path in video_paths:
            signature = _light_index_source_signature(path)
            cached = entries.get(str(signature["path"]))
            if isinstance(cached, dict) and cached.get("source") == signature:
                frame_counts[path] = int(cached.get("frame_count", 0))
            else:
                missing.append((path, signature))
        if missing:
            with ThreadPoolExecutor(max_workers=min(32, len(missing))) as executor:
                counts = list(executor.map(lambda item: _video_frame_count(item[0]), missing))
            for (path, signature), count in zip(missing, counts, strict=True):
                frame_count = int(count)
                entries[str(signature["path"])] = {"source": signature, "frame_count": frame_count}
                frame_counts[path] = frame_count
            with cache_path.open("wb") as handle:
                pickle.dump({"version": 1, "entries": entries}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return frame_counts


def _decode_if_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _intrinsics_from_focal_principal(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[0, 0] = fx
    matrix[1, 1] = fy
    matrix[0, 2] = cx
    matrix[1, 2] = cy
    return matrix


def _intrinsics_from_horizontal_fov(width: int, height: int, camera_angle_x: float) -> np.ndarray:
    focal = 0.5 * float(width) / math.tan(0.5 * float(camera_angle_x))
    return _intrinsics_from_focal_principal(focal, focal, float(width) / 2.0, float(height) / 2.0)


def _scaled_intrinsics(intrinsics: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = intrinsics.astype(np.float32).copy()
    scaled[0, 0] *= float(scale_x)
    scaled[0, 2] *= float(scale_x)
    scaled[1, 1] *= float(scale_y)
    scaled[1, 2] *= float(scale_y)
    return scaled


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _has_distortion(distortion: np.ndarray) -> bool:
    return bool(np.isfinite(distortion).all() and not np.allclose(distortion, 0.0))


def _fisheye_rectified_intrinsics(
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    if not _has_distortion(distortion):
        return intrinsics.astype(np.float32).copy()
    rectified = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsics.astype(np.float64),
        distortion.astype(np.float64).reshape(4, 1),
        image_size,
        np.eye(3, dtype=np.float64),
        balance=0.0,
        new_size=image_size,
    )
    return rectified.astype(np.float32)

def _rectification_maps(
    kind: str,
    raw_intrinsics: np.ndarray,
    distortion: np.ndarray,
    rectified_intrinsics: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    key = (
        kind, int(width), int(height),
        np.ascontiguousarray(raw_intrinsics, dtype=np.float64).tobytes(),
        np.ascontiguousarray(distortion, dtype=np.float64).reshape(-1).tobytes(),
        np.ascontiguousarray(rectified_intrinsics, dtype=np.float64).tobytes(),
    )
    cached = _lru_get(_RECTIFICATION_MAP_CACHE, key)
    if cached is not None:
        return cached
    raw = raw_intrinsics.astype(np.float64)
    dist = distortion.astype(np.float64).reshape(-1, 1)
    rectified = rectified_intrinsics.astype(np.float64)
    if kind == "fisheye":
        maps = cv2.fisheye.initUndistortRectifyMap(
            raw, dist, np.eye(3, dtype=np.float64), rectified, (width, height), cv2.CV_16SC2
        )
    elif kind == "pinhole":
        # Match cv2.undistort's fixed-point remap path exactly while retaining the maps.
        maps = cv2.initUndistortRectifyMap(raw, dist, None, rectified, (width, height), cv2.CV_16SC2)
    else:
        raise ValueError(f"unsupported rectification kind: {kind}")
    return _lru_put(_RECTIFICATION_MAP_CACHE, key, maps, limit=_RECTIFICATION_MAP_CACHE_LIMIT)



def _undistort_fisheye_rgb(
    rgb: torch.Tensor,
    raw_intrinsics: torch.Tensor,
    distortion: torch.Tensor,
    rectified_intrinsics: torch.Tensor,
) -> torch.Tensor:
    if not _has_distortion(distortion.detach().cpu().numpy()):
        return rgb
    image_rgb = (
        rgb.detach()
        .cpu()
        .to(dtype=torch.float32)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    map_one, map_two = _rectification_maps("fisheye", raw_intrinsics.detach().cpu().numpy(), distortion.detach().cpu().numpy(), rectified_intrinsics.detach().cpu().numpy(), (image_rgb.shape[1], image_rgb.shape[0]))
    rectified = cv2.remap(image_rgb, map_one, map_two, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(rectified.transpose(2, 0, 1).copy()).float() / 255.0


def _pinhole_rectified_intrinsics(
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Return the pinhole K paired with an OpenCV rational-lens undistortion."""
    if not _has_distortion(distortion):
        return intrinsics.astype(np.float32).copy()
    rectified, _ = cv2.getOptimalNewCameraMatrix(
        intrinsics.astype(np.float64),
        distortion.astype(np.float64).reshape(-1, 1),
        image_size,
        alpha=0.0,
        newImgSize=image_size,
    )
    return rectified.astype(np.float32)


def _undistort_pinhole_rgb(
    rgb: torch.Tensor,
    raw_intrinsics: torch.Tensor,
    distortion: torch.Tensor,
    rectified_intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Lazily rectify a standard OpenCV rational-tangential RGB image."""
    if not _has_distortion(distortion.detach().cpu().numpy()):
        return rgb
    image_rgb = (
        rgb.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    map_one, map_two = _rectification_maps("pinhole", raw_intrinsics.detach().cpu().numpy(), distortion.detach().cpu().numpy(), rectified_intrinsics.detach().cpu().numpy(), (image_rgb.shape[1], image_rgb.shape[0]))
    rectified = cv2.remap(image_rgb, map_one, map_two, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(rectified.transpose(2, 0, 1).copy()).float() / 255.0


def _arctic_rgb_media_ref(
    extracted_path: Path,
    tar_path: Path | None,
    member_name: str,
) -> MediaRef | None:
    """Prefer an extracted ARCTIC JPEG while extraction is being resumed."""
    if extracted_path.is_file():
        return MediaRef(kind="path", path=str(extracted_path))
    if tar_path is not None:
        return MediaRef(kind="tar_member", path=str(tar_path), member=member_name)
    return None


def _should_skip_arctic_exposure_warmup_frame(frame_id: str) -> bool:
    """Drop ARCTIC's two camera auto-exposure warmup frames per sequence."""
    try:
        return int(frame_id) <= 2
    except ValueError:
        return False


def _undistort_pinhole_points_2d(
    points_2d: np.ndarray,
    raw_intrinsics: np.ndarray,
    distortion: np.ndarray,
    rectified_intrinsics: np.ndarray,
) -> np.ndarray:
    """Map distorted OpenCV image points into the matching rectified image."""
    points = np.asarray(points_2d, dtype=np.float32).reshape(-1, 2)
    rectified = points.copy()
    valid = np.isfinite(points).all(axis=1)
    if not bool(valid.any()) or not _has_distortion(distortion):
        return rectified
    corrected = cv2.undistortPoints(
        points[valid].astype(np.float64).reshape(-1, 1, 2),
        raw_intrinsics.astype(np.float64),
        distortion.astype(np.float64).reshape(-1, 1),
        P=rectified_intrinsics.astype(np.float64),
    )
    rectified[valid] = corrected.reshape(-1, 2).astype(np.float32)
    return rectified


def _pose_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.astype(np.float32)
    pose[:3, 3] = translation.astype(np.float32)
    return pose


def _invert_pose(pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(pose, dtype=np.float32)).astype(np.float32)


_STERA_OPTICAL_TO_LINK = np.array(
    ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    dtype=np.float32,
)


def _stera_optical_world_to_camera(
    rotation_world: np.ndarray,
    translation_world: np.ndarray,
) -> np.ndarray:
    """Convert Stera camera-link C2W metadata to optical-frame W2C."""
    rotation_world = np.asarray(rotation_world, dtype=np.float32).reshape(3, 3)
    translation_world = np.asarray(translation_world, dtype=np.float32).reshape(3)
    rotation_camera_world = _STERA_OPTICAL_TO_LINK.T @ rotation_world.T
    translation_camera_world = -rotation_camera_world @ translation_world
    return _pose_from_rotation_translation(rotation_camera_world, translation_camera_world)


def _quat_wxyz_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _pose_from_quat_translation(quaternion_wxyz: np.ndarray, translation_xyz: np.ndarray) -> np.ndarray:
    return _pose_from_rotation_translation(_quat_wxyz_to_matrix(quaternion_wxyz), np.asarray(translation_xyz, dtype=np.float32))


def _hot3d_scene_objects(
    dataset_root: Path,
    object_rows: Iterable[dict[str, Any]],
    camera_from_world: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Describe available HOT3D object meshes in the rectified camera frame."""
    if camera_from_world is None:
        return []
    camera_from_world = np.asarray(camera_from_world, dtype=np.float32)
    if camera_from_world.shape != (4, 4) or not np.isfinite(camera_from_world).all():
        return []
    descriptors: list[dict[str, Any]] = []
    for object_row in object_rows:
        object_uid = str(object_row["object_uid"])
        world_from_object = np.asarray(object_row["world_from_object"], dtype=np.float32)
        mesh_path = dataset_root / "assets" / f"{object_uid}.glb"
        if (
            world_from_object.shape != (4, 4)
            or not np.isfinite(world_from_object).all()
            or not mesh_path.is_file()
        ):
            continue
        descriptors.append(
            {
                "object_id": object_uid,
                "mesh_path": str(mesh_path),
                "object_to_camera": (camera_from_world @ world_from_object).astype(np.float32),
            }
        )
    return descriptors


def _oakink_mocap_frame_for_rgb_frame(
    rgb_frame_id: int,
    mocap_frame_ids: Iterable[Any],
) -> int:
    """Return the mocap key synchronized with an OakInk RGB frame.

    Released OakInk-v2 RGB ids are one-based, while full-rate mocap ids are
    zero-based. RGB is frequently downsampled, so its position in
    ``frame_id_list`` is never a valid mocap-time index. Preserve direct-key
    compatibility for layouts whose mocap ids are not zero-based.
    """
    frame_id = int(rgb_frame_id)
    mocap_ids = {int(value) for value in mocap_frame_ids}
    zero_based_key = frame_id - 1
    if 0 in mocap_ids and zero_based_key in mocap_ids:
        return zero_based_key
    if frame_id in mocap_ids:
        return frame_id
    raise KeyError(f"OakInk mocap frame missing for RGB frame {frame_id}")


def _oakink_scene_objects(
    dataset_root: Path,
    object_ids: Iterable[Any],
    object_transforms: dict[Any, Any],
    mocap_frame_id: int,
    camera_from_world: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Return an all-or-nothing OakInk object scene in camera coordinates."""
    if camera_from_world is None:
        return []
    camera_from_world = np.asarray(camera_from_world, dtype=np.float32)
    if (
        camera_from_world.shape != (4, 4)
        or not np.isfinite(camera_from_world).all()
        or not isinstance(object_transforms, dict)
    ):
        return []
    descriptors: list[dict[str, Any]] = []
    for object_id_value in object_ids:
        object_id = str(object_id_value)
        per_frame_transforms = object_transforms.get(object_id_value)
        if per_frame_transforms is None:
            per_frame_transforms = object_transforms.get(object_id)
        if not isinstance(per_frame_transforms, dict):
            return []
        world_from_object = per_frame_transforms.get(mocap_frame_id)
        if world_from_object is None:
            world_from_object = per_frame_transforms.get(str(mocap_frame_id))
        world_from_object = np.asarray(world_from_object, dtype=np.float32) if world_from_object is not None else None
        mesh_path = dataset_root / "object_affordance" / "affordance_part" / object_id / "model.obj"
        if not mesh_path.is_file():
            mesh_path = dataset_root / "object_repair" / "align_ds" / object_id / "model.obj"
        if (
            world_from_object is None
            or world_from_object.shape != (4, 4)
            or not np.isfinite(world_from_object).all()
            or not mesh_path.is_file()
        ):
            return []
        descriptors.append(
            {
                "object_id": object_id,
                "mesh_path": str(mesh_path),
                "object_to_camera": (camera_from_world @ world_from_object).astype(np.float32),
            }
        )
    return descriptors


def _hoi4d_rigid_mesh_extents(mesh_path: Path) -> np.ndarray | None:
    cache_key = str(mesh_path.resolve(strict=False))
    if cache_key in _HOI4D_RIGID_MESH_EXTENTS_CACHE:
        return _HOI4D_RIGID_MESH_EXTENTS_CACHE[cache_key]
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    try:
        with mesh_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                values = line.split()
                if len(values) != 4 or values[0] != "v":
                    continue
                vertex = np.asarray(values[1:], dtype=np.float64)
                if not np.isfinite(vertex).all():
                    continue
                minimum = np.minimum(minimum, vertex)
                maximum = np.maximum(maximum, vertex)
    except (OSError, ValueError):
        extents = None
    else:
        candidate = maximum - minimum
        extents = candidate.astype(np.float32) if np.isfinite(candidate).all() and bool((candidate > 0.0).all()) else None
    _HOI4D_RIGID_MESH_EXTENTS_CACHE[cache_key] = extents
    return extents


def _hoi4d_sequence_category_and_instance(sequence_id: str) -> tuple[str, int] | None:
    category_code = None
    instance_id = None
    for part in Path(sequence_id).parts:
        if part.startswith("C") and part[1:].isdigit():
            category_code = int(part[1:])
        elif part.startswith("N") and part[1:].isdigit():
            instance_id = int(part[1:])
    category = None if category_code is None else _HOI4D_OBJECT_CATEGORY_BY_CODE.get(category_code)
    if category is None or instance_id is None:
        return None
    return category, instance_id


def _hoi4d_intrinsic_xyz_rotation_matrix(euler_xyz: np.ndarray) -> np.ndarray | None:
    angles = np.asarray(euler_xyz, dtype=np.float64)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        return None
    cos_x, cos_y, cos_z = np.cos(angles)
    sin_x, sin_y, sin_z = np.sin(angles)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]])
    rotation_y = np.array([[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]])
    rotation_z = np.array([[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]])
    return (rotation_x @ rotation_y @ rotation_z).astype(np.float32)


def _hoi4d_object_pose_path(dataset_root: Path, sequence_id: str, frame_id: int) -> Path | None:
    pose_dir = dataset_root / "HOI4D_annotations" / sequence_id / "objpose"
    pose_path = pose_dir / f"{frame_id:05d}.json"
    if pose_path.is_file():
        return pose_path
    fallback_path = pose_dir / f"{frame_id}.json"
    return fallback_path if fallback_path.is_file() else None


def _hoi4d_scene_objects(
    dataset_root: Path,
    sequence_id: str,
    *,
    frame_id: int,
    object_pose_payload: Any = _HOI4D_OBJECT_PAYLOAD_UNSET,
) -> list[dict[str, Any]]:
    """Return only rigid HOI4D objects whose CAD geometry is numerically verified."""
    identity = _hoi4d_sequence_category_and_instance(sequence_id)
    if identity is None:
        return []
    category, instance_id = identity
    mesh_path = dataset_root / "HOI4D_CAD_Model_for_release" / "rigid" / category / f"{instance_id:03d}.obj"
    pose_path = _hoi4d_object_pose_path(dataset_root, sequence_id, frame_id)
    if not mesh_path.is_file() or pose_path is None:
        return []
    if object_pose_payload is _HOI4D_OBJECT_PAYLOAD_UNSET:
        try:
            payload = _load_json(pose_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return []
    else:
        payload = object_pose_payload
    if not isinstance(payload, dict):
        return []
    if not bool(payload.get("isEffective", False)):
        return []
    annotations = payload.get("dataList", payload.get("objects", []))
    if not isinstance(annotations, list):
        return []
    expected_label = category.replace("_", "").lower()
    annotation = next(
        (
            candidate
            for candidate in annotations
            if isinstance(candidate, dict)
            and str(candidate.get("label", "")).replace("_", "").lower() == expected_label
        ),
        None,
    )
    if annotation is None:
        return []
    center = annotation.get("center")
    rotation = annotation.get("rotation")
    dimensions = annotation.get("dimensions")
    if not isinstance(center, dict) or not isinstance(rotation, dict) or not isinstance(dimensions, dict):
        return []
    try:
        translation = np.asarray([center[axis] for axis in ("x", "y", "z")], dtype=np.float32)
        euler_xyz = np.asarray([rotation[axis] for axis in ("x", "y", "z")], dtype=np.float64)
        expected_extents = np.asarray(
            [dimensions["length"], dimensions["width"], dimensions["height"]], dtype=np.float32
        )
    except (KeyError, TypeError, ValueError):
        return []
    rotation_matrix = _hoi4d_intrinsic_xyz_rotation_matrix(euler_xyz)
    mesh_extents = _hoi4d_rigid_mesh_extents(mesh_path)
    if (
        mesh_extents is None
        or not np.isfinite(translation).all()
        or rotation_matrix is None
        or not np.isfinite(rotation_matrix).all()
        or not np.isfinite(expected_extents).all()
        or not bool((expected_extents > 0.0).all())
        or not np.allclose(mesh_extents, expected_extents, rtol=0.15, atol=0.005)
    ):
        return []
    object_to_camera = np.eye(4, dtype=np.float32)
    object_to_camera[:3, :3] = rotation_matrix
    object_to_camera[:3, 3] = translation
    return [
        {
            "object_id": f"{category}/{instance_id:03d}",
            "mesh_path": str(mesh_path),
            "object_to_camera": object_to_camera,
        }
    ]


def _hot3d_camera_frame_wrist_transform(
    camera_from_world: np.ndarray | None,
    world_from_wrist: np.ndarray,
) -> np.ndarray | None:
    """Convert HOT3D's world-frame wrist pose into the rectified camera frame."""
    if camera_from_world is None:
        return None
    camera_from_world = np.asarray(camera_from_world, dtype=np.float32)
    world_from_wrist = np.asarray(world_from_wrist, dtype=np.float32)
    if (
        camera_from_world.shape != (4, 4)
        or world_from_wrist.shape != (4, 4)
        or not np.isfinite(camera_from_world).all()
        or not np.isfinite(world_from_wrist).all()
    ):
        return None
    return (camera_from_world @ world_from_wrist).astype(np.float32)


def _hot3d_camera_model_with_online_calibration(
    static_camera_model: dict[str, Any],
    online_camera_models: list[tuple[int, dict[str, Any]]],
    device_timestamp_ns: int | None,
) -> dict[str, Any]:
    """Use the closest Aria online calibration while preserving recorded image size."""
    resolved = dict(static_camera_model)
    if not online_camera_models or device_timestamp_ns is None:
        return resolved
    timestamps = [item[0] for item in online_camera_models]
    index = bisect_left(timestamps, int(device_timestamp_ns))
    candidates = online_camera_models[max(index - 1, 0):min(index + 1, len(online_camera_models))]
    if not candidates:
        return resolved
    _, online = min(candidates, key=lambda item: abs(item[0] - int(device_timestamp_ns)))
    quaternion = np.asarray(online.get("quaternion_wxyz"), dtype=np.float32)
    translation = np.asarray(online.get("translation_xyz"), dtype=np.float32)
    projection_params = np.asarray(online.get("projectionParams"), dtype=np.float32)
    if quaternion.shape != (4,) or translation.shape != (3,) or projection_params.size < 3:
        return resolved
    if not (np.isfinite(quaternion).all() and np.isfinite(translation).all() and np.isfinite(projection_params).all()):
        return resolved
    resolved["T_Device_Camera"] = {
        "quaternion_wxyz": quaternion.tolist(),
        "translation_xyz": translation.tolist(),
    }
    resolved["projectionParams"] = projection_params.tolist()
    return resolved


def _matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    axis_angle, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float32).reshape(3, 3))
    return axis_angle.reshape(3).astype(np.float32)


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float32).reshape(3))
    return rotation.astype(np.float32)


def _quat16_to_axis_angle_full(quat16: np.ndarray, *, root_rotation: np.ndarray | None = None) -> np.ndarray:
    quat_array = np.asarray(quat16, dtype=np.float32).reshape(16, 4)
    rotvecs = []
    for index, quaternion in enumerate(quat_array):
        rotation = _quat_wxyz_to_matrix(quaternion)
        if index == 0 and root_rotation is not None:
            rotation = np.asarray(root_rotation, dtype=np.float32).reshape(3, 3) @ rotation
        rotvecs.append(_matrix_to_axis_angle(rotation))
    return np.concatenate(rotvecs, axis=0).astype(np.float32)


def _hot3d_projection_params(camera_model: dict[str, Any]) -> np.ndarray:
    params = np.asarray(camera_model["projectionParams"], dtype=np.float64).reshape(-1)
    if params.shape[0] < 15:
        params = np.pad(params, (0, 15 - params.shape[0]))
    return params


def _hot3d_camera_calibration(camera_model: dict[str, Any]):
    from projectaria_tools.core import calibration, sophus

    transform = camera_model.get("T_Device_Camera", {})
    quaternion = transform.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
    translation = transform.get("translation_xyz", [0.0, 0.0, 0.0])
    t_device_camera = sophus.SE3.from_quat_and_translation(
        float(quaternion[0]),
        np.asarray(quaternion[1:], dtype=np.float64),
        np.asarray(translation, dtype=np.float64),
    )
    return calibration.CameraCalibration(
        str(camera_model.get("label", "camera-rgb")),
        calibration.CameraModelType.FISHEYE624,
        _hot3d_projection_params(camera_model),
        t_device_camera,
        int(camera_model["imageWidth"]),
        int(camera_model["imageHeight"]),
        None,
        float(camera_model.get("maxSolidAngle", np.pi)),
        str(camera_model.get("serialNumber", "")),
    )


def _hot3d_linear_calibration(camera_model: dict[str, Any]):
    from projectaria_tools.core import calibration

    params = _hot3d_projection_params(camera_model)
    return calibration.get_linear_camera_calibration(
        int(camera_model["imageWidth"]),
        int(camera_model["imageHeight"]),
        float(params[0]),
        str(camera_model.get("label", "camera-rgb-linear")),
    )


def _hot3d_camera_model_cache_key(camera_model: dict[str, Any]) -> str:
    """Canonicalize the per-frame online calibration for a process-local cache."""
    return json.dumps(camera_model, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=128)
def _hot3d_rectification_calibrations(camera_model_key: str):
    camera_model = json.loads(camera_model_key)
    return _hot3d_linear_calibration(camera_model), _hot3d_camera_calibration(camera_model)


def _hot3d_rectified_rgb_cache_path(
    cache_root: str | Path,
    *,
    sequence_id: str,
    timestamp_ns: int,
    worker_slot: int,
) -> Path:
    return Path(cache_root) / f"worker_{int(worker_slot):02d}" / str(sequence_id) / f"{int(timestamp_ns)}.png"


def _write_hot3d_rectified_rgb_cache(path: Path, rgb: torch.Tensor) -> None:
    """Atomically publish the final rectified RGB in a lossless, reusable form."""
    image = (
        rgb.detach()
        .cpu()
        .to(dtype=torch.float32)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        ok = cv2.imwrite(
            str(temporary_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        if not ok:
            raise OSError(f"无法写入 HOT3D rectified RGB 缓存：{temporary_path}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _prune_hot3d_rectified_rgb_cache(worker_root: Path, max_bytes: int) -> None:
    """Bound one worker's node-local cache without cross-worker coordination."""
    files = [path for path in worker_root.rglob("*.png") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes <= int(max_bytes):
        return
    for path in sorted(files, key=lambda candidate: candidate.stat().st_mtime_ns):
        if total_bytes <= int(max_bytes):
            break
        try:
            total_bytes -= path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue


def _intrinsics_from_projectaria_linear(camera_calibration: Any) -> np.ndarray:
    fx, fy, cx, cy = np.asarray(camera_calibration.get_projection_params(), dtype=np.float32)[:4]
    return _intrinsics_from_focal_principal(float(fx), float(fy), float(cx), float(cy))


def _egoforce_hot3d_camera_model(fisheye_params: dict[str, Any], rgb_camera_params: dict[str, Any]) -> dict[str, Any] | None:
    projection_params = fisheye_params.get("projection_params")
    if projection_params is None:
        focal_length = fisheye_params.get("focal_length")
        principal_point = fisheye_params.get("principal_point")
        if focal_length is None or principal_point is None:
            return None
        projection_params = [float(focal_length[0]), float(principal_point[0]), float(principal_point[1])] + [0.0] * 12
    image_width = int(rgb_camera_params.get("image_width", rgb_camera_params.get("imageWidth", rgb_camera_params.get("width", 1408))))
    image_height = int(
        rgb_camera_params.get("image_height", rgb_camera_params.get("imageHeight", rgb_camera_params.get("height", image_width)))
    )
    return {
        "label": "camera-rgb",
        "projectionParams": projection_params,
        "imageWidth": image_width,
        "imageHeight": image_height,
        "T_Device_Camera": {
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_xyz": [0.0, 0.0, 0.0],
        },
        "maxSolidAngle": float(np.pi),
        "serialNumber": "",
    }


def _hot3d_rotated_linear_intrinsics(camera_model: dict[str, Any]) -> np.ndarray:
    from projectaria_tools.core import calibration

    return _intrinsics_from_projectaria_linear(calibration.rotate_camera_calib_cw90deg(_hot3d_linear_calibration(camera_model)))


def _rectify_hot3d_rgb(rgb: torch.Tensor, camera_model: dict[str, Any]) -> torch.Tensor:
    from projectaria_tools.core import calibration

    image_rgb = (
        rgb.detach()
        .cpu()
        .to(dtype=torch.float32)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    dst_calib = _hot3d_linear_calibration(camera_model)
    src_calib = _hot3d_camera_calibration(camera_model)
    rectified = np.stack(
        [
            calibration.distort_by_calibration(image_rgb[:, :, channel], dst_calib, src_calib)
            for channel in range(image_rgb.shape[2])
        ],
        axis=2,
    )
    return torch.from_numpy(rectified.transpose(2, 0, 1).copy()).float() / 255.0


def _rectify_rotate_hot3d_aria_rgb(rgb: torch.Tensor, camera_model: dict[str, Any]) -> torch.Tensor:
    from projectaria_tools.core import calibration

    image_rgb = (
        rgb.detach()
        .cpu()
        .to(dtype=torch.float32)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    dst_calib, src_calib = _hot3d_rectification_calibrations(
        _hot3d_camera_model_cache_key(camera_model)
    )
    # projectaria's three-channel overload is not numerically equivalent to
    # the official per-channel path.  Keep the calibrated reference result.
    rectified = np.stack(
        [
            calibration.distort_by_calibration(image_rgb[:, :, channel], dst_calib, src_calib)
            for channel in range(image_rgb.shape[2])
        ],
        axis=2,
    )
    rotated = np.rot90(rectified, k=-1).copy()
    return torch.from_numpy(rotated.transpose(2, 0, 1).copy()).float() / 255.0


def _rectify_hot3d_points_2d(points_xy: np.ndarray, camera_model: dict[str, Any]) -> np.ndarray:
    src_calib = _hot3d_camera_calibration(camera_model)
    dst_calib = _hot3d_linear_calibration(camera_model)
    outputs = []
    for point in np.asarray(points_xy, dtype=np.float64).reshape(-1, 2):
        if not np.isfinite(point).all():
            outputs.append([np.nan, np.nan])
            continue
        ray = src_calib.unproject(point)
        if ray is None:
            outputs.append([np.nan, np.nan])
            continue
        projected = dst_calib.project(ray)
        if projected is None:
            outputs.append([np.nan, np.nan])
            continue
        outputs.append([float(projected[0]), float(projected[1])])
    return np.asarray(outputs, dtype=np.float32)


def _rectify_rotate_hot3d_points_2d(points_xy: np.ndarray, camera_model: dict[str, Any]) -> np.ndarray:
    src_calib = _hot3d_camera_calibration(camera_model)
    dst_calib = _hot3d_linear_calibration(camera_model)
    height = int(camera_model["imageHeight"])
    outputs = []
    for point in np.asarray(points_xy, dtype=np.float64).reshape(-1, 2):
        if not np.isfinite(point).all():
            outputs.append([np.nan, np.nan])
            continue
        ray = src_calib.unproject(point)
        if ray is None:
            outputs.append([np.nan, np.nan])
            continue
        projected = dst_calib.project(ray)
        if projected is None:
            outputs.append([np.nan, np.nan])
            continue
        outputs.append([float(height - 1 - projected[1]), float(projected[0])])
    return np.asarray(outputs, dtype=np.float32)


def _project_points(joints_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(joints_3d[:, 2], 1e-6, None)
    x = intrinsics[0, 0] * joints_3d[:, 0] / z + intrinsics[0, 2]
    y = intrinsics[1, 1] * joints_3d[:, 1] / z + intrinsics[1, 2]
    return np.stack([x, y], axis=1).astype(np.float32)


def _bbox_from_points(points_2d: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(points_2d[:, 0].min()),
            float(points_2d[:, 1].min()),
            float(points_2d[:, 0].max()),
            float(points_2d[:, 1].max()),
        ],
        dtype=np.float32,
    )


def _bbox_from_finite_points(points_2d: np.ndarray) -> np.ndarray | None:
    finite_points = np.asarray(points_2d, dtype=np.float32)
    finite_points = finite_points[np.isfinite(finite_points).all(axis=1)]
    if finite_points.shape[0] == 0:
        return None
    return _bbox_from_points(finite_points)


def _count_sides(hand_annos: list[HandAnnotation]) -> tuple[int, int]:
    left_count = sum(1 for hand in hand_annos if hand.side == "left")
    right_count = sum(1 for hand in hand_annos if hand.side == "right")
    return left_count, right_count


def _hand_has_mano_signal(hand: HandAnnotation) -> bool:
    return any(
        value is not None
        for value in (
            hand.mano_pose,
            hand.mano_global_orient,
            hand.mano_hand_pose,
        )
    )


def _any_hand_field(hand_annos: list[HandAnnotation], field_name: str) -> bool:
    return any(getattr(hand, field_name) is not None for hand in hand_annos)


def _hand_supervision_flags(hand_annos: list[HandAnnotation]) -> dict[str, bool]:
    has_mano = any(_hand_has_mano_signal(hand) for hand in hand_annos)
    has_joint_3d_gt = _any_hand_field(hand_annos, "joints_3d")
    return {
        "has_mano": has_mano,
        "has_3d_joints": has_joint_3d_gt,
        "has_bbox_gt": _any_hand_field(hand_annos, "bbox_xyxy"),
        "has_joint_3d_gt": has_joint_3d_gt,
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_optional_array(value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if shape is not None and tuple(array.shape) != shape:
        return None
    return array


def _three_r_supervision_flags(
    *,
    depth_ref: MediaRef | None,
    intrinsics: np.ndarray | None,
    camera_pose: np.ndarray | None,
) -> dict[str, bool]:
    has_depth_gt = depth_ref is not None
    has_intrinsics_gt = intrinsics is not None
    has_camera_pose_gt = camera_pose is not None
    return {
        "has_depth_gt": has_depth_gt,
        "has_intrinsics_gt": has_intrinsics_gt,
        "has_camera_pose_gt": has_camera_pose_gt,
        "has_3r_gt": has_depth_gt or has_intrinsics_gt or has_camera_pose_gt,
    }


def _stage_membership_from_text_files(root: Path, prefix: str = "pose") -> dict[str, str]:
    membership: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = root / f"{prefix}_{split}.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item:
                membership[item] = split
    return membership


def _parse_h2o_intrinsics(path: Path) -> tuple[np.ndarray, tuple[int, int] | None]:
    values = [float(item) for item in path.read_text(encoding="utf-8").split()]
    intrinsics = _intrinsics_from_focal_principal(values[0], values[1], values[2], values[3])
    native_size = (int(values[4]), int(values[5])) if len(values) >= 6 else None
    return intrinsics, native_size


def _parse_matrix_txt(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float32)
    return values.reshape(shape)


def _parse_open3d_camera_trajectory_log(path: Path) -> list[np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 5 != 0:
        raise ValueError(f"Open3D camera trajectory log must contain 5 lines per entry: {path}")
    camera_poses: list[np.ndarray] = []
    for offset in range(0, len(lines), 5):
        raw_pose = np.asarray(
            [[float(item) for item in row.split()] for row in lines[offset + 1 : offset + 5]],
            dtype=np.float32,
        )
        if raw_pose.shape != (4, 4):
            raise ValueError(f"Open3D camera trajectory pose must be 4x4: {path}")
        camera_poses.append(raw_pose)
    return camera_poses


def _parse_h2o_hand_pose_joints(path: Path) -> list[np.ndarray | None]:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float32)
    outputs: list[np.ndarray | None] = []
    offset = 0
    for _ in range(2):
        offset += 1
        joints = values[offset : offset + 63].reshape(21, 3)
        offset += 63
        outputs.append(joints if np.isfinite(joints).all() and float(np.linalg.norm(joints)) > 1e-6 else None)
    return outputs


def _parse_h2o_hand_pose_mano(path: Path) -> list[dict[str, np.ndarray] | None]:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float32)
    outputs: list[dict[str, np.ndarray] | None] = []
    offset = 0
    for _ in range(2):
        offset += 1
        trans = values[offset : offset + 3]
        offset += 3
        pose = values[offset : offset + 48]
        offset += 48
        betas = values[offset : offset + 10]
        offset += 10
        if np.isfinite(np.concatenate((trans, pose, betas))).all() and float(np.linalg.norm(trans)) > 1e-6:
            outputs.append(
                {
                    "trans": trans,
                    "global_orient": pose[:3],
                    "hand_pose": pose[3:],
                    "pose": pose,
                    "betas": betas,
                }
            )
        else:
            outputs.append(None)
    return outputs


def _mano_hand(side: str, mano_info: dict[str, np.ndarray] | None, joints_3d: np.ndarray | None, intrinsics: np.ndarray | None) -> HandAnnotation | None:
    if mano_info is None and joints_3d is None:
        return None
    joints_2d = _project_points(joints_3d, intrinsics) if joints_3d is not None and intrinsics is not None else None
    bbox = _bbox_from_points(joints_2d) if joints_2d is not None else None
    return HandAnnotation(
        side=side,
        visible=True,
        bbox_xyxy=bbox,
        joints_3d=joints_3d,
        joints_2d=joints_2d,
        mano_global_orient=None if mano_info is None else mano_info["global_orient"],
        mano_hand_pose=None if mano_info is None else mano_info["hand_pose"],
        mano_pose=None if mano_info is None else mano_info["pose"],
        mano_pose_format="axis_angle_full",
        mano_betas=None if mano_info is None else mano_info["betas"],
        mano_trans=None if mano_info is None else mano_info["trans"],
    )


def _sequence_entry_for_index(sequence_offsets: list[int], sequence_entries: list[dict[str, Any]], index: int) -> tuple[dict[str, Any], int]:
    sequence_index = bisect_right(sequence_offsets, index) - 1
    sequence_entry = sequence_entries[sequence_index]
    local_index = index - sequence_offsets[sequence_index]
    return sequence_entry, local_index

# Server H2O helpers retained during dataset migration

import ast
import pickle
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from egohandmetric_prompt.configs import default_data_root_path
from egohandmetric_prompt.data.base import BaseFrameDataset
from egohandmetric_prompt.data.marker_vertices import MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195
from egohandmetric_prompt.data.schema import FrameRecord, HandAnnotation, MediaRef


H2O_OPENPOSE_HAND_BONES: tuple[tuple[int, int], ...] = (
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
H2O_SCHEME2_CONTACT_DIRNAME = "h2o_contact_scheme2_v1"
H2O_SCHEME2_MARKER_CONTACT_DIRNAME = "h2o_marker_contact_scheme2_v1"
H2O_INTERHAND_CONTACT_DIRNAME = "h2o_interhand_contact_scheme2_v1"
H2O_INTERHAND_DISTANCE_DIRNAME = "h2o_interhand_contact_distances_v1"
H2O_MARKER_CONTACT_COUNT = len(MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195)
H2O_OBJECT_MODEL_PATHS: dict[int, str] = {
    1: "book/book.obj",
    2: "espresso/espresso.obj",
    3: "lotion/lotion.obj",
    4: "spray/lotion_spray.obj",
    5: "milk/milk.obj",
    6: "cocoa/cocoa.obj",
    7: "chips/chips.obj",
    8: "cappuccino/cappuccino.obj",
}


def default_data_root() -> Path:
    return default_data_root_path()


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _light_index_cache_path(root: Path, cache_name: str) -> Path:
    return root / ".light_index_cache" / f"{cache_name}.pkl"


def _light_index_source_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=False)
    signature: dict[str, Any] = {
        "path": str(resolved),
        "exists": path.exists(),
    }
    if signature["exists"]:
        stat = path.stat()
        signature["mtime_ns"] = stat.st_mtime_ns
        signature["size"] = stat.st_size
    return signature


def _intrinsics_from_focal_principal(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[0, 0] = fx
    matrix[1, 1] = fy
    matrix[0, 2] = cx
    matrix[1, 2] = cy
    return matrix


def _scaled_intrinsics(intrinsics: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = intrinsics.astype(np.float32).copy()
    scaled[0, 0] *= float(scale_x)
    scaled[0, 2] *= float(scale_x)
    scaled[1, 1] *= float(scale_y)
    scaled[1, 2] *= float(scale_y)
    return scaled


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _invert_pose(pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(pose, dtype=np.float32)).astype(np.float32)


def _project_points(joints_3d: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(joints_3d[:, 2], 1e-6, None)
    x = intrinsics[0, 0] * joints_3d[:, 0] / z + intrinsics[0, 2]
    y = intrinsics[1, 1] * joints_3d[:, 1] / z + intrinsics[1, 2]
    return np.stack([x, y], axis=1).astype(np.float32)


def _bbox_from_points(points_2d: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(points_2d[:, 0].min()),
            float(points_2d[:, 1].min()),
            float(points_2d[:, 0].max()),
            float(points_2d[:, 1].max()),
        ],
        dtype=np.float32,
    )


def _count_sides(hand_annos: list[HandAnnotation]) -> tuple[int, int]:
    left_count = sum(1 for hand in hand_annos if hand.side == "left")
    right_count = sum(1 for hand in hand_annos if hand.side == "right")
    return left_count, right_count


def _hand_has_mano_signal(hand: HandAnnotation) -> bool:
    return any(
        value is not None
        for value in (
            hand.mano_pose,
            hand.mano_global_orient,
            hand.mano_hand_pose,
        )
    )


def _any_hand_field(hand_annos: list[HandAnnotation], field_name: str) -> bool:
    return any(getattr(hand, field_name) is not None for hand in hand_annos)


def _hand_supervision_flags(hand_annos: list[HandAnnotation]) -> dict[str, bool]:
    has_joint_3d_gt = _any_hand_field(hand_annos, "joints_3d")
    return {
        "has_mano": any(_hand_has_mano_signal(hand) for hand in hand_annos),
        "has_3d_joints": has_joint_3d_gt,
        "has_bbox_gt": _any_hand_field(hand_annos, "bbox_xyxy"),
        "has_joint_3d_gt": has_joint_3d_gt,
    }


def _three_r_supervision_flags(
    *,
    depth_ref: MediaRef | None,
    intrinsics: np.ndarray | None,
    camera_pose: np.ndarray | None,
) -> dict[str, bool]:
    has_depth_gt = depth_ref is not None
    has_intrinsics_gt = intrinsics is not None
    has_camera_pose_gt = camera_pose is not None
    return {
        "has_depth_gt": has_depth_gt,
        "has_intrinsics_gt": has_intrinsics_gt,
        "has_camera_pose_gt": has_camera_pose_gt,
        "has_3r_gt": has_depth_gt or has_intrinsics_gt or has_camera_pose_gt,
    }


def _stage_membership_from_text_files(root: Path, prefix: str = "pose") -> dict[str, str]:
    membership: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = root / f"{prefix}_{split}.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item:
                membership[item] = split
    return membership


def _parse_h2o_intrinsics(path: Path) -> tuple[np.ndarray, tuple[int, int] | None]:
    values = [float(item) for item in path.read_text(encoding="utf-8").split()]
    intrinsics = _intrinsics_from_focal_principal(values[0], values[1], values[2], values[3])
    native_size = (int(values[4]), int(values[5])) if len(values) >= 6 else None
    return intrinsics, native_size


def _parse_matrix_txt(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float32)
    return values.reshape(shape)


def _parse_h2o_hand_pose_joints(path: Path) -> list[np.ndarray | None]:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float32)
    outputs: list[np.ndarray | None] = []
    offset = 0
    for _ in range(2):
        offset += 1
        joints = values[offset : offset + 63].reshape(21, 3)
        offset += 63
        outputs.append(joints if np.isfinite(joints).all() and float(np.linalg.norm(joints)) > 1e-6 else None)
    return outputs


def _parse_h2o_hand_pose_mano(path: Path) -> list[dict[str, np.ndarray] | None]:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float32)
    outputs: list[dict[str, np.ndarray] | None] = []
    offset = 0
    for _ in range(2):
        offset += 1
        trans = values[offset : offset + 3]
        offset += 3
        pose = values[offset : offset + 48]
        offset += 48
        betas = values[offset : offset + 10]
        offset += 10
        if np.isfinite(np.concatenate((trans, pose, betas))).all() and float(np.linalg.norm(trans)) > 1e-6:
            outputs.append(
                {
                    "trans": trans,
                    "global_orient": pose[:3],
                    "hand_pose": pose[3:],
                    "pose": pose,
                    "betas": betas,
                }
            )
        else:
            outputs.append(None)
    return outputs


def _parse_h2o_object_pose_rt(path: Path) -> tuple[int, np.ndarray]:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ", dtype=np.float32)
    if values.size != 17:
        raise ValueError(f"obj_pose_rt 应有 17 个数，实际为 {values.size}: {path}")
    object_id = int(values[0])
    if object_id not in range(9):
        raise ValueError(f"未知 H2O object id {object_id}: {path}")
    transform = values[1:].reshape(4, 4)
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-5):
        raise ValueError(f"obj_pose_rt 最后一行不是齐次刚体变换: {path}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3, dtype=np.float32), atol=2e-3):
        raise ValueError(f"obj_pose_rt 旋转矩阵不正交: {path}")
    return object_id, transform


def _parse_h2o_contact_points(path: Path) -> tuple[list[np.ndarray], bool]:
    empty = [np.empty((0, 3), dtype=np.float32) for _ in range(2)]
    if not path.exists():
        return empty, False
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"H2O contact 文件为空: {path}")
    if int(lines[0].strip()) == 0:
        return empty, True
    if len(lines) < 3:
        raise ValueError(f"H2O contact 文件缺少左右手记录: {path}")

    outputs: list[np.ndarray] = []
    for line in lines[1:3]:
        records = ast.literal_eval(line)
        points = np.asarray([record[2] for record in records], dtype=np.float32).reshape(-1, 3)
        outputs.append(points)
    return outputs, True


def _load_h2o_scheme2_contact_sequence(path: Path, *, point_count: int, label_name: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int32).copy()
        hard_contact = np.asarray(payload["hard_contact"], dtype=np.uint8).copy()
        distances_m = np.asarray(payload["distances_m"], dtype=np.float32).copy()
        valid = np.asarray(payload["valid"], dtype=np.bool_).copy()
        format_version = int(payload["format_version"])
    if format_version != 1:
        raise ValueError(f"不支持的 H2O 方案二 contact 格式版本 {format_version}: {path}")
    if frame_ids.ndim != 1 or len(np.unique(frame_ids)) != frame_ids.size:
        raise ValueError(f"H2O 方案二 contact frame_ids 非一维或有重复: {path}")
    if hard_contact.shape != (frame_ids.size, 2, point_count):
        raise ValueError(f"H2O 方案二 {label_name} hard_contact 形状错误 {hard_contact.shape}: {path}")
    if distances_m.shape != (frame_ids.size, 2, point_count):
        raise ValueError(f"H2O 方案二 {label_name} distances_m 形状错误 {distances_m.shape}: {path}")
    if valid.shape != (frame_ids.size, 2):
        raise ValueError(f"H2O 方案二 valid 形状错误 {valid.shape}: {path}")
    if not np.all((hard_contact == 0) | (hard_contact == 1)):
        raise ValueError(f"H2O 方案二 hard_contact 包含非 0/1 值: {path}")
    return {
        "frame_to_index": {int(frame_id): index for index, frame_id in enumerate(frame_ids)},
        "hard_contact": hard_contact,
        "distances_m": distances_m,
        "valid": valid,
    }


def _load_h2o_interhand_contact_sequence(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int32).copy()
        joint_hard_contact = np.asarray(payload["joint_hard_contact"], dtype=np.uint8).copy()
        marker_hard_contact = np.asarray(payload["marker_hard_contact"], dtype=np.uint8).copy()
        valid = np.asarray(payload["valid"], dtype=np.bool_).copy()
        format_version = int(payload["format_version"])
    if format_version != 1:
        raise ValueError(f"不支持的 H2O inter-hand contact 格式版本 {format_version}: {path}")
    if frame_ids.ndim != 1 or len(np.unique(frame_ids)) != frame_ids.size:
        raise ValueError(f"H2O inter-hand contact frame_ids 非一维或有重复: {path}")
    if joint_hard_contact.shape != (frame_ids.size, 2, 21):
        raise ValueError(f"H2O inter-hand joint_hard_contact 形状错误 {joint_hard_contact.shape}: {path}")
    if marker_hard_contact.shape != (frame_ids.size, 2, H2O_MARKER_CONTACT_COUNT):
        raise ValueError(f"H2O inter-hand marker_hard_contact 形状错误 {marker_hard_contact.shape}: {path}")
    if valid.shape != (frame_ids.size, 2):
        raise ValueError(f"H2O inter-hand valid 形状错误 {valid.shape}: {path}")
    if not np.all((joint_hard_contact == 0) | (joint_hard_contact == 1)):
        raise ValueError(f"H2O inter-hand joint_hard_contact 包含非 0/1 值: {path}")
    if not np.all((marker_hard_contact == 0) | (marker_hard_contact == 1)):
        raise ValueError(f"H2O inter-hand marker_hard_contact 包含非 0/1 值: {path}")
    return {
        "frame_to_index": {int(frame_id): index for index, frame_id in enumerate(frame_ids)},
        "joint_hard_contact": joint_hard_contact,
        "marker_hard_contact": marker_hard_contact,
        "valid": valid,
    }


def _load_h2o_interhand_distance_sequence(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int32).copy()
        valid_bimanual = np.asarray(payload["valid_bimanual"], dtype=np.bool_).copy()
        joint_distances = np.stack(
            [
                np.asarray(payload["left_joint_to_right_mesh_m"], dtype=np.float32),
                np.asarray(payload["right_joint_to_left_mesh_m"], dtype=np.float32),
            ],
            axis=1,
        )
        marker_distances = np.stack(
            [
                np.asarray(payload["left_marker_to_right_mesh_m"], dtype=np.float32),
                np.asarray(payload["right_marker_to_left_mesh_m"], dtype=np.float32),
            ],
            axis=1,
        )
    if frame_ids.ndim != 1 or len(np.unique(frame_ids)) != frame_ids.size:
        raise ValueError(f"H2O inter-hand distance frame_ids 非一维或有重复: {path}")
    if valid_bimanual.shape != (frame_ids.size,):
        raise ValueError(f"H2O inter-hand distance valid_bimanual 形状错误 {valid_bimanual.shape}: {path}")
    if joint_distances.shape != (frame_ids.size, 2, 21):
        raise ValueError(f"H2O inter-hand joint distance 形状错误 {joint_distances.shape}: {path}")
    if marker_distances.shape != (frame_ids.size, 2, H2O_MARKER_CONTACT_COUNT):
        raise ValueError(f"H2O inter-hand marker distance 形状错误 {marker_distances.shape}: {path}")
    valid = np.repeat(valid_bimanual[:, None], 2, axis=1)
    if not np.isfinite(joint_distances[valid]).all() or not np.isfinite(marker_distances[valid]).all():
        raise ValueError(f"H2O inter-hand 有效距离包含非有限值: {path}")
    return {
        "frame_to_index": {int(frame_id): index for index, frame_id in enumerate(frame_ids)},
        "joint_distances_m": joint_distances,
        "marker_distances_m": marker_distances,
        "valid": valid,
    }


def _joint_contact_hotspot(contact_points: np.ndarray, joints: np.ndarray) -> np.ndarray:
    hotspot = np.zeros(21, dtype=np.float32)
    if contact_points.size == 0:
        return hotspot

    points = np.asarray(contact_points, dtype=np.float32).reshape(-1, 3)
    joints = np.asarray(joints, dtype=np.float32).reshape(21, 3)
    bone_indices = np.asarray(H2O_OPENPOSE_HAND_BONES, dtype=np.int64)
    starts = joints[bone_indices[:, 0]]
    segments = joints[bone_indices[:, 1]] - starts
    segment_length_sq = np.maximum(np.sum(segments * segments, axis=1), 1e-8)
    relative_points = points[:, None, :] - starts[None, :, :]
    projection_t = np.sum(relative_points * segments[None, :, :], axis=-1) / segment_length_sq[None, :]
    projection_t = np.clip(projection_t, 0.0, 1.0)
    closest_points = starts[None, :, :] + projection_t[..., None] * segments[None, :, :]
    distances = np.sum((points[:, None, :] - closest_points) ** 2, axis=-1)
    closest_bones = np.argmin(distances, axis=1)

    selected_t = projection_t[np.arange(points.shape[0]), closest_bones]
    selected_bones = bone_indices[closest_bones]
    np.add.at(hotspot, selected_bones[:, 0], 1.0 - selected_t)
    np.add.at(hotspot, selected_bones[:, 1], selected_t)
    peak = float(hotspot.max())
    if peak > 0.0:
        hotspot /= peak
    return hotspot


def _mano_hand(
    side: str,
    mano_info: dict[str, np.ndarray] | None,
    joints_3d: np.ndarray | None,
    intrinsics: np.ndarray | None,
    *,
    contact_points: np.ndarray | None = None,
    contact_available: bool = False,
) -> HandAnnotation | None:
    if mano_info is None and joints_3d is None:
        return None
    joints_2d = _project_points(joints_3d, intrinsics) if joints_3d is not None and intrinsics is not None else None
    bbox = _bbox_from_points(joints_2d) if joints_2d is not None else None
    contact_hotspot_valid = bool(contact_available and joints_3d is not None)
    contact_hotspot = (
        _joint_contact_hotspot(contact_points, joints_3d)
        if contact_hotspot_valid and contact_points is not None
        else np.zeros(21, dtype=np.float32)
    )
    return HandAnnotation(
        side=side,
        visible=True,
        bbox_xyxy=bbox,
        joints_3d=joints_3d,
        joints_2d=joints_2d,
        mano_global_orient=None if mano_info is None else mano_info["global_orient"],
        mano_hand_pose=None if mano_info is None else mano_info["hand_pose"],
        mano_pose=None if mano_info is None else mano_info["pose"],
        mano_pose_format="axis_angle_full",
        mano_betas=None if mano_info is None else mano_info["betas"],
        mano_trans=None if mano_info is None else mano_info["trans"],
        extras={
            "contact_hotspot": contact_hotspot,
            "contact_hotspot_valid": contact_hotspot_valid,
        },
    )


def _sequence_entry_for_index(
    sequence_offsets: list[int],
    sequence_entries: list[dict[str, Any]],
    index: int,
) -> tuple[dict[str, Any], int]:
    sequence_index = bisect_right(sequence_offsets, index) - 1
    sequence_entry = sequence_entries[sequence_index]
    local_index = index - sequence_offsets[sequence_index]
    return sequence_entry, local_index

class H2OFrameDataset(BaseFrameDataset):
    dataset_name = "h2o"
    base_dataset_name = "h2o"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        load_rgb: bool = False,
        load_depth: bool = False,
        scheme2_contact_root: str | Path | None = None,
        scheme2_marker_contact_root: str | Path | None = None,
        interhand_contact_root: str | Path | None = None,
        interhand_distance_root: str | Path | None = None,
        include_scene_occlusion_in_visibility: bool = True,
    ) -> None:
        resolved_root = _as_path(root).resolve()
        self._index_cache_path = _light_index_cache_path(resolved_root, f"h2o_frame_dataset_{split}")
        self.scheme2_contact_root = (
            _as_path(scheme2_contact_root)
            if scheme2_contact_root is not None
            else resolved_root.parent / H2O_SCHEME2_CONTACT_DIRNAME
        )
        self._scheme2_contact_cache: dict[Path, dict[str, Any] | None] = {}
        self.scheme2_marker_contact_root = (
            _as_path(scheme2_marker_contact_root)
            if scheme2_marker_contact_root is not None
            else resolved_root.parent / H2O_SCHEME2_MARKER_CONTACT_DIRNAME
        )
        self._scheme2_marker_contact_cache: dict[Path, dict[str, Any] | None] = {}
        self.interhand_contact_root = (
            _as_path(interhand_contact_root)
            if interhand_contact_root is not None
            else resolved_root.parent / H2O_INTERHAND_CONTACT_DIRNAME
        )
        self._interhand_contact_cache: dict[Path, dict[str, Any] | None] = {}
        self.interhand_distance_root = (
            _as_path(interhand_distance_root)
            if interhand_distance_root is not None
            else resolved_root.parent / H2O_INTERHAND_DISTANCE_DIRNAME
        )
        self._interhand_distance_cache: dict[Path, dict[str, Any] | None] = {}
        self._cache_pid = os.getpid()
        self._sequence_runtime_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._frame_annotation_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.include_scene_occlusion_in_visibility = True
        super().__init__(resolved_root, split, load_rgb=load_rgb, load_depth=load_depth)

    def _scheme2_contact_for_frame(self, sequence_root: Path, frame_stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        relative_sequence = sequence_root.parent.relative_to(self.root)
        sequence_path = self.scheme2_contact_root / relative_sequence / "cam4.npz"
        if sequence_path not in self._scheme2_contact_cache:
            self._scheme2_contact_cache[sequence_path] = (
                _load_h2o_scheme2_contact_sequence(sequence_path, point_count=21, label_name="joint")
                if sequence_path.is_file()
                else None
            )
        sequence = self._scheme2_contact_cache[sequence_path]
        if sequence is None:
            return np.zeros((2, 21), dtype=np.float32), np.zeros(2, dtype=np.bool_), np.zeros((2, 21), dtype=np.float32)
        row_index = sequence["frame_to_index"].get(int(frame_stem))
        if row_index is None:
            return np.zeros((2, 21), dtype=np.float32), np.zeros(2, dtype=np.bool_), np.zeros((2, 21), dtype=np.float32)
        return (
            sequence["hard_contact"][row_index].astype(np.float32, copy=True),
            sequence["valid"][row_index].copy(),
            sequence["distances_m"][row_index].copy(),
        )

    def _scheme2_marker_contact_for_frame(self, sequence_root: Path, frame_stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        relative_sequence = sequence_root.parent.relative_to(self.root)
        sequence_path = self.scheme2_marker_contact_root / relative_sequence / "cam4.npz"
        if sequence_path not in self._scheme2_marker_contact_cache:
            self._scheme2_marker_contact_cache[sequence_path] = (
                _load_h2o_scheme2_contact_sequence(
                    sequence_path,
                    point_count=H2O_MARKER_CONTACT_COUNT,
                    label_name="marker",
                )
                if sequence_path.is_file()
                else None
            )
        sequence = self._scheme2_marker_contact_cache[sequence_path]
        if sequence is None:
            return np.zeros((2, H2O_MARKER_CONTACT_COUNT), dtype=np.float32), np.zeros(2, dtype=np.bool_), np.zeros((2, H2O_MARKER_CONTACT_COUNT), dtype=np.float32)
        row_index = sequence["frame_to_index"].get(int(frame_stem))
        if row_index is None:
            return np.zeros((2, H2O_MARKER_CONTACT_COUNT), dtype=np.float32), np.zeros(2, dtype=np.bool_), np.zeros((2, H2O_MARKER_CONTACT_COUNT), dtype=np.float32)
        return (
            sequence["hard_contact"][row_index].astype(np.float32, copy=True),
            sequence["valid"][row_index].copy(),
            sequence["distances_m"][row_index].copy(),
        )

    def _interhand_contact_for_frame(
        self,
        sequence_root: Path,
        frame_stem: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        relative_sequence = sequence_root.parent.relative_to(self.root)
        sequence_path = self.interhand_contact_root / relative_sequence / "cam4.npz"
        if sequence_path not in self._interhand_contact_cache:
            self._interhand_contact_cache[sequence_path] = (
                _load_h2o_interhand_contact_sequence(sequence_path)
                if sequence_path.is_file()
                else None
            )
        sequence = self._interhand_contact_cache[sequence_path]
        if sequence is None:
            return (
                np.zeros((2, 21), dtype=np.float32),
                np.zeros((2, H2O_MARKER_CONTACT_COUNT), dtype=np.float32),
                np.zeros(2, dtype=np.bool_),
            )
        row_index = sequence["frame_to_index"].get(int(frame_stem))
        if row_index is None:
            return (
                np.zeros((2, 21), dtype=np.float32),
                np.zeros((2, H2O_MARKER_CONTACT_COUNT), dtype=np.float32),
                np.zeros(2, dtype=np.bool_),
            )
        return (
            sequence["joint_hard_contact"][row_index].astype(np.float32, copy=True),
            sequence["marker_hard_contact"][row_index].astype(np.float32, copy=True),
            sequence["valid"][row_index].copy(),
        )

    def _interhand_distance_for_frame(
        self,
        sequence_root: Path,
        frame_stem: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        relative_sequence = sequence_root.parent.relative_to(self.root)
        sequence_path = self.interhand_distance_root / relative_sequence / "cam4.npz"
        if sequence_path not in self._interhand_distance_cache:
            self._interhand_distance_cache[sequence_path] = (
                _load_h2o_interhand_distance_sequence(sequence_path) if sequence_path.is_file() else None
            )
        sequence = self._interhand_distance_cache[sequence_path]
        if sequence is None:
            return np.zeros((2, 21), dtype=np.float32), np.zeros((2, H2O_MARKER_CONTACT_COUNT), dtype=np.float32), np.zeros(2, dtype=np.bool_)
        row_index = sequence["frame_to_index"].get(int(frame_stem))
        if row_index is None:
            return np.zeros((2, 21), dtype=np.float32), np.zeros((2, H2O_MARKER_CONTACT_COUNT), dtype=np.float32), np.zeros(2, dtype=np.bool_)
        return sequence["joint_distances_m"][row_index].copy(), sequence["marker_distances_m"][row_index].copy(), sequence["valid"][row_index].copy()

    def _resolve_rgb_path(self, rel_rgb: str) -> Path:
        direct = self.root / rel_rgb
        if direct.exists():
            return direct
        parts = Path(rel_rgb).parts
        if parts and parts[0].startswith("subject") and not parts[0].endswith("_ego"):
            remapped = self.root / f"{parts[0]}_ego" / Path(*parts[1:])
            if remapped.exists():
                return remapped
        return direct

    def _uses_lazy_records(self) -> bool:
        return True

    def _ensure_worker_local_caches(self) -> None:
        pid = os.getpid()
        if self._cache_pid == pid:
            return
        self._sequence_runtime_cache.clear()
        self._frame_annotation_cache.clear()
        self._scheme2_contact_cache.clear()
        self._scheme2_marker_contact_cache.clear()
        self._interhand_contact_cache.clear()
        self._interhand_distance_cache.clear()
        self._record_cache.clear()
        self._cache_pid = pid

    def _sequence_runtime(self, sequence_root: Path) -> dict[str, Any]:
        self._ensure_worker_local_caches()
        cache_key = str(sequence_root)
        cached = _lru_get(self._sequence_runtime_cache, cache_key)
        if cached is not None:
            return cached
        object_pose_root = sequence_root / "obj_pose_rt"
        runtime = {
            "intrinsics": _parse_h2o_intrinsics(sequence_root / "cam_intrinsics.txt"),
            "has_object_pose_files": bool(object_pose_root.is_dir() and next(object_pose_root.glob("*.txt"), None)),
            "image_sizes": {},
        }
        return _lru_put(
            self._sequence_runtime_cache,
            cache_key,
            runtime,
            limit=_H2O_SEQUENCE_RUNTIME_CACHE_LIMIT,
        )

    def _frame_annotations(
        self,
        sequence_root: Path,
        frame_stem: str,
        *,
        has_object_pose_files: bool,
    ) -> dict[str, Any]:
        self._ensure_worker_local_caches()
        cache_key = f"{sequence_root}\0{frame_stem}"
        cached = _lru_get(self._frame_annotation_cache, cache_key)
        if cached is not None:
            return cached
        object_pose = None
        if self.include_scene_occlusion_in_visibility and has_object_pose_files:
            object_pose = _parse_h2o_object_pose_rt(sequence_root / "obj_pose_rt" / f"{frame_stem}.txt")
        annotations = {
            "camera_pose": _invert_pose(_parse_matrix_txt(sequence_root / "cam_pose" / f"{frame_stem}.txt", (4, 4))),
            "joints": _parse_h2o_hand_pose_joints(sequence_root / "hand_pose" / f"{frame_stem}.txt"),
            "mano": _parse_h2o_hand_pose_mano(sequence_root / "hand_pose_mano" / f"{frame_stem}.txt"),
            "object_pose": object_pose,
            "contact": _parse_h2o_contact_points(sequence_root / "contact" / f"{frame_stem}.txt"),
        }
        return _lru_put(
            self._frame_annotation_cache,
            cache_key,
            annotations,
            limit=_H2O_FRAME_ANNOTATION_CACHE_LIMIT,
        )

    def _candidate_rgb_dirs(self, rel_rgb: str) -> list[Path]:
        rel_path = Path(rel_rgb)
        candidate_dirs = [self.root / rel_path.parent]
        parts = rel_path.parts
        if parts and parts[0].startswith("subject") and not parts[0].endswith("_ego"):
            candidate_dirs.append(self.root / f"{parts[0]}_ego" / Path(*parts[1:-1]))
        unique_dirs: list[Path] = []
        seen: set[str] = set()
        for candidate_dir in candidate_dirs:
            key = str(candidate_dir.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            unique_dirs.append(candidate_dir)
        return unique_dirs

    def _candidate_rgb_paths(self, rel_rgb: str) -> list[Path]:
        filename = Path(rel_rgb).name
        return [candidate_dir / filename for candidate_dir in self._candidate_rgb_dirs(rel_rgb)]

    def _build_record_index(self) -> list[dict[str, Any]]:
        label_root = self.root / "label_split"
        label_membership = _stage_membership_from_text_files(label_root)
        if self.split == "all":
            wanted_splits = {"train", "val", "test"}
        elif self.split == "trainval":
            wanted_splits = {"train", "val"}
        else:
            wanted_splits = {self.split}
        source_paths = [label_root / f"pose_{split_name}.txt" for split_name in ("train", "val", "test")]
        source_signatures = [_light_index_source_signature(path) for path in source_paths]

        if self._index_cache_path.exists():
            try:
                payload = _load_pickle(self._index_cache_path)
            except (OSError, EOFError, pickle.UnpicklingError, ModuleNotFoundError):
                payload = None
            if isinstance(payload, dict):
                record_index = payload.get("record_index")
                cached_sources = payload.get("sources")
                missing_rgb_paths = payload.get("missing_rgb_paths", [])
                if (
                    payload.get("version") == 2
                    and isinstance(record_index, list)
                    and isinstance(cached_sources, list)
                    and cached_sources == source_signatures
                    and isinstance(missing_rgb_paths, list)
                    and not any(Path(str(path)).exists() for path in missing_rgb_paths)
                ):
                    return record_index

        record_index: list[dict[str, Any]] = []
        missing_rgb_paths: list[str] = []
        for rel_rgb, split in sorted(label_membership.items()):
            if split not in wanted_splits:
                continue
            rgb_path = self._resolve_rgb_path(rel_rgb)
            if not rgb_path.exists():
                missing_rgb_paths.extend(str(path) for path in self._candidate_rgb_paths(rel_rgb))
                continue
            frame_stem = Path(rel_rgb).stem
            parts = list(Path(rel_rgb).parts)
            if parts and parts[0].startswith("subject") and not parts[0].endswith("_ego"):
                parts[0] = f"{parts[0]}_ego"
            record_index.append(
                {
                    "split": split,
                    "sequence_id": str(Path(*parts[:-2])),
                    "frame_id": frame_stem,
                    "temporal_index": int(frame_stem),
                    "rgb_path": str(rgb_path),
                }
            )
        _write_pickle(
            self._index_cache_path,
            {
                "version": 2,
                "sources": source_signatures,
                "missing_rgb_paths": missing_rgb_paths,
                "record_index": record_index,
            },
        )
        return record_index

    def _materialize_record(self, index_entry: dict[str, Any]) -> FrameRecord:
        self._ensure_worker_local_caches()
        rgb_path = Path(str(index_entry["rgb_path"]))
        frame_stem = str(index_entry["frame_id"])
        sequence_root = rgb_path.parents[1]
        runtime = self._sequence_runtime(sequence_root)
        intrinsics, native_size = runtime["intrinsics"]
        intrinsics = intrinsics.copy()
        if native_size is not None:
            image_size_key = str(rgb_path)
            image_width, image_height = runtime["image_sizes"].get(image_size_key, (0, 0))
            if image_width <= 0 or image_height <= 0:
                image_width, image_height = _image_size(rgb_path)
                runtime["image_sizes"][image_size_key] = (image_width, image_height)
            native_width, native_height = native_size
            if (image_width, image_height) != (native_width, native_height):
                intrinsics = _scaled_intrinsics(
                    intrinsics,
                    float(image_width) / float(native_width),
                    float(image_height) / float(native_height),
                )
        annotations = self._frame_annotations(
            sequence_root,
            frame_stem,
            has_object_pose_files=bool(runtime["has_object_pose_files"]),
        )
        camera_pose = annotations["camera_pose"]
        joints_left, joints_right = annotations["joints"]
        mano_left, mano_right = annotations["mano"]
        scene_extras: dict[str, Any] = {}
        if self.include_scene_occlusion_in_visibility:
            object_pose = annotations["object_pose"]
            # Hand-only subsets have no object-pose files; partially missing
            # sequences are still rejected by the original parser above.
            if object_pose is not None:
                object_id, object_to_camera = object_pose
                scene_extras["h2o_object_id"] = object_id
                if object_id != 0:
                    object_mesh_path = self.root / "object" / H2O_OBJECT_MODEL_PATHS[object_id]
                    if not object_mesh_path.is_file():
                        raise FileNotFoundError(f"缺少 H2O object mesh: {object_mesh_path}")
                    scene_extras["h2o_object_to_camera"] = object_to_camera
                    scene_extras["h2o_object_mesh_path"] = str(object_mesh_path)
                    scene_extras["scene_objects"] = [
                        {
                            "object_id": int(object_id),
                            "mesh_path": str(object_mesh_path),
                            "object_to_camera": object_to_camera,
                        }
                    ]
        scene_extras.setdefault("scene_objects", [])
        contact_points, contact_available = annotations["contact"]
        hand_annos = [
            hand
            for hand in (
                _mano_hand(
                    "left",
                    mano_left,
                    joints_left,
                    intrinsics,
                    contact_points=contact_points[0],
                    contact_available=contact_available,
                ),
                _mano_hand(
                    "right",
                    mano_right,
                    joints_right,
                    intrinsics,
                    contact_points=contact_points[1],
                    contact_available=contact_available,
                ),
            )
            if hand is not None
        ]
        left_count, right_count = _count_sides(hand_annos)
        depth_ref = MediaRef(kind="path", path=str(sequence_root / "depth" / f"{frame_stem}.png"))
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split=str(index_entry["split"]),
            sequence_id=str(index_entry["sequence_id"]),
            frame_id=frame_stem,
            temporal_index=int(index_entry["temporal_index"]),
            view_name="egocentric",
            is_egocentric=True,
            rgb_ref=MediaRef(kind="path", path=str(rgb_path)),
            depth_ref=depth_ref,
            depth_mode="png_uint16",
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=depth_ref, intrinsics=intrinsics, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras=scene_extras,
        )

# Reference non-H2O dataset loaders
class Hot3dAriaFrameDataset(BaseFrameDataset):
    dataset_name = "hot3d_aria"
    base_dataset_name = "hot3d"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        manifest_path: str | Path | None = None,
        rgb_cache_root: str | Path | None = None,
        rgb_cache_max_bytes: int = 0,
        rgb_cache_worker_count: int = 1,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else Path(root) / "hot3d_aria_local_manifest.jsonl"
        self.rgb_cache_root = Path(rgb_cache_root) if rgb_cache_root is not None else None
        self.rgb_cache_max_bytes = max(int(rgb_cache_max_bytes), 0)
        self.rgb_cache_worker_count = max(int(rgb_cache_worker_count), 1)
        self._rgb_cache_writes = 0
        self._index_cache_path = _light_index_cache_path(_as_path(root), f"hot3d_aria_frame_dataset_{split}")
        self._sequence_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        super().__init__(root, split, load_rgb=load_rgb, load_depth=load_depth)

    def _uses_lazy_records(self) -> bool:
        return True

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = BaseFrameDataset.__getitem__(self, index)
        rgb_ref = sample.get("rgb_ref") or {}
        if sample.get("rgb") is not None and rgb_ref.get("kind") == "hot3d_vrs_frame":
            sample["rgb"] = _rectify_rotate_hot3d_aria_rgb(sample["rgb"], sample["extras"]["hot3d_camera_model"])
            cache_path = self._rgb_cache_path(
                sequence_id=str(sample["sequence_id"]),
                timestamp_ns=int(sample["temporal_index"]),
            )
            if cache_path is not None:
                try:
                    _write_hot3d_rectified_rgb_cache(cache_path, sample["rgb"])
                    sample["rgb_ref"] = MediaRef(kind="path", path=str(cache_path)).to_dict()
                    self._get_record(index).rgb_ref = MediaRef(kind="path", path=str(cache_path))
                    self._rgb_cache_writes += 1
                    if self._rgb_cache_writes % _HOT3D_RGB_CACHE_PRUNE_INTERVAL == 0:
                        _prune_hot3d_rectified_rgb_cache(
                            cache_path.parents[1],
                            max(1, self.rgb_cache_max_bytes // self.rgb_cache_worker_count),
                        )
                except OSError:
                    # A node-local cache is an optimization only; VRS remains
                    # the source of truth when its filesystem is unavailable.
                    pass
        return sample

    def _rgb_cache_worker_slot(self) -> int:
        worker = get_worker_info()
        return 0 if worker is None else int(worker.id)

    def _rgb_cache_path(self, *, sequence_id: str, timestamp_ns: int) -> Path | None:
        if self.rgb_cache_root is None or self.rgb_cache_max_bytes <= 0:
            return None
        return _hot3d_rectified_rgb_cache_path(
            self.rgb_cache_root,
            sequence_id=sequence_id,
            timestamp_ns=timestamp_ns,
            worker_slot=self._rgb_cache_worker_slot(),
        )

    def _build_record_index(self) -> list[dict[str, Any]]:
        def _scan_record_index() -> tuple[list[dict[str, Any]], list[Path]]:
            if self.manifest_path.exists():
                manifest_rows = _load_jsonl(self.manifest_path)
                source_paths: list[Path] = [self.manifest_path]
            else:
                required_files = (
                    "recording.vrs",
                    "mano_hand_pose_trajectory.jsonl",
                    "box2d_hands.csv",
                    "camera_models.json",
                    "headset_trajectory.csv",
                )
                sequence_dirs = [
                    path
                    for path in sorted(self.root.glob("P*"))
                    if path.is_dir() and all((path / file_name).is_file() for file_name in required_files)
                ]
                manifest_rows = [
                    {"sequence_name": path.name, "sequence_dir": str(path)}
                    for path in sequence_dirs
                ]
                source_paths = [self.root, *sequence_dirs]
            if not manifest_rows:
                return [], source_paths

            record_index: list[dict[str, Any]] = []
            for row in manifest_rows:
                sequence_dir = Path(row["sequence_dir"])
                trajectory_path = sequence_dir / "mano_hand_pose_trajectory.jsonl"
                source_paths.append(trajectory_path)
                dynamic_objects_path = sequence_dir / "dynamic_objects.csv"
                source_paths.append(dynamic_objects_path)
                valid_object_timestamps: set[int] = set()
                if dynamic_objects_path.is_file():
                    with dynamic_objects_path.open("r", encoding="utf-8") as handle:
                        for item in csv.DictReader(handle):
                            try:
                                timestamp_ns = int(item["timestamp[ns]"])
                                object_uid = str(item["object_uid"])
                                translation = np.asarray(
                                    [
                                        float(item["t_wo_x[m]"]),
                                        float(item["t_wo_y[m]"]),
                                        float(item["t_wo_z[m]"]),
                                    ],
                                    dtype=np.float32,
                                )
                                quaternion = np.asarray(
                                    [
                                        float(item["q_wo_w"]),
                                        float(item["q_wo_x"]),
                                        float(item["q_wo_y"]),
                                        float(item["q_wo_z"]),
                                    ],
                                    dtype=np.float32,
                                )
                            except (KeyError, TypeError, ValueError):
                                continue
                            mesh_path = self.root / "assets" / f"{object_uid}.glb"
                            if (
                                mesh_path.is_file()
                                and translation.shape == (3,)
                                and quaternion.shape == (4,)
                                and np.isfinite(translation).all()
                                and np.isfinite(quaternion).all()
                                and np.linalg.norm(quaternion) > 1e-8
                            ):
                                valid_object_timestamps.add(timestamp_ns)
                timestamps_ns: list[int] = []
                with trajectory_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        timestamps_ns.append(int(json.loads(line)["timestamp_ns"]))
                for timestamp_ns in sorted(timestamps_ns):
                    if timestamp_ns not in valid_object_timestamps:
                        continue
                    record_index.append(
                        {
                            "sequence_id": row["sequence_name"],
                            "frame_id": str(timestamp_ns),
                            "temporal_index": timestamp_ns,
                            "sequence_dir": str(sequence_dir),
                        }
                    )
            return record_index, source_paths

        return _load_or_build_light_index_cache(self._index_cache_path, [], _scan_record_index, version=6)

    def _hot3d_sequence_payload(self, sequence_dir: Path) -> dict[str, Any]:
        cache_key = str(sequence_dir)
        cached = _lru_get(self._sequence_cache, cache_key)
        if cached is not None:
            return cached
        boxes = defaultdict(dict)
        with (sequence_dir / "box2d_hands.csv").open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                if item["stream_id"] != "214-1":
                    continue
                if not item["x_min[pixel]"]:
                    continue
                boxes[int(item["timestamp[ns]"])][int(item["hand_index"])] = {
                    "bbox": np.array(
                        [
                            float(item["x_min[pixel]"]),
                            float(item["y_min[pixel]"]),
                            float(item["x_max[pixel]"]),
                            float(item["y_max[pixel]"]),
                        ],
                        dtype=np.float32,
                    ),
                    "visible": float(item["visibility_ratio[%]"]) > 0.0,
                }
        hand_rows = _load_jsonl_dict(sequence_dir / "mano_hand_pose_trajectory.jsonl", "timestamp_ns")
        camera_models = _load_json(sequence_dir / "camera_models.json")
        camera_rgb = next(item for item in camera_models if item["stream_id"] == "214-1")
        timecode_to_device_time: dict[int, int] = {}
        time_mapping_path = sequence_dir / "timecode_devicetime_mapping.csv"
        if time_mapping_path.is_file():
            with time_mapping_path.open("r", encoding="utf-8") as handle:
                for item in csv.DictReader(handle):
                    try:
                        timecode_to_device_time[int(item["timecode_ns"])] = int(item["devicetime_ns"])
                    except (KeyError, TypeError, ValueError):
                        continue
        online_camera_models: list[tuple[int, dict[str, Any]]] = []
        online_calibration_path = sequence_dir / "mps" / "slam" / "online_calibration.jsonl"
        if online_calibration_path.is_file():
            with online_calibration_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        calibration_record = json.loads(line)
                        device_timestamp_ns = int(calibration_record["tracking_timestamp_us"]) * 1000
                        rgb_calibration = next(
                            item for item in calibration_record["CameraCalibrations"]
                            if item["Label"] == str(camera_rgb.get("label", "camera-rgb"))
                        )
                        quaternion_wxyz = rgb_calibration["T_Device_Camera"]["UnitQuaternion"]
                        online_camera_models.append(
                            (
                                device_timestamp_ns,
                                {
                                    "quaternion_wxyz": [quaternion_wxyz[0], *quaternion_wxyz[1]],
                                    "translation_xyz": rgb_calibration["T_Device_Camera"]["Translation"],
                                    "projectionParams": rgb_calibration["Projection"]["Params"],
                                },
                            )
                        )
                    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
                        continue
        online_camera_models.sort(key=lambda item: item[0])
        headset_poses: dict[int, np.ndarray] = {}
        with (sequence_dir / "headset_trajectory.csv").open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                timestamp_ns = int(item["timestamp[ns]"])
                translation = np.array(
                    [
                        float(item["t_wo_x[m]"]),
                        float(item["t_wo_y[m]"]),
                        float(item["t_wo_z[m]"]),
                    ],
                    dtype=np.float32,
                )
                quaternion = np.array(
                    [
                        float(item["q_wo_w"]),
                        float(item["q_wo_x"]),
                        float(item["q_wo_y"]),
                        float(item["q_wo_z"]),
                    ],
                    dtype=np.float32,
                )
                headset_poses[timestamp_ns] = _pose_from_quat_translation(quaternion, translation)
        object_rows: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        dynamic_objects_path = sequence_dir / "dynamic_objects.csv"
        if dynamic_objects_path.is_file():
            with dynamic_objects_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for item in reader:
                    try:
                        timestamp_ns = int(item["timestamp[ns]"])
                        translation = np.array(
                            [
                                float(item["t_wo_x[m]"]),
                                float(item["t_wo_y[m]"]),
                                float(item["t_wo_z[m]"]),
                            ],
                            dtype=np.float32,
                        )
                        quaternion = np.array(
                            [
                                float(item["q_wo_w"]),
                                float(item["q_wo_x"]),
                                float(item["q_wo_y"]),
                                float(item["q_wo_z"]),
                            ],
                            dtype=np.float32,
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                    object_rows[timestamp_ns].append(
                        {
                            "object_uid": str(item["object_uid"]),
                            "world_from_object": _pose_from_quat_translation(quaternion, translation),
                        }
                    )
        payload = {
            "boxes": boxes,
            "hand_rows": hand_rows,
            "headset_poses": headset_poses,
            "object_rows": object_rows,
            "camera_model": camera_rgb,
            "timecode_to_device_time": timecode_to_device_time,
            "online_camera_models": online_camera_models,
        }
        return _lru_put(self._sequence_cache, cache_key, payload, limit=_HOT3D_SEQUENCE_CACHE_LIMIT)

    def prefetch_sequence(self, sequence_index: int) -> None:
        sequence_id = list(self.sequence_to_indices.keys())[int(sequence_index)]
        first_index = self.sequence_to_indices[sequence_id][0]
        if self._record_index is None:
            return
        index_entry = self._record_index[int(first_index)]
        sequence_dir = Path(str(index_entry["sequence_dir"]))
        self._hot3d_sequence_payload(sequence_dir)
        cached_image = self._rgb_cache_path(
            sequence_id=str(index_entry["sequence_id"]),
            timestamp_ns=int(index_entry["temporal_index"]),
        )
        if cached_image is not None and cached_image.is_file():
            return
        prewarm_media_refs(
            [
                MediaRef(
                    kind="hot3d_vrs_frame",
                    path=str(sequence_dir / "recording.vrs"),
                    member="214-1",
                    timestamp_ns=int(index_entry["temporal_index"]),
                )
            ]
        )

    def _materialize_record(self, index_entry: dict[str, Any]) -> FrameRecord:
        sequence_dir = Path(str(index_entry["sequence_dir"]))
        timestamp_ns = int(index_entry["temporal_index"])
        sequence_payload = self._hot3d_sequence_payload(sequence_dir)
        camera_model = _hot3d_camera_model_with_online_calibration(
            sequence_payload["camera_model"],
            sequence_payload["online_camera_models"],
            sequence_payload["timecode_to_device_time"].get(timestamp_ns),
        )
        hand_row = sequence_payload["hand_rows"][timestamp_ns]
        hand_annos: list[HandAnnotation] = []
        t_world_device = sequence_payload["headset_poses"].get(timestamp_ns)
        t_camera_world = None
        if t_world_device is not None:
            t_device_camera = _pose_from_quat_translation(
                np.asarray(camera_model["T_Device_Camera"]["quaternion_wxyz"], dtype=np.float32),
                np.asarray(camera_model["T_Device_Camera"]["translation_xyz"], dtype=np.float32),
            )
            t_world_camera = t_world_device @ t_device_camera
            t_camera_world = _invert_pose(t_world_camera)
        rotated_camera_from_world = None
        if t_camera_world is not None:
            rotated_camera_from_world = (
                _pose_from_rotation_translation(
                    _HOT3D_CW_CAMERA_ROTATION,
                    np.zeros(3, dtype=np.float32),
                )
                @ t_camera_world
            ).astype(np.float32)
        for hand_key, side in (("0", "left"), ("1", "right")):
            if hand_key not in hand_row["hand_poses"]:
                continue
            payload = hand_row["hand_poses"][hand_key]
            box_info = sequence_payload["boxes"].get(timestamp_ns, {}).get(int(hand_key))
            bbox_xyxy = None
            if box_info is not None:
                raw_bbox = np.asarray(box_info["bbox"], dtype=np.float32)
                corners = np.array(
                    [
                        [raw_bbox[0], raw_bbox[1]],
                        [raw_bbox[2], raw_bbox[1]],
                        [raw_bbox[2], raw_bbox[3]],
                        [raw_bbox[0], raw_bbox[3]],
                    ],
                    dtype=np.float32,
                )
                bbox_xyxy = _bbox_from_points(_rectify_rotate_hot3d_points_2d(corners, camera_model))
            wrist_quaternion = np.asarray(payload.get("wrist_xform", {}).get("q_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            wrist_translation = np.asarray(payload.get("wrist_xform", {}).get("t_xyz", []), dtype=np.float32)
            mano_trans = wrist_translation if wrist_translation.shape == (3,) else None
            mano_global_orient = None
            if wrist_quaternion.shape == (4,) and np.isfinite(wrist_quaternion).all():
                mano_global_orient = _matrix_to_axis_angle(_quat_wxyz_to_matrix(wrist_quaternion))
            hand_annos.append(
                HandAnnotation(
                    side=side,
                    hand_index=int(hand_key),
                    visible=None if box_info is None else box_info["visible"],
                    bbox_xyxy=bbox_xyxy,
                    mano_global_orient=mano_global_orient,
                    mano_pose=np.asarray(payload.get("pose", []), dtype=np.float32),
                    mano_pose_format="hot3d_mano_pca",
                    mano_betas=np.asarray(payload.get("betas", []), dtype=np.float32),
                    mano_trans=None if mano_trans is None else mano_trans.astype(np.float32),
                    extras={
                        "raw_wrist_quaternion_wxyz": payload.get("wrist_xform", {}).get("q_wxyz"),
                        "post_mano_transform": rotated_camera_from_world,
                        "mano_flip_left_shapedirs": side == "left",
                    },
                )
            )
        cached_image = self._rgb_cache_path(
            sequence_id=str(index_entry["sequence_id"]),
            timestamp_ns=timestamp_ns,
        )
        if cached_image is not None and cached_image.is_file():
            rgb_ref = MediaRef(kind="path", path=str(cached_image))
        else:
            rgb_ref = MediaRef(
                kind="hot3d_vrs_frame",
                path=str(sequence_dir / "recording.vrs"),
                member="214-1",
                timestamp_ns=timestamp_ns,
            )
        intrinsics = _hot3d_rotated_linear_intrinsics(camera_model)
        camera_pose = rotated_camera_from_world
        left_count, right_count = _count_sides(hand_annos)
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split="train",
            sequence_id=str(index_entry["sequence_id"]),
            frame_id=str(index_entry["frame_id"]),
            temporal_index=timestamp_ns,
            view_name="egocentric",
            is_egocentric=True,
            rgb_ref=rgb_ref,
            depth_ref=None,
            depth_mode=None,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=None, intrinsics=intrinsics, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={
                "sequence_dir": str(sequence_dir),
                "recording_vrs": str(sequence_dir / "recording.vrs"),
                "hot3d_camera_model": camera_model,
                "scene_objects": _hot3d_scene_objects(
                    self.root,
                    sequence_payload["object_rows"].get(timestamp_ns, []),
                    rotated_camera_from_world,
                ),
            },
        )


class Hoi4dFrameDataset(BaseFrameDataset):
    dataset_name = "hoi4d"
    base_dataset_name = "hoi4d"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self._cache_pid = os.getpid()
        self._intrinsics_cache: dict[str, np.ndarray] = {}
        self._camera_pose_cache: dict[str, list[np.ndarray]] = {}
        self._sequence_runtime_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._frame_annotation_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._index_cache_path = _light_index_cache_path(_as_path(root), f"hoi4d_frame_dataset_{split}")
        self._record_cache: OrderedDict[int, FrameRecord] = OrderedDict()
        self._records_materialized: list[FrameRecord] | None = None
        self._record_index: list[dict[str, Any]] | None = None
        self._sequence_entries = self._load_hoi4d_sequence_entries()
        self._sequence_offsets: list[int] = []
        self.sequence_to_indices: dict[str, range] = {}
        record_count = 0
        for sequence_entry in self._sequence_entries:
            frame_count = int(sequence_entry["frame_count"])
            self._sequence_offsets.append(record_count)
            self.sequence_to_indices[str(sequence_entry["sequence_id"])] = range(record_count, record_count + frame_count)
            record_count += frame_count
        self._record_count = record_count

    def _uses_lazy_records(self) -> bool:
        return True

    def _ensure_worker_local_caches(self) -> None:
        pid = os.getpid()
        if self._cache_pid == pid:
            return
        self._intrinsics_cache.clear()
        self._camera_pose_cache.clear()
        self._sequence_runtime_cache.clear()
        self._frame_annotation_cache.clear()
        self._record_cache.clear()
        self._cache_pid = pid

    def _sequence_runtime(self, sequence_entry: dict[str, Any]) -> dict[str, Any]:
        """Cache immutable trajectory and calibration inputs within one worker."""
        self._ensure_worker_local_caches()
        cache_key = str(sequence_entry["sequence_id"])
        cached = _lru_get(self._sequence_runtime_cache, cache_key)
        if cached is not None:
            return cached
        camera_pose_log = Path(str(sequence_entry["camera_pose_log"]))
        runtime = {
            "intrinsics": self._hoi4d_intrinsics(str(sequence_entry["camera_uid"])),
            "camera_poses": self._hoi4d_camera_poses(camera_pose_log) if camera_pose_log.exists() else None,
        }
        return _lru_put(
            self._sequence_runtime_cache,
            cache_key,
            runtime,
            limit=_HOI4D_SEQUENCE_RUNTIME_CACHE_LIMIT,
        )

    def _frame_annotations(self, sequence_entry: dict[str, Any], frame_id: int) -> dict[str, Any]:
        """Read raw HOI4D hand/object files once without caching derived GT."""
        self._ensure_worker_local_caches()
        sequence_id = str(sequence_entry["sequence_id"])
        cache_key = f"{sequence_id}\0{frame_id}"
        cached = _lru_get(self._frame_annotation_cache, cache_key)
        if cached is not None:
            return cached
        hands: dict[str, Any] = {}
        for hand_dir_value, side in ((sequence_entry["left_dir"], "left"), (sequence_entry["right_dir"], "right")):
            hand_path = Path(str(hand_dir_value)) / f"{frame_id}.pickle"
            if not hand_path.is_file():
                continue
            with hand_path.open("rb") as handle:
                hands[side] = pickle.load(handle, encoding="latin1")
        object_pose_payload = None
        object_pose_path = _hoi4d_object_pose_path(self.root, sequence_id, frame_id)
        if object_pose_path is not None:
            try:
                object_pose_payload = _load_json(object_pose_path)
            except (OSError, ValueError, json.JSONDecodeError):
                object_pose_payload = None
        return _lru_put(
            self._frame_annotation_cache,
            cache_key,
            {"hands": hands, "object_pose_payload": object_pose_payload},
            limit=_HOI4D_FRAME_ANNOTATION_CACHE_LIMIT,
        )

    def _load_hoi4d_sequence_entries(self) -> list[dict[str, Any]]:
        source_paths = [
            self.root / "HOI4D_release",
            self.root / "HOI4D_depth_video",
            self.root / "HOI4D_annotations",
            self.root / "Hand_pose",
            self.root / "camera_params",
        ]

        def _cached_source_paths(sequence_entries: list[dict[str, Any]]) -> list[Path]:
            cached_paths = list(source_paths)
            for sequence_entry in sequence_entries:
                cached_paths.extend(
                    [
                        Path(str(sequence_entry["rgb_video"])),
                        Path(str(sequence_entry["depth_video"])),
                        Path(str(sequence_entry["left_dir"])),
                        Path(str(sequence_entry["right_dir"])),
                    ]
                )
                if "camera_pose_log" in sequence_entry:
                    cached_paths.append(Path(str(sequence_entry["camera_pose_log"])))
            return cached_paths

        def _scan_sequence_entries() -> list[dict[str, Any]]:
            sequence_entries: list[dict[str, Any]] = []
            rgb_videos = sorted(self.root.glob("HOI4D_release/**/align_rgb/image.mp4"))
            for rgb_video in rgb_videos:
                rel = rgb_video.relative_to(self.root / "HOI4D_release")
                seq_root = rel.parent.parent
                depth_video = self.root / "HOI4D_depth_video" / seq_root / "align_depth" / "depth_video.avi"
                camera_pose_log = self.root / "HOI4D_annotations" / seq_root / "3Dseg" / "output.log"
                left_dir = self.root / "Hand_pose" / "handpose_left_hand" / seq_root
                right_dir = self.root / "Hand_pose" / "handpose_right_hand" / seq_root
                frame_ids = {int(path.stem) for path in left_dir.glob("*.pickle")} | {
                    int(path.stem) for path in right_dir.glob("*.pickle")
                }
                if not frame_ids:
                    continue
                sequence_entries.append(
                    {
                        "sequence_id": str(seq_root),
                        "camera_uid": seq_root.parts[0],
                        "rgb_video": str(rgb_video),
                        "depth_video": str(depth_video),
                        "camera_pose_log": str(camera_pose_log),
                        "left_dir": str(left_dir),
                        "right_dir": str(right_dir),
                        "frame_ids": np.asarray(sorted(frame_ids), dtype=np.int32),
                        "frame_count": len(frame_ids),
                    }
                )
            return sequence_entries

        return _load_or_build_sequence_entry_cache(
            self._index_cache_path,
            source_paths,
            _scan_sequence_entries,
            cached_source_paths_builder=_cached_source_paths,
            version=7,
        )

    def _hoi4d_intrinsics(self, camera_uid: str) -> np.ndarray:
        cached = self._intrinsics_cache.get(camera_uid)
        if cached is not None:
            return cached
        intrinsics = np.load(self.root / "camera_params" / camera_uid / "intrin.npy").astype(np.float32)
        self._intrinsics_cache[camera_uid] = intrinsics
        return intrinsics

    def _hoi4d_camera_poses(self, path: Path) -> list[np.ndarray]:
        cache_key = str(path)
        cached = self._camera_pose_cache.get(cache_key)
        if cached is not None:
            return cached
        camera_poses = _parse_open3d_camera_trajectory_log(path)
        self._camera_pose_cache[cache_key] = camera_poses
        return camera_poses

    def camera_pose_for_index(self, index: int) -> np.ndarray | None:
        """Read a trajectory pose without materializing hand/object annotations."""
        index = self._normalize_index(index)
        sequence_entry, local_index = _sequence_entry_for_index(
            self._sequence_offsets,
            self._sequence_entries,
            index,
        )
        frame_id = int(sequence_entry["frame_ids"][local_index])
        poses = self._sequence_runtime(sequence_entry)["camera_poses"]
        if poses is None:
            return None
        pose_index = max(frame_id - 1, 0)
        return poses[pose_index] if pose_index < len(poses) else None

    def audit_metadata_for_index(self, index: int) -> dict[str, Any]:
        """Expose selection metadata without opening per-frame hand/object files."""
        index = self._normalize_index(index)
        sequence_entry, local_index = _sequence_entry_for_index(
            self._sequence_offsets,
            self._sequence_entries,
            index,
        )
        frame_id = int(sequence_entry["frame_ids"][local_index])
        # ``frame_ids`` is built from the union of available MANO hand-pose
        # files, so these frames have hand supervision by construction.
        return {
            "sequence_id": str(sequence_entry["sequence_id"]),
            "frame_id": str(frame_id),
            "has_mano": True,
            "has_3d_joints": False,
            "hand_annos": [True],
            "extras": {},
        }

    def prefetch_sequence(self, sequence_index: int) -> None:
        sequence_entry = self._sequence_entries[int(sequence_index)]
        refs = [MediaRef(kind="video_frame", path=str(sequence_entry["rgb_video"]), frame_index=0)]
        depth_video = Path(str(sequence_entry["depth_video"]))
        if depth_video.is_file():
            refs.append(MediaRef(kind="hoi4d_depth_video", path=str(depth_video), frame_index=0))
        prewarm_media_refs(refs)
        self._sequence_runtime(sequence_entry)

    def _get_record(self, index: int) -> FrameRecord:
        self._ensure_worker_local_caches()
        index = self._normalize_index(index)
        cached = self._record_cache.get(index)
        if cached is not None:
            self._record_cache.move_to_end(index)
            return cached
        sequence_entry, local_index = _sequence_entry_for_index(self._sequence_offsets, self._sequence_entries, index)
        record = self._materialize_record(sequence_entry, local_index)
        self._record_cache[index] = record
        self._record_cache.move_to_end(index)
        cache_limit = self._record_cache_limit()
        if cache_limit is not None and cache_limit > 0:
            while len(self._record_cache) > cache_limit:
                self._record_cache.popitem(last=False)
        return record

    def __len__(self) -> int:
        return self._record_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        return BaseFrameDataset.__getitem__(self, index)

    def _materialize_record(self, sequence_entry: dict[str, Any], local_index: int) -> FrameRecord:
        frame_id = int(sequence_entry["frame_ids"][local_index])
        raw_annotations = self._frame_annotations(sequence_entry, frame_id)
        hand_annos: list[HandAnnotation] = []
        for side, payload in raw_annotations["hands"].items():
            hand_annos.append(
                HandAnnotation(
                    side=side,
                    visible=True,
                    joints_2d=np.asarray(payload.get("kps2D"), dtype=np.float32).copy(),
                    mano_pose=np.asarray(payload.get("poseCoeff"), dtype=np.float32).copy(),
                    mano_pose_format="axis_angle_full",
                    mano_betas=np.asarray(payload.get("beta"), dtype=np.float32).copy(),
                    mano_trans=np.asarray(payload.get("trans"), dtype=np.float32).copy(),
                )
            )
        left_count, right_count = _count_sides(hand_annos)
        runtime = self._sequence_runtime(sequence_entry)
        intrinsics = runtime["intrinsics"]
        camera_pose = None
        poses = runtime["camera_poses"]
        if poses is not None:
            pose_index = max(frame_id - 1, 0)
            if pose_index < len(poses):
                camera_pose = poses[pose_index]
        depth_video = Path(str(sequence_entry["depth_video"]))
        depth_ref = (
            MediaRef(kind="hoi4d_depth_video", path=str(depth_video), frame_index=max(frame_id - 1, 0))
            if depth_video.is_file()
            else None
        )
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split="train",
            sequence_id=str(sequence_entry["sequence_id"]),
            frame_id=str(frame_id),
            temporal_index=frame_id,
            view_name="egocentric",
            is_egocentric=True,
            rgb_ref=MediaRef(kind="video_frame", path=str(sequence_entry["rgb_video"]), frame_index=max(frame_id - 1, 0)),
            depth_ref=depth_ref,
            depth_mode="hoi4d_depth_video" if depth_ref is not None else None,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=depth_ref, intrinsics=intrinsics, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={
                "raw_depth_video": str(sequence_entry["depth_video"]),
                "scene_objects": _hoi4d_scene_objects(
                    self.root,
                    str(sequence_entry["sequence_id"]),
                    frame_id=frame_id,
                    object_pose_payload=raw_annotations["object_pose_payload"],
                ),
            },
        )


class OakInkV2FrameDataset(BaseFrameDataset):
    dataset_name = "oakink_v2"
    base_dataset_name = "oakink_v2"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        manifest_path: str | Path | None = None,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self.manifest_path = Path(manifest_path) if manifest_path is not None else Path(root) / "oakink_preview_manifest.jsonl"
        manifest_cache_key = hashlib.sha1(str(self.manifest_path.resolve(strict=False)).encode("utf-8")).hexdigest()[:12]
        self._index_cache_path = _light_index_cache_path(_as_path(root), f"oakink_v2_frame_dataset_{split}_{manifest_cache_key}")
        self._preview_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._sequence_image_cache: OrderedDict[str, list[Path]] = OrderedDict()
        self._sequence_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._record_cache: OrderedDict[int, FrameRecord] = OrderedDict()
        self._records_materialized: list[FrameRecord] | None = None
        self._record_index: list[dict[str, Any]] | None = None
        self._sequence_entries = self._load_oakink_sequence_entries()
        self._sequence_offsets: list[int] = []
        self.sequence_to_indices: dict[str, range] = {}
        record_count = 0
        for sequence_entry in self._sequence_entries:
            frame_count = int(sequence_entry["frame_count"])
            self._sequence_offsets.append(record_count)
            self.sequence_to_indices[str(sequence_entry["sequence_id"])] = range(record_count, record_count + frame_count)
            record_count += frame_count
        self._record_count = record_count

    def _uses_lazy_records(self) -> bool:
        return True

    def _oakink_manifest_rows(self) -> list[dict[str, Any]]:
        if self.manifest_path.exists():
            return _load_jsonl(self.manifest_path)
        rows: list[dict[str, Any]] = []
        for tar_path in sorted((self.root / "data").glob("*.tar")):
            preview_path = self.root / "anno_preview" / f"{tar_path.stem}.pkl"
            if not preview_path.is_file():
                continue
            camera_id, frame_ids = self._oakink_tar_camera_frames(tar_path)
            if camera_id is None or not frame_ids:
                continue
            rows.append({
                "sequence_name": tar_path.stem,
                "preview_pkl": str(preview_path),
                "tar_path": str(tar_path),
                "egocentric_camera_id": camera_id,
                "camera_id_to_view": {camera_id: "egocentric"},
                "object_ids": [],
                "frame_count": len(frame_ids),
                "mocap_frame_count": None,
                "raw_mano_frame_count": None,
                "seq_beg": min(frame_ids),
                "seq_end": max(frame_ids),
                "frame_ids": frame_ids,
            })
        if rows:
            return rows
        raise FileNotFoundError(f"缺少 OakInk preview manifest 或可用 preview/tar 对: {self.manifest_path}")

    @staticmethod
    def _oakink_tar_camera_frames(tar_path: Path) -> tuple[str | None, list[int]]:
        camera_to_frames: dict[str, list[int]] = defaultdict(list)
        with tarfile.open(tar_path, "r") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".png"):
                    continue
                parts = Path(member.name).parts
                if len(parts) != 3:
                    continue
                try:
                    camera_to_frames[parts[1]].append(int(Path(parts[2]).stem))
                except ValueError:
                    continue
        frames = sorted(set(camera_to_frames.get(_OAKINK_EGOCENTRIC_CAMERA_ID, [])))
        if frames:
            return _OAKINK_EGOCENTRIC_CAMERA_ID, frames
        if len(camera_to_frames) == 1:
            camera_id, only_frames = next(iter(camera_to_frames.items()))
            return camera_id, sorted(set(only_frames))
        return None, []

    def _oakink_manifest_counts(self, row: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
        return (
            _optional_int(row.get("frame_count")),
            _optional_int(row.get("mocap_frame_count")),
            _optional_int(row.get("raw_mano_frame_count")),
        )

    def _oakink_manifest_frame_bounds(self, row: dict[str, Any]) -> tuple[int, int] | None:
        seq_beg = _optional_int(row.get("seq_beg"))
        seq_end = _optional_int(row.get("seq_end"))
        frame_count, mocap_frame_count, raw_mano_frame_count = self._oakink_manifest_counts(row)
        if seq_beg is None or seq_end is None or frame_count is None:
            return None
        if seq_end < seq_beg:
            return None
        if seq_end - seq_beg + 1 != frame_count:
            return None
        if mocap_frame_count != frame_count or raw_mano_frame_count != frame_count:
            return None
        return seq_beg, seq_end

    def _oakink_sequence_entry_base(
        self,
        row: dict[str, Any],
        seq_root: Path,
        egocentric_view_name: str,
    ) -> dict[str, Any]:
        return {
            "split": "train",
            "sequence_id": str(row["sequence_name"]),
            "view_name": egocentric_view_name,
            "camera_id": str(row.get("egocentric_camera_id", "")),
            "preview_pkl": str(self._oakink_manifest_path(row["preview_pkl"])),
            "seq_root": str(seq_root),
            "tar_path": str(self._oakink_manifest_path(row.get("tar_path", ""))),
            "object_ids": row["object_ids"],
            "seq_beg": _optional_int(row.get("seq_beg")),
            "seq_end": _optional_int(row.get("seq_end")),
        }

    def _oakink_manifest_path(self, value: Any) -> Path:
        """Rebase stale absolute manifest paths after a dataset-root migration."""
        path = Path(str(value))
        if path.is_file():
            return path
        for component in ("data", "anno_preview", "extracted_sequences"):
            if component in path.parts:
                return self.root.joinpath(*path.parts[path.parts.index(component) :])
        return path

    def _oakink_sequence_entry_from_manifest_row(
        self,
        row: dict[str, Any],
        seq_root: Path,
        egocentric_view_name: str,
    ) -> dict[str, Any] | None:
        frame_bounds = self._oakink_manifest_frame_bounds(row)
        if frame_bounds is None:
            return None
        seq_beg, seq_end = frame_bounds
        sequence_entry = self._oakink_sequence_entry_base(row, seq_root, egocentric_view_name)
        sequence_entry["frame_count"] = seq_end - seq_beg + 1
        sequence_entry["index_mode"] = "manifest_bounds"
        return sequence_entry

    def _oakink_sequence_entry_from_manifest_counts(
        self,
        row: dict[str, Any],
        seq_root: Path,
        egocentric_view_name: str,
    ) -> dict[str, Any] | None:
        frame_count, _, _ = self._oakink_manifest_counts(row)
        if frame_count is None or frame_count <= 0:
            return None
        sequence_entry = self._oakink_sequence_entry_base(row, seq_root, egocentric_view_name)
        sequence_entry["frame_count"] = frame_count
        sequence_entry["index_mode"] = "manifest_count"
        return sequence_entry

    def _oakink_sequence_entry_from_preview_row(
        self,
        row: dict[str, Any],
        seq_root: Path,
        egocentric_view_name: str,
    ) -> dict[str, Any]:
        frame_ids = [int(frame_id) for frame_id in row.get("frame_ids", [])]
        sequence_entry = self._oakink_sequence_entry_base(row, seq_root, egocentric_view_name)
        sequence_entry["frame_count"] = len(frame_ids)
        sequence_entry["index_mode"] = "preview_fallback"
        sequence_entry["frame_ids"] = frame_ids
        sequence_entry["tar_native"] = self._oakink_manifest_path(row.get("tar_path", "")).is_file()
        return sequence_entry

    def _load_oakink_sequence_entries(self) -> list[dict[str, Any]]:
        if self._index_cache_path.exists():
            try:
                payload = _load_pickle(self._index_cache_path)
                sequence_entries = payload.get("sequence_entries")
                cached_sources = payload.get("sources")
                if (
                    payload.get("version") == 6
                    and isinstance(cached_sources, list)
                    and _light_index_sources_match(cached_sources)
                    and isinstance(sequence_entries, list)
                ):
                    return sequence_entries
            except (OSError, EOFError, pickle.UnpicklingError, ValueError, ModuleNotFoundError):
                pass

        def _scan_sequence_entries() -> tuple[list[dict[str, Any]], list[Path]]:
            manifest_rows = self._oakink_manifest_rows()
            sequence_entries: list[dict[str, Any]] = []
            source_paths: list[Path] = [self.manifest_path]
            for row in manifest_rows:
                sequence_name = str(row["sequence_name"])
                egocentric_camera_id = str(row["egocentric_camera_id"])
                egocentric_view_name = str(row.get("camera_id_to_view", {}).get(egocentric_camera_id, "egocentric"))
                seq_root = self.root / "extracted_sequences" / sequence_name / sequence_name / egocentric_camera_id
                source_paths.append(self._oakink_manifest_path(row["preview_pkl"]))
                source_paths.append(seq_root)
                if row.get("tar_path"):
                    source_paths.append(self._oakink_manifest_path(row["tar_path"]))
                sequence_entry = self._oakink_sequence_entry_from_manifest_row(row, seq_root, egocentric_view_name)
                if sequence_entry is not None:
                    sequence_entries.append(sequence_entry)
                    continue
                if row.get("frame_ids"):
                    sequence_entries.append(self._oakink_sequence_entry_from_preview_row(row, seq_root, egocentric_view_name))
                    continue
                sequence_entry = self._oakink_sequence_entry_from_manifest_counts(row, seq_root, egocentric_view_name)
                if sequence_entry is not None:
                    sequence_entries.append(sequence_entry)
                    continue
                sequence_entries.append(self._oakink_sequence_entry_from_preview_row(row, seq_root, egocentric_view_name))
            return sequence_entries, source_paths

        sequence_entries, source_paths = _scan_sequence_entries()
        source_signatures = [_light_index_source_signature(path) for path in source_paths]
        self._index_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._index_cache_path.open("wb") as handle:
            pickle.dump(
                {"version": 6, "sources": source_signatures, "sequence_entries": sequence_entries},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return sequence_entries

    def _oakink_sequence_image_paths(self, seq_root: Path) -> list[Path]:
        cache_key = str(seq_root)
        cached = _lru_get(self._sequence_image_cache, cache_key)
        if cached is not None:
            return cached
        image_paths = sorted(seq_root.glob("*.png"))
        return _lru_put(self._sequence_image_cache, cache_key, image_paths, limit=_OAKINK_SEQUENCE_IMAGE_CACHE_LIMIT)

    def _oakink_runtime_preview_path(self, preview_pkl: Path) -> Path:
        identity = hashlib.sha1(str(preview_pkl.resolve(strict=False)).encode("utf-8")).hexdigest()[:16]
        return self.root / ".light_index_cache" / "oakink_v2_runtime_preview" / f"{preview_pkl.stem}.{identity}.pkl"

    @staticmethod
    def _oakink_runtime_preview_projection(preview: dict[str, Any]) -> dict[str, Any]:
        """Keep only source annotations that an ego RGB frame can consume."""
        rgb_frame_ids = [int(value) for value in preview.get("frame_id_list", [])]
        mocap_frame_ids = [int(value) for value in preview.get("mocap_frame_id_list", [])]
        raw_mano_source = preview.get("raw_mano", {})
        used_mocap_keys: set[int] = set()
        raw_mano: dict[int, Any] = {}
        for frame_id in rgb_frame_ids:
            try:
                mocap_key = _oakink_mocap_frame_for_rgb_frame(frame_id, mocap_frame_ids)
            except KeyError:
                continue
            source_value = raw_mano_source.get(mocap_key) if isinstance(raw_mano_source, dict) else None
            if source_value is None and isinstance(raw_mano_source, dict):
                source_value = raw_mano_source.get(str(mocap_key))
            if source_value is not None:
                used_mocap_keys.add(mocap_key)
                raw_mano[mocap_key] = source_value

        object_transforms: dict[Any, dict[int, Any]] = {}
        source_transforms = preview.get("obj_transf", {})
        if isinstance(source_transforms, dict):
            for object_id in preview.get("obj_list", []):
                per_frame = source_transforms.get(object_id)
                if per_frame is None:
                    per_frame = source_transforms.get(str(object_id))
                if not isinstance(per_frame, dict):
                    continue
                selected: dict[int, Any] = {}
                for mocap_key in used_mocap_keys:
                    transform = per_frame.get(mocap_key)
                    if transform is None:
                        transform = per_frame.get(str(mocap_key))
                    if transform is not None:
                        selected[mocap_key] = transform
                object_transforms[object_id] = selected

        return {
            "cam_def": preview.get("cam_def", {}),
            "frame_id_list": rgb_frame_ids,
            "cam_intr": preview.get("cam_intr", {}),
            "cam_extr": preview.get("cam_extr", {}),
            # This small id list also encodes whether the release uses
            # zero-based mocap time, so it must not be filtered with payloads.
            "mocap_frame_id_list": mocap_frame_ids,
            "obj_list": list(preview.get("obj_list", [])),
            "obj_transf": object_transforms,
            "raw_mano": raw_mano,
        }

    def _load_oakink_runtime_preview(self, preview_pkl: Path) -> dict[str, Any] | None:
        sidecar_path = self._oakink_runtime_preview_path(preview_pkl)
        if not sidecar_path.is_file():
            return None
        try:
            payload = _load_pickle(sidecar_path)
        except (OSError, EOFError, pickle.UnpicklingError, ModuleNotFoundError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _OAKINK_RUNTIME_PREVIEW_VERSION
            or payload.get("source") != _light_index_source_signature(preview_pkl)
            or not isinstance(payload.get("preview"), dict)
        ):
            return None
        return payload["preview"]

    def _write_oakink_runtime_preview(self, preview_pkl: Path, preview: dict[str, Any]) -> dict[str, Any]:
        projected = self._oakink_runtime_preview_projection(preview)
        sidecar_path = self._oakink_runtime_preview_path(preview_pkl)
        temporary_path = sidecar_path.with_name(f".{sidecar_path.name}.{os.getpid()}.tmp")
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_pickle(
                temporary_path,
                {
                    "version": _OAKINK_RUNTIME_PREVIEW_VERSION,
                    "source": _light_index_source_signature(preview_pkl),
                    "preview": projected,
                },
            )
            os.replace(temporary_path, sidecar_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return projected

    def _oakink_preview_payload(self, preview_pkl: Path) -> dict[str, Any]:
        cache_key = str(preview_pkl)
        cached = _lru_get(self._preview_cache, cache_key)
        if cached is not None:
            return cached
        sidecar = self._load_oakink_runtime_preview(preview_pkl)
        if sidecar is not None:
            return _lru_put(self._preview_cache, cache_key, sidecar, limit=_OAKINK_PREVIEW_CACHE_LIMIT)
        with preview_pkl.open("rb") as handle:
            preview = pickle.load(handle)
        return _lru_put(
            self._preview_cache,
            cache_key,
            self._write_oakink_runtime_preview(preview_pkl, preview),
            limit=_OAKINK_PREVIEW_CACHE_LIMIT,
        )

    def _oakink_sequence_cache_entries(self, sequence_entry: dict[str, Any]) -> list[dict[str, Any]]:
        cache_key = str(sequence_entry["sequence_id"])
        cached = _lru_get(self._sequence_cache, cache_key)
        if cached is not None:
            return cached
        seq_root = Path(str(sequence_entry["seq_root"]))
        tar_path = Path(str(sequence_entry.get("tar_path", "")))
        image_paths = self._oakink_sequence_image_paths(seq_root)
        image_path_by_frame = {int(path.stem): path for path in image_paths}
        frame_count = int(sequence_entry["frame_count"])
        index_mode = str(sequence_entry["index_mode"])
        local_entries: list[dict[str, Any]] = []
        if index_mode == "manifest_bounds":
            seq_beg = int(sequence_entry["seq_beg"])
            for local_index in range(frame_count):
                frame_index = seq_beg + local_index
                image_path = image_path_by_frame.get(frame_index)
                if image_path is None and not tar_path.is_file():
                    raise IndexError(f"OakInk frame 缺少图片: {sequence_entry['sequence_id']} {frame_index}")
                local_entries.append(
                    {
                        "frame_id": str(frame_index),
                        "temporal_index": frame_index,
                        "image_path": "" if image_path is None else str(image_path),
                        "mano_index": local_index,
                        "tar_path": str(tar_path),
                    }
                )
        elif index_mode == "manifest_count":
            if len(image_paths) < frame_count and not tar_path.is_file():
                raise IndexError(f"OakInk 图片数量不足: {sequence_entry['sequence_id']} {len(image_paths)} < {frame_count}")
            preview = self._oakink_preview_payload(Path(str(sequence_entry["preview_pkl"])))
            for local_index in range(frame_count):
                image_path = image_paths[local_index] if image_paths else None
                frame_index = int(image_path.stem) if image_path is not None else int(preview["frame_id_list"][local_index])
                mano_index = frame_index
                local_entries.append(
                    {
                        "frame_id": str(frame_index),
                        "temporal_index": frame_index,
                        "image_path": "" if image_path is None else str(image_path),
                        "mano_index": int(mano_index),
                        "mano_frame_id": int(frame_index),
                        "tar_path": str(sequence_entry.get("tar_path", "")),
                    }
                )
        else:
            frame_ids = sequence_entry["frame_ids"]
            for local_index, frame_index in enumerate(frame_ids):
                image_path = image_path_by_frame.get(int(frame_index))
                if image_path is None and not tar_path.is_file():
                    raise IndexError(f"OakInk frame 缺少图片: {sequence_entry['sequence_id']} {frame_index}")
                local_entries.append(
                    {
                        "frame_id": str(frame_index),
                        "temporal_index": int(frame_index),
                        "image_path": "" if image_path is None else str(image_path),
                        "mano_index": int(local_index),
                        "tar_path": str(sequence_entry.get("tar_path", "")),
                    }
                )
        for entry in local_entries:
            if not entry.get("tar_path"):
                entry["tar_path"] = str(sequence_entry.get("tar_path", ""))
        return _lru_put(self._sequence_cache, cache_key, local_entries, limit=_OAKINK_SEQUENCE_CACHE_LIMIT)

    def prefetch_sequence(self, sequence_index: int) -> None:
        sequence_entry = self._sequence_entries[int(sequence_index)]
        self._oakink_preview_payload(Path(str(sequence_entry["preview_pkl"])))
        self._oakink_sequence_cache_entries(sequence_entry)

    def _oakink_sequence_entry_for_index(self, index: int) -> tuple[dict[str, Any], int]:
        sequence_index = bisect_right(self._sequence_offsets, index) - 1
        sequence_entry = self._sequence_entries[sequence_index]
        local_index = index - self._sequence_offsets[sequence_index]
        return sequence_entry, local_index

    def _get_record(self, index: int) -> FrameRecord:
        index = self._normalize_index(index)
        cached = self._record_cache.get(index)
        if cached is not None:
            self._record_cache.move_to_end(index)
            return cached
        sequence_entry, local_index = self._oakink_sequence_entry_for_index(index)
        local_entries = self._oakink_sequence_cache_entries(sequence_entry)
        if local_index >= len(local_entries):
            raise IndexError(f"OakInk local_index 越界: {local_index} >= {len(local_entries)}")
        record = self._materialize_record(sequence_entry, local_entries[local_index])
        self._record_cache[index] = record
        self._record_cache.move_to_end(index)
        cache_limit = self._record_cache_limit()
        if cache_limit is not None and cache_limit > 0:
            while len(self._record_cache) > cache_limit:
                self._record_cache.popitem(last=False)
        return record

    def __len__(self) -> int:
        return self._record_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        return BaseFrameDataset.__getitem__(self, index)

    def _materialize_record(self, sequence_entry: dict[str, Any], local_entry: dict[str, Any]) -> FrameRecord:
        view_name = str(sequence_entry["view_name"])
        preview = self._oakink_preview_payload(Path(str(sequence_entry["preview_pkl"])))
        frame_index = int(local_entry["temporal_index"])
        image_path = Path(str(local_entry["image_path"]))
        expected_camera_id = str(sequence_entry["camera_id"])
        camera_id_to_view = preview.get("cam_def", {})
        if str(camera_id_to_view.get(expected_camera_id, "")) != "egocentric":
            raise ValueError(
                f"OakInk camera UID 与 preview 不一致: {sequence_entry['sequence_id']} "
                f"{expected_camera_id} -> {camera_id_to_view.get(expected_camera_id)!r}"
            )
        rgb_frame_ids = [int(value) for value in preview.get("frame_id_list", [])]
        mocap_frame_ids = [int(value) for value in preview.get("mocap_frame_id_list", [])]
        if frame_index not in rgb_frame_ids:
            raise KeyError(
                f"OakInk RGB frame 不在 preview.frame_id_list: {sequence_entry['sequence_id']} frame={frame_index}"
            )
        try:
            mano_key = _oakink_mocap_frame_for_rgb_frame(frame_index, mocap_frame_ids)
        except KeyError as exc:
            raise KeyError(
                f"OakInk RGB/MANO frame 未对齐: {sequence_entry['sequence_id']} frame={frame_index}"
            ) from exc
        raw_mano_by_frame = preview.get("raw_mano", {})
        raw_mano = raw_mano_by_frame.get(mano_key)
        if raw_mano is None:
            raise KeyError(
                f"OakInk RGB/MANO frame 未对齐: {sequence_entry['sequence_id']} frame={frame_index} mocap={mano_key}"
            )
        intrinsics = np.asarray(preview["cam_intr"][view_name][frame_index], dtype=np.float32)
        camera_pose = np.asarray(preview["cam_extr"][view_name][frame_index], dtype=np.float32)
        object_ids = list(preview.get("obj_list", []))
        scene_objects = _oakink_scene_objects(
            self.root,
            object_ids,
            preview.get("obj_transf", {}),
            mano_key,
            camera_pose,
        )

        hand_annos = [
            HandAnnotation(
                side="right",
                visible=True,
                mano_pose=_quat16_to_axis_angle_full(np.asarray(raw_mano.get("rh__pose_coeffs")).squeeze(0)),
                mano_pose_format="axis_angle_full",
                mano_betas=np.asarray(raw_mano.get("rh__betas")).squeeze(0),
                mano_trans=np.asarray(raw_mano.get("rh__tsl")).squeeze(0),
                extras={"mano_external_transform": camera_pose, "mano_trans_is_root_position": True},
            ),
            HandAnnotation(
                side="left",
                visible=True,
                mano_pose=_quat16_to_axis_angle_full(np.asarray(raw_mano.get("lh__pose_coeffs")).squeeze(0)),
                mano_pose_format="axis_angle_full",
                mano_betas=np.asarray(raw_mano.get("lh__betas")).squeeze(0),
                mano_trans=np.asarray(raw_mano.get("lh__tsl")).squeeze(0),
                extras={"mano_external_transform": camera_pose, "mano_trans_is_root_position": True},
            ),
        ]
        left_count, right_count = _count_sides(hand_annos)
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split=str(sequence_entry["split"]),
            sequence_id=str(sequence_entry["sequence_id"]),
            frame_id=str(local_entry["frame_id"]),
            temporal_index=frame_index,
            view_name=view_name,
            is_egocentric=True,
            rgb_ref=(
                MediaRef(
                    kind="tar_member",
                    path=str(local_entry["tar_path"]),
                    member=f"{sequence_entry['sequence_id']}/{sequence_entry['camera_id']}/{frame_index:06d}.png",
                )
                if local_entry.get("tar_path") and not image_path.is_file()
                else MediaRef(kind="path", path=str(image_path))
            ),
            depth_ref=None,
            depth_mode=None,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=None, intrinsics=intrinsics, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={
                "object_ids": object_ids,
                "seq_beg": _optional_int(sequence_entry["seq_beg"]),
                "seq_end": _optional_int(sequence_entry["seq_end"]),
                "scene_objects": scene_objects,
            },
        )


class WhimFrameDataset(BaseFrameDataset):
    dataset_name = "whim"
    base_dataset_name = "whim"
    frame_rate_fps = 30

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        segment_gap_ms: int = 2000,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.segment_gap_ms = segment_gap_ms
        self.segment_gap_frames = max(1, int(round(float(segment_gap_ms) * float(self.frame_rate_fps) / 1000.0)))
        self._index_cache_path = _light_index_cache_path(
            _as_path(root),
            f"whim_frame_dataset_{split}_gap{segment_gap_ms}_fps{self.frame_rate_fps}_480p",
        )
        super().__init__(root, split, load_rgb=load_rgb, load_depth=load_depth)

    def _uses_lazy_records(self) -> bool:
        return True

    def _split_names(self) -> tuple[str, ...]:
        if self.split == "all":
            return ("train", "test")
        return (self.split,)

    def _split_video_ids(self, split_name: str) -> set[str]:
        split_ids_path = self.root / "whim" / f"{split_name}_video_ids.json"
        if not split_ids_path.exists():
            return set()
        payload = _load_json(split_ids_path)
        if isinstance(payload, dict):
            return {str(key) for key in payload}
        if isinstance(payload, list):
            return {str(item) for item in payload}
        raise ValueError(f"WHIM split video ids 格式错误: {split_ids_path}")

    def _frame_dir_for_video(self, split_name: str, video_name: str) -> Path | None:
        split_root = self.root / "WHIM" / split_name
        for frame_dir in (
            split_root / "anno_resized_max480p" / video_name,
            split_root / "anno" / video_name,
        ):
            if frame_dir.is_dir():
                return frame_dir
        return None

    def _video_path_for_video(self, video_name: str) -> Path:
        return self.root / "Videos" / f"{video_name}.mp4"

    def _build_record_index(self) -> list[dict[str, Any]]:
        split_names = self._split_names()
        source_paths: list[Path] = []
        split_video_dirs: list[tuple[str, list[Path]]] = []
        for split_name in split_names:
            anno_root = self.root / "WHIM" / split_name / "anno"
            split_ids_path = self.root / "whim" / f"{split_name}_video_ids.json"
            allowed_video_ids = self._split_video_ids(split_name)
            video_dirs = sorted(
                path for path in anno_root.glob("*")
                if path.is_dir() and (not allowed_video_ids or path.name in allowed_video_ids)
            )
            split_video_dirs.append((split_name, video_dirs))
            frame_dirs = [
                frame_dir
                for video_dir in video_dirs
                if (frame_dir := self._frame_dir_for_video(split_name, video_dir.name)) is not None
            ]
            video_paths = [self._video_path_for_video(video_dir.name) for video_dir in video_dirs]
            source_paths.extend([split_ids_path, anno_root, *video_dirs, *frame_dirs, *video_paths])

        def _scan_record_index() -> list[dict[str, Any]]:
            record_index: list[dict[str, Any]] = []
            for split_name, video_dirs in split_video_dirs:
                for video_dir in video_dirs:
                    if not self._video_path_for_video(video_dir.name).exists():
                        continue
                    frame_dir = self._frame_dir_for_video(split_name, video_dir.name)
                    if frame_dir is None:
                        continue
                    frame_indices = sorted(int(path.stem) for path in video_dir.glob("*.npy"))
                    if not frame_indices:
                        continue
                    segment_id = 0
                    previous = None
                    for frame_index in frame_indices:
                        rgb_path = frame_dir / f"{frame_index:06d}.jpg"
                        if not rgb_path.exists():
                            continue
                        resized_anno_path = frame_dir / f"{frame_index:06d}.npy"
                        if previous is None or frame_index - previous > self.segment_gap_frames:
                            segment_id += 1
                        previous = frame_index
                        record_index.append(
                            {
                                "split": split_name,
                                "sequence_id": f"{split_name}:{video_dir.name}:segment_{segment_id}",
                                "frame_id": str(frame_index),
                                "temporal_index": frame_index,
                                "video_name": video_dir.name,
                                "anno_path": str(resized_anno_path if resized_anno_path.exists() else video_dir / f"{frame_index:06d}.npy"),
                                "rgb_path": str(rgb_path),
                            }
                        )
            return record_index

        return _load_or_build_light_index_cache(
            self._index_cache_path,
            source_paths,
            _scan_record_index,
            version=6,
        )

    def _materialize_record(self, index_entry: dict[str, Any]) -> FrameRecord:
        payloads = np.load(Path(str(index_entry["anno_path"])), allow_pickle=True)
        hand_annos: list[HandAnnotation] = []
        intrinsics = None
        for hand_index, payload in enumerate(payloads.tolist()):
            side_value = int(np.asarray(payload["side"]).item()) if "side" in payload else -1
            side = "left" if side_value == 0 else "right" if side_value == 1 else None
            mano_dict = payload.get("mano", {})
            extras: dict[str, Any] = {}
            joints_2d = _as_optional_array(payload.get("joints_2d"), shape=(21, 2))
            if joints_2d is None and payload.get("joints_2d") is not None:
                extras["raw_joints_2d"] = np.asarray(payload["joints_2d"], dtype=np.float32)
            if payload.get("joints_3d") is not None:
                extras["raw_joints_3d"] = np.asarray(payload["joints_3d"], dtype=np.float32)
            if side == "left":
                extras["mano_use_right_hand_layer"] = True
                extras["mano_mirror_local_x"] = True
            hand_annos.append(
                HandAnnotation(
                    side=side,
                    hand_index=hand_index,
                    visible=True,
                    bbox_xyxy=np.asarray(payload.get("bbox"), dtype=np.float32),
                    joints_3d=None,
                    joints_2d=joints_2d,
                    mano_global_orient=np.asarray(mano_dict.get("global_orient"), dtype=np.float32) if "global_orient" in mano_dict else None,
                    mano_hand_pose=np.asarray(mano_dict.get("hand_pose"), dtype=np.float32) if "hand_pose" in mano_dict else None,
                    mano_pose=None,
                    mano_pose_format="whim_mano",
                    mano_betas=np.asarray(mano_dict.get("betas"), dtype=np.float32) if "betas" in mano_dict else None,
                    mano_trans=np.asarray(payload.get("trans"), dtype=np.float32),
                    extras=extras,
                )
            )
            if intrinsics is None and "K" in payload:
                intrinsics = np.asarray(payload["K"], dtype=np.float32)
        frame_index = int(index_entry["temporal_index"])
        left_count, right_count = _count_sides(hand_annos)
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split=str(index_entry.get("split", self.split)),
            sequence_id=str(index_entry["sequence_id"]),
            frame_id=str(index_entry["frame_id"]),
            temporal_index=frame_index,
            view_name="third_person",
            is_egocentric=False,
            rgb_ref=MediaRef(
                kind="path",
                path=str(index_entry["rgb_path"]),
            ),
            depth_ref=None,
            depth_mode=None,
            intrinsics=intrinsics,
            camera_pose=None,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=None, intrinsics=intrinsics, camera_pose=None),
            max_left_count=left_count,
            max_right_count=right_count,
        )


class ForeHoiFrameDataset(BaseFrameDataset):
    dataset_name = "forehoi"
    base_dataset_name = "forehoi"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        load_rgb: bool = False,
        load_depth: bool = False,
        tar_view_count: int = 40,
        tar_frames_per_view: int = 155,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.load_rgb = load_rgb
        self.load_depth = load_depth
        if int(tar_view_count) <= 0:
            raise ValueError("tar_view_count must be positive")
        self.tar_view_count = int(tar_view_count)
        if int(tar_frames_per_view) <= 0:
            raise ValueError("tar_frames_per_view must be positive")
        self.tar_frames_per_view = int(tar_frames_per_view)
        self._metadata_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._frame_metadata_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._index_cache_path = _light_index_cache_path(_as_path(root), f"forehoi_frame_dataset_{split}")
        self._record_cache: OrderedDict[int, FrameRecord] = OrderedDict()
        self._records_materialized: list[FrameRecord] | None = None
        self._record_index: list[dict[str, Any]] | None = None
        self._sequence_entries = self._load_forehoi_sequence_entries()
        self._sequence_offsets: list[int] = []
        self.sequence_to_indices: dict[str, range] = {}
        record_count = 0
        for sequence_entry in self._sequence_entries:
            frame_count = int(sequence_entry["frame_count"])
            self._sequence_offsets.append(record_count)
            self.sequence_to_indices[str(sequence_entry["sequence_id"])] = range(record_count, record_count + frame_count)
            record_count += frame_count
        self._record_count = record_count

    def _uses_lazy_records(self) -> bool:
        return True

    def _load_forehoi_sequence_entries(self) -> list[dict[str, Any]]:
        source_paths = [self.root / "graspxl_renders"]

        def _cached_source_paths(sequence_entries: list[dict[str, Any]]) -> list[Path]:
            cached_paths = list(source_paths)
            for sample_dir_str in sorted({str(sequence_entry["sample_dir"]) for sequence_entry in sequence_entries}):
                sample_dir = Path(sample_dir_str)
                cached_paths.extend(
                    [
                        sample_dir / "metadata.npy",
                        sample_dir / "data.tar",
                        sample_dir / "extracted" / sample_dir.name,
                    ]
                )
            return cached_paths

        def _scan_sequence_entries() -> list[dict[str, Any]]:
            sequence_entries: list[dict[str, Any]] = []
            sample_dirs = sorted((self.root / "graspxl_renders").glob("*"))
            for sample_dir in sample_dirs:
                if not sample_dir.is_dir():
                    continue
                metadata_path = sample_dir / "metadata.npy"
                if not metadata_path.exists():
                    continue
                extracted_root = sample_dir / "extracted" / sample_dir.name
                for view_dir in sorted(path for path in extracted_root.glob("view_*") if path.is_dir()):
                    frame_meta_paths = sorted(view_dir.glob("*.meta.json"))
                    if not frame_meta_paths:
                        continue
                    frame_stems = [path.stem.split(".")[0] for path in frame_meta_paths]
                    frame_width = len(frame_stems[0])
                    numeric_frame_ids = [int(frame_stem) for frame_stem in frame_stems]
                    is_contiguous = all(
                        numeric_frame_ids[index] == numeric_frame_ids[0] + index
                        for index in range(len(numeric_frame_ids))
                    )
                    sequence_entry = {
                        "sequence_id": f"{sample_dir.name}:{view_dir.name}",
                        "sample_dir": str(sample_dir),
                        "view_dir": str(view_dir),
                        "view_name": view_dir.name,
                        "frame_count": len(frame_stems),
                        "frame_width": frame_width,
                        "contiguous": is_contiguous,
                    }
                    if is_contiguous:
                        sequence_entry["frame_start"] = numeric_frame_ids[0]
                    else:
                        sequence_entry["frame_ids"] = frame_stems
                    sequence_entries.append(sequence_entry)
                data_tar = sample_dir / "data.tar"
                if data_tar.exists() and not extracted_root.is_dir():
                    frame_count = self.tar_frames_per_view
                    frame_width = max(3, len(str(max(frame_count - 1, 0))))
                    sequence_entries.append(
                        {
                            "sequence_id": sample_dir.name,
                            "sample_dir": str(sample_dir),
                            "data_tar": str(data_tar),
                            "frame_count": frame_count * self.tar_view_count,
                            "frames_per_view": frame_count,
                            "tar_view_count": self.tar_view_count,
                            "frame_width": frame_width,
                            "contiguous": True,
                            "storage": "tar",
                            "frame_start": 0,
                        }
                    )
            return sequence_entries

        return _load_or_build_sequence_entry_cache(
            self._index_cache_path,
            source_paths,
            _scan_sequence_entries,
            cached_source_paths_builder=_cached_source_paths,
            version=8,
        )

    def _forehoi_metadata(self, sample_dir: Path) -> dict[str, Any]:
        cache_key = str(sample_dir)
        cached = _lru_get(self._metadata_cache, cache_key)
        if cached is not None:
            return cached
        metadata = np.load(sample_dir / "metadata.npy", allow_pickle=True).item()
        return _lru_put(self._metadata_cache, cache_key, metadata, limit=_FOREHOI_METADATA_CACHE_LIMIT)

    def _forehoi_frame_metadata(
        self,
        sequence_entry: dict[str, Any],
        local_index: int,
        frame_stem: str,
        view_name: str,
    ) -> dict[str, Any]:
        if sequence_entry.get("storage") != "tar":
            frame_meta_path = Path(str(sequence_entry["view_dir"])) / f"{frame_stem}.meta.json"
            return _load_json(frame_meta_path)
        meta_members = sequence_entry.get("frame_meta_members")
        if meta_members is None:
            member = f"{Path(str(sequence_entry['sample_dir'])).name}/{view_name}/{frame_stem}.meta.json"
        else:
            member = str(meta_members[local_index])
        cache_key = f"{sequence_entry['data_tar']}:{member}"
        cached = _lru_get(self._frame_metadata_cache, cache_key)
        if cached is not None:
            return cached
        with tarfile.open(str(sequence_entry["data_tar"]), "r") as handle:
            file_obj = handle.extractfile(member)
            if file_obj is None:
                raise FileNotFoundError(f"{sequence_entry['data_tar']}:{member}")
            frame_meta = json.load(file_obj)
        return _lru_put(
            self._frame_metadata_cache,
            cache_key,
            frame_meta,
            limit=_FOREHOI_FRAME_METADATA_CACHE_LIMIT,
        )

    def _get_record(self, index: int) -> FrameRecord:
        index = self._normalize_index(index)
        cached = self._record_cache.get(index)
        if cached is not None:
            self._record_cache.move_to_end(index)
            return cached
        sequence_entry, local_index = _sequence_entry_for_index(self._sequence_offsets, self._sequence_entries, index)
        record = self._materialize_record(sequence_entry, local_index)
        self._record_cache[index] = record
        self._record_cache.move_to_end(index)
        cache_limit = self._record_cache_limit()
        if cache_limit is not None and cache_limit > 0:
            while len(self._record_cache) > cache_limit:
                self._record_cache.popitem(last=False)
        return record

    def __len__(self) -> int:
        return self._record_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        return BaseFrameDataset.__getitem__(self, index)

    def _materialize_record(self, sequence_entry: dict[str, Any], local_index: int) -> FrameRecord:
        sample_dir = Path(str(sequence_entry["sample_dir"]))
        if sequence_entry.get("storage") == "tar":
            frames_per_view = int(sequence_entry["frames_per_view"])
            view_index, frame_index = divmod(local_index, frames_per_view)
            view_name = f"view_{view_index}"
            frame_stem = f"{frame_index:0{int(sequence_entry['frame_width'])}d}"
            frame_local_index = frame_index
        elif bool(sequence_entry["contiguous"]):
            frame_index = int(sequence_entry["frame_start"]) + local_index
            frame_stem = f"{frame_index:0{int(sequence_entry['frame_width'])}d}"
            view_name = str(sequence_entry["view_name"])
            frame_local_index = local_index
        else:
            frame_stem = str(sequence_entry["frame_ids"][local_index])
            frame_index = int(frame_stem)
            view_name = str(sequence_entry["view_name"])
            frame_local_index = local_index
        frame_meta = self._forehoi_frame_metadata(sequence_entry, frame_local_index, frame_stem, view_name)
        metadata = self._forehoi_metadata(sample_dir)
        hand_meta = metadata.get("right_hand", {})
        camera_pose = np.asarray(frame_meta.get("camera_pose"), dtype=np.float32)
        hand_annos: list[HandAnnotation] = []
        if hand_meta:
            betas = hand_meta.get("betas")
            mano_betas = np.zeros(10, dtype=np.float32) if betas is None else np.asarray(betas, dtype=np.float32)[frame_index]
            hand_annos.append(
                HandAnnotation(
                    side="right",
                    visible=True,
                    mano_global_orient=np.asarray(hand_meta.get("rot"), dtype=np.float32)[frame_index],
                    mano_hand_pose=np.asarray(hand_meta.get("pose"))[frame_index],
                    mano_pose=None,
                    mano_pose_format="axis_angle_pose45",
                    mano_betas=mano_betas,
                    mano_trans=np.asarray(hand_meta.get("trans"), dtype=np.float32)[frame_index],
                    extras={
                        "mano_flat_hand_mean": False,
                        "mano_external_transform": (
                            _FOREHOI_OPENGL_TO_CV
                            @ np.linalg.inv(camera_pose)
                            @ _FOREHOI_METADATA_TO_WORLD
                        ).astype(np.float32),
                    },
                )
            )
        left_count, right_count = _count_sides(hand_annos)
        intrinsics = None
        if "camera_angle_x" in frame_meta and "width" in frame_meta and "height" in frame_meta:
            intrinsics = _intrinsics_from_horizontal_fov(
                int(frame_meta["width"]),
                int(frame_meta["height"]),
                float(frame_meta["camera_angle_x"]),
            )
        if sequence_entry.get("storage") == "tar":
            rgb_ref = MediaRef(
                kind="tar_member",
                path=str(sequence_entry["data_tar"]),
                member=f"{sample_dir.name}/{view_name}/{frame_stem}.rgb.webp",
            )
        else:
            frame_meta_path = Path(str(sequence_entry["view_dir"])) / f"{frame_stem}.meta.json"
            rgb_ref = MediaRef(kind="path", path=str(frame_meta_path.with_name(f"{frame_stem}.rgb.webp")))
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split="train",
            sequence_id=str(sequence_entry["sequence_id"]),
            frame_id=frame_stem,
            temporal_index=frame_index,
            view_name=view_name,
            is_egocentric=False,
            rgb_ref=rgb_ref,
            depth_ref=None,
            depth_mode=None,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=None, intrinsics=None, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={"obj_pose": frame_meta.get("obj_pose")},
        )


class ReInterHandFrameDataset(BaseFrameDataset):
    dataset_name = "reinterhand"
    base_dataset_name = "reinterhand"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        manifest_path: str | Path | None = None,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else Path(root) / "reinterhand_ego_manifest.jsonl"
        self._index_cache_path = _light_index_cache_path(_as_path(root), f"reinterhand_frame_dataset_{split}")
        self._cache_pid = os.getpid()
        self._json_payload_cache: OrderedDict[str, Any] = OrderedDict()
        super().__init__(root, split, load_rgb=load_rgb, load_depth=load_depth)

    def _uses_lazy_records(self) -> bool:
        return True

    def _ensure_worker_local_caches(self) -> None:
        pid = os.getpid()
        if self._cache_pid == pid:
            return
        self._json_payload_cache.clear()
        self._record_cache.clear()
        self._cache_pid = pid

    def _cached_json_payload(self, path: Path) -> Any:
        self._ensure_worker_local_caches()
        cache_key = str(path)
        cached = _lru_get(self._json_payload_cache, cache_key)
        if cached is not None:
            return cached
        return _lru_put(
            self._json_payload_cache,
            cache_key,
            _load_json(path),
            limit=_REINTERHAND_JSON_CACHE_LIMIT,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = BaseFrameDataset.__getitem__(self, index)
        if sample.get("rgb") is None:
            return sample
        extras = sample.get("extras", {})
        if not bool(extras.get("rgb_rectified", False)):
            return sample
        sample["rgb"] = _undistort_fisheye_rgb(
            sample["rgb"],
            extras["raw_intrinsics"],
            extras["camera_distortion"],
            sample["intrinsics"],
        )
        return sample

    def _build_record_index(self) -> list[dict[str, Any]]:
        def _scan_record_index() -> tuple[list[dict[str, Any]], list[Path]]:
            if self.manifest_path.exists():
                manifest_rows = _load_jsonl(self.manifest_path)
                source_paths: list[Path] = [self.manifest_path]
            else:
                capture_dirs = sorted(path for path in self.root.glob("m--*--two-hands") if path.is_dir())
                archive_parts = list(self.root.rglob("*.tar.gz*"))
                if not capture_dirs:
                    if archive_parts:
                        raise FileNotFoundError(
                            "ReInterHand 仅发现分卷 archive；需先完成官方 checksum 校验并解压，"
                            "得到 cam_params、mano_fits/params 和 images 后再构建训练数据集"
                        )
                    raise FileNotFoundError(
                        f"缺少 ReInterHand manifest 或已解压 capture: {self.manifest_path}"
                    )
                has_prepared_capture = any(
                    (capture_dir / "mano_fits" / "params").is_dir()
                    and (capture_dir / "Ego_cameras" / "envmap_per_segment" / "cam_params").is_dir()
                    for capture_dir in capture_dirs
                )
                if not has_prepared_capture and archive_parts:
                    raise FileNotFoundError(
                        "ReInterHand 仅发现分卷 archive；需先完成官方 checksum 校验并解压，"
                        "得到 cam_params、mano_fits/params 和 images 后再构建训练数据集"
                    )
                manifest_rows = [
                    {"capture_id": capture_dir.name, "capture_dir": str(capture_dir)}
                    for capture_dir in capture_dirs
                ]
                source_paths = [self.root, *capture_dirs]
            record_index: list[dict[str, Any]] = []
            for row in manifest_rows:
                capture_dir = Path(row["capture_dir"])
                capture_id = str(row.get("capture_id", capture_dir.name))
                mano_dir = capture_dir / "mano_fits" / "params"
                source_paths.append(mano_dir)
                for background_mode in ("envmap_per_segment",):
                    image_dir = capture_dir / "Ego_cameras" / background_mode / "images"
                    cam_dir = capture_dir / "Ego_cameras" / background_mode / "cam_params"
                    truncation_path = image_dir.parent / "truncation_ratio.json"
                    source_paths.extend((image_dir, cam_dir, truncation_path))
                    if not image_dir.exists() or not cam_dir.exists():
                        continue
                    truncation_ratio = _load_json(truncation_path) if truncation_path.is_file() else {}
                    for image_path in sorted(image_dir.glob("*.png")):
                        frame_id = image_path.stem
                        try:
                            frame_id_int = int(frame_id)
                        except ValueError:
                            continue
                        camera_path = cam_dir / f"{frame_id}.json"
                        left_mano_path = mano_dir / f"{frame_id_int}_left.json"
                        right_mano_path = mano_dir / f"{frame_id_int}_right.json"
                        ratio = truncation_ratio.get(frame_id, truncation_ratio.get(str(frame_id_int)))
                        try:
                            is_truncated = ratio is not None and float(ratio) >= 0.2
                        except (TypeError, ValueError):
                            is_truncated = True
                        if is_truncated or not (camera_path.is_file() and left_mano_path.is_file() and right_mano_path.is_file()):
                            continue
                        if background_mode == "envmap_per_segment":
                            sequence_id = f"{capture_id}:{background_mode}"
                        else:
                            sequence_id = f"{capture_id}:{background_mode}:{frame_id}"
                        record_index.append(
                            {
                                "sequence_id": sequence_id,
                                "frame_id": frame_id,
                                "temporal_index": int(frame_id),
                                "image_path": str(image_path),
                                "cam_param_path": str(camera_path),
                                "mano_dir": str(mano_dir),
                                "background_mode": background_mode,
                            }
                        )
            return record_index, source_paths

        return _load_or_build_light_index_cache(self._index_cache_path, [], _scan_record_index)

    def _materialize_record(self, index_entry: dict[str, Any]) -> FrameRecord:
        image_path = Path(str(index_entry["image_path"]))
        camera_params = self._cached_json_payload(Path(str(index_entry["cam_param_path"])))
        raw_intrinsics = _intrinsics_from_focal_principal(
            float(camera_params["focal"][0]),
            float(camera_params["focal"][1]),
            float(camera_params["princpt"][0]),
            float(camera_params["princpt"][1]),
        )
        distortion = np.asarray(camera_params.get("D", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
        intrinsics = _fisheye_rectified_intrinsics(raw_intrinsics, distortion, _image_size(image_path))
        camera_rotation = np.asarray(camera_params["R"], dtype=np.float32)
        camera_translation = np.asarray(camera_params["t"], dtype=np.float32) / 1000.0
        camera_pose = _pose_from_rotation_translation(camera_rotation, camera_translation)
        frame_id = int(index_entry["temporal_index"])
        mano_dir = Path(str(index_entry["mano_dir"]))
        hand_annos: list[HandAnnotation] = []
        for side in ("left", "right"):
            mano_path = mano_dir / f"{frame_id}_{side}.json"
            if not mano_path.exists():
                continue
            payload = self._cached_json_payload(mano_path)
            hand_annos.append(
                HandAnnotation(
                    side=side,
                    visible=True,
                    mano_pose=np.asarray(payload.get("pose"), dtype=np.float32),
                    mano_pose_format="axis_angle_full",
                    mano_betas=np.asarray(payload.get("shape"), dtype=np.float32),
                    mano_trans=np.asarray(payload.get("trans"), dtype=np.float32),
                    extras={
                        "mano_flat_hand_mean": False,
                        "mano_flip_left_shapedirs": side == "left",
                        "post_mano_transform": camera_pose,
                    },
                )
            )
        left_count, right_count = _count_sides(hand_annos)
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split="train",
            sequence_id=str(index_entry["sequence_id"]),
            frame_id=str(index_entry["frame_id"]),
            temporal_index=frame_id,
            view_name="egocentric",
            is_egocentric=True,
            rgb_ref=MediaRef(kind="path", path=str(image_path)),
            depth_ref=None,
            depth_mode=None,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=None, intrinsics=intrinsics, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={
                "background_mode": str(index_entry["background_mode"]),
                "raw_intrinsics": raw_intrinsics,
                "camera_distortion": distortion,
                "rgb_rectified": _has_distortion(distortion),
            },
        )


def _parse_jsonl_by_frame(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            result[int(payload["frame_index"])] = payload
    return result


def _load_jsonl_offsets_by_frame(path: Path) -> dict[int, int]:
    offsets: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            offsets[int(payload["frame_index"])] = offset
    return offsets


def _load_jsonl_row_at_offset(path: Path, offset: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        line = handle.readline()
    return json.loads(line)


class EgoTouchFrameDataset(BaseFrameDataset):
    dataset_name = "egotouch"
    base_dataset_name = "egotouch"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        hand_source: str = "wilor",
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self.hand_source = hand_source
        self._index_cache_path = _light_index_cache_path(_as_path(root), f"egotouch_frame_dataset_{split}_{hand_source}")
        self._record_cache: OrderedDict[int, FrameRecord] = OrderedDict()
        self._records_materialized: list[FrameRecord] | None = None
        self._record_index: list[dict[str, Any]] | None = None
        self._sequence_entries = self._load_egotouch_sequence_entries()
        self._sequence_offsets: list[int] = []
        self.sequence_to_indices: dict[str, range] = {}
        record_count = 0
        for sequence_entry in self._sequence_entries:
            frame_count = int(sequence_entry["frame_count"])
            self._sequence_offsets.append(record_count)
            self.sequence_to_indices[str(sequence_entry["sequence_id"])] = range(record_count, record_count + frame_count)
            record_count += frame_count
        self._record_count = record_count

    def _uses_lazy_records(self) -> bool:
        return True

    def _load_egotouch_sequence_entries(self) -> list[dict[str, Any]]:
        hand_files = [
            episode_dir / f"{self.hand_source}_hands.json"
            for episode_dir in sorted(self.root.glob("*/*/*"))
            if episode_dir.is_dir() and (episode_dir / f"{self.hand_source}_hands.json").exists()
        ]
        source_paths = [
            path
            for hand_file in hand_files
            for path in (hand_file, hand_file.parent / "chest.mp4")
        ]

        def _scan_sequence_entries() -> list[dict[str, Any]]:
            sequence_entries: list[dict[str, Any]] = []
            video_frame_counts = _load_or_build_video_frame_counts(
                _light_index_cache_path(self.root, "egotouch_video_frame_counts"),
                [hand_file.parent / "chest.mp4" for hand_file in hand_files],
            )
            for hand_file in hand_files:
                episode_dir = hand_file.parent
                rgb_path = episode_dir / "chest.mp4"
                rgb_frame_count = video_frame_counts.get(rgb_path, 0)
                if rgb_frame_count <= 0:
                    continue
                sequence_id = str(episode_dir.relative_to(self.root))
                offsets_by_frame = _load_jsonl_offsets_by_frame(hand_file)
                frame_indices = np.asarray(
                    [frame_index for frame_index in sorted(offsets_by_frame) if int(frame_index) < rgb_frame_count],
                    dtype=np.int32,
                )
                if frame_indices.size == 0:
                    continue
                line_offsets = np.asarray([offsets_by_frame[int(frame_index)] for frame_index in frame_indices], dtype=np.int64)
                sequence_entries.append(
                    {
                        "sequence_id": sequence_id,
                        "episode_dir": str(episode_dir),
                        "rgb_path": str(rgb_path),
                        "hand_file": str(hand_file),
                        "frame_indices": frame_indices,
                        "line_offsets": line_offsets,
                        "frame_count": int(len(frame_indices)),
                    }
                )
            return sequence_entries

        return _load_or_build_sequence_entry_cache(
            self._index_cache_path,
            source_paths,
            _scan_sequence_entries,
            version=6,
        )

    def _get_record(self, index: int) -> FrameRecord:
        index = self._normalize_index(index)
        cached = self._record_cache.get(index)
        if cached is not None:
            self._record_cache.move_to_end(index)
            return cached
        sequence_entry, local_index = _sequence_entry_for_index(self._sequence_offsets, self._sequence_entries, index)
        record = self._materialize_record(sequence_entry, local_index)
        self._record_cache[index] = record
        self._record_cache.move_to_end(index)
        cache_limit = self._record_cache_limit()
        if cache_limit is not None and cache_limit > 0:
            while len(self._record_cache) > cache_limit:
                self._record_cache.popitem(last=False)
        return record

    def __len__(self) -> int:
        return self._record_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        return BaseFrameDataset.__getitem__(self, index)

    def _materialize_record(self, sequence_entry: dict[str, Any], local_index: int) -> FrameRecord:
        frame_index = int(sequence_entry["frame_indices"][local_index])
        row = _load_jsonl_row_at_offset(Path(str(sequence_entry["hand_file"])), int(sequence_entry["line_offsets"][local_index]))
        hand_annos = [
            HandAnnotation(side="left", visible=True, joints_3d=np.asarray(row["left_pos"], dtype=np.float32)),
            HandAnnotation(side="right", visible=True, joints_3d=np.asarray(row["right_pos"], dtype=np.float32)),
        ]
        left_count, right_count = _count_sides(hand_annos)
        episode_dir = Path(str(sequence_entry["episode_dir"]))
        rgb_path = Path(str(sequence_entry["rgb_path"]))
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split="train",
            sequence_id=str(sequence_entry["sequence_id"]),
            frame_id=str(frame_index),
            temporal_index=frame_index,
            view_name="egocentric",
            is_egocentric=True,
            rgb_ref=MediaRef(kind="video_frame", path=str(rgb_path), frame_index=frame_index),
            depth_ref=None,
            depth_mode=None,
            intrinsics=None,
            camera_pose=None,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=None, intrinsics=None, camera_pose=None),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={
                "left_camera_ref": MediaRef(kind="video_frame", path=str(episode_dir / "left.mp4"), frame_index=frame_index).to_dict(),
                "right_camera_ref": MediaRef(kind="video_frame", path=str(episode_dir / "right.mp4"), frame_index=frame_index).to_dict(),
            },
        )


class Stera10MFrameDataset(BaseFrameDataset):
    dataset_name = "stera_10m"
    base_dataset_name = "stera_10m"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self._intrinsics_cache: dict[str, np.ndarray] = {}
        self._index_cache_path = _light_index_cache_path(_as_path(root), f"stera_10m_frame_dataset_{split}")
        # Create the sidecar directory before fingerprinting ``root`` so the
        # cache's own first write cannot invalidate the root signature.
        self._index_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._row_locator_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_pid = os.getpid()
        self._annotation_handle_cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._record_cache: OrderedDict[int, FrameRecord] = OrderedDict()
        self._records_materialized: list[FrameRecord] | None = None
        self._record_index: list[dict[str, Any]] | None = None
        self._sequence_entries = self._load_stera_sequence_entries()
        self._sequence_offsets: list[int] = []
        self.sequence_to_indices: dict[str, range] = {}
        record_count = 0
        for sequence_entry in self._sequence_entries:
            frame_count = int(sequence_entry["frame_count"])
            self._sequence_offsets.append(record_count)
            self.sequence_to_indices[str(sequence_entry["sequence_id"])] = range(record_count, record_count + frame_count)
            record_count += frame_count
        self._record_count = record_count

    def _uses_lazy_records(self) -> bool:
        return True

    def _ensure_worker_local_caches(self) -> None:
        pid = os.getpid()
        if self._cache_pid == pid:
            return
        while self._annotation_handle_cache:
            _, handle = self._annotation_handle_cache.popitem(last=False)
            handle.close()
        self._row_locator_cache.clear()
        self._intrinsics_cache.clear()
        self._record_cache.clear()
        self._cache_pid = pid

    def _stera_annotation_handle(self, ann_path: Path) -> h5py.File:
        self._ensure_worker_local_caches()
        cache_key = str(ann_path)
        cached = self._annotation_handle_cache.get(cache_key)
        if cached is not None and bool(cached.id.valid):
            self._annotation_handle_cache.move_to_end(cache_key)
            return cached
        if cached is not None:
            self._annotation_handle_cache.pop(cache_key)
            cached.close()
        handle = h5py.File(ann_path, "r")
        self._annotation_handle_cache[cache_key] = handle
        self._annotation_handle_cache.move_to_end(cache_key)
        while len(self._annotation_handle_cache) > _STERA_H5_CACHE_LIMIT:
            _, stale = self._annotation_handle_cache.popitem(last=False)
            stale.close()
        return handle

    def _load_stera_sequence_entries(self) -> list[dict[str, Any]]:
        session_dirs = [session_dir for session_dir in sorted(self.root.glob("session_data_*")) if session_dir.is_dir()]

        def _scan_sequence_entries() -> list[dict[str, Any]]:
            sequence_entries: list[dict[str, Any]] = []
            for session_dir in session_dirs:
                ann_path = session_dir / "annotation.hdf5"
                rgb_path = session_dir / "rgb.mp4"
                if not ann_path.exists() or not rgb_path.exists():
                    continue
                rgb_frame_count = _video_frame_count(rgb_path)
                if rgb_frame_count <= 0:
                    continue
                with h5py.File(ann_path, "r") as handle:
                    frame_timestamps = np.asarray(handle["hand-pose/frame_timestamps"])
                    frame_indices = np.asarray(handle["hand-pose/frame_idx"])
                    unique_frames = [
                        frame_index
                        for frame_index in sorted(set(int(item) for item in frame_indices.tolist()))
                        if frame_index < rgb_frame_count
                    ]
                    if not unique_frames:
                        continue
                    sequence_entries.append(
                        {
                            "sequence_id": session_dir.name,
                            "session_dir": str(session_dir),
                            "ann_path": str(ann_path),
                            "rgb_frame_count": rgb_frame_count,
                            "frame_indices": np.asarray(unique_frames, dtype=np.int32),
                            "frame_count": len(unique_frames),
                        }
                    )
            return sequence_entries

        return _load_or_build_sequence_entry_cache(
            self._index_cache_path,
            # Stera releases are immutable; root mtime detects session additions/removals
            # without stat-ing every session's media and calibration files on startup.
            [self.root],
            _scan_sequence_entries,
            version=_STERA_SEQUENCE_ENTRY_CACHE_VERSION,
        )

    def _stera_intrinsics(self, session_dir: Path) -> np.ndarray:
        self._ensure_worker_local_caches()
        cache_key = str(session_dir)
        cached = self._intrinsics_cache.get(cache_key)
        if cached is not None:
            return cached
        intrinsics = np.load(session_dir / "calibrations" / "rgb_K.npy").astype(np.float32)
        self._intrinsics_cache[cache_key] = intrinsics
        return intrinsics

    def _stera_row_indices(self, session_dir: Path, ann_path: Path, frame_index: int) -> list[int]:
        self._ensure_worker_local_caches()
        cache_key = str(session_dir)
        cached = _lru_get(self._row_locator_cache, cache_key)
        if cached is None:
            handle = self._stera_annotation_handle(ann_path)
            row_indices_by_frame: dict[int, list[int]] = {}
            for row_index, indexed_frame in enumerate(np.asarray(handle["hand-pose/frame_idx"])):
                row_indices_by_frame.setdefault(int(indexed_frame), []).append(int(row_index))
            cached = {"row_indices_by_frame": row_indices_by_frame}
            _lru_put(self._row_locator_cache, cache_key, cached, limit=_STERA_SESSION_CACHE_LIMIT)
        return list(cached["row_indices_by_frame"].get(int(frame_index), []))

    def prefetch_sequence(self, sequence_index: int) -> None:
        sequence_entry = self._sequence_entries[int(sequence_index)]
        session_dir = Path(str(sequence_entry["session_dir"]))
        ann_path = Path(str(sequence_entry["ann_path"]))
        frame_index = int(sequence_entry["frame_indices"][0])
        self._stera_intrinsics(session_dir)
        self._stera_row_indices(session_dir, ann_path, frame_index)
        prewarm_media_refs(
            [
                MediaRef(kind="video_frame", path=str(session_dir / "rgb.mp4"), frame_index=frame_index),
                MediaRef(kind="h5_dataset", path=str(ann_path), member="depth/frames", frame_index=frame_index),
            ]
        )

    def _get_record(self, index: int) -> FrameRecord:
        self._ensure_worker_local_caches()
        index = self._normalize_index(index)
        cached = self._record_cache.get(index)
        if cached is not None:
            self._record_cache.move_to_end(index)
            return cached
        sequence_entry, local_index = _sequence_entry_for_index(self._sequence_offsets, self._sequence_entries, index)
        record = self._materialize_record(sequence_entry, local_index)
        self._record_cache[index] = record
        self._record_cache.move_to_end(index)
        cache_limit = self._record_cache_limit()
        if cache_limit is not None and cache_limit > 0:
            while len(self._record_cache) > cache_limit:
                self._record_cache.popitem(last=False)
        return record

    def __len__(self) -> int:
        return self._record_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        return BaseFrameDataset.__getitem__(self, index)

    def _materialize_record(self, sequence_entry: dict[str, Any], local_index: int) -> FrameRecord:
        session_dir = Path(str(sequence_entry["session_dir"]))
        ann_path = Path(str(sequence_entry["ann_path"]))
        frame_index = int(sequence_entry["frame_indices"][local_index])
        hand_annos: list[HandAnnotation] = []
        row_indices = self._stera_row_indices(session_dir, ann_path, frame_index)
        with nullcontext(self._stera_annotation_handle(ann_path)) as handle:
            side_values = handle["hand-pose/side"]
            for row_idx in row_indices:
                side = "left" if int(side_values[row_idx]) == 0 else "right"
                global_orient = np.asarray(handle["hand-pose/global_orient"][row_idx], dtype=np.float32)
                hand_pose = np.asarray(handle["hand-pose/hand_pose"][row_idx], dtype=np.float32)
                if side == "left":
                    global_orient = _STERA_LEFT_ROTATION_CONVENTION @ global_orient @ _STERA_LEFT_ROTATION_CONVENTION
                    hand_pose = _STERA_LEFT_ROTATION_CONVENTION @ hand_pose @ _STERA_LEFT_ROTATION_CONVENTION
                hand_annos.append(
                    HandAnnotation(
                        side=side,
                        visible=True,
                        bbox_xyxy=np.asarray(handle["hand-pose/bbox"][row_idx], dtype=np.float32),
                        joints_3d=np.asarray(handle["hand-pose/joints_cam_lidar"][row_idx], dtype=np.float32),
                        joints_2d=np.asarray(handle["hand-pose/kpts_2d"][row_idx], dtype=np.float32),
                        mano_global_orient=global_orient,
                        mano_hand_pose=hand_pose,
                        mano_pose_format="rotation_matrix",
                        mano_betas=np.asarray(handle["hand-pose/betas"][row_idx], dtype=np.float32),
                        mano_trans=np.asarray(handle["hand-pose/pred_cam_t"][row_idx], dtype=np.float32),
                        extras={"mano_align_root_to_joints_3d": True},
                    )
                )
            rotation = np.asarray(handle["cam-pose/rotations"][frame_index], dtype=np.float32)
            translation = np.asarray(handle["cam-pose/translations"][frame_index], dtype=np.float32)
            timestamp = float(handle["hand-pose/frame_timestamps"][frame_index])
        depth_ref = MediaRef(kind="h5_dataset", path=str(ann_path), member="depth/frames", frame_index=frame_index)
        camera_pose = _stera_optical_world_to_camera(rotation, translation)
        intrinsics = self._stera_intrinsics(session_dir)
        left_count, right_count = _count_sides(hand_annos)
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split="train",
            sequence_id=str(sequence_entry["sequence_id"]),
            frame_id=str(frame_index),
            temporal_index=frame_index,
            view_name="egocentric",
            is_egocentric=True,
            rgb_ref=MediaRef(kind="video_frame", path=str(session_dir / "rgb.mp4"), frame_index=frame_index),
            depth_ref=depth_ref,
            depth_mode="h5_uint16_mm",
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=depth_ref, intrinsics=intrinsics, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={"timestamp": timestamp},
        )


class EgoForceH2OFrameDataset(BaseFrameDataset):
    dataset_name = "egoforce_h2o"
    base_dataset_name = "h2o"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        camera_names: tuple[str, ...] = ("cam4",),
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.camera_names = camera_names
        camera_suffix = "_".join(camera_names)
        self._index_cache_path = _light_index_cache_path(
            _as_path(root),
            f"egoforce_h2o_frame_dataset_{split}_{camera_suffix}",
        )
        super().__init__(root, split, load_rgb=load_rgb, load_depth=load_depth)

    def _uses_lazy_records(self) -> bool:
        return True

    @staticmethod
    def _has_valid_hand_gt(row: Any, side: str) -> bool:
        fields = (f"{side}_hand_camera_j3D", f"{side}_hand_camera_global_orient", f"{side}_hand_hand_pose", f"{side}_hand_camera_betas", f"{side}_hand_camera_transl")
        values = [np.asarray(row[field], dtype=np.float32) for field in fields]
        return all(np.isfinite(value).all() for value in values) and float(np.linalg.norm(values[0])) > 1e-6 and float(np.linalg.norm(values[-1])) > 1e-6

    def _build_record_index(self) -> list[dict[str, Any]]:
        if self.split == "all":
            split_names = ("train", "val", "test")
        elif self.split == "trainval":
            split_names = ("train", "val")
        else:
            split_names = (self.split,)
        source_paths = [self.root / f"{camera_name}_annos.h5" for camera_name in self.camera_names]

        def _scan_record_index() -> list[dict[str, Any]]:
            record_index: list[dict[str, Any]] = []
            for camera_name in self.camera_names:
                h5_path = self.root / f"{camera_name}_annos.h5"
                with h5py.File(h5_path, "r") as handle:
                    for split_name in split_names:
                        dataset = handle[split_name]
                        for row_index, row in enumerate(dataset):
                            key = _decode_if_bytes(row["key"])
                            prefix, frame_str = key.rsplit("_", 1)
                            record_index.append(
                                {
                                    "sequence_id": f"{prefix}_{camera_name}",
                                    "frame_id": frame_str,
                                    "temporal_index": int(frame_str),
                                    "split_name": split_name,
                                    "camera_name": camera_name,
                                    "h5_path": str(h5_path),
                                    "row_index": row_index,
                                    "prefix": prefix,
                                }
                            )
            return record_index

        return _load_or_build_light_index_cache(self._index_cache_path, source_paths, _scan_record_index)

    def _materialize_record(self, index_entry: dict[str, Any]) -> FrameRecord:
        camera_name = str(index_entry["camera_name"])
        prefix = str(index_entry["prefix"])
        frame_str = str(index_entry["frame_id"])
        split_name = str(index_entry["split_name"])
        with h5py.File(Path(str(index_entry["h5_path"])), "r") as handle:
            row = handle[split_name][int(index_entry["row_index"])]
            intrinsics = _intrinsics_from_focal_principal(float(row["fx"]), float(row["fy"]), float(row["cx"]), float(row["cy"]))
            camera_pose = _invert_pose(row["camera_pose"])
            hand_annos: list[HandAnnotation] = []
            if self._has_valid_hand_gt(row, "left"):
                hand_annos.append(
                    HandAnnotation(
                        side="left",
                        visible=True,
                                    bbox_xyxy=np.asarray(row["left_hand_camera_hand_box"], dtype=np.float32),
                                    joints_3d=np.asarray(row["left_hand_camera_j3D"], dtype=np.float32),
                                    joints_2d=np.asarray(row["left_hand_camera_j2D"], dtype=np.float32),
                                    mano_global_orient=np.asarray(row["left_hand_camera_global_orient"], dtype=np.float32),
                                    mano_hand_pose=np.asarray(row["left_hand_hand_pose"], dtype=np.float32),
                                    mano_pose_format="axis_angle_pose45",
                                    mano_betas=np.asarray(row["left_hand_camera_betas"], dtype=np.float32),
                                    mano_trans=np.asarray(row["left_hand_camera_transl"], dtype=np.float32),
                                    extras={
                                        "mano_flat_hand_mean": False,
                                        "mano_flip_left_shapedirs": True,
                                    },
                                )
                            )
            if self._has_valid_hand_gt(row, "right"):
                hand_annos.append(
                    HandAnnotation(
                        side="right",
                        visible=True,
                                    bbox_xyxy=np.asarray(row["right_hand_camera_hand_box"], dtype=np.float32),
                                    joints_3d=np.asarray(row["right_hand_camera_j3D"], dtype=np.float32),
                                    joints_2d=np.asarray(row["right_hand_camera_j2D"], dtype=np.float32),
                                    mano_global_orient=np.asarray(row["right_hand_camera_global_orient"], dtype=np.float32),
                                    mano_hand_pose=np.asarray(row["right_hand_hand_pose"], dtype=np.float32),
                                    mano_pose_format="axis_angle_pose45",
                                    mano_betas=np.asarray(row["right_hand_camera_betas"], dtype=np.float32),
                                    mano_trans=np.asarray(row["right_hand_camera_transl"], dtype=np.float32),
                                    extras={"mano_flat_hand_mean": False},
                                )
                            )
        rgb_tar = self.root / "wds" / "shards" / f"{prefix}_{camera_name}_rgb.tar"
        depth_tar = self.root / "wds" / "shards" / f"{prefix}_{camera_name}_depth.tar"
        depth_ref = (
            MediaRef(kind="tar_member", path=str(depth_tar), member=f"{prefix}_{camera_name}_{frame_str}.depth.png")
            if depth_tar.exists()
            else None
        )
        left_count, right_count = _count_sides(hand_annos)
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split=split_name,
            sequence_id=str(index_entry["sequence_id"]),
            frame_id=frame_str,
            temporal_index=int(index_entry["temporal_index"]),
            view_name=camera_name,
            is_egocentric=camera_name == "cam4",
            rgb_ref=MediaRef(kind="tar_member", path=str(rgb_tar), member=f"{prefix}_{camera_name}_{frame_str}.rgb.png"),
            depth_ref=depth_ref,
            depth_mode="tar_depth_png",
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=depth_ref, intrinsics=intrinsics, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
        )


class EgoForceHot3dFrameDataset(BaseFrameDataset):
    dataset_name = "egoforce_hot3d"
    base_dataset_name = "hot3d"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self._index_cache_path = _light_index_cache_path(_as_path(root), f"egoforce_hot3d_frame_dataset_{split}")
        super().__init__(root, split, load_rgb=load_rgb, load_depth=load_depth)

    def _uses_lazy_records(self) -> bool:
        return True

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = BaseFrameDataset.__getitem__(self, index)
        if sample.get("rgb") is not None and bool(sample.get("extras", {}).get("rgb_rectified", False)):
            sample["rgb"] = _rectify_hot3d_rgb(sample["rgb"], sample["extras"]["hot3d_camera_model"])
        return sample

    def _build_record_index(self) -> list[dict[str, Any]]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            if self.split == "all":
                split_names = ["train", "val", "test"]
            else:
                split_names = [self.split]
        else:
            split_names = [self.split]
        if self.split == "all":
            split_names = [name for name in ("train", "val", "test") if (self.root / name).exists()]
        ann_tars = [ann_tar for split_name in split_names for ann_tar in sorted((self.root / split_name).glob("*.annotations.tar"))]

        def _scan_record_index() -> list[dict[str, Any]]:
            record_index: list[dict[str, Any]] = []
            for split_name in split_names:
                for ann_tar in sorted((self.root / split_name).glob("*.annotations.tar")):
                    rgb_tar = ann_tar.with_name(ann_tar.name.replace(".annotations.tar", ".rgb.tar"))
                    with tarfile.open(ann_tar, "r") as handle:
                        for member in handle.getnames():
                            parts = member.split(".")
                            if len(parts) < 4:
                                continue
                            sequence_id, frame_id = parts[0], parts[1]
                            record_index.append(
                                {
                                    "split": split_name,
                                    "sequence_id": sequence_id,
                                    "frame_id": frame_id,
                                    "temporal_index": int(frame_id),
                                    "ann_tar": str(ann_tar),
                                    "rgb_tar": str(rgb_tar),
                                    "member": member,
                                }
                            )
            return record_index

        return _load_or_build_light_index_cache(self._index_cache_path, ann_tars, _scan_record_index)

    def _materialize_record(self, index_entry: dict[str, Any]) -> FrameRecord:
        ann_tar = Path(str(index_entry["ann_tar"]))
        member = str(index_entry["member"])
        with tarfile.open(ann_tar, "r") as handle:
            payload = json.load(handle.extractfile(member))
        hand_annos: list[HandAnnotation] = []
        rgb_camera_params = payload.get("rgb_camera_params", {})
        fisheye_params = rgb_camera_params.get("fisheye624_params", {})
        focal_length = fisheye_params.get("focal_length")
        principal_point = fisheye_params.get("principal_point")
        camera_model = _egoforce_hot3d_camera_model(fisheye_params, rgb_camera_params)
        intrinsics = None
        if focal_length is not None and principal_point is not None:
            intrinsics = _intrinsics_from_focal_principal(
                float(focal_length[0]),
                float(focal_length[1]),
                float(principal_point[0]),
                float(principal_point[1]),
            )
        raw_intrinsics = intrinsics.copy() if intrinsics is not None else None
        if camera_model is not None:
            intrinsics = _intrinsics_from_projectaria_linear(_hot3d_linear_calibration(camera_model))
        rgb_hand_params = payload.get("rgb_hand_params", {})
        hand_params = rgb_hand_params if rgb_hand_params else payload.get("hand_params", {})
        for side in ("left", "right"):
            hand_payload = hand_params.get(side)
            if not hand_payload:
                continue
            raw_joints_2d = _as_optional_array(hand_payload.get("camera_j2D"), shape=(21, 2))
            joints_2d = _rectify_hot3d_points_2d(raw_joints_2d, camera_model) if raw_joints_2d is not None and camera_model is not None else raw_joints_2d
            raw_bbox_xyxy = _as_optional_array(hand_payload.get("bbox"), shape=(4,))
            bbox_xyxy = raw_bbox_xyxy
            if raw_bbox_xyxy is not None and camera_model is not None:
                raw_corners = np.array(
                    [
                        [raw_bbox_xyxy[0], raw_bbox_xyxy[1]],
                        [raw_bbox_xyxy[2], raw_bbox_xyxy[1]],
                        [raw_bbox_xyxy[2], raw_bbox_xyxy[3]],
                        [raw_bbox_xyxy[0], raw_bbox_xyxy[3]],
                    ],
                    dtype=np.float32,
                )
                bbox_xyxy = _bbox_from_finite_points(_rectify_hot3d_points_2d(raw_corners, camera_model))
            if bbox_xyxy is None and joints_2d is not None:
                bbox_xyxy = _bbox_from_finite_points(joints_2d)
            camera_j3d = _as_optional_array(hand_payload.get("camera_j3D"), shape=(21, 3))
            joints_3d = camera_j3d
            if joints_3d is None:
                joints_3d = _as_optional_array(hand_payload.get("world_j3D"), shape=(21, 3))
            mano_pose = _as_optional_array(hand_payload.get("pose"), shape=(48,))
            mano_pose_format = "axis_angle_full"
            mano_global_orient = None
            mano_trans = _as_optional_array(hand_payload.get("trans"), shape=(3,))
            pca_pose = _as_optional_array(hand_payload.get("hand_pose"), shape=(15,))
            if pca_pose is not None:
                mano_pose = pca_pose
                mano_pose_format = "hot3d_mano_pca"
                mano_global_orient = _as_optional_array(hand_payload.get("camera_global_orient"), shape=(3,))
                mano_trans = _as_optional_array(hand_payload.get("camera_transl"), shape=(3,))
            mano_betas = _as_optional_array(hand_payload.get("betas"), shape=(10,))
            extras: dict[str, Any] = {}
            if camera_model is not None:
                extras["raw_fisheye624_projection_params"] = fisheye_params.get("projection_params")
                if raw_joints_2d is not None:
                    extras["raw_joints_2d"] = raw_joints_2d
                if raw_bbox_xyxy is not None:
                    extras["raw_bbox_xyxy"] = raw_bbox_xyxy
            if camera_j3d is not None:
                extras["mano_align_root_to_joints_3d"] = True
                if side == "left":
                    extras["mano_flip_left_shapedirs"] = True
            hand_annos.append(
                HandAnnotation(
                    side=side,
                    visible=True,
                    bbox_xyxy=bbox_xyxy,
                    joints_3d=joints_3d,
                    joints_2d=joints_2d,
                    mano_global_orient=mano_global_orient,
                    mano_pose=mano_pose,
                    mano_pose_format=mano_pose_format,
                    mano_betas=mano_betas,
                    mano_trans=mano_trans,
                    extras=extras,
                )
            )
        camera_pose = None
        if "w2c" in rgb_camera_params:
            camera_pose = np.asarray(rgb_camera_params["w2c"], dtype=np.float32)
        elif "camera_pose" in payload:
            camera_pose = np.asarray(payload.get("camera_pose"), dtype=np.float32)
        left_count, right_count = _count_sides(hand_annos)
        sequence_id = str(index_entry["sequence_id"])
        frame_id = str(index_entry["frame_id"])
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split=str(index_entry["split"]),
            sequence_id=sequence_id,
            frame_id=frame_id,
            temporal_index=int(index_entry["temporal_index"]),
            view_name="egocentric",
            is_egocentric=True,
            rgb_ref=MediaRef(kind="tar_member", path=str(index_entry["rgb_tar"]), member=f"{sequence_id}.{frame_id}.rgb.jpg"),
            depth_ref=None,
            depth_mode=None,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=None, intrinsics=None, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={
                "timestamp_ns": payload.get("timestamp_ns"),
                "raw_intrinsics": raw_intrinsics,
                "rgb_rectified": camera_model is not None,
                "hot3d_camera_model": camera_model,
            },
        )


class EgoForceArcticFrameDataset(BaseFrameDataset):
    dataset_name = "egoforce_arctic"
    base_dataset_name = "arctic"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        camera_names: tuple[str, ...] = ("cam0",),
        egocentric_cameras: tuple[str, ...] = ("cam0",),
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.camera_names = camera_names
        self.egocentric_cameras = set(egocentric_cameras)
        self._object_resolver = ArcticObjectResolver(_as_path(root) / "official_object_sidecar")
        self._cache_pid = os.getpid()
        self._annotation_handle_cache: OrderedDict[str, h5py.File] = OrderedDict()
        camera_suffix = "_".join(camera_names)
        self._index_cache_path = _light_index_cache_path(
            _as_path(root),
            f"{self.dataset_name}_frame_dataset_{split}_{camera_suffix}",
        )
        super().__init__(root, split, load_rgb=load_rgb, load_depth=load_depth)

    def _uses_lazy_records(self) -> bool:
        return True

    def _close_arctic_annotation_handles(self) -> None:
        while self._annotation_handle_cache:
            _, handle = self._annotation_handle_cache.popitem(last=False)
            handle.close()

    def _ensure_worker_local_caches(self) -> None:
        pid = os.getpid()
        if self._cache_pid == pid:
            return
        self._close_arctic_annotation_handles()
        self._record_cache.clear()
        self._cache_pid = pid

    def _arctic_annotation_handle(self, h5_path: Path) -> h5py.File:
        self._ensure_worker_local_caches()
        cache_key = str(h5_path)
        cached = self._annotation_handle_cache.get(cache_key)
        if cached is not None and bool(cached.id.valid):
            self._annotation_handle_cache.move_to_end(cache_key)
            return cached
        if cached is not None:
            self._annotation_handle_cache.pop(cache_key)
            cached.close()
        handle = h5py.File(h5_path, "r")
        self._annotation_handle_cache[cache_key] = handle
        self._annotation_handle_cache.move_to_end(cache_key)
        while len(self._annotation_handle_cache) > _ARCTIC_H5_CACHE_LIMIT:
            _, stale = self._annotation_handle_cache.popitem(last=False)
            stale.close()
        return handle

    def __del__(self) -> None:
        cache = getattr(self, "_annotation_handle_cache", None)
        if cache is None:
            return
        while cache:
            _, handle = cache.popitem(last=False)
            try:
                handle.close()
            except Exception:
                pass

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = BaseFrameDataset.__getitem__(self, index)
        if sample.get("rgb") is None:
            return sample
        extras = sample.get("extras", {})
        if not bool(extras.get("rgb_rectified", False)):
            return sample
        sample["rgb"] = _undistort_pinhole_rgb(
            sample["rgb"],
            extras["raw_intrinsics"],
            extras["camera_distortion"],
            sample["intrinsics"],
        )
        return sample

    @staticmethod
    def _has_valid_hand_gt(row: Any, side: str) -> bool:
        fields = (
            f"{side}_hand_camera_j3D",
            f"{side}_hand_camera_global_orient",
            f"{side}_hand_hand_pose",
            f"{side}_hand_camera_betas",
            f"{side}_hand_camera_transl",
        )
        return all(np.isfinite(np.asarray(row[field], dtype=np.float32)).all() for field in fields)

    def _build_record_index(self) -> list[dict[str, Any]]:
        split_names = (self.split,) if self.split != "all" else ("train", "val", "test")
        source_paths: list[Path] = []
        for camera_name in self.camera_names:
            h5_path = self.root / f"{camera_name}_hand_arm_annotations_v4.h5"
            source_paths.append(h5_path)
            camera_suffix = camera_name.replace("cam", "").zfill(2)
            for split_name in split_names:
                source_paths.append(self.root / "ArcticDatasetSeqNoCropImages" / split_name)
                source_paths.extend(
                    sorted((self.root / "ArcticDatasetSeqNoCropImages" / split_name).glob(f"*.cam{camera_suffix}.tar"))
                )

        def _scan_record_index() -> list[dict[str, Any]]:
            record_index: list[dict[str, Any]] = []
            for camera_name in self.camera_names:
                h5_path = self.root / f"{camera_name}_hand_arm_annotations_v4.h5"
                camera_suffix = camera_name.replace("cam", "").zfill(2)
                member_to_tar: dict[str, Path] = {}
                for split_name in split_names:
                    image_root = self.root / "ArcticDatasetSeqNoCropImages" / split_name
                    for tar_path in sorted(image_root.glob(f"*.cam{camera_suffix}.tar")):
                        with tarfile.open(tar_path, "r") as handle:
                            for member in handle.getnames():
                                member_to_tar[member] = tar_path
                with h5py.File(h5_path, "r") as handle:
                    for split_name in split_names:
                        dataset = handle[split_name]
                        for row_index, row in enumerate(dataset):
                            key = _decode_if_bytes(row["key"])
                            if not key:
                                continue
                            parts = key.split("@")
                            if len(parts) < 3:
                                continue
                            seq_name, action_name, frame_id, *_ = parts
                            if _should_skip_arctic_exposure_warmup_frame(frame_id):
                                continue
                            member_name = f"{seq_name}_{action_name}.{frame_id}.rgb{camera_suffix}.jpg"
                            image_path = self.root / "ArcticDatasetSeqNoCropImages" / split_name / member_name
                            rgb_tar = member_to_tar.get(member_name)
                            if _arctic_rgb_media_ref(image_path, rgb_tar, member_name) is None:
                                continue
                            record_index.append(
                                {
                                    "split": split_name,
                                    "sequence_id": f"{seq_name}@{action_name}@{camera_name}",
                                    "frame_id": frame_id,
                                    "temporal_index": int(frame_id),
                                    "camera_name": camera_name,
                                    "h5_path": str(h5_path),
                                    "row_index": row_index,
                                    "member_name": member_name,
                                    "rgb_path": str(image_path),
                                    "rgb_tar": None if rgb_tar is None else str(rgb_tar),
                                }
                            )
            return record_index

        return _load_or_build_light_index_cache(self._index_cache_path, source_paths, _scan_record_index, version=3)

    def _materialize_record(self, index_entry: dict[str, Any]) -> FrameRecord:
        split_name = str(index_entry["split"])
        camera_name = str(index_entry["camera_name"])
        handle = self._arctic_annotation_handle(Path(str(index_entry["h5_path"])))
        row = handle[split_name][int(index_entry["row_index"])]
        raw_intrinsics = _intrinsics_from_focal_principal(
            float(row["fx"]), float(row["fy"]), float(row["cx"]), float(row["cy"])
        )
        distortion = np.asarray(row["dist"], dtype=np.float32).reshape(-1)
        image_size = (int(row["width"]), int(row["height"]))
        intrinsics = _pinhole_rectified_intrinsics(raw_intrinsics, distortion, image_size)
        camera_pose = np.asarray(row["camera_pose"], dtype=np.float32)
        left_joints_2d = _undistort_pinhole_points_2d(
            row["left_hand_camera_j2D"], raw_intrinsics, distortion, intrinsics
        )
        left_bbox_xyxy = _bbox_from_finite_points(left_joints_2d)
        right_joints_2d = _undistort_pinhole_points_2d(
            row["right_hand_camera_j2D"], raw_intrinsics, distortion, intrinsics
        )
        right_bbox_xyxy = _bbox_from_finite_points(right_joints_2d)
        hand_annos: list[HandAnnotation] = []
        if self._has_valid_hand_gt(row, "left"):
            hand_annos.append(
                HandAnnotation(
                    side="left",
                    visible=True,
                    bbox_xyxy=left_bbox_xyxy,
                    joints_3d=np.asarray(row["left_hand_camera_j3D"], dtype=np.float32),
                    joints_2d=left_joints_2d,
                    mano_global_orient=np.asarray(row["left_hand_camera_global_orient"], dtype=np.float32),
                    mano_hand_pose=np.asarray(row["left_hand_hand_pose"], dtype=np.float32),
                    mano_pose_format="axis_angle_pose45",
                    mano_betas=np.asarray(row["left_hand_camera_betas"], dtype=np.float32),
                    mano_trans=np.asarray(row["left_hand_camera_transl"], dtype=np.float32),
                    extras={
                        "mano_flat_hand_mean": False,
                        "mano_flip_left_shapedirs": True,
                    },
                )
            )
        if self._has_valid_hand_gt(row, "right"):
            hand_annos.append(
                HandAnnotation(
                    side="right",
                    visible=True,
                    bbox_xyxy=right_bbox_xyxy,
                    joints_3d=np.asarray(row["right_hand_camera_j3D"], dtype=np.float32),
                    joints_2d=right_joints_2d,
                    mano_global_orient=np.asarray(row["right_hand_camera_global_orient"], dtype=np.float32),
                    mano_hand_pose=np.asarray(row["right_hand_hand_pose"], dtype=np.float32),
                    mano_pose_format="axis_angle_pose45",
                    mano_betas=np.asarray(row["right_hand_camera_betas"], dtype=np.float32),
                    mano_trans=np.asarray(row["right_hand_camera_transl"], dtype=np.float32),
                    extras={"mano_flat_hand_mean": False},
                )
            )
        left_count, right_count = _count_sides(hand_annos)
        member_name = str(index_entry["member_name"])
        rgb_tar_value = index_entry.get("rgb_tar")
        rgb_ref = _arctic_rgb_media_ref(
            Path(str(index_entry["rgb_path"])),
            None if rgb_tar_value is None else Path(str(rgb_tar_value)),
            member_name,
        )
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split=split_name,
            sequence_id=str(index_entry["sequence_id"]),
            frame_id=str(index_entry["frame_id"]),
            temporal_index=int(index_entry["temporal_index"]),
            view_name=camera_name,
            is_egocentric=camera_name in self.egocentric_cameras,
            rgb_ref=rgb_ref,
            depth_ref=None,
            depth_mode=None,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=None, intrinsics=intrinsics, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={
                "dist": distortion,
                "raw_intrinsics": raw_intrinsics,
                "camera_distortion": distortion,
                "rgb_rectified": _has_distortion(distortion),
                "scene_objects": self._object_resolver.scene_objects(
                    str(index_entry["sequence_id"]),
                    str(index_entry["frame_id"]),
                ),
            },
        )


class ArcticFrameDataset(EgoForceArcticFrameDataset):
    dataset_name = "arctic"
    base_dataset_name = "arctic"


def _resolve_taco_depth_root(root: Path) -> Path:
    """Prefer the lossless official uint16 depth export over legacy mirrors."""
    candidates = (
        root / "Egocentric_Depth_Videos_official_uint16" / "Egocentric_Depth_Videos",
        root / "Egocentric_Depth_Videos_official_uint16",
        root / "Egocentric_Depth_Videos",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*/*/egocentric_depth.avi")):
            return candidate
    # Preserve the existing error message when preparation is incomplete.
    return candidates[0]


class TacoFrameDataset(BaseFrameDataset):
    """TACO RGB-D HOI records from the verified extracted archive layout."""

    dataset_name = "taco"
    base_dataset_name = "taco"

    def __init__(self, root: str | Path, split: str = "all", *, load_rgb: bool = False, load_depth: bool = False) -> None:
        self._index_cache_path = _light_index_cache_path(_as_path(root), f"taco_frame_dataset_{split}")
        self._cache_pid = os.getpid()
        self._sequence_source_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        super().__init__(root, split, load_rgb=load_rgb, load_depth=load_depth)

    def _uses_lazy_records(self) -> bool:
        return True

    def _ensure_worker_local_caches(self) -> None:
        pid = os.getpid()
        if self._cache_pid == pid:
            return
        self._sequence_source_cache.clear()
        self._record_cache.clear()
        self._cache_pid = pid

    def _sequence_sources(self, index_entry: dict[str, Any]) -> dict[str, Any]:
        """Load immutable per-sequence annotations once per worker process."""
        self._ensure_worker_local_caches()
        cache_key = str(index_entry["sequence_id"])
        cached = _lru_get(self._sequence_source_cache, cache_key)
        if cached is not None:
            return cached
        object_pose_paths = [Path(str(value)) for value in index_entry["object_pose_paths"]]
        source = {
            "intrinsics": np.loadtxt(Path(str(index_entry["intrinsic_path"])), dtype=np.float32).reshape(3, 3),
            "world_to_camera": np.load(Path(str(index_entry["camera_pose_path"])), mmap_mode="r"),
            "hands": {
                side: (
                    _load_pickle(Path(str(index_entry[f"{side}_path"]))),
                    _load_pickle(Path(str(index_entry[f"{side}_shape_path"]))),
                )
                for side in ("left", "right")
            },
            "objects": [
                (
                    pose_path.stem,
                    self.root / "object_models_released" / f"{pose_path.stem.rsplit('_', 1)[-1]}_cm.obj",
                    np.load(pose_path, mmap_mode="r"),
                )
                for pose_path in object_pose_paths
            ],
        }
        return _lru_put(
            self._sequence_source_cache,
            cache_key,
            source,
            limit=_TACO_SEQUENCE_SOURCE_CACHE_LIMIT,
        )

    def _build_record_index(self) -> list[dict[str, Any]]:
        components = {
            "rgb": self.root / "Egocentric_RGB_Videos",
            "depth": _resolve_taco_depth_root(self.root),
            "camera": self.root / "Egocentric_Camera_Parameters",
            "hands": self.root / "Hand_Poses",
            "objects": self.root / "Object_Poses",
            "meshes": self.root / "object_models_released",
        }

        def _scan() -> list[dict[str, Any]]:
            missing = [name for name, path in components.items() if not path.is_dir()]
            if missing:
                raise FileNotFoundError(f"TACO requires verified extracted components: {', '.join(missing)}")
            records: list[dict[str, Any]] = []
            for intrinsic_path in sorted(components["camera"].glob("*/*/egocentric_intrinsic.txt")):
                sequence_dir = intrinsic_path.parent
                relative = sequence_dir.relative_to(components["camera"])
                action, sequence = relative.parts
                rgb_path = components["rgb"] / action / sequence / "color.mp4"
                depth_path = components["depth"] / action / sequence / "egocentric_depth.avi"
                extrinsic_path = sequence_dir / "egocentric_frame_extrinsic.npy"
                left_path = components["hands"] / action / sequence / "left_hand.pkl"
                right_path = components["hands"] / action / sequence / "right_hand.pkl"
                left_shape_path = components["hands"] / action / sequence / "left_hand_shape.pkl"
                right_shape_path = components["hands"] / action / sequence / "right_hand_shape.pkl"
                object_dir = components["objects"] / action / sequence
                required = (rgb_path, depth_path, extrinsic_path, left_path, right_path, left_shape_path, right_shape_path, object_dir)
                if not all(path.exists() for path in required):
                    continue
                camera_poses = np.load(extrinsic_path, mmap_mode="r")
                frame_count = int(camera_poses.shape[0]) if camera_poses.ndim == 3 and camera_poses.shape[1:] == (4, 4) else 0
                frame_count = min(frame_count, _video_frame_count(rgb_path), _video_frame_count(depth_path))
                if frame_count <= 0:
                    continue
                object_pose_paths = sorted(object_dir.glob("*.npy"))
                records.extend(
                    {
                        "sequence_id": f"{action}/{sequence}",
                        "frame_id": f"{frame_index + 1:05d}",
                        "temporal_index": frame_index,
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_path),
                        "intrinsic_path": str(intrinsic_path),
                        "camera_pose_path": str(extrinsic_path),
                        "left_path": str(left_path),
                        "right_path": str(right_path),
                        "left_shape_path": str(left_shape_path),
                        "right_shape_path": str(right_shape_path),
                        "object_pose_paths": [str(path) for path in object_pose_paths],
                    }
                    for frame_index in range(frame_count)
                )
            return records

        # The release is immutable after preparation.  Persist only component-root
        # signatures: storing every per-sequence source here makes the next
        # startup compare an incompatible signature set and rebuild unconditionally.
        return _load_or_build_light_index_cache(self._index_cache_path, list(components.values()), _scan, version=4)

    @staticmethod
    def _hand_payload(
        payload: Any,
        shape_payload: Any,
        frame_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        frame = payload.get(frame_id) if isinstance(payload, dict) else None
        if not isinstance(frame, dict):
            return None
        pose = _as_optional_array(frame.get("hand_pose"), shape=(48,))
        translation = _as_optional_array(frame.get("hand_trans"), shape=(3,))
        betas = _as_optional_array(shape_payload.get("hand_shape") if isinstance(shape_payload, dict) else None, shape=(10,))
        if pose is None or translation is None or betas is None:
            return None
        return pose, translation, betas

    def _materialize_record(self, index_entry: dict[str, Any]) -> FrameRecord:
        sources = self._sequence_sources(index_entry)
        frame_index = int(index_entry["temporal_index"])
        frame_id = str(index_entry["frame_id"])
        intrinsics = sources["intrinsics"]
        # The TACO release explicitly defines this sequence as world-to-camera.
        # Keep the 3R camera target and all geometry in that same camera frame.
        world_to_camera = sources["world_to_camera"][frame_index].astype(np.float32)
        camera_pose = world_to_camera
        hand_annos: list[HandAnnotation] = []
        for side in ("left", "right"):
            hand_payload, shape_payload = sources["hands"][side]
            result = self._hand_payload(hand_payload, shape_payload, frame_id)
            if result is None:
                continue
            pose, translation, betas = result
            hand_annos.append(
                HandAnnotation(
                    side=side,
                    visible=True,
                    mano_pose=pose,
                    mano_pose_format="axis_angle_full",
                    mano_betas=betas,
                    mano_trans=translation,
                    extras={
                        "mano_trans_is_root_position": True,
                        "mano_external_transform": world_to_camera,
                        "mano_flip_left_shapedirs": side == "left",
                    },
                )
            )
        scene_objects: list[dict[str, Any]] = []
        for object_name, mesh_path, object_poses in sources["objects"]:
            if frame_index >= len(object_poses):
                continue
            object_to_world = np.asarray(object_poses[frame_index], dtype=np.float32)
            if not mesh_path.is_file() or object_to_world.shape != (4, 4) or not np.isfinite(object_to_world).all():
                continue
            scene_objects.append(
                {
                    "object_id": object_name,
                    "mesh_path": str(mesh_path),
                    # TACO release meshes are stored in centimetres ("*_cm.obj");
                    # the common scene contract is metres.
                    "mesh_scale": 0.01,
                    "object_to_camera": (world_to_camera @ object_to_world).astype(np.float32),
                }
            )
        left_count, right_count = _count_sides(hand_annos)
        raw_depth_ref = MediaRef(kind="taco_depth_video", path=str(index_entry["depth_path"]), frame_index=frame_index)
        return FrameRecord(
            dataset_name=self.dataset_name,
            base_dataset_name=self.base_dataset_name,
            split=self.split,
            sequence_id=str(index_entry["sequence_id"]),
            frame_id=frame_id,
            temporal_index=frame_index,
            view_name="egocentric",
            is_egocentric=True,
            rgb_ref=MediaRef(kind="video_frame", path=str(index_entry["rgb_path"]), frame_index=frame_index),
            # Official FFV1 keeps the complete uint16 depth at scale 4000.
            # The legacy H.264 mirror remains supported by the media loader.
            depth_ref=raw_depth_ref,
            depth_mode="taco_depth_video",
            intrinsics=intrinsics,
            camera_pose=camera_pose,
            hand_annos=hand_annos,
            **_hand_supervision_flags(hand_annos),
            **_three_r_supervision_flags(depth_ref=raw_depth_ref, intrinsics=intrinsics, camera_pose=camera_pose),
            max_left_count=left_count,
            max_right_count=right_count,
            extras={
                "scene_objects": scene_objects,
                "depth_quantization_m": 1.0 / 4000.0,
            },
        )


class OakInkV1FrameDataset(BaseFrameDataset):
    """Static OakInk-v1 samples backed only by an explicit verified manifest."""

    dataset_name = "oakink_v1"
    base_dataset_name = "oakink_v1"
    allowed_temporal_lengths = (1,)

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        manifest_path: str | Path | None = None,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else Path(root) / "oakink_v1_verified_manifest.jsonl"
        super().__init__(root, split, load_rgb=load_rgb, load_depth=load_depth)

    @staticmethod
    def _finite_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
        matrix = _as_optional_array(value, shape=shape)
        return matrix if matrix is not None and np.isfinite(matrix).all() else None

    def _build_records(self) -> list[FrameRecord]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"OakInk-v1 requires a verified manifest: {self.manifest_path}")
        records: list[FrameRecord] = []
        for row in _load_jsonl(self.manifest_path):
            rgb_path = Path(str(row.get("rgb_path", "")))
            hand_path = Path(str(row.get("hand_path", "")))
            mesh_path = Path(str(row.get("mesh_path", "")))
            intrinsics = self._finite_matrix(row.get("intrinsics"), (3, 3))
            camera_pose = self._finite_matrix(row.get("camera_pose"), (4, 4))
            object_to_camera = self._finite_matrix(row.get("object_to_camera"), (4, 4))
            if not (rgb_path.is_file() and hand_path.is_file() and mesh_path.is_file()):
                continue
            if intrinsics is None or camera_pose is None or object_to_camera is None:
                continue
            try:
                hand_archive = np.load(hand_path)
                pose = _as_optional_array(hand_archive["pose"], shape=(48,))
                betas = _as_optional_array(hand_archive["betas"], shape=(10,))
                translation = _as_optional_array(hand_archive["translation"], shape=(3,))
            except (KeyError, OSError, ValueError):
                continue
            if pose is None or betas is None or translation is None:
                continue
            side = str(row.get("side", ""))
            if side not in {"left", "right"}:
                continue
            hand = HandAnnotation(
                side=side,
                visible=True,
                mano_pose=pose,
                mano_pose_format="axis_angle_full",
                mano_betas=betas,
                mano_trans=translation,
                extras={"mano_flip_left_shapedirs": side == "left"},
            )
            frame_id = str(row.get("sample_id", len(records)))
            records.append(
                FrameRecord(
                    dataset_name=self.dataset_name,
                    base_dataset_name=self.base_dataset_name,
                    split=self.split,
                    sequence_id=frame_id,
                    frame_id=frame_id,
                    temporal_index=0,
                    view_name="static",
                    is_egocentric=False,
                    rgb_ref=MediaRef(kind="path", path=str(rgb_path)),
                    depth_ref=None,
                    depth_mode=None,
                    intrinsics=intrinsics,
                    camera_pose=camera_pose,
                    hand_annos=[hand],
                    **_hand_supervision_flags([hand]),
                    **_three_r_supervision_flags(depth_ref=None, intrinsics=intrinsics, camera_pose=camera_pose),
                    max_left_count=1 if side == "left" else 0,
                    max_right_count=1 if side == "right" else 0,
                    extras={
                        "scene_objects": [{
                            "object_id": str(row.get("object_id", mesh_path.stem)),
                            "mesh_path": str(mesh_path),
                            "object_to_camera": object_to_camera,
                        }]
                    },
                )
            )
        return records


class HoloAssistFrameDataset(BaseFrameDataset):
    """Calibrated HoloAssist keypoint records prepared into an explicit manifest."""

    dataset_name = "holoassist"
    base_dataset_name = "holoassist"

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        manifest_path: str | Path | None = None,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else Path(root) / "holoassist_verified_manifest.jsonl"
        super().__init__(root, split, load_rgb=load_rgb, load_depth=load_depth)

    def _build_records(self) -> list[FrameRecord]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"HoloAssist requires a verified manifest: {self.manifest_path}")
        records: list[FrameRecord] = []
        for row in _load_jsonl(self.manifest_path):
            rgb_path = Path(str(row.get("rgb_path", "")))
            camera_pose_path = Path(str(row.get("camera_pose_path", "")))
            intrinsics = _as_optional_array(row.get("intrinsics"), shape=(3, 3))
            if not rgb_path.is_file() or not camera_pose_path.is_file() or intrinsics is None or not np.isfinite(intrinsics).all():
                continue
            try:
                camera_pose = np.asarray(np.load(camera_pose_path), dtype=np.float32).reshape(4, 4)
            except (OSError, ValueError):
                continue
            if not np.isfinite(camera_pose).all():
                continue
            hand_annos: list[HandAnnotation] = []
            for side in ("left", "right"):
                joints_value = row.get(f"{side}_joints_path")
                if not isinstance(joints_value, str):
                    continue
                joints_path = Path(joints_value)
                if not joints_path.is_file():
                    continue
                try:
                    joints = np.asarray(np.load(joints_path), dtype=np.float32).reshape(21, 3)
                except (OSError, ValueError):
                    continue
                if not np.isfinite(joints).all():
                    continue
                joints_2d = _project_points(joints, intrinsics)
                hand_annos.append(
                    HandAnnotation(
                        side=side,
                        visible=True,
                        bbox_xyxy=_bbox_from_points(joints_2d),
                        joints_3d=joints,
                        joints_2d=joints_2d,
                    )
                )
            if not hand_annos:
                continue
            depth_path_value = row.get("depth_path")
            depth_path = Path(depth_path_value) if isinstance(depth_path_value, str) else None
            depth_ref = MediaRef(kind="path", path=str(depth_path)) if depth_path is not None and depth_path.is_file() else None
            left_count, right_count = _count_sides(hand_annos)
            frame_id = str(row.get("frame_id", len(records)))
            records.append(
                FrameRecord(
                    dataset_name=self.dataset_name,
                    base_dataset_name=self.base_dataset_name,
                    split=self.split,
                    sequence_id=str(row.get("sequence_id", "holoassist")),
                    frame_id=frame_id,
                    temporal_index=int(row.get("temporal_index", row.get("frame_id", len(records)))),
                    view_name="egocentric",
                    is_egocentric=True,
                    rgb_ref=MediaRef(kind="path", path=str(rgb_path)),
                    depth_ref=depth_ref,
                    depth_mode=str(row.get("depth_mode", "meter")) if depth_ref is not None else None,
                    intrinsics=intrinsics,
                    camera_pose=camera_pose,
                    hand_annos=hand_annos,
                    **_hand_supervision_flags(hand_annos),
                    **_three_r_supervision_flags(depth_ref=depth_ref, intrinsics=intrinsics, camera_pose=camera_pose),
                    max_left_count=left_count,
                    max_right_count=right_count,
                    extras={"scene_objects": []},
                )
            )
        return records
