"""Materialize exact dataloader windows as method-neutral RGB and geometry inputs."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .egofound3r_gt import SixDatasetGroundTruth, validate_window_row
from .six_dataset_gt_cache import window_cache_id


WINDOW_INPUT_VERSION = "six_dataset_window_input_v1"


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _rgb_uint8(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected RGB tensor [3,H,W] or [H,W,3], got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
    return np.asarray(array, dtype=np.uint8)


def _object_name(frame: Mapping[str, Any], dataset: str) -> str:
    objects = frame.get("scene_objects", [])
    if isinstance(objects, list) and objects and isinstance(objects[0], Mapping):
        value = objects[0].get("object_id", objects[0].get("mesh_path"))
        if value is not None:
            return str(value)
    return f"{dataset}_object"


def _materialize_30fps_video(directory: Path, rgb_paths: list[Path]) -> tuple[Path, Path]:
    """Create the common contiguous 30 FPS clip required by video baselines."""
    video = directory / "input.mp4"
    mapping_path = directory / "mapping.json"
    if video.is_file() and mapping_path.is_file():
        return video, mapping_path
    frame_dir = directory / "video_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    suffix = rgb_paths[0].suffix.lower()
    if any(path.suffix.lower() != suffix for path in rgb_paths):
        raise ValueError("window RGB extensions must match for video export")
    for index, path in enumerate(rgb_paths):
        link = frame_dir / f"{index:06d}{suffix}"
        if not link.exists():
            link.symlink_to(path.resolve())
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
        "-i", str(frame_dir / f"%06d{suffix}"), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(video),
    ], check=True)
    return video, mapping_path


def materialize_window_input(
    bridge: SixDatasetGroundTruth,
    row: Mapping[str, object],
    output_root: Path,
) -> Path:
    """Write decoded PNGs and camera-space geometry outside the source datasets."""
    from PIL import Image

    dataset, sequence_id, frame_ids = validate_window_row(row)
    cache_id = window_cache_id(row)
    directory = output_root / dataset / cache_id
    record_path = directory / "window_input.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("dataset") == dataset and record.get("sequence_id") == sequence_id and record.get("frame_ids") == frame_ids:
            return record_path
        raise ValueError(f"existing input record differs: {record_path}")

    frames = bridge.geometry_for_window(row)
    if len(frames) != len(frame_ids):
        raise RuntimeError(f"{dataset}/{sequence_id}: geometry frame count drift")
    rgb_paths: list[str] = []
    geometry_paths: list[str] = []
    for index, (frame_id, frame) in enumerate(zip(frame_ids, frames, strict=True)):
        rgb_path = directory / "rgb" / f"{index:03d}_{frame_id}.png"
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_rgb_uint8(frame["rgb"]), mode="RGB").save(rgb_path)
        vertices = np.full((2, 778, 3), np.nan, dtype=np.float32)
        joints = np.full((2, 21, 3), np.nan, dtype=np.float32)
        valid = np.asarray(frame["hand_valid"], dtype=bool)
        for side in range(2):
            if valid[side]:
                vertices[side] = frame["hand_vertices"][side]
                joints[side] = frame["hand_joints"][side]
        geometry_path = directory / "geometry" / f"{index:03d}_{frame_id}.npz"
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            geometry_path,
            hand_vertices=vertices,
            hand_joints=joints,
            hand_valid=valid,
            object_vertices=np.asarray(frame["object_vertices"] if frame["object_vertices"] is not None else np.empty((0, 3)), dtype=np.float32),
            object_faces=np.asarray(frame["object_faces"] if frame["object_faces"] is not None else np.empty((0, 3)), dtype=np.int64),
        )
        rgb_paths.append(str(rgb_path))
        geometry_paths.append(str(geometry_path))
    video_path, video_mapping_path = _materialize_30fps_video(directory, [Path(path) for path in rgb_paths])
    video_mapping = {
        "dataset": dataset,
        "sequence": sequence_id,
        "window_id": row.get("window_id"),
        "frame_ids": frame_ids,
        "context_frame_ids": frame_ids,
        "hand_indices_30fps": list(range(len(frame_ids))),
        "context_frames": len(frame_ids),
        "input_fps": 30.0,
        "duration_seconds": len(frame_ids) / 30.0,
    }
    _atomic_text(video_mapping_path, json.dumps(video_mapping, ensure_ascii=False, indent=2) + "\n")
    record = {
        "window_input_version": WINDOW_INPUT_VERSION,
        "dataset": dataset,
        "cache_id": cache_id,
        "sequence_id": sequence_id,
        "window_id": row.get("window_id"),
        "frame_ids": frame_ids,
        "rgb_paths": rgb_paths,
        "geometry_paths": geometry_paths,
        "object_names": [_object_name(frame, dataset) for frame in frames],
        "video_path": str(video_path),
        "video_mapping_path": str(video_mapping_path),
    }
    _atomic_text(record_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record_path


def load_window_input(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("window_input_version") != WINDOW_INPUT_VERSION:
        raise ValueError(f"unsupported window input: {path}")
    frame_ids, rgb_paths, geometry_paths = record.get("frame_ids"), record.get("rgb_paths"), record.get("geometry_paths")
    if not isinstance(frame_ids, list) or not isinstance(rgb_paths, list) or not isinstance(geometry_paths, list):
        raise ValueError(f"invalid window input: {path}")
    if len(frame_ids) != len(rgb_paths) or len(frame_ids) != len(geometry_paths):
        raise ValueError(f"window input frame count mismatch: {path}")
    missing = [value for value in [*rgb_paths, *geometry_paths] if not Path(str(value)).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return record
