"""Lossless, per-window GT cache produced by the current six-dataset dataloader."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .egofound3r_gt import camera_c2w_from_batch, validate_window_row


CACHE_VERSION = "six_dataset_window_gt_v1"


def window_cache_id(row: Mapping[str, object]) -> str:
    """Stable ID independent of filesystem-unsafe sequence/window names."""
    dataset, sequence_id, frame_ids = validate_window_row(row)
    value = json.dumps(
        {"dataset": dataset, "sequence_id": sequence_id, "frame_ids": frame_ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(value).hexdigest()[:24]


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def cache_arrays_from_batch(batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract only metric targets; RGB and source data never enter the cache."""
    camera_c2w, camera_valid = camera_c2w_from_batch(batch)
    joints = _numpy(batch["joints_3d_targets"])[0]
    if joints.shape[1:] != (2, 21, 3):
        raise ValueError(f"canonical evaluation requires exactly two 21-joint hands, got {joints.shape}")
    arrays = {
        "camera_c2w": camera_c2w,
        "camera_valid": camera_valid,
        "intrinsics": _numpy(batch["intrinsics"])[0],
        "intrinsics_valid": _numpy(batch["intrinsics_supervision_mask"])[0].astype(bool),
        "hand_joints_camera": joints,
        "hand_valid": (
            _numpy(batch["hand_valid_mask"])[0]
            & _numpy(batch["raw_joint_supervision_mask"])[0]
        ).astype(bool),
        "joint_contact_target": _numpy(batch["contact_targets"])[0],
        "joint_contact_mask": _numpy(batch["contact_supervision_mask"])[0].astype(bool),
        "marker_contact_target": _numpy(batch["marker_contact_targets"])[0],
        "marker_contact_mask": _numpy(batch["marker_contact_supervision_mask"])[0].astype(bool),
    }
    depth = batch.get("depth")
    depth_valid = batch.get("depth_valid_mask")
    if depth is not None and depth_valid is not None:
        arrays["depth"] = _numpy(depth)[0]
        arrays["depth_valid"] = _numpy(depth_valid)[0].astype(bool)
    return arrays


def cache_metadata(row: Mapping[str, object], *, cache_id: str) -> dict[str, object]:
    dataset, sequence_id, frame_ids = validate_window_row(row)
    return {
        "cache_version": CACHE_VERSION,
        "cache_id": cache_id,
        "dataset": dataset,
        "sequence_id": sequence_id,
        "window_id": row.get("window_id"),
        "frame_ids": frame_ids,
        "window_manifest_fields": {
            key: row[key]
            for key in ("window_size", "window_stride", "window_overlap", "split")
            if key in row
        },
    }


def cache_paths(output_root: Path, row: Mapping[str, object]) -> tuple[Path, Path]:
    dataset, _, _ = validate_window_row(row)
    cache_id = window_cache_id(row)
    directory = output_root / dataset
    return directory / f"{cache_id}.npz", directory / f"{cache_id}.json"


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def write_window_cache(output_root: Path, row: Mapping[str, object], batch: Mapping[str, Any]) -> dict[str, object]:
    """Atomically write one cache entry, refusing mismatched existing targets."""
    data_path, metadata_path = cache_paths(output_root, row)
    cache_id = window_cache_id(row)
    metadata = cache_metadata(row, cache_id=cache_id)
    if data_path.exists() or metadata_path.exists():
        if not (data_path.is_file() and metadata_path.is_file()):
            raise FileExistsError(f"incomplete cache entry: {data_path}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(f"existing cache metadata differs: {metadata_path}")
        return {**metadata, "array_path": str(data_path), "status": "reused"}

    arrays = cache_arrays_from_batch(batch)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=data_path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(data_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _atomic_bytes(metadata_path, (json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n").encode())
    return {**metadata, "array_path": str(data_path), "status": "written"}


def load_window_cache(index_entry: Mapping[str, object]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata_path = Path(str(index_entry["metadata_path"]))
    array_path = Path(str(index_entry["array_path"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"unsupported GT cache version: {metadata.get('cache_version')}")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, arrays
