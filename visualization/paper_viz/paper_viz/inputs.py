"""Loaders for per-window method predictions and per-frame RGB/geometry sources."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mano import ManoAsset


def arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def camera_points(data: dict, field_name: str) -> np.ndarray:
    """Return (T, 2, N, 3) camera-space points, converting world via own c2w when needed."""
    key = f"hand_{field_name}_camera"
    if key in data:
        return data[key].astype(float)
    world_key = f"hand_{field_name}_world"
    pose = data["camera_c2w"].astype(float)
    return np.einsum("tji,tsvj->tsvi", pose[:, :3, :3], data[world_key] - pose[:, None, None, :3, 3])


def native_windows_in_gt_world(native_world, native_c2w, gt_c2w, camera_valid=None):
    """Rigidly place each native-SLAM 60-frame window in common GT world at its first camera."""
    world = np.empty_like(native_world, dtype=float)
    cameras = np.empty_like(native_c2w, dtype=float)
    camera_valid = np.ones(len(native_c2w), bool) if camera_valid is None else np.asarray(camera_valid, bool)
    for start in range(0, len(native_c2w), 60):
        stop = start + 60
        if not camera_valid[start] or not np.isfinite(native_c2w[start]).all():
            world[start:stop] = np.nan
            cameras[start:stop] = np.nan
            continue
        transform = gt_c2w[start] @ np.linalg.inv(native_c2w[start])
        world[start:stop] = np.einsum("ij,tsvj->tsvi", transform[:3, :3], native_world[start:stop]) + transform[:3, 3]
        cameras[start:stop] = np.einsum("ij,tjk->tik", transform, native_c2w[start:stop])
    return world, cameras


@dataclass
class WindowSources:
    cache_id: str
    window_id: str
    frame_ids: list
    rgb_dir: Path
    geometry_dir: Path
    record: dict
    selected_indices: list[int] = field(default_factory=list)
    methods: dict = field(default_factory=dict)          # name -> npz dict (T=60)
    baselines: dict = field(default_factory=dict)        # name -> npz dict
    ego_native: dict | None = None                       # marker_visibility etc.
    cache: dict = field(default_factory=dict)            # per-window derived arrays (vertices, joints)


@dataclass
class SegmentSources:
    dataset: str
    sequence_id: str
    segment_id: str
    windows: list
    mano: ManoAsset
    mapping_path: Path


def load_window(cache_id: str, window_id: str, prepared_root: Path, method_files: dict,
                baseline_files: dict, mano: ManoAsset,
                selected_indices: list[int] | None = None) -> WindowSources:
    window_root = prepared_root / cache_id
    record = json.loads((window_root / "window_input.json").read_text())
    methods = {}
    for name, path in method_files.items():
        if path is None or not Path(path).is_file():
            continue
        data = arrays(Path(path))
        methods[name] = data
    baselines = {}
    for name, path in baseline_files.items():
        if path is not None and Path(path).is_file():
            baselines[name] = arrays(Path(path))
    return WindowSources(
        cache_id=cache_id,
        window_id=window_id,
        frame_ids=record["frame_ids"],
        rgb_dir=window_root / "rgb",
        geometry_dir=window_root / "geometry",
        record=record,
        selected_indices=(list(selected_indices) if selected_indices is not None
                          else list(range(len(record["frame_ids"])))),
        methods=methods,
        baselines=baselines,
    )


def rgb_path(window: WindowSources, index: int) -> Path:
    name = f"{index:03d}_{window.frame_ids[index]}.png"
    return window.rgb_dir / name


def geometry_path(window: WindowSources, index: int) -> Path:
    name = f"{index:03d}_{window.frame_ids[index]}.npz"
    return window.geometry_dir / name


def method_vertices_camera(data: dict, mano: ManoAsset) -> np.ndarray:
    """(T, 2, 778, 3) camera-space hand vertices for any method contract."""
    if "hand_vertices_camera" in data:
        return data["hand_vertices_camera"].astype(float)
    if "hand_markers_camera" in data:
        return mano.upsample(data["hand_markers_camera"].astype(float))
    if "hand_vertices_world" in data:
        pose = data["camera_c2w"].astype(float)
        return np.einsum("tji,tsvj->tsvi", pose[:, :3, :3],
                         data["hand_vertices_world"].astype(float) - pose[:, None, None, :3, 3])
    if "hand_markers_world" in data:
        pose = data["camera_c2w"].astype(float)
        markers = np.einsum("tji,tsvj->tsvi", pose[:, :3, :3],
                            data["hand_markers_world"].astype(float) - pose[:, None, None, :3, 3])
        return mano.upsample(markers)
    raise KeyError(list(data))


def method_joints_camera(data: dict) -> np.ndarray:
    return camera_points(data, "joints")
