"""Build the shared 3D display scene: world geometry, frustums, trajectories, floor."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .inputs import WindowSources, method_vertices_camera, native_windows_in_gt_world
from .joint_order import joints_in_gt_order

NATIVE_METHODS = ("ego", "hawor", "dyn_hamr")
FRUSTUM_DEPTH = 0.032
SKELETON_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                  (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
                  (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))
FRUSTUM_EDGES = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))


@dataclass
class Scene:
    xyz: dict = field(default_factory=dict)          # method -> (T, 2, 778, 3) display coords
    valid: dict = field(default_factory=dict)        # method -> (T, 2) bool
    trajectories: dict = field(default_factory=dict) # method -> (T, 2, 3) display coords
    traj_valid: dict = field(default_factory=dict)
    camera_parts: list = field(default_factory=list) # GT frustum + center path segments
    floor: np.ndarray = None
    bounds: tuple = None
    selected: np.ndarray = None
    frame_ids: list = None
    sequence_id: str = ""
    segment_id: str = ""
    dataset: str = ""
    intrinsics: np.ndarray = None
    source_hw: tuple = None
    faces: np.ndarray = None
    joints: dict = field(default_factory=dict)         # method -> (T, 2, 21, 3) display
    joints_valid: dict = field(default_factory=dict)


def stitch_windows(windows: list[WindowSources], method: str, mano) -> tuple:
    """Concatenate per-window camera-space vertices (T=300, 2, 778, 3) and validity."""
    pieces = []
    valids = []
    for window in windows:
        data = window.methods.get(method)
        if data is None:
            pieces.append(np.full((60, 2, 778, 3), np.nan))
            valids.append(np.zeros((60, 2), bool))
            continue
        try:
            vertices = method_vertices_camera(data, mano)
        except KeyError:
            pieces.append(np.full((60, 2, 778, 3), np.nan))
            valids.append(np.zeros((60, 2), bool))
            continue
        joints = data.get("hand_joints_camera")
        if joints is None and "hand_joints_world" in data:
            pose = data["camera_c2w"].astype(float)
            joints = np.einsum("tji,tsvj->tsvi", pose[:, :3, :3],
                               data["hand_joints_world"].astype(float) - pose[:, None, None, :3, 3])
        valid = data["hand_valid"].astype(bool) & np.isfinite(vertices).all(axis=(2, 3))
        if joints is not None:
            valid &= np.isfinite(joints).all(axis=(2, 3))
        pieces.append(vertices)
        valids.append(valid)
    return np.concatenate(pieces), np.concatenate(valids)


def build_scene(windows: list[WindowSources], methods: tuple, mano, selected: np.ndarray,
                sequence_id: str, segment_id: str, dataset: str) -> Scene:
    faces = mano.faces
    gt_camera = np.concatenate([w.methods["gt"]["camera_c2w"].astype(float) for w in windows])
    gt_valid_camera = np.concatenate([w.methods["gt"]["camera_valid"].astype(bool) for w in windows])
    display_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]) @ gt_camera[0, :3, :3].T
    origin = gt_camera[0, :3, 3]

    def to_display(points):
        return (points - origin) @ display_rotation.T

    def to_world(camera_points_array, c2w):
        return np.einsum("tij,tsvj->tsvi", c2w[:, :3, :3], camera_points_array) + c2w[:, None, None, :3, 3]

    scene = Scene(sequence_id=sequence_id, segment_id=segment_id, dataset=dataset,
                  selected=selected, frame_ids=[fid for w in windows for fid in w.frame_ids],
                  faces=faces)
    for method in methods:
        if method == "gt":
            camera = np.concatenate([w.methods["gt"]["hand_vertices_camera"].astype(float) for w in windows])
            valid = np.concatenate([w.methods["gt"]["hand_valid"].astype(bool) for w in windows])
            c2w = gt_camera
        elif method == "ego":
            camera, valid = stitch_windows(windows, "ego", mano)
            c2w = np.concatenate([w.methods["ego"]["camera_c2w"].astype(float) for w in windows])
        else:
            camera, valid = stitch_windows(windows, method, mano)
            data0 = windows[0].methods.get(method)
            c2w = (np.concatenate([w.methods[method]["camera_c2w"].astype(float) for w in windows])
                   if data0 is not None and "camera_c2w" in data0 else gt_camera)
        if method in NATIVE_METHODS:
            world, _ = native_windows_in_gt_world(
                to_world(camera, c2w if method != "ego" else c2w), c2w, gt_camera, gt_valid_camera)
        else:
            world = to_world(camera, gt_camera)
        scene.xyz[method] = to_display(world)
        scene.valid[method] = valid & np.isfinite(scene.xyz[method]).all(axis=(2, 3))
        joints = []
        joints_valid = []
        for window in windows:
            data = window.methods.get(method)
            if data is None:
                joints.append(np.full((60, 2, 3), np.nan))
                joints_valid.append(np.zeros((60, 2), bool))
                continue
            if "hand_joints_camera" in data:
                jc = data["hand_joints_camera"].astype(float)
            elif "hand_joints_world" in data:
                pose = data["camera_c2w"].astype(float)
                jc = np.einsum("tji,tsvj->tsvi", pose[:, :3, :3],
                               data["hand_joints_world"].astype(float) - pose[:, None, None, :3, 3])
            else:
                jc = camera_points_fallback(window, method)
            joints.append(jc.mean(2) if jc.shape[2] > 1 else jc[:, :, 0])
            joints_valid.append(data["hand_valid"].astype(bool))
        joint_world_segments = []
        for index, window in enumerate(windows):
            data = window.methods.get(method)
            jc = joints[index]
            if data is None:
                joint_world_segments.append(np.full((60, 2, 3), np.nan))
                continue
            c2w_window = (data["camera_c2w"].astype(float) if "camera_c2w" in data else
                          window.methods["gt"]["camera_c2w"].astype(float))
            jc_world = (np.einsum("tij,tsj->tsi", c2w_window[:, :3, :3], jc)
                        + c2w_window[:, None, :3, 3])
            if method in NATIVE_METHODS:
                world_j, _ = native_windows_in_gt_world(jc_world[:, :, None, :], c2w_window,
                                                        window.methods["gt"]["camera_c2w"].astype(float),
                                                        window.methods["gt"]["camera_valid"].astype(bool))
                world_j = world_j[:, :, 0, :]
            else:
                gt_c2w_window = window.methods["gt"]["camera_c2w"].astype(float)
                world_j = (np.einsum("tij,tsj->tsi", gt_c2w_window[:, :3, :3], jc)
                           + gt_c2w_window[:, None, :3, 3])
            joint_world_segments.append(world_j)
        joint_world = np.concatenate(joint_world_segments)
        scene.trajectories[method] = to_display(joint_world)
        full_joints = []
        full_valid = []
        for window in windows:
            data = window.methods.get(method)
            if data is None or "hand_joints_camera" not in data and "hand_joints_world" not in data:
                full_joints.append(np.full((60, 2, 21, 3), np.nan))
                full_valid.append(np.zeros((60, 2), bool))
                continue
            if "hand_joints_camera" in data:
                jc = joints_in_gt_order(method, data["hand_joints_camera"].astype(float))
                c2w_use = window.methods["gt"]["camera_c2w"].astype(float)
                wj = np.einsum("tij,tsvj->tsvi", c2w_use[:, :3, :3], jc) + c2w_use[:, None, None, :3, 3]
            else:
                pose = data["camera_c2w"].astype(float)
                jc = joints_in_gt_order(method, np.einsum(
                    "tji,tsvj->tsvi", pose[:, :3, :3],
                    data["hand_joints_world"].astype(float) - pose[:, None, None, :3, 3]))
                wj = np.einsum("tij,tsvj->tsvi", pose[:, :3, :3], jc) + pose[:, None, None, :3, 3]
                if method in NATIVE_METHODS:
                    wj, _ = native_windows_in_gt_world(wj, pose,
                                                       window.methods["gt"]["camera_c2w"].astype(float),
                                                       window.methods["gt"]["camera_valid"].astype(bool))
            full_joints.append(wj)
            full_valid.append(data["hand_valid"].astype(bool))
        scene.joints[method] = to_display(np.concatenate(full_joints))
        scene.joints_valid[method] = np.concatenate(full_valid)
        scene.traj_valid[method] = np.concatenate(joints_valid) & np.isfinite(joint_world).all(axis=2)

    # GT camera frustums at selected frames plus the center path, shared by all rows.
    intrinsics = np.concatenate([w.methods["gt"]["intrinsics"].astype(float) for w in windows])
    intrinsics_valid = np.concatenate([w.methods["gt"]["intrinsics_valid"].astype(bool) for w in windows])
    record = windows[0].record
    source_hw = None
    for window in windows:
        ego = window.methods.get("ego")
        if ego is not None:
            source_hw = tuple(ego.get("source_resolution_hw", (0, 0))) or None
            break
    if source_hw is None or source_hw == (0, 0):
        source_hw = (480, 640)
    scene.intrinsics = intrinsics
    scene.source_hw = source_hw
    height, width = source_hw
    for t in selected:
        if not intrinsics_valid[t]:
            continue
        pixels = np.array([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]], float)
        rays = pixels @ np.linalg.inv(intrinsics[t]).T
        corners = np.vstack([np.zeros(3), rays * FRUSTUM_DEPTH])
        corners = to_display(corners @ gt_camera[t, :3, :3].T + gt_camera[t, :3, 3])
        scene.camera_parts.extend([[corners[a], corners[b]] for a, b in FRUSTUM_EDGES])
    centers = to_display(gt_camera[:, :3, 3])
    scene.camera_parts.extend([[a, b] for a, b in zip(centers[:-1], centers[1:])
                               if np.isfinite(a).all() and np.isfinite(b).all()])

    points = [scene.xyz[m][selected][scene.valid[m][selected]].reshape(-1, 3) for m in methods
              if scene.valid[m][selected].any()]
    points.append(np.asarray(scene.camera_parts).reshape(-1, 3))
    extent_points = np.concatenate(points)
    low, high = extent_points.min(0), extent_points.max(0)
    center = (low + high) / 2
    floor_width = max(high[0] - low[0], 0.3) * 1.12
    floor_depth = max(high[1] - low[1], 0.3) * 1.12
    scene.floor = np.array([[center[0] + a * floor_width / 2, center[1] + b * floor_depth / 2,
                             low[2] - 0.03] for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    hand_points = [scene.xyz[m][selected][scene.valid[m][selected]].reshape(-1, 3) for m in methods
                   if scene.valid[m][selected].any()]
    points = np.concatenate(hand_points + [scene.floor])
    scene.bounds = (points.min(0), points.max(0))
    return scene


def camera_points_fallback(window: WindowSources, method: str) -> np.ndarray:
    data = window.methods[method]
    from .inputs import camera_points
    return camera_points(data, "joints")


def select_frames(valid_gt: np.ndarray, count: int = 5) -> np.ndarray:
    keep = np.flatnonzero(valid_gt.any(1))
    targets = np.linspace(0, len(keep) - 1, count)
    return keep[np.rint(targets).astype(int)]
