"""World-space hand sequences for the 3D multi-view comparison (8 methods).

Every method is expressed in the **same GT world frame** so the rows of the
summary/video are directly comparable:

* GT / WiLoR / PAD-Hand / EgoForce / ReViV4D live in the calibrated camera frame,
  so they are lifted with the window's calibrated ``camera_c2w``;
* HaWoR / Dyn-HaMR are native-SLAM world: each 60-frame window is rigidly placed
  in GT world at its first camera (``native_windows_in_gt_world``);
* EgoFound3R is predicted in its own camera frame: its vertices are lifted with
  the predicted ``camera_c2w`` and the window is rigidly aligned to GT world the
  same way, so the comparison keeps the model's own geometry without pretending
  its camera equals the calibrated one.

Column order matches the 2D contract; ReViV4D only has 21 joints, so it is drawn
as a skeleton instead of a 778 mesh.
"""
from __future__ import annotations

import numpy as np

from .inputs import WindowSources, native_windows_in_gt_world
from .joint_order import joints_in_gt_order

# Column order follows the 2D contract: baselines first, then EgoFound3R, then GT.
METHODS_3D = ("wilor", "pad_hand", "egoforce", "dyn_hamr", "hawor", "reviv4d",
              "ego", "gt")
METHOD_LABELS_3D = {
    "ego": "EgoFound3R", "wilor": "WiLoR", "pad_hand": "PAD-Hand",
    "egoforce": "EgoForce", "dyn_hamr": "Dyn-HaMR", "hawor": "HaWoR",
    "reviv4d": "ReViV4D", "gt": "GT",
}
SKELETON_METHODS = ("reviv4d",)
MESH_METHODS = tuple(method for method in METHODS_3D if method not in SKELETON_METHODS)


def world_from_camera(vertices_camera: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """(T,2,N,3) camera-space points -> world, per-frame ``c2w`` (T,4,4)."""
    rotation = np.asarray(c2w, float)[:, :3, :3]
    translation = np.asarray(c2w, float)[:, :3, 3]
    return np.einsum("tij,tsvj->tsvi", rotation, np.asarray(vertices_camera, float)) \
        + translation[:, None, None, :]


def per_vertex_valid(hand_valid: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """(T,2) frame flags + finite vertices -> (T,2,N) per-vertex flags."""
    finite = np.isfinite(vertices).all(-1)
    return np.asarray(hand_valid, bool)[:, :, None] & finite


def _window_world(method: str, window: WindowSources, mano, vertices_camera=None):
    """World-space vertices/joints of one 60-frame window, or None when absent."""
    gt = window.methods.get("gt")
    if gt is None:
        return None, None, None
    gt_c2w = np.asarray(gt["camera_c2w"], float)
    if method in ("gt", "wilor", "pad_hand", "egoforce", "reviv4d"):
        data = window.methods.get(method)
        if data is None:
            return None, None, None
        if method == "reviv4d":
            # ReViV4D emits the native MANO ordering, so it has to be mapped onto the
            # GT skeleton convention before the shared edge list is used (same
            # permutation the 2D pipeline applies).
            joints = joints_in_gt_order("reviv4d", np.asarray(data["hand_joints_camera"], float))
            return None, world_from_camera(joints, gt_c2w), np.asarray(data["hand_valid"], bool)
        camera = np.asarray(data["hand_vertices_camera"], float)
        return world_from_camera(camera, gt_c2w), None, np.asarray(data["hand_valid"], bool)

    if method == "ego":
        data = window.methods.get("ego")
        if data is None:
            return None, None, None
        camera = np.asarray(data["hand_vertices_camera"], float)
        c2w = np.asarray(data["camera_c2w"], float)
        predicted_world = world_from_camera(camera, c2w)
        valid = np.asarray(data["hand_valid"], bool)
        aligned, _ = native_windows_in_gt_world(predicted_world, c2w, gt_c2w,
                                                camera_valid=valid.any(axis=1))
        return aligned, None, valid

    data = window.methods.get(method)
    if data is None:
        return None, None, None
    if "hand_vertices_world" not in data:
        return None, None, None
    native_world = np.asarray(data["hand_vertices_world"], float)
    native_c2w = np.asarray(data["camera_c2w"], float)
    valid = np.asarray(data["hand_valid"], bool)
    aligned, _ = native_windows_in_gt_world(native_world, native_c2w, gt_c2w,
                                            camera_valid=valid.any(axis=1))
    return aligned, None, valid


def build_segment_sequences(segment, mano) -> dict:
    """Return per method ``{vertices, joints, valid, faces}`` in GT world.

    Every method is padded to the full segment length so all sequences share one
    time axis: a window without a registered prediction (Dyn-HaMR covers 1 of 5
    windows here) contributes NaN vertices with ``valid=False`` and simply renders
    as an empty cell for that stretch.
    """
    vertex_count = int(mano.faces.max()) + 1
    out = {method: {"vertices": [], "joints": [], "valid": []} for method in METHODS_3D}
    kinds = {}
    for window in segment.windows:
        frames = len(window.frame_ids)
        for method in METHODS_3D:
            vertices, joints, valid = _window_world(method, window, mano)
            if vertices is not None:
                kinds[method] = "mesh"
            elif joints is not None and method not in kinds:
                kinds[method] = "joints"
            if valid is not None:
                valid = np.asarray(valid, bool)
            if kinds.get(method) == "joints":
                count = joints.shape[2] if joints is not None else 21
                if joints is None:
                    joints = np.full((frames, 2, count, 3), np.nan)
                    valid = np.zeros((frames, 2), bool)
                out[method]["joints"].append(joints)
                out[method]["valid"].append(valid)
                continue
            if vertices is None:
                vertices = np.full((frames, 2, vertex_count, 3), np.nan)
                valid = np.zeros((frames, 2), bool)
            out[method]["vertices"].append(vertices)
            out[method]["valid"].append(valid)
    result = {}
    for method, store in out.items():
        if not store["valid"]:
            continue
        valid_frames = np.concatenate(store["valid"], axis=0)
        entry = {"faces": mano.faces, "valid": valid_frames}
        if store["vertices"]:
            joined = np.concatenate(store["vertices"], axis=0)
            entry["vertices"] = joined
            entry["valid"] = per_vertex_valid(valid_frames, joined)
        if store["joints"]:
            entry["joints"] = np.concatenate(store["joints"], axis=0)
        result[method] = entry
    return result
