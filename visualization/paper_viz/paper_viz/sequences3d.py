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

# Which methods predict their own camera extrinsics, and which predict intrinsics
# too.  Everything else falls back to the calibrated (GT) camera, which is what the
# frozen evaluation used for those methods in the first place.
PREDICTED_EXTRINSICS = ("ego", "hawor", "dyn_hamr", "reviv4d")
PREDICTED_INTRINSICS = ("ego",)
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
    """World-space geometry **and** camera of one window, or None when absent.

    Returns ``(vertices, joints, valid, camera)`` where ``camera`` is a dict with
    the extrinsics/intrinsics that should be drawn for this method:

    * methods that ship their own camera (EgoFound3R, HaWoR, Dyn-HaMR, ReViV4D)
      use it, rigidly rebased into the same GT world as their hands;
    * the others (GT, WiLoR, PAD-Hand, EgoForce) were evaluated in the calibrated
      camera frame, so the calibrated camera is theirs;
    * intrinsics fall back to the calibrated K unless the method predicts its own
      (only EgoFound3R does, as ``intrinsics_pred``).
    """
    gt = window.methods.get("gt")
    if gt is None:
        return None, None, None, None
    gt_c2w = np.asarray(gt["camera_c2w"], float)
    gt_valid = np.asarray(gt.get("camera_valid", np.ones(len(gt_c2w), bool)), bool)
    gt_K = np.asarray(gt["intrinsics"], float)
    gt_K_valid = np.asarray(gt.get("intrinsics_valid", np.ones(len(gt_c2w), bool)), bool)
    size_hw = tuple(int(v) for v in window.record.get("image_size_hw", (2000, 2800)))

    def calibrated(frames):
        return {"c2w": gt_c2w, "valid": gt_valid, "K": gt_K, "K_valid": gt_K_valid,
                "size_hw": size_hw, "source": "gt", "frames": frames}

    if method in ("gt", "wilor", "pad_hand", "egoforce", "reviv4d"):
        data = window.methods.get(method)
        if data is None:
            return None, None, None, None
        if method == "reviv4d":
            # ReViV4D emits the native MANO ordering, so it has to be mapped onto the
            # GT skeleton convention before the shared edge list is used (same
            # permutation the 2D pipeline applies).  Its joints live in its own
            # predicted camera frame and it ships that camera, so the window is
            # rebased into GT world exactly like the native-world methods; only the
            # intrinsics fall back to the calibrated K (ReViV4D predicts none).
            joints = joints_in_gt_order("reviv4d", np.asarray(data["hand_joints_camera"], float))
            valid = np.asarray(data["hand_valid"], bool)
            own_c2w = np.asarray(data["camera_c2w"], float)
            own_valid = np.asarray(data.get("camera_valid", np.ones(len(own_c2w), bool)), bool)
            own_world = world_from_camera(joints, own_c2w)
            aligned, aligned_c2w = native_windows_in_gt_world(own_world, own_c2w, gt_c2w,
                                                              camera_valid=own_valid)
            camera = {"c2w": aligned_c2w, "valid": np.isfinite(aligned_c2w).all(axis=(1, 2)),
                      "K": gt_K, "K_valid": gt_K_valid, "size_hw": size_hw, "source": "pred",
                      "frames": None}
            return None, aligned, valid, camera
        camera = np.asarray(data["hand_vertices_camera"], float)
        return (world_from_camera(camera, gt_c2w), None, np.asarray(data["hand_valid"], bool),
                calibrated(None))

    if method == "ego":
        data = window.methods.get("ego")
        if data is None:
            return None, None, None, None
        camera = np.asarray(data["hand_vertices_camera"], float)
        c2w = np.asarray(data["camera_c2w"], float)
        predicted_world = world_from_camera(camera, c2w)
        valid = np.asarray(data["hand_valid"], bool)
        aligned, aligned_c2w = native_windows_in_gt_world(predicted_world, c2w, gt_c2w,
                                                          camera_valid=valid.any(axis=1))
        K = gt_K
        K_valid = gt_K_valid
        pred_size = size_hw
        if "intrinsics_pred" in data:
            K_pred = np.asarray(data["intrinsics_pred"], float)
            if "input_affine" in data:
                affine = np.asarray(data["input_affine"], float)
                K_pred = np.diag([1.0 / affine[0, 0], 1.0 / affine[1, 1], 1.0]) @ K_pred
            K = K_pred
            K_valid = np.ones(len(K_pred), bool)
            if "source_size_hw" in data:
                pred_size = tuple(int(v) for v in np.asarray(data["source_size_hw"]).ravel())
        camera = {"c2w": aligned_c2w, "valid": np.isfinite(aligned_c2w).all(axis=(1, 2)),
                  "K": K, "K_valid": K_valid, "size_hw": pred_size, "source": "pred",
                  "frames": None}
        return aligned, None, valid, camera

    data = window.methods.get(method)
    if data is None:
        return None, None, None, None
    if "hand_vertices_world" not in data:
        return None, None, None, None
    native_world = np.asarray(data["hand_vertices_world"], float)
    native_c2w = np.asarray(data["camera_c2w"], float)
    valid = np.asarray(data["hand_valid"], bool)
    aligned, aligned_c2w = native_windows_in_gt_world(native_world, native_c2w, gt_c2w,
                                                      camera_valid=valid.any(axis=1))
    camera = {"c2w": aligned_c2w, "valid": np.isfinite(aligned_c2w).all(axis=(1, 2)),
              "K": gt_K, "K_valid": gt_K_valid, "size_hw": size_hw, "source": "pred",
              "frames": None}
    return aligned, None, valid, camera


def concatenate_cameras(cameras: list, total: int) -> dict:
    """Stack the per-window camera bundle over the clip (pad missing windows).

    Every camera array is padded to ``total`` frames so a method with a sparse or
    absent prediction (Dyn-HaMR) still lines up with the shared time axis.
    """
    first = cameras[0]
    c2w = np.concatenate([c["c2w"] for c in cameras], axis=0)
    valid = np.concatenate([np.asarray(c["valid"], bool) for c in cameras])
    K = np.concatenate([c["K"] for c in cameras], axis=0)
    K_valid = np.concatenate([np.asarray(c["K_valid"], bool) for c in cameras])
    if len(c2w) < total:
        pad = total - len(c2w)
        c2w = np.concatenate([c2w, np.full((pad,) + c2w.shape[1:], np.nan)], axis=0)
        valid = np.concatenate([valid, np.zeros(pad, bool)])
        K = np.concatenate([K, np.repeat(K[-1:], pad, axis=0)], axis=0)
        K_valid = np.concatenate([K_valid, np.zeros(pad, bool)])
    elif len(c2w) > total:
        c2w, valid, K, K_valid = c2w[:total], valid[:total], K[:total], K_valid[:total]
    return {"c2w": c2w, "valid": valid, "K": K, "K_valid": K_valid,
            "size_hw": tuple(int(v) for v in first["size_hw"]), "source": first["source"]}


def build_segment_sequences(segment, mano) -> dict:
    """Return per method ``{vertices, joints, valid, faces}`` in GT world.

    Every method is padded to the full segment length so all sequences share one
    time axis: a window without a registered prediction (Dyn-HaMR covers 1 of 5
    windows here) contributes NaN vertices with ``valid=False`` and simply renders
    as an empty cell for that stretch.
    """
    vertex_count = int(mano.faces.max()) + 1
    out = {method: {"vertices": [], "joints": [], "valid": [], "camera": []} for method in METHODS_3D}
    kinds = {}
    for window in segment.windows:
        frames = len(window.frame_ids)
        for method in METHODS_3D:
            vertices, joints, valid, camera = _window_world(method, window, mano)
            if camera is not None:
                out[method]["camera"].append(camera)
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
        if store["camera"]:
            entry["camera"] = concatenate_cameras(store["camera"], len(valid_frames))
        result[method] = entry
    return result
