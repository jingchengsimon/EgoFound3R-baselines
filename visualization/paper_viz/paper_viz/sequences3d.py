"""World-space hand sequences for the 3D multi-view comparison (8 methods).

Every method is expressed in the **same GT world frame** so the rows of the
summary/video are directly comparable:

* GT / WiLoR / PAD-Hand / EgoForce / ReViV4D live in the calibrated camera frame,
  so they are lifted with the window's calibrated ``camera_c2w``;
* HaWoR / Dyn-HaMR are native-SLAM world: the first 60-frame window is placed in
  GT world, then later windows inherit the previous predicted drift and advance by
  the GT camera motion across each boundary;
* EgoFound3R is predicted in its own camera frame: its vertices are lifted with
  the predicted ``camera_c2w`` and the window is rigidly aligned to GT world the
  same way, so the comparison keeps the model's own geometry without pretending
  its camera equals the calibrated one.

Column order matches the 2D contract; ReViV4D only has 21 joints, so it is drawn
as a skeleton instead of a 778 mesh.
"""
from __future__ import annotations

import numpy as np

from .inputs import (WindowSources, method_vertices_camera,
                     native_windows_in_gt_world, stitch_window_anchor)
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

# One source window is 60 frames; predicted-camera windows are stitched in order.
WINDOW_FRAMES = 60


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


def _window_world(method: str, window: WindowSources, mano, vertices_camera=None,
                  anchor_c2w=None):
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
            aligned, aligned_c2w = native_windows_in_gt_world(
                own_world, own_c2w, gt_c2w, camera_valid=own_valid,
                anchor_c2w=anchor_c2w)
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
        camera = method_vertices_camera(data, mano)
        c2w = np.asarray(data["camera_c2w"], float)
        predicted_world = world_from_camera(camera, c2w)
        valid = np.asarray(data["hand_valid"], bool)
        aligned, aligned_c2w = native_windows_in_gt_world(
            predicted_world, c2w, gt_c2w, camera_valid=valid.any(axis=1),
            anchor_c2w=anchor_c2w)
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
    aligned, aligned_c2w = native_windows_in_gt_world(
        native_world, native_c2w, gt_c2w, camera_valid=valid.any(axis=1),
        anchor_c2w=anchor_c2w)
    camera = {"c2w": aligned_c2w, "valid": np.isfinite(aligned_c2w).all(axis=(1, 2)),
              "K": gt_K, "K_valid": gt_K_valid, "size_hw": size_hw, "source": "pred",
              "frames": None}
    return aligned, None, valid, camera


def concatenate_cameras(cameras: list, total: int) -> dict:
    """Stack the per-window camera bundles over the clip, one entry per window.

    ``cameras`` is in window order and may contain ``None`` for windows where the
    method has no prediction; those windows are filled with an invalid (NaN) camera
    *in place*, so window *i* keeps the frames of window *i*.  Dropping the missing
    windows instead slid every later rig to the start of the clip: Dyn-HaMR (1 of 5
    windows here) was drawn during frames 0-59 while its hand lives in 120-179.
    """
    present = [camera for camera in cameras if camera is not None]
    if not present:
        raise ValueError("no camera to concatenate")
    first = present[0]
    frames, intrinsics = np.asarray(first["c2w"], float).shape[0], np.asarray(first["K"], float)
    placeholder_K = (intrinsics[None] if intrinsics.ndim == 2 else intrinsics[:1]).astype(float)
    parts = []
    for camera in cameras:
        if camera is not None:
            parts.append(camera)
            continue
        parts.append({"c2w": np.full((frames, 4, 4), np.nan),
                      "valid": np.zeros(frames, bool),
                      "K": np.repeat(placeholder_K, frames, axis=0),
                      "K_valid": np.zeros(frames, bool)})
    c2w = np.concatenate([c["c2w"] for c in parts], axis=0)
    valid = np.concatenate([np.asarray(c["valid"], bool) for c in parts])
    K = np.concatenate([c["K"] for c in parts], axis=0)
    K_valid = np.concatenate([np.asarray(c["K_valid"], bool) for c in parts])
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
    stitch_state = {}
    for window in segment.windows:
        frames = len(window.frame_ids)
        gt = window.methods.get("gt")
        gt_c2w = np.asarray(gt["camera_c2w"], float) if gt is not None else None
        gt_valid = (np.asarray(gt.get("camera_valid", np.ones(frames, bool)), bool)
                    if gt is not None else np.zeros(frames, bool))
        for method in METHODS_3D:
            anchor = None
            if method in PREDICTED_EXTRINSICS and method in stitch_state and gt_valid[0]:
                previous_pred, previous_gt = stitch_state[method]
                if np.isfinite(gt_c2w[0]).all():
                    anchor = stitch_window_anchor(previous_pred, previous_gt, gt_c2w[0])
            vertices, joints, valid, camera = _window_world(
                method, window, mano, anchor_c2w=anchor)
            if method in PREDICTED_EXTRINSICS:
                camera_valid = (np.asarray(camera["valid"], bool) if camera is not None
                                else np.zeros(frames, bool))
                if (camera is not None and camera_valid[-1] and gt_valid[-1]
                        and np.isfinite(gt_c2w[-1]).all()):
                    stitch_state[method] = (np.asarray(camera["c2w"][-1], float), gt_c2w[-1])
                else:
                    stitch_state.pop(method, None)
            # One slot per window, ``None`` when this window has no prediction, so the
            # stacked rig keeps the segment's time axis.
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
            assert len(store["camera"]) == len(segment.windows), (
                f"{method}: {len(store['camera'])} camera bundles for "
                f"{len(segment.windows)} windows - the time axis would drift")
            if any(camera is not None for camera in store["camera"]):
                entry["camera"] = concatenate_cameras(store["camera"], len(valid_frames))
        result[method] = entry
    return result


def level_in_place(store: dict, rotation: np.ndarray) -> None:
    """Move every method's hands **and** camera rig into the renderer's y-up world.

    The dataset world from ``camera_c2w`` has +y pointing down, so the renderer
    levels everything it draws with one rigid transform.  Hands, joints *and* the
    per-method camera bundles have to go through that same transform: rotating only
    the geometry (the state until 2026-09-18) drew every frustum, axis triad and
    trajectory tube ~2.7 m away from its hand, in a world 151 deg from the hands'.
    """
    rotation = np.asarray(rotation, float)
    rotation4 = np.eye(4)
    rotation4[:3, :3] = rotation
    for entry in store.values():
        if "vertices" in entry:
            entry["vertices"] = np.einsum("ij,tsvj->tsvi", rotation, entry["vertices"])
        if "joints" in entry:
            entry["joints"] = np.einsum("ij,tsvj->tsvi", rotation, entry["joints"])
        bundle = entry.get("camera")
        if bundle is not None:
            bundle["c2w"] = np.einsum("ij,tjk->tik", rotation4, bundle["c2w"])


def _hand_centroids(entry: dict, total: int) -> np.ndarray:
    """Mean hand position per frame, ``(T, 3)`` with NaN where the method is empty."""
    points = entry.get("vertices")
    if points is None:
        points = entry.get("joints")
    out = np.full((total, 3), np.nan)
    if points is None:
        return out
    points = np.asarray(points, float)
    valid = np.asarray(entry["valid"], bool)
    if valid.ndim == 3:
        valid = valid.any(axis=-1)
    finite = np.isfinite(points).all(-1) & valid[:, :, None]
    for frame in range(min(total, points.shape[0])):
        mask = finite[frame]
        if mask.any():
            out[frame] = points[frame][mask].mean(0)
    return out


def _cone_angle_deg(K, size_hw) -> float:
    """Half-angle of the image diagonal, i.e. the widest direction the camera sees."""
    height, width = (int(v) for v in size_hw)
    fx, fy = float(np.asarray(K, float)[0, 0]), float(np.asarray(K, float)[1, 1])
    if not np.isfinite(fx) or not np.isfinite(fy) or fx <= 0 or fy <= 0:
        return float("nan")
    return float(np.degrees(np.arctan(np.hypot(0.5 * (width - 1) / fx,
                                               0.5 * (height - 1) / fy))))


def camera_bundle_report(store: dict, camera, *, window_frames: int = WINDOW_FRAMES,
                         anchor_tol: float = 1e-3, distance_band=(0.02, 5.0),
                         hard_angle_deg: float = 90.0) -> dict:
    """Audit that hands, frustums and trajectories live in one world frame.

    Three independent, dataset-independent checks:

    * ``anchor`` - the first predicted window (and any window after a missing one)
      starts on the calibrated camera; contiguous windows must instead advance the
      previous predicted pose by the calibrated camera's boundary motion.  This
      catches both coordinate-frame mistakes and 60-frame camera resets.
    * ``up`` - the calibrated display camera is levelled, i.e. its up axis is +y.
    * ``hand`` - each method's hand sits in front of that method's own camera, at a
      plausible capture distance and (as a warning) inside its projected image cone.
    """
    calib = np.asarray(camera.camera_to_display, float)
    calib_valid = np.asarray(camera.camera_valid, bool) & np.isfinite(calib).all(axis=(1, 2))
    total = len(calib)
    rows, violations, warnings = [], [], []
    up = -np.mean(calib[calib_valid][:, :3, 1], axis=0) if calib_valid.any() else np.zeros(3)
    up_error = float(np.linalg.norm(up - np.array([0.0, 1.0, 0.0])))
    if not calib_valid.any():
        warnings.append("calibrated camera has no valid frame")
    elif up_error > anchor_tol:
        violations.append(f"calibrated camera is not levelled: |up - +y| = {up_error:.4f}")
    starts = list(range(0, total, window_frames))
    for method in METHODS_3D:
        entry = store.get(method)
        if entry is None or entry.get("camera") is None:
            continue
        bundle = entry["camera"]
        c2w = np.asarray(bundle["c2w"], float)
        valid = np.asarray(bundle["valid"], bool) & np.isfinite(c2w).all(axis=(1, 2))
        centres, axes = c2w[:, :3, 3], c2w[:, :3, 2]
        hands = _hand_centroids(entry, total)
        both = np.flatnonzero(valid & np.isfinite(hands).all(axis=1))
        distances, angles = [], []
        for frame in both:
            delta = hands[frame] - centres[frame]
            norm = float(np.linalg.norm(delta))
            distances.append(norm)
            if norm > 1e-9:
                cos = float(np.clip(axes[frame] @ delta / norm, -1.0, 1.0))
                angles.append(float(np.degrees(np.arccos(cos))))
        anchor, anchor_rot = [], []
        for start in starts:
            if not (valid[start] and calib_valid[start]):
                continue
            if start == 0 or not (valid[start - 1] and calib_valid[start - 1]):
                expected = calib[start]
            else:
                expected = c2w[start - 1] @ np.linalg.inv(calib[start - 1]) @ calib[start]
            anchor.append(float(np.linalg.norm(c2w[start, :3, 3] - expected[:3, 3])))
            relative = expected[:3, :3].T @ c2w[start, :3, :3]
            anchor_rot.append(float(np.degrees(np.arccos(np.clip(
                (np.trace(relative) - 1) / 2, -1.0, 1.0)))))
        cone = _cone_angle_deg(bundle["K"][both[0]] if len(both) else bundle["K"][0],
                               bundle["size_hw"])
        row = {"method": method, "source": bundle["source"],
               "camera_frames": int(valid.sum()), "paired_frames": int(len(both)),
               "distance_median": float(np.median(distances)) if distances else float("nan"),
               "distance_max": float(np.max(distances)) if distances else float("nan"),
               "angle_max": float(np.max(angles)) if angles else float("nan"),
               "cone_angle": cone,
               "anchor_max": float(np.max(anchor)) if anchor else float("nan"),
               "anchor_rot_max": float(np.max(anchor_rot)) if anchor_rot else float("nan")}
        rows.append(row)
        label = METHOD_LABELS_3D[method]
        if anchor and max(anchor) > anchor_tol:
            violations.append(f"{label}: camera rig violates the window-stitch contract by "
                              f"{max(anchor):.4f} m")
        if not len(both):
            warnings.append(f"{label}: no frame has both a hand and a camera")
        if distances and (min(distances) < distance_band[0] or max(distances) > distance_band[1]):
            violations.append(f"{label}: camera-to-hand distance {min(distances):.3f}-"
                              f"{max(distances):.3f} m is outside {distance_band}")
        if angles and max(angles) > hard_angle_deg:
            violations.append(f"{label}: frustum axis misses the hand by {max(angles):.1f} deg "
                              f"(>= {hard_angle_deg:.0f})")
        elif angles and np.isfinite(cone) and max(angles) > cone:
            warnings.append(f"{label}: hand is outside its own image cone "
                            f"({max(angles):.1f} deg > {cone:.1f} deg)")
    return {"up_error": up_error, "rows": rows, "violations": violations, "warnings": warnings}


def assert_camera_bundle_frame(store: dict, camera, **kwargs) -> dict:
    """Raise when the drawn rig does not share the hands' world frame."""
    report = camera_bundle_report(store, camera, **kwargs)
    if report["violations"]:
        raise RuntimeError("3D world-frame contract violated (hands / frustums / trajectory): "
                           + "; ".join(report["violations"]))
    return report
