"""World-space multi-view summary for the 8-method 3D comparison.

Renders the EgoFound3R tag `final-no-hand-completion-20260908` multi-view style
(the same `HandMultiviewRenderer` / `compose_grid` / `mesh_parts` code path):
white background, fov 19, Morandi side colours with a temporal gradient, gray
ground platform with a fake shadow, camera frustums and trajectory, and the
"rows = methods, columns = views" matrix the reference uses.

Runs in the EgoFound3R visualisation environment (torch + pytorch3d + t3drender),
*not* the numba render environment, so it loads the segment sources directly
instead of importing `paper_viz.cli`.

Example
-------
``/mnt/workspace/sjc/envs/egofound3r/bin/python tools/render_3d_summary.py \
    --inputs-dir <staged segment> --src-dir <relay dir> ... \
    --out <out dir> --views front top side --keyframes 6 --cell 320``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from paper_viz.inputs import load_window                      # noqa: E402
from paper_viz.mano import ManoAsset                          # noqa: E402
from paper_viz.sequences3d import (METHODS_3D, METHOD_LABELS_3D,  # noqa: E402
                                   SKELETON_METHODS, assert_camera_bundle_frame,
                                   build_segment_sequences, level_in_place)

REFERENCE_ROOT = Path("/mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917")
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from egohandmetric_prompt.inference_multiview import (        # noqa: E402
    CAMERA_COLORS, HandMultiviewRenderer, SIDE_DARK, SIDE_LIGHT, VIEW_ANGLES,
    _box, _tube_part, camera_overlay_parts, compose_grid, label_cell, mesh_parts,
    scene_bounds, scene_framing_points)

# Five viewpoints of the *same* world scene: the hands and the capture-camera
# trajectory never move, only the view camera does.  Azimuths are multiples of 90
# degrees so the ground plate's edges stay parallel to the image axes (no roll, no
# diagonal plate), and the overhead view sits at 85 degrees.
# Four oblique views, one from above each edge of the ground plane, plus the
# overhead view (which is rotated 90 deg counter-clockwise after rendering).
VIEWPOINTS = {
    "front": (0.0, 40.0),
    "right": (90.0, 40.0),
    "back": (180.0, 40.0),
    "left": (-90.0, 40.0),
    "top": (0.0, 85.0),
}
TOP_VIEW_ROT90_CCW = True
for _name, _angles in VIEWPOINTS.items():
    VIEW_ANGLES[_name] = _angles
EGOGRASP_VIEWS = tuple(VIEWPOINTS)

# 21-joint MANO chain (wrist + 4 joints per finger + 5 tips), same order as GT.
SKELETON_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                  (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
                  (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))


def skeleton_parts(joints, valid, indices, times, *, temporal_colors, radius=0.0035):
    """ReViV4D has no 778 mesh: draw 21 joint markers + 20 bone tubes instead.

    Colours follow the reference side palette with the same temporal gradient the
    meshes use, so the joints-only row reads as part of the same figure.
    """
    parts = []
    first, last = times[indices[0]], times[indices[-1]]
    for frame in indices:
        phase = (times[frame] - first) / max(last - first, 1e-9) if temporal_colors and len(indices) > 1 else .5
        for side in range(2):
            color = SIDE_LIGHT[side] * (1 - phase) + SIDE_DARK[side] * phase
            xyz = joints[frame, side]
            ok = valid[frame, side] if valid.ndim == 3 else np.ones(len(xyz), bool)
            segments = np.stack([xyz[list(edge)] for edge in SKELETON_EDGES], axis=0)
            keep = np.array([ok[list(edge)].all() and np.isfinite(seg).all() for edge, seg in zip(SKELETON_EDGES, segments)])
            if keep.any():
                parts += _tube_part(segments[keep], color, radius)
            for joint, good in zip(xyz, ok):
                if good and np.isfinite(joint).all():
                    parts.append(_box(joint, np.full(3, radius * 2.2), color))
    return parts





def _rot_x(deg):
    a = np.deg2rad(deg); c, sn = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])


def _rot_y(deg):
    a = np.deg2rad(deg); c, sn = np.cos(a), np.sin(a)
    return np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])


def _rot_z(deg):
    a = np.deg2rad(deg); c, sn = np.cos(a), np.sin(a)
    return np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])


# EgoGrasp's quickview stack: rotate the *scene* and keep one fixed camera, so in
# every non-overhead view the ground plate stays parallel to the image axes
# (an orbit camera instead rolls the plate diagonally).
EGOGRASP_STACK = {
    "front": np.eye(3),
    "side": _rot_y(90.0),
    "top": _rot_x(-65.0),
    "bottom": _rot_z(180.0) @ _rot_x(65.0),
    "left": _rot_x(-65.0) @ _rot_z(90.0),
    "right": _rot_z(180.0) @ _rot_x(65.0) @ _rot_z(90.0),
}
FLAT_VIEW_PITCH = 0.0     # fixed camera used with the scene rotations


def turntable_pose(center, points, yaw_deg, pitch_deg, *, fov_deg=19.0, margin=1.10):
    """View camera at a fixed direction, distance solved from the projected extent.

    Fitting a bounding *sphere* made every view too far away (and the back view
    nearly empty).  Here the framing cloud (hand trajectory + plate corners) is
    projected onto the view plane and the distance is solved so its extent fills
    the cell with a small margin.  ``up`` is always derived from world +y, so the
    horizon stays level and the plate keeps its edges parallel to the image axes.
    """
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    direction = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    forward = -direction
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    rel = np.asarray(points, float) - np.asarray(center, float)
    half = max(float(np.abs(rel @ right).max()), float(np.abs(rel @ up).max()))
    depth = max(float((rel @ forward).max()), 0.0)
    distance = margin * half / np.tan(np.deg2rad(fov_deg) * 0.5) + depth
    position = np.asarray(center, float) + direction * distance
    rotation = np.stack([right, -up, forward])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ position
    return pose


def label_font(size):
    """TTF at a readable size (PIL's default bitmap font is far too small)."""
    from PIL import ImageFont
    candidates = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  str(PACKAGE_ROOT / "paper_viz" / "assets" / "fonts" / "PatrickHand-Regular.ttf"))
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_matrix(cells, methods, views, *, title, cell_px, temporal=True, camera_legend=True):
    """Rows = views, columns = methods (the reference grid wraps 3+2 instead)."""
    from PIL import Image, ImageDraw
    label_size = max(22, int(round(cell_px * 0.115)))
    body_size = max(16, int(round(cell_px * 0.075)))
    head_font, row_font = label_font(label_size), label_font(label_size)
    small_font = label_font(body_size)
    header_h = label_size + 18
    label_w = int(round(label_size * 5.2))
    pad, legend_h = 3, int(body_size * 2.6) + 30
    width = label_w + pad + len(methods) * (cell_px + pad)
    height = header_h + len(views) * (cell_px + pad) + legend_h + 26
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for column, method in enumerate(methods):
        x = label_w + pad + column * (cell_px + pad)
        # centre the column title over its cell
        text_w = draw.textlength(method, font=head_font)
        draw.text((x + max((cell_px - text_w) / 2, 0), header_h - label_size - 12), method,
                  fill=(20, 22, 26), font=head_font)
    for row, view in enumerate(views):
        y = header_h + row * (cell_px + pad)
        # centre the row label in the label band, vertically aligned with the cell
        text_w = draw.textlength(view, font=row_font)
        draw.text((max((label_w - text_w) / 2, 0), y + cell_px // 2 - label_size // 2), view,
                  fill=(20, 22, 26), font=row_font)
        draw.line((4, y, 4, y + cell_px), fill=(214, 216, 220))
        for column, method in enumerate(methods):
            x = label_w + pad + column * (cell_px + pad)
            canvas.paste(Image.fromarray(cells[method, view]), (x, y))
    # ---- legend strip: grouped, evenly spaced, with a hairline separator ------
    y = header_h + len(views) * (cell_px + pad) + 10
    draw.line((pad, y - 6, width - pad, y - 6), fill=(226, 228, 232))
    bar_h, chip = max(10, body_size // 2), max(10, body_size // 2)
    baseline = y + 4
    group_font = label_font(max(15, int(round(body_size * 0.92))))
    x = pad + 4

    def gradient_bar(x0, y0, w, h, stops):
        for i in range(w):
            f = i / max(w - 1, 1)
            color = tuple(int(round(255 * (stops[0][k] + f * (stops[1][k] - stops[0][k]))))
                          for k in range(3))
            draw.rectangle((x0 + i, y0, x0 + i, y0 + h), fill=color)

    def group_title(text):
        nonlocal x
        draw.text((x, baseline), text, fill=(120, 124, 130), font=group_font)
        x += draw.textlength(text, font=group_font) + 8

    def colour_bar(colour, label):
        nonlocal x
        draw.text((x, baseline), label, fill=(60, 62, 66), font=small_font)
        x += draw.textlength(label, font=small_font) + 5
        gradient_bar(x, baseline + 2, int(round(cell_px * 0.34)), bar_h, (SIDE_LIGHT[colour], SIDE_DARK[colour]))
        draw.rectangle((x - 1, baseline + 1, x + int(round(cell_px * 0.34)), baseline + bar_h + 2),
                       outline=(206, 208, 214))
        x += int(round(cell_px * 0.34)) + 16

    group_title("Hand colour")
    colour_bar(0, "Left")
    colour_bar(1, "Right")
    x += 10
    group_title("Time")
    draw.text((x, baseline), "Earlier", fill=(60, 62, 66), font=small_font)
    x += draw.textlength("Earlier", font=small_font) + 5
    gradient_bar(x, baseline + 2, int(round(cell_px * 0.34)), bar_h, (SIDE_LIGHT[1], SIDE_DARK[0]))
    draw.rectangle((x - 1, baseline + 1, x + int(round(cell_px * 0.34)), baseline + bar_h + 2),
                   outline=(206, 208, 214))
    x += int(round(cell_px * 0.34)) + 5
    draw.text((x, baseline), "Later", fill=(60, 62, 66), font=small_font)
    x += draw.textlength("Later", font=small_font) + 22
    if camera_legend:
        group_title("Camera")
        for name, colour in CAMERA_COLORS.items():
            rgb = tuple(int(round(255 * value)) for value in colour)
            draw.rectangle((x, baseline + 1, x + chip, baseline + 1 + chip), fill=rgb,
                           outline=(150, 152, 158))
            x += chip + 5
            label = f"{name}"
            draw.text((x, baseline), label, fill=(60, 62, 66), font=small_font)
            x += draw.textlength(label, font=small_font) + 16
    draw.line((pad, y + bar_h + 16, width - pad, y + bar_h + 16), fill=(226, 228, 232))
    draw.text((pad + 4, y + bar_h + 24), title, fill=(128, 132, 138), font=small_font)
    return np.asarray(canvas, dtype=np.uint8)

def level_rotation(c2w) -> np.ndarray:
    """Rotation that puts the calibrated camera upright in the renderer's y-up world.

    The dataset world from ``camera_c2w`` has +y pointing *down* in the image
    (OpenCV), while the reference renderer assumes +y is up, which is why the
    camera frustum used to hang upside down with its top near the table.  This is
    a pure rotation applied to every method and to the camera, so nothing is
    mirrored and the comparison stays valid.
    """
    up = -np.asarray(c2w, float)[:, :3, 1].mean(0)
    up = up / max(float(np.linalg.norm(up)), 1e-9)
    target = np.array([0.0, 1.0, 0.0])
    axis = np.cross(up, target)
    sin_angle = float(np.linalg.norm(axis))
    cos_angle = float(up @ target)
    if sin_angle < 1e-8:
        return np.eye(3) if cos_angle > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / sin_angle
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sin_angle * skew + (1 - cos_angle) * skew @ skew

def build_registry(inputs_dir: Path, args) -> dict:
    selection = json.loads((inputs_dir / "selection.json").read_text())
    hawor_index = {}
    for line in Path(args.hawor_index).read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            hawor_index[entry["window_id"]] = entry["prediction_dir"]
    registry = {"dataset": selection["dataset"], "sequence_id": selection["sequence_id"],
                "segment_id": selection["segment_id"], "windows": []}
    for index, window in enumerate(selection["windows"]):
        cache = window["gt"]["cache_id"]
        src = Path(args.src_dir) / cache
        hawor_dir = Path(hawor_index.get(window["window_id"], Path(args.hawor_root) / cache))
        methods = {
            "gt": src / "gt.npz",
            "ego": inputs_dir / f"{index}_ego.npz",
            "ego_full": src / "ego_full.npz",
            "wilor": src / "wilor.npz",
            "hawor": hawor_dir / "predictions.npz",
            "reviv4d": src / "reviv4d.npz",
            "pad_hand": src / "pad_hand.npz",
            "egoforce": src / "egoforce.npz",
            "dyn_hamr": src / "dyn_hamr.npz",
        }
        baselines = {"ego_contact": Path(args.contact_root) /
                     "ego_vertex_contact_distance_8095_retry3_20260911/egofound3r_stride5/shards/"
                     f"{selection['dataset']}/{selection['dataset']}/{cache}.npz"}
        registry["windows"].append({"cache_id": cache, "window_id": window["window_id"],
                                    "methods": {k: str(v) for k, v in methods.items()},
                                    "baselines": {k: str(v) for k, v in baselines.items()}})
    return registry


def load_segment(args):
    mano = ManoAsset(Path(args.mapping))
    registry = build_registry(Path(args.inputs_dir), args)
    windows = [load_window(entry["cache_id"], entry["window_id"], Path(args.prepared_root),
                           entry["methods"], entry["baselines"], mano)
               for entry in registry["windows"]]
    return registry, windows, mano


def camera_sequence(windows, mano, name="Camera (calibrated)"):
    """GT world camera trajectory + per-frame intrinsics, as a HandSequence-like object."""
    from egohandmetric_prompt.inference_multiview import HandSequence
    c2w = np.concatenate([np.asarray(w.methods["gt"]["camera_c2w"], float) for w in windows])
    valid = np.concatenate([np.asarray(w.methods["gt"]["camera_valid"], bool) for w in windows])
    intrinsics = np.concatenate([np.asarray(w.record["intrinsics"], float) for w in windows])
    size = tuple(int(v) for v in windows[0].record.get("image_size_hw", (2000, 2800)))
    vertex_count = int(mano.faces.max()) + 1
    empty = np.zeros((len(c2w), 2, vertex_count, 3))
    sequence = HandSequence(name=name, vertices=empty,
                            valid=np.zeros((len(c2w), 2, vertex_count), bool),
                            faces=mano.faces, camera_source="gt",
                            camera_to_display=c2w, camera_valid=valid, camera_K=intrinsics,
                            camera_K_valid=valid.copy(), image_size_hw=size)
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--hawor-index", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", nargs="+",
                        default=list(EGOGRASP_VIEWS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS_3D))
    parser.add_argument("--keyframes", type=int, default=5)
    parser.add_argument("--rgb-column", dest="rgb_column", action="store_true", default=True,
                        help="leftmost column: the key-frame RGB, top to bottom in time")
    parser.add_argument("--no-rgb-column", dest="rgb_column", action="store_false")
    parser.add_argument("--cell", type=int, default=320)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--camera-trajectory", choices=("none", "full", "accumulate"),
                        default="full")
    parser.add_argument("--camera-scale", type=float, default=0.12)
    parser.add_argument("--fit-margin", type=float, default=None,
                        help="override the zoom; defaults are 0.82 with the camera rig and "
                             "0.58 without it (smaller = closer)")
    parser.add_argument("--camera-overlay", choices=("show", "hide"), default="show",
                        help="draw the capture-camera trajectory/frustums; hiding also zooms in")
    parser.add_argument("--view-mode", choices=("orbit", "scene"), default="orbit",
                        help="scene = EgoGrasp rotation stack (plate stays axis aligned)")
    parser.add_argument("--no-panels", action="store_true")
    args = parser.parse_args()

    registry, windows, mano = load_segment(args)
    store = build_segment_sequences(type("S", (), {"windows": windows})(), mano)
    missing = [m for m in args.methods if m not in store]
    present = [m for m in args.methods if m in store]
    print(f"segment {registry['segment_id']}: {len(present)} methods present, missing {missing}")

    from egohandmetric_prompt.inference_multiview import HandSequence
    frames = len(windows) * 60
    times = np.arange(frames, dtype=float) / 30.0
    camera = camera_sequence(windows, mano)
    rotation = level_rotation(camera.camera_to_display)
    rotation4 = np.eye(4)
    rotation4[:3, :3] = rotation
    camera.camera_to_display = np.einsum("ij,tjk->tik", rotation4, camera.camera_to_display)
    level_in_place(store, rotation)
    assert_camera_bundle_frame(store, camera)
    # Every hand row carries the same calibrated camera trajectory: the rows are
    # different hands in one shared GT world, so the frustums/trajectory must match.
    sequences = []
    for method in present:
        entry = store[method]
        if "vertices" not in entry:
            continue
        sequences.append(HandSequence(name=METHOD_LABELS_3D[method], vertices=entry["vertices"],
                                      valid=entry["valid"], faces=entry["faces"],
                                      camera_source="gt",
                                      camera_to_display=camera.camera_to_display,
                                      camera_valid=camera.camera_valid,
                                      camera_K=camera.camera_K,
                                      camera_K_valid=camera.camera_K_valid,
                                      image_size_hw=camera.image_size_hw))
    valid_indices = np.flatnonzero(store["gt"]["valid"].any(axis=-1).any(axis=-1)) if "gt" in store \
        else np.arange(frames)
    # Union policy: every frame where *any* method is valid stays eligible, so the
    # summary covers the whole clip (Dyn-HaMR is simply empty outside its window)
    # instead of collapsing onto the one stretch all methods happen to share.
    frame_ok = np.zeros(frames, bool)
    for method in present:
        entry = store[method]
        flags = entry["valid"]
        frame_ok |= flags.any(axis=-1).any(axis=-1) if flags.ndim == 3 else flags.any(axis=-1)
    candidates = np.flatnonzero(frame_ok)
    picks = np.rint(np.linspace(0, len(candidates) - 1, min(args.keyframes, len(candidates)))).astype(int)
    keyframes = candidates[picks]
    print(f"keyframes (union_valid): {keyframes.tolist()} of {len(candidates)} eligible frames")
    mesh_methods = [m for m in present if "vertices" in store[m]]
    drawn_methods = [m for m in present if m in mesh_methods or "joints" in store[m]]

    # EgoGrasp recipe: the ground plane and the framing follow the *whole hand
    # trajectory*, not a single frame and not the camera path.  The plate is placed
    # a fixed gap below the trajectory AABB and scaled to its footprint, so every
    # row sits on the same stage and the hands stay the subject of the picture.
    trajectory = [store[m]["vertices"][store[m]["valid"]] for m in mesh_methods
                  if "vertices" in store[m] and store[m]["valid"].any()]
    trajectory = [points for points in trajectory if len(points)]
    if trajectory:
        cloud = np.concatenate(trajectory, axis=0)
        low, high = cloud.min(0), cloud.max(0)
        center, size = 0.5 * (low + high), high - low
        gap = max(0.015, 0.08 * float(size[1]))
        bounds = (np.array([center[0] - 0.55 * size[0], low[1] - gap, center[2] - 0.55 * size[2]]),
                  np.array([center[0] + 0.55 * size[0], high[1] + 0.25 * size[1], center[2] + 0.55 * size[2]]))
    else:
        bounds = scene_bounds(sequences, valid_indices)
    framing = scene_framing_points(sequences, valid_indices)
    trajectory_center = 0.5 * (np.asarray(framing, float).min(0) + np.asarray(framing, float).max(0))
    renderer = HandMultiviewRenderer(bounds, cell_size=args.cell, device=args.device, ground=True,
                                     supersample=args.supersample, framing_points=framing)
    if args.view_mode == "orbit":
        # Frame the hands *and* the plate in every viewpoint: a hand-only fit lets
        # the plate fall out of frame, a bounding-sphere fit pushes everything far.
        scene_center = 0.5 * (bounds[0] + bounds[1])
        # Fit the hands (the subject) with a small margin; the plate is allowed to
        # run past the frame edge, otherwise the plate extent shrinks the hands.
        # Fit to a trimmed trajectory box (4th-96th percentile) instead of the raw
        # min/max: a few outlier frames otherwise force the camera far away.
        cloud = np.asarray(framing, float)
        show_camera = args.camera_overlay == "show"
        if show_camera:
            # The rig may be clipped, so a trimmed trajectory box keeps the hands
            # large (4th-96th percentile drops the few outlier frames).
            low_fit = np.percentile(cloud, 4, axis=0)
            high_fit = np.percentile(cloud, 96, axis=0)
            default_margin = 0.82
        else:
            # Hand-only view: same trajectory box as the rig view (so the framing
            # stays comparable between the two modes), just a smaller margin so the
            # hands come closer.  ``--fit-margin`` tunes it further.
            low_fit = np.percentile(cloud, 4, axis=0)
            high_fit = np.percentile(cloud, 96, axis=0)
            default_margin = 0.72
        fit_points = np.array(np.meshgrid(*zip(low_fit, high_fit))).T.reshape(-1, 3)
        margin = args.fit_margin if args.fit_margin else default_margin
        for name, (yaw, pitch) in VIEWPOINTS.items():
            renderer.view_poses[name] = turntable_pose(scene_center, fit_points, yaw, pitch,
                                                       margin=margin)

    cells = {}
    for method in drawn_methods:
        entry = store[method]
        sequence = next((s for s in sequences if s.name == METHOD_LABELS_3D[method]), None)
        if sequence is not None:
            parts = mesh_parts(sequence, keyframes, times, temporal_colors=True)
        elif "joints" in entry:
            parts = skeleton_parts(entry["joints"], entry["valid"], keyframes, times,
                                   temporal_colors=True)
        else:
            continue
        annotations = (camera_overlay_parts(camera, keyframes, show_frustums=True,
                                            path_indices=np.arange(frames),
                                            scale=args.camera_scale)
                       if (args.camera_overlay == "show" and args.camera_trajectory != "none") else [])
        for view in args.views:
            if args.view_mode == "scene":
                rotation = EGOGRASP_STACK.get(view, np.eye(3))
                # Rotate about the hand-trajectory centre, not the world origin: a
                # viewpoint change must keep the hands in frame and swing the
                # capture camera around them (rotating about the origin slid the
                # whole rig across the image instead).
                origin = trajectory_center
                def spin(vertices):
                    return np.einsum("ij,vj->vi", rotation, vertices - origin) + origin
                rotated = [(spin(v), f, c) for v, f, c in parts]
                rotated_annot = [(spin(v), f, c) for v, f, c in annotations]
                cloud = np.concatenate([v.reshape(-1, 3) for v, _, _ in rotated if len(v)])
                low, high = cloud.min(0), cloud.max(0)
                center, size = 0.5 * (low + high), high - low
                gap = max(0.015, 0.08 * float(size[1]))
                view_bounds = (np.array([center[0] - 0.75 * size[0], low[1] - gap, center[2] - 0.75 * size[2]]),
                               np.array([center[0] + 0.75 * size[0], high[1] + 0.35 * size[1], center[2] + 0.75 * size[2]]))
                view_renderer = HandMultiviewRenderer(view_bounds, cell_size=args.cell,
                                                      device=args.device, ground=True,
                                                      supersample=args.supersample,
                                                      framing_points=cloud)
                rgb = view_renderer.render(rotated + rotated_annot, "front", shadow_parts=rotated)
            else:
                rgb = renderer.render(parts + annotations, view, shadow_parts=parts)
            if view == "top" and TOP_VIEW_ROT90_CCW:
                rgb = np.ascontiguousarray(np.rot90(rgb, k=1))   # counter-clockwise
            # No per-cell caption: the matrix already carries the method (column
            # header) and the view (row label).
            cells[(METHOD_LABELS_3D[method], view)] = rgb
            if not args.no_panels:
                target = args.out / "summary_views" / method / f"{view}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                from PIL import Image
                Image.fromarray(rgb).save(target)
    columns = (["Input RGB"] if args.rgb_column else []) + [METHOD_LABELS_3D[m] for m in drawn_methods]
    cells = {(METHOD_LABELS_3D[m], view): cells[(METHOD_LABELS_3D[m], view)]
             for m in drawn_methods for view in args.views}
    camera_note = "with camera rig" if args.camera_overlay == "show" else "hand-only (camera hidden)"
    if args.rgb_column:
        # Leftmost column: the key-frame RGB of the clip, ordered in time so the
        # reader can line each row up with what the hands actually did.
        from PIL import Image
        from paper_viz.inputs import rgb_path
        for row, t in enumerate(keyframes):
            w_index, f_index = divmod(int(t), 60)
            picture = Image.open(rgb_path(windows[w_index], f_index))
            if picture.mode != "RGB":
                picture = picture.convert("RGB")
            scale = args.cell / picture.width
            picture = picture.resize((args.cell, max(1, int(round(picture.height * scale))),
                                      ), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (args.cell, args.cell), (255, 255, 255))
            tile.paste(picture, (0, max((args.cell - picture.height) // 2, 0)))
            cells[("Input RGB", args.views[row % len(args.views)])] = np.asarray(tile)
    grid = compose_matrix(cells, columns, list(args.views),
                          title=f"{registry['segment_id']} | {len(keyframes)} time samples | "
                                "world space | rows = views, columns = methods | "
                                "EgoFound3R = 8fc061a infer (default post-processing)",
                          cell_px=args.cell)
    args.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(grid).save(args.out / "fig1_3d_summary.png")
    print("wrote", args.out / "fig1_3d_summary.png", grid.shape)


if __name__ == "__main__":
    main()
