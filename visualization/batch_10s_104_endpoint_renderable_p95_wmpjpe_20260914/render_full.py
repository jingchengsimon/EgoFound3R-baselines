"""Render one endpoint-renderable 10s segment as a six-view PNG and MP4."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
RUNTIME_SOURCE = ROOT / "runtime_source"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_module("legacy_render_10s", RUNTIME_SOURCE / "batch_10s_41_p95_wmpjpe_20260910/render_batch.py")
video_io = load_module("local_video_io", ROOT / "video_io.py")
completion = load_module("ego_bracketed_fill", ROOT / "ego_bracketed_fill.py")
base = legacy.base
METHODS = ("ego", "wilor", "hawor", "reviv4d", "pad_hand", "gt")
LABELS = ("Ego stride5", "WiLoR", "HaWoR native camera\n(unmasked SLAM)",
          "ReViV4D joints", "PAD-Hand 778", "GT")
VIEWS = legacy.VIEWS
EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20))


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def concatenate(inputs, method):
    return {key: np.concatenate([arrays(inputs / f"{index}_{method}.npz")[key] for index in range(5)])
            for key in arrays(inputs / f"0_{method}.npz")}


def camera_points(data, field):
    return legacy.old.camera_points(data, field).astype(float)


def transform_points(transform, values):
    return values @ transform[:3, :3].T + transform[:3, 3]


def native_windows_in_gt_world(native_world, native_c2w, gt_c2w):
    """Rigidly place each native-SLAM window in the common GT world at its first camera."""
    world = np.empty_like(native_world, dtype=float)
    cameras = np.empty_like(native_c2w, dtype=float)
    transforms = []
    for start in range(0, 300, 60):
        stop = start + 60
        transform = gt_c2w[start] @ np.linalg.inv(native_c2w[start])
        world[start:stop] = transform_points(transform, native_world[start:stop])
        cameras[start:stop] = np.einsum("ij,tjk->tik", transform, native_c2w[start:stop])
        transforms.append(transform)
    return world, cameras, transforms


def camera_lines(c2w, intrinsics, height, width, selected, display):
    result = []
    edges = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    for frame in selected:
        rays = np.array([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1],
                         [0, height - 1, 1]]) @ np.linalg.inv(intrinsics[frame]).T
        corners = display(np.vstack([np.zeros(3), rays * .032]) @ c2w[frame, :3, :3].T
                          + c2w[frame, :3, 3])
        result.extend([[corners[a], corners[b]] for a, b in edges])
    centers = display(c2w[:, :3, 3])
    result.extend([[a, b] for a, b in zip(centers[:-1], centers[1:])])
    return result


def prepare(row, workspace):
    inputs = Path(workspace) / row["dataset"] / "segments" / row["segment_id"] / "inputs"
    selected = json.loads((inputs / "selection.json").read_text())
    if selected["segment_id"] != row["segment_id"] or selected["frame_ids"] != row["frame_ids"]:
        raise ValueError("staged selection identity mismatch")
    xyz, faces, _, _, cameras, floor, _, info = legacy.prepare(row["dataset"], inputs, emit_stitched=False)
    gt = concatenate(inputs, "gt")
    c2w = gt["camera_c2w"].astype(float)
    if c2w.shape != (300, 4, 4) or not np.isfinite(c2w).all():
        raise ValueError("GT camera trajectory is incomplete")
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]) @ c2w[0, :3, :3].T
    origin = c2w[0, :3, 3]
    display = lambda values: (values - origin) @ rotation.T
    world = lambda values: np.einsum("tij,tsvj->tsvi", c2w[:, :3, :3], values) + c2w[:, None, None, :3, 3]

    valid = {}
    gt_joints = gt["hand_joints_camera"].astype(float)
    gt_vertices = gt["hand_vertices_camera"].astype(float)
    valid["gt"] = (gt["hand_valid"].astype(bool)
                   & np.isfinite(gt_joints).all(axis=(2, 3))
                   & np.isfinite(gt_vertices).all(axis=(2, 3)))
    xyz["gt"] = display(world(gt_vertices))

    ego = concatenate(inputs, "ego")
    ego_joints, ego_markers, valid["ego"], fill_info = completion.fill_camera_space(
        ego["hand_joints_camera"], ego["hand_markers_camera"], ego["hand_valid"])
    with np.load(RUNTIME_SOURCE / "six_same_mask_v7_p95_bestego_20260909/mano_195_to_778.npz") as asset:
        neighbors = asset["neighbor_indices"]
        weights = asset["geometry_weights"]
        anchors = asset["source_vertex_ids"]
    center = ego_markers.mean(axis=-2, keepdims=True)
    ego_vertices = center + ((ego_markers - center)[..., neighbors, :] * weights[..., None]).sum(axis=-2)
    ego_vertices[..., anchors, :] = ego_markers
    xyz["ego"] = display(world(ego_vertices))

    for method in ("wilor",):
        data = concatenate(inputs, method)
        joints = camera_points(data, "joints")
        vertices = camera_points(data, "vertices")
        valid[method] = (data["hand_valid"].astype(bool)
                         & np.isfinite(joints).all(axis=(2, 3))
                         & np.isfinite(vertices).all(axis=(2, 3)))
        if "camera_valid" in data:
            valid[method] &= data["camera_valid"].astype(bool)[:, None]
        xyz[method] = display(world(vertices))

    hawor = concatenate(inputs, "hawor")
    native_world = hawor["hand_vertices_world"].astype(float)
    native_joints = hawor["hand_joints_world"].astype(float)
    native_c2w = hawor["camera_c2w"].astype(float)
    hawor_world, hawor_c2w, hawor_transforms = native_windows_in_gt_world(
        native_world, native_c2w, c2w)
    hawor_joints, _, _ = native_windows_in_gt_world(native_joints, native_c2w, c2w)
    valid["hawor"] = (hawor["hand_valid"].astype(bool)
                      & np.isfinite(hawor_joints).all(axis=(2, 3))
                      & np.isfinite(hawor_world).all(axis=(2, 3))
                      & hawor["camera_valid"].astype(bool)[:, None])
    xyz["hawor"] = display(hawor_world)

    for method in ("pad_hand", "reviv4d"):
        parts = []
        geometry_key = None
        for index, window in enumerate(row["windows"]):
            metadata = json.loads((inputs / f"{index}_{method}_metadata.json").read_text())
            if (metadata["dataset"] != row["dataset"] or metadata["method"] != method
                    or metadata["frame_ids"] != window["gt"]["frame_ids"]):
                raise ValueError(f"{method} metadata identity mismatch")
            path = inputs / f"{index}_{method}.npz"
            data = arrays(path)
            if method == "pad_hand" and "hand_vertices_camera" not in data:
                mesh_path = inputs / f"{index}_pad_hand_mesh.npz"
                if not mesh_path.is_file():
                    raise ValueError("PAD-Hand 778 vertices missing")
                mesh = arrays(mesh_path)
                data["hand_vertices_camera"] = mesh["hand_vertices_camera"]
            key = "hand_vertices_camera" if method == "pad_hand" else "hand_joints_camera"
            geometry_key = geometry_key or key
            if geometry_key != key:
                raise ValueError(f"mixed {method} geometry granularity")
            parts.append(data)
        joints = np.concatenate([part["hand_joints_camera"] for part in parts]).astype(float)
        points = np.concatenate([part[geometry_key] for part in parts]).astype(float)
        native_valid = np.concatenate([part["hand_valid"] for part in parts]).astype(bool)
        expected_points = 778 if method == "pad_hand" else 21
        if points.shape != (300, 2, expected_points, 3):
            raise ValueError(f"{method} geometry shape mismatch: {points.shape}")
        valid[method] = (native_valid
                         & np.isfinite(joints).all(axis=(2, 3))
                         & np.isfinite(points).all(axis=(2, 3)))
        xyz[method] = display(world(points))

    shown = np.rint(np.linspace(0, 299, 5)).astype(int)
    ego_metadata = json.loads((inputs / "0_ego_metadata.json").read_text())
    height, width = ego_metadata["source_resolution_hw"]
    cameras = {method: camera_lines(c2w, gt["intrinsics"], height, width, shown, display)
               for method in METHODS}
    cameras["hawor"] = camera_lines(hawor_c2w, gt["intrinsics"], height, width, shown, display)
    sampled = [xyz[method][valid[method]][:, ::max(1, xyz[method].shape[2] // 25), :].reshape(-1, 3)
               for method in METHODS if valid[method].any()]
    camera_points_all = [np.asarray(lines).reshape(-1, 3) for lines in cameras.values()]
    points = np.concatenate([*sampled, *camera_points_all, np.asarray(floor).reshape(-1, 3)])
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    span = max(high - low) * .53
    bounds = (center - span, center + span)
    info.update(
        methods=list(METHODS), shared_video_bounds=[bounds[0].tolist(), bounds[1].tolist()],
        rgb_indices=[0, 149, 299], rgb_source_frame_ids=[row["frame_ids"][t] for t in (0, 149, 299)],
        video_frames=300, video_fps=30, visualization_mask="none",
        valid_left_right={method: valid[method].sum(0).astype(int).tolist() for method in METHODS},
        ego_interpolation={key: value for key, value in fill_info.items() if key != "filled_mask"},
        ego_filled_mask=fill_info["filled_mask"],
        hawor_label="HaWoR native camera (unmasked SLAM)",
        hawor_camera=("native camera_c2w and hand world arrays; each 60-frame native-SLAM world is "
                      "rigidly placed in the common 10s GT world by matching its first camera pose; "
                      "no per-frame GT-camera substitution and no identity fallback"),
        hawor_window_to_gt_transforms=[value.tolist() for value in hawor_transforms],
    )
    return xyz, faces, shown, valid, cameras, floor, bounds, info, inputs


def joints_cell(xyz, selected, valid, cameras, floor, bounds, elev, azim, pixel_size=1200):
    fig = Figure(figsize=(pixel_size / 150, pixel_size / 150), dpi=150, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", proj_type="ortho", computed_zorder=False)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    low, high = bounds
    center = (low + high) / 2
    span = max(high - low) * .54
    for axis, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        axis(value - span, value + span)
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
    for scale, z, color in ([(1.10, -.012, [.87, .88, .90]), (1, 0, [.94, .945, .95])] if elev >= 0 else []):
        stage = floor.copy()
        middle = stage.mean(0)
        stage[:, :2] = middle[:2] + (stage[:, :2] - middle[:2]) * scale
        stage[:, 2] += z
        ax.add_collection3d(Poly3DCollection([stage], facecolors=[color], edgecolors="none", zorder=0 if z else 1))
    ax.add_collection3d(Line3DCollection(cameras, colors=[(.72, .48, .14, .6)], linewidths=.55, zorder=2))
    for frame in selected:
        for side in range(2):
            if not valid[frame, side]:
                continue
            joints = xyz[frame, side]
            color = tuple(base.LIGHT[side] * (1 - frame / 299) + base.DARK[side] * (frame / 299))
            ax.add_collection3d(Line3DCollection([[joints[a], joints[b]] for a, b in EDGES], colors=[color], linewidths=max(.8, 2.1 * pixel_size / 1200), zorder=3))
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=[color], s=max(.5, 5 * (pixel_size / 1200) ** 2), depthshade=False, zorder=4)
    canvas.draw()
    image = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
    fig.clear()
    return image


def cell(xyz, method, faces, frames, valid, cameras, floor, bounds, elev, azim, pixel_size=1200):
    if xyz[method].shape[2] == 21:
        return joints_cell(xyz[method], frames, valid[method], cameras, floor, bounds, elev, azim, pixel_size)
    return base.render_cell(xyz[method], faces, frames, valid[method], cameras, floor, bounds, elev, azim, pixel_size)


def hand_crop(image, padding_fraction=.14):
    values = np.asarray(image)
    red, green, blue = (values[..., index].astype(int) for index in range(3))
    colored = (blue > green + 5) & ((red > green + 5) | (blue > red + 5))
    yy, xx = np.where(colored)
    if len(xx) == 0:
        return (0, 0, image.width, image.height)
    width = max(1, int(xx.max() - xx.min() + 1))
    height = max(1, int(yy.max() - yy.min() + 1))
    padding = int(max(width, height) * padding_fraction)
    return (max(0, int(xx.min()) - padding), max(0, int(yy.min()) - padding),
            min(image.width, int(xx.max()) + 1 + padding),
            min(image.height, int(yy.max()) + 1 + padding))


def fixed_hand_crops(xyz, faces, valid, floor, bounds):
    frames = np.unique(np.rint(np.linspace(0, 299, 31)).astype(int))
    crops = {}
    placeholder_camera = [[np.zeros(3), np.zeros(3)]]
    for method in METHODS:
        for column, (_, elev, azim) in enumerate(VIEWS):
            reference = cell(xyz, method, faces, frames, valid, placeholder_camera, floor, bounds,
                             elev, azim, 1200)
            crops[method, column] = hand_crop(reference)
    return crops


def crop_and_fit(image, crop, size, resample):
    image = image.crop(crop)
    image.thumbnail(size, resample)
    return image


def rgb_strip(inputs, width, height, source_ids):
    strip = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(strip)
    slot = height // 3
    for index, frame in enumerate((0, 149, 299)):
        files = list(inputs.glob(f"rgb_{frame:03d}.*"))
        if len(files) != 1:
            raise ValueError(f"RGB sample match count: {frame}: {len(files)}")
        with Image.open(files[0]) as source:
            image = source.convert("RGB")
        image.thumbnail((width - 16, slot - 44), Image.Resampling.LANCZOS)
        y = index * slot + 12
        strip.paste(image, ((width - image.width) // 2, y))
        draw.text((8, y + image.height + 5), f"RGB {frame} | ID {source_ids[index]}", fill="#25313e")
    return strip


def render_png(row, prepared, target):
    xyz, faces, shown, valid, cameras, floor, bounds, info, inputs = prepared
    panel, rgb_width, label_width, top = 650, 350, 200, 260
    grid_x = rgb_width + label_width
    canvas = Image.new("RGB", (grid_x + 6 * panel, top + 6 * panel + 80), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{row['dataset']} | {row['sequence_id']} | source {row['frame_ids'][0]}..{row['frame_ids'][-1]} | endpoint-renderable 104"
    size = 35
    while size > 18 and draw.textlength(title, font=base.font(size, True)) > canvas.width - 40:
        size -= 1
    draw.text((20, 20), title, fill="#202a37", font=base.font(size, True))
    draw.text((20, 75), "Joint8 P95 candidate source | all 300 frames rendered | hand-focused fixed crop | five equally spaced 3D samples", fill="#475261", font=base.font(25))
    counts = info["ego_interpolation"]["filled_counts_left_right"]
    draw.text((20, 115), f"Ego intermediate completion: camera-space bracketed smoothstep, filled L/R {counts[0]}/{counts[1]}, no extrapolation | 8fc061a", fill="#475261", font=base.font(22))
    for column, (name, _, _) in enumerate(VIEWS):
        draw.text((grid_x + column * panel + 10, 175), name, fill="#202a37", font=base.font(30))
    canvas.paste(rgb_strip(inputs, rgb_width, 6 * panel, info["rgb_source_frame_ids"]), (0, top))
    crops = fixed_hand_crops(xyz, faces, valid, floor, bounds)
    info["hand_focused_crops"] = {method: [list(crops[method, column]) for column in range(len(VIEWS))]
                                  for method in METHODS}
    info["hand_focused_crop_frames"] = 31
    info["hand_focused_crop_policy"] = "fixed per method/view over 31 uniform frames; camera trajectory may be clipped"
    info["spatial_span_m"] = {}
    for row_index, method in enumerate(METHODS):
        y = top + row_index * panel
        support = info["valid_left_right"][method]
        points = xyz[method][valid[method]].reshape(-1, 3)
        span = float(np.linalg.norm(np.ptp(points, axis=0))) if len(points) else float("nan")
        info["spatial_span_m"][method] = span
        label_lines = LABELS[row_index].split("\n")
        for line_index, label in enumerate(label_lines):
            draw.text((rgb_width + 8, y + 12 + 25 * line_index), label, fill="#202a37",
                      font=base.font(17 if len(label_lines) > 1 else 20, True))
        note_y = y + 18 + 25 * len(label_lines)
        draw.text((rgb_width + 8, note_y), f"valid L/R {support[0]}/{support[1]}", fill="#475261", font=base.font(15))
        draw.text((rgb_width + 8, note_y + 23), f"hand span {span:.2f} m", fill="#475261", font=base.font(15))
        for column, (_, elev, azim) in enumerate(VIEWS):
            image = cell(xyz, method, faces, shown, valid, cameras[method], floor, bounds, elev, azim)
            image = crop_and_fit(image, crops[method, column], (panel - 12, panel - 12),
                                 Image.Resampling.LANCZOS)
            canvas.paste(image, (grid_x + column * panel + (panel - image.width) // 2,
                                 y + (panel - image.height) // 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return {"png": str(target), "size": list(canvas.size)}


def video_frames(prepared):
    xyz, faces, _, valid, cameras, floor, bounds, info, inputs = prepared
    panel, rgb_width, label_width, top = 280, 210, 110, 100
    grid_x = rgb_width + label_width
    width, height = grid_x + 6 * panel, top + 6 * panel
    strip = rgb_strip(inputs, rgb_width, 6 * panel, info["rgb_source_frame_ids"])
    crops = {(method, column): tuple(value) for method, values in info["hand_focused_crops"].items()
             for column, value in enumerate(values)}
    empty_valid = np.zeros_like(valid["gt"])
    empty_cells = {}
    for method in METHODS:
        for column, (_, elev, azim) in enumerate(VIEWS):
            image = base.render_cell(xyz[method], faces, [], empty_valid, cameras[method], floor,
                                     bounds, elev, azim, 560)
            scaled_crop = tuple(int(value * 560 / 1200) for value in crops[method, column])
            image = crop_and_fit(image, scaled_crop, (panel - 6, panel - 6), Image.Resampling.BILINEAR)
            empty_cells[method, column] = image
    for frame in range(300):
        if frame % 30 == 0:
            print(json.dumps({"segment": info["segment_id"], "video_frame": frame, "total_frames": 300}), flush=True)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"{info['dataset']} | {info['sequence_id']} | source {info['source_frame_start']}..{info['source_frame_end']}", fill="#202a37", font=base.font(15, True))
        filled = [side for side in range(2) if info["ego_filled_mask"][frame, side]]
        state = " | Ego interpolated " + "/".join("L" if side == 0 else "R" for side in filled) if filled else ""
        draw.text((12, 35), f"10s | frame {frame:03d}/299 | {frame/30:.2f}s | GT-camera world{state}", fill="#202a37", font=base.font(14))
        for column, (name, _, _) in enumerate(VIEWS):
            draw.text((grid_x + column * panel + 8, 67), name, fill="#202a37", font=base.font(14, True))
        canvas.paste(strip, (0, top))
        empty = not any(valid[method][frame].any() for method in METHODS)
        for row_index, method in enumerate(METHODS):
            y = top + row_index * panel
            label_lines = LABELS[row_index].split("\n")
            for line_index, label in enumerate(label_lines):
                draw.text((rgb_width + 4, y + 6 + 14 * line_index), label, fill="#202a37",
                          font=base.font(10 if len(label_lines) > 1 else 12, True))
            draw.text((rgb_width + 4, y + 9 + 14 * len(label_lines)),
                      f"span {info['spatial_span_m'][method]:.2f}m", fill="#475261", font=base.font(9))
            for column, (_, elev, azim) in enumerate(VIEWS):
                image = (empty_cells[method, column] if empty else
                         cell(xyz, method, faces, [frame], valid, cameras[method], floor,
                              bounds, elev, azim, 560))
                if not empty:
                    scaled_crop = tuple(int(value * 560 / 1200) for value in crops[method, column])
                    image = crop_and_fit(image, scaled_crop, (panel - 6, panel - 6),
                                         Image.Resampling.BILINEAR)
                canvas.paste(image, (grid_x + column * panel + (panel - image.width) // 2,
                                     y + (panel - image.height) // 2))
        yield canvas


def main(row, output, workspace):
    prepared = prepare(row, workspace)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    png = output / "png_gallery" / row["png_filename"]
    mp4 = output / "video_gallery" / row["video_filename"]
    if png.exists() or mp4.exists():
        raise FileExistsError("segment output already exists")
    report = render_png(row, prepared, png)
    width, height = 2000, 1780
    report.update(video_io.encode_mp4(video_frames(prepared), mp4, width=width, height=height, frame_count=300, fps=30))
    interpolation = prepared[7]["ego_interpolation"]
    report.update(segment_id=row["segment_id"], dataset=row["dataset"], source_frame_start=row["frame_ids"][0],
                  source_frame_end=row["frame_ids"][-1], methods=list(METHODS), views=[view[0] for view in VIEWS],
                  visualization_mask="none", ego_interpolation=interpolation,
                  pad_hand_geometry="native 778 vertices",
                  hawor_label=prepared[7]["hawor_label"], hawor_camera=prepared[7]["hawor_camera"])
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    (reports / (row["gallery_stem"] + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
