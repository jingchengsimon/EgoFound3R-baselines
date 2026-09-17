"""Local six-method 10s pilot: five-sample PNG and 300-frame six-view MP4."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "batch_10s_41_p95_wmpjpe_20260910"
spec = importlib.util.spec_from_file_location("legacy_render_10s", LEGACY / "render_batch.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
spec = importlib.util.spec_from_file_location("local_video_io", ROOT / "video_io.py")
video_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_io)
base = legacy.base
METHODS = ("ego", "wilor", "hawor", "reviv4d", "pad_hand", "gt")
LABELS = ("Ego stride5", "WiLoR", "HaWoR", "ReViV4D joints", "PAD-Hand native", "GT")
VIEWS = legacy.VIEWS
EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20))


def rows():
    return [json.loads(line) for line in (ROOT / "selected_manifest.jsonl").read_text().splitlines() if line.strip()]


def linked_inputs(row):
    dest = ROOT / row["dataset"] / "segments" / row["segment_id"] / "inputs"
    source = row["reusable_41_input_dir"]
    if source and Path(source).is_dir():
        source = Path(source)
        for item in source.iterdir():
            if item.name in ("sources.json", "selection.json"):
                continue
            target = dest / item.name
            if not target.exists():
                target.symlink_to(item)
    selection = dest / "selection.json"
    if selection.is_symlink():
        selection.unlink()
    if not selection.exists():
        selection.write_text(json.dumps(row, indent=2) + "\n")
    return dest


def selected_arrays(path, keys):
    with np.load(path, allow_pickle=False) as src:
        return {key: src[key].astype(float) if key != "hand_valid" else src[key].astype(bool) for key in keys}


def prepare(row):
    inputs = linked_inputs(row)
    ds = row["dataset"]
    xyz, faces, shown, valid, cameras, floor, bounds, info = legacy.prepare(ds, inputs, emit_stitched=False)
    c2w = np.concatenate([selected_arrays(inputs / f"{i}_gt.npz", ["camera_c2w"])["camera_c2w"] for i in range(5)])
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]) @ c2w[0, :3, :3].T
    origin = c2w[0, :3, 3]
    display = lambda x: (x - origin) @ rotation.T
    world = lambda x: np.einsum("tij,tsvj->tsvi", c2w[:, :3, :3], x) + c2w[:, None, None, :3, 3]
    for method in ("pad_hand", "reviv4d"):
        parts = []
        granularity = None
        for i, win in enumerate(row["windows"]):
            metadata = json.loads((inputs / f"{i}_{method}_metadata.json").read_text())
            assert metadata["dataset"] == ds and metadata["method"] == method and metadata["frame_ids"] == win["gt"]["frame_ids"]
            path = inputs / f"{i}_{method}.npz"
            with np.load(path, allow_pickle=False) as source:
                key = "hand_vertices_camera" if method == "pad_hand" and "hand_vertices_camera" in source else "hand_joints_camera"
            if granularity is None:
                granularity = key
            assert key == granularity, f"mixed {method} geometry granularity"
            parts.append(selected_arrays(path, ["hand_joints_camera", "hand_valid", key]))
        a = {key: np.concatenate([part[key] for part in parts]) for key in ("hand_joints_camera", "hand_valid", granularity)}
        points = a[granularity]
        assert points.shape == (300, 2, 778 if granularity == "hand_vertices_camera" else 21, 3)
        xyz[method] = display(world(points))
        valid[method] = (a["hand_valid"] & valid["gt"]
                         & np.isfinite(a["hand_joints_camera"]).all(axis=(2, 3))
                         & np.isfinite(points).all(axis=(2, 3)))
    # A single fixed range for all video frames and methods; never recenter each frame.
    sampled = [xyz[m][valid[m]][:, ::max(1, xyz[m].shape[2] // 25), :].reshape(-1, 3) for m in METHODS]
    points = np.concatenate([*sampled, np.asarray(cameras).reshape(-1, 3)])
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    span = max(high - low) * .53
    bounds = (center - span, center + span)
    info.update(methods=list(METHODS), shared_video_bounds=[bounds[0].tolist(), bounds[1].tolist()],
                rgb_indices=[0, 149, 299], rgb_source_frame_ids=[row["frame_ids"][t] for t in (0, 149, 299)],
                video_frames=300, video_fps=30, shared_excluded_mask=row["excluded"])
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
    for axis, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        axis(c - span, c + span)
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
    for scale, z, color in ([(1.10, -.012, [.87, .88, .90]), (1, 0, [.94, .945, .95])] if elev >= 0 else []):
        stage = floor.copy()
        mid = stage.mean(0)
        stage[:, :2] = mid[:2] + (stage[:, :2] - mid[:2]) * scale
        stage[:, 2] += z
        ax.add_collection3d(Poly3DCollection([stage], facecolors=[color], edgecolors="none", zorder=0 if z else 1))
    ax.add_collection3d(Line3DCollection(cameras, colors=[(.72, .48, .14, .6)], linewidths=.55, zorder=2))
    for t in selected:
        for side in range(2):
            if not valid[t, side]:
                continue
            joints = xyz[t, side]
            color = tuple(base.LIGHT[side] * (1 - t / 299) + base.DARK[side] * (t / 299))
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


def rgb_strip(inputs, width, height, source_ids):
    strip = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(strip)
    for i, t in enumerate((0, 149, 299)):
        files = list(inputs.glob(f"rgb_{t:03d}.*"))
        assert len(files) == 1
        with Image.open(files[0]) as source:
            image = source.convert("RGB")
        image.thumbnail((width - 20, height // 3 - 50), Image.Resampling.LANCZOS)
        y = i * (height // 3) + 25
        strip.paste(image, ((width - image.width) // 2, y))
        draw.text((12, y + image.height + 6), f"RGB index {t} | source ID {source_ids[i]}", fill="#25313e")
    return strip


def render_png(row, prepared, target):
    xyz, faces, shown, valid, cameras, floor, bounds, info, inputs = prepared
    panel = 650
    left = 720
    top = 260
    canvas = Image.new("RGB", (left + 6 * panel, top + 6 * panel + 80), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{row['dataset']} | {row['sequence_id']} | source {row['frame_ids'][0]}..{row['frame_ids'][-1]} | Joint8 P95 | W-MPJPE {row['weighted_existing_w_mpjpe_mm']:.2f} mm"
    size = 35
    while size > 18 and draw.textlength(title, font=base.font(size, True)) > canvas.width - 40:
        size -= 1
    draw.text((20, 20), title, fill="#202a37", font=base.font(size, True))
    draw.text((20, 75), "Ego stride5 / WiLoR / HaWoR / ReViV4D / PAD-Hand / GT | GT-camera stitch | 5 shared frames | 300 samples / 30 FPS", fill="#475261", font=base.font(25))
    for j, (name, _, _) in enumerate(VIEWS):
        draw.text((left + j * panel + 10, 160), name, fill="#202a37", font=base.font(30))
    canvas.paste(rgb_strip(inputs, left, 6 * panel, info["rgb_source_frame_ids"]), (0, top))
    for i, method in enumerate(METHODS):
        draw.text((10, top + i * panel + 8), LABELS[i], fill="#202a37", font=base.font(23, True))
        for j, (_, elev, azim) in enumerate(VIEWS):
            image = cell(xyz, method, faces, shown, valid, cameras, floor, bounds, elev, azim)
            image.thumbnail((panel - 12, panel - 12), Image.Resampling.LANCZOS)
            canvas.paste(image, (left + j * panel + (panel - image.width) // 2,
                                 top + i * panel + (panel - image.height) // 2))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return {"png": str(target), "size": list(canvas.size)}


def video_frames(prepared):
    xyz, faces, _, valid, cameras, floor, bounds, info, inputs = prepared
    panel = 238
    left = 360
    top = 96
    width = left + 6 * panel
    height = top + 6 * panel
    strip = rgb_strip(inputs, left, 6 * panel, info["rgb_source_frame_ids"])
    empty_valid = np.zeros_like(valid["gt"])
    empty_cells = []
    for _, elev, azim in VIEWS:
        image = base.render_cell(xyz["gt"], faces, [], empty_valid, cameras, floor, bounds, elev, azim, 480)
        image.thumbnail((panel - 6, panel - 6), Image.Resampling.BILINEAR)
        empty_cells.append(image)
    for t in range(300):
        if t % 30 == 0:
            print(json.dumps({"segment": info["segment_id"], "video_frame": t, "total_frames": 300}), flush=True)
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"{info['dataset']} | {info['sequence_id']} | source {info['source_frame_start']}..{info['source_frame_end']}", fill="#202a37")
        state = " | P95 excluded" if info["shared_excluded_mask"][t] else (" | GT hand invalid" if not valid["gt"][t].any() else "")
        draw.text((12, 34), f"10s display | frame {t:03d}/299 | {t/30:.2f}s | same GT-camera world{state}", fill="#202a37")
        for j, (name, _, _) in enumerate(VIEWS):
            draw.text((left + j * panel + 8, 58), name, fill="#202a37")
        canvas.paste(strip, (0, top))
        empty = not any(valid[method][t].any() for method in METHODS)
        for i, method in enumerate(METHODS):
            draw.text((8, top + i * panel + 6), LABELS[i], fill="#202a37")
            for j, (_, elev, azim) in enumerate(VIEWS):
                if empty:
                    image = empty_cells[j]
                else:
                    image = cell(xyz, method, faces, [t], valid, cameras, floor, bounds, elev, azim, 480)
                    image.thumbnail((panel - 6, panel - 6), Image.Resampling.BILINEAR)
                canvas.paste(image, (left + j * panel + (panel - image.width) // 2,
                                     top + i * panel + (panel - image.height) // 2))
        yield canvas


def main(segment_id, output):
    row = next(row for row in rows() if row["segment_id"] == segment_id)
    prepared = prepare(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    png = output / "png_gallery" / row["png_filename"]
    mp4 = output / "video_gallery" / row["video_filename"]
    if png.exists() or mp4.exists():
        raise FileExistsError("pilot output already exists")
    report = render_png(row, prepared, png)
    report.update(video_io.encode_mp4(video_frames(prepared), mp4, width=1788, height=1524, frame_count=300, fps=30))
    report.update(segment_id=segment_id, dataset=row["dataset"], source_frame_start=row["frame_ids"][0],
                  source_frame_end=row["frame_ids"][-1], methods=list(METHODS), views=[v[0] for v in VIEWS])
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    (reports / (row["gallery_stem"] + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
