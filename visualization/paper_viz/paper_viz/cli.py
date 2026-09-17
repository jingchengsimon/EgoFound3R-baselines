"""Command-line entry: render paper figures and videos for one 10 s segment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import render2d, render3d, style
from .inputs import SegmentSources, WindowSources, arrays, load_window
from .mano import ManoAsset
from .scene import build_scene, select_frames
from .video import VideoWriter

METHODS = style.METHODS
VIEWS_FIG = ("front", "left", "top", "right", "side", "bottom")
VIEWS_VIDEO = ("front", "left", "top")


def build_registry(args) -> dict:
    """Assemble per-window source paths for the smoke segment from explicit roots."""
    selection = json.loads((Path(args.inputs_dir) / "selection.json").read_text())
    hawor_index = {}
    if getattr(args, "hawor_index", None):
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
            "ego": Path(args.inputs_dir) / f"{index}_ego.npz",
            "ego_full": src / "ego_full.npz",
            "wilor": src / "wilor.npz",
            "hawor": hawor_dir / "predictions.npz",
            "reviv4d": src / "reviv4d.npz",
            "pad_hand": src / "pad_hand.npz",
            "egoforce": src / "egoforce.npz",
            "dyn_hamr": src / "dyn_hamr.npz",
        }
        baselines = {
            "ego_contact": Path(args.contact_root) /
                f"ego_vertex_contact_distance_8095_retry3_20260911/egofound3r_stride5/shards/"
                f"{selection['dataset']}/{selection['dataset']}/{cache}.npz",
            "s2contact": Path(args.contact_root) /
                f"s2contact_vertex_8095_retry7_20260911/predictions/{selection['dataset']}/0/"
                f"s2contact/formal/{cache}/predictions.npz",
            "contactopt": Path(args.contact_root) /
                f"contactopt_vertex_8095_retry3_20260911/predictions/{selection['dataset']}/0/"
                f"contactopt/formal/{cache}/predictions.npz",
            "interactvlm": src / "interactvlm.npz",
        }
        registry["windows"].append({
            "cache_id": cache,
            "window_id": window["window_id"],
            "methods": {k: str(v) for k, v in methods.items()},
            "baselines": {k: str(v) for k, v in baselines.items()},
        })
    return registry


def load_segment(registry: dict, args) -> SegmentSources:
    mano = ManoAsset(Path(args.mapping))
    windows = []
    for entry in registry["windows"]:
        windows.append(load_window(entry["cache_id"], entry["window_id"],
                                   Path(args.prepared_root), entry["methods"],
                                   entry["baselines"], mano))
    return SegmentSources(dataset=registry["dataset"], sequence_id=registry["sequence_id"],
                          segment_id=registry["segment_id"], windows=windows, mano=mano,
                          mapping_path=Path(args.mapping))


def gt_valid_mask(segment: SegmentSources) -> np.ndarray:
    pieces = []
    for window in segment.windows:
        gt = window.methods["gt"]
        pieces.append(gt["hand_valid"].astype(bool) & np.isfinite(gt["hand_joints_camera"]).all(axis=(2, 3)))
    return np.concatenate(pieces)


def render_segment(segment: SegmentSources, out: Path, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if args.stages == "fig2_frame":
        return render_fig2_frame(segment, out, args)
    selected = select_frames(gt_valid_mask(segment), count=5)
    if args.stages == "fig2":
        report = {"segment_id": segment.segment_id, "selected_frames": selected.tolist(),
                  "outputs": {}}
        render_2d_block(segment, out, report, args, selected)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    scene = build_scene(segment.windows, METHODS, segment.mano, selected,
                        segment.sequence_id, segment.segment_id, segment.dataset)
    report = {"segment_id": segment.segment_id, "selected_frames": selected.tolist(),
              "outputs": {}}

    figure = render3d.figure1(scene, METHODS, VIEWS_FIG, cell=args.cell3d)
    figure.save(out / "fig1_3d_multiview.png")
    report["outputs"]["fig1"] = str(out / "fig1_3d_multiview.png")
    if not args.skip_panels:
        for method in METHODS:
            for key in VIEWS_FIG:
                label, elev, azim = render3d.VIEW_PRESETS[key]
                cell = render3d.render_cell(scene, method, elev, azim, pixel_size=1024)
                panel_dir = out / "panels_3d" / key
                panel_dir.mkdir(parents=True, exist_ok=True)
                cell.save(panel_dir / f"{method}.png")
        report["outputs"]["panels_3d"] = str(out / "panels_3d")

    if not args.skip_videos:
        writer = VideoWriter(out / "video1_3d_matrix.mp4", fps=args.fps)
        for index, frame in enumerate(render3d.video1(scene, METHODS, VIEWS_VIDEO,
                                                       cell=args.cell3d_video, fps=args.fps)):
            if index % args.video_stride:
                continue
            writer.add(frame)
        report["outputs"]["video1"] = str(out / "video1_3d_matrix.mp4")
        report["video1_frames"] = writer.close()

    render_2d_block(segment, out, report, args, selected)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def render_2d_block(segment: SegmentSources, out: Path, report: dict, args, selected) -> None:
    """Figure 2 (one frame per row), per-column panels and the 2D overlay video."""
    figure_frames = select_frames(gt_valid_mask(segment), count=args.rows)
    cells = {}
    for r, t in enumerate(figure_frames):
        w_index, f_index = divmod(int(t), 60)
        window = segment.windows[w_index]
        frame = render2d.Frame2D(window, f_index, segment.mano, cell_w=args.cell2d)
        for c, (method, signal) in enumerate(render2d.COLUMNS):
            cells[(r, c)] = render2d.column_cell(frame, method, signal, f_index)
    row_labels = [f"frame {int(t)} / {segment.windows[int(t) // 60].frame_ids[int(t) % 60]}"
                  for t in figure_frames]
    figure2 = compose_figure2(cells, len(figure_frames), row_labels, segment,
                              subtitle="2D camera-space overlay | one frame per row | "
                                       "EgoFound3R = 8fc061a infer (default post-processing)")
    figure2.save(out / "fig2_2d_matrix.png")
    report["outputs"]["fig2"] = str(out / "fig2_2d_matrix.png")
    if not args.skip_panels:
        center = int(selected[len(selected) // 2])
        w_index, f_index = divmod(center, 60)
        window = segment.windows[w_index]
        frame = render2d.Frame2D(window, f_index, segment.mano, cell_w=args.cell2d)
        panel_dir = out / "panels_2d"
        panel_dir.mkdir(parents=True, exist_ok=True)
        for c, (method, signal) in enumerate(render2d.COLUMNS):
            render2d.column_cell(frame, method, signal, f_index).save(
                panel_dir / f"{c:02d}_{method}_{signal}.png")
        report["outputs"]["panels_2d"] = str(panel_dir)

    if not args.skip_videos:
        writer = VideoWriter(out / "video2_2d_matrix.mp4", fps=args.fps)
        total_frames = 60 * len(segment.windows)
        for t in range(0, total_frames, args.video_stride):
            w_index, f_index = divmod(t, 60)
            window = segment.windows[w_index]
            frame = render2d.Frame2D(window, f_index, segment.mano, cell_w=args.cell2d_video)
            cells = {}
            for c, (method, signal) in enumerate(render2d.COLUMNS):
                cells[(0, c)] = render2d.column_cell(frame, method, signal, f_index)
            writer.add(compose_figure2(cells, 1, [f"{window.frame_ids[f_index]}"], segment,
                                       subtitle=f"2D camera-space overlay | clip frame {t} "
                                                f"| source frame {window.frame_ids[f_index]}"))
        report["outputs"]["video2"] = str(out / "video2_2d_matrix.mp4")
        report["video2_frames"] = writer.close()


def render_fig2_frame(segment: SegmentSources, out: Path, args) -> dict:
    t0 = int(np.linspace(0, 299, 6).astype(int)[2])
    w_index, f_index = divmod(t0, 60)
    window = segment.windows[w_index]
    frame = render2d.Frame2D(window, f_index, segment.mano, cell_w=args.cell2d)
    cells = {}
    for c, (method, signal) in enumerate(render2d.COLUMNS):
        cells[(0, c)] = render2d.column_cell(frame, method, signal, f_index)
    row_labels = [f"frame {t0} / {window.frame_ids[f_index]}"]
    figure = compose_figure2(cells, 1, row_labels, segment,
                             subtitle=f"2D camera-space overlay | single frame {t0} review")
    figure.save(out / "fig2_frame.png")
    report = {"segment_id": segment.segment_id, "frame": t0,
              "outputs": {"fig2_frame": str(out / "fig2_frame.png")}}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compose_figure2(cells, rows, row_labels, segment, subtitle: str) -> "Image.Image":
    from .layout import compose_grid
    return compose_grid(cells, rows, len(render2d.COLUMNS), row_labels,
                        list(render2d.COLUMN_LABELS),
                        title=f"{segment.dataset} | {segment.sequence_id}",
                        subtitle=subtitle,
                        footer="Baseline columns show hand geometry only; visibility, contact "
                               "and distance are shown for EgoFound3R and GT.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", required=True, help="Staged segment inputs (selection.json + ego npz)")
    parser.add_argument("--src-dir", required=True, help="Relayed per-cache npz directory")
    parser.add_argument("--prepared-root", required=True, help="Prepared window_inputs/<dataset> root")
    parser.add_argument("--hawor-root", required=True, help="HaWoR formal root")
    parser.add_argument("--hawor-index", default=None,
                        help="Optional HaWoR predictions.jsonl mapping window_id -> prediction_dir")
    parser.add_argument("--contact-root", required=True, help="result3_completion_20260910_v1 root")
    parser.add_argument("--mapping", required=True, help="mano_195_to_778.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cell3d", type=int, default=760)
    parser.add_argument("--cell3d-video", type=int, default=420)
    parser.add_argument("--cell2d", type=int, default=480)
    parser.add_argument("--cell2d-video", type=int, default=300)
    parser.add_argument("--rows", type=int, default=5,
                        help="2D figure rows; one source frame per row")
    parser.add_argument("--palette", choices=("reference", "soft"), default="reference",
                        help="reference = 8fc061a viser/monitor colours; soft = previous ramps")
    parser.add_argument("--signal-style", choices=("wireframe", "face"), default="face",
                        help="wireframe = points + face edges; face = faint face wash + points/edges")
    parser.add_argument("--face-alpha", type=float, default=0.30,
                        help="alpha of the faint face wash used by --signal-style face")
    parser.add_argument("--face-occluded-factor", type=float, default=0.35,
                        help="multiplicative alpha for faces that are occluded (fraction of --face-alpha)")
    parser.add_argument("--geometry-brightness", type=float, default=1.60,
                        help="value/saturation lift for the mesh colours (hue preserved)")
    parser.add_argument("--signal-brightness", type=float, default=1.15,
                        help="value lift for the signal palettes (hue preserved)")
    parser.add_argument("--geometry-line-width", type=int, default=1,
                        help="stroke width of the mesh edges in geometry columns")
    parser.add_argument("--geometry-dot-radius", type=int, default=1,
                        help="radius of the vertex markers in geometry columns")
    parser.add_argument("--geometry-face-alpha", type=float, default=0.30,
                        help="alpha of the faint face wash under the geometry wireframe")
    parser.add_argument("--geometry-occluded-alpha", type=float, default=0.45,
                        help="opacity of occluded (back-side) dots and edges in geometry columns")
    parser.add_argument("--geometry-stroke-lighten", type=float, default=0.65,
                        help="white mix applied to the geometry dots and edges so they read "
                             "brighter than the face wash")
    parser.add_argument("--geometry-contour", type=int, default=0,
                        help="width of the hand silhouette contour in geometry columns (0 = off)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--skip-panels", action="store_true")
    parser.add_argument("--stages", choices=("all", "fig2", "fig2_frame"), default="all")
    return parser


def apply_style_args(args) -> None:
    style.set_palette(args.palette)
    style.set_signal_style(args.signal_style)
    style.set_face_alpha(args.face_alpha)
    style.set_face_occluded_factor(args.face_occluded_factor)
    style.set_brightness(args.geometry_brightness, args.signal_brightness)
    style.set_geometry_stroke(args.geometry_line_width, args.geometry_dot_radius)
    style.set_geometry_face_alpha(args.geometry_face_alpha)
    style.set_geometry_occluded_alpha(args.geometry_occluded_alpha)
    style.set_geometry_lighten(args.geometry_stroke_lighten)
    style.set_geometry_contour(args.geometry_contour)


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_style_args(args)
    registry = build_registry(args)
    segment = load_segment(registry, args)
    report = render_segment(segment, Path(args.out), args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
