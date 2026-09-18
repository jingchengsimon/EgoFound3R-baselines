"""Batch-render the 3D deliverables (summary figure + panels + video) for many clips.

One persistent worker per GPU pulls tasks from a queue, so the per-segment
process/CUDA/pickle startup is paid once for the whole batch and the scene of the
segment a worker is currently on stays loaded.  Task kinds:

* ``summary``  - the 9-column x 5-view matrix (`fig1_3d_summary.png`, multi-time
  gradient) plus one PNG per method and viewpoint under ``panels_3d/``;
* ``video``    - one frame chunk of the 300-frame, 9-column matrix
  (``video1_3d_matrix.mp4``), rendered with the same batched code path as the
  single-segment tool.

Everything reuses `render_3d_video.build_scene`-equivalent plumbing and
`paper_viz.batch_render`, i.e. the exact rendering code that was verified to be
pixel-identical (or, with ``--bin-size 64``, visually identical) to the
reference.  Finished outputs are skipped, so an interrupted batch can be rerun.

Example
-------
``python3 tools/batch_3d_render.py --staged-root <dir> --out-root <out> \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 --stages summary video \
    --cell 192 --panel-size 1024 --chunks 4``
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import render_3d_video as V                                        # noqa: E402
from batch_3d_video import scene_state                             # noqa: E402
from render_3d_video import R                                      # noqa: E402
from paper_viz.batch_render import batched_render                  # noqa: E402
from paper_viz.sequences3d import METHOD_LABELS_3D                 # noqa: E402

_W: dict = {}


def _renderer(scene, cell, supersample):
    """Renderer for a given cell size (the scene keeps one per size)."""
    cache = scene.setdefault("renderers", {})
    if cell not in cache:
        from egohandmetric_prompt.inference_multiview import scene_framing_points
        renderer = R.HandMultiviewRenderer(scene["bounds"], cell_size=cell,
                                           device=scene["device"], ground=True,
                                           supersample=supersample,
                                           framing_points=scene["framing"])
        for name, (yaw, pitch) in R.VIEWPOINTS.items():
            renderer.view_poses[name] = R.turntable_pose(scene["scene_center"], scene["fit_points"],
                                                         yaw, pitch, margin=scene["margin"])
        cache[cell] = renderer
    return cache[cell]


def _path_parts(scene):
    """Camera-trajectory tubes for the whole clip, built once per segment.

    ``camera_overlay_parts`` only emits the path when ``path_indices`` has more
    than one entry, and the result is independent of the frame being rendered, so
    it is cached in the scene instead of being rebuilt 300 times.
    """
    from egohandmetric_prompt.inference_multiview import camera_overlay_parts
    cached = scene.get("path_parts")
    if cached is None:
        cached = camera_overlay_parts(scene["camera"], [], show_frustums=False,
                                      path_indices=np.arange(scene["frames_total"]),
                                      scale=scene["camera_scale"]) if scene["show_camera"] else []
        scene["path_parts"] = cached
    return cached


def _cell_parts(scene, keyframes):
    """Per-method parts for the summary (temporal gradient over the keyframes)."""
    from egohandmetric_prompt.inference_multiview import mesh_parts
    parts = {}
    for name in scene["drawn"]:
        if name in scene["sequences"]:
            parts[name] = mesh_parts(scene["sequences"][name], list(keyframes), scene["times"],
                                     temporal_colors=True)
        else:
            entry = scene["store"][name]
            parts[name] = R.skeleton_parts(entry["joints"], entry["valid"], list(keyframes),
                                           scene["times"], temporal_colors=True)
    return parts


def render_summary(scene, out_dir: Path, args) -> None:
    """Matrix figure + one panel per method and viewpoint."""
    from egohandmetric_prompt.inference_multiview import camera_overlay_parts
    keyframes = np.rint(np.linspace(0, scene["frames_total"] - 1, args.keyframes)).astype(int)
    parts = _cell_parts(scene, keyframes)
    annotations = (camera_overlay_parts(scene["camera"], list(keyframes), show_frustums=True,
                                        path_indices=(), scale=scene["camera_scale"])
                   + _path_parts(scene)) if scene["show_camera"] else []
    items = [parts[n] + annotations for n in scene["drawn"]]
    shadows = [parts[n] for n in scene["drawn"]]
    cells = {}
    renderer = _renderer(scene, args.cell, args.supersample)
    for view in scene["views"]:
        rendered = batched_render(renderer, items, view, shadows, bin_size=args.bin_size)
        for name, rgb in zip(scene["drawn"], rendered):
            if view == "top" and R.TOP_VIEW_ROT90_CCW:
                rgb = np.ascontiguousarray(np.rot90(rgb, k=1))
            cells[(METHOD_LABELS_3D[name], view)] = rgb
    for row, t in enumerate(keyframes):
        tile = V.rgb_tile(scene["windows"], int(t), args.cell)
        for view in scene["views"]:
            cells[("Input RGB", view)] = tile
    columns = ["Input RGB"] + [METHOD_LABELS_3D[m] for m in scene["drawn"]]
    camera_note = "with camera rig" if scene["show_camera"] else "hand-only (camera hidden)"
    grid = R.compose_matrix(cells, columns, list(scene["views"]),
                            title=f"{scene['segment_id']} | {len(keyframes)} time samples | "
                                  f"world space | rows = views, columns = methods | {camera_note} | "
                                  "EgoFound3R = 8fc061a infer (default post-processing)",
                            cell_px=args.cell, temporal=True, camera_legend=scene["show_camera"])
    from PIL import Image as PILImage
    out_dir.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(grid).save(out_dir / "fig1_3d_summary.png")
    if args.panel_size:
        panel_renderer = _renderer(scene, args.panel_size, args.supersample)
        for view in scene["views"]:
            rendered = batched_render(panel_renderer, items, view, shadows, bin_size=args.bin_size)
            for name, rgb in zip(scene["drawn"], rendered):
                if view == "top" and R.TOP_VIEW_ROT90_CCW:
                    rgb = np.ascontiguousarray(np.rot90(rgb, k=1))
                target = out_dir / "panels_3d" / name / f"{view}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                PILImage.fromarray(rgb).save(target)


def worker(device: str, tasks, out_root: str, args_dict: dict) -> None:
    from PIL import Image as PILImage
    args = argparse.Namespace(**args_dict)
    current, scene = None, None
    while True:
        task = tasks.get()
        if task is None:
            break
        segment, kind = task[0], task[1]
        if current != segment:
            V._SCENE.clear()
            scene = V._SCENE
            scene.update(scene_state(args, Path(segment), device))
            # The video path (`_scene_frame`) reads ``scene["renderer"]``; the image
            # path builds renderers lazily per cell size, so bind the video one here.
            scene["renderer"] = _renderer(scene, args.cell, args.supersample)
            current = segment
        out_dir = Path(out_root) / Path(segment).name
        if kind == "summary":
            out_dir.mkdir(parents=True, exist_ok=True)
            render_summary(scene, out_dir, args)
        else:
            start, stop = task[2], task[3]
            frames_dir = out_dir / "_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            if scene["show_camera"]:
                # Keep the whole-clip trajectory in every video frame (same look as
                # the figure); the frustums still follow the current time step.
                scene["extra_annotations"] = _path_parts(scene)
            for t in range(start, stop, args.stride):
                PILImage.fromarray(V._scene_frame(t)).save(frames_dir / f"{t:05d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    for flag in ("src-dir", "prepared-root", "hawor-root", "hawor-index", "contact-root", "mapping"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--stages", nargs="+", default=["summary", "video"],
                        choices=("summary", "video"))
    parser.add_argument("--cell", type=int, default=192)
    parser.add_argument("--panel-size", type=int, default=1024,
                        help="side of the per-method/per-view panels (0 disables them)")
    parser.add_argument("--keyframes", type=int, default=5)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--bin-size", type=int, default=64)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--camera-overlay", choices=("show", "hide"), default="show")
    parser.add_argument("--fit-margin", type=float, default=None)
    parser.add_argument("--camera-scale", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-frames", type=int, default=None,
                        help="render only the first N frames of each video (testing)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    segments = sorted(p for p in args.staged_root.iterdir()
                      if p.is_dir() and (p / "selection.json").is_file())
    if args.limit:
        segments = segments[:args.limit]

    plan, todo = [], []
    for segment in segments:
        out_dir = args.out_root / segment.name
        selection = json.loads((segment / "selection.json").read_text())
        frames_total = 60 * len(selection["windows"])
        wanted = min(frames_total, args.video_frames) if args.video_frames else frames_total
        if "summary" in args.stages and (args.force or not (out_dir / "fig1_3d_summary.png").is_file()):
            todo.append((str(segment), "summary"))
        if "video" in args.stages and (args.force or not (out_dir / "video1_3d_matrix.mp4").is_file()):
            step = max(1, int(np.ceil(wanted / args.chunks)))
            for start in range(0, wanted, step):
                todo.append((str(segment), "video", start, min(start + step, wanted)))
        plan.append((segment, frames_total, wanted))
    print(f"{len(segments)} segments | {len(todo)} tasks | {len(args.devices)} workers | "
          f"stages={','.join(args.stages)}")
    if args.dry_run:
        for task in todo[:40]:
            print("  ", task)
        return

    start_time = time.time()
    context = multiprocessing.get_context("spawn")
    tasks = context.Queue()
    workers = [context.Process(target=worker, args=(device, tasks, str(args.out_root), vars(args)))
               for device in args.devices]
    for process in workers:
        process.start()
    for task in todo:
        tasks.put(task)
    for _ in workers:
        tasks.put(None)
    for process in workers:
        process.join()

    results = []
    for segment, frames_total, wanted in plan:
        out_dir = args.out_root / segment.name
        video = out_dir / "video1_3d_matrix.mp4"
        frames = sorted((out_dir / "_frames").glob("*.png")) if (out_dir / "_frames").is_dir() else []
        if "video" in args.stages and not video.is_file():
            if len(frames) == wanted:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                                "-i", str(out_dir / "_frames" / "%05d.png"), "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", "-crf", str(args.crf), "-movflags", "+faststart",
                                str(video)], check=True)
                status = "rendered"
            else:
                status = f"video_incomplete({len(frames)}/{wanted})"
        else:
            status = "present"
        summary = out_dir / "fig1_3d_summary.png"
        panels = sorted((out_dir / "panels_3d").glob("*/*.png")) if (out_dir / "panels_3d").is_dir() else []
        results.append({"segment": segment.name, "status": status,
                        "summary": summary.is_file(), "panels": len(panels),
                        "video_frames": len(frames), "video": str(video)})
        print(f"  {segment.name}: {status} | summary={summary.is_file()} panels={len(panels)}")
    payload = {"segments": len(segments), "tasks": len(todo),
               "seconds": round(time.time() - start_time, 1), "results": results}
    (args.out_root / "batch_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wall {payload['seconds']}s -> {args.out_root / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
