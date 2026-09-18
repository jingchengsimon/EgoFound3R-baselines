"""Batch-render many 3D videos with long-lived GPU workers.

Motivation: one video segment costs ~40 s of process/CUDA/pickle startup in the
single-segment tool, which dominates a 100+ segment batch.  Here every GPU gets
one **persistent** worker; work is a queue of (segment, frame-chunk) tasks that
workers steal as they finish, and each worker keeps the segment it is working on
loaded (rebuilding only when the segment changes).  The parent only tracks
completion, runs one ffmpeg pass per finished segment and writes a summary, so it
can also be re-run to resume an interrupted batch.

The rendering itself is `render_3d_video._scene_frame` (batched cells), i.e. the
exact same code path — and therefore the exact same pixels — as the single
segment tool.

Example
-------
``python3 tools/batch_3d_video.py --staged-root <dir with one subdir per segment> \
    --out-root <out> --devices cuda:0 cuda:1 cuda:2 cuda:3 --chunks 4 --cell 192``
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
from render_3d_video import R                                      # noqa: E402
from paper_viz.sequences3d import METHODS_3D, METHOD_LABELS_3D     # noqa: E402

_STATE: dict = {}


def scene_state(args, inputs_dir: Path, device: str) -> dict:
    """Build the render state for one segment (same math as the single tool)."""
    import argparse as ap
    from egohandmetric_prompt.inference_multiview import HandSequence, scene_framing_points

    ns = ap.Namespace(inputs_dir=inputs_dir, src_dir=args.src_dir,
                      prepared_root=args.prepared_root, hawor_root=args.hawor_root,
                      hawor_index=args.hawor_index, contact_root=args.contact_root,
                      mapping=args.mapping)
    registry, windows, mano = R.load_segment(ns)
    store = R.build_segment_sequences(type("S", (), {"windows": windows})(), mano)
    camera = R.camera_sequence(windows, mano)
    rotation = R.level_rotation(camera.camera_to_display)
    rotation4 = np.eye(4)
    rotation4[:3, :3] = rotation
    camera.camera_to_display = np.einsum("ij,tjk->tik", rotation4, camera.camera_to_display)
    for entry in store.values():
        if "vertices" in entry:
            entry["vertices"] = np.einsum("ij,tsvj->tsvi", rotation, entry["vertices"])
        if "joints" in entry:
            entry["joints"] = np.einsum("ij,tsvj->tsvi", rotation, entry["joints"])

    frames_total = len(windows) * 60
    times = np.arange(frames_total, dtype=float) / 30.0
    present = [m for m in METHODS_3D if m in store]
    mesh_methods = [m for m in present if "vertices" in store[m]]
    drawn = [m for m in present if m in mesh_methods or "joints" in store[m]]
    sequences = {m: HandSequence(name=METHOD_LABELS_3D[m], vertices=store[m]["vertices"],
                                 valid=store[m]["valid"], faces=store[m]["faces"],
                                 camera_source="gt", camera_to_display=camera.camera_to_display,
                                 camera_valid=camera.camera_valid, camera_K=camera.camera_K,
                                 camera_K_valid=camera.camera_K_valid,
                                 image_size_hw=camera.image_size_hw) for m in mesh_methods}
    trajectory = np.concatenate([store[m]["vertices"][store[m]["valid"]] for m in mesh_methods
                                 if store[m]["valid"].any()], axis=0)
    low, high = trajectory.min(0), trajectory.max(0)
    center, size = 0.5 * (low + high), high - low
    gap = max(0.015, 0.08 * float(size[1]))
    bounds = (np.array([center[0] - 0.55 * size[0], low[1] - gap, center[2] - 0.55 * size[2]]),
              np.array([center[0] + 0.55 * size[0], high[1] + 0.25 * size[1], center[2] + 0.55 * size[2]]))
    valid_indices = np.flatnonzero(store["gt"]["valid"].any(axis=-1).any(axis=-1))
    framing_all = scene_framing_points([sequences[m] for m in mesh_methods], np.arange(frames_total))
    framing = scene_framing_points([sequences[m] for m in mesh_methods], valid_indices)
    renderer = R.HandMultiviewRenderer(bounds, cell_size=args.cell, device=device, ground=True,
                                       supersample=args.supersample, framing_points=framing)
    show_camera = args.camera_overlay == "show"
    cloud = np.percentile(np.asarray(framing_all, float), [4, 96], axis=0)
    fit_points = np.array(np.meshgrid(*zip(cloud[0], cloud[1]))).T.reshape(-1, 3)
    margin = args.fit_margin if args.fit_margin else (0.82 if show_camera else 0.72)
    scene_center = 0.5 * (bounds[0] + bounds[1])
    for name, (yaw, pitch) in R.VIEWPOINTS.items():
        renderer.view_poses[name] = R.turntable_pose(scene_center, fit_points, yaw, pitch, margin=margin)
    return dict(bounds=bounds, framing=framing_all, fit_points=fit_points, scene_center=scene_center,
                margin=margin, cell=args.cell, supersample=args.supersample, store=store,
                camera=camera, times=times, sequences=sequences, windows=windows,
                show_camera=show_camera, camera_scale=args.camera_scale, views=list(V.R.VIEWPOINTS),
                drawn=drawn, columns=["Input RGB"] + [METHOD_LABELS_3D[m] for m in drawn],
                segment_id=registry["segment_id"], batch_cells=True, renderer=renderer,
                device=device, frames_total=frames_total)


def worker(device: str, tasks, out_root: str, args_dict: dict) -> None:
    """Persistent worker: steal tasks, keep the current segment loaded."""
    from PIL import Image as PILImage
    args = argparse.Namespace(**args_dict)
    current = None
    while True:
        task = tasks.get()
        if task is None:
            break
        segment_dir, start, stop = task
        if current != segment_dir:
            V._SCENE.clear()
            V._SCENE.update(scene_state(args, Path(segment_dir), device))
            current = segment_dir
        frames_dir = Path(out_root) / Path(segment_dir).name / "_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for t in range(start, stop, args.stride):
            PILImage.fromarray(V._scene_frame(t)).save(frames_dir / f"{t:05d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staged-root", type=Path, required=True,
                        help="directory with one staged segment subdirectory per clip")
    parser.add_argument("--staged-list", type=Path,
                        help="optional file with one staged directory path per line")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--hawor-index", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--chunks", type=int, default=4, help="frame chunks per segment")
    parser.add_argument("--cell", type=int, default=192)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--camera-overlay", choices=("show", "hide"), default="show")
    parser.add_argument("--fit-margin", type=float, default=None)
    parser.add_argument("--camera-scale", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.staged_list:
        segments = [Path(line.strip()) for line in args.staged_list.read_text().splitlines() if line.strip()]
    else:
        segments = sorted(p for p in args.staged_root.iterdir()
                          if p.is_dir() and (p / "selection.json").is_file())
    if args.limit:
        segments = segments[:args.limit]
    todo = [s for s in segments if args.force or not (args.out_root / s.name / "video1_3d_matrix.mp4").is_file()]
    print(f"{len(segments)} segments, {len(todo)} to render, {len(args.devices)} workers")
    if args.dry_run:
        for s in todo:
            print("  ", s.name)
        return

    start_time = time.time()
    tasks = multiprocessing.get_context("spawn").Queue()
    workers = [multiprocessing.get_context("spawn").Process(
        target=worker, args=(device, tasks, str(args.out_root), vars(args)))
        for device in args.devices]
    for process in workers:
        process.start()
    plan = {}
    total_tasks = 0
    for segment in todo:
        probe = None
        # frame count comes from the staged selection (60 frames per window)
        import json as _json
        selection = _json.loads((segment / "selection.json").read_text())
        frames_total = 60 * len(selection["windows"])
        step = max(1, int(np.ceil(frames_total / args.chunks)))
        chunks = [(segment, start, min(start + step, frames_total))
                  for start in range(0, frames_total, step)]
        plan[segment.name] = {"path": str(segment), "frames": frames_total,
                              "chunks_done": 0, "chunks": len(chunks), "status": "pending"}
        for chunk in chunks:
            tasks.put(chunk)
            total_tasks += 1
    for _ in workers:
        tasks.put(None)
    for process in workers:
        process.join()

    summary = {"segments": len(todo), "tasks": total_tasks, "seconds": round(time.time() - start_time, 1),
               "results": []}
    for segment in todo:
        frames_dir = args.out_root / segment.name / "_frames"
        video = args.out_root / segment.name / "video1_3d_matrix.mp4"
        frames = sorted(frames_dir.glob("*.png"))
        status = "rendered"
        if len(frames) != plan[segment.name]["frames"]:
            status = f"frame_count_mismatch({len(frames)}/{plan[segment.name]['frames']})"
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                            "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", str(args.crf), "-movflags", "+faststart",
                            str(video)], check=True)
        summary["results"].append({"segment": segment.name, "status": status,
                                   "frames": len(frames), "video": str(video)})
        print(f"  {segment.name}: {status}")
    (args.out_root / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wall {summary['seconds']}s -> {args.out_root / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
