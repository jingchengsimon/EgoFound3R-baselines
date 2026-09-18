"""World-space multi-view VIDEO in exactly the style of `fig1_3d_summary.png`.

Same layout as the summary figure — leftmost synchronised Input RGB column, then
the eight methods, rows = the five viewpoints — but the 3D cells show only the
*current* time step (no temporal gradient) and the RGB column plays the clip, so
one matrix frame per video frame at `fps`.

The scene setup (level-to-camera world, trajectory-driven ground plate, per-mode
fit margins, view poses) is imported from `render_3d_summary` so the video and the
figure can never drift apart.

Example
-------
``/mnt/workspace/sjc/envs/egofound3r/bin/python tools/render_3d_video.py \
    --inputs-dir <staged segment> --src-dir <relay dir> ... \
    --out <out dir> --camera-overlay show --fps 30``
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import render_3d_summary as R                                      # noqa: E402
from paper_viz.inputs import rgb_path                              # noqa: E402
from paper_viz.sequences3d import (METHODS_3D, METHOD_LABELS_3D,  # noqa: E402
                                   assert_camera_bundle_frame, level_in_place)
from paper_viz.video import VideoWriter                            # noqa: E402

# Worker state: the scene is built once in the parent and inherited by the forked
# workers, so only frames and finished cells cross the process boundary.
_WORK: dict = {}


def rgb_tile(windows, t: int, cell: int):
    """Decoded + resized source RGB for one frame (letterboxed into a square cell)."""
    from PIL import Image as PILImage
    w_index, f_index = divmod(int(t), 60)
    picture = PILImage.open(rgb_path(windows[w_index], f_index))
    if picture.mode != "RGB":
        picture = picture.convert("RGB")
    scale = cell / picture.width
    picture = picture.resize((cell, max(1, int(round(picture.height * scale)))),
                             PILImage.Resampling.LANCZOS)
    tile = PILImage.new("RGB", (cell, cell), (255, 255, 255))
    tile.paste(picture, (0, max((cell - picture.height) // 2, 0)))
    return np.asarray(tile)


def _scene_frame(t: int, tile=None) -> "np.ndarray":
    """Compose the full matrix for one time step (used by every shard mode)."""
    import torch
    from egohandmetric_prompt.inference_multiview import camera_overlay_parts, mesh_parts
    scene = _SCENE
    renderer = scene["renderer"]
    annotations = (camera_overlay_parts(scene["camera"], [t], show_frustums=True, path_indices=(),
                                        scale=scene["camera_scale"])
                   + list(scene.get("extra_annotations") or ()) if scene["show_camera"] else [])
    by_method = scene.get("annotations_by_method")     # per-row cameras (batch path)
    cells = {}
    from paper_viz.batch_render import batched_render
    with torch.no_grad():
        parts_of = {}
        for name in scene["drawn"]:
            if name in scene["sequences"]:
                parts_of[name] = mesh_parts(scene["sequences"][name], [t], scene["times"],
                                            temporal_colors=False)
            else:
                entry = scene["store"][name]
                parts_of[name] = R.skeleton_parts(entry["joints"], entry["valid"], [t], scene["times"],
                                                  temporal_colors=False)
        if scene.get("batch_cells", True):
            # One batched call per viewpoint; verified pixel-identical to the
            # per-cell path and ~2.6x faster per frame.
            for view in scene["views"]:
                rendered = batched_render(renderer,
                                          [parts_of[n] + (by_method[n] if by_method else annotations)
                                           for n in scene["drawn"]], view,
                                          [parts_of[n] for n in scene["drawn"]],
                                          bin_size=scene.get("bin_size"))
                for name, rgb in zip(scene["drawn"], rendered):
                    if view == "top" and R.TOP_VIEW_ROT90_CCW:
                        rgb = np.ascontiguousarray(np.rot90(rgb, k=1))
                    cells[(R.METHOD_LABELS_3D[name], view)] = rgb
        else:
            for name in scene["drawn"]:
                row_annotations = by_method[name] if by_method else annotations
                for view in scene["views"]:
                    rgb = renderer.render(parts_of[name] + row_annotations, view,
                                          shadow_parts=parts_of[name])
                    if view == "top" and R.TOP_VIEW_ROT90_CCW:
                        rgb = np.ascontiguousarray(np.rot90(rgb, k=1))
                    cells[(R.METHOD_LABELS_3D[name], view)] = rgb
    tile = rgb_tile(scene["windows"], t, scene["cell"]) if tile is None else tile
    for view in scene["views"]:
        cells[("Input RGB", view)] = tile
    camera_note = "with camera rig" if scene["show_camera"] else "hand-only (camera hidden)"
    return R.compose_matrix(cells, scene["columns"], list(scene["views"]),
                            title=f"{scene['segment_id']} | frame {int(t)} | world space | "
                                  f"rows = views, columns = methods | {camera_note} | "
                                  "EgoFound3R = 8fc061a infer (default post-processing)",
                            cell_px=scene["cell"], temporal=False,
                            camera_legend=scene["show_camera"])


_SCENE: dict = {}


def _worker_scene(state: dict) -> None:
    """Spawned worker for frame-range sharding: adopt the scene state."""
    _SCENE.update(state)


def _render_range(task) -> int:
    """Render one contiguous frame range and write lossless PNG frames."""
    from PIL import Image as PILImage
    device, indices, frames_dir = task
    if _SCENE.get("device") != device:            # build this worker's own renderer
        _SCENE["device"] = device
        _SCENE["renderer"] = R.HandMultiviewRenderer(_SCENE["bounds"], cell_size=_SCENE["cell"],
                                                     device=device, ground=True,
                                                     supersample=_SCENE["supersample"],
                                                     framing_points=_SCENE["framing"])
        for name, (yaw, pitch) in R.VIEWPOINTS.items():
            _SCENE["renderer"].view_poses[name] = R.turntable_pose(_SCENE["scene_center"],
                                                                   _SCENE["fit_points"], yaw, pitch,
                                                                   margin=_SCENE["margin"])
    written = 0
    for t in indices:
        PILImage.fromarray(_scene_frame(t)).save(Path(frames_dir) / f"{t:05d}.png")
        written += 1
    return written


def _worker_init(state: dict) -> None:
    """Spawned worker: adopt the (pickled) scene state."""
    _WORK.update(state)


def _render_cells(task) -> dict:
    """Render the (method, view) cells of one shard for frame ``t`` (lazy renderer)."""
    import torch
    from egohandmetric_prompt.inference_multiview import camera_overlay_parts, mesh_parts
    shard, t = task
    cache = _WORK.setdefault("renderers", {})
    if shard not in cache:
        renderer = R.HandMultiviewRenderer(_WORK["bounds"], cell_size=_WORK["cell"],
                                           device=_WORK["devices"][shard], ground=True,
                                           supersample=_WORK["supersample"],
                                           framing_points=_WORK["framing"])
        for name, (yaw, pitch) in R.VIEWPOINTS.items():
            renderer.view_poses[name] = R.turntable_pose(_WORK["scene_center"], _WORK["fit_points"],
                                                         yaw, pitch, margin=_WORK["margin"])
        cache[shard] = renderer
    renderer = cache[shard]
    methods = _WORK["groups"][shard]
    store, camera, times = _WORK["store"], _WORK["camera"], _WORK["times"]
    sequences, show_camera, camera_scale = _WORK["sequences"], _WORK["show_camera"], _WORK["camera_scale"]
    out = {}
    with torch.no_grad():
      for name in methods:
        if name in sequences:
            parts = mesh_parts(sequences[name], [t], times, temporal_colors=False)
        else:
            entry = store[name]
            parts = R.skeleton_parts(entry["joints"], entry["valid"], [t], times, temporal_colors=False)
        annotations = (camera_overlay_parts(camera, [t], show_frustums=True, path_indices=(),
                                            scale=camera_scale) if show_camera else [])
        for view in _WORK["views"]:
            rgb = renderer.render(parts + annotations, view, shadow_parts=parts)
            if view == "top" and R.TOP_VIEW_ROT90_CCW:
                rgb = np.ascontiguousarray(np.rot90(rgb, k=1))
            out[(R.METHOD_LABELS_3D[name], view)] = rgb
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("inputs-dir", "src-dir", "prepared-root", "hawor-root", "hawor-index",
                 "contact-root", "mapping"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=list(R.VIEWPOINTS))
    parser.add_argument("--cell", type=int, default=320)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None,
                        help="shard the eight methods across several GPUs (e.g. cuda:0 cuda:1)")
    parser.add_argument("--camera-overlay", choices=("show", "hide"), default="show")
    parser.add_argument("--fit-margin", type=float, default=None)
    parser.add_argument("--camera-scale", type=float, default=0.12)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--bin-size", type=int, default=64,
                        help="rasterizer bin grid; 64 is 1.66x faster and visually identical, "
                             "0 restores the bit-exact reference rasterizer")
    parser.add_argument("--no-batch-cells", dest="batch_cells", action="store_false", default=True,
                        help="one render() call per cell instead of the batched path")
    parser.add_argument("--shard-mode", choices=("method", "frames"), default="frames",
                        help="frames = each GPU renders a contiguous time range into lossless "
                             "PNGs, then one ffmpeg pass (no per-frame sync/IPC)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="limit for a preview")
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    devices = args.devices or [args.device]
    sharded = len(devices) > 1 and args.shard_mode == "method"
    frame_sharded = len(devices) > 1 and args.shard_mode == "frames"
    registry, windows, mano = R.load_segment(args)
    store = R.build_segment_sequences(type("S", (), {"windows": windows})(), mano)
    from egohandmetric_prompt.inference_multiview import (HandSequence, camera_overlay_parts,
                                                          mesh_parts, scene_framing_points)

    camera = R.camera_sequence(windows, mano)
    rotation = R.level_rotation(camera.camera_to_display)
    rotation4 = np.eye(4)
    rotation4[:3, :3] = rotation
    camera.camera_to_display = np.einsum("ij,tjk->tik", rotation4, camera.camera_to_display)
    level_in_place(store, rotation)
    assert_camera_bundle_frame(store, camera)

    frames_total = len(windows) * 60
    times = np.arange(frames_total, dtype=float) / 30.0
    present = [m for m in METHODS_3D if m in store]
    mesh_methods = [m for m in present if "vertices" in store[m]]
    drawn = [m for m in present if m in mesh_methods or "joints" in store[m]]
    sequences = {}
    for name in mesh_methods:
        entry = store[name]
        sequences[name] = HandSequence(name=METHOD_LABELS_3D[name], vertices=entry["vertices"],
                                       valid=entry["valid"], faces=entry["faces"],
                                       camera_source="gt",
                                       camera_to_display=camera.camera_to_display,
                                       camera_valid=camera.camera_valid,
                                       camera_K=camera.camera_K,
                                       camera_K_valid=camera.camera_K_valid,
                                       image_size_hw=camera.image_size_hw)

    # Identical scene framing to the summary figure.
    valid_indices = np.flatnonzero(store["gt"]["valid"].any(axis=-1).any(axis=-1))
    framing_all = scene_framing_points([sequences[m] for m in mesh_methods], np.arange(frames_total))
    trajectory = np.concatenate([store[m]["vertices"][store[m]["valid"]] for m in mesh_methods
                                 if store[m]["valid"].any()], axis=0)
    low, high = trajectory.min(0), trajectory.max(0)
    center, size = 0.5 * (low + high), high - low
    gap = max(0.015, 0.08 * float(size[1]))
    bounds = (np.array([center[0] - 0.55 * size[0], low[1] - gap, center[2] - 0.55 * size[2]]),
              np.array([center[0] + 0.55 * size[0], high[1] + 0.25 * size[1], center[2] + 0.55 * size[2]]))
    renderer = None
    if not sharded:
        renderer = R.HandMultiviewRenderer(bounds, cell_size=args.cell, device=args.device, ground=True,
                                           supersample=args.supersample,
                                           framing_points=scene_framing_points([sequences[m] for m in mesh_methods],
                                                                               valid_indices))
    show_camera = args.camera_overlay == "show"
    cloud = np.percentile(np.asarray(framing_all, float), [4, 96], axis=0)
    fit_points = np.array(np.meshgrid(*zip(cloud[0], cloud[1]))).T.reshape(-1, 3)
    margin = args.fit_margin if args.fit_margin else (0.82 if show_camera else 0.72)
    scene_center = 0.5 * (bounds[0] + bounds[1])
    if renderer is not None:
        for name, (yaw, pitch) in R.VIEWPOINTS.items():
            renderer.view_poses[name] = R.turntable_pose(scene_center, fit_points, yaw, pitch,
                                                         margin=margin)

    args.out.mkdir(parents=True, exist_ok=True)
    stop = frames_total if args.frames is None else min(frames_total, args.start + args.frames)
    writer = VideoWriter(args.out / "video1_3d_matrix.mp4", fps=args.fps)
    if frame_sharded:
        # Frame-range sharding: every GPU renders a contiguous time range into
        # lossless PNGs (identical pixels to the piped path), then one ffmpeg pass.
        frames_dir = args.out / "_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        indices = list(range(args.start, stop, args.stride))
        per = int(np.ceil(len(indices) / len(devices)))
        tasks = [(device, indices[i * per:(i + 1) * per], frames_dir)
                 for i, device in enumerate(devices) if indices[i * per:(i + 1) * per]]
        state = dict(bounds=bounds, framing=framing_all, fit_points=fit_points,
                     scene_center=scene_center, margin=margin, cell=args.cell,
                     supersample=args.supersample, store=store, camera=camera, times=times,
                     sequences=sequences, windows=windows, show_camera=show_camera,
                     camera_scale=args.camera_scale, views=list(args.views), drawn=drawn,
                     columns=["Input RGB"] + [METHOD_LABELS_3D[m] for m in drawn],
                     segment_id=registry["segment_id"], batch_cells=args.batch_cells,
                     bin_size=args.bin_size)
        context = multiprocessing.get_context("spawn")
        with context.Pool(len(tasks), initializer=_worker_scene, initargs=(state,)) as pool:
            for count in pool.map(_render_range, tasks):
                print(f"  shard done ({count} frames)", flush=True)
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", str(args.crf), "-movflags", "+faststart",
                        str(args.out / "video1_3d_matrix.mp4")], check=True)
        print(f"wrote {args.out / 'video1_3d_matrix.mp4'} ({len(indices)} frames, cell {args.cell}, "
              f"camera {args.camera_overlay}, {len(tasks)} frame shards)")
        return
    pool = None
    if sharded:
        # Fork *after* the scene exists: children inherit it and only exchange
        # finished cells, so the matrix stays byte-identical to the single-GPU run.
        _WORK.update(bounds=bounds, framing=framing_all, fit_points=fit_points,
                     scene_center=scene_center, margin=margin, cell=args.cell,
                     supersample=args.supersample, store=store, camera=camera, times=times,
                     sequences=sequences, show_camera=show_camera, camera_scale=args.camera_scale,
                     devices=devices, views=list(args.views),
                     groups=[drawn[index::len(devices)] for index in range(len(devices))])
        state = {key: value for key, value in _WORK.items() if key != "renderers"}
        pool = multiprocessing.get_context("spawn").Pool(len(devices), initializer=_worker_init,
                                                         initargs=(state,))

    written = 0
    for t in range(args.start, stop, args.stride):
        cells = {}
        if pool is not None:
            for shard_cells in pool.map(_render_cells, [(i, t) for i in range(len(devices))]):
                cells.update(shard_cells)
        frame_annotations = (camera_overlay_parts(camera, [t], show_frustums=True, path_indices=(),
                                                  scale=args.camera_scale) if show_camera else [])
        for name in ([] if pool is not None else mesh_methods):
            parts = mesh_parts(sequences[name], [t], times, temporal_colors=False)
            annotations = frame_annotations
            for view in args.views:
                rgb = renderer.render(parts + annotations, view, shadow_parts=parts)
                if view == "top" and R.TOP_VIEW_ROT90_CCW:
                    rgb = np.ascontiguousarray(np.rot90(rgb, k=1))
                cells[(METHOD_LABELS_3D[name], view)] = rgb
        if pool is None and "reviv4d" in drawn:
            entry = store["reviv4d"]
            parts = R.skeleton_parts(entry["joints"], entry["valid"], [t], times, temporal_colors=False)
            annotations = frame_annotations
            for view in args.views:
                rgb = renderer.render(parts + annotations, view, shadow_parts=parts)
                if view == "top" and R.TOP_VIEW_ROT90_CCW:
                    rgb = np.ascontiguousarray(np.rot90(rgb, k=1))
                cells[(METHOD_LABELS_3D["reviv4d"], view)] = rgb
        # Leftmost column: the same synchronised RGB frame on every row.
        from PIL import Image as PILImage
        w_index, f_index = divmod(int(t), 60)
        picture = PILImage.open(rgb_path(windows[w_index], f_index))
        if picture.mode != "RGB":
            picture = picture.convert("RGB")
        scale = args.cell / picture.width
        picture = picture.resize((args.cell, max(1, int(round(picture.height * scale)))), PILImage.Resampling.LANCZOS)
        tile = PILImage.new("RGB", (args.cell, args.cell), (255, 255, 255))
        tile.paste(picture, (0, max((args.cell - picture.height) // 2, 0)))
        for view in args.views:
            cells[("Input RGB", view)] = np.asarray(tile)
        columns = ["Input RGB"] + [METHOD_LABELS_3D[m] for m in drawn]
        camera_note = "with camera rig" if show_camera else "hand-only (camera hidden)"
        grid = R.compose_matrix(cells, columns, list(args.views),
                                title=f"{registry['segment_id']} | frame {int(t)} | world space | "
                                      f"rows = views, columns = methods | {camera_note} | "
                                      "EgoFound3R = 8fc061a infer (default post-processing)",
                                cell_px=args.cell,
                                temporal=False, camera_legend=show_camera)
        writer.add(PILImage.fromarray(grid))
        written += 1
        if written % 10 == 0:
            print(f"  {written} frames written", flush=True)
    total = writer.close()
    if pool is not None:
        pool.close()
        pool.join()
    print(f"wrote {args.out / 'video1_3d_matrix.mp4'} ({total} frames, cell {args.cell}, "
          f"camera {args.camera_overlay})")


if __name__ == "__main__":
    main()
