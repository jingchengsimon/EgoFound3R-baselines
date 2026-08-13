#!/usr/bin/env python3
"""Measure ReViV4D's native 60-frame RGB-clip throughput for 500 H2O targets."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

TARGET_FRAMES = 500
CLIP_FRAMES = 60


def _select_sequence(data_root: Path) -> tuple[str, list[Path]]:
    candidates = sorted(
        rgb.parent.parent.relative_to(data_root).as_posix()
        for rgb in data_root.glob("subject4_ego/*/*/cam4/rgb")
        if sum(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in rgb.iterdir()) >= 540
    )
    if not candidates:
        raise RuntimeError("no H2O test sequence has the 540 frames needed for 500 ReViV targets")
    sequence = random.Random(0).choice(candidates)
    frames = sorted(
        path for path in (data_root / sequence / "cam4/rgb").iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return sequence, frames


def _clip_plan() -> list[tuple[int, int]]:
    return [(start, CLIP_FRAMES) for start in range(0, TARGET_FRAMES, CLIP_FRAMES)]


def _write_clip(frames: list[Path], start: int, output: Path) -> Path:
    clip = frames[start : start + CLIP_FRAMES]
    if len(clip) != CLIP_FRAMES:
        raise RuntimeError(f"clip at {start} has {len(clip)}, expected {CLIP_FRAMES}")
    suffix = clip[0].suffix.lower()
    if any(path.suffix.lower() != suffix for path in clip):
        raise RuntimeError("ReViV clip frames must use one image extension")
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True)
    for index, frame in enumerate(clip):
        os.symlink(frame.resolve(), frame_dir / f"{index:06d}{suffix}")
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-framerate", "30", "-i", str(frame_dir / f"%06d{suffix}"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output / f"{output.name}.mp4"),
    ], check=True)
    return output / f"{output.name}.mp4"


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> float:
    start = time.perf_counter()
    subprocess.run(command, cwd=cwd, env=env, check=True)
    return time.perf_counter() - start


def _verify_outputs(root: Path, stems: list[str]) -> None:
    import numpy as np

    for stem in stems:
        scene = root / "scene" / stem
        hand = root / "hand" / stem
        camera = np.load(scene / f"{stem}_tok_cam.npy")
        depth_paths = list(scene.glob(f"{stem}_tok_depth*.npy"))
        if camera.shape != (60, 9) or len(depth_paths) != 1 or np.load(depth_paths[0]).ndim != 4:
            raise RuntimeError(f"unexpected scene output for {stem}")
        for side in ("l", "r"):
            if np.load(hand / f"{stem}_tok_{side}hand.npy").shape != (60, 21, 3):
                raise RuntimeError(f"unexpected {side}hand output for {stem}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--scene-cosmos-dir", type=Path, required=True)
    parser.add_argument("--hand-cosmos-dir", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--cuda-visible-devices", default="7")
    parser.add_argument("--pythonpath", default="")
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    sequence, frames = _select_sequence(args.data_root)
    plan = _clip_plan()
    args.output_dir.mkdir(parents=True)
    clips = args.output_dir / "clips"
    videos_dir = args.output_dir / "videos"
    videos_dir.mkdir()
    for index, (start, _) in enumerate(plan):
        video = _write_clip(frames, start, clips / f"clip_{index:03d}")
        os.symlink(video.resolve(), videos_dir / video.name)
    videos = sorted(videos_dir.glob("*.mp4"))
    stems = [video.stem for video in videos]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath
    native = args.output_dir / "native"
    scene_seconds = _run([
        args.python_executable, "demo_infer.py", "--video", str(videos_dir), "--output_dir", str(native / "scene"),
        "--ckpt_root", str(args.checkpoint_root), "--cosmos_dir", str(args.scene_cosmos_dir),
        "--targets", "tok_cam", "tok_depth", "--amp_dtype", args.amp_dtype,
    ], cwd=args.source_root, env=env)
    hand_seconds = _run([
        args.python_executable, "demo_hand.py", "--video", str(videos_dir), "--output_dir", str(native / "hand"),
        "--ckpt_root", str(args.checkpoint_root), "--cosmos_dir", str(args.hand_cosmos_dir),
        "--amp_dtype", args.amp_dtype,
    ], cwd=args.source_root, env=env)
    _verify_outputs(native, stems)
    total = scene_seconds + hand_seconds
    report = {
        "status": "success",
        "sequence": sequence,
        "target_frame_ids": [path.stem for path in frames[:TARGET_FRAMES]],
        "target_output_frames": TARGET_FRAMES,
        "model_input_frames": len(videos) * CLIP_FRAMES,
        "strategy": "9 contiguous ReViV 60-frame clips; clips 0-7 each contribute 60 targets, clip 8 contributes its first 20 targets",
        "timing_boundary": "official scene and hand demo execution, including checkpoint load, video decode, Cosmos tokenization, generation, detokenization, and native output writes; excludes H2O RGB-to-MP4 preparation",
        "scene_seconds": scene_seconds,
        "hand_seconds": hand_seconds,
        "total_seconds": total,
        "effective_target_fps": TARGET_FRAMES / total,
        "clip_stems": stems,
    }
    (args.output_dir / "benchmark_reviv4d_500.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
