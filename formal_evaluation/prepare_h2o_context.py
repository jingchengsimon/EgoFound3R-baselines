#!/usr/bin/env python3
"""Build a contiguous RGB context clip for an H2O evaluation window."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from PIL import Image

from formal_evaluation.common.io import load_manifest, select_manifest_window


def _sequence_frames(data_root: Path, sequence: str) -> list[Path]:
    rgb_dir = data_root / sequence / "cam4" / "rgb"
    frames = sorted(path for path in rgb_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        raise FileNotFoundError(f"no RGB frames in {rgb_dir}")
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=60,
                        help="contiguous RGB context length; ReViV requires 60")
    parser.add_argument("--sequence")
    parser.add_argument("--window-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    sequence, window_id, frame_ids = select_manifest_window(
        manifest, phase=args.phase, sequence=args.sequence, window_id=args.window_id
    )
    frames = _sequence_frames(args.data_root, sequence)
    positions = {frame.stem: index for index, frame in enumerate(frames)}
    if any(frame_id not in positions for frame_id in frame_ids):
        raise FileNotFoundError("evaluation frame missing from sequence RGB directory")
    if args.context_frames <= 0 or len(frames) < args.context_frames:
        raise ValueError("sequence is shorter than the requested context length")
    first = positions[frame_ids[0]]
    start = min(max(first - (args.context_frames - len(frame_ids)) // 2, 0), len(frames) - args.context_frames)
    clip = frames[start:start + args.context_frames]
    target_indices = [positions[frame_id] - start for frame_id in frame_ids]
    if any(index < 0 or index >= args.context_frames for index in target_indices):
        raise ValueError("failed to include every evaluation frame in the RGB context clip")

    if (args.output_dir / "input.mp4").exists() or (args.output_dir / "mapping.json").exists():
        raise FileExistsError(f"refusing to overwrite prepared input: {args.output_dir}")
    suffix = clip[0].suffix.lower()
    if any(frame.suffix.lower() != suffix for frame in clip):
        raise ValueError("ReViV context frames must use one image extension")
    with Image.open(clip[0]) as image:
        width, height = image.size
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir()
    for index, frame in enumerate(clip):
        with Image.open(frame) as image:
            if image.size != (width, height):
                raise ValueError(f"inconsistent RGB resolution: {frame}")
        os.symlink(frame.resolve(), frame_dir / f"{index:06d}{suffix}")
    video_path = args.output_dir / "input.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-framerate", "30", "-i", str(frame_dir / f"%06d{suffix}"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path),
    ], check=True)

    mapping = {
        "sequence": sequence,
        "window_id": window_id,
        "frame_ids": frame_ids,
        "context_frame_ids": [frame.stem for frame in clip],
        "hand_indices_30fps": target_indices,
        "context_frames": args.context_frames,
        "input_fps": 30.0,
        "duration_seconds": 2.0,
    }
    (args.output_dir / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"input": str(video_path), "mapping": str(args.output_dir / "mapping.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
