#!/usr/bin/env python3
"""Run ReViV's RGB-only 3R+hand demos and write a canonical H2O window output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from formal_evaluation.common.io import load_manifest, select_manifest_window, write_comparison_output
from formal_evaluation.common.schema import SCHEMA_VERSION
from formal_evaluation.datasets.window_inputs import load_window_input


def decode_camera_9d(camera: np.ndarray) -> np.ndarray:
    """Decode ReViV's documented canonical c2w [T,9] representation."""
    camera = np.asarray(camera, dtype=np.float32)
    if camera.ndim == 3 and camera.shape[1:] == (4, 4):
        return camera
    if camera.ndim != 2 or camera.shape[1] != 9:
        raise ValueError(f"expected ReViV camera [T,9] or [T,4,4], got {camera.shape}")
    packed = camera.reshape(-1, 3, 3).transpose(0, 2, 1)
    first = packed[:, :, 0]
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-8)
    second = packed[:, :, 1] - np.sum(first * packed[:, :, 1], axis=1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-8)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(camera), axis=0)
    poses[:, :3, :3] = np.stack((first, second, np.cross(first, second)), axis=-1)
    poses[:, :3, 3] = packed[:, :, 2]
    return poses


def _nearest_indices(target_indices_30fps: list[int], output_frames: int) -> np.ndarray:
    indices = np.rint(np.asarray(target_indices_30fps, dtype=float) * output_frames / 60.0).astype(int)
    return np.clip(indices, 0, output_frames - 1)


def _load_output(path: Path, expected_ndim: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path)
    if value.ndim != expected_ndim:
        raise ValueError(f"unexpected ReViV output shape for {path}: {value.shape}")
    return value


def _run_demos(args: argparse.Namespace, input_video: Path, native_dir: Path) -> None:
    scene_dir, hand_dir = native_dir / "scene", native_dir / "hand"
    subprocess.run([
        sys.executable, "demo_infer.py", "--video", str(input_video), "--output_dir", str(scene_dir),
        "--ckpt_root", str(args.checkpoint_root), "--cosmos_dir", str(args.cosmos_dir),
        "--targets", "tok_cam", "tok_depth", "--amp_dtype", args.amp_dtype,
    ], cwd=args.source_root, check=True)
    subprocess.run([
        sys.executable, "demo_hand.py", "--video", str(input_video), "--output_dir", str(hand_dir),
        "--ckpt_root", str(args.checkpoint_root), "--cosmos_dir", str(args.hand_cosmos_dir),
        "--amp_dtype", args.amp_dtype,
    ], cwd=args.source_root, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path,
                        help="directory produced by scripts/prepare_reviv4d_input.py")
    parser.add_argument("--window-input", type=Path,
                        help="method-neutral record from materialize_six_dataset_window_inputs.py")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--cosmos-dir", type=Path, required=True)
    parser.add_argument(
        "--hand-cosmos-dir",
        type=Path,
        default=Path("Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8"),
        help="Cosmos DV4x8x8 directory for ReViV's 256-pathway hand demo.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--window-id")
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--reuse-native", action="store_true", help="skip official inference and import existing raw files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_input is not None:
        if args.manifest is not None or args.prepared_dir is not None:
            raise ValueError("--window-input cannot be combined with --manifest/--prepared-dir")
        window_input = load_window_input(args.window_input)
        sequence = str(window_input["sequence_id"])
        window_id = str(window_input["window_id"])
        frame_ids = [str(item) for item in window_input["frame_ids"]]
        prepared_dir = args.window_input.parent
        dataset = str(window_input["dataset"])
        output_window_id = str(window_input["cache_id"])
    else:
        if args.manifest is None or args.prepared_dir is None:
            raise ValueError("provide --window-input or both --manifest and --prepared-dir")
        manifest = load_manifest(args.manifest)
        sequence, window_id, frame_ids = select_manifest_window(
            manifest, phase=args.phase, sequence=args.sequence, window_id=args.window_id
        )
        prepared_dir = args.prepared_dir
        dataset = "h2o"
        output_window_id = window_id
    mapping_path, input_video = prepared_dir / "mapping.json", prepared_dir / "input.mp4"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping.get("sequence") != sequence or mapping.get("window_id") != window_id or mapping.get("frame_ids") != frame_ids:
        raise ValueError("prepared ReViV input does not match the requested manifest window")

    output_dir = args.output_root / "reviv4d" / args.phase / output_window_id
    native_dir = output_dir / "native" / "reviv4d"
    start = time.perf_counter()
    if not args.reuse_native:
        _run_demos(args, input_video, native_dir)
    stem = input_video.stem
    scene_dir, hand_dir = native_dir / "scene" / stem, native_dir / "hand" / stem
    camera = decode_camera_9d(_load_output(scene_dir / f"{stem}_tok_cam.npy", 2))
    depth = _load_output(scene_dir / f"{stem}_tok_depth.npy", 4)
    left = _load_output(hand_dir / f"{stem}_tok_lhand.npy", 3)
    right = _load_output(hand_dir / f"{stem}_tok_rhand.npy", 3)
    if left.shape[1:] != (21, 3) or right.shape != left.shape:
        raise ValueError("ReViV hand outputs must be matching [T,21,3] arrays")
    hand_indices = np.asarray(mapping["hand_indices_30fps"], dtype=np.int64)
    if np.any(hand_indices < 0) or np.any(hand_indices >= len(left)):
        raise ValueError("invalid ReViV hand frame mapping")
    scene_indices = _nearest_indices(mapping["hand_indices_30fps"], len(depth))
    camera_indices = _nearest_indices(mapping["hand_indices_30fps"], len(camera))
    depth = depth[..., 0] if depth.shape[-1] in (1, 3) else depth
    if depth.ndim != 3:
        raise ValueError("ReViV depth output must be [T,H,W] or [T,H,W,C]")
    arrays = {
        "camera_c2w": camera[camera_indices],
        "camera_valid": np.isfinite(camera[camera_indices]).all(axis=(1, 2)),
        "depth": depth[scene_indices].astype(np.float32),
        "depth_valid": np.isfinite(depth[scene_indices]) & (depth[scene_indices] > 0),
        "hand_joints_camera": np.stack((left[hand_indices], right[hand_indices]), axis=1).astype(np.float32),
    }
    arrays["hand_valid"] = np.isfinite(arrays["hand_joints_camera"]).all(axis=(2, 3))
    arrays["hand_joints_camera"] = np.nan_to_num(arrays["hand_joints_camera"])
    methods = json.loads(args.methods_config.read_text(encoding="utf-8"))["methods"]
    config = methods["reviv4d"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": "reviv4d",
        "source": config["source"],
        "phase": args.phase,
        "dataset": dataset,
        "sequence": sequence,
        "window_id": window_id,
        "frame_ids": frame_ids,
        "capabilities": {name: True for name in arrays},
        "scale_type": config["scale_type"],
        "camera_convention": "ReViV canonical c2w; no GT first-frame pose anchor applied",
        "hand_coordinate": "ReViV camera-space joints",
        "targets": ["tok_cam", "tok_depth", "tok_lhand", "tok_rhand"],
        "excluded_targets": ["tok_body", "tok_gaze"],
        "scene_indices": scene_indices.tolist(),
        "camera_indices": camera_indices.tolist(),
        "hand_indices": hand_indices.tolist(),
    }
    write_comparison_output(
        output_dir,
        metadata=metadata,
        arrays=arrays,
        run={"status": "success", "elapsed_seconds": time.perf_counter() - start, "reuse_native": args.reuse_native},
        native_metadata={"source": "official ReViV demo outputs", "raw_root": str(native_dir)},
        native_arrays={"camera_9d": camera, "depth": depth, "lhand": left, "rhand": right},
    )
    print(json.dumps({"output_dir": str(output_dir), "status": "success"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
