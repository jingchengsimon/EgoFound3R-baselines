#!/usr/bin/env python3
"""Run released PAD-Hand on a prepared H2O clip and export canonical joints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from formal_evaluation.common.io import load_manifest, select_manifest_window, write_comparison_output
from formal_evaluation.common.schema import SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wilor-python", type=Path, required=True,
                        help="verified interpreter for PAD-Hand's bundled WiLoR front end")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--window-id")
    return parser.parse_args()


def _run_wilor(args: argparse.Namespace, video: Path, native: Path) -> Path:
    result = native / "wilor.npz"
    command = [str(args.wilor_python), str(Path(__file__).with_name("pad_wilor_inference.py")),
        "--source-root", str(args.source_root), "--video", str(video), "--output", str(result),
    ]
    subprocess.run(command, check=True)
    return result


def _joints_from_refined(frame_predictions: list[object], mano: object) -> tuple[np.ndarray, np.ndarray]:
    import torch

    joints = np.full((len(frame_predictions), 2, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((len(frame_predictions), 2), dtype=bool)
    for index, prediction in enumerate(frame_predictions):
        if prediction is None:
            continue
        vertices = prediction["refined_vertices"].to(mano.device).unsqueeze(0)
        with torch.no_grad():
            hand_joints = torch.matmul(vertices.transpose(1, 2), mano.J_regressor).transpose(1, 2)[0]
        hand_joints = hand_joints.detach().cpu().numpy() + np.asarray(prediction["cam_t"], dtype=np.float32)
        slot = 1 if prediction["is_right"] else 0
        joints[index, slot] = hand_joints
        valid[index, slot] = np.isfinite(hand_joints).all()
    return joints, valid


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    sequence, window_id, frame_ids = select_manifest_window(
        manifest, phase=args.phase, sequence=args.sequence, window_id=args.window_id
    )
    mapping = json.loads((args.prepared_dir / "mapping.json").read_text(encoding="utf-8"))
    if mapping.get("sequence") != sequence or mapping.get("window_id") != window_id or mapping.get("frame_ids") != frame_ids:
        raise ValueError("prepared PAD-Hand input does not match the requested manifest window")
    video = args.prepared_dir / "input.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    if int(mapping.get("context_frames", 0)) < 16:
        raise ValueError("PAD-Hand requires at least 16 contiguous context frames")

    output_dir = args.output_root / "pad_hand" / args.phase / window_id
    native_dir = output_dir / "native" / "pad_hand"
    native_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    wilor_npz = _run_wilor(args, video, native_dir)

    source_root = args.source_root.resolve()
    if not (source_root / "demo.py").is_file():
        raise FileNotFoundError(source_root / "demo.py")
    sys.path.insert(0, str(source_root))
    previous_cwd = Path.cwd()
    try:
        os.chdir(source_root)
        import torch
        from demo import MANO, load_pad_hand_model, load_wilor_results, refine_with_pad_hand

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = args.checkpoint or source_root / "checkpoints" / "pad_hand.pt"
        model = load_pad_hand_model(checkpoint, device)
        predictions, _ = load_wilor_results(wilor_npz)
        refined = refine_with_pad_hand(predictions, model, MANO("RIGHT", device), device)
        all_joints, all_valid = _joints_from_refined(refined, MANO("RIGHT", device))
    finally:
        os.chdir(previous_cwd)

    indices = np.asarray(mapping["hand_indices_30fps"], dtype=np.int64)
    if indices.shape != (len(frame_ids),) or np.any(indices < 0) or np.any(indices >= len(all_joints)):
        raise ValueError("invalid PAD-Hand frame mapping")
    arrays = {
        "hand_joints_camera": np.nan_to_num(all_joints[indices]),
        "hand_valid": all_valid[indices],
    }
    methods = json.loads(args.methods_config.read_text(encoding="utf-8"))["methods"]
    write_comparison_output(
        output_dir,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "method": "pad_hand",
            "source": methods["pad_hand"]["source"],
            "phase": args.phase,
            "sequence": sequence,
            "window_id": window_id,
            "frame_ids": frame_ids,
            "capabilities": {name: True for name in arrays},
            "scale_type": methods["pad_hand"]["scale_type"],
            "coordinate_space": "camera",
            "native_frame_indices": indices.tolist(),
        },
        arrays=arrays,
        run={"status": "success", "elapsed_seconds": time.perf_counter() - start},
        native_metadata={"source": "official PAD-Hand and WiLoR inference", "wilor_npz": str(wilor_npz)},
        native_arrays={"hand_joints_camera": all_joints, "hand_valid": all_valid},
    )
    print(json.dumps({"output_dir": str(output_dir), "status": "success"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
