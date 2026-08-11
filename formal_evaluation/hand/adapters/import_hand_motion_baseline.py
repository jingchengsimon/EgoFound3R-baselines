#!/usr/bin/env python3
"""Import normalized external hand-motion predictions into the canonical format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from formal_evaluation.common.io import load_manifest, select_manifest_window, write_comparison_output
from formal_evaluation.common.schema import SCHEMA_VERSION


def _indices(values: np.ndarray, frame_count: int, requested: list[int] | None) -> np.ndarray:
    if requested is None:
        if len(values) != frame_count:
            raise ValueError("native frame count differs from the evaluation window; pass --frame-indices")
        return np.arange(frame_count)
    indices = np.asarray(requested, dtype=np.int64)
    if indices.shape != (frame_count,) or np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("--frame-indices must contain one valid native index per evaluation frame")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dyn_hamr", "pad_hand"), required=True)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--native-predictions", type=Path, required=True,
                        help=".npz with a (T,2,21,3) joint tensor and optional validity/camera tensors")
    parser.add_argument("--joints-key", default="hand_joints_camera")
    parser.add_argument("--valid-key", default="hand_valid")
    parser.add_argument("--camera-key", default=None)
    parser.add_argument("--coordinate-space", choices=("camera", "world"), default="camera")
    parser.add_argument("--frame-indices", nargs="+", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--window-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    sequence, window_id, frame_ids = select_manifest_window(
        manifest, phase=args.phase, sequence=args.sequence, window_id=args.window_id
    )
    with np.load(args.native_predictions, allow_pickle=False) as archive:
        if args.joints_key not in archive:
            raise KeyError(f"missing {args.joints_key!r} in {args.native_predictions}")
        joints = np.asarray(archive[args.joints_key], dtype=np.float32)
        if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
            raise ValueError(f"{args.joints_key} must have shape (T,2,21,3), got {joints.shape}")
        frame_indices = _indices(joints, len(frame_ids), args.frame_indices)
        joints = joints[frame_indices]
        if args.valid_key in archive:
            valid = np.asarray(archive[args.valid_key], dtype=bool)[frame_indices]
            if valid.shape != joints.shape[:2]:
                raise ValueError(f"{args.valid_key} must have shape (T,2)")
        else:
            valid = np.isfinite(joints).all(axis=(2, 3))
        valid &= np.isfinite(joints).all(axis=(2, 3))
        arrays: dict[str, np.ndarray] = {
            f"hand_joints_{args.coordinate_space}": np.nan_to_num(joints),
            "hand_valid": valid,
        }
        if args.camera_key:
            camera = np.asarray(archive[args.camera_key], dtype=np.float32)[frame_indices]
            if camera.shape != (len(frame_ids), 4, 4):
                raise ValueError(f"{args.camera_key} must have shape (T,4,4)")
            arrays["camera_c2w"] = np.nan_to_num(camera)
            arrays["camera_valid"] = np.isfinite(camera).all(axis=(1, 2))

    methods = json.loads(args.methods_config.read_text(encoding="utf-8"))["methods"]
    config = methods[args.method]
    output_dir = args.output_root / args.method / args.phase / window_id
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": args.method,
        "source": config["source"],
        "phase": args.phase,
        "sequence": sequence,
        "window_id": window_id,
        "frame_ids": frame_ids,
        "capabilities": {name: True for name in arrays},
        "scale_type": config["scale_type"],
        "coordinate_space": args.coordinate_space,
        "native_frame_indices": frame_indices.tolist(),
    }
    write_comparison_output(
        output_dir,
        metadata=metadata,
        arrays=arrays,
        run={"status": "imported", "native_predictions": str(args.native_predictions)},
        native_metadata={"joints_key": args.joints_key, "valid_key": args.valid_key},
    )
    print(json.dumps({"output_dir": str(output_dir), "status": "imported"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
