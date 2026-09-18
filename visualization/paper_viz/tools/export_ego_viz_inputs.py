"""Export the compact NPZ consumed by paper_viz from multiclip inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def tensor(value) -> np.ndarray:
    return value.detach().cpu().float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-output", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-root-k-source", choices=("pred", "gt"))
    args = parser.parse_args()

    payload = torch.load(args.inference_output, map_location="cpu", weights_only=False)
    root = payload.get("root_intrinsics", {})
    source = root.get("source", payload.get("provenance", {}).get("root_k_source"))
    if source is None:
        source = "gt" if payload.get("gt_assisted_inference") else "pred"
    if args.require_root_k_source is not None and source != args.require_root_k_source:
        raise ValueError(f"root K source {source!r} != {args.require_root_k_source!r}")
    if source == "gt" and not bool(payload.get("gt_assisted_inference")):
        raise ValueError("GT-K export requires gt_assisted_inference=true")

    metric = payload["metric_predictions"]
    hand = metric["hand"]
    hand778 = metric["hand_778"]
    arrays = {
        "vertex778_xyz_camera": tensor(hand778["vertex_xyz_metric"]),
        "joint21_xyz_camera": tensor(hand["joint_xyz_metric"]),
        "vertex778_visibility_probability": tensor(hand778["vertex_visibility_probability"]),
        "vertex778_contact_probability": tensor(hand778["vertex_contact_probability"]),
        "hand_valid": hand778["valid"].detach().cpu().bool().numpy(),
        "intrinsics_full": tensor(payload["intrinsics_K_full"]),
        "camera_pose_c2w": tensor(payload["camera_pose_metric_full"]),
    }
    frame_counts = {key: len(value) for key, value in arrays.items()}
    if len(set(frame_counts.values())) != 1:
        raise ValueError(f"inference arrays do not share one frame axis: {frame_counts}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print({"root_k_source": source, "frames": next(iter(frame_counts.values())),
           "output": str(args.out)})


if __name__ == "__main__":
    main()
