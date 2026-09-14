#!/usr/bin/env python3
"""Run released PAD-Hand per hand and export canonical two-hand geometry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from formal_evaluation.common.io import load_manifest, select_manifest_window, write_comparison_output
from formal_evaluation.common.mano_sampling import downsample_mano_vertices
from formal_evaluation.common.schema import SCHEMA_VERSION
from formal_evaluation.datasets.window_inputs import load_window_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--window-input", type=Path,
                        help="method-neutral record from materialize_six_dataset_window_inputs.py")
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
        "--both-hands",
    ]
    subprocess.run(command, check=True)
    return result


def _split_wilor_by_side(source_path: Path, native_dir: Path) -> tuple[Path, Path]:
    """Give the released single-hand PAD loader one complete temporal track per side."""
    result = []
    with np.load(source_path, allow_pickle=False) as source:
        count = source["vertices"].shape[0]
        if source["vertices"].shape != (count, 2, 778, 3) or source["is_right"].shape != (count, 2):
            raise ValueError("PAD WiLoR frontend did not preserve both hand slots")
        for side, label in enumerate(("left", "right")):
            flags = source["is_right"][:, side]
            present = np.isfinite(source["vertices"]).all(axis=(2, 3))[:, side]
            if np.any(present & (~np.isfinite(flags) | ((flags > 0.5) != bool(side)))):
                raise ValueError(f"PAD WiLoR {label} slot has inconsistent handedness")
            target = native_dir / f"wilor_{label}.npz"
            np.savez_compressed(target, **{
                key: source[key][:, side] for key in (
                    "vertices", "cam_t", "global_orient", "hand_pose", "betas",
                    "is_right", "img_size", "scaled_focal")}, fps=source["fps"])
            result.append(target)
    return result[0], result[1]


def _geometry_from_refined(
    frame_predictions: list[object],
    mano: object,
    fx_per_frame: "list[float | None] | None" = None,
    scaled_focal: "np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Camera-space MANO geometry for each frame, at the real-camera depth.

    PAD-Hand refines WiLoR's mesh but keeps WiLoR's translation, which solves depth
    against WiLoR's rendering focal length. The official demo rescales it with the
    camera's fx; without that, absolute MPJPE measures a fictitious depth while
    root-relative metrics stay correct because cam_t is a constant per-hand offset.
    """
    import torch

    vertices_out = np.full((len(frame_predictions), 2, 778, 3), np.nan, dtype=np.float32)
    joints = np.full((len(frame_predictions), 2, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((len(frame_predictions), 2), dtype=bool)
    rescaled = 0
    for index, prediction in enumerate(frame_predictions):
        if prediction is None:
            continue
        vertices = prediction["refined_vertices"].to(mano.device).unsqueeze(0)
        with torch.no_grad():
            hand_joints = torch.matmul(vertices.transpose(1, 2), mano.J_regressor).transpose(1, 2)[0]
        cam_t = np.asarray(prediction["cam_t"], dtype=np.float32).copy()
        fx = None if fx_per_frame is None else fx_per_frame[index]
        focal = None if scaled_focal is None else float(scaled_focal[index])
        if fx is not None and focal is not None and np.isfinite(focal) and focal > 1e-9:
            cam_t[2] *= fx / focal
            rescaled += 1
        camera_vertices = vertices[0].detach().cpu().numpy() + cam_t
        hand_joints = hand_joints.detach().cpu().numpy() + cam_t
        slot = 1 if prediction["is_right"] else 0
        vertices_out[index, slot] = camera_vertices
        joints[index, slot] = hand_joints
        valid[index, slot] = np.isfinite(camera_vertices).all() and np.isfinite(hand_joints).all()
    return vertices_out, joints, valid, rescaled


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
        window_intrinsics = window_input.get("intrinsics")
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
        window_intrinsics = None
    fx_per_frame = None
    if window_intrinsics is not None:
        fx_per_frame = []
        for matrix in window_intrinsics:
            value = None
            if matrix is not None:
                candidate = float(np.asarray(matrix, dtype=np.float64)[0, 0])
                value = candidate if np.isfinite(candidate) and candidate > 0 else None
            fx_per_frame.append(value)
        if len(fx_per_frame) != len(frame_ids):
            raise ValueError("intrinsics must provide one entry per frame")
    mapping = json.loads((prepared_dir / "mapping.json").read_text(encoding="utf-8"))
    if mapping.get("sequence") != sequence or mapping.get("window_id") != window_id or mapping.get("frame_ids") != frame_ids:
        raise ValueError("prepared PAD-Hand input does not match the requested manifest window")
    video = prepared_dir / "input.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    if int(mapping.get("context_frames", 0)) < 16:
        raise ValueError("PAD-Hand requires at least 16 contiguous context frames")

    output_dir = args.output_root / "pad_hand" / args.phase / output_window_id
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
        mano = MANO("RIGHT", device)
        side_paths = _split_wilor_by_side(wilor_npz, native_dir)
        with np.load(wilor_npz, allow_pickle=False) as source:
            native_count = source["vertices"].shape[0]
        all_vertices = np.full((native_count, 2, 778, 3), np.nan, dtype=np.float32)
        all_joints = np.full((native_count, 2, 21, 3), np.nan, dtype=np.float32)
        all_valid = np.zeros((native_count, 2), dtype=bool)
        rescaled_count = 0
        for side, side_path in enumerate(side_paths):
            predictions, _ = load_wilor_results(side_path)
            if len(predictions) != native_count:
                raise ValueError("PAD hand track length does not match the source video")
            refined = refine_with_pad_hand(predictions, model, mano, device)
            with np.load(side_path, allow_pickle=False) as source:
                focal = source["scaled_focal"]
            vertices, joints, valid, rescaled = _geometry_from_refined(
                refined, mano, fx_per_frame, focal)
            if valid[:, 1 - side].any():
                raise ValueError("PAD refinement wrote a hand into the wrong slot")
            all_vertices[:, side] = vertices[:, side]
            all_joints[:, side] = joints[:, side]
            all_valid[:, side] = valid[:, side]
            rescaled_count += rescaled
    finally:
        os.chdir(previous_cwd)

    indices = np.asarray(mapping["hand_indices_30fps"], dtype=np.int64)
    if indices.shape != (len(frame_ids),) or np.any(indices < 0) or np.any(indices >= len(all_joints)):
        raise ValueError("invalid PAD-Hand frame mapping")
    arrays = {
        "hand_joints_camera": np.nan_to_num(all_joints[indices]),
        "hand_vertices_camera": np.nan_to_num(all_vertices[indices]),
        "hand_markers_camera": np.nan_to_num(downsample_mano_vertices(all_vertices[indices])),
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
            "dataset": dataset,
            "sequence": sequence,
            "window_id": window_id,
            "frame_ids": frame_ids,
            "capabilities": {name: True for name in arrays},
            "scale_type": methods["pad_hand"]["scale_type"],
            "coordinate_space": "camera",
            "native_frame_indices": indices.tolist(),
            "hand_tracks": "independent_left_right_refinement",
            "valid_side_frame_count": all_valid[indices].sum(axis=0).astype(int).tolist(),
            "both_hands_valid_frame_count": int(all_valid[indices].all(axis=1).sum()),
            "metric_depth_rescale": (
                "cam_t[2] *= fx / scaled_focal, following WiLoR's official demo"
                if fx_per_frame is not None else
                "unavailable: no intrinsics supplied, absolute depth left on WiLoR's rendering focal length"
            ),
            "frames_with_intrinsic_rescale": int(rescaled_count),
            "intrinsic_rescale_count_unit": "hand-side frames",
        },
        arrays=arrays,
        run={"status": "success", "elapsed_seconds": time.perf_counter() - start},
        native_metadata={"source": "official PAD-Hand and WiLoR inference", "wilor_npz": str(wilor_npz)},
        native_arrays={"hand_joints_camera": all_joints, "hand_vertices_camera": all_vertices, "hand_valid": all_valid},
    )
    print(json.dumps({"output_dir": str(output_dir), "status": "success"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
