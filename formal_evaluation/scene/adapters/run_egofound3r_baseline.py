#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from egohandmetric_prompt import (
    build_runtime_marker_model,
    load_marker_model_weights,
    load_project_config,
    marker_model_floating_dtype,
)
from egohandmetric_prompt.configs import active_marker_model_config
from egohandmetric_prompt.marker_runtime import reconstruct_metric_scale_outputs
from formal_evaluation.common.io import (
    load_manifest,
    resolve_rgb_paths,
    select_manifest_window,
    write_comparison_output,
)
from formal_evaluation.common.schema import SCHEMA_VERSION
from formal_evaluation.datasets.window_inputs import load_window_input
from train_marker_model import _build_prediction_marker_forward_contract, _call_marker_model


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _invert_w2c(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(w2c).astype(np.float32)


def _camera_to_world(points: np.ndarray, camera_c2w: np.ndarray) -> np.ndarray:
    return np.einsum("tij,thpj->thpi", camera_c2w[:, :3, :3], points) + camera_c2w[
        :, None, None, :3, 3
    ]


def _load_frames(frame_paths: list[Path], *, height: int, width: int) -> torch.Tensor:
    frames = []
    for path in frame_paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0)
    return torch.stack(frames).unsqueeze(0)


def _load_training_config_compat(config_path: Path):
    try:
        return load_project_config(config_path)
    except ValueError as error:
        retired_fields = {
            "interhand_contact_loss_weight",
            "interhand_marker_contact_loss_weight",
        }
        if not all(field in str(error) for field in retired_fields):
            raise
        retained_lines = [
            line
            for line in config_path.read_text(encoding="utf-8").splitlines()
            if line.split("=", 1)[0].strip() not in retired_fields
        ]
        with tempfile.TemporaryDirectory(prefix="egofound3r_eval_config_") as temp_dir:
            compatible_path = Path(temp_dir) / config_path.name
            compatible_path.write_text("\n".join(retained_lines) + "\n", encoding="utf-8")
            return load_project_config(compatible_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 EgoFound3R 并写入 comparison canonical 输出")
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--window-input", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--window-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--global-stride", type=int, choices=range(1, 6))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_input is not None:
        if args.manifest is not None or args.data_root is not None:
            raise ValueError("--window-input cannot be combined with --manifest/--data-root")
        window_input = load_window_input(args.window_input)
        sequence = str(window_input["sequence_id"])
        window_id = str(window_input["window_id"])
        frame_ids = [str(item) for item in window_input["frame_ids"]]
        frame_paths = [Path(str(item)) for item in window_input["rgb_paths"]]
        dataset = str(window_input["dataset"])
        output_window_id = str(window_input["cache_id"])
    else:
        if args.manifest is None or args.data_root is None:
            raise ValueError("provide --window-input or both --manifest and --data-root")
        manifest = load_manifest(args.manifest)
        sequence, window_id, frame_ids = select_manifest_window(
            manifest, phase=args.phase, sequence=args.sequence, window_id=args.window_id
        )
        frame_paths = resolve_rgb_paths(args.data_root, sequence, frame_ids)
        dataset = "h2o"
        output_window_id = window_id
    method_config = json.loads(args.methods_config.read_text(encoding="utf-8"))["methods"][
        "egofound3r"
    ]
    project_config = _load_training_config_compat(args.config)
    if project_config.marker_runtime.backend != "vggt_omega":
        raise ValueError(f"EgoFound3R 正式比较要求 vggt_omega backend，实际为 {project_config.marker_runtime.backend}")
    project_config.marker_runtime.vggt_checkpoint_path = str(args.backbone_checkpoint)
    global_stride = args.global_stride or int(active_marker_model_config(project_config).global_stride)
    global_anchor_phase = global_stride // 2

    if not torch.cuda.is_available():
        raise RuntimeError("EgoFound3R smoke/pilot 需要 CUDA")
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(args.device)
    start = time.perf_counter()

    model = build_runtime_marker_model(project_config).to(device)
    load_marker_model_weights(
        model,
        args.checkpoint,
        model_config=active_marker_model_config(project_config),
        resume_mode="weights",
        align_model_floating_dtype=True,
    )
    model.eval()
    frames = _load_frames(
        frame_paths,
        height=project_config.marker_runtime.image_height,
        width=project_config.marker_runtime.image_width,
    ).to(device=device, dtype=marker_model_floating_dtype(model))
    with torch.inference_mode():
        forward_contract, _ = _build_prediction_marker_forward_contract(
            project_config=project_config,
            batch=None,
            images=frames,
            global_stride=global_stride,
            global_anchor_phase=global_anchor_phase,
        )
        outputs = reconstruct_metric_scale_outputs(
            _call_marker_model(model, frames, **forward_contract)
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated()

    camera_w2c = _as_numpy(outputs["camera_pose"])[0]
    camera_c2w = _invert_w2c(camera_w2c)
    intrinsics = _as_numpy(outputs["intrinsics"])[0]
    depth = _as_numpy(outputs["depth"])[0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_confidence = _as_numpy(outputs["depth_conf"])[0]
    if depth_confidence.ndim == 4 and depth_confidence.shape[-1] == 1:
        depth_confidence = depth_confidence[..., 0]
    joints_camera = _as_numpy(outputs["dense_joint_xyz"])[0]
    markers_camera = _as_numpy(outputs["dense_vertex_xyz"])[0]
    presence_probability = _as_numpy(outputs["in_view_probability"])[0]
    root_translation_valid = outputs.get("root_translation_valid")
    if (
        not isinstance(root_translation_valid, torch.Tensor)
        or tuple(root_translation_valid.shape[:3]) != tuple(outputs["dense_joint_xyz"].shape[:3])
    ):
        raise ValueError(
            "EgoFound3R comparison requires root_translation_valid aligned with [B, T, side]"
        )
    hand_valid = _as_numpy(root_translation_valid)[0].astype(bool, copy=False)
    joint_visibility = torch.sigmoid(outputs["dense_joint_visibility_logits"])[0].detach().float().cpu().numpy()
    marker_visibility = torch.sigmoid(outputs["dense_vertex_visibility_logits"])[0].detach().float().cpu().numpy()
    joint_contact = torch.sigmoid(outputs["dense_joint_contact_logits"])[0].detach().float().cpu().numpy()
    marker_contact = torch.sigmoid(outputs["dense_vertex_contact_logits"])[0].detach().float().cpu().numpy()

    arrays = {
        "camera_c2w": camera_c2w,
        "camera_valid": np.isfinite(camera_c2w).reshape(len(frame_ids), -1).all(axis=-1),
        "intrinsics": intrinsics,
        "intrinsics_valid": np.isfinite(intrinsics).reshape(len(frame_ids), -1).all(axis=-1),
        "depth": depth,
        "depth_valid": np.isfinite(depth) & (depth > 0),
        "depth_confidence": depth_confidence,
        "hand_joints_camera": joints_camera,
        "hand_joints_world": _camera_to_world(joints_camera, camera_c2w),
        "hand_markers_camera": markers_camera,
        "hand_markers_world": _camera_to_world(markers_camera, camera_c2w),
        "hand_valid": hand_valid,
        "hand_presence_probability": presence_probability,
        "hand_visibility": joint_visibility,
        "marker_visibility": marker_visibility,
        "joint_contact_probability": joint_contact,
        "marker_contact_probability": marker_contact,
    }
    output_dir = args.output_root / "egofound3r" / args.phase / output_window_id
    with Image.open(frame_paths[0]) as image:
        source_width, source_height = image.size
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": "egofound3r",
        "source": method_config["source"],
        "source_commit": method_config["source_commit"],
        "source_tag": method_config.get("source_tag"),
        "inference_commit": method_config.get("inference_commit"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": method_config.get("checkpoint_sha256"),
        "checkpoint_role": method_config["checkpoint_role"],
        "model_compute_dtype": str(marker_model_floating_dtype(model)),
        "global_stride": global_stride,
        "global_anchor_phase": global_anchor_phase,
        "phase": args.phase,
        "dataset": dataset,
        "sequence": sequence,
        "window_id": window_id,
        "frame_ids": frame_ids,
        "capabilities": {name: True for name in arrays},
        "scale_type": "metric",
        "camera_convention": "OpenCV x-right y-down z-forward; camera_c2w maps camera to world",
        "units": "meters",
        "source_resolution_hw": [source_height, source_width],
        "processed_resolution_hw": [
            project_config.marker_runtime.image_height,
            project_config.marker_runtime.image_width,
        ],
        "runner_detail": {
            "hand_geometry": "native 21 joints and 195 MANO markers; no fabricated 778-vertex mesh",
            "world_hand_geometry": "derived from native camera-space hands and predicted camera_c2w",
            "metric_reconstruction": (
                "reconstruct_metric_scale_outputs applies only to VGGT depth/camera; "
                "Hand XYZ is already metric"
            ),
            "hand_valid_semantics": "analytic Root translation validity per physical side",
        },
    }
    run = {
        "status": "success",
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
        "device": args.device,
        "frame_count": len(frame_ids),
    }
    native_arrays = {
        "camera_w2c": camera_w2c,
        "camera_pose_encoding": _as_numpy(outputs["camera_pose_encoding"])[0],
        "metric_scale_factor": _as_numpy(outputs["metric_scale_factor"]),
    }
    write_comparison_output(
        output_dir,
        metadata=metadata,
        arrays=arrays,
        run=run,
        native_metadata={
            "official_output_keys_retained": sorted(native_arrays),
            "canonical_derivations": {
                "camera_c2w": "inverse of native camera_w2c",
                "hand_joints_world/hand_markers_world": "predicted camera_c2w applied to native camera-space geometry",
            },
        },
        native_arrays=native_arrays,
    )
    print(json.dumps({"output_dir": str(output_dir), **run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
