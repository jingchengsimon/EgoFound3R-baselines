#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from functools import lru_cache
import json
import sys
import tempfile
import time
from pathlib import Path

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
from egohandmetric_prompt.data import MediaRef, load_media_ref
from egohandmetric_prompt.inference_contracts import reconstruct_multirate_metric_scene_outputs
from egohandmetric_prompt.inference_hand_depth_scale import (
    HandDepthScaleOptions,
    apply_hand_depth_scale,
)
from egohandmetric_prompt.marker_runtime import MarkerRuntimeCollator
from formal_evaluation.common.io import (
    load_manifest,
    resolve_rgb_paths,
    select_manifest_window,
    write_comparison_output,
)
from formal_evaluation.common.schema import SCHEMA_VERSION
from formal_evaluation.datasets.window_inputs import load_window_input
from train_marker_model import _build_prediction_marker_forward_contract, _call_marker_model, _public_marker_outputs


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _invert_w2c(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(w2c).astype(np.float32)


def _multirate_scene_arrays(outputs, frame_map):
    # Official inference reconstruction already makes cameras/depth metric and clip-local.
    camera = _as_numpy(outputs["camera_pose_refined_high"])[0].copy()
    camera_valid = _as_numpy(outputs["camera_refined_valid_high"])[0].astype(bool)
    camera[~camera_valid] = np.nan
    indices = _as_numpy(frame_map.global_anchor_indices)[0].astype(np.int64)
    present = _as_numpy(frame_map.global_frame_present)[0].astype(bool)
    scale_valid = bool(_as_numpy(outputs["interpolation_scene_metric_scale_valid"])[0])
    selected = present & scale_valid
    values = []
    for key in ("intrinsics_global", "depth_global", "depth_conf_global"):
        source = _as_numpy(outputs[key])[0]
        if key != "intrinsics_global" and source.ndim == 4 and source.shape[-1] == 1:
            source = source[..., 0]
        dense = np.full((len(camera), *source.shape[1:]), np.nan, dtype=np.float32)
        dense[indices[selected]] = source[selected]
        values.append(dense)
    return camera, *values


def _camera_to_world(points: np.ndarray, camera_c2w: np.ndarray) -> np.ndarray:
    return np.einsum("tij,thpj->thpi", camera_c2w[:, :3, :3], points) + camera_c2w[
        :, None, None, :3, 3
    ]


INPUT_RESOLUTIONS_HW = {
    "384x512": (384, 512),
    "448x448": (448, 448),
    "512x512": (512, 512),
}


def _first_chunk(chunks: list[dict]) -> dict:
    return chunks[0]


def _load_frames(frame_paths: list[Path], *, target_hw: tuple[int, int], crop_config) -> torch.Tensor:
    """Use the label-independent center-crop path from the checkpoint training code."""
    if not hasattr(MarkerRuntimeCollator, "_should_apply_center_crop"):
        raise RuntimeError("EgoFound3R inference requires the label-independent training crop implementation")
    samples = []
    for path in frame_paths:
        samples.append({
            "rgb": load_media_ref(MediaRef(kind="path", path=str(path)), is_rgb=True),
            "hand_annos": [],
        })
    deterministic_crop = replace(
        crop_config,
        enabled=True,
        crop_probability=1.0,
        target_shapes=[list(target_hw)],
    )
    collator = MarkerRuntimeCollator(
        _first_chunk,
        target_height=target_hw[0],
        target_width=target_hw[1],
        random_crop_resize=deterministic_crop,
    )
    chunk = collator([{"samples": samples, "target_image_size_hw": target_hw}])
    return chunk["rgb"].unsqueeze(0)


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


def _verified_checkpoint_sha256(path: Path, expected: str) -> str:
    if not expected:
        raise ValueError("EgoFound3R requires a registered checkpoint_sha256")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"checkpoint SHA-256 mismatch: expected={expected}, actual={actual}")
    return actual


def parse_args(argv=None) -> argparse.Namespace:
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
    parser.add_argument("--global-stride", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--input-resolution", choices=tuple(INPUT_RESOLUTIONS_HW), required=True)
    return parser.parse_args(argv)


@lru_cache(maxsize=1)
def _runtime(config, checkpoint, backbone, expected_sha256, device, size, mtime_ns, height, width):
    """One immutable checkpoint per worker; retain no per-window predictions."""
    checkpoint_sha256 = _verified_checkpoint_sha256(Path(checkpoint), expected_sha256)
    project_config = _load_training_config_compat(Path(config))
    if project_config.marker_runtime.backend != "vggt_omega":
        raise RuntimeError("Unsupported backbone for this registered comparison adapter")
    configured_shapes = {
        tuple(int(value) for value in shape)
        for shape in project_config.marker_runtime.random_crop_resize.target_shapes
    }
    if (height, width) not in configured_shapes:
        raise ValueError(
            f"input resolution {(height, width)} is absent from training target_shapes "
            f"{sorted(configured_shapes)}"
        )
    project_config.marker_runtime.image_height = height
    project_config.marker_runtime.image_width = width
    project_config.marker_runtime.vggt_checkpoint_path = backbone
    if not torch.cuda.is_available():
        raise RuntimeError("EgoFound3R comparison requires CUDA")
    model = build_runtime_marker_model(project_config).to(device)
    load_marker_model_weights(
        model, Path(checkpoint), model_config=active_marker_model_config(project_config),
        resume_mode="weights", align_model_floating_dtype=True,
    )
    model.eval()
    return checkpoint_sha256, project_config, model, marker_model_floating_dtype(model)


def main(argv=None) -> None:
    args = parse_args(argv)
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
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(args.device)
    input_height, input_width = INPUT_RESOLUTIONS_HW[args.input_resolution]
    start = time.perf_counter()
    checkpoint_sha256, project_config, model, input_dtype = _runtime(
        str(args.config), str(args.checkpoint), str(args.backbone_checkpoint),
        method_config.get("checkpoint_sha256"), args.device,
        args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns,
        input_height, input_width,
    )
    global_stride = args.global_stride
    global_anchor_phase = global_stride // 2
    frames = _load_frames(
        frame_paths,
        target_hw=(input_height, input_width),
        crop_config=project_config.marker_runtime.random_crop_resize,
    ).to(device=device, dtype=input_dtype)
    with torch.inference_mode():
        forward_contract, _ = _build_prediction_marker_forward_contract(
            project_config=project_config,
            # These RGB windows have no padding or teacher/GT observations.
            batch={},
            images=frames,
            global_stride=global_stride,
            global_anchor_phase=global_anchor_phase,
        )
        outputs = _public_marker_outputs(
            _call_marker_model(model, frames, **forward_contract)
        )
        frame_map = forward_contract["multirate_frame_map"]
        query_map = forward_contract["temporal_interpolation_query_map"]
        outputs = reconstruct_multirate_metric_scene_outputs(
            outputs,
            global_frame_present=frame_map.global_frame_present,
        )
        outputs, hand_depth_scale = apply_hand_depth_scale(
            outputs,
            frame_map,
            query_map,
            30.0,
            HandDepthScaleOptions(),
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated()

    camera_w2c, intrinsics, depth, depth_confidence = _multirate_scene_arrays(
        outputs, frame_map
    )
    camera_c2w = _invert_w2c(camera_w2c)
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
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_role": method_config["checkpoint_role"],
        "model_compute_dtype": str(frames.dtype),
        "model_input_dtype": str(frames.dtype),
        "model_floating_dtypes_after_forward": sorted({
            str(value.dtype) for value in (*model.parameters(), *model.buffers())
            if value.is_floating_point()
        }),
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
        "temporal_fps": 30.0,
        "source_resolution_hw": [source_height, source_width],
        "input_resolution": args.input_resolution,
        "processed_resolution_hw": [input_height, input_width],
        "image_preprocessing": "training_marker_runtime_collator_label_independent_center_crop",
        "preprocessing_uses_hand_annotations": False,
        "preprocessing_random_crop": False,
        "preprocessing_center_crop": True,
        "runner_detail": {
            "hand_geometry": "native 21 joints and 195 MANO markers; no fabricated 778-vertex mesh",
            "world_hand_geometry": "derived from native camera-space hands and predicted camera_c2w",
            "metric_reconstruction": (
                "official inference_contracts reconstruction followed by "
                "official inference_hand_depth_scale fusion; outputs are consumed as metric once"
            ),
            "hand_depth_scale": hand_depth_scale,
            "scene_sampling": "native G depth/intrinsics only at registered global anchors; other frames are NaN/invalid",
            "hand_valid_semantics": "analytic Root translation validity per physical side",
            "image_preprocessing": (
                "checkpoint training MarkerRuntimeCollator with its label-independent centered "
                "aspect crop forced on deterministically and the selected training target shape"
            ),
        },
    }
    run = {
        "status": "success",
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
        "device": args.device,
        "frame_count": len(frame_ids),
        "provenance": {
            key: metadata[key] for key in (
                "source_commit", "source_tag", "inference_commit", "checkpoint",
                "checkpoint_sha256", "model_compute_dtype", "global_stride", "global_anchor_phase",
                "input_resolution", "processed_resolution_hw", "image_preprocessing",
            )
        },
    }
    native_arrays = {
        "camera_w2c": camera_w2c,
        "camera_pose_encoding_global": _as_numpy(outputs["camera_pose_encoding_global"])[0],
        "global_anchor_indices": _as_numpy(forward_contract["multirate_frame_map"].global_anchor_indices)[0],
        "metric_scale_factor": _as_numpy(outputs["interpolation_scene_metric_scale_factor"]),
        "metric_scale_factor_fused": _as_numpy(outputs["raw_to_meter_fused"]),
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
