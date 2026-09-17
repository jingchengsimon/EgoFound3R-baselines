#!/usr/bin/env python3
"""Canonical RGB-only EgoForce adapter.

The adapter intentionally consumes only ``window_input.json`` RGB paths and camera
calibration.  It never opens the paired geometry paths.  Inputs are anisotropically
resized to 256x256 before official inference and their intrinsics are scaled with the
same pixel transform.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from formal_evaluation.common.io import write_comparison_output
from formal_evaluation.hand.adapters.yolo_rgb import configure_rgb_yolo_input
from formal_evaluation.common.marker_vertices import MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195
from formal_evaluation.common.schema import SCHEMA_VERSION
from formal_evaluation.datasets.window_inputs import WINDOW_INPUT_VERSION


OFFICIAL_SOURCE_COMMIT = "480ffb358516d0d7f971ec9da99b7bb07635731e"
OFFICIAL_CHECKPOINT_SHA256 = "722fe84ba2b6ab3569c3a95000c61a9a77c7b3a41cc76585f891b8bc4fe82870"
INPUT_HW = (256, 256)
MARKER_IDS_195 = np.asarray(MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195, dtype=np.int64)


def _verified_sha256(path: Path, expected: str) -> str:
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
        raise ValueError("--checkpoint-sha256 must be a SHA-256 hex digest")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if digest != expected.lower():
        raise ValueError(f"checkpoint SHA-256 mismatch: expected={expected}, actual={digest}")
    return digest


def _resize_rgb_and_intrinsics(rgb: np.ndarray, intrinsics: Any) -> tuple[np.ndarray, np.ndarray]:
    """Resize RGB to the shared 256x256 input and apply its pixel transform to K."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB HxWx3, got {image.shape}")
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("EgoForce requires a finite positive 3x3 intrinsic matrix for every frame")
    height, width = image.shape[:2]
    resized = np.asarray(Image.fromarray(image, mode="RGB").resize((INPUT_HW[1], INPUT_HW[0]), Image.Resampling.BILINEAR))
    scale = np.array([[INPUT_HW[1] / width, 0.0, 0.0], [0.0, INPUT_HW[0] / height, 0.0], [0.0, 0.0, 1.0]])
    return resized, (scale @ K).astype(np.float32)


def _camera_model(source_root: Path, K: np.ndarray):
    sys.path.insert(0, str(source_root))
    from camera_models import PinholeCameraModel

    return PinholeCameraModel(np.array([K[0, 0], K[1, 1]]), np.array([K[0, 2], K[1, 2]]), INPUT_HW[1], INPUT_HW[0])


def _import_official_inference(source_root: Path):
    """Load the official numerical path without TensorRT or visualization imports."""
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "demo"))
    source_path = source_root / "demo" / "inference.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    keep: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "torch_tensorrt" for alias in node.names):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "renderer":
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name) and func.value.value.id == "torch_tensorrt":
                continue
        keep.append(node)
    tree.body = keep
    module = types.ModuleType("_egoforce_inference_without_tensorrt")
    module.__file__ = str(source_path)
    exec(compile(tree, str(source_path), "exec"), module.__dict__)
    return module


def _verified_source_commit(source_root: Path) -> str:
    actual = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if actual != OFFICIAL_SOURCE_COMMIT:
        raise ValueError(f"EgoForce source commit mismatch: expected={OFFICIAL_SOURCE_COMMIT}, actual={actual}")
    return actual


def _build_runner(source_root: Path, checkpoint: Path, device_name: str):
    """Use official detector/loader/solver methods, replacing only TensorRT model compilation."""
    import torch

    official = _import_official_inference(source_root)

    class NonTensorRTInference(official.Inference):
        def __init__(self):
            self.device = torch.device(device_name)
            self.undistort_inp = True
            self.enable_kalman_filter = True
            self.kalman_filter_kwargs = {"q_pos": 0.001, "q_vel": 1e-05, "r_meas": 0.001}
            self.kalman_filter_freq = 30.0
            self.box_inferencer = official.DetInferencer(
                str(source_root / "demo" / "rtmdet_tiny_8xb32-300e_combined_cutmix.py"),
                weights=official.cfg.DETECTION.HAND_ARM_PATH,
                device=self.device,
            )
            self.box_inferencer.model = official.optimize_mmdet_model_for_inference(self.box_inferencer.model.eval().half())
            self.hand_detector = official.YOLO(official.cfg.DETECTION.HAND_PATH, task="pose")
            configure_rgb_yolo_input(self.hand_detector)
            self.classes = ["left_forearm", "right_forearm", "left_hand", "right_hand"]
            official.init_tracking_defaults(self)
            self.model = official.HALO(official.cfg)
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device), strict=True)
            self.model = self.model.to(self.device).half().eval()
            self.limb_model = official.LimbModel(official.cfg, device=self.device, use_pose_pca=False, n_components=5)
            self.camera_model = self.left_dataset = self.right_dataset = None
            self.left_kalman_filter = self.right_kalman_filter = None
            self.set_kalman_filter_frequency(self.kalman_filter_freq)
            self.renderer = self._last_renderer_meta = None
            self.frame_index = 0
            self.stream_det, self.stream_yolo = torch.cuda.Stream(), torch.cuda.Stream()
            self.grouped_hand_track_ids = {"left": None, "right": None}
            self.grouped_hand_track_misses = {"left": 0, "right": 0}
            self.grouped_hand_track_max_misses = 2

    return NonTensorRTInference()


def _canonical_arrays(outputs: dict[str, Any], frame_count: int, frame_index: int, arrays: dict[str, np.ndarray]) -> None:
    vertices = np.asarray(outputs["pred_vertices"], dtype=np.float32)
    joints = np.asarray(outputs["pred_j3d"], dtype=np.float32)
    valid = np.asarray(outputs["visible_hand"], dtype=bool).reshape(2)
    if vertices.shape != (2, 778, 3) or joints.shape != (2, 21, 3):
        raise ValueError(f"unexpected EgoForce output shapes: vertices={vertices.shape}, joints={joints.shape}")
    arrays["hand_valid"][frame_index] = valid
    arrays["hand_vertices_camera"][frame_index, valid] = vertices[valid]
    arrays["hand_joints_camera"][frame_index, valid] = joints[valid]
    arrays["hand_markers_camera"][frame_index, valid] = vertices[valid][:, MARKER_IDS_195]


def _empty_arrays(frame_count: int) -> dict[str, np.ndarray]:
    return {
        "hand_joints_camera": np.full((frame_count, 2, 21, 3), np.nan, dtype=np.float32),
        "hand_vertices_camera": np.full((frame_count, 2, 778, 3), np.nan, dtype=np.float32),
        "hand_markers_camera": np.full((frame_count, 2, 195, 3), np.nan, dtype=np.float32),
        "hand_valid": np.zeros((frame_count, 2), dtype=bool),
    }


def _load_rgb_window_input(path: Path) -> dict[str, object]:
    """Load method-neutral RGB/K input without opening or requiring GT geometry."""
    record = json.loads(path.read_text(encoding="utf-8"))
    frame_ids, rgb_paths = record.get("frame_ids"), record.get("rgb_paths")
    if record.get("window_input_version") != WINDOW_INPUT_VERSION or not isinstance(frame_ids, list) or not isinstance(rgb_paths, list):
        raise ValueError(f"unsupported RGB window input: {path}")
    if not frame_ids or len(frame_ids) != len(rgb_paths) or not all(Path(str(value)).is_file() for value in rgb_paths):
        raise ValueError(f"invalid RGB paths in window input: {path}")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official EgoForce as a canonical hand baseline")
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--window-input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=OFFICIAL_CHECKPOINT_SHA256)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("--source-root and --checkpoint must exist")
    record = _load_rgb_window_input(args.window_input)
    frame_ids = [str(value) for value in record["frame_ids"]]
    intrinsics = record.get("intrinsics")
    if not isinstance(intrinsics, list) or len(intrinsics) != len(frame_ids):
        raise ValueError("window input lacks one intrinsic matrix per frame")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("EgoForce baseline requires CUDA")
    source_commit = _verified_source_commit(args.source_root)
    checkpoint_sha256 = _verified_sha256(args.checkpoint, args.checkpoint_sha256)
    runner = _build_runner(args.source_root, args.checkpoint, args.device)
    arrays = _empty_arrays(len(frame_ids))
    start = time.perf_counter()
    for index, (path, K) in enumerate(zip(record["rgb_paths"], intrinsics, strict=True)):
        with Image.open(path) as handle:
            rgb = np.asarray(ImageOps.exif_transpose(handle).convert("RGB"))
        resized, K_resized = _resize_rgb_and_intrinsics(rgb, K)
        runner.set_camera_model(_camera_model(args.source_root, K_resized), undistort_inp=True)
        _canonical_arrays(runner.run_outputs(resized, args.device), len(frame_ids), index, arrays)
    torch.cuda.synchronize()
    output_dir = args.output_root / "egoforce" / args.phase / str(record["cache_id"])
    metadata = {
        "schema_version": SCHEMA_VERSION, "method": "egoforce", "phase": args.phase,
        "dataset": str(record["dataset"]), "sequence": str(record["sequence_id"]),
        "window_id": str(record["window_id"]), "frame_ids": frame_ids,
        "capabilities": {name: True for name in arrays}, "scale_type": "metric_camera",
        "detector_color_input": "RGB_to_BGR_at_YOLO_predict_only",
        "camera_convention": "OpenCV camera frame (x-right, y-down, z-forward)", "units": "meters",
        "processed_resolution_hw": list(INPUT_HW),
        "runner_detail": {"official_source_commit": source_commit, "checkpoint_sha256": checkpoint_sha256,
                          "image_preprocessing": "shared_default_anisotropic_resize_256x256_with_K_scale",
                          "camera_input": "rectified-pinhole", "tensor_rt": "not imported", "kalman_filter": "official_default_enabled"},
    }
    write_comparison_output(output_dir, metadata=metadata, arrays=arrays,
                            run={"status": "success", "elapsed_seconds": time.perf_counter() - start, "frame_count": len(frame_ids), "device": args.device},
                            native_metadata={"canonical_derivations": {"hand_markers_camera": "195 fixed MANO vertex subset"}})
    print(json.dumps({"output_dir": str(output_dir), "valid_hands": int(arrays["hand_valid"].sum())}))


if __name__ == "__main__":
    main()
