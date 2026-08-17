#!/usr/bin/env python3
"""WiLoR canonical adapter for EgoFound3R baseline comparison.

Uses official YOLO hand detector (conf=0.3), ViTDetDataset, cam_crop_to_full.
Outputs camera-space 21 joints, 778 vertices, and 195 marker subset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from PIL import Image, ImageOps

from formal_evaluation.common.io import (
    load_manifest,
    resolve_rgb_paths,
    select_manifest_window,
    write_comparison_output,
)
from formal_evaluation.common.marker_vertices import MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195
from formal_evaluation.common.schema import SCHEMA_VERSION
from formal_evaluation.datasets.window_inputs import load_window_input

# ---------------------------------------------------------------------------
# Mock pyrender/OpenGL before WiLoR imports (headless server, no rendering needed)
# ---------------------------------------------------------------------------
for _mod_name in ("pyrender", "OpenGL", "OpenGL.GL", "OpenGL.platform"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

MARKER_IDS_195 = np.array(MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195, dtype=np.int64)

# ---------------------------------------------------------------------------
# WiLoR inference
# ---------------------------------------------------------------------------

# Canonical hand slots: 0=left, 1=right
_SLOT_LEFT = 0
_SLOT_RIGHT = 1


def _run_wilor(
    frame_paths: list[Path],
    source_root: Path,
    checkpoint: Path,
    detector_path: Path,
    device_name: str,
    rescale_factor: float = 2.0,
):
    """Run WiLoR on a list of frames, return canonical arrays."""
    import cv2
    import torch

    sys.path.insert(0, str(source_root))
    from wilor.models import load_wilor
    from wilor.utils import recursive_to
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils.renderer import cam_crop_to_full
    from ultralytics import YOLO

    device = torch.device(device_name)

    # Load model (load_wilor uses relative ./mano_data/ paths, must cwd to source_root)
    import os
    source_root_abs = source_root.resolve()
    prev_cwd = os.getcwd()
    os.chdir(source_root_abs)
    cfg_path = source_root_abs / "pretrained_models" / "model_config.yaml"
    model, model_cfg = load_wilor(str(checkpoint.resolve()), str(cfg_path))
    os.chdir(prev_cwd)
    model = model.to(device).eval()

    # Load detector
    detector = YOLO(str(detector_path))
    detector.to(device)

    frame_count = len(frame_paths)
    joints_out = np.zeros((frame_count, 2, 21, 3), dtype=np.float32)
    vertices_out = np.zeros((frame_count, 2, 778, 3), dtype=np.float32)
    hand_valid_out = np.zeros((frame_count, 2), dtype=bool)
    detection_meta = []  # per-frame metadata

    for fi, frame_path in enumerate(frame_paths):
        img_cv2 = cv2.imread(str(frame_path))
        if img_cv2 is None:
            detection_meta.append({"frame": frame_path.name, "error": "cv2.imread returned None"})
            continue

        # Official YOLO detection, conf=0.3
        detections = detector(img_cv2, conf=0.3, verbose=False)[0]

        bboxes = []
        is_right_list = []
        confidences = []
        for det in detections:
            box_data = det.boxes.data.cpu().detach().squeeze().numpy()
            cls = int(det.boxes.cls.cpu().detach().squeeze().item())
            conf = float(det.boxes.conf.cpu().detach().squeeze().item())
            bboxes.append(box_data[:4].tolist())
            is_right_list.append(cls)
            confidences.append(conf)

        if len(bboxes) == 0:
            detection_meta.append({"frame": frame_path.name, "detections": 0})
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right_list)
        confs = np.array(confidences)

        # Deterministic selection: for each side, pick highest confidence
        # right==1 → right hand, right==0 → left hand
        frame_meta = {"frame": frame_path.name, "detections": len(bboxes)}

        for side_class, slot in [(0, _SLOT_LEFT), (1, _SLOT_RIGHT)]:
            side_mask = right == side_class
            if not side_mask.any():
                continue
            side_indices = np.where(side_mask)[0]
            # Pick highest confidence among same-side detections
            best_idx = side_indices[np.argmax(confs[side_indices])]
            if len(side_indices) > 1:
                frame_meta[f"slot_{slot}_multi_det"] = int(len(side_indices))
                frame_meta[f"slot_{slot}_selected_conf"] = float(confs[best_idx])

            # Run WiLoR on this single detection
            sel_boxes = boxes[best_idx : best_idx + 1]
            sel_right = right[best_idx : best_idx + 1]

            dataset = ViTDetDataset(
                model_cfg, img_cv2, sel_boxes, sel_right,
                rescale_factor=rescale_factor, fp16=False,
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                # Post-process following official demo.py logic
                multiplier = 2 * batch["right"] - 1  # +1 for right, -1 for left
                pred_cam = out["pred_cam"].clone()
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = (
                    model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                )
                pred_cam_t_full = (
                    cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                    .detach()
                    .cpu()
                    .numpy()
                )

                verts = out["pred_vertices"][0].detach().cpu().numpy()  # (778, 3)
                joints = out["pred_keypoints_3d"][0].detach().cpu().numpy()  # (21, 3)
                is_right_val = batch["right"][0].cpu().numpy()

                # Flip x for left hand (official convention)
                verts[:, 0] = (2 * is_right_val - 1) * verts[:, 0]
                joints[:, 0] = (2 * is_right_val - 1) * joints[:, 0]

                # Translate to full camera space
                cam_t = pred_cam_t_full[0]  # (3,)
                verts = verts + cam_t[None, :]
                joints = joints + cam_t[None, :]

                joints_out[fi, slot] = joints
                vertices_out[fi, slot] = verts
                hand_valid_out[fi, slot] = True

        detection_meta.append(frame_meta)

    # Extract 195 marker subset from vertices
    markers_out = vertices_out[:, :, MARKER_IDS_195, :]  # (T, 2, 195, 3)

    arrays = {
        "hand_joints_camera": joints_out,
        "hand_vertices_camera": vertices_out,
        "hand_markers_camera": markers_out,
        "hand_valid": hand_valid_out,
    }
    native = {}  # No extra native arrays needed beyond canonical
    detail = {
        "detector": "official YOLO, conf=0.3",
        "dataset": "official ViTDetDataset",
        "cam_crop_to_full": "official",
        "rescale_factor": rescale_factor,
        "multi_detection_rule": "highest detector confidence per side",
        "hand_slot_convention": "slot 0=left, slot 1=right",
        "coordinate": "OpenCV camera frame (x-right, y-down, z-forward), full-image",
        "per_frame_detection_log": detection_meta,
    }
    # Resolution: WiLoR processes per-crop; report original resolution
    img0 = Image.open(frame_paths[0])
    img0 = ImageOps.exif_transpose(img0)
    orig_hw = (img0.height, img0.width)
    return arrays, native, orig_hw, detail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _original_resolution(frame_path: Path) -> tuple[int, int]:
    with Image.open(frame_path) as image:
        width, height = ImageOps.exif_transpose(image).size
    return height, width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 WiLoR baseline 并写入 canonical 输出")
    parser.add_argument("--phase", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--window-input", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--window-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rescale-factor", type=float, default=2.0)
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
            manifest, phase=args.phase, sequence=args.sequence, window_id=args.window_id,
        )
        frame_paths = resolve_rgb_paths(args.data_root, sequence, frame_ids)
        dataset = "h2o"
        output_window_id = window_id
    methods = json.loads(args.methods_config.read_text(encoding="utf-8"))["methods"]
    method_config = methods["wilor"]
    output_dir = args.output_root / "wilor" / args.phase / output_window_id

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("WiLoR baseline 需要 CUDA")
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    arrays, native_arrays, processed_resolution, runner_detail = _run_wilor(
        frame_paths,
        args.source_root,
        args.checkpoint,
        args.detector,
        args.device,
        rescale_factor=args.rescale_factor,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated()

    capabilities = {name: True for name in arrays}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": "wilor",
        "source": method_config["source"],
        "source_commit": method_config["source_commit"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_id": method_config.get("checkpoint_id"),
        "phase": args.phase,
        "dataset": dataset,
        "sequence": sequence,
        "window_id": window_id,
        "frame_ids": frame_ids,
        "capabilities": capabilities,
        "scale_type": method_config["scale_type"],
        "camera_convention": "OpenCV x-right y-down z-forward; camera-space full-image coordinates",
        "units": "metric (camera space)",
        "source_resolution_hw": list(_original_resolution(frame_paths[0])),
        "processed_resolution_hw": list(processed_resolution),
        "runner_detail": runner_detail,
    }
    run = {
        "status": "success",
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
        "device": args.device,
        "frame_count": len(frame_ids),
    }
    native_metadata = {
        "official_output_keys_retained": [],
        "canonical_derivations": {
            "hand_joints_camera": "pred_keypoints_3d + cam_crop_to_full translation",
            "hand_vertices_camera": "pred_vertices + cam_crop_to_full translation",
            "hand_markers_camera": "195 vertex subset of hand_vertices_camera",
            "hand_valid": "True only when YOLO detects corresponding hand side",
        },
    }
    write_comparison_output(
        output_dir,
        metadata=metadata,
        arrays=arrays,
        run=run,
        native_metadata=native_metadata,
        native_arrays=native_arrays if native_arrays else None,
    )
    print(json.dumps({"output_dir": str(output_dir), **run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
