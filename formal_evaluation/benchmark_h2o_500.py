#!/usr/bin/env python3
"""Isolated, in-memory 500-frame H2O inference benchmark for available baselines."""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path


METHODS = (
    "egofound3r", "wilor", "hawor", "dyn_hamr", "pad_hand", "reviv4d",
    "s2contact", "contactopt", "vggt", "pi3", "da3_large_1_1",
    "lingbot_map_long", "vggt_omega", "interactvlm",
)


def _frames(data_root: Path) -> tuple[str, list[Path]]:
    candidates = sorted(
        rgb.parent.parent.relative_to(data_root).as_posix()
        for rgb in data_root.glob("subject4_ego/*/*/cam4/rgb")
        if sum(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in rgb.iterdir()) >= 500
    )
    if not candidates:
        raise RuntimeError("no H2O test sequence has 500 frames")
    sequence = random.Random(0).choice(candidates)
    paths = sorted(
        path for path in (data_root / sequence / "cam4/rgb").iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[:500]
    if len(paths) != 500:
        raise RuntimeError(f"{sequence} has only {len(paths)} selected frames")
    return sequence, paths


def _chunks(items: list, size: int = 12) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _parameter_counts(modules: dict[str, object]) -> tuple[dict[str, int], int]:
    import torch

    per_module: dict[str, int] = {}
    seen: set[tuple[object, ...]] = set()
    total = 0
    for name, module in modules.items():
        count = 0
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            count += parameter.numel()
            storage = parameter.untyped_storage()
            key = (parameter.device.type, parameter.device.index, storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += parameter.numel()
        per_module[name] = count
    return per_module, total


def _timed(run_once, *, torch) -> dict[str, object]:
    for _ in range(2):
        run_once()
        torch.cuda.synchronize()
    seconds = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    ordered = sorted(seconds)
    p90_index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.9 + 0.999999))
    median = statistics.median(seconds)
    return {
        "trial_seconds": seconds,
        "median_seconds": median,
        "mean_seconds": statistics.fmean(seconds),
        "p90_seconds": ordered[p90_index],
        "output_frames": 500,
        "output_fps": 500.0 / median,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def _load_vggt(paths: list[Path], device: str):
    import sys
    import torch

    sys.path.insert(0, "/mnt/workspace/sjc/EgoFound3R-baselines/VGGT")
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    checkpoint = Path("/mnt/workspace/sjc/models/pretrained/VGGT-1B/model.pt")
    model = VGGT()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    inputs = load_and_preprocess_images([str(path) for path in paths], mode="crop")
    inputs = [chunk.to(device) for chunk in _chunks(inputs)]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    def forward():
        for batch in inputs:
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
                output = model(batch)
            del output

    return {"main_model": model}, forward, {"strategy": "42 chunks: 41x12 + 1x8; no overlap", "model_input_frames": 500}


def _load_wilor(paths: list[Path], device: str):
    import os
    import sys
    import time
    from unittest.mock import MagicMock

    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    for name in ("pyrender", "OpenGL", "OpenGL.GL", "OpenGL.platform"):
        sys.modules.setdefault(name, MagicMock())
    root = Path("/mnt/workspace/sjc/EgoFound3R-baselines/WiLoR")
    sys.path.insert(0, str(root))
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to

    previous = Path.cwd()
    os.chdir(root)
    try:
        model, config = load_wilor(
            "/mnt/workspace/sjc/models/pretrained/WiLoR/wilor_final.ckpt",
            str(root / "pretrained_models/model_config.yaml"),
        )
    finally:
        os.chdir(previous)
    model = model.to(device).eval()
    detector = YOLO("/mnt/workspace/sjc/models/pretrained/WiLoR/detector.pt")
    detector.to(device)
    frames = [cv2.imread(str(path)) for path in paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("cv2 failed to decode a selected H2O frame")
    # Detector outputs determine official crop inputs.  This prepass is input
    # preparation only; each timed trial still invokes the actual detector.
    samples = []
    for image, result in zip(frames, detector(frames, conf=0.3, verbose=False), strict=True):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        handedness = boxes.cls.detach().cpu().numpy().astype(np.int64)
        confidence = boxes.conf.detach().cpu().numpy()
        for side in (0, 1):
            candidates = np.flatnonzero(handedness == side)
            if candidates.size == 0:
                continue
            index = candidates[np.argmax(confidence[candidates])]
            dataset = ViTDetDataset(config, image, xyxy[index : index + 1], handedness[index : index + 1], rescale_factor=2.0, fp16=False)
            samples.append(dataset[0])
    if not samples:
        raise RuntimeError("official detector produced no hand crops on selected 500 frames")
    batches = [recursive_to(batch, device) for batch in DataLoader(samples, batch_size=16, shuffle=False, num_workers=0)]

    def detector_forward():
        outputs = detector(frames, conf=0.3, verbose=False)
        del outputs

    def hand_forward():
        for batch in batches:
            with torch.inference_mode():
                output = model(batch)
            del output

    def timed():
        for _ in range(2):
            detector_forward()
            torch.cuda.synchronize()
            hand_forward()
            torch.cuda.synchronize()
        totals, detectors, hands = [], [], []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            detector_forward()
            torch.cuda.synchronize()
            detector_end = time.perf_counter()
            hand_forward()
            torch.cuda.synchronize()
            end = time.perf_counter()
            totals.append(end - start)
            detectors.append(detector_end - start)
            hands.append(end - detector_end)
        median = statistics.median(totals)
        return {
            "trial_seconds": totals,
            "median_seconds": median,
            "mean_seconds": statistics.fmean(totals),
            "p90_seconds": max(totals),
            "output_frames": 500,
            "output_fps": 500.0 / median,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "stage_trial_seconds": {"detector": detectors, "wilor_main_model": hands},
            "stage_median_seconds": {"detector": statistics.median(detectors), "wilor_main_model": statistics.median(hands)},
        }

    return {"detector": detector.model, "wilor_main_model": model}, timed, {
        "strategy": "500 independently detected frames; one crop per detected hand side; WiLoR batches of 16",
        "model_input_frames": 500,
        "timing_kind": "complete neural pipeline: detector + WiLoR; detection-dependent CPU crop preparation excluded",
        "prepared_hand_crops": len(samples),
    }


def _load_egofound3r(paths: list[Path], device: str):
    import sys

    import cv2
    import torch

    root = Path("/mnt/workspace/sjc/EgoFound3R_dev_dataloader_shm_compat_20260808")
    config = root / "configs/h2o_contact_or_conf_flow_workers4_step006099_100steps_20260809.toml"
    checkpoint = Path("/mnt/workspace/sjc/EgoFound3R_dev/outputs/experiments/step_006099_newflow.pt")
    sys.path.insert(0, str(root))
    from egohandmetric_prompt import build_runtime_marker_model, load_marker_model_weights, load_project_config
    from egohandmetric_prompt.marker_runtime import reconstruct_metric_scale_outputs
    from infer_marker_video import _build_windows, _inference_egocentric

    project_config = load_project_config(config)
    height, width = project_config.marker_runtime.image_height, project_config.marker_runtime.image_width
    frames = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cv2 failed to decode {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0))
    windows = _build_windows(len(frames), chunk_size=12, stride=12)
    if len(windows) != 42 or sum(len(window) for window in windows) != 504:
        raise RuntimeError(f"unexpected 500-frame window plan: {len(windows)} windows, {sum(map(len, windows))} model frames")
    inputs = [torch.stack([frames[index] for index in window]).unsqueeze(0).to(device) for window in windows]
    model = build_runtime_marker_model(project_config).to(device)
    load_marker_model_weights(model, checkpoint)
    model.eval()
    egocentric = _inference_egocentric(project_config)

    def forward():
        for batch in inputs:
            with torch.inference_mode():
                output = reconstruct_metric_scale_outputs(model(batch, inference_egocentric=egocentric))
            del output

    return {"marker_runtime_model": model}, forward, {
        "strategy": "42 windows: 41x12 stride 12 plus final 12-frame window [488..499], overlap 4; outputs reassembled by frame index",
        "model_input_frames": 504,
        "window_count": 42,
        "timing_kind": "complete EgoFound3R model forward plus metric-scale output reconstruction; excludes RGB read/resize, aggregation, metrics, saving, and visualization",
    }


def _load_pi3(paths: list[Path], device: str):
    import sys
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import load_file
    from torchvision.transforms.functional import pil_to_tensor

    sys.path.insert(0, "/mnt/workspace/sjc/EgoFound3R-baselines/Pi3")
    from pi3.models.pi3 import Pi3

    checkpoint = Path("/mnt/workspace/sjc/models/pretrained/Pi3/model.safetensors")
    model = Pi3()
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    model = model.to(device).eval()
    sample = ImageOps.exif_transpose(Image.open(paths[0])).convert("RGB")
    width0, height0 = sample.size
    scale = (255_000 / (width0 * height0)) ** 0.5
    grid_w, grid_h = round(width0 * scale / 14), round(height0 * scale / 14)
    while (grid_w * 14) * (grid_h * 14) > 255_000:
        if grid_w / grid_h > width0 / height0:
            grid_w -= 1
        else:
            grid_h -= 1
    size = (max(1, grid_w) * 14, max(1, grid_h) * 14)
    inputs = [
        pil_to_tensor(ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize(size, Image.Resampling.LANCZOS)).float() / 255.0
        for path in paths
    ]
    inputs = [torch.stack(chunk).to(device) for chunk in _chunks(inputs)]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    def forward():
        for batch in inputs:
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
                output = model(batch[None])
            del output

    return {"main_model": model}, forward, {"strategy": "42 chunks: 41x12 + 1x8; no overlap", "model_input_frames": 500}


def _load_da3(paths: list[Path], device: str):
    import sys
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    sys.path.insert(0, "/mnt/workspace/sjc/EgoFound3R-baselines/Depth-Anything-3/src")
    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.io.input_processor import InputProcessor

    root = Path("/mnt/workspace/sjc/models/pretrained/Depth-Anything-3/DA3-LARGE-1.1")
    config = json.loads((root / "config.json").read_text())["config"]
    model = create_object(OmegaConf.create(config))
    wrapped = load_file(str(root / "model.safetensors"), device="cpu")
    state = {key.removeprefix("model."): value for key, value in wrapped.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    disallowed = [key for key in missing if not key.startswith("head.scratch.output_conv2_aux.")]
    if disallowed or unexpected:
        raise RuntimeError(f"checkpoint mismatch missing={disallowed} unexpected={unexpected}")
    model = model.to(device).eval()
    inputs, _, _ = InputProcessor()([str(path) for path in paths], process_res=504, process_res_method="upper_bound_resize", num_workers=8)
    inputs = [chunk.to(device) for chunk in _chunks(inputs)]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def forward():
        for batch in inputs:
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                output = model(batch[None])
            del output

    return {"main_model": model}, forward, {"strategy": "42 chunks: 41x12 + 1x8; no overlap", "model_input_frames": 500}


def _load_vggt_omega(paths: list[Path], device: str):
    import sys
    import torch

    sys.path.insert(0, "/mnt/workspace/sjc/EgoFound3R-baselines/vggt-omega")
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    checkpoint = Path("/mnt/workspace/sjc/models/pretrained/VGGT-Omega/vggt_omega_1b_512.pt")
    model = VGGTOmega().to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    inputs = load_and_preprocess_images([str(path) for path in paths], mode="balanced", image_resolution=512, patch_size=16)
    inputs = [chunk.to(device) for chunk in _chunks(inputs)]

    def forward():
        for batch in inputs:
            with torch.inference_mode():
                output = model(batch)
            del output

    return {"main_model": model}, forward, {"strategy": "42 chunks: 41x12 + 1x8; no overlap", "model_input_frames": 500}


def _load_lingbot(paths: list[Path], device: str):
    import sys
    import torch

    root = Path("/mnt/workspace/sjc/EgoFound3R-baselines/lingbot-map")
    sys.path.insert(0, str(root))
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=4,
    )
    checkpoint = Path("/mnt/workspace/sjc/models/pretrained/LingBot-Map/lingbot-map-long.pt")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data.get("model", checkpoint_data)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    inputs = load_and_preprocess_images(
        [str(path) for path in paths], mode="crop", image_size=518, patch_size=14
    ).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    model.aggregator = model.aggregator.to(dtype=dtype)

    def forward():
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
            output = model.inference_streaming(
                inputs,
                num_scale_frames=8,
                keyframe_interval=1,
                output_device=torch.device("cpu"),
            )
        del output

    return {"main_model": model}, forward, {
        "strategy": "one continuous 500-frame official streaming call; SDPA, sliding KV window 64, 8 scale frames",
        "model_input_frames": 500,
        "checkpoint_missing_keys": list(missing),
        "checkpoint_unexpected_keys": list(unexpected),
    }


RUNNERS = {
    "egofound3r": _load_egofound3r,
    "wilor": _load_wilor,
    "vggt": _load_vggt,
    "pi3": _load_pi3,
    "da3_large_1_1": _load_da3,
    "lingbot_map_long": _load_lingbot,
    "vggt_omega": _load_vggt_omega,
}

BLOCKED = {
    "hawor": "blocked: source-local DROID and Metric3D weights are absent; no source modification permitted",
    "dyn_hamr": "delegated: run formal_evaluation/benchmark_dyn_hamr_500.py; this legacy single-forward driver must not substitute an optimization-only measurement",
    "pad_hand": "blocked: no verified PAD-Hand runtime environment exists under /mnt/workspace/sjc/envs",
    "reviv4d": "blocked: /mnt/workspace/sjc/external/reviv4d/Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit is missing",
    "s2contact": "blocked: verified adapter consumes H2O contact cache, not preprocessed raw H2O RGB",
    "contactopt": "blocked: verified adapter consumes H2O contact cache, not preprocessed raw H2O RGB",
    "interactvlm": "blocked: cached predictions only; no executable inference entrypoint verified",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    sequence, paths = _frames(args.data_root)
    if args.self_check:
        assert sequence == "subject4_ego/k2/5" and paths[0].stem == "000000" and paths[-1].stem == "000499"
        print("self-check OK")
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "timing_boundary": "loaded model and fully preprocessed in-memory GPU inputs to model prediction; excludes checkpoint load, file read/decode, preprocessing, metrics, saving, and visualization",
        "seed": 0,
        "sequence": sequence,
        "frame_ids": [path.stem for path in paths],
        "methods": {},
    }
    for method in METHODS:
        if method not in args.methods:
            continue
        if method not in RUNNERS:
            report["methods"][method] = {"status": BLOCKED[method]}
            continue
        started = time.perf_counter()
        try:
            modules, forward, detail = RUNNERS[method](paths, args.device)
            counts, total = _parameter_counts(modules)
            result = _timed(forward, torch=torch)
            report["methods"][method] = {
                "status": "success: real neural-network forward",
                "checkpoint_load_and_input_prepare_seconds": time.perf_counter() - started,
                "parameter_count": {"modules": counts, "unique_total": total, "rule": "requires_grad parameters; shared storage deduplicated"},
                **detail,
                **result,
            }
        except Exception as error:
            report["methods"][method] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            (args.output_dir / "benchmark_500.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
