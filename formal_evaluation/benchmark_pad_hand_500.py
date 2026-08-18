#!/usr/bin/env python3
"""Benchmark loaded-once WiLoR plus PAD-Hand on 500 in-memory H2O frames."""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock


def _select(data_root: Path, frame_count: int) -> tuple[str, list[Path]]:
    from formal_evaluation.datasets import get_dataset_adapter
    selected = get_dataset_adapter("h2o", data_root).select_contiguous("test", frame_count, seed=0)
    return selected.sequence_id, list(selected.frame_paths)


def _parameter_counts(modules: dict[str, object]) -> tuple[dict[str, int], int]:
    counts, seen, total = {}, set(), 0
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
        counts[name] = count
    return counts, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--wilor-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-weight", type=Path, required=True)
    parser.add_argument("--pad-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-count", type=int, default=500)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    for name in ("pyrender", "OpenGL", "OpenGL.GL", "OpenGL.platform"):
        sys.modules.setdefault(name, MagicMock())
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.frame_count < 16 or args.warmups < 0 or args.trials < 1:
        raise ValueError("frame-count must be >=16, warmups >=0 and trials >=1")
    sequence, paths = _select(args.data_root, args.frame_count)
    frames = [cv2.imread(str(path)) for path in paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("failed to decode selected H2O frames")

    pad_root = args.source_root.resolve()
    wilor_root = pad_root / "WiLoR"
    sys.path.insert(0, str(wilor_root))
    sys.path.insert(0, str(pad_root))
    import wilor.models as wilor_models
    from ultralytics import YOLO
    from wilor.configs import get_config
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    previous = Path.cwd()
    os.chdir(wilor_root)
    try:
        config = get_config("pretrained_models/model_config.yaml", update_cachedir=True)
        if "vit" in config.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in config.MODEL:
            config.defrost(); config.MODEL.BBOX_SHAPE = [192, 256]; config.freeze()
        if "PRETRAINED_WEIGHTS" in config.MODEL.BACKBONE:
            config.defrost(); config.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS"); config.freeze()
        config.defrost()
        config.MANO.DATA_DIR = "./mano_data/"
        config.MANO.MODEL_PATH = "./mano_data/"
        config.MANO.MEAN_PARAMS = "./mano_data/mano_mean_params.npz"
        config.freeze()
        wilor = wilor_models.WiLoR.load_from_checkpoint(
            str(args.wilor_checkpoint.resolve()), strict=False, cfg=config, init_renderer=False
        ).to(device).eval()
    finally:
        os.chdir(previous)
    detector = YOLO(str(args.detector_weight.resolve()))
    detector.to(device)

    os.chdir(pad_root)
    try:
        from demo import MANO, load_pad_hand_model, refine_with_pad_hand
        pad = load_pad_hand_model(args.pad_checkpoint.resolve(), device)
        mano = MANO("RIGHT", device)
    finally:
        os.chdir(previous)

    def run_wilor() -> list[dict[str, object] | None]:
        predictions: list[dict[str, object] | None] = []
        detections = detector(frames, conf=0.3, verbose=False)
        for frame, result in zip(frames, detections, strict=True):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                predictions.append(None); continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            sides = boxes.cls.detach().cpu().numpy().astype(np.int64)
            confidence = boxes.conf.detach().cpu().numpy()
            right = np.flatnonzero(sides == 1)
            candidates = right if right.size else np.arange(len(sides))
            selected = candidates[np.argmax(confidence[candidates])]
            dataset = ViTDetDataset(config, frame, xyxy[selected:selected + 1], sides[selected:selected + 1])
            batch = recursive_to(next(iter(DataLoader(dataset, batch_size=1, num_workers=0))), device)
            with torch.inference_mode():
                output = wilor(batch)
            multiplier = 2 * batch["right"] - 1
            pred_cam = output["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            image_size = batch["img_size"].float()
            focal = config.EXTRA.FOCAL_LENGTH / config.MODEL.IMAGE_SIZE * image_size.max()
            camera_t = cam_crop_to_full(pred_cam, batch["box_center"].float(),
                                        batch["box_size"].float(), image_size, focal)[0].detach().cpu().numpy()
            is_right = float(batch["right"][0].item())
            vertices = output["pred_vertices"][0].detach().cpu().clone()
            vertices[:, 0] *= 2 * is_right - 1
            predictions.append({
                "vertices": vertices, "cam_t": camera_t,
                "global_orient": output["pred_mano_params"]["global_orient"][0].detach().cpu(),
                "hand_pose": output["pred_mano_params"]["hand_pose"][0].detach().cpu(),
                "betas": output["pred_mano_params"]["betas"][0].detach().cpu(),
                "is_right": is_right, "img_size": image_size[0].cpu().numpy(),
                "scaled_focal": float(focal.item()),
            })
        return predictions

    def run_once() -> dict[str, float]:
        torch.cuda.synchronize(device); started = time.perf_counter()
        predictions = run_wilor()
        torch.cuda.synchronize(device); wilor_end = time.perf_counter()
        padded_predictions = list(predictions)
        remainder = len(padded_predictions) % 16
        if remainder:
            padded_predictions.extend([padded_predictions[-1]] * (16 - remainder))
        os.chdir(pad_root)
        try:
            refined = refine_with_pad_hand(padded_predictions, pad, mano, device)[:args.frame_count]
        finally:
            os.chdir(previous)
        torch.cuda.synchronize(device); end = time.perf_counter()
        if len(refined) != args.frame_count:
            raise RuntimeError(f"expected {args.frame_count} PAD outputs, got {len(refined)}")
        return {"wilor_frontend": wilor_end - started, "pad_refinement": end - wilor_end}

    for _ in range(args.warmups):
        run_once()
    torch.cuda.reset_peak_memory_stats(device)
    trials, stage_trials = [], {"wilor_frontend": [], "pad_refinement": []}
    for _ in range(args.trials):
        torch.cuda.synchronize(device); started = time.perf_counter()
        stages = run_once()
        trials.append(time.perf_counter() - started)
        for name, seconds in stages.items():
            stage_trials[name].append(seconds)
    median = statistics.median(trials)
    counts, unique_total = _parameter_counts({"detector": detector.model, "wilor": wilor, "pad_hand": pad})
    report = {
        "status": "success: loaded-once complete WiLoR plus PAD-Hand pipeline",
        "seed": 0, "sequence": sequence, "frame_ids": [path.stem for path in paths],
        "output_frames": args.frame_count,
        "model_input_frames": {"wilor": args.frame_count,
                               "pad_hand": ((args.frame_count + 15) // 16) * 16},
        "strategy": f"WiLoR on {args.frame_count} frames; PAD official non-overlapping 16-frame windows; final window repeats the last input {(-args.frame_count) % 16} times and is cropped back to requested outputs",
        "timing_boundary": "decoded RGB arrays in memory through detector, WiLoR and PAD refinement; excludes model/checkpoint load, image decode, metrics and saves",
        "trial_seconds": trials, "median_seconds": median, "mean_seconds": statistics.fmean(trials),
        "p90_seconds": sorted(trials)[min(len(trials) - 1, int(0.9 * len(trials)))],
        "output_fps": args.frame_count / median,
        "stage_trial_seconds": stage_trials,
        "stage_median_seconds": {name: statistics.median(values) for name, values in stage_trials.items()},
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "parameter_count": {"modules": counts, "unique_total": unique_total,
                            "rule": "requires_grad parameters; shared storage deduplicated"},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "benchmark_pad_hand_500.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    del detector, wilor, pad, mano
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
