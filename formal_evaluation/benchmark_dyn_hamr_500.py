#!/usr/bin/env python3
"""Benchmark the complete loaded-once Dyn-HaMR RGB pipeline on 500 H2O frames.

This is the only supported Dyn-HaMR speed entry. It refuses prepared tracks or
cameras: every trial runs RGB -> YOLO -> HaMeR -> DROID-SLAM -> Dyn-HaMR.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def _select(data_root: Path, count: int) -> tuple[str, list[Path]]:
    candidates = sorted(
        rgb.parent.parent.relative_to(data_root).as_posix()
        for rgb in data_root.glob("subject4_ego/*/*/cam4/rgb")
        if sum(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in rgb.iterdir()) >= count
    )
    if not candidates:
        raise RuntimeError(f"no H2O test sequence has {count} frames")
    sequence = random.Random(0).choice(candidates)
    frames = [p for p in sorted((data_root / sequence / "cam4/rgb").iterdir())
              if p.suffix.lower() in {".png", ".jpg", ".jpeg"}][:count]
    return sequence, frames


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parameter_counts(modules):
    counts, seen, total = {}, set(), 0
    for name, module in modules.items():
        count = 0
        for p in module.parameters():
            if not p.requires_grad:
                continue
            count += p.numel()
            storage = p.untyped_storage()
            key = (p.device.type, p.device.index, storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key); total += p.numel()
        counts[name] = count
    return counts, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hamer-checkpoint-root", type=Path, required=True)
    parser.add_argument("--detector-weight", type=Path, required=True)
    parser.add_argument("--droid-weight", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True,
                        help="memory-backed directory such as /dev/shm")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=500, help=argparse.SUPPRESS)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--root-iters", type=int, default=50)
    parser.add_argument("--smooth-iters", type=int, default=300)
    args = parser.parse_args()
    if args.frames != 500:
        raise ValueError("formal Dyn-HaMR speed benchmark requires exactly 500 RGB frames")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in (args.data_root, args.source_root, args.hamer_checkpoint_root,
                 args.detector_weight, args.droid_weight, args.scratch_dir):
        path.resolve(strict=True)

    import cv2
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from ultralytics import YOLO

    source_root = args.source_root.resolve()
    dyn_root = source_root / "dyn-hamr"
    hamer_root = source_root / "third-party/hamer"
    droid_root = source_root / "third-party/DROID-SLAM"
    for path in (dyn_root, hamer_root, droid_root, droid_root / "droid_slam"):
        sys.path.insert(0, str(path))

    hamer = _module(hamer_root / "run.py", "dyn_hamr_loaded_hamer")
    from body_model import MANO
    from data.dataset import MultiPeopleDataset
    from preproc.export_hamer import export_sequence_results
    from preproc.run_slam import get_slam_parser, run_loaded, save_cameras
    from droid import Droid
    from run_opt import run_opt, set_seed
    from util.loaders import resolve_cfg_paths

    sequence, frame_paths = _select(args.data_root, args.frames)
    images = [cv2.imread(str(path)) for path in frame_paths]
    if any(image is None for image in images):
        raise RuntimeError("failed to decode selected H2O RGB frames")
    height, width = images[0].shape[:2]
    # The official video path emits a constant-resolution frame stream. H2O's
    # selected raw sequence mixes two 16:9 encodings, so normalize to the first
    # frame before timing, matching the already-verified official smoke path.
    images = [image if image.shape[:2] == (height, width)
              else cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
              for image in images]

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    load_start = time.perf_counter()
    hamer_model, hamer_cfg = hamer.load_hamer(str(args.hamer_checkpoint_root))
    hamer_model = hamer_model.to(device).eval()
    detector = YOLO(str(args.detector_weight)); detector.to(device)
    droid_net = Droid.load_network(str(args.droid_weight))
    load_seconds = time.perf_counter() - load_start

    with initialize_config_dir(version_base=None, config_dir=str(dyn_root / "confs")):
        cfg = compose(config_name="config", overrides=[
            "data=video_driod", "data.seq=dyn_hamr_h2o500", "data.end_idx=500",
            f"optim.root.num_iters={args.root_iters}",
            f"optim.smooth.num_iters={args.smooth_iters}",
            "run_prior=False", "run_vis=False", "is_static=False",
        ])
    cfg = resolve_cfg_paths(cfg)
    cfg.paths.base_dir = str(source_root)
    cfg.data.frame_opts.fps = 30

    hamer_args = SimpleNamespace(rescale_factor=1.3, render=False, res_folder=None)
    slam_args = get_slam_parser().parse_args([])
    slam_args.weights = str(args.droid_weight)
    slam_args.t0 = 0; slam_args.disable_vis = True; slam_args.stereo = False
    focal = 0.5 * (height + width)
    intrins = torch.tensor([focal, focal, width / 2, height / 2, width, height])[None].repeat(args.frames, 1)
    mano_model = None

    def run_once(load_mano=False):
        nonlocal mano_model, load_seconds
        work = Path(tempfile.mkdtemp(prefix="dyn_hamr_500_", dir=args.scratch_dir))
        try:
            image_dir = work / "images/dyn_hamr_h2o500"
            track_dir = work / "dynhamr/track_preds/dyn_hamr_h2o500"
            shot_path = work / "dynhamr/shot_idcs/dyn_hamr_h2o500.json"
            camera_dir = work / "dynhamr/cameras/dyn_hamr_h2o500/shot-0"
            image_dir.mkdir(parents=True)

            torch.cuda.synchronize(device); total_start = time.perf_counter()
            raw = hamer.extract_raw_bboxes(frame_paths, detector, images=images)
            cleaned = hamer.clean_bbox_sequences(raw)
            hamer_result = hamer.run_hamer_on_cleaned_bboxes(
                cleaned, hamer_model, hamer_cfg, None, hamer_args, images=images)

            # Official inter-stage representation, materialized in memory-backed scratch.
            hamer_pickle = work / "hamer.pkl"
            with hamer_pickle.open("wb") as handle:
                pickle.dump(hamer_result, handle)
            export_sequence_results(str(hamer_pickle), str(track_dir), str(shot_path))

            frame_w2c, droid = run_loaded(slam_args, frame_paths, intrins, images, droid_net)
            save_cameras(str(camera_dir), frame_w2c, intrins)

            sources = {"images": str(image_dir), "tracks": str(track_dir),
                       "shots": str(shot_path), "cameras": str(camera_dir)}
            dataset = MultiPeopleDataset(sources, "dyn_hamr_h2o500", end_idx=500,
                                         is_static=False, split_cameras=True,
                                         img_size=(width, height))
            if dataset.seq_len != 500:
                raise RuntimeError(f"full pipeline produced {dataset.seq_len}, expected 500")
            if mano_model is None:
                if not load_mano:
                    raise RuntimeError("MANO must be initialized during untimed setup")
                mano_start = time.perf_counter()
                mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
                mano_model = MANO(batch_size=len(dataset) * 500, pose2rot=True, **mano_cfg).to(device)
                load_seconds += time.perf_counter() - mano_start

            set_seed(cfg.get("seed", 42))
            _, prediction = run_opt(
                cfg, dataset, str(work / "unused"), device,
                hand_model=mano_model, save_io=False)
            torch.cuda.synchronize(device)
            if prediction["world"]["trans"].shape[1] != 500:
                raise RuntimeError("Dyn-HaMR did not produce 500-frame trajectories")
            total = time.perf_counter() - total_start
            del droid, dataset, hamer_result, raw, cleaned, prediction
            return total
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # One untimed setup pass determines track count and loads shape-dependent MANO once.
    run_once(load_mano=True)
    for _ in range(args.warmups):
        run_once()
    torch.cuda.reset_peak_memory_stats(device)
    trials = []
    for _ in range(args.trials):
        trials.append(run_once())

    counts, unique_total = _parameter_counts({
        "YOLO_detector": detector.model, "HaMeR": hamer_model,
        "DROID_SLAM": droid_net, "MANO": mano_model,
    })
    median = statistics.median(trials)
    report = {
        "status": "success: complete loaded-once Dyn-HaMR RGB pipeline",
        "seed": 0, "sequence": sequence,
        "frame_ids": [path.stem for path in frame_paths],
        "output_frames": 500, "model_input_frames": 500,
        "strategy": "one continuous 500-frame sequence; fresh tracker, SLAM and optimizer state per trial",
        "input_preprocessing": f"decoded RGB normalized before timing to the first-frame size {width}x{height}, matching the official constant-resolution video path",
        "timing_boundary": "decoded RGB arrays in memory through YOLO, HaMeR, DROID-SLAM, official inter-stage conversion and Dyn-HaMR optimization; excludes checkpoint/model load, source image IO/decode, final result save, metrics and visualization",
        "checkpoint_and_model_load_seconds_excluded": load_seconds,
        "trial_seconds": trials, "median_seconds": median,
        "mean_seconds": statistics.fmean(trials), "p90_seconds": sorted(trials)[-1],
        "output_fps": 500 / median,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "parameter_count": {"modules": counts, "unique_total": unique_total,
                            "rule": "requires_grad parameters; shared storage deduplicated"},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "benchmark_dyn_hamr_500.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
