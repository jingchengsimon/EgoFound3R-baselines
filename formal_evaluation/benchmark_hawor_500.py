#!/usr/bin/env python3
"""Benchmark the loaded-once full HaWoR pipeline on 500 H2O frames."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def _select(data_root: Path) -> tuple[str, list[Path]]:
    from formal_evaluation.datasets import get_dataset_adapter
    selected = get_dataset_adapter("h2o", data_root).select_contiguous("test", 500, seed=0)
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
                seen.add(key); total += parameter.numel()
        counts[name] = count
    return counts, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--infiller-weight", type=Path, required=True)
    parser.add_argument("--detector-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    import cv2
    import torch
    from formal_evaluation.hand.adapters.run_hawor_baseline import HaworRuntime

    sequence, paths = _select(args.data_root)
    frames = [cv2.imread(str(path)) for path in paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("failed to decode selected H2O frames")
    torch.cuda.set_device(args.device)
    load_start = time.perf_counter()
    runtime = HaworRuntime(
        args.source_root, args.checkpoint, args.infiller_weight, args.detector_weight, args.device
    )
    load_seconds = time.perf_counter() - load_start
    counts, unique_total = _parameter_counts(runtime.parameter_modules())

    def run_once() -> dict[str, float]:
        arrays, _, _, detail = runtime.run(frames)
        if arrays["hand_joints_world"].shape[:3] != (500, 2, 21):
            raise RuntimeError("unexpected HaWoR output shape")
        return detail["stage_seconds"]

    for _ in range(2):
        run_once(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    trials = []
    stage_trials: dict[str, list[float]] = {}
    for _ in range(5):
        torch.cuda.synchronize(); start = time.perf_counter()
        stages = run_once(); torch.cuda.synchronize()
        trials.append(time.perf_counter() - start)
        for name, seconds in stages.items():
            stage_trials.setdefault(name, []).append(seconds)
    median = statistics.median(trials)
    report = {
        "status": "success: full loaded-once HaWoR online pipeline",
        "seed": 0, "sequence": sequence, "frame_ids": [path.stem for path in paths],
        "output_frames": 500, "model_input_frames": 500,
        "strategy": "one continuous 500-frame sequence; tracker and SLAM state reset before every trial",
        "timing_boundary": "decoded RGB arrays in memory through detector/tracker, HaWoR, DROID-SLAM, Metric3D, infiller and MANO outputs; excludes model construction/checkpoint load, metrics and result saving",
        "checkpoint_and_model_load_seconds_excluded": load_seconds,
        "trial_seconds": trials, "median_seconds": median,
        "mean_seconds": statistics.fmean(trials), "p90_seconds": max(trials),
        "stage_trial_seconds": stage_trials,
        "stage_median_seconds": {name: statistics.median(values) for name, values in stage_trials.items()},
        "output_fps": 500 / median, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "parameter_count": {"modules": counts, "unique_total": unique_total,
                            "rule": "requires_grad parameters; shared storage deduplicated"},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "benchmark_hawor_500.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
