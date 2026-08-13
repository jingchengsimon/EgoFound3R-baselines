#!/usr/bin/env python3
"""Strict loaded-once timing entry for Dyn-HaMR optimization on prepared 500-frame inputs.

HaMeR and DROID preparation must come from the same 500 decoded RGB frames.  This
entry deliberately reports the Dyn optimization stage only; frontend stage times
must be added by the loaded HaMeR/DROID hooks, never by timing cached-file reads.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def _params(module):
    seen, total = set(), 0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        storage = p.untyped_storage()
        key = (p.device.type, p.device.index, storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += p.numel()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[],
                        help="Hydra override used by the verified official run; repeat as needed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    dyn_root = (args.source_root / "dyn-hamr").resolve()
    sys.path.insert(0, str(dyn_root))
    import torch
    from hydra import compose, initialize_config_dir
    from data import expand_source_paths, get_dataset_from_cfg
    from body_model import MANO
    from run_opt import run_opt, set_seed
    from util.loaders import resolve_cfg_paths

    with initialize_config_dir(version_base=None, config_dir=str(dyn_root / "confs")):
        cfg = compose(config_name="config", overrides=args.override)
    cfg = resolve_cfg_paths(cfg)
    cfg.paths.base_dir = str(args.source_root.resolve())
    cfg.data.sources = expand_source_paths(cfg.data.sources)
    dataset = get_dataset_from_cfg(cfg)
    if dataset.seq_len != 500:
        raise ValueError(f"prepared Dyn-HaMR input must contain exactly 500 frames, got {dataset.seq_len}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
    load_start = time.perf_counter()
    hand_model = MANO(batch_size=len(dataset) * dataset.seq_len,
                      pose2rot=True, **mano_cfg).to(device)
    load_seconds = time.perf_counter() - load_start
    mano_params = _params(hand_model)

    def once():
        set_seed(cfg.get("seed", 42))
        torch.cuda.synchronize()
        start = time.perf_counter()
        stages = run_opt(cfg, dataset, str(args.output_dir), device,
                         hand_model=hand_model, save_io=False)
        torch.cuda.synchronize()
        return time.perf_counter() - start, stages

    for _ in range(args.warmups):
        once()
    torch.cuda.reset_peak_memory_stats(device)
    trials, stage_trials = [], {}
    for _ in range(args.trials):
        elapsed, stages = once()
        trials.append(elapsed)
        for name, seconds in stages.items():
            stage_trials.setdefault(name, []).append(seconds)
    median = statistics.median(trials)
    report = {
        "status": "success: pure Dyn-HaMR optimization; not full RGB pipeline",
        "output_frames": 500, "model_input_frames": 500,
        "timing_boundary": "prepared HaMeR tracks and DROID cameras in memory through fresh BaseSceneModel initialization, root optimization and smooth optimization; excludes MANO load and all file IO",
        "checkpoint_and_model_load_seconds_excluded": load_seconds,
        "trial_seconds": trials, "median_seconds": median,
        "mean_seconds": statistics.fmean(trials), "p90_seconds": max(trials),
        "output_fps": 500 / median,
        "stage_trial_seconds": stage_trials,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "parameter_count": {"MANO": mano_params, "unique_total": mano_params,
                            "rule": "requires_grad parameters; shared storage deduplicated"},
        "full_pipeline_note": "Use loaded extract_raw_bboxes/run_hamer_on_cleaned_bboxes and preproc.run_slam.run_loaded for frontend stage timing; cached prediction reads are excluded and are not inference."
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "benchmark_dyn_hamr_500.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
