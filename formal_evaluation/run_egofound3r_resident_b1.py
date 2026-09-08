#!/usr/bin/env python3
"""Run one dataset with one resident EgoFound3R model and B=1 clips."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formal_evaluation.run_egofound3r_stride_evaluation import prepare_inputs
from formal_evaluation.scene.adapters import run_egofound3r_baseline as adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-resolution", choices=("384x512", "448x448", "512x512"), required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if len(spec["datasets"]) != 1:
        raise ValueError("resident B=1 worker requires exactly one dataset")
    args.output_root.mkdir(parents=True, exist_ok=False)

    dataset, dataset_spec = next(iter(spec["datasets"].items()))
    _, gt_index, records = prepare_inputs(dataset, dataset_spec, args.output_root / "inputs")
    dataset_root = args.output_root / dataset
    dataset_root.mkdir()
    progress_path = dataset_root / "progress.jsonl"
    with progress_path.open("x") as progress:
        for completed, record in enumerate(records, start=1):
            started = time.monotonic()
            adapter.main([
                "--phase", "formal",
                "--window-input", record["window_input"],
                "--methods-config", spec["methods_config"],
                "--config", spec["config"],
                "--checkpoint", spec["checkpoint"],
                "--backbone-checkpoint", spec["backbone"],
                "--global-stride", "5",
                "--input-resolution", args.input_resolution,
                "--output-root", str(dataset_root),
            ])
            cache = adapter._runtime.cache_info()
            if cache.misses != 1:
                raise RuntimeError(f"model was not resident: {cache}")
            row = {
                "dataset": dataset,
                "completed": completed,
                "total": len(records),
                "model_load_count": cache.misses,
                "model_cache_hits": cache.hits,
                "wall_seconds": time.monotonic() - started,
                "input_resolution": args.input_resolution,
            }
            progress.write(json.dumps(row) + "\n")
            progress.flush()
            print(json.dumps(row), flush=True)

    prediction_index = dataset_root / "predictions.jsonl"
    prediction_index.write_text("".join(
        json.dumps({
            "method": "egofound3r",
            "dataset": dataset,
            "window_id": record["window_id"],
            "prediction_dir": str(dataset_root / "egofound3r/formal" / record["cache_id"]),
        }) + "\n"
        for record in records
    ))
    report = dataset_root / "report.json"
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("evaluate_six_dataset.py")),
        "--gt-index", str(gt_index),
        "--prediction-index", str(prediction_index),
        "--methods-config", spec["methods_config"],
        "--report-path", str(report),
    ], check=True)
    summary = {
        "status": "complete",
        "dataset": dataset,
        "windows": len(records),
        "report": str(report),
        "batch_size": 1,
        "model_resident": True,
        "global_stride": 5,
        "global_anchor_phase": 2,
        "clip_frames": 60,
        "input_resolution": args.input_resolution,
        "source_commit": spec["source_commit"],
        "inference_commit": spec["inference_commit"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
