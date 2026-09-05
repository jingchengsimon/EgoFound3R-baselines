#!/usr/bin/env python3
"""Run and validate one registered EgoFound3R comparison window."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from formal_evaluation.common.schema import validate_comparison_output
from formal_evaluation.datasets.window_inputs import load_window_input


def _idle_gpu(initial: str) -> str:
    if initial != "-1":
        return initial
    while True:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        busy = {row.strip() for row in apps.stdout.splitlines() if row.strip()}
        for row in probe.stdout.splitlines() if probe.returncode == 0 else []:
            index, uuid, memory = (value.strip() for value in row.split(",", 2))
            if apps.returncode == 0 and uuid not in busy and memory.isdigit() and int(memory) <= 1024:
                return index
        time.sleep(60)


def _first_window(index: Path) -> Path:
    for line in index.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            return Path(str(row["window_input"]))
    raise ValueError(f"empty smoke input index: {index}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--inference-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--global-stride", type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()

    window_input = _first_window(args.input_index)
    record = load_window_input(window_input)
    gpu = _idle_gpu(os.environ.get("TASKCTL_GPU", "-1"))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
    subprocess.run([
        str(args.model_python), str(args.runner),
        "--phase", "smoke",
        "--window-input", str(window_input),
        "--methods-config", str(args.methods_config),
        "--config", str(args.config),
        "--checkpoint", str(args.checkpoint),
        "--backbone-checkpoint", str(args.backbone_checkpoint),
        "--output-root", str(args.output_root),
        "--device", "cuda:0",
        "--global-stride", str(args.global_stride),
    ], env=environment, check=True)

    output = args.output_root / "egofound3r" / "smoke" / str(record["cache_id"])
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    with np.load(output / "predictions.npz", allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    validate_comparison_output(metadata, arrays)
    expected = {
        "source_commit": args.source_commit,
        "inference_commit": args.inference_commit,
        "checkpoint_sha256": args.checkpoint_sha256,
        "model_compute_dtype": "torch.bfloat16",
        "phase": "smoke",
        "global_stride": args.global_stride,
        "global_anchor_phase": args.global_stride // 2,
    }
    mismatches = {key: metadata.get(key) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches or run.get("status") != "success":
        raise ValueError(f"smoke identity mismatch: {mismatches}, run_status={run.get('status')}")

    summary = {
        "status": "complete",
        "gpu": gpu,
        "window_input": str(window_input),
        "output": str(output),
        "frame_count": len(record["frame_ids"]),
        "elapsed_seconds": run.get("elapsed_seconds"),
        "peak_gpu_memory_bytes": run.get("peak_gpu_memory_bytes"),
        **expected,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
