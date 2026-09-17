#!/usr/bin/env python3
"""Run one exact registered prediction backfill after a full-matrix audit."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def idle_gpu(initial: str) -> str:
    if initial != "-1":
        return initial
    while True:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        if probe.returncode == 0:
            for row in probe.stdout.splitlines():
                fields = [value.strip() for value in row.split(",")]
                if len(fields) == 2 and fields[1].isdigit() and int(fields[1]) <= 10:
                    return fields[0]
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--expected-failures", required=True, type=int)
    parser.add_argument("--window-count", required=True, type=int)
    parser.add_argument("--model-python", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runner-arg", action="append", default=[])
    parser.add_argument("--input-index", action="append", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    runtime = Path(__file__).parent
    index = root / "inputs" / f"{args.dataset}_{args.method}_missing_{args.expected_failures}.jsonl"
    audit = root / f"audit_{args.method}_{args.expected_failures}.json"
    subprocess.run([
        args.model_python, str(runtime / "prepare_prediction_backfill.py"),
        "--dataset", args.dataset, "--method", args.method,
        *sum((["--input-index", value] for value in args.input_index), []),
        "--window-count", str(args.window_count),
        "--prediction-root", args.prediction_root, "--expected-failures", str(args.expected_failures),
        "--audit-path", str(audit), "--output-index", str(index),
    ], check=True)
    gpu = idle_gpu(os.environ.get("TASKCTL_GPU", "-1"))
    subprocess.run([
        args.model_python, str(runtime / "run_formal_window_queue.py"),
        "--method", args.method, "--python", args.model_python, "--runner", args.runner,
        *("--runner-arg=" + value for value in args.runner_arg),
        "--input-index", str(index), "--output-root", str(root),
        "--queue-root", str(root / f"queue_{args.method}_{args.expected_failures}"),
        "--gpus", gpu, "--resume",
    ], check=True)


if __name__ == "__main__":
    main()
