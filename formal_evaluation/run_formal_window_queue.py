#!/usr/bin/env python3
"""Run one canonical baseline over completed input shards on a GPU pool."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formal_evaluation.common.schema import validate_comparison_output
from formal_evaluation.datasets.window_inputs import load_window_input


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _validated_output(path: Path, record: dict[str, object], method: str) -> None:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    with np.load(path / "predictions.npz", allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    validate_comparison_output(metadata, arrays)
    if metadata.get("method") != method or metadata.get("dataset") != record["dataset"]:
        raise ValueError("method/dataset identity mismatch")
    if metadata.get("window_id") != record["window_id"] or metadata.get("frame_ids") != record["frame_ids"]:
        raise ValueError("window/frame identity mismatch")
    if run.get("status") != "success" or run.get("frame_count", len(record["frame_ids"])) != len(record["frame_ids"]):
        raise ValueError("run.json does not mark this exact window successful")


def _parse_env(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, val = value.partition("=")
        if not separator or not key or "=" in key:
            raise ValueError(f"--env must be KEY=VALUE: {value!r}")
        result[key] = val
    return result


def _input_tasks(indices: list[Path]) -> list[dict[str, object]]:
    tasks = []
    for index in indices:
        sentinel = index.with_suffix(".status.json")
        status = json.loads(sentinel.read_text(encoding="utf-8"))
        if status.get("status") != "complete" or status.get("index") != str(index):
            raise ValueError(f"input shard is not complete: {index}")
        rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            raise ValueError(f"empty input index: {index}")
        if status.get("window_count") != len(rows):
            raise ValueError(f"input shard count mismatch: {index}")
        records = []
        for row in rows:
            path = Path(str(row["window_input"]))
            records.append({**load_window_input(path), "_path": str(path)})
        datasets = {str(record["dataset"]) for record in records}
        if len(datasets) != 1:
            raise ValueError(f"input index mixes datasets: {index}")
        tasks.append({"index": index, "dataset": datasets.pop(), "records": records})
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runner-arg", action="append", default=[],
                        help="one exact runner argument; {window_input} and {output_root} are substituted")
    parser.add_argument("--input-index", action="append", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must be a non-empty, unique comma-separated list")
    if not args.python.is_file() or not args.runner.is_file():
        raise FileNotFoundError("--python and --runner must exist")
    if args.queue_root.exists():
        raise FileExistsError(f"refusing to reuse queue root: {args.queue_root}")
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse output root without --resume: {args.output_root}")
    tasks = _input_tasks(args.input_index)
    args.queue_root.mkdir(parents=True)
    log_root, sentinel_root = args.queue_root / "logs", args.queue_root / "sentinels"
    log_root.mkdir(); sentinel_root.mkdir()
    pending: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()
    for task in tasks:
        pending.put(task)
    environment = _parse_env(args.env)

    def worker(gpu: str) -> list[dict[str, object]]:
        results = []
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return results
            index = Path(task["index"])
            stem = index.stem
            sentinel = sentinel_root / f"{args.method}_{stem}.status.json"
            started = time.monotonic()
            log = log_root / f"{args.method}_{stem}.log"
            succeeded, reused, failed = 0, 0, []
            with log.open("x", encoding="utf-8") as handle:
                for record in task["records"]:
                    output = args.output_root / args.method / "formal" / str(record["cache_id"])
                    try:
                        if output.is_dir():
                            _validated_output(output, record, args.method)
                            reused += 1
                            continue
                        command = [str(args.python), str(args.runner)]
                        command.extend(argument.format(window_input=str(record["_path"]), output_root=str(args.output_root))
                                       for argument in args.runner_arg)
                        env = {**os.environ, **environment, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
                        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
                        if completed.returncode:
                            raise RuntimeError(f"runner exit={completed.returncode}")
                        _validated_output(output, record, args.method)
                        succeeded += 1
                    except Exception as error:
                        failed.append({"cache_id": record["cache_id"], "error": repr(error)})
            result = {
                "method": args.method, "dataset": task["dataset"], "index": str(index), "gpu": gpu,
                "status": "success" if not failed else "failed", "window_count": len(task["records"]),
                "success_count": succeeded, "reused_count": reused, "failed": failed,
                "elapsed_seconds": time.monotonic() - started, "log": str(log),
            }
            if not failed:
                _atomic_json(sentinel, {**result, "sentinel": str(sentinel)})
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        for worker_results in executor.map(worker, gpus):
            results.extend(worker_results)
    counts = Counter(str(result["status"]) for result in results)
    _atomic_json(args.queue_root / "summary.json", {
        "method": args.method, "status": "complete" if not counts.get("failed") else "completed_with_failures",
        "gpus": gpus, "counts": dict(counts), "tasks": results,
    })
    if counts.get("failed"):
        raise SystemExit("one or more formal shards failed; see queue summary")


if __name__ == "__main__":
    main()
