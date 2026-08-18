#!/usr/bin/env python3
"""Run resumable per-dataset window-input shards concurrently."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

from formal_evaluation.datasets.egofound3r_gt import DATASET_LOADERS, validate_window_row


def _roots(values: list[str]) -> list[str]:
    roots = dict(value.split("=", 1) for value in values)
    if set(roots) != set(DATASET_LOADERS):
        raise ValueError(f"--root must name exactly {sorted(DATASET_LOADERS)}")
    return [f"{dataset}={roots[dataset]}" for dataset in DATASET_LOADERS]


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _tasks(windows: Path, shard_size: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {dataset: [] for dataset in DATASET_LOADERS}
    for line in windows.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        dataset, _, frame_ids = validate_window_row(row)
        if len(frame_ids) != 60 or row.get("window_stride") != 60 or row.get("window_overlap") != 0:
            raise ValueError(f"{row.get('window_id')}: expected strict 60-frame non-overlap window")
        grouped[dataset].append(row)
    tasks = []
    for dataset in DATASET_LOADERS:
        count = len(grouped[dataset])
        if not count:
            raise ValueError(f"manifest contains no {dataset} windows")
        shard_count = math.ceil(count / shard_size)
        for shard_index in range(shard_count):
            tasks.append({
                "dataset": dataset,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "window_count": min(shard_size, count - shard_index * shard_size),
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="reuse only completed shard sentinels in --output-root")
    args = parser.parse_args()
    if args.shard_size < 1 or args.workers < 1:
        raise ValueError("--shard-size and --workers must be positive")
    if not args.windows.is_file() or not args.python.is_file() or not args.repo_root.is_dir():
        raise FileNotFoundError("--windows, --python, and --repo-root must exist")
    roots = _roots(args.root)
    tasks = _tasks(args.windows, args.shard_size)
    if args.queue_root.exists():
        raise FileExistsError(f"refusing to reuse queue root: {args.queue_root}")
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse output root without --resume: {args.output_root}")
    args.queue_root.mkdir(parents=True)
    log_root = args.queue_root / "logs"
    log_root.mkdir()
    runner = args.repo_root / "formal_evaluation/materialize_six_dataset_window_inputs.py"

    def run(task: dict[str, object]) -> dict[str, object]:
        dataset = str(task["dataset"])
        shard_index, shard_count = int(task["shard_index"]), int(task["shard_count"])
        stem = f"{dataset}_shard_{shard_index:03d}_of_{shard_count:03d}"
        sentinel = args.output_root / f"window_inputs_{stem}.status.json"
        if sentinel.is_file():
            return {**task, "status": "reused", "sentinel": str(sentinel)}
        command = [
            str(args.python), str(runner), "--windows", str(args.windows), "--mano-dir", str(args.mano_dir),
            "--output-root", str(args.output_root), "--dataset", dataset,
            "--shard-size", str(args.shard_size), "--shard-index", str(shard_index),
        ]
        for root in roots:
            command.extend(("--root", root))
        started = time.monotonic()
        log = log_root / f"{stem}.log"
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=args.repo_root, stdout=handle, stderr=subprocess.STDOUT)
        status = "success" if completed.returncode == 0 and sentinel.is_file() else "failed"
        return {
            **task, "status": status, "returncode": completed.returncode,
            "elapsed_seconds": time.monotonic() - started, "log": str(log), "sentinel": str(sentinel),
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(run, tasks):
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    status_counts = Counter(str(result["status"]) for result in results)
    summary = {
        "status": "complete" if not status_counts.get("failed") else "completed_with_failures",
        "windows": str(args.windows),
        "windows_sha256": hashlib.sha256(args.windows.read_bytes()).hexdigest(),
        "output_root": str(args.output_root),
        "shard_size": args.shard_size,
        "workers": args.workers,
        "counts": dict(status_counts),
        "tasks": results,
    }
    _atomic_json(args.queue_root / "summary.json", summary)
    if status_counts.get("failed"):
        raise SystemExit("one or more input shards failed; see queue summary")


if __name__ == "__main__":
    main()
