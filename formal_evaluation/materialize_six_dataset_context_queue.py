#!/usr/bin/env python3
"""Add one shared RGB context length to completed window-input shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formal_evaluation.datasets.egofound3r_gt import DATASET_LOADERS


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-index", action="append", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.context_frames < 1 or args.workers < 1:
        raise ValueError("--context-frames and --workers must be positive")
    if not args.python.is_file() or not args.repo_root.is_dir() or args.queue_root.exists():
        raise FileExistsError("--python/repo-root missing or queue root already exists")
    roots = _roots(args.root)
    indices = sorted(args.input_index)
    for index in indices:
        status = json.loads(index.with_suffix(".status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete" or status.get("index") != str(index):
            raise ValueError(f"input shard is not complete: {index}")
    args.queue_root.mkdir(parents=True)
    log_root, sentinel_root = args.queue_root / "logs", args.queue_root / "sentinels"
    log_root.mkdir(); sentinel_root.mkdir()
    runner = args.repo_root / "formal_evaluation/materialize_six_dataset_video_contexts.py"

    def run(index: Path) -> dict[str, object]:
        stem = index.stem
        sentinel = sentinel_root / f"{stem}.status.json"
        log = log_root / f"{stem}.log"
        command = [
            str(args.python), str(runner), "--window-input-index", str(index), "--mano-dir", str(args.mano_dir),
            "--context-frames", str(args.context_frames), "--sentinel", str(sentinel),
        ]
        for root in roots:
            command.extend(("--root", root))
        started = time.monotonic()
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=args.repo_root, stdout=handle, stderr=subprocess.STDOUT)
        status = "success" if completed.returncode == 0 and sentinel.is_file() else "failed"
        result = {"index": str(index), "status": status, "returncode": completed.returncode,
                  "elapsed_seconds": time.monotonic() - started, "log": str(log), "sentinel": str(sentinel)}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(run, indices))
    counts = Counter(result["status"] for result in results)
    _atomic_json(args.queue_root / "summary.json", {
        "status": "complete" if not counts.get("failed") else "completed_with_failures",
        "context_frames": args.context_frames, "counts": dict(counts), "tasks": results,
    })
    if counts.get("failed"):
        raise SystemExit("one or more context shards failed; see queue summary")


if __name__ == "__main__":
    main()
