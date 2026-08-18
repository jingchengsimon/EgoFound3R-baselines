#!/usr/bin/env python3
"""Export exact dataloader RGB and canonical geometry for method adapters."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formal_evaluation.datasets.egofound3r_gt import DATASET_LOADERS, SixDatasetGroundTruth, validate_window_row
from formal_evaluation.datasets.window_inputs import materialize_window_input


def _roots(values: list[str]) -> dict[str, str]:
    roots = dict(value.split("=", 1) for value in values)
    if set(roots) != set(DATASET_LOADERS):
        raise ValueError(f"--root must name exactly {sorted(DATASET_LOADERS)}")
    return roots


def _atomic_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    """Publish a completed shard index without exposing a partial file."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _select_shard(rows: list[dict[str, object]], dataset: str, shard_size: int, shard_index: int) -> tuple[list[dict[str, object]], int]:
    selected = [row for row in rows if row["dataset"] == dataset]
    shard_count = math.ceil(len(selected) / shard_size)
    if not shard_count:
        raise ValueError(f"manifest contains no {dataset} windows")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"{dataset} shard index must satisfy 0 <= index < {shard_count}")
    start = shard_index * shard_size
    return selected[start:start + shard_size], shard_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_LOADERS), required=True,
                        help="materialize one dataset at a time so shards remain independently resumable")
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    rows = [json.loads(line) for line in args.windows.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        validate_window_row(row)
    rows, shard_count = _select_shard(rows, args.dataset, args.shard_size, args.shard_index)
    index_stem = f"window_inputs_{args.dataset}_shard_{args.shard_index:03d}_of_{shard_count:03d}"
    index_path = args.output_root / f"{index_stem}.jsonl"
    sentinel_path = args.output_root / f"{index_stem}.status.json"
    if index_path.exists() or sentinel_path.exists():
        raise FileExistsError(f"refusing to replace completed shard index/sentinel: {index_stem}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    bridge = SixDatasetGroundTruth(_roots(args.root), args.mano_dir, datasets=(args.dataset,))
    records = []
    for index, row in enumerate(rows, start=1):
        record = materialize_window_input(bridge, row, args.output_root)
        records.append(str(record))
        print(json.dumps({"index": index, "total": len(rows), "input": str(record)}), flush=True)
    _atomic_jsonl(index_path, [{"window_input": item} for item in records])
    _atomic_json(sentinel_path, {
        "status": "complete",
        "dataset": args.dataset,
        "shard_index": args.shard_index,
        "shard_count": shard_count,
        "shard_size": args.shard_size,
        "window_count": len(rows),
        "index": str(index_path),
    })
    print(json.dumps({"status": "complete", "index": str(index_path), "sentinel": str(sentinel_path)}), flush=True)


if __name__ == "__main__":
    main()
