#!/usr/bin/env python3
"""Precompute each exact evaluation window's GT once for reuse by all baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from formal_evaluation.datasets.egofound3r_gt import DATASET_LOADERS, SixDatasetGroundTruth, validate_window_row
from formal_evaluation.datasets.six_dataset_gt_cache import cache_paths, write_window_cache


def _roots(values: list[str]) -> dict[str, str]:
    roots = dict(value.split("=", 1) for value in values)
    if set(roots) != set(DATASET_LOADERS):
        raise ValueError(f"--root must name exactly {sorted(DATASET_LOADERS)}")
    return roots


def _rows(path: Path, datasets: set[str], shard_count: int, shard_index: int) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        dataset, _, frame_ids = validate_window_row(row)
        if dataset not in datasets:
            continue
        if len(frame_ids) != 60 or row.get("window_stride") != 60 or row.get("window_overlap") != 0:
            raise ValueError(f"{row.get('window_id')}: expected strict 60-frame non-overlap window")
        digest = int(hashlib.sha256(str(row["window_id"]).encode()).hexdigest(), 16)
        if digest % shard_count == shard_index:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASET_LOADERS), default=sorted(DATASET_LOADERS))
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-windows", type=int, help="bounded validation only; omit for a full shard")
    parser.add_argument("--scene-visibility-device", default="cpu")
    parser.add_argument("--interhand-contact-compute-device", default="cpu")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must satisfy 0 <= index < count")

    selected = _rows(args.windows, set(args.datasets), args.shard_count, args.shard_index)
    if args.max_windows is not None:
        if args.max_windows < 1:
            raise ValueError("--max-windows must be positive")
        selected = selected[:args.max_windows]
    if not selected:
        raise ValueError("selected shard has no windows")
    args.output_root.mkdir(parents=True, exist_ok=True)
    bridge = SixDatasetGroundTruth(
        _roots(args.root), args.mano_dir,
        scene_visibility_device=args.scene_visibility_device,
        interhand_contact_compute_device=args.interhand_contact_compute_device,
    )
    entries = []
    for index, row in enumerate(selected, start=1):
        data_path, metadata_path = cache_paths(args.output_root, row)
        if data_path.exists() and metadata_path.exists():
            entry = write_window_cache(args.output_root, row, {})
        else:
            entry = write_window_cache(
                args.output_root,
                row,
                bridge.batch_for_window(row),
                geometry_frames=bridge.geometry_for_window(row),
            )
        entries.append({
            "dataset": entry["dataset"], "sequence_id": entry["sequence_id"], "window_id": entry["window_id"],
            "frame_ids": entry["frame_ids"], "cache_id": entry["cache_id"], "array_path": entry["array_path"],
            "metadata_path": str(metadata_path), "status": entry["status"],
        })
        print(json.dumps({"index": index, "total": len(selected), "dataset": entry["dataset"], "status": entry["status"]}), flush=True)
    index_path = args.output_root / f"index_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.jsonl"
    index_path.write_text("".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
    print(json.dumps({"status": "complete", "index": str(index_path), "counts": dict(Counter(item["dataset"] for item in entries))}), flush=True)


if __name__ == "__main__":
    main()
