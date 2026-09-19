#!/usr/bin/env python3
"""Copy only jump-cut HaWoR files absent from the frozen-114 CPFS source tree."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil


DATASETS = {"h2o", "oakink_v2"}


def readable(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stitched-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-missing", type=int, required=True)
    parser.add_argument("--reserve-gib", type=int, default=20)
    args = parser.parse_args()

    entries = [json.loads(line) for line in args.stitched_manifest.read_text().splitlines()
               if line.strip()]
    if len(entries) != 114:
        raise ValueError(f"STITCHED_SEGMENT_COUNT:{len(entries)}")
    required = sorted({(entry["dataset"], ref["cache_id"])
                       for entry in entries if entry["dataset"] in DATASETS
                       for ref in entry["frame_refs"]})
    missing_old = [(dataset, cache) for dataset, cache in required
                   if not readable(args.old_root / dataset / cache / "predictions.npz")]
    if len(missing_old) != args.expected_missing:
        raise ValueError(f"OLD_MISSING_COUNT:{len(missing_old)}:{args.expected_missing}")

    source_missing = []
    inventory = []
    for dataset, cache in missing_old:
        source = args.source_root / dataset / cache / "predictions.npz"
        if not readable(source):
            source_missing.append({"dataset": dataset, "cache_id": cache,
                                   "source": str(source)})
        else:
            inventory.append((dataset, cache, source, source.stat().st_size))
    required_bytes = sum(row[3] for row in inventory)
    free_bytes = shutil.disk_usage(args.output_root.parent).free
    report = {
        "status": "ready" if not source_missing else "blocked",
        "segments": len(entries), "required_windows": len(required),
        "old_present": len(required) - len(missing_old),
        "old_missing": len(missing_old), "source_present": len(inventory),
        "source_missing": source_missing, "required_bytes": required_bytes,
        "cpfs_free_bytes": free_bytes,
        "dataset_counts": dict(Counter(dataset for dataset, _ in missing_old)),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if source_missing:
        raise SystemExit(2)
    if free_bytes - required_bytes < args.reserve_gib * 1024**3:
        raise RuntimeError("INSUFFICIENT_CPFS_HEADROOM")

    rows = []
    for index, (dataset, cache, source, size) in enumerate(inventory, 1):
        destination = args.output_root / "files" / dataset / cache / "predictions.npz"
        if destination.exists():
            if destination.stat().st_size != size:
                raise RuntimeError(f"DESTINATION_SIZE_CONFLICT:{destination}")
            copied = False
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(".npz.partial")
            shutil.copy2(source, partial)
            if partial.stat().st_size != size:
                raise RuntimeError(f"PARTIAL_SIZE_MISMATCH:{partial}")
            partial.replace(destination)
            copied = True
        rows.append({"dataset": dataset, "cache_id": cache, "bytes": size,
                     "source": str(source), "destination": str(destination),
                     "copied": copied})
        if index % 10 == 0:
            print(json.dumps({"processed": index, "total": len(inventory)}), flush=True)

    (args.output_root / "transfer_manifest.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    report.update(status="complete", copied_files=sum(row["copied"] for row in rows),
                  resumed_files=sum(not row["copied"] for row in rows),
                  copied_bytes=sum(row["bytes"] for row in rows if row["copied"]))
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text(
        json.dumps({"status": "complete", "files": len(rows)}, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "old_missing", "copied_files", "resumed_files",
                       "copied_bytes", "dataset_counts")}), flush=True)


if __name__ == "__main__":
    main()
