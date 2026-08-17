#!/usr/bin/env python3
"""Make one deterministic 20-frame smoke clip per dataset from the strict test windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_evaluation.datasets.egofound3r_gt import DATASET_LOADERS, validate_window_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    if args.frames < 1 or args.frames > 60:
        raise ValueError("--frames must be in [1, 60]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite pilot manifest: {args.output}")

    rows: dict[str, dict[str, object]] = {}
    for line in args.windows.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        dataset, _, frame_ids = validate_window_row(row)
        if dataset in rows:
            continue
        if len(frame_ids) != 60 or row.get("window_stride") != 60 or row.get("window_overlap") != 0:
            raise ValueError(f"{row.get('window_id')}: expected strict 60-frame input")
        rows[dataset] = {
            **row,
            "window_id": f"{row['window_id']}:pilot20",
            "frame_ids": frame_ids[:args.frames],
            "window_size": args.frames,
            "window_stride": args.frames,
            "window_overlap": 0,
            "window_source": "first_frames_of_strict_60f_test_window",
            "phase": "pilot20",
        }
    missing = set(DATASET_LOADERS) - set(rows)
    if missing:
        raise ValueError(f"missing datasets: {sorted(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(rows[name], ensure_ascii=False, sort_keys=True) + "\n" for name in sorted(rows)), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frames_per_dataset": args.frames, "datasets": sorted(rows)}))


if __name__ == "__main__":
    main()
