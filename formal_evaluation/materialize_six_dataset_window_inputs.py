#!/usr/bin/env python3
"""Export exact dataloader RGB and canonical geometry for method adapters."""

from __future__ import annotations

import argparse
import json
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.windows.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        validate_window_row(row)
    bridge = SixDatasetGroundTruth(_roots(args.root), args.mano_dir)
    records = []
    for index, row in enumerate(rows, start=1):
        record = materialize_window_input(bridge, row, args.output_root)
        records.append(str(record))
        print(json.dumps({"index": index, "total": len(rows), "input": str(record)}), flush=True)
    index_path = args.output_root / "window_inputs.jsonl"
    index_path.write_text("".join(json.dumps({"window_input": item}) + "\n" for item in records), encoding="utf-8")
    print(json.dumps({"status": "complete", "index": str(index_path)}), flush=True)


if __name__ == "__main__":
    main()
