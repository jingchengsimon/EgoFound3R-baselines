#!/usr/bin/env python3
"""Smoke-check exact 60-frame evaluation windows through the latest dataloader GT path."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from formal_evaluation.datasets.egofound3r_gt import DATASET_LOADERS, SixDatasetGroundTruth, validate_window_row


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-dataset", type=int, default=1)
    parser.add_argument("--scene-visibility-device", default="cpu")
    parser.add_argument("--interhand-contact-compute-device", default="cpu")
    args = parser.parse_args()
    if args.limit_per_dataset < 1:
        raise ValueError("--limit-per-dataset must be positive")

    selected: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for line in args.windows.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        dataset, _, frame_ids = validate_window_row(row)
        if len(frame_ids) != 60 or row.get("window_stride") != 60 or row.get("window_overlap") != 0:
            raise ValueError(f"{row.get('window_id')}: expected strict 60-frame non-overlapping window")
        if counts[dataset] < args.limit_per_dataset:
            selected.append(row)
            counts[dataset] += 1
    if set(counts) != set(DATASET_LOADERS):
        raise ValueError(f"window manifest lacks datasets: {sorted(set(DATASET_LOADERS) - set(counts))}")

    bridge = SixDatasetGroundTruth(
        _roots(args.root),
        args.mano_dir,
        scene_visibility_device=args.scene_visibility_device,
        interhand_contact_compute_device=args.interhand_contact_compute_device,
    )
    report = {
        "windows": str(args.windows),
        "selected_count_per_dataset": dict(counts),
        "audits": [bridge.audit_window(row) for row in selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
