#!/usr/bin/env python3
"""Flatten method-neutral RGB windows into InteractVLM's official JSONL contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from formal_evaluation.datasets.window_inputs import load_window_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-input-index", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_manifest}")
    rows: list[dict[str, object]] = []
    for line in args.window_input_index.read_text(encoding="utf-8").splitlines():
        record = load_window_input(Path(str(json.loads(line)["window_input"])))
        for time_index, (rgb_path, object_name, frame_id) in enumerate(zip(
            record["rgb_paths"], record["object_names"], record["frame_ids"], strict=True
        )):
            rows.append({
                "rgb_path": rgb_path,
                "object_name": object_name,
                "dataset": record["dataset"],
                "sequence": record["sequence_id"],
                "window_id": record["window_id"],
                "window_cache_id": record["cache_id"],
                "time_index": time_index,
                "frame_id": frame_id,
            })
    if not rows:
        raise ValueError("window input index is empty")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"frames": len(rows), "output_manifest": str(args.output_manifest)}))


if __name__ == "__main__":
    main()
