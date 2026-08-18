#!/usr/bin/env python3
"""Add contiguous RGB context clips to existing six-dataset window inputs."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formal_evaluation.datasets.egofound3r_gt import DATASET_LOADERS, SixDatasetGroundTruth
from formal_evaluation.datasets.window_inputs import materialize_video_context

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-input-index", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=60)
    parser.add_argument("--sentinel", type=Path,
                        help="write only after every context mapping for this input index is complete")
    args = parser.parse_args()
    roots = dict(value.split("=", 1) for value in args.root)
    if set(roots) != set(DATASET_LOADERS):
        raise ValueError(f"--root must name exactly {sorted(DATASET_LOADERS)}")
    bridge = SixDatasetGroundTruth(roots, args.mano_dir)
    records = [line for line in args.window_input_index.read_text(encoding="utf-8").splitlines() if line]
    for line in records:
        path = Path(str(json.loads(line)["window_input"]))
        print(json.dumps({"context_mapping": str(materialize_video_context(bridge, path, context_frames=args.context_frames))}), flush=True)
    if args.sentinel is not None:
        if args.sentinel.exists():
            raise FileExistsError(f"refusing to replace context sentinel: {args.sentinel}")
        args.sentinel.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.sentinel.parent, delete=False) as handle:
            json.dump({
                "status": "complete",
                "context_frames": args.context_frames,
                "window_count": len(records),
                "window_input_index": str(args.window_input_index),
            }, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(args.sentinel)

if __name__ == "__main__":
    main()
