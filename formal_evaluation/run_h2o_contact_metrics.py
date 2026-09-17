#!/usr/bin/env python3
"""Build an exact Contact prediction index and evaluate both methods."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _prediction_directories(root: Path, method: str) -> list[Path]:
    formal = root / method / "formal"
    try:
        return [child for child in formal.iterdir() if child.is_dir()
                and (child / "metadata.json").is_file()
                and (child / "predictions.npz").is_file()]
    except OSError:
        return []


def _canonical_prediction_window_id(window_id: str, aliases: dict[str, str]) -> str:
    return aliases.get(window_id, window_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("h2o", "taco", "hot3d", "arctic", "oakink_v2"), default="h2o")
    parser.add_argument("--expected-windows", type=int, default=283)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--prediction-root", action="append", required=True,
                        help="METHOD=PATH; repeat for every shard")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--existing-gt-index", type=Path)
    parser.add_argument("--existing-gt-root", type=Path)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "report.json").exists():
        raise FileExistsError("refusing to overwrite an existing H2O Contact report")
    if bool(args.existing_gt_index) != bool(args.existing_gt_root):
        raise ValueError("--existing-gt-index and --existing-gt-root must be supplied together")
    existing_gt_rows = None
    if args.existing_gt_index:
        existing_gt_rows = [
            json.loads(line) for line in args.existing_gt_index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    windows = []
    window_aliases: dict[str, str] = {}
    if existing_gt_rows is not None:
        for row in existing_gt_rows:
            window_id = str(row["window_id"])
            window_aliases[str(row["cache_id"])] = window_id
            windows.append({
                "dataset": args.dataset, "sequence_id": row["sequence_id"],
                "window_id": window_id, "frame_ids": row["frame_ids"],
                "window_size": 60, "window_stride": 60, "window_overlap": 0,
            })
    else:
        for line in args.input_index.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            record = json.loads(Path(item["window_input"]).read_text(encoding="utf-8"))
            window_id = str(record["window_id"])
            window_aliases[str(record.get("cache_id", window_id))] = window_id
            windows.append({
                "dataset": args.dataset, "sequence_id": record["sequence_id"],
                "window_id": window_id, "frame_ids": record["frame_ids"],
                "window_size": 60, "window_stride": 60, "window_overlap": 0,
            })
    if len(windows) != args.expected_windows:
        raise ValueError(
            f"{args.dataset} input index has {len(windows)}/{args.expected_windows} windows"
        )
    windows_path = args.output_root / "windows.jsonl"
    windows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in windows), encoding="utf-8")

    roots: dict[str, list[Path]] = {"s2contact": [], "contactopt": []}
    for value in args.prediction_root:
        method, path = value.split("=", 1)
        roots[method].append(Path(path))
    prediction_rows = []
    for method, method_roots in roots.items():
        found: dict[str, Path] = {}
        for root in method_roots:
            for directory in _prediction_directories(root, method):
                metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                window_id = _canonical_prediction_window_id(str(metadata["window_id"]), window_aliases)
                if metadata.get("dataset") != args.dataset or window_id in found:
                    raise ValueError(f"invalid or duplicate {method} prediction: {window_id}")
                found[window_id] = directory
        if len(found) != args.expected_windows:
            raise ValueError(
                f"{method} has {len(found)}/{args.expected_windows} predictions"
            )
        prediction_rows.extend({"method": method, "dataset": args.dataset, "window_id": window_id,
                                "prediction_dir": str(directory)}
                               for window_id, directory in sorted(found.items()))
    prediction_index = args.output_root / "predictions_index.jsonl"
    prediction_index.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows), encoding="utf-8")

    gt_root = args.output_root / "gt_cache"
    if existing_gt_rows is not None:
        rows = existing_gt_rows
        if len(rows) != args.expected_windows:
            raise ValueError(f"existing GT index has {len(rows)}/{args.expected_windows} rows")
        for row in rows:
            for key in ("array_path", "metadata_path"):
                path = args.existing_gt_root / args.dataset / Path(row[key]).name
                if not path.is_file():
                    raise FileNotFoundError(path)
                row[key] = str(path)
        gt_root.mkdir(parents=True, exist_ok=True)
        (gt_root / "index_shard_000_of_001.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    else:
        roots_args = [
            f"h2o={args.data_root / 'H2O/h2o_data'}",
            f"hot3d={args.data_root / 'HOT3D/hot3d/hot3d/dataset'}",
            f"arctic={args.data_root / 'EgoForce/ARCTIC'}",
            f"oakink_v2={args.data_root / 'OakInk-v2'}",
            "taco=/mnt/cpfs/sjc/DATA/TACO_resized",
            f"hoi4d={args.data_root / 'HOI4D'}",
        ]
        build = [sys.executable, str(args.runtime_root / "formal_evaluation/build_six_dataset_gt_cache.py"),
                 "--windows", str(windows_path), "--mano-dir", str(args.mano_dir),
                 "--output-root", str(gt_root), "--datasets", args.dataset]
        for value in roots_args:
            build.extend(("--root", value))
        subprocess.run(build, check=True)

    source = json.loads((args.worktree / "formal_evaluation/config/methods_v1.json").read_text(encoding="utf-8"))["methods"]
    methods_path = args.output_root / "methods_contact2.json"
    methods_path.write_text(json.dumps({"methods": {name: source[name] for name in roots}}, indent=2) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, str(args.runtime_root / "formal_evaluation/evaluate_six_dataset.py"),
        "--gt-index", str(gt_root / "index_shard_000_of_001.jsonl"),
        "--prediction-index", str(prediction_index), "--methods-config", str(methods_path),
        "--report-path", str(args.output_root / "report.json"),
    ], check=True)


if __name__ == "__main__":
    main()
