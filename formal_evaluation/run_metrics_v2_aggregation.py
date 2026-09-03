#!/usr/bin/env python3
"""Build one exact standard10 prediction index, then run the v2 evaluator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STANDARD10 = (
    "egofound3r", "wilor", "hawor", "pad_hand", "reviv4d",
    "vggt", "pi3", "da3_large_1_1", "lingbot_map_long", "vggt_omega",
)


def outputs(root: Path) -> list[Path]:
    found: list[Path] = []
    for formal in (root / "formal", root / root.name / "formal"):
        try:
            found.extend(child for child in formal.iterdir()
                         if child.is_dir() and (child / "metadata.json").is_file()
                         and (child / "predictions.npz").is_file())
        except OSError:
            continue
    return found


def remap_gt_index(source: Path, destination: Path) -> int:
    """Rewrite stale pre-migration cache paths to the index's current GT root."""
    rows = []
    marker = "/gt_cache/"
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("metadata_path", "array_path"):
            value = str(row[field])
            if marker in value:
                row[field] = str(source.parent / value.split(marker, 1)[1])
        rows.append(row)
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-windows", required=True, type=int)
    parser.add_argument("--main-root", required=True, type=Path)
    parser.add_argument("--history-root", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--gt-index", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--scene-only", action="store_true",
                        help="Evaluate only scene pose metrics for every selected method.")
    parser.add_argument("--method", action="append", choices=STANDARD10,
                        help="Evaluate only this method; repeat as needed (default: all standard10 methods).")
    args = parser.parse_args()
    methods = tuple(dict.fromkeys(args.method or STANDARD10))

    gt_rows = [json.loads(line) for line in args.gt_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    allowed_window_ids = {str(row["window_id"]) for row in gt_rows}

    histories: dict[str, Path] = {}
    for value in args.history_root:
        method, raw_path = value.split("=", 1)
        if method not in methods or method in histories:
            raise ValueError(f"invalid history root: {value}")
        histories[method] = Path(raw_path)

    index: dict[tuple[str, str], Path] = {}
    counts: dict[str, int] = {}
    for method in methods:
        roots = ([histories[method]] if method in histories else []) + [args.main_root / method]
        for root in roots:
            for directory in outputs(root):
                metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                if metadata.get("dataset") != args.dataset:
                    continue
                window_id = str(metadata["window_id"])
                if window_id not in allowed_window_ids:
                    continue
                key = (method, window_id)
                if key in index:
                    continue
                index[key] = directory
        count = sum(key[0] == method for key in index)
        counts[method] = count

    args.work_dir.mkdir(parents=True, exist_ok=True)
    remapped_gt_index = args.work_dir / "gt_index_current_root.jsonl"
    gt_windows = remap_gt_index(args.gt_index, remapped_gt_index)
    failures = [f"{method}: {count}/{args.expected_windows} canonical predictions"
                for method, count in counts.items() if count != args.expected_windows]
    if gt_windows != args.expected_windows:
        failures.append(f"GT cache: {gt_windows}/{args.expected_windows} windows")
    if args.audit_path is not None:
        args.audit_path.parent.mkdir(parents=True, exist_ok=True)
        args.audit_path.write_text(json.dumps({
            "dataset": args.dataset,
            "expected_windows": args.expected_windows,
            "gt_windows": gt_windows,
            "method_prediction_counts": counts,
            "status": "complete" if not failures else "incomplete",
            "failures": failures,
        }, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise ValueError("; ".join(failures))

    prediction_index = args.work_dir / "predictions_index.jsonl"
    with prediction_index.open("w", encoding="utf-8") as handle:
        for (method, window_id), directory in sorted(index.items()):
            handle.write(json.dumps({"method": method, "dataset": args.dataset,
                                     "window_id": window_id, "prediction_dir": str(directory)}) + "\n")
    source_config = json.loads((args.worktree / "formal_evaluation/config/methods_v1.json").read_text(encoding="utf-8"))
    selected_config = {method: dict(source_config["methods"][method]) for method in methods}
    for method, config in selected_config.items():
        config["group"] = (["scene"] if args.scene_only else
                           ["hand", "scene"] if method in {"egofound3r", "reviv4d"} else
                           ["hand"] if method == "hawor" else ["scene"])
        if "scene" in config["group"]:
            config["scene_pose_only"] = True
    config_path = args.work_dir / "methods_selected.json"
    config_path.write_text(json.dumps({"methods": selected_config}, indent=2) + "\n", encoding="utf-8")
    evaluator = Path(__file__).with_name("evaluate_six_dataset.py")
    if not evaluator.is_file():
        evaluator = args.worktree / "formal_evaluation/evaluate_six_dataset.py"
    subprocess.run([
        sys.executable, str(evaluator),
        "--gt-index", str(remapped_gt_index), "--prediction-index", str(prediction_index),
        "--methods-config", str(config_path), "--report-path", str(args.report_path),
    ], check=True)


if __name__ == "__main__":
    main()
