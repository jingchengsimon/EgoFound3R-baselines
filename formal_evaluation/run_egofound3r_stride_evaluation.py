#!/usr/bin/env python3
"""Run one registered fixed stride using existing inference and metric entrypoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from formal_evaluation.datasets.window_inputs import load_window_input
from formal_evaluation.run_metrics_v2_aggregation import remap_gt_index


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare_inputs(dataset: str, spec: dict, root: Path) -> tuple[Path, Path, list[dict]]:
    """Select precisely the registered GT windows, rejecting incomplete input coverage."""
    root.mkdir(parents=True, exist_ok=True)
    gt = root / "gt_index.jsonl"
    source_gt = Path(spec["gt_index"])
    remap_gt_index(source_gt, gt)
    gt_rows = read_rows(gt)
    expected = int(spec["expected_windows"])
    wanted = {str(row["window_id"]) for row in gt_rows}
    if len(gt_rows) != expected or len(wanted) != expected:
        raise ValueError(f"{dataset}: GT count/identity mismatch, expected {expected}")
    for row in gt_rows:
        for field in ("metadata_path", "array_path"):
            with Path(row[field]).open("rb") as handle:
                if not handle.read(1):
                    raise ValueError(f"empty GT artifact: {row[field]}")
    selected = {}
    for raw in spec["input_indices"]:
        index = Path(raw)
        status = json.loads(index.with_suffix(".status.json").read_text())
        rows = read_rows(index)
        if status.get("status") != "complete" or status.get("window_count") != len(rows):
            raise ValueError(f"incomplete input shard: {index}")
        for row in rows:
            record = load_window_input(Path(row["window_input"]))
            key = str(record["window_id"])
            if record["dataset"] != dataset or key not in wanted:
                continue
            if key in selected:
                raise ValueError(f"duplicate input window: {dataset}:{key}")
            if len(record["frame_ids"]) != 60:
                raise ValueError(f"non-60-frame input: {dataset}:{key}")
            for path in record["rgb_paths"]:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            selected[key] = {**record, "window_input": row["window_input"]}
    if set(selected) != wanted:
        raise ValueError(f"{dataset}: input coverage {len(selected)}/{expected}")
    records = [selected[str(row["window_id"])] for row in gt_rows]
    index = root / "inputs.jsonl"
    index.write_text("".join(json.dumps({"window_input": row["window_input"]}) + "\n" for row in records))
    index.with_suffix(".status.json").write_text(json.dumps({
        "status": "complete", "index": str(index), "window_count": expected,
    }))
    (root / "input_audit.json").write_text(json.dumps({
        "dataset": dataset, "windows": expected, "source_gt": str(source_gt),
        "source_gt_sha256": hashlib.sha256(source_gt.read_bytes()).hexdigest(),
        "window_identity_sha256": hashlib.sha256(json.dumps([
            [r["window_id"], r["frame_ids"]] for r in records
        ], sort_keys=True).encode()).hexdigest(),
    }, indent=2))
    return index, gt, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--global-stride", required=True, type=int, choices=(1, 2, 5))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    worktree = Path(__file__).resolve().parents[1]
    args.work_root.mkdir(parents=True, exist_ok=True)
    prepared = {dataset: prepare_inputs(dataset, item, args.work_root / dataset)
                for dataset, item in spec["datasets"].items()}
    if args.preflight_only:
        print(json.dumps({"status": "preflight_complete", "windows": {
            d: len(value[2]) for d, value in prepared.items()}}), flush=True)
        return
    smoke_strides = set()
    for root in spec["smoke_roots"]:
        path = Path(root)
        summary = json.loads((path / "smoke_summary.json").read_text())
        if not (path / "COMPLETE").is_file() or summary.get("status") != "complete":
            raise ValueError(f"smoke gate not complete: {path}")
        if spec.get("require_metric_smoke"):
            if summary.get("dataset") not in spec["datasets"] or summary.get("metric_validation") != "complete":
                raise ValueError(f"dataset metric smoke not complete: {path}")
            report = json.loads(Path(summary["metric_report"]).read_text())
            method_report = report["methods"]["egofound3r"]
            if (report.get("gt_windows") != 1 or method_report.get("missing_prediction_windows") != 0
                    or method_report.get("datasets", {}).get(summary["dataset"], {}).get("n_windows") != 1):
                raise ValueError(f"dataset metric smoke coverage mismatch: {path}")
        for key in ("source_commit", "inference_commit", "checkpoint_sha256"):
            if summary.get(key) != spec[key]:
                raise ValueError(f"smoke model identity mismatch: {path}:{key}")
        stride = summary.get("global_stride")
        if summary.get("global_anchor_phase") != stride // 2:
            raise ValueError(f"smoke phase mismatch: {path}")
        smoke_strides.add(stride)
    required_smoke_strides = set(spec.get("required_smoke_strides", [1, 5]))
    if not required_smoke_strides or not required_smoke_strides.issubset({1, 2, 5}):
        raise ValueError("invalid required_smoke_strides")
    # Preserve the original formal gate; single-stride ablations explicitly require [5].
    if not required_smoke_strides.issubset(smoke_strides):
        raise ValueError(f"required smoke strides missing: {sorted(required_smoke_strides - smoke_strides)}")
    if "required_smoke_strides" in spec and args.global_stride not in required_smoke_strides:
        raise ValueError("requested stride must be included in required_smoke_strides")
    methods_path = Path(spec.get("methods_config", worktree / "formal_evaluation/config/methods_v1.json"))
    methods = json.loads(methods_path.read_text())
    method = methods["methods"]["egofound3r"]
    for key in ("source_commit", "inference_commit", "checkpoint_sha256"):
        if method.get(key) != spec[key]:
            raise ValueError(f"method configuration model identity mismatch: {key}")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    selected_methods = args.work_root / "methods.json"
    selected_methods.write_text(json.dumps({"methods": {"egofound3r": method}}))
    errors, reports = {}, {}
    for dataset, (index, gt, records) in prepared.items():
        root = args.output_root / dataset
        work = args.work_root / dataset
        runner_args = ["--phase", "formal", "--window-input", "{window_input}",
                       "--methods-config", str(selected_methods), "--config", spec["config"],
                       "--checkpoint", spec["checkpoint"], "--backbone-checkpoint", spec["backbone"],
                       "--output-root", "{output_root}", "--global-stride", str(args.global_stride)]
        try:
            subprocess.run([
                sys.executable, str(worktree / "formal_evaluation/run_formal_window_queue.py"),
                "--method", "egofound3r", "--python", sys.executable,
                "--runner", str(worktree / "formal_evaluation/scene/adapters/run_egofound3r_baseline.py"),
                *("--runner-arg=" + value for value in runner_args),
                "--input-index", str(index), "--output-root", str(root),
                "--queue-root", str(work / "queue"), "--gpus", os.environ["TASKCTL_GPU"],
            ], check=True)
            predictions = work / "predictions.jsonl"
            rows = []
            for record in records:
                directory = root / "egofound3r/formal" / str(record["cache_id"])
                metadata = json.loads((directory / "metadata.json").read_text())
                expected = {"global_stride": args.global_stride, "global_anchor_phase": args.global_stride // 2,
                            **{key: spec[key] for key in ("source_commit", "inference_commit", "checkpoint_sha256")}}
                if any(metadata.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"prediction provenance mismatch: {directory}")
                rows.append({"method": "egofound3r", "dataset": dataset,
                             "window_id": record["window_id"], "prediction_dir": str(directory)})
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = root / "report.json"
            subprocess.run([
                sys.executable, str(worktree / "formal_evaluation/evaluate_six_dataset.py"),
                "--gt-index", str(gt), "--prediction-index", str(predictions),
                "--methods-config", str(selected_methods), "--report-path", str(report),
            ], check=True)
            reports[dataset] = str(report)
            print(json.dumps({"dataset": dataset, "status": "complete", "report": str(report)}), flush=True)
        except Exception as error:
            errors[dataset] = repr(error)
            print(json.dumps({"dataset": dataset, "status": "failed", "error": repr(error)}), flush=True)
    summary = {"status": "failed" if errors else "complete", "reports": reports, "errors": errors,
               "global_stride": args.global_stride, "global_anchor_phase": args.global_stride // 2}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)
    (args.output_root / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
