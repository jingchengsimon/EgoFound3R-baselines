#!/usr/bin/env python3
"""Run one dataset through HaWoR while requiring its native camera trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formal_evaluation.common.schema import validate_comparison_output
from formal_evaluation.datasets.six_dataset_gt_cache import window_cache_id


def load_rgb_window_input(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text())
    frame_ids, rgb_paths, intrinsics = (
        record.get("frame_ids"), record.get("rgb_paths"), record.get("intrinsics")
    )
    if record.get("input_kind") != "rgb_intrinsics_only_no_geometry":
        raise ValueError(f"unexpected HaWoR window input kind: {path}")
    if not isinstance(frame_ids, list) or not isinstance(rgb_paths, list) or not isinstance(intrinsics, list):
        raise ValueError(f"invalid HaWoR RGB window input: {path}")
    if len(frame_ids) != len(rgb_paths) or len(frame_ids) != len(intrinsics):
        raise ValueError(f"HaWoR RGB window frame count mismatch: {path}")
    return record


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def validate_native_output(path: Path, row: dict[str, object]) -> None:
    metadata = json.loads((path / "metadata.json").read_text())
    run = json.loads((path / "run.json").read_text())
    with np.load(path / "predictions.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    validate_comparison_output(metadata, arrays)
    if metadata["method"] != "hawor" or metadata["dataset"] != row["dataset"]:
        raise ValueError("method/dataset identity mismatch")
    if metadata["window_id"] != row["window_id"] or metadata["frame_ids"] != row["frame_ids"]:
        raise ValueError("window/frame identity mismatch")
    if run.get("status") != "success":
        raise ValueError("run.json is not successful")
    detail = metadata.get("detail", {})
    if detail.get("slam_failed_identity_fallback") is not False:
        raise ValueError("identity camera fallback was emitted")
    camera = arrays["camera_c2w"]
    valid = arrays["camera_valid"]
    if not valid.all() or not np.isfinite(camera).all():
        raise ValueError("native camera trajectory is invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root = Path(spec["output_root"])
    if root.exists():
        unexpected = [path.name for path in root.iterdir() if path.name not in {"control", "run.log"}]
        if unexpected:
            raise FileExistsError(f"refusing to reuse populated output root {root}: {unexpected}")
    else:
        root.mkdir(parents=True)

    manifest = Path(spec["manifest"])
    raw_manifest = manifest.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != spec["manifest_sha256"]:
        raise ValueError("frozen manifest SHA256 mismatch")
    rows = [json.loads(line) for line in raw_manifest.splitlines() if line]
    rows = [row for row in rows if row["dataset"] == spec["dataset"]]
    if len(rows) != spec["expected_windows"]:
        raise ValueError(f"window count mismatch: {len(rows)} != {spec['expected_windows']}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]

    gt_index = Path(spec["gt_index"])
    if hashlib.sha256(gt_index.read_bytes()).hexdigest() != spec["gt_sha256"]:
        raise ValueError("GT index SHA256 mismatch")

    input_root = Path(spec["input_root"])
    prepared = []
    for row in rows:
        cache_id = window_cache_id(row)
        path = input_root / "inputs" / spec["dataset"] / cache_id / "window_input.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen RGB window input: {path}")
        record = load_rgb_window_input(path)
        if record["window_id"] != row["window_id"] or record["frame_ids"] != row["frame_ids"]:
            raise ValueError(f"window input identity mismatch: {path}")
        if not all(Path(item).is_file() for item in record["rgb_paths"]):
            raise FileNotFoundError(f"one or more RGB frames are missing for {path}")
        prepared.append((row, cache_id, path))

    summary = {
        "status": "running",
        "dataset": spec["dataset"],
        "expected_windows": len(rows),
        "completed_windows": 0,
        "failed_windows": [],
        "native_camera_required": True,
        "existing_hawor_roots_preserved": spec["existing_hawor_roots"],
    }
    atomic_json(root / "summary.json", summary)
    (root / "logs").mkdir()
    predictions = []
    for index, (row, cache_id, window_input) in enumerate(prepared, 1):
        output = root / "hawor" / "formal" / cache_id
        command = [
            spec["python"], spec["adapter"],
            "--phase", "formal",
            "--window-input", str(window_input),
            "--methods-config", spec["methods_config"],
            "--source-root", spec["source_root"],
            "--checkpoint", spec["checkpoint"],
            "--infiller-weight", spec["infiller_weight"],
            "--detector-weight", spec["detector_weight"],
            "--output-root", str(root),
            "--device", "cuda:0",
            "--require-native-camera",
        ]
        started = time.monotonic()
        try:
            with (root / "logs" / f"{cache_id}.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=spec.get("window_timeout_seconds", 3600))
            validate_native_output(output, row)
            predictions.append({
                "method": "hawor", "dataset": spec["dataset"],
                "window_id": row["window_id"], "prediction_dir": str(output),
            })
            summary["completed_windows"] += 1
        except Exception:
            summary["failed_windows"].append({
                "window_id": row["window_id"], "cache_id": cache_id,
                "traceback": traceback.format_exc(),
            })
        summary["last_window_seconds"] = time.monotonic() - started
        (root / "predictions.jsonl").write_text("".join(json.dumps(item) + "\n" for item in predictions))
        atomic_json(root / "summary.json", summary)
        print(json.dumps({
            "dataset": spec["dataset"], "window": index, "total": len(rows),
            "completed": summary["completed_windows"],
            "failed": len(summary["failed_windows"]),
        }), flush=True)
        if summary["failed_windows"] and spec.get("stop_on_first_failure", True):
            break

    if summary["failed_windows"] or summary["completed_windows"] != len(rows):
        summary["status"] = "failed"
        atomic_json(root / "summary.json", summary)
        raise SystemExit(1)

    (root / "INFERENCE_COMPLETE").write_text("all exact windows have valid native HaWoR camera trajectories\n")
    if args.limit is not None:
        summary["status"] = "complete"
        summary["pilot"] = True
        atomic_json(root / "summary.json", summary)
        (root / "COMPLETE").write_text("native HaWoR camera pilot complete\n")
        return

    methods = root / "methods.json"
    atomic_json(methods, {"methods": {"hawor": {"group": ["hand"], "scale_type": "metric_hand_camera"}}})
    subprocess.run([
        spec["python"], spec["evaluator"],
        "--gt-index", str(gt_index),
        "--prediction-index", str(root / "predictions.jsonl"),
        "--methods-config", str(methods),
        "--report-path", str(root / "report.json"),
    ], check=True)
    report = json.loads((root / "report.json").read_text())
    if report["gt_windows"] != len(rows) or report["methods"]["hawor"]["missing_prediction_windows"] != 0:
        raise ValueError("evaluation report coverage mismatch")
    summary["status"] = "complete"
    atomic_json(root / "summary.json", summary)
    (root / "COMPLETE").write_text("native HaWoR inference and camera metrics complete\n")


if __name__ == "__main__":
    main()
