#!/usr/bin/env python3
"""Infer native-camera HaWoR only for jump-cut windows absent from frozen sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from formal_evaluation.run_hawor_native_camera_dataset import atomic_json, validate_native_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root = Path(spec["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(line) for line in Path(spec["stitched_manifest"]).read_text().splitlines()
               if line.strip()]
    if len(entries) != 114:
        raise ValueError(f"STITCHED_SEGMENT_COUNT:{len(entries)}")

    refs = {(ref["cache_id"], ref["window_id"])
            for entry in entries if entry["dataset"] == spec["dataset"]
            for ref in entry["frame_refs"]}
    old_root = Path(spec["old_root"]) / spec["dataset"]
    missing = sorted((cache, window_id) for cache, window_id in refs
                     if not (old_root / cache / "predictions.npz").is_file())
    if len(missing) != spec["expected_windows"]:
        raise ValueError(f"MISSING_COUNT:{len(missing)}:{spec['expected_windows']}")

    prepared = []
    input_root = Path(spec["prepared_root"])
    for cache, window_id in missing:
        path = input_root / cache / "window_input.json"
        record = json.loads(path.read_text())
        if record["dataset"] != spec["dataset"] or record["window_id"] != window_id:
            raise ValueError(f"WINDOW_INPUT_IDENTITY:{path}")
        if not all(isinstance(record.get(key), list)
                   for key in ("frame_ids", "rgb_paths", "intrinsics")):
            raise ValueError(f"WINDOW_INPUT_FIELDS:{path}")
        if not (len(record["frame_ids"]) == len(record["rgb_paths"])
                == len(record["intrinsics"])):
            raise ValueError(f"WINDOW_INPUT_LENGTHS:{path}")
        absent_rgb = [item for item in record["rgb_paths"] if not Path(item).is_file()]
        if absent_rgb:
            raise FileNotFoundError(f"RGB_MISSING:{path}:{len(absent_rgb)}:{absent_rgb[0]}")
        prepared.append((cache, record, path))

    summary = {"status": "running", "dataset": spec["dataset"],
               "expected_windows": len(prepared), "completed_windows": 0,
               "failed_windows": [], "native_camera_required": True}
    atomic_json(root / "summary.json", summary)
    (root / "logs").mkdir(exist_ok=True)
    predictions = []
    for index, (cache, row, window_input) in enumerate(prepared, 1):
        output = root / "hawor" / "formal" / cache
        log_path = root / "logs" / f"{cache}.log"
        command = [
            spec["python"], spec["adapter"], "--phase", "formal",
            "--window-input", str(window_input), "--methods-config", spec["methods_config"],
            "--source-root", spec["source_root"], "--checkpoint", spec["checkpoint"],
            "--infiller-weight", spec["infiller_weight"],
            "--detector-weight", spec["detector_weight"], "--output-root", str(root),
            "--device", "cuda:0", "--require-native-camera",
        ]
        started = time.monotonic()
        try:
            with log_path.open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=spec.get("window_timeout_seconds", 3600))
            validate_native_output(output, row)
            predictions.append({"method": "hawor", "dataset": spec["dataset"],
                                "window_id": row["window_id"],
                                "prediction_dir": str(output)})
            summary["completed_windows"] += 1
        except Exception:
            summary["failed_windows"].append({
                "window_id": row["window_id"], "cache_id": cache,
                "traceback": traceback.format_exc(), "log": str(log_path),
                "log_tail": "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])[-12000:]
                if log_path.exists() else "",
            })
        summary["last_window_seconds"] = time.monotonic() - started
        (root / "predictions.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in predictions))
        atomic_json(root / "summary.json", summary)
        print(json.dumps({"dataset": spec["dataset"], "window": index,
                          "total": len(prepared), "completed": summary["completed_windows"],
                          "failed": len(summary["failed_windows"])}), flush=True)
        if summary["failed_windows"]:
            break

    if summary["failed_windows"] or summary["completed_windows"] != len(prepared):
        summary["status"] = "failed"
        atomic_json(root / "summary.json", summary)
        raise SystemExit(1)
    summary["status"] = "complete"
    atomic_json(root / "summary.json", summary)
    (root / "INFERENCE_COMPLETE").write_text("all missing jump-cut windows inferred\n")
    (root / "COMPLETE").write_text("native-camera HaWoR jump-cut inference complete\n")


if __name__ == "__main__":
    main()
