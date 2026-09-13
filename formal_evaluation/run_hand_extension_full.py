#!/usr/bin/env python3
"""Run exact whole-window shards for the two RGB hand extensions."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def select_windows(rows, dataset, shard, shards):
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("invalid shard")
    selected = [r for r in rows if r["dataset"] == dataset]
    if len({r["window_id"] for r in selected}) != len(selected):
        raise ValueError("duplicate window identity")
    return selected[shard::shards]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".incoming")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def readonly_loaders(root, roots):
    from egohandmetric_prompt.data import datasets
    for name, read, signature in [
        ("_load_or_build_light_index_cache", "_read_light_index_cache", "_light_index_source_signature"),
        ("_load_or_build_sequence_entry_cache", "_read_sequence_entry_cache", "_light_index_source_signature"),
    ]:
        original = getattr(datasets, name)
        reader = getattr(datasets, read)
        source_signature = getattr(datasets, signature)
        def cached(cache_path, source_paths, build, _original=original, _reader=reader,
                   _signature=source_signature, **kwargs):
            paths = list(source_paths)
            signatures = [_signature(p) for p in paths]
            value = _reader(cache_path, signatures, **kwargs)
            if value is not None:
                return value
            local = root / "loader_cache" / (hashlib.sha256(str(cache_path).encode()).hexdigest() + ".pkl")
            local.parent.mkdir(exist_ok=True)
            return _original(local, paths, build, **kwargs)
        setattr(datasets, name, cached)
    datasets._hot3d_rectified_rgb_cache_path = lambda *a, **k: None
    datasets.OakInkV2FrameDataset._write_oakink_runtime_preview = lambda self, path, preview: self._oakink_runtime_preview_projection(preview)
    protected = [os.path.realpath(p) for p in roots.values()]
    def guard(event, args):
        if event == "open":
            path, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and isinstance(path, (str, bytes)):
                value = os.path.realpath(os.fsdecode(path))
                if any(value == p or value.startswith(p + "/") for p in protected):
                    raise PermissionError("dataset write forbidden: " + value)
    sys.addaudithook(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root = Path(spec["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "summary.json").exists():
        raise FileExistsError("run already has a summary; use a new output root")
    manifest = Path(spec["manifest"])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == spec["manifest_sha256"]
    all_rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    dataset = spec["dataset"]
    rows = select_windows(all_rows, dataset, spec.get("shard", 0), spec.get("shards", 1))
    assert len(rows) == spec["expected_windows"]
    assert all(len(r["frame_ids"]) == 60 for r in rows)
    from formal_evaluation.datasets.egofound3r_gt import VENDORED_DATALOADER_ROOT, DATASET_LOADERS, SixDatasetGroundTruth
    from formal_evaluation.run_egoforce_six_smoke import _materialize_rgb_input
    from formal_evaluation.datasets.six_dataset_gt_cache import window_cache_id, write_window_cache, VISIBILITY_CACHE_VERSION
    from formal_evaluation.common.schema import validate_comparison_output
    sys.path.insert(0, str(VENDORED_DATALOADER_ROOT))
    readonly_loaders(root, spec["roots"])
    from egohandmetric_prompt.data.stages import build_named_frame_dataset
    import numpy as np
    name = spec["method"]
    code = Path(spec["code_root"])
    method = json.loads((code / "formal_evaluation/config/baseline_runtime_registry_dsw.json").read_text())["methods"][name]
    summary = dict(status="preparing", method=name, dataset=dataset, expected_windows=len(rows),
                   completed_windows=0, failed_windows=[], zero_detection_windows=0, valid_hands=0,
                   shard=spec.get("shard", 0), shards=spec.get("shards", 1))
    save(root / "summary.json", summary)
    print("DATASET_BUILD_START", dataset, flush=True)
    frames = build_named_frame_dataset(DATASET_LOADERS[dataset], root_override=spec["roots"][dataset], split="all", load_rgb=True, load_depth=False)
    print("DATASET_BUILD_COMPLETE", dataset, flush=True)
    bridge = None
    gt_rows = []
    if spec.get("gt_index"):
        from formal_evaluation.run_metrics_v2_aggregation import remap_gt_index
        remap_gt_index(Path(spec["gt_index"]), root / "gt_index.jsonl")
        gt_rows = [json.loads(l) for l in (root / "gt_index.jsonl").read_text().splitlines() if l.strip()]
        wanted = {r["window_id"] for r in rows}
        gt_rows = [r for r in gt_rows if r["window_id"] in wanted]
        assert len(gt_rows) == len(rows)
    elif name == "hand_visibility_detector":
        bridge = SixDatasetGroundTruth(spec["roots"], Path("/mnt/workspace/sjc/models/human"), datasets=[dataset], scene_visibility_device="cuda:0", interhand_contact_compute_device="cuda:0")
    predictions = []
    shared = Path(spec["input_root"])
    shared.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    for index, row in enumerate(rows):
        started = time.monotonic()
        cache_id = window_cache_id(row)
        print("WINDOW_START", dataset, index + 1, len(rows), cache_id, flush=True)
        try:
            with (shared / (cache_id + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                record_path = shared / "inputs" / dataset / cache_id / "window_input.json"
                if not record_path.exists():
                    record_path = _materialize_rgb_input(frames, row, shared)
                record = json.loads(record_path.read_text())
                assert record["frame_ids"] == row["frame_ids"] and record["window_id"] == row["window_id"]
            command = [method["python"], str(code / "formal_evaluation/hand/adapters" / ("run_egoforce_baseline.py" if name == "egoforce" else "run_hand_visibility_detector.py")), "--phase", "formal", "--window-input", str(record_path), "--output-root", str(root), "--device", "cuda:0", "--checkpoint", spec["checkpoint"]]
            if name == "egoforce":
                command.extend(["--source-root", method["source_root"]])
            with (root / "logs" / (cache_id + ".log")).open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
            pred = root / name / "formal" / cache_id
            metadata = json.loads((pred / "metadata.json").read_text())
            with np.load(pred / "predictions.npz", allow_pickle=False) as a:
                arrays = dict(a)
            validate_comparison_output(metadata, arrays)
            assert metadata["frame_ids"] == row["frame_ids"] and metadata["processed_resolution_hw"] == [256, 256]
            valid = int(arrays["hand_valid"].sum())
            summary["valid_hands"] += valid
            summary["zero_detection_windows"] += int(valid == 0)
            predictions.append(dict(method=name, dataset=dataset, window_id=row["window_id"], prediction_dir=str(pred)))
            if bridge is not None:
                entry = write_window_cache(root / "gt_cache", row, bridge.batch_for_window(row), geometry_frames=bridge.geometry_for_window(row), cache_version=VISIBILITY_CACHE_VERSION)
                entry["metadata_path"] = str(Path(entry["array_path"]).with_suffix(".json"))
                gt_rows.append(entry)
            summary["completed_windows"] += 1
        except Exception:
            summary["failed_windows"].append(dict(window_id=row["window_id"], traceback=traceback.format_exc()))
            print(summary["failed_windows"][-1]["traceback"], flush=True)
        (root / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
        (root / "gt_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gt_rows))
        summary.update(status="running", last_window_seconds=time.monotonic() - started)
        save(root / "summary.json", summary)
        print("WINDOW_RESULT", json.dumps(summary), flush=True)
    if summary["failed_windows"]:
        summary["status"] = "failed"
        save(root / "summary.json", summary)
        raise SystemExit(1)
    assert len(predictions) == len(gt_rows) == len(rows)
    (root / "INFERENCE_COMPLETE").write_text("All exact windows inferred and schema validated\n")
    config = {"group": ["hand"], "scale_type": "metric_hand_camera"} if name == "egoforce" else {"group": ["contact"], "scale_type": "not_applicable"}
    save(root / "methods.json", {"methods": {name: config}})
    subprocess.run([sys.executable, str(code / "formal_evaluation/evaluate_six_dataset.py"), "--gt-index", str(root / "gt_index.jsonl"), "--prediction-index", str(root / "predictions.jsonl"), "--methods-config", str(root / "methods.json"), "--report-path", str(root / "report.json")], check=True)
    report = json.loads((root / "report.json").read_text())
    assert report["gt_windows"] == len(rows) and report["methods"][name]["missing_prediction_windows"] == 0
    summary["status"] = "complete"
    save(root / "summary.json", summary)
    (root / "COMPLETE").write_text("Exact window inference and metrics complete\n")


if __name__ == "__main__":
    main()
