#!/usr/bin/env python3
"""Aggregate Result3 camera metrics for HaWoR, Dyn-HaMR, and matched Ego windows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
from formal_evaluation.evaluate_six_dataset import _prediction_arrays


METRIC_KEYS = ("camera_ate_aligned", "camera_rot_error_deg", "camera_pose_auc_30")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scene(path: Path):
    spec = importlib.util.spec_from_file_location("result3_frozen_scene", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aliases(rows: list[dict]) -> dict[str, str]:
    result = {}
    for row in rows:
        canonical = str(row["window_id"])
        result[canonical] = canonical
        if row.get("cache_id") is not None:
            result[str(row["cache_id"])] = canonical
    return result


def remap_gt_row(row: dict, index_parent: Path) -> dict:
    """Resolve migrated GT-cache references from the verified index location."""
    result = dict(row)
    for key in ("metadata_path", "array_path"):
        original = Path(str(row[key]))
        if original.is_file():
            continue
        marker = "/gt_cache/"
        if marker not in str(original):
            raise FileNotFoundError(original)
        candidate = index_parent / str(original).split(marker, 1)[1]
        if not candidate.is_file():
            raise FileNotFoundError(f"GT artifact unavailable at {original} or {candidate}")
        result[key] = str(candidate)
    return result


def indexed_predictions(path: Path, dataset: str, expected: int, mapping: dict[str, str]) -> dict[str, Path]:
    if not (path.parent / "COMPLETE").is_file():
        raise ValueError(f"prediction source COMPLETE absent: {path.parent}")
    found = {}
    for row in read_jsonl(path):
        if str(row.get("dataset", dataset)) != dataset:
            continue
        key = mapping.get(str(row["window_id"]), str(row["window_id"]))
        directory = Path(str(row["prediction_dir"]))
        if key in found and found[key] != directory:
            raise ValueError(f"duplicate indexed prediction: {dataset}:{key}")
        found[key] = directory
    if len(found) != expected:
        raise ValueError(f"prediction index coverage: {dataset}:{len(found)}/{expected}:{path}")
    return found


def directory_predictions(roots: list[str], dataset: str, expected: int,
                          mapping: dict[str, str], wanted: set[str]) -> dict[str, Path]:
    found = {}
    checked = []
    for raw in roots:
        root = Path(raw)
        if not root.is_dir():
            continue
        checked.append(str(root))
        for directory in root.iterdir():
            if not directory.is_dir():
                continue
            metadata_path = directory / "metadata.json"
            prediction_path = directory / "predictions.npz"
            if not metadata_path.is_file() or not prediction_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("dataset")) != dataset:
                continue
            key = mapping.get(str(metadata["window_id"]), str(metadata["window_id"]))
            if key not in wanted:
                continue
            if key in found and found[key] != directory:
                raise ValueError(f"duplicate directory prediction: {dataset}:{key}")
            found[key] = directory
    if len(found) != expected:
        raise ValueError(
            f"directory prediction coverage: {dataset}:{len(found)}/{expected}; roots={checked}"
        )
    return found


def p95_keep(path: Path, expected_ids: set[str]) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    masks = {
        str(window_id): ~np.asarray(excluded, dtype=bool)
        for window_id, excluded in zip(payload["window_ids"], payload["excluded"], strict=True)
    }
    if set(masks) != expected_ids or any(value.shape != (60,) for value in masks.values()):
        raise ValueError(f"P95 mask coverage/shape mismatch: {path}")
    return masks


def normalize(summary: dict) -> dict:
    result = {}
    for key, value in summary.items():
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            result[key] = None
        elif isinstance(value, np.integer):
            result[key] = int(value)
        else:
            result[key] = value
    return result


def aggregate_method(scene, *, method: str, dataset: str, gt_rows: dict[str, dict],
                     predictions: dict[str, Path], keep: dict[str, np.ndarray],
                     output: Path) -> tuple[dict, dict]:
    if set(gt_rows) != set(predictions) or set(gt_rows) != set(keep):
        raise ValueError(f"{dataset}:{method}: GT/prediction/mask identity mismatch")
    rows = []
    fallback_windows = 0
    for number, window_id in enumerate(sorted(gt_rows), 1):
        gt_metadata, target = load_window_cache(gt_rows[window_id])
        metadata, prediction = _prediction_arrays(predictions[window_id])
        if list(metadata["frame_ids"]) != list(gt_metadata["frame_ids"]):
            raise ValueError(f"frame identity mismatch: {dataset}:{method}:{window_id}")
        frozen = scene.freeze_scene(prediction, target)
        metric = scene.aggregate_scene_window(frozen, keep[window_id])
        row = {"dataset": dataset, "method": method, "window_id": window_id,
               "frame_ids": metadata["frame_ids"], **metric}
        rows.append(row)
        fallback_windows += int(bool(metadata.get("detail", {}).get("slam_failed_identity_fallback")))
        if number % 100 == 0:
            print(json.dumps({"stage": method, "dataset": dataset,
                              "processed": number, "total": len(gt_rows)}), flush=True)
    aggregate = normalize(aggregate_windows(rows, method=method))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")
    audit = {
        "windows": len(rows),
        "slam_failed_identity_fallback_windows": fallback_windows if method == "hawor" else 0,
        "undefined_window_counts": {
            key: aggregate.get(key + "_undefined_window_count", len(rows)) for key in METRIC_KEYS
        },
    }
    return aggregate, audit


def aggregate_existing_ego(dataset: str, source: Path, selected: set[str], output: Path) -> tuple[dict, dict]:
    found = {}
    for row in read_jsonl(source):
        key = str(row["window_id"])
        if key in selected:
            found[key] = {"dataset": dataset, "method": "egofound3r_stride5", **row}
    if set(found) != selected:
        raise ValueError(f"Ego matched coverage: {dataset}:{len(found)}/{len(selected)}")
    rows = [found[key] for key in sorted(found)]
    aggregate = normalize(aggregate_windows(rows, method="egofound3r_stride5"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")
    audit = {
        "windows": len(rows),
        "source": str(source),
        "source_sha256": sha256(source),
        "undefined_window_counts": {
            key: aggregate.get(key + "_undefined_window_count", len(rows)) for key in METRIC_KEYS
        },
    }
    return aggregate, audit


def run(spec: dict, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(output_root)
    scene_path = Path(spec["scene_script"])
    scene = load_scene(scene_path)
    output_root.mkdir(parents=True)
    report = {
        "status": "running",
        "protocol": {
            "scene_script": str(scene_path),
            "scene_script_sha256": sha256(scene_path),
            "camera_convention": "OpenCV camera_c2w",
            "ate": "per-window Sim(3)-aligned camera-center RMSE, then window macro mean",
            "rotation": "per-window first-valid-frame gauge, mean later-frame geodesic error, then window macro mean",
            "auc_30": "per-window PCK AUC of max rotation/translation-direction error, then window macro mean",
            "hawor_selection": "full 2378 Result3 windows with frozen All8 P95 masks",
            "dyn_and_ego_selection": "Dyn-HaMR original100 identities, unfiltered",
            "fit_on_full_unfiltered_window": True,
            "no_refit_after_filter": True,
        },
        "methods": {"hawor": {}, "dyn_hamr": {}, "egofound3r_stride5": {}},
        "audit": {},
        "sources": {},
    }
    total_hawor = total_dyn = 0
    for dataset, cfg in spec["datasets"].items():
        gt_index = Path(cfg["gt_index"])
        if sha256(gt_index) != cfg["gt_sha256"]:
            raise ValueError(f"GT index SHA256 mismatch: {dataset}")
        raw_gt = read_jsonl(gt_index)
        if len(raw_gt) != cfg["expected_windows"]:
            raise ValueError(f"GT coverage mismatch: {dataset}")
        gt_rows = {
            str(row["window_id"]): remap_gt_row(row, gt_index.parent)
            for row in raw_gt
        }
        mapping = aliases(raw_gt)
        wanted = set(gt_rows)

        hawor = directory_predictions(cfg["hawor_roots"], dataset, len(gt_rows), mapping, wanted)
        hawor_keep = p95_keep(Path(cfg["p95_mask"]), wanted)
        hawor_summary, hawor_audit = aggregate_method(
            scene, method="hawor", dataset=dataset, gt_rows=gt_rows,
            predictions=hawor, keep=hawor_keep,
            output=output_root / dataset / "hawor_all8_p95_window_metrics.jsonl",
        )
        report["methods"]["hawor"][dataset] = hawor_summary
        report["audit"].setdefault(dataset, {})["hawor"] = hawor_audit
        total_hawor += len(hawor)

        dyn_index = Path(cfg["dyn_prediction_index"])
        dyn = indexed_predictions(dyn_index, dataset, cfg["dyn_windows"], mapping)
        dyn_gt = {key: gt_rows[key] for key in dyn}
        dyn_keep = {key: np.ones(60, dtype=bool) for key in dyn}
        dyn_summary, dyn_audit = aggregate_method(
            scene, method="dyn_hamr", dataset=dataset, gt_rows=dyn_gt,
            predictions=dyn, keep=dyn_keep,
            output=output_root / dataset / "dyn_hamr_unfiltered_window_metrics.jsonl",
        )
        report["methods"]["dyn_hamr"][dataset] = dyn_summary
        report["audit"][dataset]["dyn_hamr"] = dyn_audit
        total_dyn += len(dyn)

        ego_summary, ego_audit = aggregate_existing_ego(
            dataset, Path(cfg["ego_unfiltered_metrics"]), set(dyn),
            output_root / dataset / "ego_stride5_dyn100_unfiltered_window_metrics.jsonl",
        )
        report["methods"]["egofound3r_stride5"][dataset] = ego_summary
        report["audit"][dataset]["egofound3r_stride5"] = ego_audit
        report["sources"][dataset] = {
            "gt_index": str(gt_index), "gt_index_sha256": cfg["gt_sha256"],
            "dyn_prediction_index": str(dyn_index), "dyn_prediction_index_sha256": sha256(dyn_index),
            "p95_mask": cfg["p95_mask"], "p95_mask_sha256": sha256(Path(cfg["p95_mask"])),
            "hawor_roots": cfg["hawor_roots"],
        }
        print(json.dumps({"stage": "dataset_complete", "dataset": dataset,
                          "hawor_windows": len(hawor), "dyn_windows": len(dyn)}), flush=True)

    if total_hawor != 2378 or total_dyn != 100:
        raise ValueError(f"total coverage mismatch: hawor={total_hawor}, dyn={total_dyn}")
    report["status"] = "complete"
    report["windows"] = {"hawor_all8_p95": total_hawor, "dyn_unfiltered": total_dyn,
                         "ego_matched_dyn_unfiltered": total_dyn}
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete", "windows": report["windows"],
        "report_sha256": sha256(report_path),
        "metric_keys": list(METRIC_KEYS),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def self_check() -> None:
    sample = normalize({"a": float("nan"), "b": np.int64(2), "c": 1.5})
    assert sample == {"a": None, "b": 2, "c": 1.5}
    print(json.dumps({"status": "self_check_passed"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.spec or not args.output_root:
        parser.error("--spec and --output-root are required")
    run(json.loads(args.spec.read_text(encoding="utf-8")), args.output_root)


if __name__ == "__main__":
    main()
