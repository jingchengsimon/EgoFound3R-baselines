#!/usr/bin/env python3
"""Recompute Dyn-HaMR W/WA with the historical hand-space Sim(3) protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from egocentric_metrics import world_aligned_mpjpe
from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
from formal_evaluation.evaluate_six_dataset import (
    _camera_to_world,
    _prediction_geometry,
    _target_geometry,
)
from formal_evaluation.recompute_hand_literature import read_hand_prediction
from formal_evaluation.recompute_same_mask_all_methods import remap_gt_row


GRANULARITIES = (
    ("joint", "joints", "mpjpe"),
    ("marker", "markers", "mpmpe"),
    ("vertex", "vertices", "mpvpe"),
)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alignment_metrics(
    world_prediction: np.ndarray,
    world_target: np.ndarray,
    valid_frames: np.ndarray,
) -> dict[str, float | int]:
    """Return first-two-frame W and whole-window WA for one hand/granularity."""
    point_mask = np.broadcast_to(valid_frames[:, None], world_prediction.shape[:2]).copy()
    point_mask &= np.isfinite(world_prediction).all(axis=-1)
    point_mask &= np.isfinite(world_target).all(axis=-1)
    result: dict[str, float | int] = {}
    for name, mode in (("w", "first2"), ("wa", "all")):
        errors = world_aligned_mpjpe(
            world_prediction,
            world_target,
            joint_mask=point_mask,
            mode=mode,
            chunk_length=len(world_prediction),
            unit_scale=1000.0,
        )
        finite = np.isfinite(errors)
        result[name] = float(errors[finite].mean()) if finite.any() else float("nan")
        result[name + "_frame_count"] = int(finite.sum())
    return result


def recompute(spec: dict[str, object], output_root: Path) -> None:
    source_root = Path(str(spec["source_root"]))
    source_report = source_root / "report.json"
    source_summary = source_root / "summary.json"
    source_complete = source_root / "COMPLETE"
    prediction_index = source_root / "predictions.jsonl"
    if not source_complete.is_file():
        raise ValueError("SOURCE_COMPLETE_MISSING")
    summary = json.loads(source_summary.read_text())
    expected = int(spec["expected_windows"])
    if summary.get("status") != "complete" or summary.get("windows") != expected:
        raise ValueError("SOURCE_SUMMARY_INCOMPLETE")
    if summary.get("report_sha256") != _digest(source_report):
        raise ValueError("SOURCE_REPORT_SHA256_MISMATCH")
    if summary.get("prediction_index_sha256") != _digest(prediction_index):
        raise ValueError("SOURCE_PREDICTION_INDEX_SHA256_MISMATCH")

    gt_index = Path(str(spec["gt_index"]))
    if _digest(gt_index) != spec["gt_index_sha256"]:
        raise ValueError("GT_INDEX_SHA256_MISMATCH")
    gt_rows = _jsonl(gt_index)
    by_window = {str(row["window_id"]): row for row in gt_rows}
    aliases = {str(row["cache_id"]): str(row["window_id"]) for row in gt_rows}
    prediction_rows = _jsonl(prediction_index)
    if len(prediction_rows) != expected:
        raise ValueError("SOURCE_PREDICTION_COUNT_MISMATCH")

    output_root.mkdir(parents=True, exist_ok=False)
    window_results: list[dict[str, object]] = []
    with (output_root / "metrics.jsonl").open("x") as stream:
        for position, prediction_row in enumerate(prediction_rows, 1):
            prediction_dir = Path(str(prediction_row["prediction_dir"]))
            metadata, prediction = read_hand_prediction(prediction_dir)
            window_id = aliases.get(str(metadata["window_id"]), str(metadata["window_id"]))
            gt_row = by_window.get(window_id)
            if gt_row is None:
                raise ValueError(f"UNREGISTERED_WINDOW:{window_id}")
            remapped = remap_gt_row(gt_row, Path("/unused"), gt_index.parent, direct_oss=True)
            gt_metadata, target = load_window_cache(remapped)
            if metadata["frame_ids"] != gt_metadata["frame_ids"]:
                raise ValueError(f"FRAME_ID_MISMATCH:{window_id}")

            result: dict[str, object] = {
                "dataset": spec["dataset"],
                "window_id": window_id,
                "prediction_dir": str(prediction_dir),
            }
            for granularity, field, position_name in GRANULARITIES:
                points, coordinate, provenance = _prediction_geometry(prediction, granularity)
                if points is None or coordinate is None:
                    raise ValueError(f"MISSING_{granularity.upper()}_GEOMETRY:{window_id}")
                if coordinate == "world":
                    world_prediction = points
                    world_source = "native_world"
                else:
                    pose = prediction.get("camera_c2w")
                    if pose is None:
                        raise ValueError(f"MISSING_PREDICTED_CAMERA:{window_id}")
                    world_prediction = _camera_to_world(points, pose)
                    world_source = "predicted_camera_c2w"
                world_target = _target_geometry(target, field, "world")
                camera_valid = np.asarray(target["camera_valid"], dtype=bool).copy()
                if "camera_valid" in prediction:
                    camera_valid &= np.asarray(prediction["camera_valid"], dtype=bool)
                hand_valid = np.asarray(prediction["hand_valid"], dtype=bool) & np.asarray(target["hand_valid"], dtype=bool)
                result[f"hand_{granularity}_geometry_provenance"] = provenance
                result[f"hand_{granularity}_world_pose_source"] = world_source
                for side_index, side in enumerate(("left", "right")):
                    prefix = f"hand_{side}_" if granularity == "joint" else f"hand_{side}_{granularity}_"
                    values = alignment_metrics(
                        world_prediction[:, side_index],
                        world_target[:, side_index],
                        hand_valid[:, side_index] & camera_valid,
                    )
                    result[f"{prefix}w_{position_name}"] = values["w"]
                    result[f"{prefix}w_{position_name}_frame_count"] = values["w_frame_count"]
                    result[f"{prefix}wa_{position_name}"] = values["wa"]
                    result[f"{prefix}wa_{position_name}_frame_count"] = values["wa_frame_count"]
            window_results.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            print(json.dumps({"processed": position, "total": expected, "window_id": window_id}), flush=True)

    aggregate = aggregate_windows(window_results, method="dyn_hamr")
    # W follows the historical valid-frame weighting. WA deliberately retains
    # the historical mean of valid per-hand window means.
    for granularity, _, position_name in GRANULARITIES:
        for side in ("left", "right"):
            prefix = f"hand_{side}_" if granularity == "joint" else f"hand_{side}_{granularity}_"
            stem = f"{prefix}w_{position_name}"
            samples = [
                (float(row[stem]), int(row[stem + "_frame_count"]))
                for row in window_results
                if np.isfinite(row[stem]) and int(row[stem + "_frame_count"]) > 0
            ]
            sample_count = sum(count for _, count in samples)
            aggregate[stem + "_window_macro_mean"] = aggregate[stem + "_mean"]
            aggregate[stem + "_sample_count"] = sample_count
            aggregate[stem + "_mean"] = (
                sum(value * count for value, count in samples) / sample_count
                if sample_count else float("nan")
            )

    complete6 = {}
    for granularity, _, position_name in GRANULARITIES:
        label = "" if granularity == "joint" else granularity + "_"
        for metric in ("w", "wa"):
            weighted = count = 0
            for side in ("left", "right"):
                stem = f"hand_{side}_{label}{metric}_{position_name}"
                value = aggregate.get(stem + "_mean", float("nan"))
                windows = int(aggregate.get(stem + "_count", 0))
                if np.isfinite(value) and windows:
                    weighted += float(value) * windows
                    count += windows
            if not count:
                raise ValueError(f"NO_DEFINED_METRIC:{label}{metric}_{position_name}")
            complete6[f"{label}{metric}_{position_name}"] = weighted / count

    protocol = {
        "id": "ego_final_hand_first2_world_wa_all_wrist_temporal_60f_v1",
        "w_alignment": "world_hand_first_two_frames_sim3",
        "wa": "world_hand_entire_60f_sim3",
        "world_pose_source": "predicted_camera_c2w_or_native_world",
        "w_aggregation": "valid_frame_weighted",
        "wa_aggregation": "valid_hand_window_mean_preserves_historical_convention",
    }
    report = {
        "status": "complete",
        "method": "dyn_hamr",
        "windows": expected,
        "datasets": {str(spec["dataset"]): aggregate},
        "complete6": complete6,
        "protocol": protocol,
        "source": {
            "run_id": spec["source_run_id"],
            "output_root": str(source_root),
            "report_sha256": _digest(source_report),
            "prediction_index_sha256": _digest(prediction_index),
            "gt_index_sha256": spec["gt_index_sha256"],
        },
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    result_summary = {
        "status": "complete",
        "windows": expected,
        "metric_count": len(complete6),
        "report_sha256": _digest(report_path),
        "source_prediction_index_sha256": _digest(prediction_index),
    }
    (output_root / "summary.json").write_text(json.dumps(result_summary, indent=2))
    (output_root / "COMPLETE").write_text("complete\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    recompute(json.loads(args.spec.read_text()), args.output_root)


if __name__ == "__main__":
    main()
