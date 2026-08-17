#!/usr/bin/env python3
"""Evaluate canonical predictions against the reusable current-dataloader GT cache."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.common.schema import validate_comparison_output
from formal_evaluation.contact.metrics import compute_contact_metrics
from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
from formal_evaluation.hand.metrics import compute_hand_metrics
from formal_evaluation.scene.metrics import compute_scene_metrics


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _prediction_arrays(directory: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_comparison_output(metadata, arrays)
    return metadata, arrays


def _camera_space_joints(predictions: Mapping[str, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if "hand_joints_camera" in predictions:
        return predictions["hand_joints_camera"], predictions.get("hand_valid")
    if "hand_joints_world" not in predictions or "camera_c2w" not in predictions:
        return None, None
    w2c = np.linalg.inv(predictions["camera_c2w"])
    world = predictions["hand_joints_world"]
    camera = np.einsum("tik,tnjk->tnji", w2c[:, :3, :3], world) + w2c[:, None, None, :3, 3]
    return camera, predictions.get("hand_valid")


def _depth_pairs(prediction: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    if "depth" not in prediction or "depth" not in target:
        return []
    result = []
    valid = target.get("depth_valid")
    for index, (pred, gt) in enumerate(zip(prediction["depth"], target["depth"], strict=True)):
        if valid is not None:
            gt = np.where(valid[index], gt, np.nan)
        if pred.shape != gt.shape:
            import cv2

            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        result.append((np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)))
    return result


def evaluate_window(
    *, method: str, config: Mapping[str, object], metadata: Mapping[str, object], predictions: Mapping[str, np.ndarray], gt_metadata: Mapping[str, object], targets: Mapping[str, np.ndarray]
) -> dict[str, object]:
    if metadata.get("frame_ids") != gt_metadata.get("frame_ids"):
        raise ValueError(f"{method}: prediction frame IDs differ from GT cache")
    result: dict[str, object] = {
        "dataset": gt_metadata["dataset"],
        "sequence": gt_metadata["sequence_id"],
        "window_id": gt_metadata["window_id"],
    }
    groups = set(config.get("group", []))
    if "hand" in groups:
        pred_joints, pred_valid = _camera_space_joints(predictions)
        if pred_joints is not None:
            if pred_valid is None:
                pred_valid = np.ones(pred_joints.shape[:2], dtype=bool)
            result.update(compute_hand_metrics(pred_joints, targets["hand_joints_camera"], pred_valid, targets["hand_valid"]))
    if "scene" in groups:
        target_pose = np.where(targets["camera_valid"][:, None, None], targets["camera_c2w"], np.nan)
        result.update(compute_scene_metrics(
            predictions.get("camera_c2w"), target_pose, _depth_pairs(predictions, targets),
            scale_type=str(config.get("scale_type", "relative")),
        ))
    if "contact" in groups:
        hand_valid = predictions.get("hand_valid")
        for prefix, prediction_key, target_key, mask_key in (
            ("joint", "joint_contact_probability", "joint_contact_target", "joint_contact_mask"),
            ("marker", "marker_contact_probability", "marker_contact_target", "marker_contact_mask"),
        ):
            if prediction_key not in predictions:
                continue
            mask = targets[mask_key].copy()
            if hand_valid is not None:
                mask &= np.broadcast_to(hand_valid[..., None], mask.shape)
            for name, value in compute_contact_metrics(predictions[prediction_key], targets[target_key], mask).items():
                result[f"{prefix}_contact_{name}"] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-index", action="append", type=Path, required=True)
    parser.add_argument("--prediction-index", type=Path, required=True,
                        help="JSONL: method, dataset, window_id, prediction_dir")
    parser.add_argument("--methods-config", type=Path, default=Path("formal_evaluation/config/methods_v1.json"))
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    methods = json.loads(args.methods_config.read_text(encoding="utf-8"))["methods"]
    gt_rows = [row for path in args.gt_index for row in _load_jsonl(path)]
    predictions = {}
    for row in _load_jsonl(args.prediction_index):
        key = (str(row["method"]), str(row["dataset"]), str(row["window_id"]))
        if key in predictions:
            raise ValueError(f"duplicate prediction index entry: {key}")
        predictions[key] = Path(str(row["prediction_dir"]))

    results: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    missing: dict[str, int] = defaultdict(int)
    for gt_row in gt_rows:
        gt_metadata, targets = load_window_cache(gt_row)
        for method, config in methods.items():
            directory = predictions.get((method, str(gt_metadata["dataset"]), str(gt_metadata["window_id"])))
            if directory is None:
                missing[method] += 1
                continue
            metadata, arrays = _prediction_arrays(directory)
            results[method][str(gt_metadata["dataset"])].append(evaluate_window(
                method=method, config=config, metadata=metadata, predictions=arrays,
                gt_metadata=gt_metadata, targets=targets,
            ))
    report = {
        "gt_windows": len(gt_rows),
        "methods": {
            method: {
                "missing_prediction_windows": missing[method],
                "datasets": {
                    dataset: aggregate_windows(rows, method=method)
                    for dataset, rows in sorted(results[method].items())
                },
            }
            for method in methods
        },
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
