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
from formal_evaluation.common.mano_sampling import downsample_mano_vertices, upsample_mano_markers
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


_POINT_FIELDS = {"joint": "joints", "marker": "markers", "vertex": "vertices"}


def _camera_to_world(points: np.ndarray, camera_c2w: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    pose = np.asarray(camera_c2w, dtype=float)
    if value.ndim != 4 or value.shape[0] != pose.shape[0] or value.shape[-1] != 3 or pose.shape != (value.shape[0], 4, 4):
        raise ValueError("camera-to-world requires (T,2,N,3) points and matching (T,4,4) c2w poses")
    return np.einsum("tij,tnpj->tnpi", pose[:, :3, :3], value) + pose[:, None, None, :3, 3]


def _prediction_geometry(predictions: Mapping[str, np.ndarray], granularity: str) -> tuple[np.ndarray | None, str | None, str | None]:
    field = _POINT_FIELDS[granularity]
    for coordinate in ("camera", "world"):
        direct = predictions.get(f"hand_{field}_{coordinate}")
        if direct is not None:
            return direct, coordinate, "native"
    if granularity == "marker":
        for coordinate in ("camera", "world"):
            vertices = predictions.get(f"hand_vertices_{coordinate}")
            if vertices is not None:
                return downsample_mano_vertices(vertices), coordinate, "derived_from_778_vertices"
    if granularity == "vertex":
        for coordinate in ("camera", "world"):
            markers = predictions.get(f"hand_markers_{coordinate}")
            if markers is not None:
                return upsample_mano_markers(markers), coordinate, "derived_from_195_markers"
    return None, None, None


def _target_geometry(targets: Mapping[str, np.ndarray], field: str, coordinate: str) -> np.ndarray:
    camera = np.asarray(targets[f"hand_{field}_camera"], dtype=float)
    return camera if coordinate == "camera" else _camera_to_world(camera, targets["camera_c2w"])


def _roots_in_coordinate(predictions: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray], coordinate: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    field = f"hand_joints_{coordinate}"
    if field in predictions:
        pred_root = predictions[field][..., 0, :]
    elif coordinate == "world" and "hand_joints_camera" in predictions:
        pred_root = _camera_to_world(predictions["hand_joints_camera"], _world_pose_for_prediction(predictions, targets))[..., 0, :]
    else:
        return None, None
    return pred_root, _target_geometry(targets, "joints", coordinate)[..., 0, :]


def _world_pose_for_prediction(predictions: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]) -> np.ndarray:
    """Use a method pose where it exists; camera-only hand methods use GT pose explicitly."""
    target_pose = np.asarray(targets["camera_c2w"], dtype=float)
    prediction_pose = predictions.get("camera_c2w")
    if prediction_pose is None:
        return target_pose
    prediction_pose = np.asarray(prediction_pose, dtype=float)
    if prediction_pose.shape != target_pose.shape:
        raise ValueError("prediction camera_c2w shape differs from GT cache")
    return np.where(np.isfinite(prediction_pose).all(axis=(1, 2))[:, None, None], prediction_pose, target_pose)


def _world_pose_source(predictions: Mapping[str, np.ndarray]) -> str:
    prediction_pose = predictions.get("camera_c2w")
    if prediction_pose is None:
        return "gt_camera_c2w"
    valid = np.isfinite(np.asarray(prediction_pose, dtype=float)).all(axis=(1, 2))
    if np.all(valid):
        return "predicted_camera_c2w"
    return "mixed_predicted_and_gt_camera_c2w" if np.any(valid) else "gt_camera_c2w"


def _world_geometry(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    points: np.ndarray,
    coordinate: str,
    granularity: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    field = _POINT_FIELDS[granularity]
    if coordinate == "world":
        return points, _target_geometry(targets, field, "world"), "native_world"
    pose = _world_pose_for_prediction(predictions, targets)
    return _camera_to_world(points, pose), _target_geometry(targets, field, "world"), _world_pose_source(predictions)


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
        pred_valid = predictions.get("hand_valid")
        if pred_valid is not None:
            pred_valid = np.asarray(pred_valid, dtype=bool)
        temporal_fps = float(metadata.get("temporal_fps", 30.0))
        if not np.isfinite(temporal_fps) or temporal_fps <= 0.0:
            raise ValueError(f"{method}: temporal_fps must be positive")
        for granularity in ("joint", "marker", "vertex"):
            pred_points, coordinate, provenance = _prediction_geometry(predictions, granularity)
            if pred_points is None or coordinate is None:
                continue
            if pred_valid is None:
                pred_valid = np.ones(pred_points.shape[:2], dtype=bool)
            field = _POINT_FIELDS[granularity]
            target_points = _target_geometry(targets, field, coordinate)
            roots_pred, roots_gt = _roots_in_coordinate(predictions, targets, coordinate)
            world_pred, world_gt, world_pose_source = _world_geometry(
                predictions, targets, pred_points, coordinate, granularity
            )
            result.update(compute_hand_metrics(
                pred_points,
                target_points,
                pred_valid,
                targets["hand_valid"],
                granularity=granularity,
                root_prediction=roots_pred,
                root_target=roots_gt,
                world_prediction=world_pred,
                world_target=world_gt,
                temporal_fps=temporal_fps,
            ))
            result[f"hand_{granularity}_geometry_provenance"] = provenance
            result[f"hand_{granularity}_world_pose_source"] = world_pose_source
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
