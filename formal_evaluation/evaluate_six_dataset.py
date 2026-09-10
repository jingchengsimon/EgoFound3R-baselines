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
from egocentric_metrics.alignment import SimilarityTransform, apply_transform, umeyama


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _prediction_arrays(
    directory: Path, *, scene_pose_only: bool = False
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        arrays = ({"camera_c2w": archive["camera_c2w"]} if scene_pose_only else
                  {key: archive[key] for key in archive.files})
    if not scene_pose_only:
        validate_comparison_output(metadata, arrays)
    return metadata, arrays


def _scene_window_cache(index_entry: Mapping[str, object]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata = json.loads(Path(str(index_entry["metadata_path"])).read_text(encoding="utf-8"))
    with np.load(Path(str(index_entry["array_path"])), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("camera_c2w", "camera_valid")}
    return metadata, arrays


_POINT_FIELDS = {"joint": "joints", "marker": "markers", "vertex": "vertices"}


def _camera_to_world(points: np.ndarray, camera_c2w: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    pose = np.asarray(camera_c2w, dtype=float)
    if value.ndim != 4 or value.shape[0] != pose.shape[0] or value.shape[-1] != 3 or pose.shape != (value.shape[0], 4, 4):
        raise ValueError("camera-to-world requires (T,2,N,3) points and matching (T,4,4) c2w poses")
    return np.einsum("tij,tnpj->tnpi", pose[:, :3, :3], value) + pose[:, None, None, :3, 3]


def _world_to_camera(points: np.ndarray, camera_c2w: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    pose = np.asarray(camera_c2w, dtype=float)
    if value.ndim != 4 or value.shape[0] != pose.shape[0] or value.shape[-1] != 3 or pose.shape != (value.shape[0], 4, 4):
        raise ValueError("world-to-camera requires (T,2,N,3) points and matching (T,4,4) c2w poses")
    centered = value - pose[:, None, None, :3, 3]
    return np.einsum("tji,tnpj->tnpi", pose[:, :3, :3], centered)


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


def _prediction_geometry_camera(
    predictions: Mapping[str, np.ndarray], granularity: str
) -> tuple[np.ndarray | None, str | None]:
    field = _POINT_FIELDS[granularity]
    direct = predictions.get(f"hand_{field}_camera")
    if direct is not None:
        return direct, "native_camera"
    if granularity == "marker" and "hand_vertices_camera" in predictions:
        return downsample_mano_vertices(predictions["hand_vertices_camera"]), "derived_camera"
    if granularity == "vertex" and "hand_markers_camera" in predictions:
        return upsample_mano_markers(predictions["hand_markers_camera"]), "derived_camera"
    pose = predictions.get("camera_c2w")
    if pose is None:
        return None, "unavailable_without_predicted_camera_c2w"
    direct = predictions.get(f"hand_{field}_world")
    if direct is not None:
        return _world_to_camera(direct, pose), "world_to_camera_via_predicted_camera_c2w"
    if granularity == "marker" and "hand_vertices_world" in predictions:
        camera_vertices = _world_to_camera(predictions["hand_vertices_world"], pose)
        return downsample_mano_vertices(camera_vertices), "world_to_camera_via_predicted_camera_c2w"
    if granularity == "vertex" and "hand_markers_world" in predictions:
        camera_markers = _world_to_camera(predictions["hand_markers_world"], pose)
        return upsample_mano_markers(camera_markers), "world_to_camera_via_predicted_camera_c2w"
    return None, None


def _target_geometry(targets: Mapping[str, np.ndarray], field: str, coordinate: str) -> np.ndarray:
    camera = np.asarray(targets[f"hand_{field}_camera"], dtype=float)
    return camera if coordinate == "camera" else _camera_to_world(camera, targets["camera_c2w"])


def _camera_trajectory_alignments(
    predictions: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]
) -> tuple[dict[str, SimilarityTransform], np.ndarray]:
    """Fit one SE(3) and one Sim(3) from predicted to GT camera centres."""
    target_pose = np.asarray(targets["camera_c2w"], dtype=float)
    prediction_pose = predictions.get("camera_c2w")
    if prediction_pose is None:
        return {}, np.zeros(target_pose.shape[0], dtype=bool)
    prediction_pose = np.asarray(prediction_pose, dtype=float)
    if prediction_pose.shape != target_pose.shape:
        raise ValueError("prediction camera_c2w shape differs from GT cache")
    valid = np.isfinite(prediction_pose).all(axis=(1, 2)) & np.isfinite(target_pose).all(axis=(1, 2))
    if "camera_valid" in predictions:
        valid &= np.asarray(predictions["camera_valid"], dtype=bool)
    if "camera_valid" in targets:
        valid &= np.asarray(targets["camera_valid"], dtype=bool)
    if np.count_nonzero(valid) < 3:
        return {}, valid
    source = prediction_pose[valid, :3, 3]
    destination = target_pose[valid, :3, 3]
    transforms = {
        "se3": umeyama(source, destination, fix_scale=True),
        "sim3": umeyama(source, destination),
    }
    transforms = {
        name: transform for name, transform in transforms.items()
        if np.isfinite(transform.scale).all()
        and np.isfinite(transform.rotation).all()
        and np.isfinite(transform.translation).all()
    }
    return transforms, valid


def _apply_world_alignment(points: np.ndarray, transform: SimilarityTransform, valid: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    aligned = apply_transform(value.reshape(-1, 3), transform).reshape(value.shape)
    return np.where(valid[:, None, None, None], aligned, np.nan)


def _world_geometry(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    points: np.ndarray,
    coordinate: str,
    granularity: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    field = _POINT_FIELDS[granularity]
    if coordinate == "world":
        return points, _target_geometry(targets, field, "world"), "native_world"
    pose = predictions.get("camera_c2w")
    if pose is None:
        return None, None, "unavailable_without_predicted_camera_c2w"
    return _camera_to_world(points, pose), _target_geometry(targets, field, "world"), "predicted_camera_c2w"


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
        world_transforms, world_camera_valid = _camera_trajectory_alignments(predictions, targets)
        scale_type = str(config.get("scale_type", "relative"))
        emit_w = scale_type in {"metric", "metric_world_hand"} and "se3" in world_transforms
        emit_wa = scale_type in {"metric", "metric_world_hand", "relative", "up_to_scale"} and "sim3" in world_transforms
        if world_transforms:
            result["hand_world_alignment_source"] = "camera_trajectory"
            result["hand_world_alignment_valid_frame_count"] = int(np.count_nonzero(world_camera_valid))
            if "sim3" in world_transforms:
                result["hand_world_alignment_sim3_scale"] = float(world_transforms["sim3"].scale)
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
            world_pred, world_gt, world_pose_source = _world_geometry(
                predictions, targets, pred_points, coordinate, granularity
            )
            w_prediction = (
                _apply_world_alignment(world_pred, world_transforms["se3"], world_camera_valid)
                if emit_w and world_pred is not None else None
            )
            wa_prediction = (
                _apply_world_alignment(world_pred, world_transforms["sim3"], world_camera_valid)
                if emit_wa and world_pred is not None else None
            )
            metric_points = pred_points
            metric_coordinate = coordinate
            metric_coordinate_source = "native"
            emit_point_metrics = True
            camera_points, metric_coordinate_source = _prediction_geometry_camera(predictions, granularity)
            if camera_points is None:
                emit_point_metrics = False
                metric_coordinate = "unavailable"
            else:
                metric_points = camera_points
                metric_coordinate = "camera"
            target_points = _target_geometry(targets, field, "camera" if emit_point_metrics else coordinate)
            roots_pred = roots_gt = None
            if emit_point_metrics:
                camera_joints, _ = _prediction_geometry_camera(predictions, "joint")
                if camera_joints is not None:
                    roots_pred = camera_joints[..., 0, :]
                    roots_gt = targets["hand_joints_camera"][..., 0, :]
            result.update(compute_hand_metrics(
                metric_points,
                target_points,
                pred_valid,
                targets["hand_valid"],
                granularity=granularity,
                root_prediction=roots_pred,
                root_target=roots_gt,
                world_prediction=w_prediction,
                world_aligned_prediction=wa_prediction,
                world_target=world_gt if w_prediction is not None or wa_prediction is not None else None,
                temporal_fps=temporal_fps,
                emit_point_metrics=emit_point_metrics,
            ))
            result[f"hand_{granularity}_geometry_provenance"] = provenance
            result[f"hand_{granularity}_metric_coordinate"] = metric_coordinate
            result[f"hand_{granularity}_metric_coordinate_source"] = metric_coordinate_source
            result[f"hand_{granularity}_world_pose_source"] = world_pose_source
    if "scene" in groups:
        target_pose = np.where(targets["camera_valid"][:, None, None], targets["camera_c2w"], np.nan)
        result.update(compute_scene_metrics(
            predictions.get("camera_c2w"), target_pose,
            [] if config.get("scene_pose_only") else _depth_pairs(predictions, targets),
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
        for prefix, prediction_key, target_key, mask_key in (
            ("joint", "hand_visibility", "joint_visibility_target", "joint_visibility_mask"),
            ("marker", "marker_visibility", "marker_visibility_target", "marker_visibility_mask"),
        ):
            if prediction_key not in predictions or target_key not in targets or mask_key not in targets:
                continue
            mask = targets[mask_key].copy()
            if hand_valid is not None:
                mask &= np.broadcast_to(hand_valid[..., None], mask.shape)
            for name, value in compute_contact_metrics(
                predictions[prediction_key], targets[target_key], mask
            ).items():
                result[f"{prefix}_visibility_{name}"] = value
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
    scene_pose_only = all(
        set(config.get("group", [])) == {"scene"} and config.get("scene_pose_only")
        for config in methods.values()
    )
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
        gt_metadata, targets = (_scene_window_cache(gt_row) if scene_pose_only else
                                load_window_cache(gt_row))
        for method, config in methods.items():
            directory = predictions.get((method, str(gt_metadata["dataset"]), str(gt_metadata["window_id"])))
            if directory is None:
                missing[method] += 1
                continue
            metadata, arrays = _prediction_arrays(directory, scene_pose_only=scene_pose_only)
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
