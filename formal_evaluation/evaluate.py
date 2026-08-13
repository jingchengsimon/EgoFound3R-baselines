#!/usr/bin/env python3
"""Evaluate canonical baseline predictions with the formal protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.common.h2o_gt import H2OGTLoader
from formal_evaluation.contact.metrics import compute_contact_metrics
from formal_evaluation.hand.metrics import compute_hand_metrics
from formal_evaluation.scene.metrics import compute_scene_metrics


def _phase_sequences(manifest: dict[str, object], phase: str) -> list[dict[str, object]]:
    key = "formal_test" if phase == "formal" else "pilot"
    section = manifest.get(key, {})
    sequences = section.get("sequences", []) if isinstance(section, dict) else []
    if not sequences:
        raise ValueError(f"manifest has no {phase} sequences")
    return sequences


def _camera_space_joints(predictions: dict[str, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if "hand_joints_camera" in predictions:
        return predictions["hand_joints_camera"], predictions.get("hand_valid")
    if "hand_joints_world" not in predictions or "camera_c2w" not in predictions:
        return None, None
    world = predictions["hand_joints_world"]
    w2c = np.linalg.inv(predictions["camera_c2w"])
    camera = np.einsum("tik,tnjk->tnji", w2c[:, :3, :3], world) + w2c[:, None, None, :3, 3]
    return camera, predictions.get("hand_valid")


def _depth_pairs(predictions: dict[str, np.ndarray], targets) -> list[tuple[np.ndarray, np.ndarray]]:
    if "depth" not in predictions or targets is None:
        return []
    pairs = []
    for prediction, target in zip(predictions["depth"], targets, strict=True):
        if target is None:
            continue
        if prediction.shape != target.shape:
            import cv2

            prediction = cv2.resize(prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
        pairs.append((np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)))
    return pairs


def _load_predictions(path: Path, config: dict[str, object]) -> dict[str, np.ndarray]:
    """Load only canonical fields consumed by the selected metric groups."""
    groups = set(config.get("group", []))
    keys: set[str] = set()
    if "hand" in groups:
        keys.update(("hand_joints_camera", "hand_joints_world", "camera_c2w", "hand_valid"))
    if "scene" in groups:
        keys.update(("camera_c2w", "depth"))
    if "contact" in groups:
        keys.update(("hand_valid", "joint_contact_probability", "marker_contact_probability"))
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in keys if key in archive}


def evaluate_window(method: str, predictions: dict[str, np.ndarray], gt: H2OGTLoader, sequence: str, frame_ids: list[str], config: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {"sequence": sequence, "window_id": "_".join((sequence.replace("/", "_"), frame_ids[0], frame_ids[-1]))}
    groups = set(config.get("group", []))
    if "hand" in groups:
        gt_joints, _, gt_valid = gt.load_hand_gt(sequence, frame_ids)
        pred_joints, pred_valid = _camera_space_joints(predictions)
        if pred_joints is not None:
            if pred_valid is None:
                pred_valid = np.ones(pred_joints.shape[:2], dtype=bool)
            result.update(compute_hand_metrics(pred_joints, gt_joints, pred_valid, gt_valid))
    if "scene" in groups:
        gt_pose, _ = gt.load_camera_gt(sequence, frame_ids)
        result.update(compute_scene_metrics(predictions.get("camera_c2w"), gt_pose, _depth_pairs(predictions, gt.load_depth_gt(sequence, frame_ids)), scale_type=str(config.get("scale_type", "relative"))))
    if "contact" in groups:
        contact_gt = gt.load_contact_gt(sequence, frame_ids)
        hand_valid = predictions.get("hand_valid")
        for point_set, prediction_key in (("joint", "joint_contact_probability"), ("marker", "marker_contact_probability")):
            if prediction_key not in predictions or contact_gt[point_set] is None:
                continue
            target, mask = contact_gt[point_set]
            if hand_valid is not None:
                mask = mask & np.broadcast_to(hand_valid[..., None], mask.shape)
            for name, value in compute_contact_metrics(predictions[prediction_key], target, mask).items():
                result[f"{point_set}_contact_{name}"] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--methods-config", type=Path, default=Path("formal_evaluation/config/methods_v1.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    methods = json.loads(args.methods_config.read_text())["methods"]
    selected_methods = args.methods or list(methods)
    gt = H2OGTLoader(args.data_root, args.mano_dir)
    report: dict[str, object] = {"phase": args.phase, "methods": {}}
    for method in selected_methods:
        if method not in methods:
            raise ValueError(f"unknown method: {method}")
        window_results = []
        for sequence_entry in _phase_sequences(manifest, args.phase):
            sequence = str(sequence_entry["sequence"])
            for window in sequence_entry.get("windows", []):
                frame_ids = list(window["frame_ids"])
                window_id = str(window["window_id"])
                prediction_path = args.output_root / method / args.phase / window_id / "predictions.npz"
                if not prediction_path.exists():
                    continue
                predictions = _load_predictions(prediction_path, methods[method])
                window_results.append(evaluate_window(method, predictions, gt, sequence, frame_ids, methods[method]))
        report["methods"][method] = aggregate_windows(window_results, method=method)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
