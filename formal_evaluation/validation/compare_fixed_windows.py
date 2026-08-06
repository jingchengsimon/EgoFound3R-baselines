#!/usr/bin/env python3
"""Store legacy and aligned metrics for a fixed subset of saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from formal_evaluation.common.h2o_gt import H2OGTLoader
from formal_evaluation.evaluate import _camera_space_joints, _depth_pairs
from formal_evaluation.hand.metrics import compute_hand_metrics
from formal_evaluation.scene.metrics import compute_scene_metrics
from formal_evaluation.contact.metrics import compute_contact_metrics
from formal_evaluation.validation import legacy_reference


def _differences(legacy: dict[str, object], aligned: dict[str, object]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key in sorted(set(legacy) | set(aligned)):
        old, new = legacy.get(key), aligned.get(key)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            output[key] = float(new - old) if np.isfinite(old) and np.isfinite(new) else None
    return output


def _selected_windows(manifest: dict[str, object], phase: str, max_windows: int):
    section = manifest["formal_test" if phase == "formal" else "pilot"]
    for sequence_entry in section["sequences"]:
        for window in sequence_entry["windows"]:
            yield str(sequence_entry["sequence"]), str(window["window_id"]), list(window["frame_ids"])
            max_windows -= 1
            if max_windows == 0:
                return


def _contact_pair(predictions, contact_gt, hand_valid, point_set: str, key: str):
    if key not in predictions or contact_gt[point_set] is None:
        return None
    target, mask = contact_gt[point_set]
    if hand_valid is not None:
        mask = mask & np.broadcast_to(hand_valid[..., None], mask.shape)
    return predictions[key], target, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mano-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--groups", nargs="+", required=True, choices=("hand", "scene", "contact"))
    parser.add_argument("--scale-type", default="relative")
    parser.add_argument("--phase", choices=("pilot", "formal"), default="formal")
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()

    if args.max_windows < 1:
        raise ValueError("max-windows must be positive")
    manifest = json.loads(args.manifest.read_text())
    gt = H2OGTLoader(args.data_root, args.mano_dir)
    records = []
    for sequence, window_id, frame_ids in _selected_windows(manifest, args.phase, args.max_windows):
        path = args.output_root / args.method / args.phase / window_id / "predictions.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            predictions = dict(archive)
        legacy, aligned = {}, {}
        if "hand" in args.groups:
            gt_joints, _, gt_valid = gt.load_hand_gt(sequence, frame_ids)
            pred_joints, pred_valid = _camera_space_joints(predictions)
            if pred_joints is not None:
                pred_valid = np.ones(pred_joints.shape[:2], dtype=bool) if pred_valid is None else pred_valid
                legacy.update(legacy_reference.hand_metrics(pred_joints, gt_joints, pred_valid, gt_valid))
                aligned.update(compute_hand_metrics(pred_joints, gt_joints, pred_valid, gt_valid))
        if "scene" in args.groups:
            gt_pose, _ = gt.load_camera_gt(sequence, frame_ids)
            pairs = _depth_pairs(predictions, gt.load_depth_gt(sequence, frame_ids))
            legacy.update(legacy_reference.scene_metrics(predictions.get("camera_c2w"), gt_pose, pairs, scale_type=args.scale_type))
            aligned.update(compute_scene_metrics(predictions.get("camera_c2w"), gt_pose, pairs, scale_type=args.scale_type))
        if "contact" in args.groups:
            contact_gt = gt.load_contact_gt(sequence, frame_ids)
            hand_valid = predictions.get("hand_valid")
            for point_set, key in (("joint", "joint_contact_probability"), ("marker", "marker_contact_probability")):
                pair = _contact_pair(predictions, contact_gt, hand_valid, point_set, key)
                if pair is None:
                    continue
                probability, target, mask = pair
                legacy.update({f"{point_set}_contact_{name}": value for name, value in legacy_reference.contact_metrics(probability, target, mask).items()})
                aligned.update({f"{point_set}_contact_{name}": value for name, value in compute_contact_metrics(probability, target, mask).items()})
        records.append({"sequence": sequence, "window_id": window_id, "legacy": legacy, "aligned": aligned, "aligned_minus_legacy": _differences(legacy, aligned)})
    report = {"method": args.method, "phase": args.phase, "requested_window_count": args.max_windows, "evaluated_window_count": len(records), "windows": records}
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
