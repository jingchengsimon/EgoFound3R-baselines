#!/usr/bin/env python3
"""Recompute registered baseline metrics on result-3 joint-derived frame masks."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.contact.metrics import compute_contact_metrics
from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
from formal_evaluation.evaluate_six_dataset import _prediction_arrays


OSS_ROOT = Path("/mnt/oss/pre-train/ego/eval_artifacts")
SCHEMES = ("unfiltered", "p97_5", "all8_p95", "temporal_p95_other_p97_5")
_MODULE_CACHE = {}


def load_module(name: str, path: Path):
    key = (name, str(path))
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def remap(path: str, migration_root: Path, *, direct_oss: bool = False) -> Path:
    source = Path(path)
    if direct_oss:
        return source
    try:
        relative = source.relative_to(OSS_ROOT)
    except ValueError:
        return source
    return migration_root / "payload" / relative


def remap_gt_artifact(
    path: str, migration_root: Path, gt_index_parent: Path, *, direct_oss: bool = False
) -> Path:
    if direct_oss and "/gt_cache/" in path:
        return gt_index_parent / path.split("/gt_cache/", 1)[1]
    return remap(path, migration_root, direct_oss=direct_oss)


def remap_gt_row(
    row: dict, migration_root: Path, gt_index_parent: Path, *, direct_oss: bool = False
) -> dict:
    result = dict(row)
    result["array_path"] = str(remap_gt_artifact(
        str(row["array_path"]), migration_root, gt_index_parent, direct_oss=direct_oss
    ))
    if "metadata_path" in row:
        result["metadata_path"] = str(remap_gt_artifact(
            str(row["metadata_path"]), migration_root, gt_index_parent, direct_oss=direct_oss
        ))
    return result


def prediction_rows(
    method: str, item: dict, migration_root: Path, aliases: dict[str, str],
    *, direct_oss: bool = False
) -> dict[str, Path]:
    method_spec = item["predictions"][method]
    rows: dict[str, Path] = {}
    wanted = set(aliases.values())
    for raw_root in method_spec.get("formal_roots", []):
        root = remap(str(raw_root), migration_root, direct_oss=direct_oss)
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            metadata_path = directory / "metadata.json"
            prediction_path = directory / "predictions.npz"
            if not metadata_path.is_file() or not prediction_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("dataset")) != item["dataset"]:
                continue
            raw_window_id = str(metadata["window_id"])
            window_id = aliases.get(raw_window_id, raw_window_id)
            if window_id not in wanted:
                continue
            if window_id in rows and rows[window_id] != directory:
                raise ValueError(f"{item['dataset']}:{method}: duplicate {window_id}")
            rows[window_id] = directory
    expected = int(item["expected_windows"])
    if len(rows) != expected:
        raise ValueError(f"{item['dataset']}:{method}: {len(rows)} predictions, expected {expected}")
    return rows


def indexed_prediction_rows(index: Path, dataset: str, expected: int) -> dict[str, Path]:
    rows = {
        str(row["window_id"]): Path(str(row["prediction_dir"]))
        for row in read_jsonl(index)
        if str(row.get("dataset")) == dataset and str(row.get("method")) == "egofound3r"
    }
    if len(rows) != expected:
        raise ValueError(f"{dataset}:egofound3r_stride5: {len(rows)} predictions, expected {expected}")
    return rows


def read_masks(spec: dict, dataset: str, window_ids: set[str]) -> dict[str, dict[str, np.ndarray]]:
    roots = spec["mask_roots"]
    paths = {
        "p97_5": Path(roots["p97_5"]) / dataset / "p97_5_mask.json",
        "all8_p95": Path(roots["variants"]) / dataset / "all8_p95_mask.json",
        "temporal_p95_other_p97_5": (
            Path(roots["variants"]) / dataset / "temporal_p95_other_p97_5_mask.json"
        ),
    }
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = {
            str(window_id): np.asarray(mask, dtype=bool)
            for window_id, mask in zip(payload["window_ids"], payload["excluded"], strict=True)
        }
        if values.keys() != window_ids:
            raise ValueError(f"{dataset}:{name}: mask/window mismatch")
        result[name] = values
    result["unfiltered"] = {window_id: np.zeros(60, dtype=bool) for window_id in window_ids}
    return result


def process_window(task: tuple) -> tuple[str, dict[str, dict]]:
    (method, groups, window_id, prediction_dir, gt_row, masks, hand_script, scene_script) = task
    hand = load_module("same_mask_hand", Path(hand_script))
    scene = load_module("same_mask_scene", Path(scene_script))
    gt_metadata, target = load_window_cache(gt_row)
    metadata, prediction = _prediction_arrays(Path(prediction_dir))
    if metadata["frame_ids"] != gt_metadata["frame_ids"]:
        raise ValueError(f"{method}:{window_id}: frame identity mismatch")
    frozen_hand = freeze_hand_window(hand, prediction, target) if "hand" in groups else None
    frozen_scene = scene.freeze_scene(prediction, target) if "scene" in groups else None
    output = {}
    for scheme in SCHEMES:
        excluded = masks[scheme]
        keep = ~excluded
        row = {"window_id": window_id, "excluded_frame_count": int(excluded.sum())}
        if frozen_hand is not None:
            _, frame, pair, triplet = frozen_hand
            _, hand_rows = hand.aggregate_scheme(
                frame[None], pair[None], triplet[None], excluded[None]
            )
            row.update(hand_rows[0])
        if frozen_scene is not None:
            row.update(scene.aggregate_scene_window(frozen_scene, keep))
        if "contact" in groups:
            row.update(scene.aggregate_contact_window(prediction, target, keep))
            row.update(aggregate_visibility_window(prediction, target, keep))
        output[scheme] = row
    return window_id, output


def aggregate_visibility_window(
    prediction: dict, target: dict, keep: np.ndarray
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    hand_valid = prediction.get("hand_valid")
    for prefix, prediction_key, target_key, mask_key in (
        ("joint", "hand_visibility", "joint_visibility_target", "joint_visibility_mask"),
        ("marker", "marker_visibility", "marker_visibility_target", "marker_visibility_mask"),
    ):
        if prediction_key not in prediction or target_key not in target or mask_key not in target:
            continue
        mask = np.asarray(target[mask_key], dtype=bool).copy()
        mask &= np.broadcast_to(np.asarray(keep, dtype=bool)[:, None, None], mask.shape)
        if hand_valid is not None:
            mask &= np.broadcast_to(np.asarray(hand_valid, dtype=bool)[..., None], mask.shape)
        metrics = compute_contact_metrics(
            prediction[prediction_key], target[target_key], mask
        )
        result.update({f"{prefix}_visibility_{name}": value for name, value in metrics.items()})
    return result


def freeze_hand_window(hand, prediction: dict, target: dict):
    """Freeze hand residuals using predicted camera poses when available."""
    pred_joints, _ = hand._prediction_geometry_camera(prediction, "joint")
    gt_joints = target["hand_joints_camera"]
    hand_valid = prediction["hand_valid"].astype(bool) & target["hand_valid"].astype(bool)
    pose = prediction.get("camera_c2w")
    oracle = pose is None
    pose = target["camera_c2w"] if oracle else np.asarray(pose, dtype=float)
    camera_valid = target["camera_valid"].astype(bool)
    if not oracle and "camera_valid" in prediction:
        camera_valid &= prediction["camera_valid"].astype(bool)
    frame_count = pred_joints.shape[0]
    frozen_frame = np.full((3, 2, len(hand.FRAME_METRICS), frame_count), np.nan)
    frozen_pair = np.full((3, 2, frame_count - 1), np.nan)
    frozen_triplet = np.full((3, 2, frame_count - 2), np.nan)
    for granularity_index, (_, granularity, field, _, _, _, _) in enumerate(hand.GRANULARITIES):
        pred_points, _ = hand._prediction_geometry_camera(prediction, granularity)
        if pred_points is None:
            continue
        gt_points = hand._target_geometry(target, field, "camera")
        world_pred = hand._camera_to_world(pred_points, pose)
        world_gt = hand._target_geometry(target, field, "world")
        for side in range(2):
            raw_valid = (
                hand_valid[:, side]
                & np.isfinite(pred_points[:, side]).all(axis=(1, 2))
                & np.isfinite(gt_points[:, side]).all(axis=(1, 2))
            )
            point_mask = np.broadcast_to(raw_valid[:, None], pred_points[:, side].shape[:2])
            raw = hand._per_frame(pred_points[:, side], gt_points[:, side], raw_valid)
            aligned = hand._procrustes_per_frame(pred_points[:, side], gt_points[:, side], point_mask)
            pa_valid = raw_valid & np.isfinite(aligned).all(axis=(1, 2))
            pa = hand._per_frame(aligned, gt_points[:, side], pa_valid)
            roots_valid = (
                np.isfinite(pred_joints[:, side, 0]).all(axis=-1)
                & np.isfinite(gt_joints[:, side, 0]).all(axis=-1)
            )
            relative_valid = raw_valid & roots_valid
            relative_pred = pred_points[:, side] - pred_joints[:, side, :1]
            relative_gt = gt_points[:, side] - gt_joints[:, side, :1]
            rr = hand._per_frame(relative_pred, relative_gt, relative_valid)
            sim3 = hand.world_aligned_mpjpe(
                pred_points[:, side], gt_points[:, side], joint_mask=point_mask,
                mode="all", chunk_length=frame_count, unit_scale=1000.0,
            )
            world_valid = (
                raw_valid & camera_valid
                & np.isfinite(world_pred[:, side]).all(axis=(1, 2))
                & np.isfinite(world_gt[:, side]).all(axis=(1, 2))
            )
            world_mask = np.broadcast_to(world_valid[:, None], world_pred[:, side].shape[:2])
            w = hand.world_aligned_mpjpe(
                world_pred[:, side], world_gt[:, side], joint_mask=world_mask,
                mode="first2", chunk_length=frame_count, unit_scale=1000.0,
            )
            wa = hand.world_aligned_mpjpe(
                world_pred[:, side], world_gt[:, side], joint_mask=world_mask,
                mode="all", chunk_length=frame_count, unit_scale=1000.0,
            )
            pair, triplet = hand._pair_triplet(relative_pred, relative_gt, relative_valid)
            frozen_frame[granularity_index, side] = np.stack((raw, rr, pa, sim3, w, wa))
            frozen_pair[granularity_index, side] = pair
            frozen_triplet[granularity_index, side] = triplet
    return None, frozen_frame, frozen_pair, frozen_triplet


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def process_method_block(task: tuple) -> tuple[str, str, int, dict[str, dict]]:
    (
        dataset, method, groups, pred_rows, gt_rows, masks,
        hand_script, scene_script, output_root,
    ) = task
    if pred_rows.keys() != gt_rows.keys():
        raise ValueError(f"{dataset}:{method}: prediction/GT mismatch")
    rows = {scheme: [] for scheme in SCHEMES}
    for _, result in map(process_window, (
        (
            method, groups, window_id, str(pred_rows[window_id]), gt_rows[window_id],
            {scheme: masks[scheme][window_id] for scheme in SCHEMES},
            hand_script, scene_script,
        )
        for window_id in sorted(gt_rows)
    )):
        for scheme in SCHEMES:
            rows[scheme].append(result[scheme])
    method_root = Path(output_root) / dataset / method
    method_root.mkdir(parents=True)
    aggregates = {}
    for scheme in SCHEMES:
        aggregates[scheme] = aggregate_windows(rows[scheme], method=method)
        (method_root / f"{scheme}_window_metrics.jsonl").write_text(
            "".join(json.dumps(json_safe(row), allow_nan=False) + "\n" for row in rows[scheme]),
            encoding="utf-8",
        )
    return dataset, method, len(gt_rows), aggregates


def run(spec: dict, output_root: Path, workers: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in output_root.iterdir() if path.name != "runtime"]
    if unexpected:
        raise FileExistsError(f"output root already contains result artifacts: {unexpected[:5]}")
    methods_config = json.loads(Path(spec["methods_config"]).read_text(encoding="utf-8"))["methods"]
    summary = {
        "status": "running",
        "protocol": {
            "mask_source": "result3 joint8 dataset-local masks",
            "same_frame_mask_for_all_aligned_methods": True,
            "fit_on_full_unfiltered_window": True,
            "no_refit_after_filter": True,
            "excluded_gaps_are_never_bridged": True,
            "parallelism": {
                "level": "dataset_method_block",
                "block_workers": workers,
                "window_workers_per_block": 1,
            },
            "interactvlm_dyn_hamr": "excluded from recomputation; retain their 100-window unfiltered values",
        },
        "schemes": {name: defaultdict(dict) for name in SCHEMES},
        "coverage": {},
    }
    direct_oss = spec.get("source_mode") == "direct_oss"
    result3_catalogs = {
        item["dataset"]: item
        for item in json.loads(Path(spec["result3_catalog"]).read_text(encoding="utf-8"))["catalogs"]
    }
    block_tasks = []
    for item in spec["catalogs"]:
        dataset = item["dataset"]
        migration_root = Path(spec.get("migration_root") or "/nonexistent")
        gt_index = remap(str(item["gt_index"]), migration_root, direct_oss=direct_oss)
        raw_gt_rows = read_jsonl(gt_index)
        aliases = {
            str(row["cache_id"]): str(row["window_id"])
            for row in raw_gt_rows
        }
        gt_rows = {
            str(row["window_id"]): remap_gt_row(
                row, migration_root, gt_index.parent, direct_oss=direct_oss
            )
            for row in raw_gt_rows
        }
        expected = int(item["expected_windows"])
        if len(gt_rows) != expected:
            raise ValueError(f"{dataset}: {len(gt_rows)} GT windows, expected {expected}")
        masks = read_masks(spec, dataset, set(gt_rows))
        summary["coverage"][dataset] = {"gt_windows": expected, "methods": {}}
        method_inputs = []
        for method in spec["methods"]:
            method_inputs.append((
                method,
                methods_config[method],
                prediction_rows(
                    method, item, migration_root, aliases, direct_oss=direct_oss
                ),
            ))
        result3 = result3_catalogs[dataset]
        method_inputs.append((
            "egofound3r_stride5",
            methods_config["egofound3r"],
            indexed_prediction_rows(Path(result3["prediction_index"]), dataset, expected),
        ))
        for method, config, pred_rows in method_inputs:
            groups = set(config.get("group", []))
            block_tasks.append((
                dataset, method, groups, pred_rows, gt_rows, masks,
                spec["hand_script"], spec["scene_script"], str(output_root),
            ))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_method_block, task) for task in block_tasks]
        for future in as_completed(futures):
            dataset, method, expected, aggregates = future.result()
            for scheme in SCHEMES:
                summary["schemes"][scheme][dataset][method] = aggregates[scheme]
            summary["coverage"][dataset]["methods"][method] = expected
            (output_root / "progress_summary.json").write_text(
                json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(json.dumps({"stage": "method_complete", "dataset": dataset, "method": method,
                              "windows": expected}), flush=True)
    summary["status"] = "complete"
    payload = json_safe(summary)
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "datasets": len(spec["catalogs"]),
                      "methods": len(spec["methods"]) + 1, "schemes": list(SCHEMES)}), flush=True)


def self_check() -> None:
    assert SCHEMES[0] == "unfiltered" and len(SCHEMES) == 4
    print(json.dumps({"status": "self_check_passed"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-b64")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.spec_b64 or not args.output_root:
        parser.error("--spec-b64 and --output-root are required")
    spec = json.loads(base64.urlsafe_b64decode(args.spec_b64))
    run(spec, args.output_root, args.workers)


if __name__ == "__main__":
    main()
