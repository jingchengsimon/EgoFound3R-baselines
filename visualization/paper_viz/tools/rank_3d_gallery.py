#!/usr/bin/env python3
"""Rank the frozen 104-segment 3D gallery with strict HaWoR and partial Dyn-HaMR.

All ordinary baselines must cover all five 60-frame windows.  Dyn-HaMR is an
explicit exception because its formal evaluation sampled a smaller window set:
when at least one selected window intersects that set, Dyn-HaMR is compared to
EgoFound3R on those same window indices and participates in the ranking.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from egocentric_metrics import world_aligned_mpjpe
from paper_viz.inputs import arrays, resolve_prediction_file


METHODS = ("ego", "wilor", "pad_hand", "egoforce", "dyn_hamr", "hawor", "reviv4d")
BASELINES = METHODS[1:]
EXPECTED_MANIFEST_SHA256 = "e88ac84845a5a61e6e472b12d160816ef0d8a48eaeb0e2085985335ff33c92a5"
EXPECTED_COUNTS = {"arctic": 48, "h2o": 5, "hot3d": 44, "oakink_v2": 7}


def mappings(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or key in result:
            raise ValueError(f"expected one unique DATASET=PATH mapping, got {value!r}")
        result[key] = Path(raw_path)
    return result


def json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def to_world(method: str, data: dict, gt: dict) -> tuple[np.ndarray, str, np.ndarray]:
    if "hand_joints_world" in data:
        points = np.asarray(data["hand_joints_world"], float)
        pose_source = "native_world"
    else:
        points = np.asarray(data["hand_joints_camera"], float)
        if "camera_c2w" in data:
            pose = np.asarray(data["camera_c2w"], float)
            pose_source = "predicted_camera_c2w"
        else:
            pose = np.asarray(gt["camera_c2w"], float)
            pose_source = "gt_camera_c2w_oracle"
        points = np.einsum("tij,tspj->tspi", pose[:, :3, :3], points) + pose[:, None, None, :3, 3]
    valid = np.asarray(data["hand_valid"], bool) & np.asarray(gt["hand_valid"], bool)
    valid &= np.asarray(gt["camera_valid"], bool)[:, None]
    if pose_source == "predicted_camera_c2w" and "camera_valid" in data:
        valid &= np.asarray(data["camera_valid"], bool)[:, None]
    return points, pose_source, valid


def side_metric(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, mode: str) -> tuple[float, int]:
    mask = np.broadcast_to(valid[:, None], prediction.shape[:2]).copy()
    mask &= np.isfinite(prediction).all(axis=-1) & np.isfinite(target).all(axis=-1)
    errors = world_aligned_mpjpe(
        prediction, target, joint_mask=mask, mode=mode,
        chunk_length=len(prediction), unit_scale=1000.0)
    finite = np.isfinite(errors)
    return ((float(errors[finite].mean()), int(finite.sum()))
            if finite.any() else (float("nan"), 0))


def score_window(dataset: str, segment_id: str, window_index: int, window: dict,
                 staged: Path, src_dir: Path, hawor_dirs: dict[str, str],
                 hawor_root: Path) -> list[dict]:
    cache = window["gt"]["cache_id"]
    gt = arrays(src_dir / cache / "gt.npz")
    gt_world = np.einsum(
        "tij,tspj->tspi", gt["camera_c2w"][:, :3, :3], gt["hand_joints_camera"]
    ) + gt["camera_c2w"][:, None, None, :3, 3]
    hawor_file = resolve_prediction_file(
        hawor_dirs.get(window["window_id"]), hawor_root, cache, "hawor", required=True)
    files = {
        "ego": staged / f"{window_index}_ego.npz",
        "wilor": src_dir / cache / "wilor.npz",
        "pad_hand": src_dir / cache / "pad_hand.npz",
        "egoforce": src_dir / cache / "egoforce.npz",
        "dyn_hamr": src_dir / cache / "dyn_hamr.npz",
        "hawor": hawor_file,
        "reviv4d": src_dir / cache / "reviv4d.npz",
    }
    output = []
    for method in METHODS:
        if not files[method].is_file():
            if method != "dyn_hamr":
                raise FileNotFoundError(f"missing required {method}: {files[method]}")
            output.append({
                "dataset": dataset, "segment_id": segment_id, "window_index": window_index,
                "window_id": window["window_id"], "cache_id": cache, "method": method,
                "pose_source": "missing", "sides": [
                    {"side": side, "w_valid_frame_count": 0, "wa_valid_frame_count": 0,
                     "w_mpjpe_mm": None, "wa_mpjpe_mm": None}
                    for side in ("left", "right")],
            })
            continue
        data = arrays(files[method])
        world, pose_source, valid = to_world(method, data, gt)
        sides = []
        for side_index, side in enumerate(("left", "right")):
            w_value, w_count = side_metric(world[:, side_index], gt_world[:, side_index],
                                           valid[:, side_index], "first2")
            wa_value, wa_count = side_metric(world[:, side_index], gt_world[:, side_index],
                                             valid[:, side_index], "all")
            sides.append({
                "side": side, "w_valid_frame_count": w_count,
                "wa_valid_frame_count": wa_count,
                "w_mpjpe_mm": json_value(w_value), "wa_mpjpe_mm": json_value(wa_value),
            })
        output.append({
            "dataset": dataset, "segment_id": segment_id, "window_index": window_index,
            "window_id": window["window_id"], "cache_id": cache, "method": method,
            "pose_source": pose_source, "sides": sides,
        })
    return output


def aggregate(rows: list[dict], window_indices: set[int] | None = None) -> dict:
    selected = rows if window_indices is None else [row for row in rows if row["window_index"] in window_indices]
    result = {"windows_total": 5}
    covered = set()
    for metric, count_key in (("w_mpjpe_mm", "w_valid_frame_count"),
                              ("wa_mpjpe_mm", "wa_valid_frame_count")):
        weighted_sum = 0.0
        count = 0
        for row in selected:
            for side in row["sides"]:
                value, weight = side[metric], int(side[count_key])
                if value is not None and weight > 0:
                    weighted_sum += float(value) * weight
                    count += weight
                    covered.add(row["window_index"])
        result[metric] = weighted_sum / count if count else None
        result[count_key] = count
    result["coverage_windows"] = len(covered)
    result["full_coverage"] = len(covered) == 5
    result["pose_sources"] = sorted({row["pose_source"] for row in selected})
    return result


def finite_margin(baseline: dict, ego: dict, metric: str) -> float | None:
    left, right = baseline.get(metric), ego.get(metric)
    return float(left - right) if left is not None and right is not None else None


def rank_segment(entry: dict, rows: list[dict]) -> dict:
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    methods = {method: aggregate(by_method[method]) for method in METHODS}
    margins_w, margins_wa = [], []
    beat_count = 0
    comparable_count = 0
    for method in BASELINES:
        baseline = methods[method]
        ego_compare = methods["ego"]
        if method == "dyn_hamr":
            intersection = {
                row["window_index"] for row in by_method[method]
                if any(side["w_valid_frame_count"] or side["wa_valid_frame_count"]
                       for side in row["sides"])
            }
            baseline["comparison_scope"] = "available_window_intersection"
            baseline["intersection_window_indices"] = sorted(intersection)
            baseline["intersection_windows"] = len(intersection)
            if intersection:
                ego_compare = aggregate(by_method["ego"], intersection)
                baseline["ego_w_mpjpe_on_intersection_mm"] = ego_compare["w_mpjpe_mm"]
                baseline["ego_wa_mpjpe_on_intersection_mm"] = ego_compare["wa_mpjpe_mm"]
            comparable = bool(intersection)
        else:
            baseline["comparison_scope"] = "full_five_windows"
            comparable = baseline["full_coverage"]
        w_margin = finite_margin(baseline, ego_compare, "w_mpjpe_mm") if comparable else None
        wa_margin = finite_margin(baseline, ego_compare, "wa_mpjpe_mm") if comparable else None
        comparable = comparable and w_margin is not None and wa_margin is not None
        baseline["comparable"] = comparable
        baseline["w_margin_vs_ego_mm"] = w_margin if comparable else None
        baseline["wa_margin_vs_ego_mm"] = wa_margin if comparable else None
        if comparable:
            comparable_count += 1
            margins_w.append(w_margin)
            margins_wa.append(wa_margin)
            beat_count += int(w_margin > 0)
    return {
        "dataset": entry["dataset"], "sequence_id": entry["sequence_id"],
        "segment_id": entry["gallery_stem"], "methods": methods,
        "beat_count": beat_count, "comparable_count": comparable_count,
        "all_comparable_beaten": bool(margins_w) and all(value > 0 for value in margins_w),
        "worst_w_margin_mm": min(margins_w) if margins_w else None,
        "median_w_margin_mm": float(np.median(margins_w)) if margins_w else None,
        "worst_wa_margin_mm": min(margins_wa) if margins_wa else None,
    }


def descending(value: float | None) -> float:
    return -float(value) if value is not None else math.inf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--staged", action="append", default=[], metavar="DATASET=PATH", required=True)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--hawor-root", action="append", default=[], metavar="DATASET=PATH", required=True)
    parser.add_argument("--hawor-index", action="append", default=[], metavar="DATASET=PATH", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ego-commit", default="8fc061a615895bd3b5a556f7387bae306e32d9db")
    parser.add_argument("--visualization-commit", required=True)
    args = parser.parse_args()

    raw_manifest = args.manifest.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("MANIFEST_SHA256_MISMATCH")
    entries = [json.loads(line) for line in raw_manifest.splitlines() if line]
    if len(entries) != 104 or Counter(row["dataset"] for row in entries) != EXPECTED_COUNTS:
        raise RuntimeError("MANIFEST_COUNT_MISMATCH")
    staged_roots, hawor_roots, hawor_indexes = map(
        mappings, (args.staged, args.hawor_root, args.hawor_index))
    if set(staged_roots) != set(EXPECTED_COUNTS) or set(hawor_roots) != set(EXPECTED_COUNTS) or set(hawor_indexes) != set(EXPECTED_COUNTS):
        raise RuntimeError("DATASET_MAPPING_MISMATCH")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)

    hawor_maps = {}
    for dataset, index in hawor_indexes.items():
        hawor_maps[dataset] = {
            row["window_id"]: row.get("prediction_dir")
            for row in (json.loads(line) for line in index.read_text().splitlines() if line.strip())
        }

    all_window_rows, ranked = [], []
    for number, entry in enumerate(entries, 1):
        dataset, segment_id = entry["dataset"], entry["gallery_stem"]
        staged = staged_roots[dataset] / segment_id
        selection = json.loads((staged / "selection.json").read_text())
        if [window["window_id"] for window in selection["windows"]] != [window["window_id"] for window in entry["windows"]]:
            raise RuntimeError(f"WINDOW_ID_MISMATCH:{segment_id}")
        rows = []
        for window_index, window in enumerate(entry["windows"]):
            rows.extend(score_window(
                dataset, segment_id, window_index, window, staged, args.src_dir,
                hawor_maps[dataset], hawor_roots[dataset]))
        all_window_rows.extend(rows)
        ranked.append(rank_segment(entry, rows))
        print(f"scored {number}/104 {segment_id}", flush=True)

    ranked.sort(key=lambda row: (
        -row["beat_count"], descending(row["worst_w_margin_mm"]),
        descending(row["median_w_margin_mm"]), descending(row["worst_wa_margin_mm"]),
        row["methods"]["ego"]["w_mpjpe_mm"] or math.inf, row["segment_id"]))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
        row["ranked_stem"] = f"{rank:03d}__{row['segment_id']}"

    with (args.output_root / "window_metrics.jsonl").open("w") as stream:
        for row in all_window_rows:
            stream.write(json.dumps(json_value(row), sort_keys=True) + "\n")
    with (args.output_root / "ranking_104.jsonl").open("w") as stream:
        for row in ranked:
            stream.write(json.dumps(json_value(row), sort_keys=True) + "\n")
    fieldnames = ["rank", "ranked_stem", "dataset", "segment_id", "beat_count",
                  "comparable_count", "all_comparable_beaten", "worst_w_margin_mm",
                  "median_w_margin_mm", "worst_wa_margin_mm"]
    with (args.output_root / "ranking_104.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in ranked)
    summary = {
        "status": "complete", "segments": 104, "windows": 520,
        "dataset_counts": EXPECTED_COUNTS,
        "dyn_hamr_intersection_segments": sum(
            row["methods"]["dyn_hamr"].get("intersection_windows", 0) > 0 for row in ranked),
        "dyn_hamr_intersection_windows": sum(
            row["methods"]["dyn_hamr"].get("intersection_windows", 0) for row in ranked),
        "hawor_full_coverage_segments": sum(row["methods"]["hawor"]["full_coverage"] for row in ranked),
        "ranking_protocol": ["beat_count descending", "worst W margin descending",
                             "median W margin descending", "worst WA margin descending",
                             "Ego W-MPJPE ascending"],
        "comparison_rule": "full five-window coverage except Dyn-HaMR, which uses its exact available-window intersection and the matching EgoFound3R windows",
        "metric_protocol": "one Sim(3) per 60f window; W fits first two valid frames, WA fits all valid points",
        "metric_geometry": "21 joints for all methods", "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "ego_inference_commit": args.ego_commit,
        "visualization_commit": args.visualization_commit,
        "ranked_stems": [row["ranked_stem"] for row in ranked],
    }
    if summary["hawor_full_coverage_segments"] != 104:
        raise RuntimeError("HAWOR_COVERAGE_INCOMPLETE")
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
