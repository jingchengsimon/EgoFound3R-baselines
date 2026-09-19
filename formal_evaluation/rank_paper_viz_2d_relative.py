#!/usr/bin/env python3
"""Rank the frozen 114 windows by Ego's 2D advantage over every baseline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

BASELINES = ("wilor", "pad_hand", "hawor", "reviv4d")
EXCLUDED_BASELINES = {
    "egoforce": "113/114 positive-depth joint coverage; excluded globally, never per-window",
}
JOINT_REORDER = {
    "pad_hand": (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 20, 7, 8, 9, 19),
    "reviv4d": (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20),
}


def segment_id(entry: dict) -> str:
    sequence = str(entry.get("sequence_id") or entry.get("window_id") or "unknown")
    key = hashlib.sha1(sequence.encode()).hexdigest()[:10]
    return f"{entry['dataset']}__{key}__{entry['frame_ids'][0]}-{entry['frame_ids'][-1]}"


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def camera_joints(data: dict) -> np.ndarray:
    if "hand_joints_camera" in data:
        return data["hand_joints_camera"].astype(float)
    if "hand_joints_world" in data:
        pose = data["camera_c2w"].astype(float)
        return np.einsum("tji,tsvj->tsvi", pose[:, :3, :3],
                         data["hand_joints_world"].astype(float) - pose[:, None, None, :3, 3])
    raise KeyError("NO_HAND_JOINTS:" + ",".join(sorted(data)))


def valid_hands(data: dict, joints: np.ndarray, index: int) -> np.ndarray:
    valid = (np.isfinite(joints[index]).all(axis=(1, 2))
             & (joints[index, :, :, 2] > 1e-6).all(axis=1))
    if "hand_valid" in data:
        valid &= np.asarray(data["hand_valid"][index], bool)
    return valid


def project(joints: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = joints @ intrinsics.T
    return homogeneous[..., :2] / np.where(homogeneous[..., 2:] > 1e-6,
                                             homogeneous[..., 2:], np.nan)


def frame_metrics(entry: dict, args) -> dict:
    wanted = str(entry["frame_id"])
    matches = [index for index, frame in enumerate(entry["frame_ids"])
               if str(frame) == wanted]
    if len(matches) != 1:
        raise ValueError(f"ANCHOR_NOT_UNIQUE:{entry['window_id']}:{wanted}:{len(matches)}")
    ref = entry["frame_refs"][matches[0]]
    cache, local = str(ref["cache_id"]), int(ref["index"])
    staged = args.staged_root / segment_id(entry)
    selection = json.loads((staged / "selection.json").read_text())
    windows = [window for window in selection["windows"] if window["cache_id"] == cache]
    if len(windows) != 1:
        raise ValueError(f"STAGED_CACHE_NOT_UNIQUE:{cache}:{len(windows)}")
    staged_index = int(windows[0]["index"])
    paths = {
        "ego": staged / f"{staged_index}_ego.npz",
        **{method: args.source_root / cache / f"{method}.npz"
           for method in BASELINES if method != "hawor"},
        "hawor": args.hawor_root / entry["dataset"] / cache / "predictions.npz",
        "gt": args.source_root / cache / "gt.npz",
    }
    missing = [method for method, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MISSING_METHODS:{cache}:{','.join(missing)}")
    data = {method: load_npz(path) for method, path in paths.items()}
    joints = {method: camera_joints(value) for method, value in data.items()}
    for method, order in JOINT_REORDER.items():
        joints[method] = joints[method][..., np.asarray(order), :]
    prepared_window = args.prepared_root / entry["dataset"] / cache
    record = json.loads((prepared_window / "window_input.json").read_text())
    if str(record["frame_ids"][local]) != wanted:
        raise ValueError(f"FRAME_ID_MISMATCH:{cache}:{local}")
    rgb_path = prepared_window / "rgb" / f"{local:03d}_{wanted}.png"
    with Image.open(rgb_path) as image:
        width, height = image.size
    scale = 512 / max(width, height)
    display = np.array([[scale, 0, (scale - 1) / 2],
                        [0, scale, (scale - 1) / 2], [0, 0, 1.]])
    calibrated = display @ np.asarray(record["intrinsics"][local], float)
    ego_k = np.asarray(data["ego"]["intrinsics_pred"][local], float)
    affine = np.asarray(data["ego"].get("input_affine", np.eye(3)), float)
    intrinsics = {method: calibrated for method in (*BASELINES, "gt")}
    intrinsics["ego"] = display @ np.linalg.inv(affine) @ ego_k
    methods = ("ego", *BASELINES)
    common = valid_hands(data["gt"], joints["gt"], local)
    for method in methods:
        common &= valid_hands(data[method], joints[method], local)
    if not common.any():
        raise ValueError(f"NO_COMMON_VALID_HAND:{cache}:{local}")
    output = {}
    target = project(joints["gt"][local], intrinsics["gt"])
    diagonal = float(np.hypot(round(width * scale), round(height * scale)))
    for method in methods:
        prediction = project(joints[method][local], intrinsics[method])
        errors = np.linalg.norm(prediction[common] - target[common], axis=-1)
        if not np.isfinite(errors).all() or errors.size != int(common.sum()) * 21:
            raise ValueError(f"INVALID_PROJECTED_JOINTS:{method}:{cache}:{local}")
        output[method] = {"joint_error_px": float(errors.mean()),
                          "joint_error_normalized": float(errors.mean() / diagonal)}
    return {
        "dataset": entry["dataset"], "dataset_rank": entry.get("dataset_rank"),
        "window_id": entry["window_id"], "sequence_id": entry["sequence_id"],
        "frame_id": wanted, "cache_id": cache, "local_index": local,
        "common_hands": int(common.sum()), "original_ego_score": entry.get("score"),
        "methods": output,
    }


def add_relative_scores(rows: list[dict]) -> list[dict]:
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset_rows in by_dataset.values():
        for row in dataset_rows:
            best_method = min(BASELINES,
                              key=lambda method: row["methods"][method]["joint_error_normalized"])
            best_error = row["methods"][best_method]["joint_error_normalized"]
            ego_error = row["methods"]["ego"]["joint_error_normalized"]
            row.update(best_baseline=best_method, best_baseline_error=best_error,
                       ego_error=ego_error, ego_advantage=best_error - ego_error,
                       ego_beats_all_baselines=ego_error < best_error)
    return sorted(rows, key=lambda row: (-row["ego_advantage"],
                                         -(row.get("original_ego_score") or -1),
                                         row["dataset"], row["window_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.manifest_sha256:
        raise ValueError("MANIFEST_SHA256_MISMATCH")
    entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(entries) != 114:
        raise ValueError(f"FROZEN_SEGMENT_COUNT:{len(entries)}")
    rows, errors = [], []
    for index, entry in enumerate(entries, 1):
        try:
            rows.append(frame_metrics(entry, args))
        except Exception as error:
            errors.append({"dataset": entry["dataset"], "window_id": entry["window_id"],
                           "error": type(error).__name__ + ":" + str(error)})
            if len(errors) == 1:
                print(json.dumps({"first_error": errors[0]}), flush=True)
        if index % 10 == 0:
            print(json.dumps({"processed": index, "valid": len(rows), "errors": len(errors)}),
                  flush=True)
    ranked = add_relative_scores(rows) if rows else []
    for rank, row in enumerate(ranked, 1):
        row["global_rank"] = rank
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "report.json").exists():
        raise FileExistsError(args.output_root / "report.json")
    (args.output_root / "ranking.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in ranked))
    summary = {
        "status": "complete" if len(ranked) == 114 and not errors else "incomplete",
        "manifest_sha256": args.manifest_sha256, "windows": len(entries),
        "ranked_windows": len(ranked), "errors": errors,
        "baseline_roster": list(BASELINES),
        "excluded_baselines": EXCLUDED_BASELINES,
        "ranking": "descending ego_advantage = min baseline alignment_error - ego alignment_error",
        "alignment_error": "mean 21-joint 2D reprojection error / rendered image diagonal",
        "dataset_counts": dict(Counter(row["dataset"] for row in ranked)),
        "ego_beats_all_baselines": sum(row["ego_beats_all_baselines"] for row in ranked),
        "top10": [
            {key: row[key] for key in (
                "global_rank", "dataset", "window_id", "frame_id", "ego_advantage",
                "ego_error", "best_baseline", "best_baseline_error")}
            for row in ranked[:10]
        ],
    }
    (args.output_root / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "complete":
        raise SystemExit(2)
    (args.output_root / "COMPLETE").write_text(json.dumps({
        "status": "complete", "ranked_windows": len(ranked),
        "ego_beats_all_baselines": summary["ego_beats_all_baselines"],
    }, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("status", "ranked_windows", "dataset_counts",
                       "ego_beats_all_baselines", "top10")}))


if __name__ == "__main__":
    main()
