#!/usr/bin/env python3
"""Evaluate cached InteractVLM hcontact predictions on H2O geometry.

This adapter deliberately does not run InteractVLM.  It evaluates saved
SMPL-H hcontact probabilities against an H2O-derived MANO vertex contact map.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


OBJECT_NAMES = {
    1: "book", 2: "espresso", 3: "lotion", 4: "spray",
    5: "milk", 6: "cocoa", 7: "chips", 8: "cappuccino",
}


def metric(counts: list[int]) -> tuple[float, float, float]:
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def sample_surface(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministically sample triangle surface points without optional rtree."""
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.any(areas > 0):
        raise ValueError("object mesh has zero total area")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=n, replace=True, p=areas / areas.sum())
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    t = tri[chosen]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True,
                    help="ContactOpt right-hand H2O cache created from the formal manifest")
    ap.add_argument("--frame-index", type=Path, required=True)
    ap.add_argument("--input-manifest", type=Path, required=True,
                    help="Manifest written by prepare_interactvlm_hcontact_inputs.py")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--segmentation", type=Path, required=True)
    ap.add_argument("--h2o-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gt-distance-m", type=float, default=0.018)
    ap.add_argument("--prediction-threshold", type=float, default=0.5)
    ap.add_argument("--surface-samples", type=int, default=200_000)
    ap.add_argument("--distance-workers", type=int, default=8,
                    help="Bound CPU use while shared GPU training is active")
    ap.add_argument("--expect-frames", type=int, default=30_724)
    ap.add_argument("--expect-windows", type=int, default=2_579)
    args = ap.parse_args()

    # ContactOpt's pickle contains HandObject instances.  Import its class from
    # the current working directory before unpickling; no baseline source is modified.
    sys.path.insert(0, ".")

    entries = {}
    for line in args.frame_index.open():
        row = json.loads(line)
        entries[(row["sequence"], str(row["frame_id"]).zfill(6))] = row
    if len(entries) != args.expect_frames:
        raise RuntimeError(f"frame index has {len(entries)}, expected {args.expect_frames}")
    prediction_stems = {}
    for line in args.input_manifest.open():
        row = json.loads(line)
        key = (row["sequence"], str(row["frame_id"]).zfill(6))
        prediction_stems[key] = Path(row["input_path"]).stem
    if len(prediction_stems) != args.expect_frames:
        raise RuntimeError(f"input manifest has {len(prediction_stems)}, expected {args.expect_frames}")

    with args.segmentation.open("rb") as f:
        segments = pickle.load(f)
    right_smplh = np.asarray(segments["right hand"], dtype=np.int64)
    if len(right_smplh) != 778 or len(np.unique(right_smplh)) != 778:
        raise RuntimeError("unexpected SMPL-H right-hand segmentation")

    # In local-index space, MANO vertex i is SMPL-H-hand local index
    # (i - 682) mod 778.  The mapping was topology-verified against all 1538
    # right-hand faces of SMPL-H and MANO_RIGHT.
    mano_to_smplh = right_smplh[(np.arange(778, dtype=np.int64) - 682) % 778]

    surface_trees: dict[int, cKDTree] = {}
    for object_id, name in OBJECT_NAMES.items():
        candidates = sorted((args.h2o_root / "object" / name).glob("*.obj"))
        if not candidates:
            raise FileNotFoundError(f"missing CAD mesh for {name}")
        import trimesh
        mesh = trimesh.load(candidates[0], process=False, force="mesh")
        points = sample_surface(np.asarray(mesh.vertices, np.float32),
                                np.asarray(mesh.faces, np.int64),
                                args.surface_samples, seed=object_id)
        surface_trees[object_id] = cKDTree(points)

    # The cache holds the H2O GT MANO vertices used by prior S²Contact and
    # ContactOpt runs.  Loading it needs the ContactOpt runtime import path.
    with args.cache.open("rb") as f:
        rows = pickle.load(f)
    if len(rows) != args.expect_frames:
        raise RuntimeError(f"cache has {len(rows)}, expected {args.expect_frames}")

    total = [0, 0, 0]
    per_window: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    missing_predictions: list[str] = []
    object_counts: dict[str, int] = defaultdict(int)

    for index, row in enumerate(rows):
        key = (row["h2o_sequence"], str(row["h2o_frame_id"]).zfill(6))
        entry = entries.get(key)
        if entry is None:
            raise KeyError(f"cache frame not in manifest: {key}")
        pose = np.loadtxt(entry["obj_pose_rt_path"], dtype=np.float32).reshape(-1)
        object_id = int(round(float(pose[0])))
        object_name = OBJECT_NAMES[object_id]
        transform = pose[1:].reshape(4, 4)
        rotation, translation = transform[:3, :3], transform[:3, 3]

        hand_camera = np.asarray(row["ho_gt"].hand_verts, dtype=np.float32)
        hand_object = (hand_camera - translation) @ rotation
        gt = surface_trees[object_id].query(hand_object, workers=args.distance_workers)[0] <= args.gt_distance_m

        stem = prediction_stems[key]
        pred_path = args.output_dir / f"{stem}_hcontact_vertices.npz"
        if not pred_path.exists():
            missing_predictions.append(str(pred_path))
            continue
        scores = np.load(pred_path)["pred_contact_3d_smplh"].reshape(-1)
        if scores.shape != (6890,):
            raise RuntimeError(f"unexpected prediction shape {scores.shape}: {pred_path}")
        pred = scores[mano_to_smplh] >= args.prediction_threshold
        counts = [int(np.sum(pred & gt)), int(np.sum(pred & ~gt)), int(np.sum(~pred & gt))]
        total = [total[i] + counts[i] for i in range(3)]
        for window in row["h2o_window_ids"]:
            per_window[window] = [per_window[window][i] + counts[i] for i in range(3)]
        object_counts[object_name] += 1
        if (index + 1) % 1000 == 0:
            print(f"processed {index + 1}/{len(rows)}", flush=True)

    if missing_predictions:
        raise FileNotFoundError(f"missing {len(missing_predictions)} predictions; first: {missing_predictions[0]}")
    if len(per_window) != args.expect_windows:
        raise RuntimeError(f"evaluated {len(per_window)} windows, expected {args.expect_windows}")
    f1s = [metric(counts)[0] for counts in per_window.values()]
    f1, precision, recall = metric(total)
    result = {
        "baseline": "interactvlm",
        "checkpoint": "interactvlm-3d-hcontact-damon",
        "cache_samples": len(rows),
        "evaluated_frames": len(rows),
        "evaluated_windows": len(per_window),
        "h2o_protocol": "right-hand 778-vertex hcontact; GT MANO vertex-to-object-surface distance",
        "prediction": {"threshold": args.prediction_threshold, "field": "pred_contact_3d_smplh"},
        "ground_truth": {
            "distance_threshold_m": args.gt_distance_m,
            "surface_samples_per_object": args.surface_samples,
            "source": "H2O GT MANO hand mesh + GT object CAD/object-to-camera pose",
            "mano_to_smplh_mapping": "right-hand topology verified; smplh_local=(mano_index-682) mod 778",
        },
        "global_micro": {"f1": f1, "precision": precision, "recall": recall, "counts": total},
        "window_macro_f1": float(np.mean(f1s)),
        "object_frame_counts": dict(sorted(object_counts.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
