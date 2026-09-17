"""Stage per-window Ego inputs from the authoritative 8fc061a infer artifacts.

The Ego columns of every paper figure/video must come from
``infer_marker_multiclip.py`` at commit 8fc061a (tag
``final-no-hand-completion-20260908``) with all default post-processing on.
This tool slices that frozen 300-frame output into the 60-frame windows of one
segment, renames the fields to the renderer contract, records the keep_aspect
input affine, and prints a projection sanity report against calibrated-K GT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_WINDOW = 60


def project(K: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = points @ K.T
    depth = np.where(homogeneous[..., 2:] > 1e-6, homogeneous[..., 2:], np.nan)
    return homogeneous[..., :2] / depth


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a).all(-1)], b[np.isfinite(b).all(-1)]
    if not len(a) or not len(b):
        return float("nan")
    lo = np.maximum(a.min(0), b.min(0))
    hi = np.minimum(a.max(0), b.max(0))
    inter = np.clip(hi - lo, 0, None).prod()
    area = np.ptp(a, axis=0).prod() + np.ptp(b, axis=0).prod() - inter
    return float(inter / max(area, 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer-dir", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--caches", nargs="+", required=True,
                        help="60-frame window cache ids in chronological order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames-per-window", type=int, default=DEFAULT_WINDOW)
    args = parser.parse_args()

    payload = torch.load(args.infer_dir / "inference_output.pt", map_location="cpu",
                         weights_only=False)
    with np.load(args.npz, allow_pickle=True) as archive:
        arrays = {key: archive[key] for key in archive.files}
    provenance = payload["provenance"]
    source_hw = tuple(int(v) for v in provenance["source_video_size_hw"])
    model_hw = tuple(int(v) for v in provenance["model_input_size_hw"])
    affine = np.diag([model_hw[1] / source_hw[1], model_hw[0] / source_hw[0], 1.0])
    if provenance.get("git_commit") != "8fc061a615895bd3b5a556f7387bae306e32d9db":
        raise ValueError(f"unexpected infer commit: {provenance.get('git_commit')}")
    print(f"infer commit {provenance['git_commit'][:8]} dirty={provenance.get('git_dirty')}")
    print(f"source {source_hw} -> model {model_hw}; affine sx={affine[0, 0]:.10f} sy={affine[1, 1]:.10f}")

    distance = payload["metric_predictions"]["hand_778"]["vertex_contact_distance_metric"]
    distance = distance.float().numpy().astype(np.float32)
    vertices = arrays["vertex778_xyz_camera"].astype(np.float32)
    joints = arrays["joint21_xyz_camera"].astype(np.float32)
    visibility = arrays["vertex778_visibility_probability"].astype(np.float32)
    contact = arrays["vertex778_contact_probability"].astype(np.float32)
    validity = arrays["hand_valid"].astype(bool)
    intrinsics = arrays["intrinsics_full"].astype(np.float32)
    c2w = arrays["camera_pose_c2w"].astype(np.float32)

    window = args.frames_per_window
    total = len(vertices)
    if len(args.caches) * window != total:
        raise ValueError(f"{len(args.caches)} windows x {window} != {total} infer frames")
    args.out.mkdir(parents=True, exist_ok=True)

    sx, sy = affine[0, 0], affine[1, 1]
    records = []
    print("[projection check] Ego(pred K+affine) vs GT(calibrated K) hand bbox IoU")
    for index, cache in enumerate(args.caches):
        root = args.prepared_root / cache
        record = json.loads((root / "window_input.json").read_text())
        frame_ids = list(record["frame_ids"])
        if len(frame_ids) != window:
            raise ValueError(f"{cache}: expected {window} frames, got {len(frame_ids)}")
        sl = slice(index * window, (index + 1) * window)
        np.savez_compressed(
            args.out / f"{index}_ego.npz",
            hand_vertices_camera=vertices[sl],
            hand_joints_camera=joints[sl],
            hand_valid=validity[sl],
            vertex_visibility_probability=visibility[sl],
            vertex_contact_probability=contact[sl],
            vertex_contact_distance=distance[sl],
            intrinsics_pred=intrinsics[sl],
            camera_c2w=c2w[sl],
            frame_ids=np.array(frame_ids),
            input_affine=affine,
            source_size_hw=np.array(source_hw),
            model_size_hw=np.array(model_hw),
        )
        ious = []
        for local in range(0, window, 15):
            K_cal = np.array(record["intrinsics"][local], float)
            K_ego = np.diag([1.0 / sx, 1.0 / sy, 1.0]) @ intrinsics[index * window + local].astype(float)
            geometry = np.load(root / "geometry" / f"{local:03d}_{frame_ids[local]}.npz",
                               allow_pickle=False)
            side = int(np.argmax(geometry["hand_valid"].astype(bool)))
            ious.append(bbox_iou(project(K_ego, vertices[index * window + local][side].astype(float)),
                                 project(K_cal, geometry["hand_vertices"][side].astype(float))))
        records.append({"cache_id": cache, "window_id": record["window_id"],
                        "sequence_id": record["sequence_id"], "frame_ids": frame_ids})
        print(f"  window {index} {cache[:8]} frames {frame_ids[0]}..{frame_ids[-1]} "
              f"bboxIoU={np.nanmean(ious):.3f}")

    selection = {
        "dataset": args.dataset,
        "sequence_id": records[0]["sequence_id"],
        "segment_id": (f"{args.dataset}__{records[0]['frame_ids'][0]}-"
                       f"{records[-1]['frame_ids'][-1]}"),
        "windows": [{"index": i, "window_id": r["window_id"], "cache_id": r["cache_id"],
                     "gt": {"cache_id": r["cache_id"]}, "frame_ids": r["frame_ids"]}
                    for i, r in enumerate(records)],
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"wrote {len(records)} ego windows + selection.json to {args.out}")


if __name__ == "__main__":
    main()
