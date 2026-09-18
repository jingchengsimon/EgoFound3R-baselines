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


def entry_frame_plan(entry: dict) -> tuple[list[str], dict[str, list[tuple[int, int, object]]]]:
    """Map exact clip-order frames to their source cache and local window index."""
    frame_ids = list(entry.get("frame_ids") or [])
    refs = list(entry.get("frame_refs") or [])
    if not frame_ids or len(frame_ids) != len(refs):
        raise ValueError(f"frame_ids/frame_refs mismatch: {len(frame_ids)} vs {len(refs)}")
    caches: list[str] = []
    plan: dict[str, list[tuple[int, int, object]]] = {}
    for source_index, (frame_id, ref) in enumerate(zip(frame_ids, refs)):
        cache = ref.get("cache_id")
        if not cache or "index" not in ref:
            raise ValueError(f"invalid frame_ref at {source_index}: {ref}")
        if cache not in plan:
            caches.append(cache)
            plan[cache] = []
        plan[cache].append((source_index, int(ref["index"]), frame_id))
    return caches, plan


def padded_window(array: np.ndarray, assignments: list[tuple[int, int, object]],
                  window: int, fill_value) -> np.ndarray:
    result = np.full((window, *array.shape[1:]), fill_value, dtype=array.dtype)
    for source_index, local_index, _ in assignments:
        result[local_index] = array[source_index]
    return result


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
    parser.add_argument("--caches", nargs="*", default=None,
                        help="legacy full 60-frame cache ids in chronological order")
    parser.add_argument("--entry-json", type=Path,
                        help="exact manifest entry with frame_ids/frame_refs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method-name", choices=("ego", "ego_gt_k"), default="ego")
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
    arrays_by_name = {
        "hand_vertices_camera": vertices,
        "hand_joints_camera": joints,
        "hand_valid": validity,
        "vertex_visibility_probability": visibility,
        "vertex_contact_probability": contact,
        "vertex_contact_distance": distance,
        "intrinsics_pred": intrinsics,
        "camera_c2w": c2w,
    }
    lengths = {name: len(value) for name, value in arrays_by_name.items()}
    if set(lengths.values()) != {total}:
        raise ValueError(f"inference array length mismatch: {lengths}")
    entry = json.loads(args.entry_json.read_text()) if args.entry_json else None
    if entry is not None:
        if entry.get("dataset") != args.dataset:
            raise ValueError(f"entry dataset {entry.get('dataset')} != {args.dataset}")
        caches, plan = entry_frame_plan(entry)
        grouped_order = [(cache, local) for cache in caches
                         for _, local, _ in plan[cache]]
        manifest_order = [(ref["cache_id"], int(ref["index"]))
                          for ref in entry["frame_refs"]]
        if grouped_order != manifest_order:
            raise ValueError("frame_refs revisit a cache; non-contiguous cache spans are unsupported")
        if total != len(entry["frame_ids"]):
            raise ValueError(f"manifest has {len(entry['frame_ids'])} frames, inference has {total}")
        if args.caches and list(args.caches) != caches:
            raise ValueError(f"--caches {args.caches} != manifest caches {caches}")
    else:
        caches = list(args.caches or [])
        if not caches:
            parser.error("one of --entry-json or --caches is required")
        if len(caches) * window != total:
            raise ValueError(f"{len(caches)} windows x {window} != {total} infer frames")
        plan = {cache: [(i * window + j, j, None) for j in range(window)]
                for i, cache in enumerate(caches)}
    args.out.mkdir(parents=True, exist_ok=True)

    sx, sy = affine[0, 0], affine[1, 1]
    records = []
    print("[projection check] Ego(pred K+affine) vs GT(calibrated K) hand bbox IoU")
    for index, cache in enumerate(caches):
        root = args.prepared_root / cache
        record = json.loads((root / "window_input.json").read_text())
        frame_ids = list(record["frame_ids"])
        if len(frame_ids) != window:
            raise ValueError(f"{cache}: expected {window} frames, got {len(frame_ids)}")
        assignments = plan[cache]
        for source_index, local_index, expected_frame_id in assignments:
            if not 0 <= local_index < window:
                raise ValueError(f"{cache}: frame index {local_index} outside 0..{window - 1}")
            if expected_frame_id is not None and str(frame_ids[local_index]) != str(expected_frame_id):
                raise ValueError(
                    f"{cache}[{local_index}]={frame_ids[local_index]} != manifest {expected_frame_id}")
        np.savez_compressed(
            args.out / f"{index}_{args.method_name}.npz",
            hand_vertices_camera=padded_window(vertices, assignments, window, np.nan),
            hand_joints_camera=padded_window(joints, assignments, window, np.nan),
            hand_valid=padded_window(validity, assignments, window, False),
            vertex_visibility_probability=padded_window(visibility, assignments, window, np.nan),
            vertex_contact_probability=padded_window(contact, assignments, window, np.nan),
            vertex_contact_distance=padded_window(distance, assignments, window, np.nan),
            intrinsics_pred=padded_window(intrinsics, assignments, window, np.nan),
            camera_c2w=padded_window(c2w, assignments, window, np.nan),
            frame_ids=np.array(frame_ids),
            input_affine=affine,
            source_size_hw=np.array(source_hw),
            model_size_hw=np.array(model_hw),
        )
        ious = []
        for source_index, local, _ in assignments[::15] or assignments[:1]:
            K_cal = np.array(record["intrinsics"][local], float)
            K_ego = np.diag([1.0 / sx, 1.0 / sy, 1.0]) @ intrinsics[source_index].astype(float)
            geometry = np.load(root / "geometry" / f"{local:03d}_{frame_ids[local]}.npz",
                               allow_pickle=False)
            side = int(np.argmax(geometry["hand_valid"].astype(bool)))
            ious.append(bbox_iou(project(K_ego, vertices[source_index][side].astype(float)),
                                 project(K_cal, geometry["hand_vertices"][side].astype(float))))
        records.append({"cache_id": cache, "window_id": record["window_id"],
                        "sequence_id": record["sequence_id"], "frame_ids": frame_ids,
                        "frame_indices": [local for _, local, _ in assignments]})
        print(f"  window {index} {cache[:8]} selected={len(assignments)} "
              f"frames {frame_ids[assignments[0][1]]}..{frame_ids[assignments[-1][1]]} "
              f"bboxIoU={np.nanmean(ious):.3f}")

    sequence_id = str(entry.get("sequence_id")) if entry is not None else records[0]["sequence_id"]
    if any(str(record["sequence_id"]) != sequence_id for record in records):
        raise ValueError("selected caches do not belong to one sequence")
    selection = {
        "dataset": args.dataset,
        "sequence_id": sequence_id,
        "segment_id": (entry.get("segment_id") if entry is not None else
                       f"{args.dataset}__{records[0]['frame_ids'][0]}-"
                       f"{records[-1]['frame_ids'][-1]}"),
        "frame_ids": (list(entry["frame_ids"]) if entry is not None else
                      [frame_id for record in records for frame_id in record["frame_ids"]]),
        "frame_refs": (list(entry["frame_refs"]) if entry is not None else
                       [{"cache_id": record["cache_id"], "index": local}
                        for record in records for local in record["frame_indices"]]),
        "center_index_in_clip": (entry.get("center_index_in_clip") if entry is not None else None),
        "windows": [{"index": i, "window_id": r["window_id"], "cache_id": r["cache_id"],
                     "gt": {"cache_id": r["cache_id"]}, "frame_ids": r["frame_ids"],
                     "frame_indices": r["frame_indices"]}
                    for i, r in enumerate(records)],
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"wrote {len(records)} {args.method_name} windows + selection.json to {args.out}")


if __name__ == "__main__":
    main()
