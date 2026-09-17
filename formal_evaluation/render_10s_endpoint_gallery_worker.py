"""Render the frozen 104 endpoint-renderable segments with a bounded CPU pool."""

import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np


RELATIVE = "visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914"


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def resolve_registered_path(path, spec):
    """Resolve the frozen CPFS spelling through its registered workspace alias."""
    source = Path(path)
    alias = Path(spec["cpfs_alias"])
    target = Path(spec["cpfs_alias_target"])
    return target / source.relative_to(alias) if source.is_relative_to(alias) else source


def index_rows(paths):
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    by_id = {}
    for row in rows:
        for key in (row.get("window_id"), row.get("cache_id"), Path(row.get("window_input", "")).parent.name):
            if key:
                by_id.setdefault(key, []).append(row)
    return by_id


def one_prediction(roots, cache_id, method, dataset, ids):
    found = [Path(root) / cache_id for root in roots
             if (Path(root) / cache_id / "predictions.npz").is_file()
             and (Path(root) / cache_id / "metadata.json").is_file()]
    if len(found) != 1:
        raise ValueError(f"{method}_SOURCE_MATCH_COUNT:{cache_id}:{len(found)}")
    metadata = json.loads((found[0] / "metadata.json").read_text())
    if metadata.get("dataset") != dataset or metadata.get("method") != method or metadata.get("frame_ids") != ids:
        raise ValueError(f"{method}_IDENTITY_MISMATCH:{cache_id}")
    return found[0]


def prediction_index(path, label):
    index = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        window_id = row.get("window_id")
        if not window_id or window_id in index:
            raise ValueError(f"{label}_WINDOW_ID_INVALID:{window_id}")
        index[window_id] = row
    return index


def hawor_native_prediction(index_path, root, window_id, dataset, ids, index=None, path=None):
    if path is None:
        index = prediction_index(index_path, "HAWOR_NATIVE_INDEX") if index is None else index
        row = index.get(window_id)
        if row is None:
            raise ValueError(f"HAWOR_NATIVE_WINDOW_MISSING:{window_id}")
        path = Path(row["prediction_dir"])
    else:
        path = Path(path)
    formal_root = Path(root) / "hawor" / "formal"
    if not path.is_relative_to(formal_root):
        raise ValueError(f"HAWOR_NATIVE_PATH_OUTSIDE_REGISTERED_ROOT:{path}")
    required = (path / "predictions.npz", path / "metadata.json", path / "run.json",
                path / "native" / "metadata.json", path / "native" / "predictions.npz")
    if any(not name.is_file() for name in required):
        raise ValueError(f"HAWOR_NATIVE_SOURCE_MISSING:{window_id}:{path}")
    metadata = json.loads((path / "metadata.json").read_text())
    run = json.loads((path / "run.json").read_text())
    if (metadata.get("dataset") != dataset or metadata.get("method") != "hawor"
            or metadata.get("window_id") != window_id or metadata.get("frame_ids") != ids):
        raise ValueError(f"HAWOR_NATIVE_IDENTITY_MISMATCH:{window_id}")
    if run.get("status") != "success" or metadata.get("detail", {}).get("slam_failed_identity_fallback") is not False:
        raise ValueError(f"HAWOR_NATIVE_CAMERA_PROVENANCE_INVALID:{window_id}")
    return path


def validate_hawor_native_geometry(path, window_id):
    expected = {
        "camera_c2w": (60, 4, 4), "camera_valid": (60,),
        "hand_joints_world": (60, 2, 21, 3), "hand_markers_world": (60, 2, 195, 3),
        "hand_vertices_world": (60, 2, 778, 3), "hand_valid": (60, 2),
    }
    with np.load(path / "predictions.npz", allow_pickle=False) as source:
        shapes = {key: source[key].shape for key in expected if key in source.files}
        if shapes != expected:
            raise ValueError(f"HAWOR_NATIVE_GEOMETRY_SHAPE:{window_id}:{shapes}")
        camera_valid = source["camera_valid"].astype(bool)
        hand_valid = source["hand_valid"].astype(bool)
        if not camera_valid.all() or not np.isfinite(source["camera_c2w"]).all():
            raise ValueError(f"HAWOR_NATIVE_CAMERA_INVALID:{window_id}")
        for key in ("hand_joints_world", "hand_markers_world", "hand_vertices_world"):
            if not np.isfinite(source[key][hand_valid]).all():
                raise ValueError(f"HAWOR_NATIVE_HAND_INVALID:{window_id}:{key}")


def indexed_native_prediction(index, formal_root, window_id, method, dataset, ids):
    """Resolve a frozen prediction index through its registered live formal root."""
    row = index.get(window_id)
    if row is None:
        return None
    cache_id = row.get("cache_id") or Path(row.get("prediction_dir", "")).name
    path = Path(formal_root) / cache_id
    required = (path / "predictions.npz", path / "metadata.json")
    if not cache_id or any(not item.is_file() for item in required):
        raise ValueError(f"{method.upper()}_SOURCE_MISSING:{window_id}:{path}")
    metadata = json.loads((path / "metadata.json").read_text())
    if (metadata.get("dataset") != dataset or metadata.get("method") != method
            or metadata.get("frame_ids") != ids):
        raise ValueError(f"{method.upper()}_IDENTITY_MISMATCH:{window_id}")
    return path


def validate_native_world_geometry(path, window_id, method):
    expected = {
        "camera_c2w": (60, 4, 4), "camera_valid": (60,),
        "hand_joints_world": (60, 2, 21, 3),
        "hand_vertices_world": (60, 2, 778, 3), "hand_valid": (60, 2),
    }
    with np.load(path / "predictions.npz", allow_pickle=False) as source:
        shapes = {key: source[key].shape for key in expected if key in source.files}
        if shapes != expected:
            raise ValueError(f"{method.upper()}_GEOMETRY_SHAPE:{window_id}:{shapes}")
        camera_valid = source["camera_valid"].astype(bool)
        hand_valid = source["hand_valid"].astype(bool)
        if not np.isfinite(source["camera_c2w"][camera_valid]).all():
            raise ValueError(f"{method.upper()}_CAMERA_INVALID:{window_id}")
        for key in ("hand_joints_world", "hand_vertices_world"):
            if not np.isfinite(source[key][hand_valid]).all():
                raise ValueError(f"{method.upper()}_HAND_INVALID:{window_id}:{key}")


def pad_prediction(progress_path, roots, window_id, dataset, ids, index=None, path=None, rewrites=()):
    if path is None:
        if index is None:
            index = {}
            for line in Path(progress_path).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("window_id")
                if not key or key in index:
                    raise ValueError(f"PAD_PROGRESS_WINDOW_ID_INVALID:{key}")
                index[key] = row
        row = index.get(window_id)
        if row is None:
            raise ValueError(f"PAD_PROGRESS_WINDOW_MISSING:{window_id}")
        path = Path(row["prediction_dir"])
    else:
        path = Path(path)
    candidates = [path]
    for rewrite in rewrites:
        source = Path(rewrite["source"])
        if path.is_relative_to(source):
            candidates.append(Path(rewrite["target"]) / path.relative_to(source))
    found = [candidate for candidate in candidates
             if any(candidate.is_relative_to(Path(root)) for root in roots)
             and (candidate / "predictions.npz").is_file()
             and (candidate / "metadata.json").is_file()]
    if len(found) != 1:
        raise ValueError(f"PAD_SOURCE_MATCH_COUNT:{window_id}:{len(found)}:{path}")
    path = found[0]
    if not any(path.is_relative_to(Path(root)) for root in roots):
        raise ValueError(f"PAD_PATH_OUTSIDE_REGISTERED_ROOT:{path}")
    if not (path / "predictions.npz").is_file() or not (path / "metadata.json").is_file():
        raise ValueError(f"PAD_SOURCE_MISSING:{window_id}:{path}")
    metadata = json.loads((path / "metadata.json").read_text())
    if (metadata.get("dataset") != dataset or metadata.get("method") != "pad_hand"
            or metadata.get("frame_ids") != ids):
        raise ValueError(f"PAD_IDENTITY_MISMATCH:{window_id}")
    return path


def has_pad_mesh(path):
    with zipfile.ZipFile(path) as source:
        return "hand_vertices_camera.npy" in source.namelist()


def link(source, target, evidence):
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"SOURCE_MISSING:{source}")
    target.symlink_to(source)
    evidence.append({"name": target.name, "source_path": str(source), "source_bytes": source.stat().st_size})


def rgb_record(row, by_id):
    matches = by_id.get(row["window_id"]) or by_id.get(row["gt"]["cache_id"])
    if not matches or len(matches) != 1:
        raise ValueError(f"RGB_INDEX_MATCH_COUNT:{row['window_id']}:{len(matches or [])}")
    path = matches[0].get("window_input", matches[0].get("window_input_path", matches[0].get("input_path", matches[0].get("record_path"))))
    record = json.loads(Path(path).read_text())
    if record.get("dataset") != row["gt"]["dataset"] or record.get("frame_ids") != row["gt"]["frame_ids"] or len(record["rgb_paths"]) != 60:
        raise ValueError(f"RGB_IDENTITY_MISMATCH:{row['window_id']}")
    return record


def source_paths(window, spec, by_id=None, record=None, pad_index=None, pad_path=None,
                 hawor_index=None, hawor_path=None):
    dataset = window["gt"]["dataset"]
    ids = window["gt"]["frame_ids"]
    cache_id = window["gt"]["cache_id"]
    if record is None:
        record = rgb_record(window, by_id)
    elif (record.get("dataset") != dataset or record.get("frame_ids") != ids
          or len(record.get("rgb_paths", [])) != 60):
        raise ValueError(f"RGB_IDENTITY_MISMATCH:{window['window_id']}")
    frozen_ego = Path(window["pred"]["prediction_dir"])
    if not frozen_ego.is_relative_to(Path(spec[dataset]["ego_output_root"])) or frozen_ego.name != cache_id:
        raise ValueError(f"EGO_PATH_OUTSIDE_REGISTERED_ROOT:{frozen_ego}")
    ego = resolve_registered_path(frozen_ego, spec)
    metadata = json.loads((ego / "metadata.json").read_text())
    if (metadata.get("dataset") != dataset or metadata.get("method") != "egofound3r"
            or metadata.get("global_stride") != 5 or digest_ids(metadata["frame_ids"]) != digest_ids(ids)):
        raise ValueError(f"EGO_IDENTITY_MISMATCH:{window['window_id']}")
    gt = Path(window["gt_path"])
    if not gt.is_relative_to(Path(spec[dataset]["gt_root"])) or gt.stem != cache_id:
        raise ValueError(f"GT_PATH_OUTSIDE_REGISTERED_ROOT:{gt}")
    requested = ["wilor", "reviv4d"]
    if "egoforce" in spec[dataset]["method_roots"]:
        requested.append("egoforce")
    methods = {method: one_prediction(spec[dataset]["method_roots"][method], cache_id, method, dataset, ids)
               for method in requested}
    if "hawor_prediction_index" in spec[dataset]:
        methods["hawor"] = hawor_native_prediction(
            spec[dataset]["hawor_prediction_index"], spec[dataset]["hawor_output_root"],
            window["window_id"], dataset, ids, index=hawor_index, path=hawor_path)
    else:
        methods["hawor"] = one_prediction(
            spec[dataset]["method_roots"]["hawor"], cache_id, "hawor", dataset, ids)
    methods["pad_hand"] = pad_prediction(
        spec[dataset]["pad_progress"], spec[dataset]["method_roots"]["pad_hand"],
        window["window_id"], dataset, ids, index=pad_index, path=pad_path,
        rewrites=spec[dataset].get("pad_path_rewrites", ()))
    return record, ego, gt, methods


def preflight(rows, spec):
    indices = {dataset: index_rows(spec[dataset]["rgb_indices"]) for dataset in {row["dataset"] for row in rows}}
    pad_indices = {}
    hawor_indices = {}
    dyn_indices = {}
    for dataset in indices:
        pad_indices[dataset] = prediction_index(spec[dataset]["pad_progress"], "PAD_PROGRESS")
        if "hawor_prediction_index" in spec[dataset]:
            hawor_indices[dataset] = prediction_index(
                spec[dataset]["hawor_prediction_index"], "HAWOR_NATIVE_INDEX")
        if "dyn_hamr_prediction_index" in spec[dataset]:
            dyn_indices[dataset] = prediction_index(
                spec[dataset]["dyn_hamr_prediction_index"], "DYN_HAMR_INDEX")
    windows = 0
    records = {}
    missing_side_frames = {dataset: [0, 0] for dataset in indices}
    segments_with_fill = []
    dyn_overlap_windows = 0
    for dataset in indices:
        if not Path(spec[dataset]["pad_lane_complete"]).is_file():
            raise ValueError(f"PAD_BIMANUAL_LANE_INCOMPLETE:{dataset}:{spec[dataset]['pad_lane_complete']}")
        if ("hawor_complete" in spec[dataset]
                and not Path(spec[dataset]["hawor_complete"]).is_file()):
            raise ValueError(f"HAWOR_NATIVE_INCOMPLETE:{dataset}:{spec[dataset]['hawor_complete']}")
    for row in rows:
        segment_records = []
        trusted_parts = []
        for window in row["windows"]:
            record, ego, _, methods = source_paths(
                window, spec, indices[row["dataset"]], pad_index=pad_indices[row["dataset"]],
                hawor_index=hawor_indices.get(row["dataset"]))
            if not has_pad_mesh(methods["pad_hand"] / "predictions.npz"):
                raise ValueError(f"PAD_HAND_VERTEX778_MISSING:{window['window_id']}")
            if row["dataset"] in hawor_indices:
                validate_hawor_native_geometry(methods["hawor"], window["window_id"])
            dyn_path = None
            if row["dataset"] in dyn_indices:
                dyn_path = indexed_native_prediction(
                    dyn_indices[row["dataset"]], spec[row["dataset"]]["dyn_hamr_formal_root"],
                    window["window_id"], "dyn_hamr", row["dataset"], window["gt"]["frame_ids"])
                if dyn_path is not None:
                    validate_native_world_geometry(dyn_path, window["window_id"], "dyn_hamr")
                    dyn_overlap_windows += 1
            with np.load(ego / "predictions.npz", allow_pickle=False) as source:
                hand_valid = source["hand_valid"].astype(bool)
                joints = source["hand_joints_camera"]
                markers = source["hand_markers_camera"]
            if hand_valid.shape != (60, 2) or joints.shape != (60, 2, 21, 3) or markers.shape != (60, 2, 195, 3):
                raise ValueError(f"EGO_GEOMETRY_SHAPE:{window['window_id']}")
            trusted_parts.append(hand_valid & np.isfinite(joints).all(axis=(2, 3)) & np.isfinite(markers).all(axis=(2, 3)))
            segment_records.append({
                "rgb": record,
                "pad_prediction_dir": str(methods["pad_hand"]),
                "hawor_prediction_dir": str(methods["hawor"]),
                "dyn_hamr_prediction_dir": None if dyn_path is None else str(dyn_path),
            })
            windows += 1
        trusted = np.concatenate(trusted_parts)
        if not trusted[[0, -1]].all():
            raise ValueError(f"EGO_ENDPOINT_NOT_TRUSTED:{row['segment_id']}")
        missing = (~trusted).sum(0).astype(int)
        missing_side_frames[row["dataset"]][0] += int(missing[0])
        missing_side_frames[row["dataset"]][1] += int(missing[1])
        if missing.any():
            segments_with_fill.append({"segment_id": row["segment_id"], "missing_left_right": missing.tolist()})
        records[row["segment_id"]] = segment_records
    return records, {"segments": len(rows), "windows": windows, "status": "complete",
                     "hawor_native_windows_verified": windows if hawor_indices else 0,
                     "dyn_hamr_overlap_windows_verified": dyn_overlap_windows,
                     "hawor_label": spec.get("hawor_visualization_label", "HaWoR"),
                     "ego_missing_side_frames_by_dataset": missing_side_frames,
                     "ego_segments_with_fill": segments_with_fill,
                     "ego_segments_with_fill_count": len(segments_with_fill)}


def wait_for_pad_lanes(rows, spec, timeout_seconds):
    sentinels = sorted({spec[row["dataset"]]["pad_lane_complete"] for row in rows})
    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [path for path in sentinels if not Path(path).is_file()]
        if not missing:
            print(json.dumps({"status": "pad_bimanual_ready", "sentinels": sentinels}), flush=True)
            return
        remaining = deadline - time.monotonic()
        if timeout_seconds <= 0 or remaining <= 0:
            raise ValueError("PAD_BIMANUAL_WAIT_TIMEOUT:" + ":".join(missing))
        print(json.dumps({"status": "waiting_for_pad_bimanual", "missing": missing,
                          "remaining_seconds": int(remaining)}), flush=True)
        time.sleep(min(60, remaining))


def prepare_inputs(row, spec, root, records, rebuild=False):
    dataset = row["dataset"]
    target = root / dataset / "segments" / row["segment_id"] / "inputs"
    if target.exists():
        if not rebuild:
            raise FileExistsError(f"INPUTS_ALREADY_EXIST:{target}")
        os.replace(target, target.with_name(f"inputs.before_resume.{time.time_ns()}"))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / "inputs.incoming"
    staging.mkdir()
    evidence = []
    try:
        for index, window in enumerate(row["windows"]):
            record, ego, gt, methods = source_paths(
                window, spec, record=records[index]["rgb"],
                pad_path=records[index]["pad_prediction_dir"],
                hawor_path=records[index]["hawor_prediction_dir"])
            source_npz = ego / "predictions.npz"
            with zipfile.ZipFile(source_npz) as source, zipfile.ZipFile(staging / f"{index}_ego.npz", "w", zipfile.ZIP_DEFLATED) as output:
                for key in ("hand_valid", "hand_joints_camera", "hand_markers_camera", "camera_c2w", "camera_valid"):
                    output.writestr(key + ".npy", source.read(key + ".npy"))
            evidence.append({"name": f"{index}_ego.npz", "source_path": str(source_npz), "source_bytes": source_npz.stat().st_size})
            link(ego / "metadata.json", staging / f"{index}_ego_metadata.json", evidence)
            link(gt, staging / f"{index}_gt.npz", evidence)
            for method, directory in methods.items():
                link(directory / "predictions.npz", staging / f"{index}_{method}.npz", evidence)
                link(directory / "metadata.json", staging / f"{index}_{method}_metadata.json", evidence)
            dyn_path = records[index].get("dyn_hamr_prediction_dir")
            if dyn_path:
                directory = Path(dyn_path)
                link(directory / "predictions.npz", staging / f"{index}_dyn_hamr.npz", evidence)
                link(directory / "metadata.json", staging / f"{index}_dyn_hamr_metadata.json", evidence)
            for frame in (0, 149, 299):
                if frame // 60 == index:
                    source = Path(record["rgb_paths"][frame % 60])
                    if record["frame_ids"][frame % 60] != row["frame_ids"][frame]:
                        raise ValueError(f"RGB_SAMPLE_ID_MISMATCH:{frame}")
                    link(source, staging / f"rgb_{frame:03d}{source.suffix.lower()}", evidence)
        (staging / "selection.json").write_text(json.dumps(row) + "\n")
        (staging / "sources.json").write_text(json.dumps(evidence, indent=2) + "\n")
        os.replace(staging, target)
        return target
    except Exception:
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
        raise


def render_one(row, spec, records, runtime, output, workspace, rebuild=False):
    prepare_inputs(row, spec, workspace, records, rebuild=rebuild)
    renderer_path = runtime / RELATIVE / spec.get("renderer", "render_full.py")
    module_spec = importlib.util.spec_from_file_location("render_10s_endpoint_gallery", renderer_path)
    renderer = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = renderer
    module_spec.loader.exec_module(renderer)
    renderer.main(row, output, workspace)
    return row["segment_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--segment")
    parser.add_argument("--wait-pad-seconds", type=int, default=0)
    parser.add_argument("--source-alignment", default="source_alignment_5001.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16")
    root = args.runtime / RELATIVE
    manifest_gz = root / "selected_manifest_hydrated.jsonl.gz"
    manifest_bytes = gzip.decompress(manifest_gz.read_bytes())
    rows = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    spec = json.loads((root / args.source_alignment).read_text())
    if len(rows) != 104 or len({row["segment_id"] for row in rows}) != 104:
        raise ValueError("FROZEN_MANIFEST_NOT_104_UNIQUE_SEGMENTS")
    if hashlib.sha256(manifest_bytes).hexdigest() != spec["manifest_sha256"]:
        raise ValueError("FROZEN_MANIFEST_IDENTITY_MISMATCH")
    if args.segment:
        rows = [row for row in rows if row["segment_id"] == args.segment]
        if len(rows) != 1:
            raise ValueError("PILOT_SEGMENT_NOT_UNIQUE")
    args.output.mkdir(parents=True, exist_ok=True)
    wanted = {row["gallery_stem"]: row for row in rows}
    png = {path.stem: path for path in (args.output / "png_gallery").glob("*.png")}
    mp4 = {path.stem: path for path in (args.output / "video_gallery").glob("*.mp4")
           if ".partial." not in path.name}
    partial = list((args.output / "video_gallery").glob("*partial*"))
    if not args.resume and ((args.output / "summary.json").exists() or (args.output / "COMPLETE").exists()
                            or png or mp4):
        raise FileExistsError(f"OUTPUT_ALREADY_CONTAINS_RESULTS:{args.output}")
    if args.resume:
        if (args.output / "COMPLETE").exists():
            raise FileExistsError(f"OUTPUT_ALREADY_COMPLETE:{args.output}")
        if partial or set(png) != set(mp4) or not set(png) <= set(wanted):
            raise ValueError("RESUME_OUTPUT_PAIRING_INVALID")
    existing = set(png) if args.resume else set()
    pending_rows = [row for row in rows if row["gallery_stem"] not in existing]
    wait_for_pad_lanes(pending_rows, spec, args.wait_pad_seconds)
    workspace = args.output / "input_workspace"
    records, audit = preflight(pending_rows if args.resume else rows, spec)
    audit["existing_verified_pairs"] = len(existing)
    audit["remaining_segments"] = len(pending_rows)
    audit["target_segments"] = len(rows)
    (args.output / "preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
    completed, failed = [], []
    if args.workers == 1:
        future_rows = [(row, None) for row in pending_rows]
    else:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, max(1, len(pending_rows))))
        future_rows = [(row, pool.submit(render_one, row, spec, records[row["segment_id"]],
                                         args.runtime, args.output, workspace, args.resume))
                       for row in pending_rows]
    try:
        for row, future in future_rows:
            try:
                result = (render_one(row, spec, records[row["segment_id"]], args.runtime,
                                     args.output, workspace, args.resume)
                          if future is None else future.result())
                completed.append(result)
                print(json.dumps({"segment_id": row["segment_id"], "status": "complete",
                                  "completed": len(existing) + len(completed), "target": len(rows)}), flush=True)
            except Exception as error:
                failed.append({"segment_id": row["segment_id"], "error": str(error), "traceback": traceback.format_exc()[-4000:]})
                print(json.dumps({"segment_id": row["segment_id"], "status": "failed", "error": str(error)}), flush=True)
            summary = {"status": "running", "completed": len(existing) + len(completed),
                       "target": len(rows), "failures": failed}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        if args.workers != 1:
            pool.shutdown()
    summary = {
        "status": "complete" if not failed and len(existing) + len(completed) == len(rows) else "incomplete",
        "completed": len(existing) + len(completed), "target": len(rows), "failures": failed,
        "manifest_sha256": spec["manifest_sha256"], "workers": args.workers,
        "png_gallery": str(args.output / "png_gallery"), "video_gallery": str(args.output / "video_gallery"),
        "ego_interpolation": spec["ego_interpolation"], "visualization_mask": spec["visualization_mask"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "complete":
        raise SystemExit(1)
    (args.output / "COMPLETE").write_text("104 endpoint-renderable PNG/MP4 pairs complete\n")


if __name__ == "__main__":
    main()
