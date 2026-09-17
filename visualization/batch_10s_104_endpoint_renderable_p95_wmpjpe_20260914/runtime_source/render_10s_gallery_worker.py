"""Render the frozen 177-segment gallery from registered 5000 sources."""

import argparse
import gzip
import hashlib
import json
import os
import traceback
import zipfile
from pathlib import Path


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


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


def link(source, target, evidence):
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"SOURCE_MISSING:{source}")
    target.symlink_to(source)
    evidence.append({"name": target.name, "source_path": str(source), "source_bytes": source.stat().st_size})


def prepare_inputs(row, spec, root, by_id):
    ds = row["dataset"]
    segment = row["segment_id"]
    target = root / ds / "segments" / segment / "inputs"
    if target.exists():
        raise FileExistsError(f"INPUTS_ALREADY_EXIST:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / "inputs.incoming"
    staging.mkdir()
    evidence = []
    try:
        for i, win in enumerate(row["windows"]):
            ids = win["gt"]["frame_ids"]
            cache_id = win["gt"]["cache_id"]
            matches = by_id.get(win["window_id"]) or by_id.get(cache_id)
            if not matches or len(matches) != 1:
                raise ValueError(f"RGB_INDEX_MATCH_COUNT:{win['window_id']}:{len(matches or [])}")
            record_path = matches[0].get("window_input", matches[0].get("window_input_path", matches[0].get("input_path", matches[0].get("record_path"))))
            record = json.loads(Path(record_path).read_text())
            if record.get("dataset") != ds or record.get("frame_ids") != ids or len(record["rgb_paths"]) != 60:
                raise ValueError(f"RGB_IDENTITY_MISMATCH:{win['window_id']}")
            ego = Path(win["pred"]["prediction_dir"])
            ego_root = Path(spec[ds]["ego_output_root"])
            if not ego.is_relative_to(ego_root) or ego.name != cache_id:
                raise ValueError(f"EGO_PATH_OUTSIDE_REGISTERED_ROOT:{ego}")
            ego_meta_path = ego / "metadata.json"
            ego_meta = json.loads(ego_meta_path.read_text())
            if (ego_meta.get("dataset") != ds or ego_meta.get("method") != "egofound3r"
                    or ego_meta.get("global_stride") != 5 or digest_ids(ego_meta["frame_ids"]) != digest_ids(ids)):
                raise ValueError(f"EGO_IDENTITY_MISMATCH:{win['window_id']}")
            source_npz = ego / "predictions.npz"
            with zipfile.ZipFile(source_npz) as source, zipfile.ZipFile(staging / f"{i}_ego.npz", "w", zipfile.ZIP_DEFLATED) as output:
                for key in ("hand_valid", "hand_joints_camera", "hand_markers_camera", "camera_c2w", "camera_valid"):
                    output.writestr(key + ".npy", source.read(key + ".npy"))
            evidence.append({"name": f"{i}_ego.npz", "source_path": str(source_npz), "source_bytes": source_npz.stat().st_size})
            link(ego_meta_path, staging / f"{i}_ego_metadata.json", evidence)
            gt_path = Path(win["gt_path"])
            if not gt_path.is_relative_to(Path(spec[ds]["gt_root"])) or gt_path.stem != cache_id:
                raise ValueError(f"GT_PATH_OUTSIDE_REGISTERED_ROOT:{gt_path}")
            link(gt_path, staging / f"{i}_gt.npz", evidence)
            for method in ("wilor", "hawor", "pad_hand", "reviv4d"):
                directory = one_prediction(spec[ds]["method_roots"][method], cache_id, method, ds, ids)
                link(directory / "predictions.npz", staging / f"{i}_{method}.npz", evidence)
                link(directory / "metadata.json", staging / f"{i}_{method}_metadata.json", evidence)
            for t in (0, 149, 299):
                if t // 60 == i:
                    source = Path(record["rgb_paths"][t % 60])
                    if record["frame_ids"][t % 60] != row["frame_ids"][t]:
                        raise ValueError(f"RGB_SAMPLE_ID_MISMATCH:{t}")
                    link(source, staging / f"rgb_{t:03d}{source.suffix.lower()}", evidence)
        (staging / "selection.json").write_text(json.dumps(row) + "\n")
        (staging / "sources.json").write_text(json.dumps(evidence, indent=2) + "\n")
        os.replace(staging, target)
        return target
    except Exception:
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--shard-number", type=int)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--plan-relative", default="visualization/batch_10s_177_p95_wmpjpe_20260912/parallel8_handoff_20260913/plan.json")
    args = parser.parse_args()
    if (args.start_index is None) != (args.end_index is None) or (args.segment and args.start_index is not None):
        parser.error("select either one segment or a complete index range")
    if (args.shard_number is None) != (args.plan_sha256 is None):
        parser.error("handoff shard number and plan hash must be specified together")
    if args.shard_number is not None and (args.segment or args.start_index is not None):
        parser.error("handoff selection cannot be combined with segment or range selection")
    manifest_gz = args.runtime / "visualization/batch_10s_177_p95_wmpjpe_20260912/selected_manifest.jsonl.gz"
    manifest = manifest_gz.with_suffix("")
    spec_path = args.runtime / "visualization/batch_10s_177_p95_wmpjpe_20260912/source_alignment_5000.json"
    manifest_bytes = gzip.decompress(manifest_gz.read_bytes())
    rows = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    spec = json.loads(spec_path.read_text())
    if len(rows) != 177 or hashlib.sha256(manifest_bytes).hexdigest() != spec["manifest_sha256"]:
        raise ValueError("FROZEN_MANIFEST_IDENTITY_MISMATCH")
    manifest.write_bytes(manifest_bytes)
    if args.start_index is not None:
        if not 0 <= args.start_index < args.end_index <= len(rows):
            parser.error("invalid frozen-manifest index range")
        rows = rows[args.start_index:args.end_index]
    if args.shard_number is not None:
        plan_path = args.runtime / args.plan_relative
        plan_bytes = plan_path.read_bytes()
        plan = json.loads(plan_bytes)
        shard_count = len(plan["shards"])
        if (hashlib.sha256(plan_bytes).hexdigest() != args.plan_sha256
                or plan.get("schema") != f"10s_gallery_parallel{shard_count}_handoff_v1"
                or plan.get("manifest_sha256") != spec["manifest_sha256"]):
            raise ValueError("HANDOFF_PLAN_IDENTITY_MISMATCH")
        shards = plan["shards"]
        if shard_count not in (8, 16) or [shard["number"] for shard in shards] != list(range(1, shard_count + 1)):
            raise ValueError("HANDOFF_SHARD_IDENTITY_MISMATCH")
        selected = [index for shard in shards for index in shard["indices"]]
        completed_indices = [entry["index"] for entry in plan["completed"]]
        if (len(selected) != plan["remaining_count"] or len(completed_indices) != plan["completed_count"]
                or sorted(selected + completed_indices) != list(range(177))):
            raise ValueError("HANDOFF_PLAN_PARTITION_MISMATCH")
        if not 1 <= args.shard_number <= shard_count:
            parser.error(f"handoff shard number must be 1–{shard_count}")
        rows = [rows[index] for index in shards[args.shard_number - 1]["indices"]]
    if args.segment:
        rows = [row for row in rows if row["segment_id"] == args.segment]
        if len(rows) != 1:
            raise ValueError("PILOT_SEGMENT_NOT_UNIQUE")
    import importlib.util
    renderer_path = args.runtime / "visualization/batch_10s_177_p95_wmpjpe_20260912/render_full.py"
    module_spec = importlib.util.spec_from_file_location("render_10s_gallery", renderer_path)
    renderer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(renderer)
    # Renderer resolves its frozen manifest from its own deployed runtime.
    completed, failed, indices = 0, [], {}
    for row in rows:
        ds = row["dataset"]
        if ds not in indices:
            indices[ds] = index_rows(spec[ds]["rgb_indices"])
        try:
            prepare_inputs(row, spec, args.runtime / "visualization/batch_10s_177_p95_wmpjpe_20260912", indices[ds])
            renderer.main(row["segment_id"], args.output)
            completed += 1
            print(json.dumps({"segment_id": row["segment_id"], "status": "complete", "completed": completed, "target": len(rows)}), flush=True)
        except Exception as error:
            failed.append({"segment_id": row["segment_id"], "error": str(error), "traceback": traceback.format_exc()[-3000:]})
            print(json.dumps({"segment_id": row["segment_id"], "status": "failed", "error": str(error)}), flush=True)
    summary = {"status": "complete" if not failed and completed == len(rows) else "incomplete",
               "completed": completed, "target": len(rows), "failures": failed,
               "manifest_start_index": args.start_index, "manifest_end_index": args.end_index,
               "handoff_shard_number": args.shard_number, "handoff_plan_sha256": args.plan_sha256,
               "png_gallery": str(args.output / "png_gallery"), "video_gallery": str(args.output / "video_gallery")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "complete":
        (args.output / "COMPLETE").write_text("complete\n")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
