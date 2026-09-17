"""Assemble verified old pairs and eight frozen continuation shards."""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from merge_10s_gallery_shards import atomic_json, verify_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--plan-relative", default="visualization/batch_10s_177_p95_wmpjpe_20260912/parallel8_handoff_20260913/plan.json")
    args = parser.parse_args()
    relative = Path("visualization/batch_10s_177_p95_wmpjpe_20260912")
    plan_path = args.runtime / args.plan_relative
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    shard_count = len(plan["shards"])
    manifest = gzip.decompress((args.runtime / relative / "selected_manifest.jsonl.gz").read_bytes())
    alignment = json.loads((args.runtime / relative / "source_alignment_5000.json").read_text())
    rows = [json.loads(line) for line in manifest.splitlines() if line]
    if (hashlib.sha256(plan_bytes).hexdigest() != args.plan_sha256
            or plan.get("schema") != f"10s_gallery_parallel{shard_count}_handoff_v1"
            or shard_count not in (8, 16)
            or len(rows) != 177 or hashlib.sha256(manifest).hexdigest() != plan["manifest_sha256"]
            or plan["manifest_sha256"] != alignment["manifest_sha256"]
            or str(args.output) != plan["aggregate_output_root"]):
        raise ValueError("HANDOFF_IDENTITY_MISMATCH")
    if [shard["number"] for shard in plan["shards"]] != list(range(1, shard_count + 1)):
        raise ValueError("HANDOFF_SHARD_IDENTITY_MISMATCH")
    sources = {}
    for entry in plan["completed"]:
        index = entry["index"]
        if rows[index]["gallery_stem"] != entry["gallery_stem"] or index in sources:
            raise ValueError("COMPLETED_SOURCE_PLAN_MISMATCH")
        sources[index] = Path(entry["source_root"])
    for shard in plan["shards"]:
        for index in shard["indices"]:
            if index in sources:
                raise ValueError("SHARD_SOURCE_PLAN_OVERLAP")
            sources[index] = Path(shard["output_root"])
    if len(sources) != 177 or set(sources) != set(range(177)):
        raise ValueError("HANDOFF_PARTITION_MISMATCH")
    args.output.mkdir(parents=True, exist_ok=True)
    for gallery in ("png_gallery", "video_gallery"):
        (args.output / gallery).mkdir(exist_ok=True)
    completed = set()
    while len(completed) < 177:
        for shard in plan["shards"]:
            source = Path(shard["output_root"])
            summary = source / "summary.json"
            if summary.is_file() and json.loads(summary.read_text()).get("status") == "incomplete":
                raise RuntimeError(f"SOURCE_SHARD_INCOMPLETE:{source}")
        for index, row in enumerate(rows):
            if index in completed:
                continue
            source = sources[index]
            stem = row["gallery_stem"]
            png = source / "png_gallery" / f"{stem}.png"
            mp4 = source / "video_gallery" / f"{stem}.mp4"
            if not (png.is_file() and mp4.is_file() and png.stat().st_size and mp4.stat().st_size):
                continue
            verify_pair(png, mp4)
            for file, gallery in ((png, "png_gallery"), (mp4, "video_gallery")):
                link = args.output / gallery / file.name
                if link.is_symlink() and link.resolve() == file.resolve():
                    continue
                if link.exists() or link.is_symlink():
                    raise FileExistsError(f"AGGREGATE_COLLISION:{link}")
                link.symlink_to(file)
            completed.add(index)
            print(json.dumps({"paired": len(completed), "target": 177, "stem": stem}), flush=True)
        atomic_json(args.output / "progress.json", {"paired": len(completed), "target": 177})
        if len(completed) < 177:
            time.sleep(30)
    atomic_json(args.output / "summary.json", {
        "status": "complete", "paired_verified": 177, "target": 177,
        "manifest_sha256": plan["manifest_sha256"], "plan_sha256": args.plan_sha256,
        "old_completed_count": plan["completed_count"], "new_shard_count": len(plan["shards"]),
    })
    (args.output / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
