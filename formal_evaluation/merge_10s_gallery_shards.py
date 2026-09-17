"""Assemble the four disjoint 10-second render ranges without touching their outputs."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".incoming")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def verify_pair(png, mp4):
    with Image.open(png) as image:
        if image.size != (4620, 4240):
            raise ValueError(f"PNG_SIZE:{png}:{image.size}")
        image.verify()
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(mp4)],
        text=True, capture_output=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    if (stream["codec_name"] != "h264" or (stream["width"], stream["height"]) != (1788, 1524)
            or stream["r_frame_rate"] != "30/1" or int(stream["nb_frames"]) != 300
            or abs(float(stream["duration"]) - 10) > 0.02):
        raise ValueError(f"MP4_CONTRACT:{mp4}:{stream}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    args = parser.parse_args()
    if len(args.source) != 4 or len(set(args.source)) != 4 or args.output in args.source:
        parser.error("exactly four distinct source roots are required")
    relative = "visualization/batch_10s_177_p95_wmpjpe_20260912"
    manifest = gzip.decompress((args.runtime / relative / "selected_manifest.jsonl.gz").read_bytes())
    alignment = json.loads((args.runtime / relative / "source_alignment_5000.json").read_text())
    rows = [json.loads(line) for line in manifest.splitlines() if line]
    if len(rows) != 177 or hashlib.sha256(manifest).hexdigest() != alignment["manifest_sha256"]:
        raise ValueError("FROZEN_MANIFEST_IDENTITY_MISMATCH")
    bounds = (0, 45, 89, 133, 177)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "png_gallery").mkdir(exist_ok=True)
    (args.output / "video_gallery").mkdir(exist_ok=True)
    completed = set()
    while len(completed) < len(rows):
        for shard, source in enumerate(args.source):
            shard_rows = rows[bounds[shard]:bounds[shard + 1]]
            summary = source / "summary.json"
            if shard and summary.is_file() and json.loads(summary.read_text()).get("status") == "incomplete":
                raise RuntimeError(f"SOURCE_SHARD_INCOMPLETE:{source}")
            if shard == 0:
                log = source / "logs/worker.log"
                if log.is_file():
                    front_ids = {row["segment_id"] for row in shard_rows}
                    for line in log.read_text(errors="replace").splitlines():
                        if not line.startswith("{"):
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event.get("status") == "failed" and event.get("segment_id") in front_ids:
                            raise RuntimeError(f"ORIGINAL_FRONT_SEGMENT_FAILED:{event['segment_id']}")
            for row in shard_rows:
                stem = row["gallery_stem"]
                if stem in completed:
                    continue
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
                completed.add(stem)
                print(json.dumps({"paired": len(completed), "target": len(rows), "stem": stem}), flush=True)
        atomic_json(args.output / "progress.json", {"paired": len(completed), "target": len(rows)})
        if len(completed) < len(rows):
            time.sleep(30)
    atomic_json(args.output / "summary.json", {
        "status": "complete", "paired_verified": len(completed), "target": len(rows),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "source_roots": [str(source) for source in args.source], "bounds": bounds,
    })
    (args.output / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
