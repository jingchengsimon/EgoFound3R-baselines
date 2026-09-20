#!/usr/bin/env python3
"""Assemble verified 104-pair PNG/MP4 galleries using ranked filename prefixes."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path


EXPECTED_COUNTS = {"arctic": 48, "h2o": 5, "hot3d": 44, "oakink_v2": 7}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--ranking-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    for root in (args.render_root, args.ranking_root):
        if not (root / "COMPLETE").is_file():
            raise RuntimeError(f"SOURCE_NOT_COMPLETE:{root}")
    rows = [json.loads(line) for line in (args.ranking_root / "ranking_104.jsonl").read_text().splitlines() if line]
    if len(rows) != 104 or [row["rank"] for row in rows] != list(range(1, 105)):
        raise RuntimeError("RANKING_CONTRACT_FAILED")
    if Counter(row["dataset"] for row in rows) != EXPECTED_COUNTS:
        raise RuntimeError("RANKING_DATASET_COUNTS_FAILED")

    png_root = args.output_root / "png_gallery"
    mp4_root = args.output_root / "mp4_gallery"
    report_root = args.output_root / "reports"
    for root in (png_root, mp4_root, report_root):
        root.mkdir(parents=True)
    files = []
    for row in rows:
        source = args.render_root / row["dataset"] / row["segment_id"]
        png_source = source / "fig1_3d_summary.png"
        mp4_source = source / "video1_3d_matrix.mp4"
        if not png_source.is_file() or not mp4_source.is_file():
            raise FileNotFoundError(source)
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate", "-of", "json", str(mp4_source),
        ], check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        if int(stream["nb_frames"]) != 300 or stream["r_frame_rate"] != "30/1":
            raise RuntimeError(f"VIDEO_CONTRACT_FAILED:{mp4_source}:{stream}")
        png_target = png_root / f"{row['ranked_stem']}.png"
        mp4_target = mp4_root / f"{row['ranked_stem']}.mp4"
        shutil.copy2(png_source, png_target)
        shutil.copy2(mp4_source, mp4_target)
        report = {"rank": row["rank"], "ranked_stem": row["ranked_stem"],
                  "dataset": row["dataset"], "segment_id": row["segment_id"],
                  "beat_count": row["beat_count"], "comparable_count": row["comparable_count"],
                  "dyn_hamr": row["methods"]["dyn_hamr"], "hawor": row["methods"]["hawor"],
                  "png": png_target.name, "mp4": mp4_target.name}
        (report_root / f"{row['ranked_stem']}.json").write_text(json.dumps(report, indent=2) + "\n")
        files.append({"rank": row["rank"], "dataset": row["dataset"],
                      "segment_id": row["segment_id"], "ranked_stem": row["ranked_stem"]})

    for name in ("ranking_104.jsonl", "ranking_104.csv"):
        shutil.copy2(args.ranking_root / name, args.output_root / name)
    summary = {
        "status": "complete", "segments": 104, "png_count": 104, "mp4_count": 104,
        "report_count": 104, "dataset_counts": EXPECTED_COUNTS,
        "render_root": str(args.render_root), "ranking_root": str(args.ranking_root),
        "ranking_summary": json.loads((args.ranking_root / "summary.json").read_text()),
        "files": files,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "preflight.json").write_text(json.dumps({
        "status": "passed", "render_complete": True, "ranking_complete": True,
        "segments": 104}, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text("complete\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    main()
