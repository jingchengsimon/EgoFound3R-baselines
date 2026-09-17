"""Batch-run the locked 2D visualization over a frozen selection manifest.

One manifest entry = one 300-frame (5 x 60-frame window) segment.  For every
segment the runner

1. locates the authoritative Ego inference output
   (``<ego-infer-root>/<dataset>/<segment_id>/``, see USAGE_2D.md step 2),
2. stages the per-window Ego inputs (``stage_ego_windows.py``),
3. renders the figure, the 15 panels and the video with the locked style,
4. verifies ``report.json`` and records the outcome.

Segments run in parallel (``--parallel``) and each segment renders its video
frames with ``--jobs`` workers, so a node with N cores can be saturated without
touching the style.  Finished segments are skipped unless ``--force`` is given.

Example
-------
``python3 tools/batch_2d_render.py \
    --manifest ../overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/selected_manifest.jsonl \
    --ego-infer-root /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_infer_8fc061a_batch \
    --src-dir /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917 \
    --out-root /mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_2d_batch \
    --parallel 6 --jobs 8``
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent

DEFAULT_PREPARED_ROOT = Path("/mnt/workspace/sjc/eval_artifacts/"
                             "prep_6dataset_60f_strict_arctic_20260820T000131Z/window_inputs")
DEFAULT_HAWOR_ROOT = Path("/mnt/workspace/sjc/DATA/eval_artifacts/"
                          "hawor_native_camera_repair_20260915_v7_5000/datasets")
DEFAULT_CONTACT_ROOT = Path("/mnt/cpfs/sjc/eval_artifacts/result3_completion_20260910_v1")
DEFAULT_MAPPING = Path("/mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917/"
                       "egohandmetric_prompt/data/mano_upsampling/mano_195_to_778.npz")


def cache_ids_of(entry: dict) -> list[str]:
    ids: list[str] = []
    for ref in entry.get("frame_refs") or []:
        cache = ref.get("cache_id")
        if cache and cache not in ids:
            ids.append(cache)
    return ids


def segment_id_of(entry: dict) -> str:
    frames = entry["frame_ids"]
    return f"{entry['dataset']}__{frames[0]}-{frames[-1]}"


def run(command: list[str], log) -> None:
    log.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
    log.flush()
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def render_segment(entry: dict, args) -> dict:
    dataset, segment = entry["dataset"], segment_id_of(entry)
    caches = cache_ids_of(entry)
    frames = len(entry["frame_ids"])
    result = {"dataset": dataset, "segment_id": segment, "caches": caches,
              "frames": frames, "status": "pending"}
    out_dir = args.out_root / segment
    report_path = out_dir / "report.json"
    if report_path.is_file() and not args.force:
        report = json.loads(report_path.read_text())
        result.update(status="skipped_existing", video2_frames=report.get("video2_frames"))
        return result

    log_path = args.out_root / "_logs" / f"{segment}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    infer_dir = args.ego_infer_root / dataset / segment
    npz_candidates = sorted((infer_dir / "viz_inputs").glob("*_plus.npz")) or \
        sorted((infer_dir / "viz_inputs").glob("ego_infer_*f.npz"))
    if not (infer_dir / "inference_output.pt").is_file() or not npz_candidates:
        result.update(status="missing_ego_infer", infer_dir=str(infer_dir))
        return result
    npz = npz_candidates[0]

    staged = args.staged_root / segment
    with log_path.open("w") as log:
        try:
            if not (staged / "selection.json").is_file() or args.force:
                run([sys.executable, str(HERE / "stage_ego_windows.py"),
                     "--infer-dir", str(infer_dir), "--npz", str(npz),
                     "--prepared-root", str(args.prepared_root / dataset),
                     "--dataset", dataset, "--caches", *caches,
                     "--out", str(staged)], log)
            command = [sys.executable, "-m", "paper_viz.cli",
                       "--stages", "fig2", "--rows", str(args.rows),
                       "--cell2d", str(args.cell2d), "--cell2d-video", str(args.cell2d_video),
                       "--fps", str(args.fps), "--jobs", str(args.jobs),
                       "--inputs-dir", str(staged), "--src-dir", str(args.src_dir),
                       "--prepared-root", str(args.prepared_root / dataset),
                       "--hawor-root", str(args.hawor_root / dataset),
                       "--hawor-index", str(args.hawor_root / dataset / "predictions.jsonl"),
                       "--contact-root", str(args.contact_root),
                       "--mapping", str(args.mapping), "--out", str(out_dir)]
            if args.skip_panels:
                command.append("--skip-panels")
            environment = dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT))
            log.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
            log.flush()
            subprocess.run(command, check=True, env=environment, stdout=log,
                           stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            result.update(status="failed", returncode=error.returncode,
                          log=str(log_path))
            return result
    report = json.loads(report_path.read_text())
    result.update(status="rendered", video2_frames=report.get("video2_frames"),
                  log=str(log_path))
    if report.get("video2_frames") != frames // max(args.video_stride, 1):
        result["status"] = "rendered_frame_count_mismatch"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ego-infer-root", type=Path, required=True,
                        help="<root>/<dataset>/<segment_id>/inference_output.pt (+ viz_inputs/)")
    parser.add_argument("--src-dir", type=Path, required=True,
                        help="relayed per-window npz (tools/relay_sources.py)")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, default=None,
                        help="staged per-window Ego inputs (default: <out-root>/_staged)")
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--hawor-root", type=Path, default=DEFAULT_HAWOR_ROOT)
    parser.add_argument("--contact-root", type=Path, default=DEFAULT_CONTACT_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cell2d", type=int, default=720)
    parser.add_argument("--cell2d-video", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=8,
                        help="render workers per segment (video rows are independent)")
    parser.add_argument("--parallel", type=int, default=1, help="segments rendered at once")
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--skip-panels", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None,
                        help="segment ids or cache ids to render")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.staged_root = args.staged_root or (args.out_root / "_staged")
    args.out_root.mkdir(parents=True, exist_ok=True)

    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.datasets:
        entries = [e for e in entries if e["dataset"] in args.datasets]
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries
                   if segment_id_of(e) in wanted or wanted & set(cache_ids_of(e))]
    if args.limit:
        entries = entries[:args.limit]
    print(f"{len(entries)} segments selected; parallel={args.parallel} jobs/segment={args.jobs}")
    if args.dry_run:
        for entry in entries:
            print(" ", segment_id_of(entry), cache_ids_of(entry))
        return

    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max(args.parallel, 1)) as pool:
        for result in pool.map(lambda entry: render_segment(entry, args), entries):
            results.append(result)
            print(f"[{len(results):3d}/{len(entries)}] {result['segment_id']}: {result['status']}"
                  + (f" (video2_frames={result.get('video2_frames')})" if result.get("video2_frames") else ""),
                  flush=True)
    summary = {"manifest": str(args.manifest), "segments": len(entries),
               "seconds": round(time.time() - started, 1),
               "status_counts": {}, "results": results}
    for result in results:
        summary["status_counts"][result["status"]] = summary["status_counts"].get(result["status"], 0) + 1
    (args.out_root / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["status_counts"], indent=2))
    print(f"wall {summary['seconds']}s -> {args.out_root/'batch_summary.json'}")


if __name__ == "__main__":
    main()
