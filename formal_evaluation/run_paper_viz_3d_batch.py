#!/usr/bin/env python3
"""Stage, preflight, and render the frozen 104-segment paper 3D gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "e88ac84845a5a61e6e472b12d160816ef0d8a48eaeb0e2085985335ff33c92a5"
EXPECTED_COUNTS = {"arctic": 48, "h2o": 5, "hot3d": 44, "oakink_v2": 7}


def segment_id(entry: dict) -> str:
    return str(entry["gallery_stem"])


def cache_ids(entry: dict) -> list[str]:
    return list(dict.fromkeys(window["gt"]["cache_id"] for window in entry["windows"]))


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ego-infer-root", type=Path, required=True)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--hawor-root", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.worktree / "visualization/paper_viz"))
    from paper_viz.inputs import resolve_prediction_file

    relative = Path("visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914")
    manifest = args.worktree / relative / "selected_manifest_hydrated.jsonl"
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("MANIFEST_SHA256_MISMATCH")
    entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(entries) != 104 or Counter(row["dataset"] for row in entries) != EXPECTED_COUNTS:
        raise RuntimeError("MANIFEST_COUNT_MISMATCH")
    if args.output_root.exists() or not args.smoke_root.is_dir():
        raise RuntimeError("OUTPUT_EXISTS_OR_SMOKE_MISSING")
    smoke_bytes = tree_bytes(args.smoke_root)
    required_free = smoke_bytes * len(entries) + 50 * 1024**3
    free = shutil.disk_usage(args.output_root.parent).free
    if free < required_free:
        raise RuntimeError(f"INSUFFICIENT_OUTPUT_SPACE:{free}:{required_free}")

    alignment = json.loads((args.worktree / relative / "source_alignment_auxmethods_full104_5001.json").read_text())
    prepared_roots = {
        dataset: Path(alignment[dataset]["rgb_indices"][0]).parent / dataset
        for dataset in EXPECTED_COUNTS
    }
    hawor_roots = {
        dataset: Path(alignment[dataset]["hawor_output_root"])
        for dataset in EXPECTED_COUNTS
    }
    hawor_indexes = {}
    missing = []
    for dataset in EXPECTED_COUNTS:
        index = Path(alignment[dataset]["hawor_prediction_index"])
        hawor_indexes[dataset] = {
            row["window_id"]: Path(row["prediction_dir"])
            for row in (json.loads(line) for line in index.read_text().splitlines() if line.strip())
        }
    for entry in entries:
        dataset, name = entry["dataset"], segment_id(entry)
        for window in entry["windows"]:
            cache = window["gt"]["cache_id"]
            window_id = window["window_id"]
            prepared = prepared_roots[dataset] / cache
            required = {
                "prepared": prepared / "window_input.json",
                "ego": args.src_dir / cache / "ego.npz",
                "gt": args.src_dir / cache / "gt.npz",
                "wilor": args.src_dir / cache / "wilor.npz",
                "pad_hand": args.src_dir / cache / "pad_hand.npz",
                "egoforce": args.src_dir / cache / "egoforce.npz",
                "reviv4d": args.src_dir / cache / "reviv4d.npz",
                "hawor": resolve_prediction_file(
                    hawor_indexes[dataset].get(window_id), hawor_roots[dataset], cache,
                    "hawor", required=False),
            }
            missing.extend((kind, str(path)) for kind, path in required.items() if not path.is_file())
    if missing:
        raise RuntimeError("MISSING_BATCH_INPUTS:" + json.dumps({
            "counts": Counter(kind for kind, _ in missing),
            "samples": [{"kind": kind, "path": path} for kind, path in missing[:20]],
        }, sort_keys=True))

    tools = args.worktree / "visualization/paper_viz/tools"
    python = "/mnt/workspace/sjc/envs/egofound3r/bin/python"
    args.output_root.mkdir(parents=True)
    staged_root = args.output_root / "_staged"
    for number, entry in enumerate(entries, 1):
        dataset, name, caches = entry["dataset"], segment_id(entry), cache_ids(entry)
        staged = staged_root / dataset / name
        if not (staged / "selection.json").is_file():
            incoming = staged.with_name(staged.name + ".incoming")
            incoming.mkdir(parents=True)
            windows = []
            for index, (cache, window) in enumerate(zip(caches, entry["windows"])):
                os.symlink(args.src_dir / cache / "ego.npz", incoming / f"{index}_ego.npz")
                windows.append({"index": index, "window_id": window["window_id"],
                                "cache_id": cache, "gt": {"cache_id": cache},
                                "frame_ids": window["gt"]["frame_ids"]})
            selection = {"dataset": dataset, "sequence_id": entry["sequence_id"],
                         "segment_id": name, "windows": windows}
            (incoming / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
            incoming.replace(staged)
        print(f"STAGED {number}/104 {name}", flush=True)

    for dataset in EXPECTED_COUNTS:
        common = ["--staged-root", str(staged_root / dataset), "--src-dir", str(args.src_dir),
                  "--prepared-root", str(prepared_roots[dataset]),
                  "--hawor-root", str(hawor_roots[dataset]),
                  "--hawor-index", str(alignment[dataset]["hawor_prediction_index"]),
                  "--contact-root", str(args.contact_root), "--mapping", str(args.mapping)]
        run([python, str(tools / "check_3d_world_frame.py"), *common, "--device", "cpu"])
        run([python, str(tools / "batch_3d_render.py"), *common,
             "--out-root", str(args.output_root / dataset), "--stages", "summary", "video",
             "--devices", *args.devices, "--chunks", str(len(args.devices)), "--cell", "320",
             "--panel-size", "0", "--keyframes", "5", "--camera-overlay", "show"])

    results = []
    for entry in entries:
        root = args.output_root / entry["dataset"] / segment_id(entry)
        frames = len(list((root / "_frames").glob("*.png")))
        ok = (root / "fig1_3d_summary.png").is_file() and (root / "video1_3d_matrix.mp4").is_file() and frames == 300
        results.append({"dataset": entry["dataset"], "segment": segment_id(entry), "frames": frames, "ok": ok})
    if not all(row["ok"] for row in results):
        raise RuntimeError("BATCH_OUTPUT_INCOMPLETE")
    summary = {"status": "complete", "segments": len(results), "manifest_sha256": EXPECTED_MANIFEST_SHA256,
               "smoke_bytes": smoke_bytes, "required_free": required_free, "results": results}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text("complete\n")
    print(json.dumps({"status": "complete", "segments": len(results)}), flush=True)


if __name__ == "__main__":
    main()
