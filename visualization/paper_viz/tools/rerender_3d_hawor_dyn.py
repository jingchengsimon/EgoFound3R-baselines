#!/usr/bin/env python3
"""Rerender the frozen 104-segment gallery with strict HaWoR and Dyn-HaMR reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "e88ac84845a5a61e6e472b12d160816ef0d8a48eaeb0e2085985335ff33c92a5"
EXPECTED_COUNTS = {"arctic": 48, "h2o": 5, "hot3d": 44, "oakink_v2": 7}
EXPECTED_DYN_SEGMENTS = 24
EXPECTED_DYN_WINDOWS = 27


def mappings(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or key in result:
            raise ValueError(f"expected one unique DATASET=PATH mapping, got {value!r}")
        result[key] = Path(raw_path)
    if set(result) != set(EXPECTED_COUNTS):
        raise RuntimeError("DATASET_MAPPING_MISMATCH")
    return result


def tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def run(command: list[str], worktree: Path) -> None:
    print("$ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    additions = [str(worktree), str(worktree / "visualization/paper_viz")]
    environment["PYTHONPATH"] = ":".join(additions + [environment.get("PYTHONPATH", "")])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--worktree-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--staged", action="append", default=[], metavar="DATASET=PATH", required=True)
    parser.add_argument("--prepared", action="append", default=[], metavar="DATASET=PATH", required=True)
    parser.add_argument("--hawor-root", action="append", default=[], metavar="DATASET=PATH", required=True)
    parser.add_argument("--hawor-index", action="append", default=[], metavar="DATASET=PATH", required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    args = parser.parse_args()

    observed_commit = subprocess.run(
        ["git", "-C", str(args.worktree), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if observed_commit != args.worktree_commit:
        raise RuntimeError(f"WORKTREE_COMMIT_MISMATCH:{observed_commit}:{args.worktree_commit}")

    manifest = args.worktree / "visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest_hydrated.jsonl"
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("MANIFEST_SHA256_MISMATCH")
    entries = [json.loads(line) for line in raw.splitlines() if line]
    if len(entries) != 104 or Counter(row["dataset"] for row in entries) != EXPECTED_COUNTS:
        raise RuntimeError("MANIFEST_COUNT_MISMATCH")
    staged, prepared, hawor_roots, hawor_indexes = map(
        mappings, (args.staged, args.prepared, args.hawor_root, args.hawor_index))
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if not args.smoke_root.is_dir():
        raise FileNotFoundError(args.smoke_root)

    import sys
    sys.path[:0] = [str(args.worktree / "visualization/paper_viz"), str(args.worktree)]
    from paper_viz.inputs import resolve_prediction_file

    hawor_maps = {}
    for dataset, index in hawor_indexes.items():
        hawor_maps[dataset] = {
            row["window_id"]: row.get("prediction_dir")
            for row in (json.loads(line) for line in index.read_text().splitlines() if line.strip())
        }

    missing = []
    dyn_segments = set()
    dyn_windows = 0
    observed_segments = Counter()
    for entry in entries:
        dataset, segment = entry["dataset"], entry["gallery_stem"]
        observed_segments[dataset] += 1
        staged_segment = staged[dataset] / segment
        selection = staged_segment / "selection.json"
        if not selection.is_file():
            missing.append(("staged", str(selection)))
            continue
        selected = json.loads(selection.read_text())
        if [row["window_id"] for row in selected["windows"]] != [row["window_id"] for row in entry["windows"]]:
            raise RuntimeError(f"WINDOW_ID_MISMATCH:{segment}")
        for window in entry["windows"]:
            cache, window_id = window["gt"]["cache_id"], window["window_id"]
            required = {
                "prepared": prepared[dataset] / cache / "window_input.json",
                "gt": args.src_dir / cache / "gt.npz",
                "ego": args.src_dir / cache / "ego.npz",
                "hawor": resolve_prediction_file(
                    hawor_maps[dataset].get(window_id), hawor_roots[dataset], cache,
                    "hawor", required=False),
            }
            missing.extend((kind, str(path)) for kind, path in required.items() if not path.is_file())
            dyn = args.src_dir / cache / "dyn_hamr.npz"
            if dyn.is_file():
                dyn_windows += 1
                dyn_segments.add(segment)
    if observed_segments != EXPECTED_COUNTS or missing:
        raise RuntimeError("INPUT_COVERAGE_FAILED:" + json.dumps({
            "counts": Counter(kind for kind, _ in missing),
            "samples": [{"kind": kind, "path": path} for kind, path in missing[:20]],
        }, sort_keys=True))
    if len(dyn_segments) != EXPECTED_DYN_SEGMENTS or dyn_windows != EXPECTED_DYN_WINDOWS:
        raise RuntimeError(f"DYN_INTERSECTION_MISMATCH:{len(dyn_segments)}:{dyn_windows}")

    smoke_bytes = tree_bytes(args.smoke_root)
    required_free = smoke_bytes * len(entries) + 50 * 1024**3
    free = shutil.disk_usage(args.output_root.parent).free
    if free < required_free:
        raise RuntimeError(f"INSUFFICIENT_OUTPUT_SPACE:{free}:{required_free}")
    args.output_root.mkdir(parents=True)
    preflight = {
        "status": "passed", "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "segments": len(entries), "windows": 520, "hawor_windows": 520,
        "dyn_hamr_intersection_segments": len(dyn_segments),
        "dyn_hamr_intersection_windows": dyn_windows,
        "worktree_commit": observed_commit, "free_bytes": free,
        "required_free_bytes": required_free,
    }
    (args.output_root / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")

    tools = args.worktree / "visualization/paper_viz/tools"
    python = "/usr/local/bin/python3"
    for dataset in EXPECTED_COUNTS:
        common = [
            "--staged-root", str(staged[dataset]), "--src-dir", str(args.src_dir),
            "--prepared-root", str(prepared[dataset]),
            "--hawor-root", str(hawor_roots[dataset]),
            "--hawor-index", str(hawor_indexes[dataset]),
            "--contact-root", str(args.contact_root), "--mapping", str(args.mapping),
        ]
        run([python, str(tools / "check_3d_world_frame.py"), *common, "--device", "cpu"], args.worktree)
        run([python, str(tools / "batch_3d_render.py"), *common,
             "--out-root", str(args.output_root / dataset), "--stages", "summary", "video",
             "--devices", *args.devices, "--chunks", str(len(args.devices)), "--cell", "320",
             "--panel-size", "0", "--keyframes", "5", "--camera-overlay", "show"], args.worktree)

    results = []
    for entry in entries:
        root = args.output_root / entry["dataset"] / entry["gallery_stem"]
        frames = len(list((root / "_frames").glob("*.png")))
        ok = ((root / "fig1_3d_summary.png").is_file()
              and (root / "video1_3d_matrix.mp4").is_file() and frames == 300)
        results.append({"dataset": entry["dataset"], "segment": entry["gallery_stem"],
                        "frames": frames, "ok": ok})
    if not all(row["ok"] for row in results):
        raise RuntimeError("BATCH_OUTPUT_INCOMPLETE")
    summary = {**preflight, "status": "complete", "results": results}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text("complete\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
