#!/usr/bin/env python3
"""Build the frozen-114 jump-cut plan and audit its exact cached inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path


STITCH_DATASETS = {"h2o", "oakink_v2"}
PREPARED_RUNS = {
    "h2o": "prep_h2o_60f_strict_20260822T030835Z_5001_g1",
    "oakink_v2": "prep_6dataset_60f_strict_oakink_v2_20260819T144050Z",
}
METHODS = ("gt", "wilor", "pad_hand", "egoforce", "reviv4d", "hawor")
OLD_ASSEMBLY = Path("/mnt/cpfs/paper_viz_2d_render_sources_frozen114_20260917_r1")
OLD_HAWOR = Path("/mnt/cpfs/paper_viz_2d_stage_6001_frozen114_20260917_r1/hawor")
CPFS_BASES = (
    Path("/mnt/cpfs/sjc/DATA/eval_artifacts"),
    Path("/mnt/cpfs/sjc/eval_artifacts"),
)


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part
                 for part in re.split(r"(\d+)", str(value)))


def readable_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def find_prepared_root(dataset: str) -> Path:
    relative = Path(PREPARED_RUNS[dataset]) / "window_inputs" / dataset
    candidates = [base / relative for base in CPFS_BASES]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("PREPARED_ROOT:" + ",".join(map(str, candidates)))


def load_sequence_axes(dataset: str, wanted: set[str]) -> tuple[dict[str, list[dict]], Path]:
    root = find_prepared_root(dataset)
    windows = defaultdict(list)
    for record_path in root.glob("*/window_input.json"):
        record = json.loads(record_path.read_text())
        window_id = str(record["window_id"])
        sequence = window_id.rsplit(":", 1)[0]
        if sequence not in wanted:
            continue
        frames = list(map(str, record["frame_ids"]))
        windows[sequence].append((natural_key(frames[0]), window_id,
                                  record_path.parent.name, frames))
    axes = {}
    for sequence in wanted:
        seen = set()
        axis = []
        for _, window_id, cache_id, frames in sorted(windows.get(sequence, [])):
            for index, frame_id in enumerate(frames):
                if frame_id in seen:
                    continue
                seen.add(frame_id)
                axis.append({"window_id": window_id, "cache_id": cache_id,
                             "index": index, "frame_id": frame_id})
        axes[sequence] = axis
    return axes, root


def centered_slice(axis: list[dict], anchor_frame: str, length: int = 300) -> list[dict]:
    positions = [index for index, frame in enumerate(axis)
                 if frame["frame_id"] == anchor_frame]
    if len(positions) != 1:
        raise ValueError(f"ANCHOR_NOT_UNIQUE:{anchor_frame}:{len(positions)}")
    if len(axis) <= length:
        return axis
    anchor = positions[0]
    start = max(0, min(anchor - length // 2, len(axis) - length))
    return axis[start:start + length]


def jump_blocks(frames: list[dict], expected_step: int) -> list[dict]:
    starts = [0]
    for index, (left, right) in enumerate(zip(frames, frames[1:]), 1):
        try:
            gap = int(right["frame_id"]) - int(left["frame_id"])
        except ValueError:
            gap = expected_step if (left["cache_id"] == right["cache_id"]
                                    and right["index"] == left["index"] + 1) else expected_step + 1
        if gap > expected_step:
            starts.append(index)
    starts.append(len(frames))
    return [{"start": start, "end_exclusive": stop,
             "first_frame_id": frames[start]["frame_id"],
             "last_frame_id": frames[stop - 1]["frame_id"]}
            for start, stop in zip(starts, starts[1:])]


def stitched_entry(entry: dict, axis: list[dict] | None) -> dict:
    result = dict(entry)
    original_frames = list(map(str, entry["frame_ids"]))
    if entry["dataset"] not in STITCH_DATASETS:
        result["jump_stitch"] = {"policy": "unchanged", "original_length": len(original_frames),
                                  "stitched_length": len(original_frames), "jump_count": 0}
        return result
    if not axis:
        raise ValueError(f"SEQUENCE_AXIS_EMPTY:{entry['dataset']}:{entry['sequence_id']}")
    chosen = centered_slice(axis, str(entry["frame_id"]))
    blocks = jump_blocks(chosen, max(1, int(entry.get("within_window_step_p95", 1))))
    result.update(
        frame_ids=[frame["frame_id"] for frame in chosen],
        frame_refs=[{key: frame[key] for key in ("window_id", "cache_id", "index")}
                    for frame in chosen],
        first_frame_id=chosen[0]["frame_id"], last_frame_id=chosen[-1]["frame_id"],
        actual_length=len(chosen), center_index_in_clip=next(
            index for index, frame in enumerate(chosen) if frame["frame_id"] == str(entry["frame_id"])),
    )
    result["jump_stitch"] = {
        "policy": "same_dataset_sequence_registered_axis_no_repeat",
        "original_length": len(original_frames), "stitched_length": len(chosen),
        "target_length": 300, "jump_count": len(blocks) - 1, "blocks": blocks,
        "ranking_anchor_window_id": entry["window_id"],
        "ranking_anchor_frame_id": str(entry["frame_id"]),
    }
    return result


def direct_candidates(alignment: dict, dataset: str, cache: str,
                      window_id: str, method: str) -> list[Path]:
    spec = alignment[dataset]
    candidates = []
    if method == "hawor":
        candidates.extend(Path(root) / cache / "predictions.npz"
                          for root in spec.get("hawor_roots", []))
        candidates.extend((Path(spec.get("hawor_output_root", "")) / cache / "predictions.npz",
                           OLD_HAWOR / dataset / cache / "predictions.npz"))
    elif method == "gt":
        roots = spec.get("gt_roots") or ([spec["gt_root"]] if spec.get("gt_root") else [])
        candidates.extend(Path(root) / f"{cache}.npz" for root in roots)
        candidates.extend(Path(root) / dataset / f"{cache}.npz" for root in roots)
    else:
        candidates.extend(Path(root) / cache / "predictions.npz"
                          for root in spec.get("method_roots", {}).get(method, []))
    candidates.append(OLD_ASSEMBLY / "src" / cache / f"{method}.npz")
    # Several formal artifacts have an exact CPFS mirror. Probe it without
    # assuming that the mirror exists.
    for path in list(candidates):
        prefix = "/mnt/oss/pre-train/ego/eval_artifacts/"
        if str(path).startswith(prefix):
            candidates.extend(base / str(path)[len(prefix):] for base in CPFS_BASES)
    return list(dict.fromkeys(candidates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    raw_manifest = args.manifest.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != args.manifest_sha256:
        raise ValueError("MANIFEST_SHA256_MISMATCH")
    entries = [json.loads(line) for line in raw_manifest.splitlines() if line.strip()]
    if len(entries) != 114:
        raise ValueError(f"FROZEN_SEGMENT_COUNT:{len(entries)}")
    alignment = json.loads(args.alignment.read_text())
    wanted = {dataset: {entry["sequence_id"] for entry in entries if entry["dataset"] == dataset}
              for dataset in STITCH_DATASETS}
    axes, prepared_roots = {}, {}
    for dataset in sorted(STITCH_DATASETS):
        dataset_axes, root = load_sequence_axes(dataset, wanted[dataset])
        axes[dataset] = dataset_axes
        prepared_roots[dataset] = str(root)

    stitched = [stitched_entry(entry, axes.get(entry["dataset"], {}).get(entry["sequence_id"]))
                for entry in entries]
    if any(entry["dataset"] == "taco" and entry["frame_ids"] != original["frame_ids"]
           for entry, original in zip(stitched, entries)):
        raise AssertionError("TACO_CHANGED")

    windows = {}
    for entry in stitched:
        for ref in entry["frame_refs"]:
            windows[(entry["dataset"], ref["cache_id"])] = ref["window_id"]
    missing = []
    coverage = Counter()
    for (dataset, cache), window_id in sorted(windows.items()):
        if dataset not in STITCH_DATASETS:
            continue
        for method in METHODS:
            candidates = direct_candidates(alignment, dataset, cache, window_id, method)
            source = next((path for path in candidates if readable_file(path)), None)
            if source is None:
                missing.append({"dataset": dataset, "cache_id": cache, "window_id": window_id,
                                "method": method, "candidates": list(map(str, candidates))})
            else:
                coverage[(dataset, method)] += 1

    lengths = defaultdict(Counter)
    jump_counts = Counter()
    for entry in stitched:
        lengths[entry["dataset"]][entry["actual_length"]] += 1
        jump_counts[entry["dataset"]] += entry["jump_stitch"]["jump_count"]
    unique_windows = Counter(dataset for dataset, _ in windows)
    required = sum(unique_windows[dataset] * len(METHODS) for dataset in STITCH_DATASETS)
    report = {
        "status": "complete", "render_ready": not missing,
        "manifest_sha256": args.manifest_sha256, "segments": len(stitched),
        "policy": {
            "h2o_oakink_v2": "same dataset+sequence; registered-axis order; no repeats; target <=300",
            "taco": "unchanged", "ranking": "frozen-window relative Ego-vs-baseline alignment advantage",
        },
        "prepared_roots": prepared_roots,
        "lengths": {dataset: dict(sorted(counts.items())) for dataset, counts in lengths.items()},
        "jump_counts": dict(jump_counts), "unique_windows": dict(unique_windows),
        "source_checks": required, "source_present": required - len(missing),
        "coverage": {f"{dataset}:{method}": count for (dataset, method), count in sorted(coverage.items())},
        "missing": missing,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "report.json").exists():
        raise FileExistsError(args.output_root / "report.json")
    (args.output_root / "stitched_manifest.jsonl").write_text(
        "".join(json.dumps(entry, separators=(",", ":")) + "\n" for entry in stitched))
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text(json.dumps({
        "status": "audit_complete", "render_ready": report["render_ready"],
        "segments": len(stitched), "missing_sources": len(missing),
    }, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "render_ready", "segments", "lengths", "jump_counts",
                       "unique_windows", "source_checks", "source_present")}))


if __name__ == "__main__":
    main()
