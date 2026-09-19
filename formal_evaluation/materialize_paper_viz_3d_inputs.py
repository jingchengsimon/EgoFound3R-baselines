#!/usr/bin/env python3
"""Materialize the frozen full104 formal prediction caches from OSS to CPFS."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
import zipfile


REQUIRED_METHODS = ("ego", "gt", "wilor", "pad_hand", "egoforce", "reviv4d")
OPTIONAL_METHODS = ("dyn_hamr",)
EGO_KEYS = {
    "hand_valid.npy", "hand_joints_camera.npy", "hand_markers_camera.npy",
    "camera_c2w.npy", "camera_valid.npy",
}
GT_KEYS = {"hand_valid.npy", "hand_vertices_camera.npy", "camera_c2w.npy", "intrinsics.npy"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def rewrite(path: str | Path, alignment: dict) -> Path:
    source = Path(path)
    # The two allocations expose the same shared CPFS under different mount
    # entrances.  Frozen 5000 records use /mnt/workspace/sjc, while 8093 mounts
    # that namespace at /mnt/cpfs/sjc.
    workspace = Path("/mnt/workspace/sjc")
    try:
        source = Path("/mnt/cpfs/sjc") / source.relative_to(workspace)
    except ValueError:
        pass
    alias = Path(alignment["cpfs_alias"])
    try:
        relative = source.relative_to(alias)
    except ValueError:
        return source
    return Path(alignment["cpfs_alias_target"]) / relative


def method_source(window: dict, spec: dict, method: str, alignment: dict) -> Path | None:
    cache = window["gt"]["cache_id"]
    if method == "ego":
        return rewrite(window["pred"]["prediction_dir"], alignment) / "predictions.npz"
    if method == "gt":
        candidates = [rewrite(window["gt_path"], alignment)]
        roots = spec.get("gt_roots") or ([spec["gt_root"]] if spec.get("gt_root") else [])
        candidates.extend(Path(root) / f"{cache}.npz" for root in roots)
    elif method == "dyn_hamr":
        root = spec.get("dyn_hamr_formal_root")
        candidates = [Path(root) / cache / "predictions.npz"] if root else []
    else:
        candidates = [Path(root) / cache / "predictions.npz"
                      for root in spec.get("method_roots", {}).get(method, [])]
    return next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)


def metadata_path(window: dict, source: Path, method: str, alignment: dict) -> Path:
    if method == "gt":
        # The published GT cache is self-contained on OSS: each ``.npz`` has an
        # adjacent ``.json``.  The frozen manifest still records the historical
        # workspace metadata path, which is deliberately not mounted on 8093.
        return source.with_suffix(".json")
    return source.with_name("metadata.json")


def validate_identity(path: Path, dataset: str, method: str, frame_ids: list[str]) -> None:
    payload = json.loads(path.read_text())
    expected_method = "egofound3r" if method == "ego" else method
    method_ok = payload.get("method") == expected_method
    if method == "gt":
        method_ok = payload.get("method") in (None, "gt")
    if payload.get("dataset") != dataset or not method_ok:
        raise ValueError(f"METADATA_IDENTITY_MISMATCH:{method}:{path}")
    if [str(value) for value in payload.get("frame_ids", [])] != frame_ids:
        raise ValueError(f"METADATA_FRAME_IDS_MISMATCH:{method}:{path}")
    if method == "ego" and int(payload.get("global_stride", -1)) != 5:
        raise ValueError(f"EGO_STRIDE_MISMATCH:{path}")


def validate_archive(path: Path, method: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    required = EGO_KEYS if method == "ego" else GT_KEYS if method == "gt" else {"hand_valid.npy"}
    missing = required - names
    if missing:
        raise ValueError(f"ARCHIVE_KEYS_MISSING:{method}:{path}:{sorted(missing)}")


def copy_one(item: dict) -> dict:
    source, target = Path(item["source"]), Path(item["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        source_sha = digest(source)
        target_sha = digest(target)
        if source_sha != target_sha:
            raise ValueError(f"EXISTING_TARGET_DIGEST_MISMATCH:{target}")
        return {**item, "bytes": target.stat().st_size, "sha256": target_sha, "status": "verified_existing"}
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(target.name + ".incoming-" + uuid.uuid4().hex)
    value = hashlib.sha256()
    with source.open("rb") as incoming, temporary.open("xb") as outgoing:
        for chunk in iter(lambda: incoming.read(8 * 1024**2), b""):
            outgoing.write(chunk)
            value.update(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    shutil.copystat(source, temporary)
    temporary.replace(target)
    return {**item, "bytes": target.stat().st_size, "sha256": value.hexdigest(), "status": "copied"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    raw = args.manifest.read_bytes()
    entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    alignment = json.loads(args.alignment.read_text())
    if len(entries) != 104 or hashlib.sha256(raw).hexdigest() != alignment["manifest_sha256"]:
        raise ValueError("FROZEN_MANIFEST_IDENTITY_MISMATCH")

    windows = {}
    for entry in entries:
        for window in entry["windows"]:
            key = (entry["dataset"], window["gt"]["cache_id"])
            if key in windows and windows[key]["window_id"] != window["window_id"]:
                raise ValueError(f"CACHE_IDENTITY_COLLISION:{key}")
            windows[key] = window
    if len(windows) != 520:
        raise ValueError(f"WINDOW_COUNT_MISMATCH:{len(windows)}")

    items, missing = [], []
    for (dataset, cache), window in sorted(windows.items()):
        spec = alignment[dataset]
        frame_ids = [str(value) for value in window["gt"]["frame_ids"]]
        for method in (*REQUIRED_METHODS, *OPTIONAL_METHODS):
            source = method_source(window, spec, method, alignment)
            if source is None or not source.is_file():
                if method in REQUIRED_METHODS:
                    missing.append({"dataset": dataset, "cache_id": cache, "method": method})
                continue
            metadata = metadata_path(window, source, method, alignment)
            if not metadata.is_file():
                if method in REQUIRED_METHODS:
                    missing.append({"dataset": dataset, "cache_id": cache, "method": method,
                                    "metadata": str(metadata)})
                continue
            validate_identity(metadata, dataset, method, frame_ids)
            validate_archive(source, method)
            items.append({"dataset": dataset, "cache_id": cache, "window_id": window["window_id"],
                          "method": method, "source": str(source),
                          "target": str(args.output_root / "src" / cache / f"{method}.npz")})
    if missing:
        raise RuntimeError("MISSING_FORMAL_INPUTS:" + json.dumps(missing[:40], sort_keys=True))

    required_bytes = sum(Path(item["source"]).stat().st_size for item in items
                         if not Path(item["target"]).is_file())
    free = shutil.disk_usage(args.output_root.parent).free
    if free < required_bytes + 20 * 1024**3:
        raise RuntimeError(f"INSUFFICIENT_CPFS_SPACE:{free}:{required_bytes}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        records = list(pool.map(copy_one, items))
    records.sort(key=lambda row: (row["dataset"], row["cache_id"], row["method"]))
    with (args.output_root / "sources.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    counts = Counter(record["method"] for record in records)
    expected = {method: 520 for method in REQUIRED_METHODS}
    if any(counts[method] != count for method, count in expected.items()):
        raise RuntimeError(f"MATERIALIZED_COUNT_MISMATCH:{dict(counts)}")
    summary = {
        "status": "complete", "segments": 104, "windows": 520,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "counts": dict(counts),
        "bytes": sum(record["bytes"] for record in records),
        "copied": sum(record["status"] == "copied" for record in records),
        "verified_existing": sum(record["status"] == "verified_existing" for record in records),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
