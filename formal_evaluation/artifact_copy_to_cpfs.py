#!/usr/bin/env python3
"""Copy exact registered OSS evaluation inputs to a resumable CPFS mirror."""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import shutil
import tempfile
import time
from pathlib import Path


OSS_ROOT = Path("/mnt/oss/pre-train/ego/eval_artifacts")
CPFS_ROOT = Path("/mnt/cpfs/sjc/eval_artifacts")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _within(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path outside allowed root: {path}") from error


def tree_files(root: Path) -> list[tuple[Path, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return [
        (path, path.relative_to(root))
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def source_files(root: Path) -> list[tuple[Path, Path]]:
    _within(root, OSS_ROOT)
    return tree_files(root)


def copy_file(source: Path, destination: Path, size: int) -> bool:
    if destination.exists():
        if destination.stat().st_size != size:
            raise RuntimeError(f"DESTINATION_SIZE_CONFLICT:{destination}")
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with source.open("rb") as reader, partial.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
    if partial.stat().st_size != size:
        raise RuntimeError(f"PARTIAL_SIZE_MISMATCH:{partial}")
    partial.replace(destination)
    return False


def run(spec: dict, output_root: Path, reserve_bytes: int) -> dict:
    _within(output_root, CPFS_ROOT)
    copied: list[dict] = []
    external: list[dict] = []
    seen: set[str] = set()
    for row in spec["sources"]:
        raw = str(row["source"])
        if raw.startswith(str(OSS_ROOT) + "/"):
            if raw not in seen:
                copied.append(row)
                seen.add(raw)
        else:
            external.append(row)
    payload = output_root / "payload"

    inventory = []
    missing_bytes = 0
    for row in copied:
        source = Path(row["source"])
        files = source_files(source)
        bytes_total = sum(path.stat().st_size for path, _ in files)
        destination = payload / _within(source, OSS_ROOT)
        destination_existing = {
            relative: (destination / relative).stat().st_size
            for _, relative in files
            if (destination / relative).is_file()
        }
        for path, relative in files:
            size = path.stat().st_size
            observed = destination_existing.get(relative)
            if observed is not None and observed != size:
                raise RuntimeError(f"DESTINATION_SIZE_CONFLICT:{destination / relative}")
            if observed is None:
                missing_bytes += size
        inventory.append({**row, "destination": str(destination), "files": len(files), "bytes": bytes_total})

    free_bytes = shutil.disk_usage(CPFS_ROOT).free
    preflight = {
        "status": "ready",
        "source_roots": len(inventory),
        "source_files": sum(row["files"] for row in inventory),
        "source_bytes": sum(row["bytes"] for row in inventory),
        "missing_bytes": missing_bytes,
        "cpfs_free_bytes": free_bytes,
        "reserve_bytes": reserve_bytes,
        "catalog_run_ids": spec["catalog_run_ids"],
        "methods": spec["methods"],
        "inventory": inventory,
        "external_registered_roots_not_copied": external,
    }
    atomic_json(output_root / "preflight.json", preflight)
    print(json.dumps({key: preflight[key] for key in (
        "status", "source_roots", "source_files", "source_bytes", "missing_bytes", "cpfs_free_bytes"
    )}), flush=True)
    if free_bytes - missing_bytes < reserve_bytes:
        raise RuntimeError("INSUFFICIENT_CPFS_HEADROOM")

    copied_files = resumed_files = copied_bytes = 0
    for root_number, row in enumerate(inventory, 1):
        source = Path(row["source"])
        destination = Path(row["destination"])
        for number, (path, relative) in enumerate(source_files(source), 1):
            size = path.stat().st_size
            resumed = copy_file(path, destination / relative, size)
            resumed_files += int(resumed)
            copied_files += int(not resumed)
            copied_bytes += 0 if resumed else size
            if number % 100 == 0:
                print(json.dumps({"stage": "copy", "root": root_number, "roots": len(inventory),
                                  "files": number, "root_files": row["files"]}), flush=True)
        observed = tree_files(destination)
        observed_stats = {"files": len(observed), "bytes": sum(path.stat().st_size for path, _ in observed)}
        expected_stats = {"files": row["files"], "bytes": row["bytes"]}
        if observed_stats != expected_stats:
            raise RuntimeError(f"DESTINATION_TREE_MISMATCH:{destination}")
        print(json.dumps({"stage": "root_complete", "root": root_number, "roots": len(inventory),
                          "source": str(source), **observed_stats}), flush=True)

    result = {
        **preflight,
        "status": "complete",
        "completed_at_epoch": int(time.time()),
        "copied_files": copied_files,
        "resumed_files": resumed_files,
        "copied_bytes": copied_bytes,
    }
    atomic_json(output_root / "migration_manifest.json", result)
    (output_root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-b64", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=int, default=20)
    args = parser.parse_args()
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        spec = json.loads(base64.urlsafe_b64decode(args.sources_b64))
        result = run(spec, args.output_root, args.reserve_gib * 1024**3)
    print(json.dumps({"status": result["status"], "copied_files": result["copied_files"],
                      "copied_bytes": result["copied_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
