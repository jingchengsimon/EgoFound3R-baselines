#!/usr/bin/env python3
"""Resumably archive registered contact artifacts to an OSSFS bundle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def tree_stats(root: Path) -> dict[str, int]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = bytes_total = 0
    for current, _, names in os.walk(root):
        for name in names:
            path = Path(current) / name
            if path.is_symlink():
                continue
            files += 1
            bytes_total += path.stat().st_size
    return {"files": files, "bytes": bytes_total}


def ossfs_preflight(destination: Path) -> None:
    allowed = Path("/mnt/oss/pre-train/ego/eval_artifacts")
    if destination != allowed and allowed not in destination.parents:
        raise ValueError(f"destination outside authorized OSS prefix: {destination}")
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3:
            mounts.append((Path(fields[1].replace("\\040", " ")), fields[2]))
    matching = [item for item in mounts if item[0] == allowed or item[0] in allowed.parents]
    if not matching or max(matching, key=lambda item: len(str(item[0])))[1] != "fuse.ossfs2":
        raise RuntimeError("OSSFS_MOUNT_UNHEALTHY")
    probe = allowed / f".migration_probe_{os.getpid()}"
    try:
        probe.write_text("probe")
        if probe.read_text() != "probe":
            raise RuntimeError("OSSFS_PROBE_MISMATCH")
    finally:
        probe.unlink(missing_ok=True)


def source_files(source: Path) -> list[tuple[Path, Path]]:
    if source.is_file():
        return [(source, Path(source.name))]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return [
        (path, path.relative_to(source))
        for path in sorted(source.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def copy_small(source: Path, destination: Path, size: int) -> dict[str, Any]:
    if destination.exists():
        if destination.stat().st_size != size:
            raise RuntimeError(f"DESTINATION_SIZE_CONFLICT:{destination}")
        return {"storage": "file", "bytes": size, "path": str(destination), "resumed": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    with source.open("rb") as reader, partial.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
    if partial.stat().st_size != size:
        raise RuntimeError(f"PARTIAL_SIZE_MISMATCH:{partial}")
    partial.replace(destination)
    return {"storage": "file", "bytes": size, "path": str(destination), "resumed": False}


def copy_parts(source: Path, destination: Path, size: int, chunk_bytes: int) -> dict[str, Any]:
    parts_root = destination.with_name(destination.name + ".parts")
    parts_root.mkdir(parents=True, exist_ok=True)
    parts = []
    with source.open("rb") as reader:
        for index, offset in enumerate(range(0, size, chunk_bytes)):
            expected = min(chunk_bytes, size - offset)
            part = parts_root / f"part-{index:05d}"
            resumed = False
            if part.exists() and part.stat().st_size == expected:
                reader.seek(expected, os.SEEK_CUR)
                resumed = True
            else:
                part.unlink(missing_ok=True)
                reader.seek(offset)
                remaining = expected
                with part.open("wb") as writer:
                    while remaining:
                        block = reader.read(min(16 * 1024 * 1024, remaining))
                        if not block:
                            raise EOFError(source)
                        writer.write(block)
                        remaining -= len(block)
                if part.stat().st_size != expected:
                    raise RuntimeError(f"PART_SIZE_MISMATCH:{part}")
            parts.append({"path": str(part), "bytes": expected, "offset": offset, "resumed": resumed})
            print(json.dumps({"status": "part", "source": str(source), "index": index,
                              "parts": (size + chunk_bytes - 1) // chunk_bytes,
                              "bytes": expected, "resumed": resumed}), flush=True)
    return {"storage": "parts", "bytes": size, "path": str(parts_root), "parts": parts}


def verify_external(reference: dict[str, Any]) -> dict[str, Any]:
    destination = Path(reference["destination"])
    destination_stats = tree_stats(destination)
    if "source" in reference:
        source_stats = tree_stats(Path(reference["source"]))
        if source_stats != destination_stats:
            raise RuntimeError(f"EXTERNAL_REF_MISMATCH:{reference['role']}")
    else:
        expected = {"files": int(reference["expected_files"]), "bytes": int(reference["expected_bytes"])}
        if destination_stats != expected:
            raise RuntimeError(f"EXTERNAL_REF_MISMATCH:{reference['role']}")
        source_stats = None
    return {**reference, "source_stats": source_stats, "destination_stats": destination_stats}


def migrate(spec: dict[str, Any], dataset: str, destination: Path) -> dict[str, Any]:
    dataset_spec = spec["datasets"][dataset]
    chunk_bytes = int(spec["chunk_bytes"])
    external = [verify_external(row) for row in dataset_spec["external_refs"]]
    for entry in dataset_spec["copy"]:
        source_files(Path(entry["source"]))
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in dataset_spec["copy"]:
        source = Path(entry["source"])
        relative_root = Path(entry["relative_destination"])
        for path, relative in source_files(source):
            size = path.stat().st_size
            target = destination / relative_root / relative
            record = (copy_parts(path, target, size, chunk_bytes)
                      if size > chunk_bytes else copy_small(path, target, size))
            records.append({"source": str(path), "relative_destination": str(relative_root / relative), **record})
            print(json.dumps({"status": "file", "source": str(path), "bytes": size,
                              "storage": record["storage"]}), flush=True)
    result = {
        "status": "complete",
        "dataset": dataset,
        "destination_root": str(destination),
        "completed_at_epoch": int(time.time()),
        "files": records,
        "external_refs": external,
        "source_bytes": sum(row["bytes"] for row in records),
    }
    atomic_json(destination / "migration_manifest.json", result)
    (destination / "COMPLETE").write_text("complete\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if spec.get("schema_version") != "contact_artifact_migration_v1":
        raise ValueError("unsupported migration spec")
    ossfs_preflight(args.destination_root)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = migrate(spec, args.dataset, args.destination_root)
    print(json.dumps({"status": "complete", "dataset": args.dataset,
                      "source_bytes": result["source_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
