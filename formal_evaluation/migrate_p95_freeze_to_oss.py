#!/usr/bin/env python3
"""Mirror frozen P95 result artifacts to OSSFS and release verified CPFS sources."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


OSS_ROOT = Path("/mnt/oss/pre-train/ego/eval_artifacts/p95_table_freeze_20260915")
HAWOR_GUARDS = (
    "hawor_native_camera_repair_20260915_v7_5000",
    "hand_extensions_full_20260913T142830Z_v2/shared_inputs",
    "/HaWoR/",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    files: list[dict[str, Any]] = []
    links: list[dict[str, str]] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        link_directories = [name for name in directories if (current_path / name).is_symlink()]
        directories[:] = [name for name in directories if name not in link_directories]
        for name in sorted(link_directories):
            path = current_path / name
            links.append({"relative_path": str(path.relative_to(root)), "target": os.readlink(path)})
        for name in sorted(names):
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                links.append({"relative_path": str(path.relative_to(root)), "target": os.readlink(path)})
            elif stat.S_ISREG(info.st_mode):
                files.append({
                    "relative_path": str(path.relative_to(root)),
                    "bytes": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                })
            else:
                raise RuntimeError(f"UNSUPPORTED_SOURCE_TYPE:{path}")
    files.sort(key=lambda row: row["relative_path"])
    links.sort(key=lambda row: row["relative_path"])
    return files, links


def normalized_aliases(path: Path) -> set[str]:
    raw = str(path)
    aliases = {raw}
    if raw.startswith("/mnt/cpfs/"):
        aliases.add("/mnt/workspace/" + raw.removeprefix("/mnt/cpfs/"))
    elif raw.startswith("/mnt/workspace/"):
        aliases.add("/mnt/cpfs/" + raw.removeprefix("/mnt/workspace/"))
    return aliases


def process_references(source: Path) -> list[dict[str, Any]]:
    aliases = normalized_aliases(source)
    own = {os.getpid(), os.getppid()}
    matches: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in own:
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            hits = {alias for alias in aliases if alias in command}
            for descriptor in (proc / "fd").iterdir():
                try:
                    target = os.readlink(descriptor)
                except (FileNotFoundError, PermissionError):
                    continue
                hits.update(alias for alias in aliases if target == alias or target.startswith(alias + "/"))
            if hits:
                matches.append({"pid": int(proc.name), "paths": sorted(hits), "command": command[:500]})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return matches


def rewrite_link_target(
    source_link: Path,
    target_text: str,
    source_to_destination: list[tuple[Path, Path]],
    destination_link: Path,
) -> str:
    if not os.path.isabs(target_text):
        return target_text
    target = Path(target_text)
    for source, destination in source_to_destination:
        for alias in normalized_aliases(source):
            alias_path = Path(alias)
            try:
                relative = target.relative_to(alias_path)
            except ValueError:
                continue
            mapped = destination / relative
            return os.path.relpath(mapped, destination_link.parent)
    return target_text


def copy_file(source: Path, destination: Path, expected: dict[str, Any]) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = int(expected["bytes"])
    resumed = destination.is_file() and destination.stat().st_size == size
    if resumed:
        source_hash = file_sha256(source)
        destination_hash = file_sha256(destination)
        if source_hash != destination_hash:
            raise RuntimeError(f"EXISTING_DESTINATION_HASH_MISMATCH:{destination}")
        return {**expected, "sha256": source_hash, "resumed": True}
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"DESTINATION_CONFLICT:{destination}")
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}.{threading.get_ident()}")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    written = 0
    with source.open("rb") as reader, temporary.open("xb") as writer:
        while True:
            block = reader.read(8 * 1024 * 1024)
            if not block:
                break
            writer.write(block)
            digest.update(block)
            written += len(block)
        writer.flush()
        try:
            os.fsync(writer.fileno())
        except OSError:
            pass
    if written != size or temporary.stat().st_size != size:
        raise RuntimeError(f"COPY_SIZE_MISMATCH:{source}:{written}:{size}")
    temporary.replace(destination)
    if destination.stat().st_size != size:
        raise RuntimeError(f"DESTINATION_SIZE_MISMATCH:{destination}")
    return {**expected, "sha256": digest.hexdigest(), "resumed": False}


def migrate_entry(
    entry: dict[str, Any],
    source_to_destination: list[tuple[Path, Path]],
    file_workers: int,
    delete_verified_source: bool,
    progress_path: Path,
    progress_lock: threading.Lock,
) -> dict[str, Any]:
    source = Path(entry["cpfs_path"])
    destination = Path(entry["proposed_oss_path"])
    if any(token in str(source) for token in HAWOR_GUARDS):
        raise RuntimeError(f"HAWOR_INPUT_GUARD:{source}")
    if not source.exists() and (destination / "VERIFIED").is_file() and (destination / "SOURCE_RELEASED").is_file():
        manifest_path = destination / "migration_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest_hash = file_sha256(manifest_path)
        if (destination / "VERIFIED").read_text().strip() != manifest_hash:
            raise RuntimeError(f"RELEASED_MANIFEST_HASH_MISMATCH:{destination}")
        return {
            "id": entry["id"], "source": str(source), "destination": str(destination),
            "regular_files": int(manifest["regular_files"]),
            "logical_bytes": int(manifest["logical_bytes"]),
            "manifest_sha256": manifest_hash, "source_released": True,
        }
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError(f"INVALID_SOURCE:{source}")
    if destination != OSS_ROOT and OSS_ROOT not in destination.parents:
        raise RuntimeError(f"DESTINATION_OUTSIDE_FREEZE_ROOT:{destination}")
    active = process_references(source)
    if active:
        raise RuntimeError(f"SOURCE_IN_USE:{source}:{json.dumps(active, sort_keys=True)}")
    before_files, before_links = inventory(source)
    before_bytes = sum(int(row["bytes"]) for row in before_files)
    if len(before_files) != int(entry["regular_files"]) or before_bytes != int(entry["logical_bytes"]):
        raise RuntimeError(
            f"FROZEN_SOURCE_DRIFT:{source}:files={len(before_files)}/{entry['regular_files']}:"
            f"bytes={before_bytes}/{entry['logical_bytes']}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for marker in ("migration_manifest.json", "VERIFIED", "SOURCE_RELEASED"):
        (destination / marker).unlink(missing_ok=True)
    copied: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=file_workers) as executor:
        future_to_row = {
            executor.submit(copy_file, source / row["relative_path"], destination / row["relative_path"], row): row
            for row in before_files
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_row):
            copied.append(future.result())
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == len(before_files):
                event = {"status": "copying", "id": entry["id"], "completed_files": completed,
                         "total_files": len(before_files), "at_epoch": int(time.time())}
                with progress_lock:
                    with progress_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event, sort_keys=True), flush=True)
    for row in before_links:
        target = destination / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        desired = rewrite_link_target(
            source / row["relative_path"], row["target"], source_to_destination, target
        )
        if target.is_symlink():
            if os.readlink(target) != desired:
                raise RuntimeError(f"DESTINATION_LINK_CONFLICT:{target}")
        elif target.exists():
            raise RuntimeError(f"DESTINATION_LINK_TYPE_CONFLICT:{target}")
        else:
            target.symlink_to(desired)
        row["stored_target"] = desired
    after_files, after_links = inventory(source)
    if before_files != after_files or before_links != after_links:
        raise RuntimeError(f"SOURCE_CHANGED_DURING_COPY:{source}")
    destination_files, destination_links = inventory(destination)
    if len(destination_files) != len(before_files) or sum(int(row["bytes"]) for row in destination_files) != before_bytes:
        raise RuntimeError(f"DESTINATION_INVENTORY_MISMATCH:{destination}")
    expected_sizes = {row["relative_path"]: int(row["bytes"]) for row in before_files}
    observed_sizes = {row["relative_path"]: int(row["bytes"]) for row in destination_files}
    if expected_sizes != observed_sizes:
        raise RuntimeError(f"DESTINATION_FILESET_MISMATCH:{destination}")
    if {row["relative_path"] for row in destination_links} != {row["relative_path"] for row in before_links}:
        raise RuntimeError(f"DESTINATION_LINKSET_MISMATCH:{destination}")
    for row in destination_links:
        resolved = str((destination / row["relative_path"]).resolve())
        if any(resolved == alias or resolved.startswith(alias + "/")
               for mapped_source, _ in source_to_destination for alias in normalized_aliases(mapped_source)):
            raise RuntimeError(f"OSS_LINK_DEPENDS_ON_CPFS:{destination / row['relative_path']}")
    copied.sort(key=lambda row: row["relative_path"])
    manifest = {
        "schema_version": "p95_artifact_mirror_v1",
        "id": entry["id"],
        "source": str(source),
        "destination": str(destination),
        "regular_files": len(copied),
        "logical_bytes": before_bytes,
        "symlinks": before_links,
        "files": copied,
        "source_snapshot_verified_unchanged": True,
        "destination_size_and_fileset_verified": True,
        "completed_at_epoch": int(time.time()),
    }
    atomic_json(destination / "migration_manifest.json", manifest)
    manifest_hash = file_sha256(destination / "migration_manifest.json")
    (destination / "VERIFIED").write_text(manifest_hash + "\n")
    released = False
    if delete_verified_source:
        active = process_references(source)
        if active:
            raise RuntimeError(f"SOURCE_BECAME_IN_USE:{source}:{json.dumps(active, sort_keys=True)}")
        shutil.rmtree(source)
        if source.exists():
            raise RuntimeError(f"SOURCE_DELETE_FAILED:{source}")
        (destination / "SOURCE_RELEASED").write_text(str(int(time.time())) + "\n")
        released = True
    result = {
        "id": entry["id"], "source": str(source), "destination": str(destination),
        "regular_files": len(copied), "logical_bytes": before_bytes,
        "manifest_sha256": manifest_hash, "source_released": released,
    }
    event = {"status": "verified_and_released" if released else "verified", **result,
             "at_epoch": int(time.time())}
    with progress_lock:
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(event, sort_keys=True), flush=True)
    return result


def ossfs_preflight() -> None:
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3:
            mounts.append((Path(fields[1].replace("\\040", " ")), fields[2]))
    matching = [row for row in mounts if row[0] == OSS_ROOT or row[0] in OSS_ROOT.parents]
    if not matching or max(matching, key=lambda row: len(str(row[0])))[1] != "fuse.ossfs2":
        raise RuntimeError("OSSFS_MOUNT_UNHEALTHY")
    OSS_ROOT.mkdir(parents=True, exist_ok=True)
    probe = OSS_ROOT / f".write_probe_{os.getpid()}"
    probe.write_text("probe\n")
    if probe.read_text() != "probe\n":
        raise RuntimeError("OSSFS_WRITE_PROBE_MISMATCH")
    probe.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OSS_ROOT)
    parser.add_argument("--entry-workers", type=int, default=6)
    parser.add_argument("--file-workers", type=int, default=8)
    parser.add_argument("--delete-verified-source", action="store_true")
    args = parser.parse_args()
    if args.output_root != OSS_ROOT:
        raise ValueError(f"OUTPUT_ROOT_MISMATCH:{args.output_root}")
    frozen = json.loads(args.freeze.read_text())
    if frozen.get("schema_version") != "p95_table_artifact_freeze_v1":
        raise ValueError("INVALID_FREEZE_SCHEMA")
    ossfs_preflight()
    entries = [row for row in frozen["entries"] if row["migration_status"] != "already_oss_backed_via_cpfs_symlink"]
    source_to_destination = [(Path(row["cpfs_path"]), Path(row["proposed_oss_path"])) for row in entries]
    progress_path = OSS_ROOT / "progress.jsonl"
    progress_lock = threading.Lock()
    started = int(time.time())
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.entry_workers) as executor:
        future_to_entry = {
            executor.submit(
                migrate_entry, row, source_to_destination, args.file_workers,
                args.delete_verified_source, progress_path, progress_lock,
            ): row
            for row in entries
        }
        for future in concurrent.futures.as_completed(future_to_entry):
            row = future_to_entry[future]
            try:
                results.append(future.result())
            except Exception as error:  # keep independent entries progressing
                failure = {"id": row["id"], "error": f"{type(error).__name__}:{error}"}
                failures.append(failure)
                print(json.dumps({"status": "failed", **failure}, sort_keys=True), flush=True)
    summary = {
        "schema_version": "p95_artifact_migration_summary_v1",
        "status": "complete" if not failures else "failed",
        "started_at_epoch": started,
        "completed_at_epoch": int(time.time()),
        "entry_workers": args.entry_workers,
        "file_workers_per_entry": args.file_workers,
        "source_release_requested": args.delete_verified_source,
        "completed": sorted(results, key=lambda row: row["id"]),
        "failures": sorted(failures, key=lambda row: row["id"]),
        "completed_entries": len(results),
        "expected_entries": len(entries),
        "verified_bytes": sum(int(row["logical_bytes"]) for row in results),
        "released_bytes": sum(int(row["logical_bytes"]) for row in results if row["source_released"]),
        "already_oss_backed_entries": [row["id"] for row in frozen["entries"]
                                      if row["migration_status"] == "already_oss_backed_via_cpfs_symlink"],
    }
    atomic_json(OSS_ROOT / "summary.json", summary)
    if failures:
        raise SystemExit(1)
    (OSS_ROOT / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
