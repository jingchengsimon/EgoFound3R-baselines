#!/usr/bin/env python3
"""Copy one registered CPFS result tree to OSS, then replace it with a compatibility link."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path


CPFS_ROOTS = (Path("/mnt/workspace/sjc/DATA/eval_artifacts"), Path("/mnt/cpfs/sjc/eval_artifacts"))
OSS_ROOT = Path("/mnt/oss/pre-train/ego/eval_artifacts")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def inventory(root: Path) -> list[dict[str, object]]:
    rows = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError("SOURCE_SYMLINK:" + str(path))
        for name in names:
            path = current_path / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("UNSUPPORTED_SOURCE_OBJECT:" + str(path))
            rows.append({"relative": str(path.relative_to(root)), "bytes": info.st_size,
                         "mtime_ns": info.st_mtime_ns})
    return sorted(rows, key=lambda row: str(row["relative"]))


def preflight(source: Path, target: Path, expected_files: int, expected_bytes: int) -> list[dict[str, object]]:
    if not any(root in source.parents for root in CPFS_ROOTS) or source.is_symlink() or not source.is_dir():
        raise RuntimeError("INVALID_CPFS_SOURCE")
    if OSS_ROOT not in target.parents:
        raise RuntimeError("INVALID_OSS_TARGET")
    files = inventory(source)
    observed_bytes = sum(int(row["bytes"]) for row in files)
    if len(files) != expected_files or observed_bytes != expected_bytes:
        raise RuntimeError(f"SOURCE_INVENTORY_MISMATCH:{len(files)}:{observed_bytes}")
    probe = OSS_ROOT / f".tree_rehome_probe_{os.getpid()}"
    try:
        probe.write_text("probe")
        if probe.read_text() != "probe":
            raise RuntimeError("OSSFS_PROBE_MISMATCH")
    finally:
        probe.unlink(missing_ok=True)
    return files


def stage(source: Path, target: Path, expected_files: int, expected_bytes: int) -> dict[str, object]:
    files = preflight(source, target, expected_files, expected_bytes)
    target.mkdir(parents=True, exist_ok=True)
    copied = resumed = 0
    for index, row in enumerate(files, 1):
        source_file = source / str(row["relative"])
        target_file = target / str(row["relative"])
        target_file.parent.mkdir(parents=True, exist_ok=True)
        size = int(row["bytes"])
        if target_file.is_file() and target_file.stat().st_size == size:
            resumed += 1
            continue
        if target_file.exists():
            raise RuntimeError("TARGET_CONFLICT:" + str(target_file))
        partial = target_file.with_name(target_file.name + ".partial")
        if partial.exists() and partial.stat().st_size != size:
            partial.unlink()
        if not partial.exists():
            with source_file.open("rb") as reader, partial.open("wb") as writer:
                shutil.copyfileobj(reader, writer, 16 * 1024 * 1024)
        if partial.stat().st_size != size:
            raise RuntimeError("PARTIAL_SIZE_MISMATCH:" + str(partial))
        partial.replace(target_file)
        copied += 1
        if index % 500 == 0:
            print(json.dumps({"status": "copy", "files": index, "total": len(files)}), flush=True)
    if any((target / str(row["relative"])).stat().st_size != int(row["bytes"]) for row in files):
        raise RuntimeError("TARGET_VERIFICATION_FAILED")
    result = {"status": "complete", "source_root": str(source), "target_root": str(target),
              "files": files, "file_count": len(files), "source_bytes": expected_bytes,
              "copied_files": copied, "resumed_files": resumed, "completed_at_epoch": int(time.time())}
    atomic_json(target / "migration_manifest.json", result)
    (target / "MIGRATION_COMPLETE").write_text("complete\n")
    return result


def active_references(source: Path) -> list[dict[str, object]]:
    root = str(source.resolve())
    ancestors, pid = set(), os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestors:
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            hit = root in command
            for entry in (proc / "fd").iterdir():
                try:
                    value = os.readlink(entry)
                except FileNotFoundError:
                    continue
                hit = hit or value == root or value.startswith(root + "/")
            if hit:
                matches.append({"pid": int(proc.name), "executable": command.split(" ", 1)[0]})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return matches


def release(source: Path, target: Path, report_root: Path) -> dict[str, object]:
    manifest = json.loads((target / "migration_manifest.json").read_text())
    if (target / "MIGRATION_COMPLETE").read_text().strip() != "complete":
        raise RuntimeError("TARGET_NOT_COMPLETE")
    if manifest.get("source_root") != str(source) or manifest.get("target_root") != str(target):
        raise RuntimeError("MANIFEST_IDENTITY_MISMATCH")
    if source.is_symlink():
        if source.resolve() != target.resolve():
            raise RuntimeError("SOURCE_LINK_TARGET_MISMATCH")
        result = {"status": "complete", "resumed": True, "released_bytes": manifest["source_bytes"]}
        atomic_json(report_root / "summary.json", result)
        (report_root / "COMPLETE").write_text("complete\n")
        return result
    matches = active_references(source)
    if matches:
        raise RuntimeError("ACTIVE_SOURCE_REFERENCES:" + json.dumps(matches[:20]))
    observed = inventory(source)
    if observed != manifest["files"]:
        raise RuntimeError("SOURCE_CHANGED_SINCE_MIGRATION")
    for row in observed:
        if (target / str(row["relative"])).stat().st_size != int(row["bytes"]):
            raise RuntimeError("TARGET_CHANGED_SINCE_MIGRATION")
    placeholder = source.with_name(source.name + ".rehome_link")
    backup = source.with_name(source.name + ".rehome_pending_delete")
    if placeholder.exists() or placeholder.is_symlink() or backup.exists():
        raise RuntimeError("REHOME_SIBLING_CONFLICT")
    placeholder.symlink_to(target, target_is_directory=True)
    source.rename(backup)
    try:
        placeholder.rename(source)
    except Exception:
        backup.rename(source)
        placeholder.unlink(missing_ok=True)
        raise
    before = os.statvfs("/mnt/cpfs")
    shutil.rmtree(backup)
    after = os.statvfs("/mnt/cpfs")
    result = {"status": "complete", "source_path": str(source), "target_root": str(target),
              "released_bytes": int(manifest["source_bytes"]),
              "cpfs_available_before": before.f_bavail * before.f_frsize,
              "cpfs_available_after": after.f_bavail * after.f_frsize,
              "completed_at_epoch": int(time.time())}
    atomic_json(report_root / "summary.json", result)
    (report_root / "COMPLETE").write_text("complete\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "release"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = (stage(args.source_root, args.target_root, args.expected_files, args.expected_bytes)
                  if args.mode == "stage" else release(args.source_root, args.target_root, args.report_root))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
