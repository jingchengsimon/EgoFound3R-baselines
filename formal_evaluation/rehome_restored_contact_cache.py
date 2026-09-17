#!/usr/bin/env python3
"""Rehome the registered Result3 contact-cache restore through its original OSS objects."""

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


CPFS_ROOTS = (
    Path("/mnt/workspace/sjc/DATA/eval_artifacts"),
    Path("/mnt/cpfs/sjc/eval_artifacts"),
)
OSS_ROOT = Path("/mnt/oss/pre-train/ego/eval_artifacts")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def tree_stats(path: Path) -> dict[str, int]:
    if path.is_file():
        return {"files": 1, "bytes": path.stat().st_size}
    files = bytes_total = 0
    for current, directories, names in os.walk(path, followlinks=False):
        for name in directories:
            if (Path(current) / name).is_symlink():
                raise RuntimeError("NESTED_SYMLINK:" + str(Path(current) / name))
        for name in names:
            item = Path(current) / name
            if item.is_symlink():
                raise RuntimeError("NESTED_SYMLINK:" + str(item))
            info = item.stat()
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("UNSUPPORTED_OBJECT:" + str(item))
            files += 1
            bytes_total += info.st_size
    return {"files": files, "bytes": bytes_total}


def oss_preflight(target: Path) -> None:
    if OSS_ROOT not in target.parents:
        raise RuntimeError("TARGET_OUTSIDE_OSS:" + str(target))
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3:
            mounts.append((Path(fields[1].replace("\\040", " ")), fields[2]))
    matching = [item for item in mounts if item[0] == OSS_ROOT or item[0] in OSS_ROOT.parents]
    if not matching or max(matching, key=lambda item: len(str(item[0])))[1] != "fuse.ossfs2":
        raise RuntimeError("OSSFS_MOUNT_UNHEALTHY")
    probe = OSS_ROOT / f".rehome_probe_{os.getpid()}"
    try:
        probe.write_text("probe")
        if probe.read_text() != "probe":
            raise RuntimeError("OSSFS_PROBE_MISMATCH")
    finally:
        probe.unlink(missing_ok=True)


def records(source_root: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in (source_root / "restored_files.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("EMPTY_RESTORE_RECORDS")
    for row in rows:
        source = Path(str(row["source"]))
        destination = Path(str(row["destination"]))
        if OSS_ROOT not in source.parents or source_root not in destination.parents:
            raise RuntimeError("RECORD_PATH_OUTSIDE_REGISTERED_ROOTS")
    return rows


def stage(source_root: Path, target_root: Path) -> dict[str, object]:
    oss_preflight(target_root)
    if not any(root in source_root.parents for root in CPFS_ROOTS) or source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeError("INVALID_CPFS_SOURCE")
    for name in ("report.json", "summary.json", "restored_files.jsonl", "COMPLETE"):
        if not (source_root / name).is_file():
            raise RuntimeError("MISSING_SOURCE_ARTIFACT:" + name)
    rows = records(source_root)
    mapped_roots: list[Path] = []
    links = []
    for row in rows:
        source = Path(str(row["source"]))
        destination = Path(str(row["destination"]))
        source_stats, destination_stats = tree_stats(source), tree_stats(destination)
        if source_stats != destination_stats or source_stats["bytes"] != int(row["bytes"]):
            raise RuntimeError("RESTORE_RECORD_MISMATCH:" + json.dumps({
                "source": str(source), "destination": str(destination),
                "record_bytes": int(row["bytes"]), "source_stats": source_stats,
                "destination_stats": destination_stats,
            }, sort_keys=True))
        mapped_roots.append(destination)
        links.append({"relative": str(destination.relative_to(source_root)), "source": str(source), **source_stats})
    extras = []
    for current, directories, names in os.walk(source_root, followlinks=False):
        current_path = Path(current)
        directories[:] = [name for name in directories if not any(
            current_path / name == root or root in (current_path / name).parents for root in mapped_roots
        )]
        for name in names:
            path = current_path / name
            if any(path == root or root in path.parents for root in mapped_roots):
                continue
            if path.is_symlink():
                raise RuntimeError("UNMAPPED_SYMLINK:" + str(path))
            extras.append((path, path.relative_to(source_root), path.stat().st_size))
    if sum(size for _, _, size in extras) > 256 * 1024 * 1024:
        raise RuntimeError("UNMAPPED_BYTES_EXCEED_LIMIT")
    if target_root.exists():
        raise RuntimeError("TARGET_ALREADY_EXISTS:" + str(target_root))
    target_root.mkdir(parents=True)
    for row in links:
        target = target_root / str(row["relative"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(str(row["source"]), target_is_directory=Path(str(row["source"])).is_dir())
    for source, relative, _ in extras:
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    result = {
        "status": "complete",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "links": links,
        "copied_files": [{"relative": str(relative), "bytes": size} for _, relative, size in extras],
        "source_bytes": sum(int(row["bytes"]) for row in links) + sum(size for _, _, size in extras),
        "completed_at_epoch": int(time.time()),
    }
    atomic_json(target_root / "rehome_manifest.json", result)
    (target_root / "REHOME_COMPLETE").write_text("complete\n")
    return result


def process_references(source_root: Path) -> list[dict[str, object]]:
    root = str(source_root.resolve())
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
                    target = os.readlink(entry)
                except FileNotFoundError:
                    continue
                hit = hit or target == root or target.startswith(root + "/")
            if hit:
                matches.append({"pid": int(proc.name), "executable": command.split(" ", 1)[0]})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return matches


def release(source_root: Path, target_root: Path, report_root: Path) -> dict[str, object]:
    oss_preflight(target_root)
    manifest = json.loads((target_root / "rehome_manifest.json").read_text())
    if (target_root / "REHOME_COMPLETE").read_text().strip() != "complete":
        raise RuntimeError("TARGET_NOT_COMPLETE")
    if manifest.get("source_root") != str(source_root) or manifest.get("target_root") != str(target_root):
        raise RuntimeError("REHOME_MANIFEST_IDENTITY_MISMATCH")
    if source_root.is_symlink():
        if source_root.resolve() != target_root.resolve():
            raise RuntimeError("SOURCE_LINK_TARGET_MISMATCH")
        result = {"status": "complete", "resumed": True, "released_bytes": int(manifest["source_bytes"])}
        atomic_json(report_root / "summary.json", result)
        (report_root / "COMPLETE").write_text("complete\n")
        return result
    matches = process_references(source_root)
    if matches:
        raise RuntimeError("ACTIVE_SOURCE_REFERENCES:" + json.dumps(matches[:20]))
    for row in manifest["links"]:
        source = Path(str(row["source"]))
        destination = source_root / str(row["relative"])
        target = target_root / str(row["relative"])
        expected = {"files": int(row["files"]), "bytes": int(row["bytes"])}
        if tree_stats(source) != expected or tree_stats(destination) != expected:
            raise RuntimeError("SOURCE_CHANGED:" + str(destination))
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise RuntimeError("TARGET_LINK_CHANGED:" + str(target))
    placeholder = source_root.with_name(source_root.name + ".rehome_link")
    backup = source_root.with_name(source_root.name + ".rehome_pending_delete")
    if placeholder.exists() or placeholder.is_symlink() or backup.exists():
        raise RuntimeError("REHOME_SIBLING_CONFLICT")
    placeholder.symlink_to(target_root, target_is_directory=True)
    source_root.rename(backup)
    try:
        placeholder.rename(source_root)
    except Exception:
        backup.rename(source_root)
        placeholder.unlink(missing_ok=True)
        raise
    before = os.statvfs("/mnt/cpfs")
    shutil.rmtree(backup)
    after = os.statvfs("/mnt/cpfs")
    result = {
        "status": "complete",
        "source_path": str(source_root),
        "target_root": str(target_root),
        "released_bytes": int(manifest["source_bytes"]),
        "cpfs_available_before": before.f_bavail * before.f_frsize,
        "cpfs_available_after": after.f_bavail * after.f_frsize,
        "completed_at_epoch": int(time.time()),
    }
    atomic_json(report_root / "summary.json", result)
    (report_root / "COMPLETE").write_text("complete\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stage", "release"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = (stage(args.source_root, args.target_root) if args.mode == "stage" else
                  release(args.source_root, args.target_root, args.report_root))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
