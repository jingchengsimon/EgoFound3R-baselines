#!/usr/bin/env python3
"""Verify a source snapshot against hashes exported from one Git tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "git_tree_sha256_v1":
        raise SystemExit("unsupported manifest schema")
    if manifest.get("commit") != args.expected_commit:
        raise SystemExit("manifest commit mismatch")

    entries = manifest.get("entries", [])
    expected_paths = {str(entry["path"]) for entry in entries}
    observed_paths: set[str] = set()
    for directory, dirnames, filenames in os.walk(args.root, followlinks=False):
        base = Path(directory)
        for name in list(dirnames):
            candidate = base / name
            relative = candidate.relative_to(args.root)
            if name in {".git", ".pytest_cache"} or "__pycache__" in relative.parts:
                dirnames.remove(name)
                continue
            if candidate.is_symlink():
                observed_paths.add(relative.as_posix())
                dirnames.remove(name)
        for name in filenames:
            relative = (base / name).relative_to(args.root)
            if name == ".git" or ".pytest_cache" in relative.parts:
                continue
            if "__pycache__" in relative.parts or relative.suffix in {".pyc", ".pyo"}:
                continue
            observed_paths.add(relative.as_posix())

    failures: list[dict[str, object]] = [
        {"path": path, "error": "unexpected file"}
        for path in sorted(observed_paths - expected_paths)
    ]
    checked = 0
    for entry in entries:
        relative = PurePosixPath(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"unsafe manifest path: {relative}")
        path = args.root.joinpath(*relative.parts)
        try:
            if entry["mode"] == "120000":
                if not path.is_symlink():
                    raise OSError("not a symlink")
                payload = os.readlink(path).encode("utf-8")
            else:
                if not path.is_file() or path.is_symlink():
                    raise OSError("not a regular file")
                payload = path.read_bytes()
                expected_executable = entry["mode"] == "100755"
                if os.access(path, os.X_OK) != expected_executable:
                    raise OSError("executable mode mismatch")
        except OSError as error:
            failures.append({"path": str(relative), "error": str(error)})
            continue
        checked += 1
        observed = hashlib.sha256(payload).hexdigest()
        if len(payload) != int(entry["bytes"]) or observed != entry["sha256"]:
            failures.append(
                {
                    "path": str(relative),
                    "bytes": len(payload),
                    "sha256": observed,
                }
            )

    result = {
        "commit": manifest["commit"],
        "root": str(args.root),
        "expected_files": len(entries),
        "observed_files": len(observed_paths),
        "checked_files": checked,
        "failures": failures[:20],
        "failure_count": len(failures),
        "ok": not failures,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
