#!/usr/bin/env python3
"""Delete exact CPFS sources only after validating their OSS migration manifest."""

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


ALLOWED_CPFS_ROOT = Path("/mnt/workspace/sjc/DATA")
ALLOWED_OSS_ROOT = Path("/mnt/oss/pre-train/ego/eval_artifacts")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def tree_bytes(root: Path) -> int:
    return sum((Path(current) / name).stat().st_size
               for current, _, names in os.walk(root) for name in names
               if not (Path(current) / name).is_symlink())


def validate_manifest(target: Path, dataset: str, expected_bytes: int) -> dict[str, Any]:
    if target != ALLOWED_OSS_ROOT and ALLOWED_OSS_ROOT not in target.parents:
        raise ValueError(f"target outside authorized OSS prefix: {target}")
    if (target / "COMPLETE").read_text().strip() != "complete":
        raise RuntimeError(f"INVALID_COMPLETE:{target}")
    manifest = json.loads((target / "migration_manifest.json").read_text())
    if (manifest.get("status"), manifest.get("dataset"), int(manifest.get("source_bytes", -1))) != (
        "complete", dataset, expected_bytes
    ):
        raise RuntimeError(f"MANIFEST_MISMATCH:{dataset}")
    for row in manifest["files"]:
        stored = Path(row["path"])
        if target not in stored.parents:
            raise RuntimeError(f"MANIFEST_PATH_ESCAPE:{stored}")
        if row["storage"] == "file":
            if not stored.is_file() or stored.stat().st_size != int(row["bytes"]):
                raise RuntimeError(f"MIGRATED_FILE_MISMATCH:{stored}")
        elif row["storage"] == "parts":
            observed = sum(Path(part["path"]).stat().st_size for part in row["parts"])
            if observed != int(row["bytes"]):
                raise RuntimeError(f"MIGRATED_PARTS_MISMATCH:{stored}")
        else:
            raise RuntimeError(f"UNKNOWN_STORAGE:{row['storage']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if spec.get("schema_version") != "contact_artifact_cleanup_v1":
        raise ValueError("unsupported cleanup spec")
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        verified: list[dict[str, Any]] = []
        for dataset, row in spec["datasets"].items():
            expected = int(row["source_bytes"])
            validate_manifest(Path(row["oss_target"]), dataset, expected)
            sources = [Path(value) for value in row["sources"]]
            for source in sources:
                if source.is_symlink() or ALLOWED_CPFS_ROOT not in source.parents or not source.is_dir():
                    raise RuntimeError(f"UNSAFE_SOURCE:{source}")
            observed = sum(tree_bytes(source) for source in sources)
            if observed != expected:
                raise RuntimeError(f"SOURCE_BYTES_MISMATCH:{dataset}:{observed}:{expected}")
            verified.append({"dataset": dataset, "source_bytes": observed,
                             "sources": [str(source) for source in sources]})
        for row in verified:
            for source in row["sources"]:
                shutil.rmtree(source)
                if Path(source).exists():
                    raise RuntimeError(f"DELETE_FAILED:{source}")
                print(json.dumps({"status": "deleted", "source": source}), flush=True)
        result = {"status": "complete", "completed_at_epoch": int(time.time()),
                  "deleted": verified, "deleted_bytes": sum(row["source_bytes"] for row in verified)}
        atomic_json(args.output_root / "cleanup_report.json", result)
        (args.output_root / "COMPLETE").write_text("complete\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
