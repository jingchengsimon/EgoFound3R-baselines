#!/usr/bin/env python3
"""Copy completed materialized datasets to OSSFS and rewrite their path contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formal_evaluation.datasets.window_inputs import load_window_input


def _replace_prefix(value: str, source_root: Path, destination_root: Path) -> str:
    source, destination = str(source_root), str(destination_root)
    return destination + value[len(source) :] if value == source or value.startswith(source + "/") else value


def _rewrite(value: Any, source_root: Path, destination_root: Path) -> Any:
    if isinstance(value, str):
        return _replace_prefix(value, source_root, destination_root)
    if isinstance(value, list):
        return [_rewrite(item, source_root, destination_root) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite(item, source_root, destination_root) for key, item in value.items()}
    return value


def _atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _rewrite_tree(root: Path, source_root: Path, destination_root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            target = os.readlink(path)
            rewritten = _replace_prefix(target, source_root, destination_root)
            if rewritten != target:
                path.unlink()
                path.symlink_to(rewritten)
        elif path.suffix == ".json":
            _atomic_json(path, _rewrite(json.loads(path.read_text(encoding="utf-8")), source_root, destination_root))


def _index_paths(root: Path, dataset: str) -> list[Path]:
    paths = sorted(root.glob(f"window_inputs_{dataset}_shard_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no {dataset} input indexes under {root}")
    for path in paths:
        sentinel = path.with_suffix(".status.json")
        status = json.loads(sentinel.read_text(encoding="utf-8"))
        if status.get("status") != "complete" or status.get("index") != str(path):
            raise ValueError(f"incomplete input shard: {path}")
    return paths


def _copy_index(source: Path, source_root: Path, destination_root: Path) -> Path:
    destination = destination_root / source.name
    if destination.exists():
        raise FileExistsError(destination)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    payload = "".join(json.dumps(_rewrite(row, source_root, destination_root), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", dir=destination_root, encoding="utf-8", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(destination)
    source_status = json.loads(source.with_suffix(".status.json").read_text(encoding="utf-8"))
    _atomic_json(destination.with_suffix(".status.json"), _rewrite(source_status, source_root, destination_root))
    return destination


def _validate(destination_indexes: list[Path]) -> int:
    count = 0
    for index in destination_indexes:
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            record = load_window_input(Path(str(json.loads(line)["window_input"])))
            if not record["rgb_paths"] or not record["geometry_paths"]:
                raise ValueError(f"empty materialized record in {index}")
            count += 1
    return count


def prepare_oss_destination(destination_root: Path) -> Path:
    """Refuse a stale mount before creating an OSSFS destination directory."""
    probe = destination_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        raise FileNotFoundError(destination_root)
    mounts = []
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 3:
            mounts.append((Path(fields[1].replace("\\040", " ")), fields[2]))
    matching = [item for item in mounts if probe == item[0] or item[0] in probe.parents]
    if not matching or max(matching, key=lambda item: len(str(item[0])))[1] != "fuse.ossfs2":
        raise RuntimeError(f"destination is not on an ossfs2 mount: {destination_root}")
    connections = Path("/sys/fs/fuse/connections")
    if not connections.is_dir() or not any(connections.iterdir()):
        raise RuntimeError(f"ossfs2 mount has no live FUSE connection: {destination_root}")
    destination_root.mkdir(parents=True, exist_ok=True)
    return destination_root.resolve(strict=True)


def promote_dataset(source_root: Path, destination_root: Path, dataset: str) -> dict[str, object]:
    source_root = source_root.resolve(strict=True)
    destination_root = destination_root.resolve(strict=True)
    source_dataset, destination_dataset = source_root / dataset, destination_root / dataset
    indexes = _index_paths(source_root, dataset)
    marker = destination_root / f"promotion_{dataset}.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    if destination_dataset.exists():
        raise FileExistsError(f"destination dataset exists without promotion marker: {destination_dataset}")
    incoming = destination_root / f".incoming_{dataset}"
    if incoming.exists():
        raise FileExistsError(f"incomplete prior OSS promotion exists: {incoming}")
    shutil.copytree(source_dataset, incoming, symlinks=True)
    _rewrite_tree(incoming, source_root, destination_root)
    incoming.replace(destination_dataset)
    destination_indexes = [_copy_index(index, source_root, destination_root) for index in indexes]
    result = {
        "status": "complete",
        "dataset": dataset,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "indexes": [str(path) for path in destination_indexes],
        "window_count": _validate(destination_indexes),
    }
    _atomic_json(marker, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--prepare-destination", action="store_true",
                        help="verify live ossfs2 and create --destination-root; performs no copy")
    args = parser.parse_args()
    destination = prepare_oss_destination(args.destination_root)
    if args.prepare_destination:
        if args.source_root is not None or args.dataset:
            raise ValueError("--prepare-destination cannot be combined with --source-root or --dataset")
        print(json.dumps({"status": "ready", "destination_root": str(destination)}), flush=True)
        return
    if args.source_root is None or not args.dataset:
        raise ValueError("--source-root and --dataset are required unless --prepare-destination is used")
    for dataset in args.dataset:
        print(json.dumps(promote_dataset(args.source_root, destination, dataset), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
