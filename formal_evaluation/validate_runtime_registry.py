#!/usr/bin/env python3
"""Validate exact DSW baseline paths from the runtime registry; never search for alternatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_REGISTRY = Path(__file__).parent / "config" / "baseline_runtime_registry_dsw.json"


def inspect(item: dict) -> dict:
    path = Path(item["path"])
    expected = item.get("state", "present")
    exists = path.exists()
    result = {"role": item.get("role", "unnamed"), "path": str(path), "expected": expected, "exists": exists}
    if exists and path.is_file():
        result["bytes"] = path.stat().st_size
        if "bytes" in item:
            result["size_matches"] = result["bytes"] == item["bytes"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--method", action="append", dest="methods", help="Method key to check; repeatable. Defaults to all.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero for required missing paths or size mismatches.")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text())
    known = registry["methods"]
    selected = args.methods or registry["method_order"]
    unknown = [method for method in selected if method not in known]
    if unknown:
        parser.error("unknown method(s): " + ", ".join(unknown))

    shared_entries = [inspect(item) for item in registry.get("shared_required_paths", [])]
    report = {"registry": str(args.registry), "shared_paths": shared_entries, "methods": {}}
    failures = []
    for entry in shared_entries:
        required = entry["expected"] == "present"
        invalid_size = entry.get("size_matches") is False
        if required and (not entry["exists"] or invalid_size):
            failures.append(f"shared: {entry['path']}")
    for method in selected:
        spec = known[method]
        required = list(spec.get("required_paths", []))
        for key in ("source_root", "python", "conda_executable"):
            if spec.get(key):
                required.append({"role": key, "path": spec[key], "state": "present"})
        for mapping in spec.get("path_mappings_required", []):
            required.append({"role": "mapping canonical", "path": mapping["canonical_path"], "state": "present"})
            required.append({"role": "mapping consumer", "path": mapping["consumer_path"], "state": mapping["state"]})
        entries = [inspect(item) for item in required]
        report["methods"][method] = {
            "execution_status": known[method]["execution_status"],
            "paths": entries,
        }
        for entry in entries:
            required = entry["expected"] == "present"
            invalid_size = entry.get("size_matches") is False
            if required and (not entry["exists"] or invalid_size):
                failures.append(f"{method}: {entry['path']}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and failures:
        print("\nStrict validation failed for:\n" + "\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
