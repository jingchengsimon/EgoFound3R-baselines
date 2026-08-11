#!/usr/bin/env python3
"""Run resumable PAD-Hand compatibility inference for every formal H2O window."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--preparer", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--conda-executable", type=Path, required=True)
    parser.add_argument("--environment-python", type=Path, required=True)
    parser.add_argument("--pad-env", default="pad_hand_h20")
    parser.add_argument("--wilor-env", default="pad_hand_h20")
    parser.add_argument("--max-windows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    windows = [(entry["sequence"], window["window_id"])
               for entry in manifest["formal_test"]["sequences"] for window in entry["windows"]]
    if args.max_windows is not None:
        windows = windows[:args.max_windows]
    if not windows:
        raise ValueError("formal manifest has no selected windows")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.prepared_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "status.jsonl"
    print(json.dumps({"windows": len(windows), "status": str(status_path)}), flush=True)
    for index, (sequence, window_id) in enumerate(windows, start=1):
        result = args.output_root / "pad_hand" / "formal" / window_id
        run_json = result / "run.json"
        if run_json.is_file() and json.loads(run_json.read_text(encoding="utf-8")).get("status") == "success":
            print(json.dumps({"index": index, "window_id": window_id, "status": "skipped"}), flush=True)
            continue
        prepared = args.prepared_root / window_id
        try:
            if not (prepared / "mapping.json").is_file():
                subprocess.run([
                    str(args.environment_python), str(args.preparer), "--phase", "formal",
                    "--manifest", str(args.manifest), "--data-root", str(args.data_root),
                    "--output-dir", str(prepared), "--sequence", sequence, "--window-id", window_id,
                ], check=True)
            subprocess.run([
                str(args.environment_python), str(args.runner), "--phase", "formal",
                "--manifest", str(args.manifest), "--methods-config", str(args.methods_config),
                "--prepared-dir", str(prepared), "--source-root", str(args.source_root),
                "--conda-executable", str(args.conda_executable), "--pad-env", args.pad_env,
                "--wilor-env", args.wilor_env, "--output-root", str(args.output_root),
                "--sequence", sequence, "--window-id", window_id,
            ], check=True)
        except subprocess.CalledProcessError as error:
            record = {"index": index, "window_id": window_id, "status": "failed", "returncode": error.returncode}
            with status_path.open("a", encoding="utf-8") as status_file:
                status_file.write(json.dumps(record) + "\n")
            raise
        record = {"index": index, "window_id": window_id, "status": "success"}
        with status_path.open("a", encoding="utf-8") as status_file:
            status_file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
