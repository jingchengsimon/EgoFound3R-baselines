#!/usr/bin/env python3
"""Audit existing canonical inputs and recompute selected metrics for six datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    reports: dict[str, str] = {}
    for dataset, item in spec["datasets"].items():
        dataset_root = args.output_root / dataset
        command = [
            sys.executable, str(args.runtime / "formal_evaluation/run_metrics_v2_aggregation.py"),
            "--dataset", dataset,
            "--expected-windows", str(item["expected_windows"]),
            "--main-root", item["main_root"],
            "--gt-index", item["gt_index"],
            "--worktree", str(args.worktree),
            "--work-dir", str(dataset_root / "work"),
            "--report-path", str(dataset_root / "report.json"),
            "--audit-path", str(dataset_root / "input_audit.json"),
        ]
        for method in spec["methods"]:
            command.extend(("--method", method))
        if spec.get("scene_only"):
            command.append("--scene-only")
        for method, root in item.get("history_roots", {}).items():
            if method in spec["methods"]:
                command.extend(("--history-root", f"{method}={root}"))
        subprocess.run(command, check=True)
        reports[dataset] = str(dataset_root / "report.json")
        print(json.dumps({"status": "dataset_complete", "dataset": dataset,
                          "report": reports[dataset]}), flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps({
        "status": "complete", "methods": spec["methods"], "reports": reports,
    }, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
