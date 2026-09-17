#!/usr/bin/env python3
"""Validate existing formal predictions and write queue-style summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_evaluation.run_formal_window_queue import _validated_output
from formal_evaluation.datasets.window_inputs import load_window_input


def rows(indices: list[Path], limit: int) -> list[dict[str, object]]:
    result = []
    for index in indices:
        for line in index.read_text(encoding="utf-8").splitlines():
            if line:
                result.append(load_window_input(Path(json.loads(line)["window_input"])))
                if len(result) == limit:
                    return result
    raise ValueError(f"input indices contain only {len(result)} windows, expected {limit}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--input-index", action="append", required=True, type=Path)
    parser.add_argument("--prediction-root", action="append", required=True, type=Path)
    parser.add_argument("--window-count", required=True, type=int)
    parser.add_argument("--summary-path", required=True, type=Path)
    parser.add_argument("--prediction-index", required=True, type=Path)
    args = parser.parse_args()

    expected = rows(args.input_index, args.window_count)
    valid, failures, entries = 0, [], []
    for record in expected:
        candidates = (
            root / args.method / "formal" / str(record["cache_id"])
            for root in args.prediction_root
        )
        nested = (
            root / args.method / args.method / "formal" / str(record["cache_id"])
            for root in args.prediction_root
        )
        output = next((path for path in (*candidates, *nested) if path.is_dir()), None)
        if output is None:
            failures.append({"cache_id": record["cache_id"], "error": "missing output directory"})
            continue
        try:
            _validated_output(output, record, args.method)
        except Exception as error:
            failures.append({"cache_id": record["cache_id"], "error": repr(error), "prediction_dir": str(output)})
            continue
        valid += 1
        entries.append({"method": args.method, "dataset": args.dataset,
                        "window_id": record["window_id"], "prediction_dir": str(output)})

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps({
        "dataset": args.dataset, "method": args.method,
        "status": "complete" if not failures else "completed_with_failures",
        "expected_window_count": len(expected), "validated_success_count": valid,
        "failures": failures,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.prediction_index.parent.mkdir(parents=True, exist_ok=True)
    with args.prediction_index.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    if failures:
        raise SystemExit(f"{args.dataset}/{args.method}: {len(failures)} invalid windows")


if __name__ == "__main__":
    main()
