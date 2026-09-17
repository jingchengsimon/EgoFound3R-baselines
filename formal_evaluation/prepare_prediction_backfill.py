#!/usr/bin/env python3
"""Audit one registered prediction matrix and materialize only its failed windows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

from formal_evaluation.datasets.window_inputs import load_window_input


def _queue_validator():
    """Load the exact queue validator without relying on package search order."""
    path = Path(__file__).with_name("run_formal_window_queue.py")
    spec = importlib.util.spec_from_file_location("registered_formal_queue", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._validated_output


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--input-index", action="append", required=True, type=Path)
    parser.add_argument("--window-count", required=True, type=int)
    parser.add_argument("--prediction-root", action="append", required=True, type=Path)
    parser.add_argument("--expected-failures", required=True, type=int)
    parser.add_argument("--audit-path", required=True, type=Path)
    parser.add_argument("--output-index", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    validate = _queue_validator()
    expected: list[dict[str, object]] = []
    for index in args.input_index:
        for line in index.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                expected.append({**load_window_input(Path(str(row["window_input"]))), "_path": str(row["window_input"])})
                if len(expected) == args.window_count:
                    break
        if len(expected) == args.window_count:
            break
    if len(expected) != args.window_count:
        raise ValueError(f"input indices contain {len(expected)} windows, expected {args.window_count}")

    failures: list[dict[str, object]] = []
    for record in expected:
        candidates = [
            *(root / args.method / "formal" / str(record["cache_id"]) for root in args.prediction_root),
            *(root / args.method / args.method / "formal" / str(record["cache_id"]) for root in args.prediction_root),
        ]
        output = next((path for path in candidates if path.is_dir()), None)
        if output is None:
            failures.append({"cache_id": record["cache_id"], "window_input": record["_path"], "error": "missing output directory"})
            continue
        try:
            validate(output, record, args.method)
        except Exception as error:
            failures.append({"cache_id": record["cache_id"], "window_input": record["_path"], "prediction_dir": str(output), "error": repr(error)})

    audit = {
        "dataset": args.dataset,
        "method": args.method,
        "expected_window_count": len(expected),
        "expected_failure_count": args.expected_failures,
        "observed_failure_count": len(failures),
        "status": "exact_failure_count" if len(failures) == args.expected_failures else "failure_count_mismatch",
        "failures": failures,
    }
    _atomic_json(args.audit_path, audit)
    if args.audit_only:
        return
    if len(failures) != args.expected_failures:
        raise SystemExit(f"{args.dataset}/{args.method}: observed {len(failures)} failures, expected {args.expected_failures}")
    if args.output_index.exists():
        raise FileExistsError(f"refusing to overwrite backfill index: {args.output_index}")
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.write_text("".join(json.dumps({"window_input": row["window_input"]}, sort_keys=True) + "\n" for row in failures), encoding="utf-8")
    _atomic_json(args.output_index.with_suffix(".status.json"), {
        "status": "complete", "index": str(args.output_index), "window_count": len(failures),
        "source_audit": str(args.audit_path),
    })


if __name__ == "__main__":
    main()
