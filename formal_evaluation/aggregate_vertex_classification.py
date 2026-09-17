#!/usr/bin/env python3
"""Aggregate 778-vertex contact classification from registered overlays and GT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.contact.metrics import compute_contact_metrics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def canonical_key(row: Mapping[str, object], aliases: Mapping[str, str]) -> str:
    value = str(row.get("window_id", row.get("cache_id", "")))
    if not value:
        raise ValueError("index row lacks window_id/cache_id")
    return aliases.get(value, value)


def verify_file(path: Path, expected_sha256: str | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 and sha256(path) != expected_sha256:
        raise ValueError(f"SHA256 mismatch: {path}")


def vertex_metrics(prediction: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]) -> dict[str, object]:
    probability = np.asarray(prediction["vertex_contact_probability"], dtype=np.float32)
    labels = np.asarray(target["vertex_contact_target"], dtype=np.float32)
    mask = np.asarray(target["vertex_contact_mask"], dtype=bool)
    hand_valid = np.asarray(prediction["hand_valid"], dtype=bool)
    if probability.shape != labels.shape or probability.shape != mask.shape:
        raise ValueError(f"vertex shape mismatch: {probability.shape}/{labels.shape}/{mask.shape}")
    if probability.ndim != 3 or probability.shape[1:] != (2, 778):
        raise ValueError(f"expected [T,2,778], got {probability.shape}")
    if hand_valid.shape != probability.shape[:2]:
        raise ValueError(f"hand_valid shape mismatch: {hand_valid.shape}")
    mask &= hand_valid[..., None]
    metrics = compute_contact_metrics(probability, labels, mask)
    return {
        f"vertex_contact_{name}": np.asarray(
            value["ap"] if name == "average_precision" and isinstance(value, dict) else value
        ).item()
        for name, value in metrics.items()
    }


def run(spec: Mapping[str, object], output_root: Path) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=False)
    source_index = Path(str(spec["prediction_index"]))
    target_index = Path(str(spec["target_index"]))
    for sentinel in spec.get("required_complete", []):
        verify_file(Path(str(sentinel)))
    verify_file(source_index, str(spec.get("prediction_index_sha256") or "") or None)
    verify_file(target_index, str(spec.get("target_index_sha256") or "") or None)

    source_rows = rows(source_index)
    target_rows = rows(target_index)
    target_by_dataset: dict[str, dict[str, dict[str, object]]] = {}
    for dataset in spec["datasets"]:
        selected = [row for row in target_rows if row.get("dataset") == dataset]
        keyed = {canonical_key(row, {}): row for row in selected}
        if len(keyed) != len(selected):
            raise ValueError(f"duplicate target window: {dataset}")
        target_by_dataset[str(dataset)] = keyed

    expected_total = int(spec["expected_windows"])
    if len(source_rows) != expected_total:
        raise ValueError(f"prediction count mismatch: {len(source_rows)}/{expected_total}")
    report: dict[str, object] = {
        "status": "complete",
        "method": str(spec["method"]),
        "protocol": str(spec["protocol"]),
        "windows": 0,
        "datasets": {},
        "schemes": {"unfiltered": {}},
        "prediction_index": str(source_index),
        "prediction_index_sha256": sha256(source_index),
        "target_index": str(target_index),
        "target_index_sha256": sha256(target_index),
    }
    output_rows = []
    for dataset, expected in spec["datasets"].items():
        selected = [row for row in source_rows if row.get("dataset") == dataset]
        if len(selected) != int(expected):
            raise ValueError(f"dataset count mismatch: {dataset} {len(selected)}/{expected}")
        targets = target_by_dataset[str(dataset)]
        values = []
        seen = set()
        for row in selected:
            key = str(row["window_id"])
            if key in seen or key not in targets:
                raise ValueError(f"prediction/target identity mismatch: {dataset}/{key}")
            seen.add(key)
            target_row = targets[key]
            if row["frame_ids"] != target_row["frame_ids"]:
                raise ValueError(f"frame identity mismatch: {dataset}/{key}")
            prediction = load_arrays(Path(str(row["array_path"])))
            target = load_arrays(Path(str(target_row["array_path"])))
            metrics = vertex_metrics(prediction, target)
            values.append({"window_id": key, **metrics})
            output_rows.append({
                "dataset": dataset,
                "window_id": key,
                "frame_ids": row["frame_ids"],
                "metrics": metrics,
            })
        aggregate = aggregate_windows(values, method=str(spec["method"]))
        report["schemes"]["unfiltered"][dataset] = aggregate
        report["datasets"][dataset] = {"windows": len(values)}
        report["windows"] += len(values)
        print(json.dumps({"dataset": dataset, "windows": len(values)}), flush=True)
    if report["windows"] != expected_total or len(output_rows) != expected_total:
        raise ValueError("total coverage mismatch")

    index_path = output_root / "index.jsonl"
    report_path = output_root / "report.json"
    index_path.write_text("".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "method": str(spec["method"]),
        "protocol": str(spec["protocol"]),
        "windows": expected_total,
        "datasets": dict(spec["datasets"]),
        "index_sha256": sha256(index_path),
        "report_sha256": sha256(report_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.spec.read_text(encoding="utf-8")), args.output_root)))


if __name__ == "__main__":
    main()
