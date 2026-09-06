"""Validate one ablation/dataset window with the existing inference and metrics."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from formal_evaluation.run_egofound3r_stride_evaluation import prepare_inputs, read_rows


def run(spec, output):
    if len(spec["datasets"]) != 1:
        raise ValueError("ablation smoke requires exactly one dataset")
    if not os.environ.get("TASKCTL_GPU", "").isdigit():
        raise ValueError("ablation smoke requires one assigned physical GPU")
    output.mkdir(parents=True, exist_ok=False)
    dataset, data = next(iter(spec["datasets"].items()))
    index, gt, records = prepare_inputs(dataset, data, output / "inputs")
    root = Path(__file__).resolve().parent
    prediction_root = output / "prediction_smoke"
    command = [sys.executable, str(root / "run_egofound3r_smoke.py"),
               "--input-index", str(index), "--model-python", sys.executable,
               "--runner", str(root / "scene/adapters/run_egofound3r_baseline.py"),
               "--methods-config", spec["methods_config"], "--config", spec["config"],
               "--checkpoint", spec["checkpoint"], "--backbone-checkpoint", spec["backbone"],
               "--output-root", str(prediction_root), "--global-stride", "5"]
    for key in ("source_commit", "inference_commit", "checkpoint_sha256"):
        command.extend(["--" + key.replace("_", "-"), spec[key]])
    subprocess.run(command, check=True)
    summary = json.loads((prediction_root / "smoke_summary.json").read_text())
    if not (prediction_root / "COMPLETE").is_file() or summary.get("status") != "complete":
        raise ValueError("inference smoke did not complete")
    first = records[0]
    selected_gt = [row for row in read_rows(gt) if str(row["window_id"]) == str(first["window_id"])]
    if len(selected_gt) != 1:
        raise ValueError("smoke GT window identity is not unique")
    gt_index = output / "gt_smoke.jsonl"
    gt_index.write_text(json.dumps(selected_gt[0]) + "\n")
    prediction_index = output / "predictions_smoke.jsonl"
    prediction_index.write_text(json.dumps({"method": "egofound3r", "dataset": dataset,
        "window_id": first["window_id"], "prediction_dir": summary["output"]}) + "\n")
    report = output / "metric_report.json"
    subprocess.run([sys.executable, str(root / "evaluate_six_dataset.py"),
                    "--gt-index", str(gt_index), "--prediction-index", str(prediction_index),
                    "--methods-config", spec["methods_config"], "--report-path", str(report)], check=True)
    metrics = json.loads(report.read_text())
    method = metrics["methods"]["egofound3r"]
    if (metrics.get("gt_windows") != 1 or method.get("missing_prediction_windows") != 0
            or method.get("datasets", {}).get(dataset, {}).get("n_windows") != 1):
        raise ValueError("smoke metric coverage is incomplete")
    summary.update(dataset=dataset, window_id=first["window_id"], metric_report=str(report),
                   metric_validation="complete")
    (output / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()), args.output_root)
