"""Run true GT-K multiclip inference for the frozen final 13 segments."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
import uuid

import torch


def segment_id(entry: dict) -> str:
    sequence = str(entry.get("sequence_id") or entry.get("window_id") or "unknown")
    key = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:10]
    frames = entry["frame_ids"]
    return f"{entry['dataset']}__{key}__{frames[0]}-{frames[-1]}"


def replace_option(argv: list[str], name: str, value: str | None) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == name:
            index += 2
        else:
            result.append(argv[index])
            index += 1
    if value is not None:
        result.extend((name, value))
    return result


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def validate_output(target: Path, source_pred: Path) -> dict:
    summary = json.loads((target / "summary.json").read_text())
    if summary.get("root_k_source") != "gt" or not summary.get("gt_assisted_inference"):
        raise ValueError(f"{target}: not a true GT-K inference")
    current = torch.load(target / "inference_output.pt", map_location="cpu", weights_only=False)
    pred = torch.load(source_pred / "inference_output.pt", map_location="cpu", weights_only=False)
    current_xyz = current["metric_predictions"]["hand"]["vertex_xyz_metric"]
    pred_xyz = pred["metric_predictions"]["hand"]["vertex_xyz_metric"]
    if current_xyz.shape != pred_xyz.shape or torch.equal(current_xyz, pred_xyz):
        raise ValueError(f"{target}: GT-K geometry is identical to pred-K geometry")
    delta = torch.nan_to_num((current_xyz.float() - pred_xyz.float()).abs()).max().item()
    return {"frames": int(current_xyz.shape[0]), "max_abs_geometry_delta_m": float(delta),
            "root_k_source": "gt", "gt_assisted_inference": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--only-file", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ego-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpus", required=True,
                        help="comma-separated physical GPU indices selected by taskctl")
    args = parser.parse_args()

    wanted = set(args.only_file.read_text().split())
    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    chosen = [(segment_id(entry), entry) for entry in entries if segment_id(entry) in wanted]
    if len(chosen) != len(wanted) or {item[0] for item in chosen} != wanted:
        raise ValueError(f"selected manifest coverage {len(chosen)} != {len(wanted)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "progress.jsonl"
    progress_lock = threading.Lock()
    gpu_queue: queue.Queue[int] = queue.Queue()
    gpus = [int(value) for value in args.gpus.split(",") if value]
    if not gpus:
        raise ValueError("at least one GPU is required")
    for gpu in gpus:
        gpu_queue.put(gpu)

    def emit(record: dict) -> None:
        with progress_lock, progress.open("a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def process(item: tuple[str, dict]) -> dict:
        sid, entry = item
        dataset = entry["dataset"]
        source = args.pred_root / dataset / sid
        target = args.output_root / dataset / sid
        if (target / "summary.json").is_file() and (target / "viz_inputs").is_dir():
            result = validate_output(target, source)
            return dict(segment_id=sid, dataset=dataset, status="skipped_valid", **result)
        gpu = gpu_queue.get()
        started = time.time()
        log_path = args.output_root / "_logs" / f"{sid}.log"
        incoming = target.with_name(target.name + ".incoming-" + uuid.uuid4().hex)
        gt_dir = args.output_root / "_gt_sidecars" / dataset / sid
        try:
            source_summary = json.loads((source / "summary.json").read_text())
            provenance = source_summary["provenance"]
            if source_summary.get("root_k_source") != "pred" or provenance.get("git_commit") != "8fc061a615895bd3b5a556f7387bae306e32d9db":
                raise ValueError(f"{sid}: unexpected pred-K source identity")
            gt_path = gt_dir / "ground_truth.pt"
            if not gt_path.is_file():
                run([
                    str(args.python), str(args.ego_repo / "export_marker_inference_gt.py"),
                    "--inference-output", str(source / "inference_output.pt"),
                    "--config", provenance["config_path"], "--output-dir", str(gt_dir),
                    "--dataset", dataset, "--sequence", entry["sequence_id"],
                    "--start-frame-id", str(entry["frame_ids"][0]),
                ], log_path)
            argv = list(provenance["command_argv"][1:])
            if Path(argv[0]).name != "infer_marker_multiclip.py":
                raise ValueError(f"{sid}: unexpected inference command {argv[0]}")
            argv[0] = str(args.ego_repo / "infer_marker_multiclip.py")
            for option in ("--output-dir", "--device", "--root-k-source", "--gt-file"):
                argv = replace_option(argv, option, None)
            argv.extend(("--output-dir", str(incoming), "--device", f"cuda:{gpu}",
                         "--root-k-source", "gt", "--gt-file", str(gt_path)))
            run([str(args.python), *argv], log_path)
            result = validate_output(incoming, source)
            viz = incoming / "viz_inputs" / f"ego_infer_{result['frames']}f_gt_k.npz"
            run([
                str(args.python), str(Path(__file__).with_name("export_ego_viz_inputs.py")),
                "--inference-output", str(incoming / "inference_output.pt"),
                "--require-root-k-source", "gt", "--out", str(viz),
            ], log_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            os.replace(incoming, target)
            record = dict(segment_id=sid, dataset=dataset, status="complete", gpu=gpu,
                          seconds=round(time.time() - started, 2), **result)
            emit(record)
            return record
        except Exception as error:
            emit({"segment_id": sid, "dataset": dataset, "status": "failed",
                  "error": repr(error)})
            if incoming.exists():
                shutil.rmtree(incoming)
            raise
        finally:
            gpu_queue.put(gpu)

    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {pool.submit(process, item): item[0] for item in chosen}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda row: row["segment_id"])
    if len(results) != len(wanted) or any(row["status"] not in {"complete", "skipped_valid"}
                                           for row in results):
        raise SystemExit(2)
    summary = {"status": "complete", "segments": len(results), "results": results}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_root / "COMPLETE").write_text(json.dumps(
        {"segments": len(results), "root_k_source": "gt", "true_model_rerun": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
