#!/usr/bin/env python3
"""Dispatch ready dataset baselines without waiting for unrelated datasets.

Run one controller per node.  It consumes only complete input-shard sentinels,
claims each (dataset, method) once, and leaves a durable JSON record for every
launch.  ``--watch-seconds`` turns the otherwise single scheduling pass into a
small, restart-safe supervisor.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


EGO_PYTHON = "/mnt/workspace/sjc/envs/egofound3r/bin/python"
METHODS_CONFIG = "formal_evaluation/config/methods_v1.json"
MODELS = Path("/mnt/workspace/sjc/models/pretrained")
BASELINES = Path("/mnt/workspace/sjc/EgoFound3R-baselines")


def _direct_specs(repo: Path) -> dict[str, tuple[str, str, list[str], dict[str, str]]]:
    """Return runner python, runner path, exact runner args, and extra env."""
    return {
        "egofound3r": (
            EGO_PYTHON, "formal_evaluation/scene/adapters/run_egofound3r_baseline.py",
            ["--config", "/mnt/workspace/sjc/EgoFound3R_dev_dataloader_shm_compat_20260808/configs/h2o_contact_or_conf_flow_workers4_step006099_100steps_20260809.toml",
             "--checkpoint", "/mnt/workspace/sjc/EgoFound3R_dev/outputs/experiments/step_006099_newflow.pt",
             "--backbone-checkpoint", str(MODELS / "VGGT-Omega/vggt_omega_1b_512.pt")],
            {"PYTHONPATH": os.pathsep.join((str(repo), "/mnt/workspace/sjc/EgoFound3R_dev_dataloader_shm_compat_20260808"))}),
        "wilor": (EGO_PYTHON, "formal_evaluation/hand/adapters/run_wilor_baseline.py",
                  ["--source-root", str(BASELINES / "WiLoR"), "--checkpoint", str(MODELS / "WiLoR/wilor_final.ckpt"),
                   "--detector", str(MODELS / "WiLoR/detector.pt")], {}),
        "hawor": (EGO_PYTHON, "formal_evaluation/hand/adapters/run_hawor_baseline.py",
                  ["--source-root", str(BASELINES / "HaWoR"), "--checkpoint", str(MODELS / "HaWoR/hawor/checkpoints/hawor.ckpt"),
                   "--infiller-weight", str(MODELS / "HaWoR/hawor/checkpoints/infiller.pt"),
                   "--detector-weight", str(MODELS / "HaWoR/external/detector.pt")], {}),
        "pad_hand": ("/mnt/workspace/sjc/miniconda3/envs/pad_hand_h20/bin/python", "formal_evaluation/hand/adapters/run_pad_hand_baseline.py",
                     ["--source-root", str(BASELINES / "PAD-Hand"), "--wilor-python", EGO_PYTHON,
                      "--checkpoint", str(BASELINES / "PAD-Hand/checkpoints/pad_hand.pt")], {}),
        "reviv4d": ("/mnt/workspace/sjc/miniconda3/envs/reviv4d_eval_py312_20260818/bin/python", "formal_evaluation/scene/adapters/run_reviv4d_baseline.py",
                    ["--source-root", "/mnt/workspace/sjc/external/reviv4d", "--python", "/mnt/workspace/sjc/miniconda3/envs/reviv4d_eval_py312_20260818/bin/python",
                     "--checkpoint-root", "/mnt/workspace/sjc/external/reviv4d/reviv_checkpoints/metric_depth",
                     "--cosmos-dir", "Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16"], {}),
        "vggt": (EGO_PYTHON, "formal_evaluation/scene/adapters/run_scene_baseline.py",
                 ["--method", "vggt", "--source-root", str(BASELINES / "VGGT"), "--checkpoint", str(MODELS / "VGGT-1B/model.pt")], {}),
        "pi3": (EGO_PYTHON, "formal_evaluation/scene/adapters/run_scene_baseline.py",
                ["--method", "pi3", "--source-root", str(BASELINES / "Pi3"), "--checkpoint", str(MODELS / "Pi3/model.safetensors")], {}),
        "da3_large_1_1": (EGO_PYTHON, "formal_evaluation/scene/adapters/run_scene_baseline.py",
                           ["--method", "da3_large_1_1", "--source-root", str(BASELINES / "Depth-Anything-3"),
                            "--checkpoint", str(MODELS / "Depth-Anything-3/DA3-LARGE-1.1/model.safetensors")], {}),
        "lingbot_map_long": ("/mnt/workspace/sjc/envs/lingbot_map/bin/python", "formal_evaluation/scene/adapters/run_scene_baseline.py",
                              ["--method", "lingbot_map_long", "--source-root", str(BASELINES / "lingbot-map"),
                               "--checkpoint", str(MODELS / "LingBot-Map/lingbot-map-long.pt")], {}),
        "vggt_omega": (EGO_PYTHON, "formal_evaluation/scene/adapters/run_scene_baseline.py",
                        ["--method", "vggt_omega", "--source-root", str(BASELINES / "vggt-omega"),
                         "--checkpoint", str(MODELS / "VGGT-Omega/vggt_omega_1b_512.pt")], {}),
    }


def _atomic_json(path: Path, payload: dict[str, object], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if replace else "x"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ready_indices(input_root: Path, dataset: str) -> list[Path]:
    statuses = []
    for path in sorted(input_root.glob(f"window_inputs_{dataset}_shard_*.status.json")):
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            return []
        index = Path(str(status.get("index", "")))
        if not index.is_file() or index.with_suffix(".status.json") != path:
            return []
        statuses.append((int(status["shard_index"]), int(status["shard_count"]), index))
    if not statuses:
        return []
    count = statuses[0][1]
    if any(shard_count != count for _, shard_count, _ in statuses):
        return []
    if [shard for shard, _, _ in statuses] != list(range(count)):
        return []
    return [index for _, _, index in statuses]


def _gpu_free(gpu: str, threshold_mib: int) -> bool:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    usage = {line.split(",", 1)[0].strip(): int(line.split(",", 1)[1].strip()) for line in output.splitlines() if "," in line}
    return usage.get(gpu, threshold_mib + 1) <= threshold_mib


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _claim_path(state_root: Path, name: str) -> Path:
    return state_root / "claims" / f"{name}.json"


def _active_claim(state_root: Path, gpu: str) -> bool:
    for path in (state_root / "claims").glob("*.json"):
        claim = json.loads(path.read_text(encoding="utf-8"))
        if str(claim.get("gpu")) == gpu and claim.get("status") == "running" and _alive(int(claim["pid"])):
            return True
    return False


def _claim(state_root: Path, name: str, payload: dict[str, object]) -> bool:
    try:
        _atomic_json(_claim_path(state_root, name), payload)
    except FileExistsError:
        return False
    return True


def _start_direct(args: argparse.Namespace, dataset: str, method: str, gpu: str, indices: list[Path]) -> bool:
    claim_name = f"direct__{dataset}__{method}"
    output_root = args.output_parent / dataset / method
    queue_root = args.state_root / "queues" / dataset / method
    if _claim_path(args.state_root, claim_name).exists() or output_root.exists() or queue_root.exists():
        return False
    runner_python, runner, extra, extra_env = _direct_specs(args.repo)[method]
    command = [args.driver_python, str(args.repo / "formal_evaluation/run_formal_window_queue.py"),
               "--method", method, "--python", runner_python, "--runner", str(args.repo / runner),
               "--runner-arg=--phase", "--runner-arg=formal", "--runner-arg=--window-input", "--runner-arg={window_input}",
               "--runner-arg=--methods-config", f"--runner-arg={METHODS_CONFIG}",
               "--runner-arg=--output-root", "--runner-arg={output_root}"]
    for key, value in zip(extra[::2], extra[1::2], strict=True):
        command.extend((f"--runner-arg={key}", f"--runner-arg={value}"))
    for index in indices:
        command.extend(("--input-index", str(index)))
    command.extend(("--output-root", str(output_root), "--queue-root", str(queue_root), "--gpus", gpu))
    env = {**os.environ, **extra_env, "PYTHONUNBUFFERED": "1"}
    if "PYTHONPATH" not in env:
        env["PYTHONPATH"] = str(args.repo)
    log = args.state_root / "logs" / f"direct__{dataset}__{method}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"would_launch": command, "gpu": gpu}))
        return True
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=args.repo, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    if not _claim(args.state_root, claim_name, {"kind": "direct", "dataset": dataset, "method": method, "gpu": gpu,
                                                  "pid": process.pid, "status": "running", "output_root": str(output_root),
                                                  "queue_root": str(queue_root), "log": str(log)}):
        raise RuntimeError("claim race after process launch; preserve process and inspect its log")
    print(json.dumps({"launched": claim_name, "gpu": gpu, "pid": process.pid}), flush=True)
    return True


def _start_interactvlm(args: argparse.Namespace, dataset: str, gpu: str, indices: list[Path]) -> bool:
    claim_name = f"interactvlm__{dataset}"
    lane = args.state_root / "interactvlm" / dataset
    manifest, work_dir, output_manifest = lane / "inputs.jsonl", lane / "work", lane / "predictions.jsonl"
    if _claim_path(args.state_root, claim_name).exists() or lane.exists():
        return False
    lane.mkdir(parents=True)
    materialize = [args.driver_python, str(args.repo / "formal_evaluation/contact/adapters/materialize_interactvlm_inputs.py")]
    for index in indices:
        materialize.extend(("--window-input-index", str(index)))
    materialize.extend(("--output-manifest", str(manifest)))
    manifest_log = lane / "materialize.log"
    if args.dry_run:
        print(json.dumps({"would_materialize_interactvlm": materialize, "gpu": gpu}))
        return True
    with manifest_log.open("x", encoding="utf-8") as handle:
        subprocess.run(materialize, cwd=args.repo, stdout=handle, stderr=subprocess.STDOUT, check=True)
    command = ["/mnt/workspace/sjc/envs/interactvlm_eval/bin/python", str(args.repo / "formal_evaluation/contact/adapters/run_interactvlm.py"),
               "--source-root", "/mnt/workspace/sjc/external/InteractVLM",
               "--checkpoint", "/mnt/workspace/sjc/models/pretrained/InteractVLM/interactvlm-3d-hcontact-damon",
               "--input-manifest", str(manifest), "--work-dir", str(work_dir), "--output-manifest", str(output_manifest),
               "--python", "/mnt/workspace/sjc/envs/interactvlm_eval/bin/python", "--precision", "bf16",
               "--vision-tower", "/mnt/workspace/sjc/models/pretrained/CLIP/clip-vit-large-patch14"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
    log = lane / "run.log"
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=args.repo, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    if not _claim(args.state_root, claim_name, {"kind": "interactvlm", "dataset": dataset, "method": "interactvlm", "gpu": gpu,
                                                  "pid": process.pid, "status": "running", "input_manifest": str(manifest),
                                                  "output_manifest": str(output_manifest), "log": str(log)}):
        raise RuntimeError("claim race after InteractVLM launch; preserve process and inspect its log")
    print(json.dumps({"launched": claim_name, "gpu": gpu, "pid": process.pid, "manifest": str(manifest)}), flush=True)
    return True


def _dispatch_once(args: argparse.Namespace) -> None:
    datasets = _csv(args.datasets)
    ready = {dataset: ready_indices(args.input_root, dataset) for dataset in datasets}
    methods = _csv(args.methods)
    specs = _direct_specs(args.repo)
    unknown = set(methods) - set(specs)
    if unknown:
        raise ValueError(f"unsupported direct method(s): {sorted(unknown)}")
    for gpu in _csv(args.gpus):
        if _active_claim(args.state_root, gpu) or not _gpu_free(gpu, args.gpu_memory_threshold_mib):
            continue
        launched = False
        for dataset in datasets:
            for method in methods:
                if ready[dataset] and _start_direct(args, dataset, method, gpu, ready[dataset]):
                    launched = True
                    break
            if launched:
                break
    if args.ivlm_gpu:
        gpu = args.ivlm_gpu
        ivlm_datasets = _csv(args.ivlm_datasets or args.datasets)
        if not _active_claim(args.state_root, gpu) and _gpu_free(gpu, args.gpu_memory_threshold_mib):
            for dataset in ivlm_datasets:
                indices = ready_indices(args.input_root, dataset)
                if indices and _start_interactvlm(args, dataset, gpu, indices):
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--datasets", required=True, help="priority-ordered ready datasets")
    parser.add_argument("--methods", required=True, help="priority-ordered direct methods owned by this node")
    parser.add_argument("--gpus", required=True, help="physical GPUs owned by this controller")
    parser.add_argument("--driver-python", default=EGO_PYTHON)
    parser.add_argument("--ivlm-gpu", help="one dedicated physical GPU for serial InteractVLM")
    parser.add_argument("--ivlm-datasets", help="priority order; defaults to --datasets")
    parser.add_argument("--gpu-memory-threshold-mib", type=int, default=64)
    parser.add_argument("--watch-seconds", type=float, default=0, help="0 runs one safe pass; positive repeats")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.repo.is_dir() or not args.input_root.is_dir() or not Path(args.driver_python).is_file():
        raise FileNotFoundError("--repo, --input-root, or --driver-python missing")
    args.state_root.mkdir(parents=True, exist_ok=True)
    (args.state_root / "claims").mkdir(exist_ok=True)
    while True:
        _dispatch_once(args)
        if args.watch_seconds <= 0:
            return
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
