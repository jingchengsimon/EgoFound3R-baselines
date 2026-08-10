#!/usr/bin/env python3
"""Persistently submit the pending contact formal evaluation when a GPU is free.

Run this program under ``nohup`` on 5000.  It checks 5000/5001/6001 every
30 minutes and starts the S²Contact + ContactOpt 2579-window pipeline on the
first eligible device.  A shared ``flock`` prevents duplicate submissions.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import PurePosixPath


HOST = "39.106.218.186"
WORKTREE = PurePosixPath("/mnt/workspace/sjc/EgoFound3R-baselines-formal-evaluation")
OUTPUT = WORKTREE / "outputs/formal_evaluation_validation"
LOCK = OUTPUT / ".contact_formal_2579.lock"
PYTHON = PurePosixPath("/mnt/workspace/sjc/envs/contactopt/bin/python")
FRAME_INDEX = PurePosixPath("/mnt/workspace/sjc/DATA/H2O/h2o_contact_baseline/frame_index.jsonl")


@dataclass(frozen=True)
class Node:
    port: int
    allowed_gpus: tuple[int, ...]


# 5000 GPU0--6 are reserved by the user for a separate task.
NODES = (Node(5000, (7,)), Node(5001, tuple(range(8))), Node(6001, tuple(range(8))))


def _ssh(port: int, command: str, *, timeout: int = 45) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "root@" + HOST, "-p", str(port), command],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _completed() -> bool:
    command = f"test -s {OUTPUT / 'contact_s2contact_formal_2579.new.json'} && test -s {OUTPUT / 'contact_contactopt_formal_2579.new.json'}"
    return _ssh(5000, command).returncode == 0


def _available_gpus(node: Node, *, minimum_free_mb: int) -> list[int]:
    result = _ssh(
        node.port,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits",
    )
    if result.returncode:
        return []
    available = []
    for line in result.stdout.splitlines():
        try:
            index, utilization, free = (int(value.strip()) for value in line.split(","))
        except ValueError:
            continue
        if index in node.allowed_gpus and utilization <= 5 and free >= minimum_free_mb:
            available.append(index)
    return available


def _pipeline(gpu: int) -> str:
    manifest = OUTPUT / "h2o_formal_manifest_reconstructed.json"
    common = (
        f"--manifest {FRAME_INDEX} --methods-config {WORKTREE / 'formal_evaluation/config/methods_v1.json'} "
        f"--output-root {OUTPUT} --batch-size 32 --device cuda:0"
    )
    evaluate = (
        f"--data-root /mnt/workspace/sjc/DATA/H2O/h2o_data --mano-dir /mnt/workspace/sjc/models/human/mano "
        f"--manifest {manifest} --methods-config {WORKTREE / 'formal_evaluation/config/methods_v1.json'} "
        f"--output-root {OUTPUT} --phase formal"
    )
    return " ; ".join(
        (
            "set -euo pipefail",
            f"export CUDA_VISIBLE_DEVICES={gpu}",
            f"cd {WORKTREE}",
            f"{PYTHON} -m formal_evaluation.contact.adapters.run_s2_contactopt --baseline s2contact "
            f"--source-root /mnt/workspace/sjc/EgoFound3R-baselines/contact_src/S2Contact "
            f"--cache /mnt/workspace/sjc/DATA/H2O/h2o_contact_baseline/cache/s2_right_h2o_30724.pkl "
            f"--checkpoint /mnt/workspace/sjc/EgoFound3R-baselines/contact_src/S2Contact/checkpoints/20211027-212322.pt "
            f"{common} --materialized-manifest {manifest}",
            f"{PYTHON} -m formal_evaluation.evaluate {evaluate} "
            f"--report-path {OUTPUT / 'contact_s2contact_formal_2579.new.json'} --methods s2contact",
            f"{PYTHON} -m formal_evaluation.contact.adapters.run_s2_contactopt --baseline contactopt "
            f"--source-root /mnt/workspace/sjc/EgoFound3R-baselines/contact_src/ContactOpt "
            f"--cache /mnt/workspace/sjc/DATA/H2O/h2o_contact_baseline/cache/contactopt_right_h2o_30724.pkl "
            f"--checkpoint /mnt/workspace/sjc/EgoFound3R-baselines/contact_src/ContactOpt/checkpoints/deepcontact_checkpoint.pt "
            f"{common}",
            f"{PYTHON} -m formal_evaluation.evaluate {evaluate} "
            f"--report-path {OUTPUT / 'contact_contactopt_formal_2579.new.json'} --methods contactopt",
        )
    )


def _submit(node: Node, gpu: int) -> dict[str, object]:
    log = OUTPUT / "logs" / f"contact_formal_monitor_{node.port}_gpu{gpu}.log"
    remote = (
        f"mkdir -p {OUTPUT / 'logs'} && cd {WORKTREE} && "
        f"nohup flock -n {LOCK} bash -lc {shlex.quote(_pipeline(gpu))} "
        f"> {log} 2>&1 < /dev/null & echo $!"
    )
    result = _ssh(node.port, remote)
    return {"node": node.port, "gpu": gpu, "returncode": result.returncode, "pid": result.stdout.strip(), "stderr": result.stderr.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--minimum-free-mb", type=int, default=60000)
    args = parser.parse_args()
    while True:
        if _completed():
            print(json.dumps({"status": "completed"}), flush=True)
            return
        for node in NODES:
            gpus = _available_gpus(node, minimum_free_mb=args.minimum_free_mb)
            if gpus:
                print(json.dumps({"status": "submission_attempt", **_submit(node, gpus[0])}), flush=True)
                break
        else:
            print(json.dumps({"status": "no_eligible_gpu"}), flush=True)
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
