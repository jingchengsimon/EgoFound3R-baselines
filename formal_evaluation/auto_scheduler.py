#!/usr/bin/env python3
"""Persistent scheduler for the formal-evaluation (dataset x method) matrix.

Runs locally (not on any DSW node — see note below), SSHing to 5000/5001/6001
to launch/inspect/pause jobs. Scope (2026-08-22 continuation): all six
datasets. Dyn-HaMR and InteractVLM are deliberately excluded. H2O, TACO, and
HOI4D run only the 10 standard baselines after their current materialization
controllers publish every required shard. Hot3D, ARCTIC, and OakInk-v2 also
run S²Contact/ContactOpt, but only after their own 400-window geometry caches
are complete. TACO and Hot3D are capped at 400; every other dataset runs its
complete strict manifest.

Runs off-cluster (e.g. on the operator's own machine) because the 5000/5001/
6001 DSW pods cannot reach each other or even themselves through the shared
external IP/port — only a machine with genuine outbound access to all three
ports can coordinate across nodes.

State is persisted to STATE_PATH (shared CPFS) so the scheduler is safe to
restart. Designed to be started once under nohup and left running.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from formal_evaluation import taskctl as task_registry
except ModuleNotFoundError:  # direct: python formal_evaluation/auto_scheduler.py
    import taskctl as task_registry

HOST = "39.106.218.186"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed_pai")
READ_NODE = 5000  # any node works for shared-CPFS reads; picked arbitrarily
SCHEDULER_ID = "continuation_20260822T112000Z"
SCHEDULER_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".auto_scheduler", SCHEDULER_ID)
STATE_PATH = os.path.join(SCHEDULER_ROOT, "state.json")
LOG_PATH = os.path.join(SCHEDULER_ROOT, "scheduler.log")
CONTROL_PATH = os.path.join(os.path.dirname(SCHEDULER_ROOT), "task_controls.json")
REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "evaluation_task_registry.json")
W = "/mnt/workspace/sjc/EgoFound3R-baselines_formal_7c79a98"
REMOTE_CONTROLLER = f"{W}/formal_evaluation/remote_task_control.py"
METHODS_CONFIG = f"{W}/formal_evaluation/config/methods_v1.json"
PY_EGO = "/mnt/workspace/sjc/envs/egofound3r/bin/python"
PY_LINGBOT = "/mnt/workspace/sjc/envs/lingbot_map/bin/python"
PY_REVIV4D = "/mnt/workspace/sjc/miniconda3/envs/reviv4d_eval_py312_20260818/bin/python"
PY_PADHAND = "/mnt/workspace/sjc/miniconda3/envs/pad_hand_h20/bin/python"
PY_CONTACT = "/mnt/workspace/sjc/envs/contactopt/bin/python"
MANO_RIGHT = "/mnt/workspace/sjc/models/human/mano/MANO_RIGHT.pkl"

# Only TACO and Hot3D have artificial evaluation caps. All other datasets run
# their complete strict manifest (including Arctic=434 and HOI4D=461).
DATASET_WINDOW_CAPS = {"taco": 400, "hot3d": 400}
POLL_SECONDS = 120
MIN_FREE_MB = 80000
NODES = (5000, 5001, 6001)
# GPUs known to host long-lived third-party jobs unrelated to this scheduler;
# never touch these even if nvidia-smi briefly reports them idle.
NODE_ALLOWED_GPUS = {
    5000: (0, 1, 5, 6, 7),  # 2/3/4 host an unattributed persistent job
    5001: tuple(range(8)),
    6001: tuple(range(8)),
}

DATASETS = {
    "h2o": {"strict_count": 283, "output_root": "/mnt/workspace/sjc/eval_artifacts/formal_h2o_60f_20260822T112000Z_continuation"},
    "taco": {"strict_count": 400, "output_root": "/mnt/workspace/sjc/eval_artifacts/formal_taco_60f_20260822T112000Z_continuation"},
    "hoi4d": {"strict_count": 461, "output_root": "/mnt/workspace/sjc/eval_artifacts/formal_hoi4d_60f_20260822T112000Z_continuation"},
    "hot3d": {"strict_count": 1008, "output_root": "/mnt/workspace/sjc/eval_artifacts/formal_hot3d_60f_20260819T203000Z_e6c4a5e_6001"},
    "oakink_v2": {"strict_count": 400, "output_root": "/mnt/workspace/sjc/eval_artifacts/formal_oakink_v2_60f_20260820T0022Z_e6c4a5e"},
    "arctic": {"strict_count": 434, "output_root": "/mnt/workspace/sjc/eval_artifacts/formal_arctic_60f_20260820T0400Z_e6c4a5e"},
}

# window_input shard globs, in canonical forward order, per dataset.
DATASET_SHARDS = {
    "h2o": [f"/mnt/workspace/sjc/eval_artifacts/prep_h2o_60f_strict_20260822T030835Z_5001_g1/window_inputs/window_inputs_h2o_shard_{i:03d}_of_006.jsonl" for i in range(6)],
    "taco": [f"/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_taco_20260822T101500Z_5001_g4_bin0/window_inputs/window_inputs_taco_shard_{i:03d}_of_008.jsonl" for i in range(8)],
    "hoi4d": [f"/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_hoi4d_20260822T015700Z_5001_g0/window_inputs/window_inputs_hoi4d_shard_{i:03d}_of_010.jsonl" for i in range(10)],
    "hot3d": [f"/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_20260818T0410Z_8b1a806/window_inputs/window_inputs_hot3d_shard_{i:03d}_of_021.jsonl" for i in range(21)],
    "oakink_v2": [f"/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_oakink_v2_20260819T144050Z/window_inputs/window_inputs_oakink_v2_shard_{i:03d}_of_008.jsonl" for i in range(8)],
    "arctic": [f"/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_arctic_20260820T000131Z/window_inputs/window_inputs_arctic_shard_{i:03d}_of_009.jsonl" for i in range(9)],
}

STANDARD_METHODS = (
    "egofound3r", "wilor", "hawor", "pad_hand", "reviv4d",
    "vggt", "pi3", "da3_large_1_1", "lingbot_map_long", "vggt_omega",
)
CONTACT_METHODS = ("s2contact", "contactopt")
ALL_METHODS = STANDARD_METHODS + CONTACT_METHODS
CONTACT_DATASETS = ("hot3d", "arctic", "oakink_v2")
CONTACT_CACHE_CAPS = {"hot3d": 400, "arctic": 400, "oakink_v2": 400}
CACHE_WORK_ROOT = f"/mnt/workspace/sjc/eval_artifacts/auto_eval_scheduler/{SCHEDULER_ID}"
HANDLE_ROOT = f"{CACHE_WORK_ROOT}/handles"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    os.makedirs(SCHEDULER_ROOT, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{now()} {msg}\n")


def ssh(port: int, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         "-i", SSH_KEY, "-p", str(port), f"root@{HOST}", command],
        text=True, capture_output=True, timeout=timeout, check=False,
    )


def local(command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Runs on the shared CPFS via any node (this daemon itself runs off-cluster)."""
    return ssh(READ_NODE, command, timeout=timeout)


def load_state() -> dict:
    if os.path.isfile(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"jobs": {}}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    json.dump(state, open(tmp, "w"), indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)


def load_controls() -> dict[str, str]:
    try:
        payload = json.load(open(CONTROL_PATH))
    except FileNotFoundError:
        return {}
    if payload.get("schema_version") != "task_controls_v1" or not isinstance(payload.get("jobs"), dict):
        raise ValueError(f"invalid task controls: {CONTROL_PATH}")
    return {str(key): str(value) for key, value in payload["jobs"].items()}


def job_key(dataset: str, method: str) -> str:
    return f"{dataset}::{method}"


def target_windows(dataset: str) -> int:
    strict_count = DATASETS[dataset]["strict_count"]
    return min(strict_count, DATASET_WINDOW_CAPS.get(dataset, strict_count))


def register_scheduler_runs() -> None:
    """Registration is part of submission; retries reuse the same run IDs."""
    state_file = os.path.relpath(STATE_PATH, os.path.join(os.path.dirname(__file__), ".."))
    for dataset in DATASETS:
        method_sets = ["standard10"] + (["contact2"] if dataset in CONTACT_DATASETS else [])
        for method_set in method_sets:
            suffix = "contact2" if method_set == "contact2" else "standard10"
            methods = CONTACT_METHODS if method_set == "contact2" else STANDARD_METHODS
            logical_task_id = f"formal:{dataset}:{method_set}:60f"
            run_id = f"formal-{dataset}-{suffix}-60f-{SCHEDULER_ID.replace('_', '-')}"
            existing = task_registry.load_registry(Path(REGISTRY_PATH)).get("runs", {}).get(run_id)
            if existing is not None:
                if existing.get("logical_task_id") != logical_task_id:
                    raise task_registry.TaskError(f"RUN_ID_COLLISION:{run_id}")
                continue
            output_root = DATASETS[dataset]["output_root"]
            overrides = {
                method: root for (registered_dataset, method), root in DATASET_METHOD_ROOT_OVERRIDE.items()
                if registered_dataset == dataset and method in methods
            }
            if overrides and len(set(overrides.values())) == 1 and set(overrides) == set(methods):
                output_root = next(iter(overrides.values()))
                overrides = {}
            task_registry.register_run(
                Path(REGISTRY_PATH),
                logical_task_id=logical_task_id,
                dataset=dataset, method_set=method_set, phase="formal", protocol="60f",
                run_id=run_id,
                run_record={
                    "scheduler_id": SCHEDULER_ID,
                    "state_file": state_file,
                    "output_root": output_root,
                    **({"output_root_overrides": overrides} if overrides else {}),
                    "target_windows_per_method": (
                        CONTACT_CACHE_CAPS[dataset] if method_set == "contact2" else target_windows(dataset)
                    ),
                },
            )


# hot3d predates this scheduler and is split across two pre-existing roots:
# reviv4d/wilor/da3_large_1_1 were launched under the 5000 "rootA", the rest
# under the 6001 root that DATASETS["hot3d"]["output_root"] points to.
DATASET_METHOD_ROOT_OVERRIDE = {
    ("hot3d", "reviv4d"): "/mnt/workspace/sjc/eval_artifacts/formal_hot3d_60f_20260819T202019Z_e6c4a5e",
    ("hot3d", "wilor"): "/mnt/workspace/sjc/eval_artifacts/formal_hot3d_60f_20260819T202019Z_e6c4a5e",
    ("hot3d", "da3_large_1_1"): "/mnt/workspace/sjc/eval_artifacts/formal_hot3d_60f_20260819T202019Z_e6c4a5e",
    # The already-running Hot3D cache controller owns these two unique result
    # directories; counting them here prevents a duplicate post-cache launch.
    ("hot3d", "s2contact"): "/mnt/workspace/sjc/eval_artifacts/formal_hot3d_contact_400_20260822T101900Z/results",
    ("hot3d", "contactopt"): "/mnt/workspace/sjc/eval_artifacts/formal_hot3d_contact_400_20260822T101900Z/results",
}

# A single Hot3D PAD-Hand directory was left incomplete by an earlier queue.
# Its recovery is deliberately written outside the original result root so the
# partial directory is never overwritten.  Count this validated, canonical
# recovery output alongside the primary root when reconciling scheduler state.
RECOVERY_METHOD_OUTPUT_ROOTS = {
    ("hot3d", "pad_hand"): (
        "/mnt/workspace/sjc/eval_artifacts/hot3d_pad_hand_single_recovery_20260822T145800Z/pad_hand",
    ),
}


def dataset_method_root(dataset: str, method: str) -> str:
    return DATASET_METHOD_ROOT_OVERRIDE.get((dataset, method), DATASETS[dataset]["output_root"])


def method_output_root(dataset: str, method: str) -> str:
    return f"{dataset_method_root(dataset, method)}/{method}"


def dataset_ready(dataset: str) -> bool:
    """True once enough window_inputs shards are materialized to hit the cap."""
    need = target_windows(dataset)
    shard_count = -(-need // 50)
    statuses = [shard.replace(".jsonl", ".status.json") for shard in DATASET_SHARDS[dataset][:shard_count]]
    # One remote shell avoids an SSH round-trip per 50-window shard.
    command = "for s in " + " ".join(shlex.quote(status) for status in statuses) + "; do cat \"$s\" || exit 1; done"
    try:
        result = local(command, timeout=30)
    except subprocess.TimeoutExpired:
        log(f"materialization-status timeout: {dataset}")
        return False
    if result.returncode != 0:
        return False
    covered = 0
    for line in result.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            return False
        if d.get("status") != "complete":
            return False
        covered += int(d.get("window_count", 0))
        if covered >= need:
            return True
    return covered >= need


def ensure_combined_index(dataset: str, cap: int | None = None) -> str:
    """Builds (remotely, on shared CPFS) a first-N-window combined index file."""
    cap = target_windows(dataset) if cap is None else cap
    out = combined_index_path_remote(dataset, cap)
    r = local(f"wc -l < {shlex.quote(out)} 2>/dev/null")
    try:
        if int((r.stdout or "0").strip()) == cap:
            return out
    except ValueError:
        pass
    shards = DATASET_SHARDS[dataset]
    n_shards_needed = -(-cap // 50) if dataset != "h2o" else len(shards)
    picked = shards[:max(n_shards_needed, 1)]
    cat_cmd = " ".join(shlex.quote(s) for s in picked)
    local(f"mkdir -p {shlex.quote(os.path.dirname(out))} && cat {cat_cmd} | head -n {cap} > {shlex.quote(out)}")
    return out


def combined_index_path_remote(dataset: str, cap: int) -> str:
    return f"/mnt/workspace/sjc/eval_artifacts/auto_eval_scheduler/combined_index/{dataset}_first{cap}.jsonl"


def contact_cache_paths(dataset: str) -> dict[str, str] | None:
    cap = CONTACT_CACHE_CAPS[dataset]
    cache_dir = f"/mnt/workspace/sjc/DATA/{dataset}_contact_baseline/cache"
    paths = {
        "s2contact_cache": f"{cache_dir}/s2_right_{dataset}_{cap}.pkl",
        "s2contact_index": f"{cache_dir}/s2_right_{dataset}_{cap}_index.jsonl",
        "contactopt_cache": f"{cache_dir}/contactopt_right_{dataset}_{cap}.pkl",
        "contactopt_index": f"{cache_dir}/contactopt_right_{dataset}_{cap}_index.jsonl",
    }
    test_cmd = " && ".join(f"test -f {shlex.quote(p)}" for p in paths.values())
    r = local(test_cmd)
    if r.returncode == 0:
        return paths
    return None


def ensure_contact_cache(dataset: str, state: dict, handle_states: dict[str, dict]) -> bool:
    """Accept an exact cache artifact or exact registered handle; never discover processes."""
    if dataset not in CONTACT_DATASETS:
        return False
    if contact_cache_paths(dataset) is not None:
        return True
    cache_job = state.setdefault("cache_jobs", {}).get(dataset)
    if cache_job and cache_job.get("handle_path"):
        observed = handle_states.get(cache_job["handle_path"], {}).get("status")
        cache_job["status"] = observed or "status_unavailable"
        return False
    state.setdefault("cache_jobs", {})[dataset] = {
        "status": "blocked_unregistered_cache_controller",
        "reason": "register an exact handle before control; process discovery is forbidden",
    }
    return False


def standard_shard_args(dataset: str) -> str:
    n_needed = -(-target_windows(dataset) // 50) if dataset != "h2o" else len(DATASET_SHARDS["h2o"])
    shards = DATASET_SHARDS[dataset][:max(n_needed, 1)]
    return " ".join(f"--input-index {s}" for s in shards)


def build_command(dataset: str, method: str, gpu: int) -> tuple[str, str] | None:
    """Return (python_env_label, full shell command) for launching this job, or None if not launchable yet."""
    out_root = dataset_method_root(dataset, method)
    idx_args = standard_shard_args(dataset)
    common_env = f"PYTHONPATH={W}:/mnt/cpfs/sjc/EgoFound3R_final_bf16_inference_8b0c44a_20260905"

    def qrun(method_name: str, python: str, runner_args: str, env: str | None = None) -> str:
        queue_root = f"{out_root}/queues/{method_name}"
        return (
            f"mkdir -p {out_root} && cd {out_root} && "
            f"exec {python} {W}/formal_evaluation/run_formal_window_queue.py "
            f"--method {method_name} --python {python} {runner_args} "
            f"{idx_args} --output-root {out_root}/{method_name} --queue-root {queue_root} "
            f"--gpus {gpu} --resume"
        )

    if method == "egofound3r":
        return "run_formal_window_queue", qrun(
            method, PY_EGO,
            f"--runner {W}/formal_evaluation/scene/adapters/run_egofound3r_baseline.py "
            f"--env {common_env} "
            f"--runner-arg=--phase --runner-arg=formal --runner-arg=--window-input --runner-arg={{window_input}} "
            f"--runner-arg=--methods-config --runner-arg={METHODS_CONFIG} "
            f"--runner-arg=--config --runner-arg=/mnt/cpfs/sjc/EgoFound3R_archive/20260905/protected_checkpoints/final_dynamic_multirate_root_fusion_v2_e73dcd8_step001599/seven_dataset_dynamic_multirate_resume1400_memory_safe_1600_8gpu_zero2_20260904.toml "
            f"--runner-arg=--checkpoint --runner-arg=/mnt/cpfs/sjc/EgoFound3R_archive/20260905/protected_checkpoints/final_dynamic_multirate_root_fusion_v2_e73dcd8_step001599/checkpoints/step_001599.pt "
            f"--runner-arg=--backbone-checkpoint --runner-arg=/mnt/workspace/sjc/models/pretrained/VGGT-Omega/vggt_omega_1b_512.pt "
            f"--runner-arg=--output-root --runner-arg={{output_root}}"
        )
    if method == "wilor":
        return "run_formal_window_queue", qrun(
            method, PY_EGO,
            f"--runner {W}/formal_evaluation/hand/adapters/run_wilor_baseline.py "
            f"--runner-arg=--phase --runner-arg=formal --runner-arg=--window-input --runner-arg={{window_input}} "
            f"--runner-arg=--methods-config --runner-arg={METHODS_CONFIG} "
            f"--runner-arg=--source-root --runner-arg=/mnt/workspace/sjc/EgoFound3R-baselines/WiLoR "
            f"--runner-arg=--checkpoint --runner-arg=/mnt/workspace/sjc/models/pretrained/WiLoR/wilor_final.ckpt "
            f"--runner-arg=--detector --runner-arg=/mnt/workspace/sjc/models/pretrained/WiLoR/detector.pt "
            f"--runner-arg=--detect-batch --runner-arg=16 --runner-arg=--forward-batch --runner-arg=32 "
            f"--runner-arg=--output-root --runner-arg={{output_root}}"
        )
    if method == "hawor":
        return "run_formal_window_queue", qrun(
            method, PY_EGO,
            f"--runner {W}/formal_evaluation/hand/adapters/run_hawor_baseline.py "
            f"--runner-arg=--phase --runner-arg=formal --runner-arg=--window-input --runner-arg={{window_input}} "
            f"--runner-arg=--methods-config --runner-arg={METHODS_CONFIG} "
            f"--runner-arg=--source-root --runner-arg=/mnt/workspace/sjc/EgoFound3R-baselines/HaWoR "
            f"--runner-arg=--checkpoint --runner-arg=/mnt/workspace/sjc/models/pretrained/HaWoR/hawor/checkpoints/hawor.ckpt "
            f"--runner-arg=--infiller-weight --runner-arg=/mnt/workspace/sjc/models/pretrained/HaWoR/hawor/checkpoints/infiller.pt "
            f"--runner-arg=--detector-weight --runner-arg=/mnt/workspace/sjc/models/pretrained/HaWoR/external/detector.pt "
            f"--runner-arg=--output-root --runner-arg={{output_root}}"
        )
    if method == "pad_hand":
        return "run_formal_window_queue", qrun(
            method, PY_PADHAND,
            f"--runner {W}/formal_evaluation/hand/adapters/run_pad_hand_baseline.py "
            f"--runner-arg=--phase --runner-arg=formal --runner-arg=--window-input --runner-arg={{window_input}} "
            f"--runner-arg=--methods-config --runner-arg={METHODS_CONFIG} "
            f"--runner-arg=--source-root --runner-arg=/mnt/workspace/sjc/EgoFound3R-baselines/PAD-Hand "
            f"--runner-arg=--wilor-python --runner-arg={PY_EGO} "
            f"--runner-arg=--checkpoint --runner-arg=/mnt/workspace/sjc/EgoFound3R-baselines/PAD-Hand/checkpoints/pad_hand.pt "
            f"--runner-arg=--output-root --runner-arg={{output_root}}"
        )
    if method == "reviv4d":
        return "run_formal_window_queue", qrun(
            method, PY_EGO,
            f"--runner {W}/formal_evaluation/scene/adapters/run_reviv4d_baseline.py "
            f"--runner-arg=--phase --runner-arg=formal --runner-arg=--window-input --runner-arg={{window_input}} "
            f"--runner-arg=--methods-config --runner-arg={METHODS_CONFIG} "
            f"--runner-arg=--source-root --runner-arg=/mnt/workspace/sjc/external/reviv4d "
            f"--runner-arg=--python --runner-arg={PY_REVIV4D} "
            f"--runner-arg=--checkpoint-root --runner-arg=/mnt/workspace/sjc/external/reviv4d/reviv_checkpoints/metric_depth "
            f"--runner-arg=--cosmos-dir --runner-arg=/mnt/workspace/sjc/external/reviv4d/Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16 "
            f"--runner-arg=--hand-cosmos-dir --runner-arg=/mnt/workspace/sjc/external/reviv4d/Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8 "
            f"--runner-arg=--amp-dtype --runner-arg=bf16 "
            f"--runner-arg=--output-root --runner-arg={{output_root}}"
        )
    if method in ("vggt", "pi3", "da3_large_1_1", "vggt_omega"):
        src = {
            "vggt": "/mnt/workspace/sjc/EgoFound3R-baselines/VGGT",
            "pi3": "/mnt/workspace/sjc/EgoFound3R-baselines/Pi3",
            "da3_large_1_1": "/mnt/workspace/sjc/EgoFound3R-baselines/Depth-Anything-3",
            "vggt_omega": "/mnt/workspace/sjc/EgoFound3R-baselines/vggt-omega",
        }[method]
        ckpt = {
            "vggt": "/mnt/workspace/sjc/models/pretrained/VGGT-1B/model.pt",
            "pi3": "/mnt/workspace/sjc/models/pretrained/Pi3/model.safetensors",
            "da3_large_1_1": "/mnt/workspace/sjc/models/pretrained/Depth-Anything-3/DA3-LARGE-1.1/model.safetensors",
            "vggt_omega": "/mnt/workspace/sjc/models/pretrained/VGGT-Omega/vggt_omega_1b_512.pt",
        }[method]
        return "run_formal_window_queue", qrun(
            method, PY_EGO,
            f"--runner {W}/formal_evaluation/scene/adapters/run_scene_baseline.py "
            f"--runner-arg=--method --runner-arg={method} "
            f"--runner-arg=--phase --runner-arg=formal --runner-arg=--window-input --runner-arg={{window_input}} "
            f"--runner-arg=--methods-config --runner-arg={METHODS_CONFIG} "
            f"--runner-arg=--source-root --runner-arg={src} "
            f"--runner-arg=--checkpoint --runner-arg={ckpt} "
            f"--runner-arg=--output-root --runner-arg={{output_root}}"
        )
    if method == "lingbot_map_long":
        return "run_formal_window_queue", qrun(
            method, PY_LINGBOT,
            f"--runner {W}/formal_evaluation/scene/adapters/run_scene_baseline.py "
            f"--runner-arg=--method --runner-arg=lingbot_map_long "
            f"--runner-arg=--phase --runner-arg=formal --runner-arg=--window-input --runner-arg={{window_input}} "
            f"--runner-arg=--methods-config --runner-arg={METHODS_CONFIG} "
            f"--runner-arg=--source-root --runner-arg=/mnt/workspace/sjc/EgoFound3R-baselines/lingbot-map "
            f"--runner-arg=--checkpoint --runner-arg=/mnt/workspace/sjc/models/pretrained/LingBot-Map/lingbot-map-long.pt "
            f"--runner-arg=--output-root --runner-arg={{output_root}}"
        )
    if method in CONTACT_METHODS:
        caches = contact_cache_paths(dataset)
        if caches is None:
            return None
        src = {
            "s2contact": "/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/S2Contact",
            "contactopt": "/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/ContactOpt",
        }[method]
        ckpt = {
            "s2contact": "/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/S2Contact/checkpoints/20211027-212322.pt",
            "contactopt": "/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/ContactOpt/checkpoints/deepcontact_checkpoint.pt",
        }[method]
        cache = caches[f"{method}_cache"]
        index = caches[f"{method}_index"]
        combined = ensure_combined_index(dataset, CONTACT_CACHE_CAPS[dataset])
        cmd = (
            f"mkdir -p {out_root} && cd {out_root} && "
            f"PYTHONPATH={W} CUDA_VISIBLE_DEVICES={gpu} exec {PY_CONTACT} -m formal_evaluation.contact.adapters.run_s2_contactopt "
            f"--baseline {method} --source-root {src} --cache {cache} --cache-index {index} "
            f"--window-input-index {combined} --checkpoint {ckpt} --mano-right {MANO_RIGHT} "
            f"--methods-config {METHODS_CONFIG} --output-root {out_root}/{method} "
            f"--phase formal --batch-size 32 --device cuda:0"
        )
        return "single_shot", cmd
    return None


def remote_handles(state: dict) -> dict[str, dict]:
    grouped: dict[int, list[str]] = {}
    for job in [*state.get("jobs", {}).values(), *state.get("cache_jobs", {}).values()]:
        if job.get("node") and job.get("handle_path"):
            grouped.setdefault(int(job["node"]), []).append(str(job["handle_path"]))
    result: dict[str, dict] = {}
    for port, handles in grouped.items():
        command = " ".join([
            shlex.quote(PY_EGO), shlex.quote(REMOTE_CONTROLLER), "status",
            *[f"--handle {shlex.quote(handle)}" for handle in handles],
        ])
        response = ssh(port, command, timeout=30)
        try:
            payload = json.loads(response.stdout)
        except ValueError:
            continue
        if response.returncode == 0 and payload.get("ok"):
            result.update(payload.get("handles", {}))
    return result


def signal_handles(requests: dict[tuple[int, str], list[str]]) -> None:
    for (port, signal_name), handles in requests.items():
        command = " ".join([
            shlex.quote(PY_EGO), shlex.quote(REMOTE_CONTROLLER), "signal",
            "--signal", signal_name,
            *[f"--handle {shlex.quote(handle)}" for handle in handles],
        ])
        response = ssh(port, command, timeout=30)
        if response.returncode != 0:
            log(f"exact signal failed node={port} signal={signal_name}: {response.stdout.strip()[:200]}")


def exact_queue_progress(dataset: str, method: str, job: dict) -> int | None:
    queue_root = job.get("queue_root")
    if not queue_root or method not in STANDARD_METHODS:
        return None
    n_needed = -(-target_windows(dataset) // 50) if dataset != "h2o" else len(DATASET_SHARDS["h2o"])
    shards = DATASET_SHARDS[dataset][:max(n_needed, 1)]
    sentinels = [
        f"{queue_root}/sentinels/{method}_{os.path.splitext(os.path.basename(shard))[0]}.status.json"
        for shard in shards
    ]
    command = "for s in " + " ".join(shlex.quote(path) for path in sentinels) + "; do test ! -f \"$s\" || cat \"$s\"; done"
    response = local(command, timeout=20)
    if response.returncode != 0:
        return None
    completed = 0
    for line in response.stdout.splitlines():
        try:
            payload = json.loads(line)
        except ValueError:
            return None
        if payload.get("status") == "success":
            completed += int(payload.get("success_count", 0)) + int(payload.get("reused_count", 0))
    return completed


def available_gpus(port: int) -> list[int]:
    marker = "__COMPUTE_APPS__"
    response = ssh(
        port,
        "nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits; "
        f"echo {marker}; nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader",
    )
    if response.returncode != 0 or marker not in response.stdout:
        return []
    gpu_text, compute_text = response.stdout.split(marker, 1)
    allowed = NODE_ALLOWED_GPUS.get(port, ())
    busy = {line.strip() for line in compute_text.splitlines() if line.strip()}
    out = []
    for line in gpu_text.splitlines():
        try:
            idx_text, uuid, free_text = (v.strip() for v in line.split(",", 2))
            idx, free = int(idx_text), int(free_text)
        except ValueError:
            continue
        if idx in allowed and uuid not in busy and free >= MIN_FREE_MB:
            out.append(idx)
    return out


def launch_registered(port: int, command: str, handle_path: str, log_path: str) -> dict | None:
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    digest = hashlib.sha256(command.encode("utf-8")).hexdigest()
    remote = " ".join([
        shlex.quote(PY_EGO), shlex.quote(REMOTE_CONTROLLER), "launch",
        "--handle", shlex.quote(handle_path),
        "--log", shlex.quote(log_path),
        "--shell-b64", encoded,
        "--command-sha256", digest,
    ])
    response = ssh(port, remote, timeout=30)
    try:
        payload = json.loads(response.stdout)
    except ValueError:
        return None
    return payload if response.returncode == 0 and payload.get("ok") else None


def dispatch(port: int, gpu: int, dataset: str, method: str) -> dict | None:
    check = local(
        f"cd {shlex.quote(W)} && {shlex.quote(PY_EGO)} "
        f"formal_evaluation/validate_runtime_registry.py --method {shlex.quote(method)} "
        f"--dataset {shlex.quote(dataset)} --strict"
    )
    if check.returncode != 0:
        log(f"strict registry validation failed for {dataset}/{method}: {check.stderr.strip()[:200]}")
        return None
    built = build_command(dataset, method, gpu)
    if built is None:
        return None
    _, cmd = built
    stamp = now().replace(":", "").replace("-", "")
    queue_old = f"/queues/{method}"
    queue_new = f"/queues/{method}_{SCHEDULER_ID}_{stamp}_{port}_g{gpu}"
    cmd = cmd.replace(queue_old, queue_new, 1)
    queue_root = f"{dataset_method_root(dataset, method)}{queue_new}"
    launch_dir = f"{CACHE_WORK_ROOT}/launches/{dataset}"
    log_path = f"{launch_dir}/{method}_{stamp}_{port}_g{gpu}.out"
    handle_path = f"{HANDLE_ROOT}/{dataset}/{method}_{stamp}_{port}_g{gpu}.json"
    launched = launch_registered(port, cmd, handle_path, log_path)
    if launched is None:
        log(f"exact dispatch failed {dataset}/{method} on {port}:GPU{gpu}")
        return None
    log(f"exact dispatch {dataset}/{method} on {port}:GPU{gpu} pid={launched['pid']}")
    return {
        "node": port,
        "gpu": gpu,
        "pid": launched["pid"],
        "pgid": launched["pgid"],
        "handle_path": handle_path,
        "queue_root": queue_root if method in STANDARD_METHODS else None,
        "launch_log": log_path,
    }


def run_cycle(state: dict) -> None:
    jobs = state.setdefault("jobs", {})
    controls = load_controls()
    handle_states = remote_handles(state)
    signals: dict[tuple[int, str], list[str]] = {}

    for dataset in DATASETS:
        if not dataset_ready(dataset):
            log(f"waiting for materialization: {dataset}")
            continue
        cache_ready = ensure_contact_cache(dataset, state, handle_states) if dataset in CONTACT_DATASETS else False
        methods = STANDARD_METHODS + (CONTACT_METHODS if dataset in CONTACT_DATASETS else ())
        for method in methods:
            key = job_key(dataset, method)
            job = jobs.setdefault(key, {"status": "unknown"})
            desired = controls.get(key, job.get("desired_state", "running"))
            if desired not in ("running", "paused"):
                raise ValueError(f"invalid desired state for {key}: {desired}")
            job["desired_state"] = desired
            if method in CONTACT_METHODS and not cache_ready:
                job["status"] = "blocked_on_cache"
                continue
            observed = handle_states.get(job.get("handle_path", ""), {}).get("status")
            if observed in ("running", "paused"):
                job["status"] = observed
                count = exact_queue_progress(dataset, method, job)
                if count is not None:
                    job["count"] = count
                if desired == "paused" and observed == "running":
                    signals.setdefault((int(job["node"]), "STOP"), []).append(job["handle_path"])
                    job["status"] = "pausing"
                elif desired == "running" and observed == "paused":
                    signals.setdefault((int(job["node"]), "CONT"), []).append(job["handle_path"])
                    job["status"] = "resuming"
                continue
            if job.get("status") in ("done", "queue_exited_needs_audit"):
                continue
            if observed == "exited" or job.get("status") in ("running", "paused", "pausing", "resuming"):
                job["status"] = "queue_exited_needs_audit"
                job["pid"] = None
                log(f"{key} exact handle exited; retained for audit, not duplicated")
                continue
            if job.get("handle_path") and observed in (None, "missing_handle", "stale_handle"):
                job["status"] = "control_handle_error"
                log(f"{key} exact handle unavailable: {observed or 'status_unavailable'}")
                continue
            if desired == "paused":
                job["status"] = "paused_pending"
                continue
            job["status"] = "pending"
    signal_handles(signals)
    save_state(state)

    # dispatch pending jobs onto any currently-free GPU, across all nodes
    pending = [k for k, j in jobs.items() if j.get("status") == "pending" and j.get("desired_state") == "running"]
    for port in NODES:
        gpus = available_gpus(port)
        for gpu in gpus:
            if not pending:
                break
            key = pending.pop(0)
            dataset, method = key.split("::", 1)
            launched = dispatch(port, gpu, dataset, method)
            if launched:
                jobs[key]["status"] = "running"
                jobs[key].update({field: value for field, value in launched.items() if value is not None})
                jobs[key]["dispatched_at"] = now()
            else:
                pending.append(key)
    save_state(state)


def main() -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    register_scheduler_runs()
    state = load_state()
    log("scheduler start")

    while True:
        try:
            run_cycle(state)
        except Exception as exc:  # noqa: BLE001 - a long-running daemon must not die on one bad cycle
            log(f"ERROR cycle failed: {exc!r}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
