#!/usr/bin/env python3
"""Exact-ID registration and control for formal-evaluation tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import fcntl
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_REGISTRY = PROJECT_ROOT / "formal_evaluation/config/evaluation_task_registry.json"
DEFAULT_CONTROLS = PROJECT_ROOT / ".auto_scheduler/task_controls.json"
TERMINAL_STATES = {"done", "queue_exited_needs_audit", "failed", "blocked"}
ACTIVE_STATES = {"running", "paused", "pausing", "resuming", "pending"}
SIX_DATASETS = {"h2o", "hot3d", "arctic", "oakink_v2", "taco", "hoi4d"}
OVERVIEW_PHASES = {
    "formal", "gt-cache-v2", "metrics-scene-v1", "metrics-v2", "contact-geometry",
}
RESOURCE_NODES = (5000, 5001, 6001, 8093, 8094, 8095, 8096)
EGOFOUND3R_FORMAL_MODEL_ROOT = Path(
    "/mnt/cpfs/sjc/EgoFound3R_archive/20260905/protected_checkpoints/"
    "final_dynamic_multirate_root_fusion_v2_e73dcd8_step001599"
)
EGOFOUND3R_FORMAL_CHECKPOINT = EGOFOUND3R_FORMAL_MODEL_ROOT / "checkpoints/step_001599.pt"
EGOFOUND3R_FORMAL_CONFIG = (
    EGOFOUND3R_FORMAL_MODEL_ROOT
    / "seven_dataset_dynamic_multirate_resume1400_memory_safe_1600_8gpu_zero2_20260904.toml"
)
EGOFOUND3R_INFERENCE_ROOT = Path(
    "/mnt/cpfs/sjc/EgoFound3R_final_bf16_inference_8b0c44a_20260905"
)
EGOFOUND3R_TRAINING_SOURCE_ROOT = Path(
    "/mnt/cpfs/sjc/EgoFound3R_root_depth_fusion_v2_e73dcd8_20260904"
)
EGOFOUND3R_MODEL_PYTHON = "/mnt/workspace/sjc/envs/egofound3r/bin/python"
EGOFOUND3R_FINAL_SMOKE_TASK_ID = "smoke:h2o:egofound3r_final_rootfusionv2_step1599:60f"
EGOFOUND3R_FINAL_SMOKE_METHOD_SET = "egofound3r_final_rootfusionv2_step1599"
EGOFOUND3R_FINAL_SMOKE_RUN_ID = "smoke-h2o-egofound3r-final-rootfusionv2-step1599-839020c-20260905T014720Z"
EGOFOUND3R_FINAL_SMOKE_OUTPUT = Path(
    "/mnt/workspace/sjc/DATA/eval_artifacts/"
    "egofound3r_final_rootfusionv2_step1599_smoke_839020c_20260905T014720Z"
)
EGOFOUND3R_FINAL_SMOKE_STATE = (
    ".auto_scheduler/egofound3r_final_rootfusionv2_step1599_smoke_20260905T014720Z/state.json"
)
EGOFOUND3R_FINAL_SMOKE_JOB = "h2o::egofound3r_final_rootfusionv2_step1599_smoke"
EGOFOUND3R_FINAL_SMOKE_INPUT_INDEX = Path(
    "/mnt/workspace/sjc/eval_artifacts/prep_h2o_60f_strict_20260822T030835Z_5001_g1/"
    "window_inputs/window_inputs_h2o_shard_000_of_006.jsonl"
)
EGOFOUND3R_BASELINES_COMMIT = "839020c7fb750079c4d98f7c068fc4fb1df33047"
EGOFOUND3R_TRAINING_COMMIT = "e73dcd8a51b0a06c1790fc18e1b1aa7b5180aea8"
EGOFOUND3R_INFERENCE_COMMIT = "8b0c44a721bded978373a3d9a8222b096d8c9930"
EGOFOUND3R_CHECKPOINT_SHA256 = "f358de97ff0c9f9ba4f35f48a68f62582403580a2095f08a9362b94ad2416a6f"
EGOFOUND3R_BACKBONE = Path(
    "/mnt/workspace/sjc/models/pretrained/VGGT-Omega/vggt_omega_1b_512.pt"
)

REMOTE_EGOFOUND3R_FORMAL_MODEL_AUDIT = r"""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from egohandmetric_prompt import load_project_config
from egohandmetric_prompt.checkpoints import validate_marker_checkpoint_contract
from egohandmetric_prompt.configs import active_marker_model_config

model_root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
inference_root = Path(sys.argv[3])
training_source_root = Path(sys.argv[4])

def file_row(path):
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }

formal_configs = sorted(model_root.glob("*.toml")) if model_root.is_dir() else []
evidence_paths = [
    model_root / "PAYLOAD_VERIFICATION.json",
    model_root / "PROVENANCE.md",
    model_root / "launch_contract.txt",
    *formal_configs,
    training_source_root / "scripts/comparison/run_egofound3r_baseline.py",
    inference_root / "scripts/comparison/run_egofound3r_baseline.py",
]
candidates = [file_row(path) for path in evidence_paths if path.is_file()]
evidence = {
    str(path): path.read_text(encoding="utf-8", errors="replace")[:20000]
    for path in evidence_paths
    if path.is_file() and path.suffix.lower() in {".toml", ".json", ".md", ".txt"}
}
adapter_contracts = {}
for label, path in {
    "training": training_source_root / "scripts/comparison/run_egofound3r_baseline.py",
    "inference": inference_root / "scripts/comparison/run_egofound3r_baseline.py",
}.items():
    if not path.is_file():
        continue
    source = path.read_text(encoding="utf-8")
    adapter_contracts[label] = {
        **file_row(path),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "dynamic_forward_contract": all(
            token in source for token in ("_build_prediction_marker_forward_contract", "_call_marker_model")
        ),
        "checkpoint_native_dtype": "align_model_floating_dtype=True" in source,
        "new_hand_validity": all(
            token in source for token in ("root_translation_valid", "in_view_probability")
        ),
    }
train_log = model_root / "train.log"
train_log_evidence = []
if train_log.is_file():
    with train_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "stage 结束 checkpoint" in line or "模式=monitor step=1599" in line:
                train_log_evidence.append(line.rstrip())

result = {
    "status": "ok" if checkpoint.is_file() else "missing_checkpoint",
    "authorized_roots": [str(model_root), str(training_source_root), str(inference_root)],
    "checkpoint": file_row(checkpoint) if checkpoint.is_file() else {"path": str(checkpoint)},
    "candidate_count": len(candidates),
    "candidates": candidates,
    "evidence": evidence,
    "adapters": adapter_contracts,
    "train_log_evidence": train_log_evidence[-4:],
}
if checkpoint.is_file():
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    result["checkpoint"]["sha256"] = digest.hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    state = payload.get("model", payload.get("state_dict", {}))
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)] if isinstance(state, dict) else []
    contract = payload.get("checkpoint_contract")
    contract_configs = contract.get("configs", {}) if isinstance(contract, dict) else {}
    config_contract_valid = False
    if len(formal_configs) == 1:
        project_config = load_project_config(formal_configs[0])
        validate_marker_checkpoint_contract(
            payload,
            active_marker_model_config(project_config),
            resume_mode="weights",
        )
        config_contract_valid = True
    result["metadata"] = {
        "payload_keys": sorted(str(key) for key in payload),
        "checkpoint_schema_version": payload.get("checkpoint_schema_version"),
        "checkpoint_namespace": payload.get("checkpoint_namespace"),
        "temporal_architecture": payload.get("temporal_architecture"),
        "temporal_architecture_hash": payload.get("temporal_architecture_hash"),
        "checkpoint_contract_keys": sorted(contract) if isinstance(contract, dict) else [],
        "checkpoint_contract_config_hashes": {
            str(name): value.get("sha256")
            for name, value in contract_configs.items()
            if isinstance(value, dict) and value.get("sha256")
        },
        "backend": payload.get("backend"),
        "step": payload.get("step"),
        "best_step": payload.get("best_step"),
        "best_metric": payload.get("best_metric"),
        "config_contract_valid": config_contract_valid,
        "model_tensor_count": len(tensors),
        "model_parameter_count": sum(tensor.numel() for tensor in tensors),
        "model_dtypes": dict(sorted(Counter(str(tensor.dtype) for tensor in tensors).items())),
    }
print(json.dumps(result, ensure_ascii=False))
"""

REMOTE_RESOURCE_PROBE = r"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

def cpu_times():
    values = [int(value) for value in open("/proc/stat", encoding="utf-8").readline().split()[1:]]
    return sum(values), values[3] + values[4]

total_before, idle_before = cpu_times()
time.sleep(0.25)
total_after, idle_after = cpu_times()
total_delta = total_after - total_before
cpu_usage = (100.0 * (1.0 - (idle_after - idle_before) / total_delta)
             if total_delta > 0 else None)
cpu_count = os.cpu_count() or 1
load_1m, load_5m, load_15m = os.getloadavg()
meminfo = {}
with open("/proc/meminfo", encoding="utf-8") as handle:
    for line in handle:
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            meminfo[key] = int(value.split()[0]) * 1024

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free", "--format=csv,noheader,nounits"],
    text=True, capture_output=True,
)
apps = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
    text=True, capture_output=True,
)
if gpu.returncode or apps.returncode:
    print(json.dumps({"ok": False, "error": "GPU_QUERY_FAILED"})); raise SystemExit(0)
processes = {}
for line in apps.stdout.splitlines():
    if not line.strip():
        continue
    uuid, pid, name, memory = (value.strip() for value in line.split(",", 3))
    processes.setdefault(uuid, []).append({"pid": pid, "process_name": name, "memory_mb": memory})
busy = set(processes)
rows = []
for line in gpu.stdout.splitlines():
    try:
        index_text, uuid, used_text, free_text = (value.strip() for value in line.split(",", 3))
        index, used, free = int(index_text), int(used_text), int(free_text)
    except ValueError:
        continue
    rows.append({
        "index": index, "memory_used_mb": used, "memory_free_mb": free,
        "allowed": True,
        "processes": processes.get(uuid, []),
        "idle": uuid not in busy,
    })
mount = sys.argv[1]
usage = shutil.disk_usage(mount)
oss_root = Path("/mnt/oss/pre-train/ego/eval_artifacts")
try:
    next(oss_root.iterdir(), None)
    oss = {"root": str(oss_root), "readable": True}
except OSError as error:
    oss = {"root": str(oss_root), "readable": False, "error": type(error).__name__}
print(json.dumps({
    "ok": True, "gpus": rows,
    "cpu": {"count": cpu_count, "usage_percent": cpu_usage,
            "load_1m": load_1m, "load_5m": load_5m, "load_15m": load_15m,
            "load_1m_per_cpu": load_1m / cpu_count},
    "memory": {"total_bytes": meminfo.get("MemTotal"),
               "available_bytes": meminfo.get("MemAvailable")},
    "oss": oss,
    "storage": {"mount": mount, "total_bytes": usage.total,
                "used_bytes": usage.used, "available_bytes": usage.free},
}))
"""

REMOTE_READABILITY_PROBE = r"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset_dir = root / sys.argv[2]
index = root / sys.argv[3]
try:
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()] if index.is_file() else []
    print(json.dumps({
        "ok": True,
        "root_exists": root.is_dir(),
        "complete_exists": (root / "COMPLETE").is_file(),
        "index_exists": index.is_file(),
        "index_lines": len(rows),
        "unique_cache_ids": len({row.get("cache_id") for row in rows}),
        "missing_index_artifacts": sum(not Path(row[key]).is_file() for row in rows for key in ("array_path", "metadata_path")),
        "index_bytes": index.stat().st_size if index.is_file() else 0,
        "npz_files": sum(1 for path in dataset_dir.iterdir() if path.suffix == ".npz") if dataset_dir.is_dir() else 0,
        "json_files": sum(1 for path in dataset_dir.iterdir() if path.suffix == ".json") if dataset_dir.is_dir() else 0,
    }))
except OSError as error:
    print(json.dumps({"ok": False, "error": type(error).__name__}))
"""

REMOTE_BASELINE_GIT_AUDIT = r"""
import json
import subprocess
from pathlib import Path

repo = Path("/mnt/workspace/sjc/EgoFound3R-baselines")
target = "f8332ee5b86222c8fe4b6beabfc56c5d692bda87"
def git(*args):
    value = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, timeout=90
    )
    if value.returncode:
        raise RuntimeError(" | ".join((value.stdout + value.stderr).splitlines()[-4:]))
    return value.stdout.strip()
try:
    status = git("status", "--porcelain=v1")
    paths = status.splitlines()
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    try:
        origin_head = git("rev-parse", "refs/remotes/origin/formal-hand-fix")
    except RuntimeError:
        origin_head = None
    print(json.dumps({
        "ok": True,
        "repo": str(repo),
        "branch": branch,
        "head": head,
        "target": target,
        "head_matches_target": head == target,
        "origin_formal_hand_fix": origin_head,
        "origin_ref_matches_target": origin_head == target,
        "clean": not paths,
        "status_count": len(paths),
        "status_sample": paths[:20],
    }))
except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
    print(json.dumps({"ok": False, "repo": str(repo), "error": type(error).__name__ + ":" + str(error)}))
"""

REMOTE_HOT3D_CONTACT_INPUT_AUDIT = r"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/workspace/sjc/EgoFound3R-baselines_formal_taco_v2_87e8f03")
from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache

prediction_roots = {
    "s2contact": Path("/mnt/workspace/sjc/DATA/eval_artifacts/formal_contact_hot3d_400_20260827T234639Z/s2contact/s2contact/formal"),
    "contactopt": Path("/mnt/workspace/sjc/DATA/eval_artifacts/formal_contact_hot3d_400_20260827T234639Z/contactopt/contactopt/formal"),
}
gt_candidates = [
    Path("/mnt/oss/pre-train/ego/eval_artifacts/formal_standard_reports_10methods_20260824/hot3d/gt_cache/index_shard_000_of_001.jsonl"),
    Path("/mnt/oss/pre-train/ego/eval_artifacts/formal_hand6_v2_gt_report_20260825T041010Z/hot3d/gt_cache/index_shard_000_of_001.jsonl"),
    Path("/mnt/workspace/sjc/DATA/eval_artifacts/formal_standard_reports_10methods_20260824/hot3d/gt_cache/index_shard_000_of_001.jsonl"),
    Path("/mnt/workspace/sjc/DATA/eval_artifacts/formal_hand6_v2_gt_report_20260825T041010Z/hot3d/gt_cache/index_shard_000_of_001.jsonl"),
    Path("/mnt/workspace/sjc/eval_artifacts/formal_hand6_v2_gt_report_20260825T041010Z/hot3d/gt_cache/index_shard_000_of_001.jsonl"),
]
errors = []
prediction_counts = {}
for method, root in prediction_roots.items():
    count = 0
    try:
        children = sorted(root.iterdir())
    except OSError as error:
        errors.append(f"{method}: {error}")
        continue
    for child in children:
        try:
            json.loads((child / "metadata.json").read_text(encoding="utf-8"))
            with np.load(child / "predictions.npz", allow_pickle=False) as archive:
                tuple(archive.files)
        except (OSError, ValueError) as error:
            errors.append(f"{method}/{child.name}: {error}")
            if len(errors) >= 3:
                break
        else:
            count += 1
    prediction_counts[method] = count

gt_index = None
for path in gt_candidates:
    try:
        if path.is_file():
            gt_index = path
            break
    except OSError:
        continue
gt_count = 0
if gt_index is None:
    errors.append("HOT3D_GT_INDEX_NOT_READABLE")
else:
    for line in gt_index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            load_window_cache(json.loads(line))
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"gt/{gt_count}: {error}")
            if len(errors) >= 3:
                break
        else:
            gt_count += 1
print(json.dumps({
    "ok": not errors and prediction_counts == {"s2contact": 400, "contactopt": 400} and gt_count == 400,
    "prediction_counts": prediction_counts,
    "gt_index": str(gt_index) if gt_index else None,
    "gt_count": gt_count,
    "errors": errors[:3],
}))
"""


# This runs only on the registered node and only reads registered paths.  It is
# deliberately layout-aware rather than a filesystem search: formal inference
# outputs live immediately under <method-root>/formal, and v2 cache progress is
# the number of rows in its exact index shard.
REMOTE_PROGRESS_PROBE = r"""
import hashlib
import json
import os
import sys
from pathlib import Path


def proc_identity(pid):
    try:
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
        return {"state": tail[0], "pgid": int(tail[2]), "start_ticks": int(tail[19])}
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def handle_status(spec):
    handle = Path(spec.get("handle_path") or "")
    registered = None
    if handle.suffix == ".json":
        try:
            registered = json.loads(handle.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return "missing_handle"
    pid = (registered or spec).get("pid")
    if pid is None:
        return None
    current = proc_identity(int(pid))
    if current is None:
        return "exited"
    expected_pgid = (registered or spec).get("pgid")
    expected_ticks = (registered or spec).get("start_ticks")
    if expected_pgid is not None and current["pgid"] != int(expected_pgid):
        return "stale_handle"
    if expected_ticks is not None and current["start_ticks"] != int(expected_ticks):
        return "stale_handle"
    return "paused" if current["state"] == "T" else "running"


def log_count(path):
    highest = None
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                    index = int(value["index"])
                except (KeyError, TypeError, ValueError):
                    continue
                highest = index if highest is None else max(highest, index)
    except (FileNotFoundError, OSError):
        return None
    # build_six_dataset_gt_cache.py deliberately reports one-based indices.
    return highest


def line_count(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return sum(bool(line.strip()) for line in handle)
    except (FileNotFoundError, OSError):
        return None


def completion_artifacts_ready(items):
    if not items:
        return False
    for item in items:
        path = Path(item["path"])
        try:
            if path.stat().st_size < int(item.get("min_bytes", 1)):
                return False
        except (FileNotFoundError, OSError, ValueError):
            return False
        if "min_lines" in item:
            count = line_count(path)
            if count is None or count < int(item["min_lines"]):
                return False
        if "json_field" in item:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if payload.get(item["json_field"]) != item.get("equals"):
                return False
    return True


def completion_artifact_progress(items):
    counts = []
    for item in items or []:
        if "progress_divisor" not in item:
            continue
        count = line_count(Path(item["path"]))
        if count is None:
            count = 0
        counts.append(min(count, int(item.get("min_lines", count))) // int(item["progress_divisor"]))
    return sum(counts) if counts else None


def prediction_count(path):
    method_root = Path(path)
    outputs = set()
    found_layout = False
    for formal in (method_root / "formal", method_root / method_root.name / "formal"):
        try:
            for child in formal.iterdir():
                if child.is_dir() and (child / "metadata.json").is_file() and (child / "predictions.npz").is_file():
                    outputs.add(str(child))
            found_layout = True
        except (FileNotFoundError, OSError):
            continue
    return len(outputs) if found_layout else None


def log_tail_from_handle(path):
    try:
        log = Path(json.loads(Path(path).read_text(encoding="utf-8"))["log"])
        with log.open("rb") as handle:
            handle.seek(max(0, handle.seek(0, os.SEEK_END) - 16384))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None
    return "\n".join(line for line in lines if line.strip())[-4000:] or None


observed = {}
for spec in json.loads(sys.argv[1]):
    root = Path(spec["output_root"])
    value = {"process_status": handle_status(spec)}
    complete = (root / "COMPLETE").is_file()
    value["completion_evidence"] = {
        "output_root": str(root),
        "complete_exists": complete,
        "complete_mtime": (root / "COMPLETE").stat().st_mtime if complete else None,
    }
    value["completion_evidence"]["runtime_sources"] = {}
    for path, expected in spec.get("runtime_sources", {}).items():
        try:
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            value["completion_evidence"]["runtime_sources"][path] = {
                "sha256": actual, "matches_local": actual == expected,
            }
        except OSError as error:
            value["completion_evidence"]["runtime_sources"][path] = {"error": type(error).__name__}
    summary_payload = None
    for name in ("summary.json", "smoke_summary.json"):
        path = root / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value["completion_evidence"][name] = payload
            if name == "summary.json" and isinstance(payload, dict):
                summary_payload = payload
        except (OSError, ValueError):
            value["completion_evidence"][name] = None
    if spec.get("completion_path"):
        try:
            json.loads(Path(spec["completion_path"]).read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            pass
        else:
            complete = True
    registered_artifacts_ready = completion_artifacts_ready(spec.get("completion_artifacts"))
    value["completion_evidence"]["registered_artifacts_ready"] = registered_artifacts_ready
    value["completion_evidence"]["registered_artifacts"] = spec.get("completion_artifacts", [])
    if spec.get("partial_output_stats"):
        files = bytes_total = partial_files = 0
        payload_root = root / "payload"
        try:
            for current, _, names in os.walk(payload_root):
                for name in names:
                    path = Path(current) / name
                    if path.is_symlink():
                        continue
                    files += 1
                    bytes_total += path.stat().st_size
                    partial_files += int(name.endswith(".partial"))
            value["completion_evidence"]["partial_output_stats"] = {
                "root": str(payload_root), "files": files,
                "logical_bytes": bytes_total, "partial_files": partial_files,
            }
        except OSError as error:
            value["completion_evidence"]["partial_output_stats"] = {"error": type(error).__name__}
    if registered_artifacts_ready:
        complete = True
    artifact_progress = completion_artifact_progress(spec.get("completion_artifacts"))
    if artifact_progress is not None:
        value.update(count=artifact_progress, source="completion_artifacts")
    if spec.get("accept_report"):
        for name in ("report.json", "hand6_report.json"):
            try:
                json.loads((root / name).read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, ValueError):
                continue
            complete = True
            break
    if complete:
        value["status"] = "done"
    if spec.get("distance_window_progress") and not complete:
        import zipfile
        counts = {}
        for dataset in ("h2o", "hot3d", "arctic", "oakink_v2", "taco", "hoi4d"):
            counts[dataset] = sum(zipfile.is_zipfile(path) for path in (root / "windows" / dataset).glob("*.npz"))
        value.update(count=sum(counts.values()), source="distance_window_npz")
        value["completion_evidence"]["exported_distance_windows"] = counts
    if spec.get("task_type") == "gt-cache":
        count = line_count(root / "index_shard_000_of_001.jsonl")
        if count is not None:
            value.update(count=count, source="cache_index")
    elif spec.get("prediction_root"):
        count = prediction_count(spec["prediction_root"])
        if count is not None:
            value.update(count=count, source="prediction_dirs")
            if count >= int(spec.get("target_windows", count + 1)):
                value["status"] = "done"
    handle = spec.get("handle_path")
    if handle and str(handle).endswith(".log"):
        count = log_count(handle)
        if count is not None:
            value.update(count=count, source="json_log")
    if spec.get("audit_log") and handle:
        tail = log_tail_from_handle(handle)
        if tail is not None:
            value["audit_log_tail"] = tail
    if spec.get("audit_path"):
        try:
            audit = json.loads(Path(spec["audit_path"]).read_text(encoding="utf-8"))
            failures = audit.get("failures", [])
            value["audit_summary"] = {
                key: audit[key] for key in ("status", "expected_window_count", "expected_failure_count", "observed_failure_count")
                if key in audit
            }
            if failures:
                value["audit_summary"]["first_failure"] = {
                    key: failures[0][key] for key in ("error", "prediction_dir") if key in failures[0]
                }
        except (FileNotFoundError, OSError, ValueError, TypeError):
            pass
    if spec.get("report_path"):
        try:
            report = json.loads(Path(spec["report_path"]).read_text(encoding="utf-8"))
            tokens = tuple(spec.get("report_metric_tokens", []))
            dataset = str(spec.get("report_dataset"))
            methods = {}
            for method, method_report in report.get("methods", {}).items():
                metrics = method_report.get("datasets", {}).get(dataset, {})
                methods[method] = {
                    "missing_prediction_windows": method_report.get("missing_prediction_windows"),
                    "n_windows": metrics.get("n_windows"),
                    "metrics": {
                        key: val for key, val in metrics.items()
                        if tokens and any(token in key for token in tokens)
                        and key.endswith(("_mean", "_count", "_undefined_window_count"))
                    },
                }
            value["report_summary"] = {
                "gt_windows": report.get("gt_windows"),
                "units": spec.get("report_units"),
                "report_path": spec["report_path"],
                "report_sha256": hashlib.sha256(Path(spec["report_path"]).read_bytes()).hexdigest(),
                "methods": methods,
            }
        except (FileNotFoundError, OSError, ValueError, TypeError):
            pass
    if "report_summary" not in value and (isinstance(summary_payload, dict) or spec.get("report_paths") or spec.get("completion_artifacts")):
        raw_reports = summary_payload.get("reports", {}) if isinstance(summary_payload, dict) else {}
        report_paths = []
        stack = [raw_reports,
                 summary_payload.get("report") if isinstance(summary_payload, dict) else None,
                 *(spec.get("report_paths") or []),
                 *[item.get("path") for item in (spec.get("completion_artifacts") or [])
                   if isinstance(item, dict) and str(item.get("path", "")).endswith("/report.json")]]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                stack.extend(current.values())
            elif isinstance(current, (list, tuple)):
                stack.extend(current)
            elif isinstance(current, str) and current.endswith("report.json"):
                report_paths.append(current)
        reports = {}
        for report_path in sorted(set(report_paths)):
            try:
                path = Path(report_path)
                report = json.loads(path.read_text(encoding="utf-8"))
                methods = {}
                for method, method_report in report.get("methods", {}).items():
                    datasets = {}
                    for dataset, metrics in method_report.get("datasets", {}).items():
                        datasets[dataset] = {
                            "missing_prediction_windows": method_report.get("missing_prediction_windows"),
                            "metrics": {
                                key: val for key, val in metrics.items()
                                if key == "n_windows" or key.endswith(("_mean", "_count", "_undefined_window_count"))
                            },
                        }
                    methods[method] = datasets
                reports[report_path] = {
                    "gt_windows": report.get("gt_windows"),
                    "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "methods": methods,
                    **{key: report[key] for key in ("status", "method", "windows", "datasets", "schemes", "filtering", "selection_sha256") if key in report},
                }
            except (FileNotFoundError, OSError, ValueError, TypeError):
                continue
        if reports:
            value["report_summary"] = {"reports": reports}
    observed[spec["key"]] = value
print(json.dumps(observed, sort_keys=True))
"""


EVAL_ARTIFACT_TREE_COMPARE_PROBE = r"""
import json
import os
import stat
import subprocess
import tempfile

source = "/mnt/workspace/sjc/DATA/eval_artifacts"
destination = "/mnt/oss/pre-train/ego/eval_artifacts"

if not os.path.isdir(source) or os.path.islink(source):
    raise SystemExit("INVALID_CPFS_SOURCE")
if not os.path.isdir(destination) or os.path.islink(destination):
    raise SystemExit("INVALID_OSSFS_DESTINATION")
mount = subprocess.run(
    ["findmnt", "-T", destination, "-o", "FSTYPE", "-n"],
    text=True, capture_output=True, check=False,
)
if mount.returncode or mount.stdout.strip() != "fuse.ossfs2":
    raise SystemExit("OSSFS_MOUNT_UNHEALTHY")

def scan(root, name, temporary):
    raw = os.path.join(temporary, name + ".raw")
    ordered = os.path.join(temporary, name + ".sorted")
    counts = {"objects": 0, "regular_files": 0, "directories": 0, "symlinks": 0,
              "regular_file_bytes": 0, "specials": 0}
    stack = [("", root)]
    with open(raw, "w", encoding="utf-8") as output:
        while stack:
            relative_parent, current = stack.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    relative = entry.name if not relative_parent else relative_parent + "/" + entry.name
                    info = entry.stat(follow_symlinks=False)
                    counts["objects"] += 1
                    if stat.S_ISLNK(info.st_mode):
                        kind, value = "l", os.readlink(entry.path)
                        counts["symlinks"] += 1
                    elif stat.S_ISDIR(info.st_mode):
                        kind, value = "d", None
                        counts["directories"] += 1
                        stack.append((relative, entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        kind, value = "f", info.st_size
                        counts["regular_files"] += 1
                        counts["regular_file_bytes"] += info.st_size
                    else:
                        kind, value = "s", [stat.S_IFMT(info.st_mode), info.st_size]
                        counts["specials"] += 1
                    output.write(json.dumps([relative, kind, value], separators=(",", ":")) + "\n")
    env = {**os.environ, "LC_ALL": "C"}
    subprocess.run(["sort", raw, "-o", ordered], env=env, check=True)
    return ordered, counts

def compare(left_path, right_path):
    differences, examples = 0, []
    with open(left_path, encoding="utf-8") as left, open(right_path, encoding="utf-8") as right:
        left_line, right_line = left.readline(), right.readline()
        while left_line or right_line:
            if left_line and left_line == right_line:
                left_line, right_line = left.readline(), right.readline()
            elif not right_line or (left_line and left_line < right_line):
                differences += 1
                if len(examples) < 20:
                    examples.append({"only": "cpfs", "entry": json.loads(left_line)})
                left_line = left.readline()
            else:
                differences += 1
                if len(examples) < 20:
                    examples.append({"only": "ossfs", "entry": json.loads(right_line)})
                right_line = right.readline()
    return differences, examples

with tempfile.TemporaryDirectory(prefix="eval_artifact_compare_") as temporary:
    source_manifest, source_stats = scan(source, "cpfs", temporary)
    destination_manifest, destination_stats = scan(destination, "ossfs", temporary)
    difference_count, difference_examples = compare(source_manifest, destination_manifest)
consistent = difference_count == 0 and source_stats == destination_stats
print(json.dumps({"status": "ok", "source": source, "destination": destination,
                  "comparison": "relative path, object type, symlink target, and file size; no hashes",
                  "cpfs": source_stats, "ossfs": destination_stats,
                  "difference_entry_count": difference_count,
                  "difference_examples": difference_examples,
                  "consistent": consistent}, sort_keys=True))
"""

STANDARD10_CPFS_STORAGE_PROBE = r"""
import json
import os
import stat
from pathlib import Path

root = Path("/mnt/workspace/sjc/DATA/eval_artifacts")
requested = json.loads(__import__("sys").argv[1])
names = requested or sorted(path.name for path in root.iterdir())
rows = []
for name in names:
    path = root / name
    files = size = links = 0
    if path.is_dir() and not path.is_symlink():
        for current, dirs, names_in_dir in os.walk(path):
            links += sum((Path(current) / child).is_symlink() for child in dirs)
            for filename in names_in_dir:
                try:
                    info = os.lstat(os.path.join(current, filename))
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    files += 1
                    size += info.st_size
                elif stat.S_ISLNK(info.st_mode):
                    links += 1
    rows.append({"name": name, "path": str(path), "exists": path.is_dir(),
                 "top_level_symlink": path.is_symlink(), "symlinks": links,
                 "regular_files": files, "regular_file_bytes": size})
print(json.dumps({"status": "ok", "rows": rows}, sort_keys=True))
"""

STANDARD10_CPFS_ROOTS = [
    "formal_arctic_60f_20260820T0400Z_e6c4a5e",
    "formal_h2o_60f_20260822T112000Z_continuation",
    "formal_hot3d_60f_20260819T203000Z_e6c4a5e_6001",
    "formal_hot3d_60f_20260819T202019Z_e6c4a5e",
    "formal_oakink_v2_60f_20260820T0022Z_e6c4a5e",
    "formal_taco_60f_20260822T112000Z_continuation",
    "formal_hoi4d_60f_20260822T112000Z_continuation",
]


EVAL_ARTIFACT_SYMLINK_AUDIT_PROBE = r"""
import json
import os
from collections import Counter

root = "/mnt/workspace/sjc/DATA/eval_artifacts"
old_root = "/mnt/workspace/sjc/eval_artifacts"
if not os.path.isdir(root) or os.path.islink(root):
    raise SystemExit("INVALID_CPFS_SOURCE")

counts = Counter()
source_tops = Counter()
target_tops = Counter()
examples = []
present = set()
links = []
stack = [("", root)]
while stack:
    relative_parent, current = stack.pop()
    with os.scandir(current) as entries:
        for entry in entries:
            relative = entry.name if not relative_parent else relative_parent + "/" + entry.name
            if entry.is_symlink():
                target = os.readlink(entry.path)
                links.append((relative, target))
                continue
            present.add(relative)
            if entry.is_dir(follow_symlinks=False):
                stack.append((relative, entry.path))

for relative, target in links:
    counts["symlinks"] += 1
    source_tops[relative.split("/", 1)[0]] += 1
    if target == old_root or target.startswith(old_root + "/"):
        counts["old_root_targets"] += 1
        counts["broken"] += 1
        suffix = target[len(old_root):].lstrip("/")
        target_tops[suffix.split("/", 1)[0] if suffix else "."] += 1
        rebased_exists = suffix in present
        counts["rebased_exists" if rebased_exists else "rebased_missing"] += 1
        if len(examples) < 20:
            examples.append({"link": relative, "target": target,
                             "rebased": os.path.join(root, suffix),
                             "rebased_exists": rebased_exists})

print(json.dumps({"status": "ok", "root": root, "old_root": old_root,
                  "counts": dict(counts),
                  "source_top_counts": dict(sorted(source_tops.items())),
                  "target_top_counts": dict(sorted(target_tops.items())),
                  "examples": examples}, sort_keys=True))
"""


# This is intentionally a one-off, bounded audit capability.  It exists for
# the 2026-08-26 authorization to recover historical H2O/TACO/HOI4D backfill
# roots that predate the registry.  It never inspects processes or changes a
# remote file, and it returns only directories containing both canonical
# prediction artifacts.
HISTORICAL_BACKFILL_AUDIT_PROBE = r"""
import json
from collections import defaultdict
from pathlib import Path

allowed = set(json.loads(__import__("sys").argv[1]))
base = Path("/mnt/workspace/sjc/eval_artifacts")
historical_markers = ("backfill", "retry", "rerun")
method_names = {
    "egofound3r", "wilor", "hawor", "pad_hand", "pad-hand", "pi3", "vggt",
    "vggt_omega", "vggt-omega", "da3", "lingbot", "reviv4d", "dyn_hamr",
}

def metadata_dataset(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    for key in ("dataset", "dataset_name", "source_dataset"):
        candidate = value.get(key)
        if isinstance(candidate, str):
            return candidate.lower().replace("-", "_")
    return None

def method_for(root, output):
    for part in reversed(output.relative_to(root).parts[:-1]):
        normalized = part.lower().replace("-", "_")
        if normalized in {name.replace("-", "_") for name in method_names}:
            return normalized
    return "unknown"

records = defaultdict(lambda: {"count": 0, "examples": [], "metadata_dataset": set()})
scanned_roots = []
for top in sorted(base.iterdir() if base.is_dir() else []):
    if not top.is_dir():
        continue
    top_name = top.name.lower().replace("-", "_")
    matching = {dataset for dataset in allowed if dataset in top_name}
    if not matching or not any(marker in top_name for marker in historical_markers):
        continue
    scanned_roots.append(str(top))
    for current, dirs, files in __import__("os").walk(top):
        relative_depth = len(Path(current).relative_to(top).parts)
        if relative_depth >= 7:
            dirs[:] = []
        if "predictions.npz" not in files or "metadata.json" not in files:
            continue
        output = Path(current)
        # A number of adapters retain a native-format mirror below the
        # canonical output directory.  Only formal/<window-id> is one window.
        if output.parent.name != "formal":
            continue
        meta_dataset = metadata_dataset(output / "metadata.json")
        if meta_dataset is not None and meta_dataset not in allowed:
            continue
        inferred = next(iter(matching)) if len(matching) == 1 else None
        dataset = meta_dataset if meta_dataset in allowed else inferred
        if dataset not in allowed:
            continue
        key = (dataset, str(top), method_for(top, output))
        row = records[key]
        row["count"] += 1
        if len(row["examples"]) < 3:
            row["examples"].append(str(output))
        if meta_dataset:
            row["metadata_dataset"].add(meta_dataset)

rows = []
for (dataset, root, method), row in sorted(records.items()):
    rows.append({
        "dataset": dataset,
        "candidate_root": root,
        "method": method,
        "valid_output_count": row["count"],
        "examples": row["examples"],
        "metadata_datasets": sorted(row["metadata_dataset"]),
    })
print(json.dumps({"status": "ok", "scope": str(base), "scanned_candidate_roots": scanned_roots,
                  "rows": rows}, sort_keys=True))
"""


HISTORICAL_BACKFILL_COVERAGE = {
    "h2o": {
        "target": 283,
        "methods": {
            "hawor": [
                "/mnt/workspace/sjc/eval_artifacts/formal_h2o_60f_20260822T112000Z_continuation/hawor",
                "/mnt/workspace/sjc/eval_artifacts/formal_h2o_hot3d_backfill_20260824T074500Z/hawor/hawor",
            ],
            "reviv4d": [
                "/mnt/workspace/sjc/eval_artifacts/formal_h2o_60f_20260822T112000Z_continuation/reviv4d",
                "/mnt/workspace/sjc/eval_artifacts/formal_h2o_hot3d_backfill_20260824T074500Z/reviv4d/reviv4d",
            ],
            "vggt_omega": [
                "/mnt/workspace/sjc/eval_artifacts/formal_h2o_60f_20260822T112000Z_continuation/vggt_omega",
                "/mnt/workspace/sjc/eval_artifacts/formal_h2o_hot3d_backfill_20260824T074500Z/vggt_omega/vggt_omega",
            ],
        },
    },
    "taco": {
        "target": 400,
        "methods": {
            "reviv4d": [
                "/mnt/workspace/sjc/eval_artifacts/formal_taco_60f_20260822T112000Z_continuation/reviv4d",
                "/mnt/workspace/sjc/eval_artifacts/formal_taco_hoi4d_backfill_20260825T103000Z/taco_reviv4d/reviv4d",
            ],
            "vggt": [
                "/mnt/workspace/sjc/eval_artifacts/formal_taco_60f_20260822T112000Z_continuation/vggt",
                "/mnt/workspace/sjc/eval_artifacts/formal_taco_hoi4d_backfill_20260825T103000Z/taco_vggt_retry_20260825T102500Z/vggt",
            ],
        },
    },
    "hoi4d": {
        "target": 461,
        "methods": {
            "hawor": [
                "/mnt/workspace/sjc/eval_artifacts/formal_hoi4d_60f_20260822T112000Z_continuation/hawor",
                "/mnt/workspace/sjc/eval_artifacts/formal_taco_hoi4d_backfill_20260825T103000Z/hoi4d_hawor/hawor",
            ],
            "reviv4d": [
                "/mnt/workspace/sjc/eval_artifacts/formal_hoi4d_60f_20260822T112000Z_continuation/reviv4d",
                "/mnt/workspace/sjc/eval_artifacts/formal_taco_hoi4d_backfill_20260825T103000Z/hoi4d_reviv4d/reviv4d",
            ],
            "vggt_omega": [
                "/mnt/workspace/sjc/eval_artifacts/formal_hoi4d_60f_20260822T112000Z_continuation/vggt_omega",
                "/mnt/workspace/sjc/eval_artifacts/formal_taco_hoi4d_backfill_20260825T103000Z/hoi4d_vggt_omega/vggt_omega",
            ],
        },
    },
}


HISTORICAL_BACKFILL_COVERAGE_PROBE = r"""
import json
import sys
from pathlib import Path

spec = json.loads(sys.argv[1])

def ids(root):
    root = Path(root)
    observed = set()
    for formal in (root / "formal", root / root.name / "formal"):
        try:
            observed.update(child.name for child in formal.iterdir()
                            if child.is_dir() and (child / "predictions.npz").is_file()
                            and (child / "metadata.json").is_file())
        except OSError:
            continue
    return observed

observed = {}
for dataset, dataset_spec in spec.items():
    target = int(dataset_spec["target"])
    rows = {}
    for method, roots in dataset_spec["methods"].items():
        main, backfill = map(ids, roots)
        overlap = main & backfill
        rows[method] = {
            "main_count": len(main), "backfill_count": len(backfill),
            "overlap_count": len(overlap), "union_count": len(main | backfill),
            "target": target,
            "verified": not overlap and len(main | backfill) == target,
        }
    observed[dataset] = rows
print(json.dumps(observed, sort_keys=True))
"""


CONTACT_ARTIFACT_AUDIT_PROBE = r"""
import json
import os
from collections import defaultdict
from pathlib import Path

base = Path("/mnt/workspace/sjc/eval_artifacts")
data_base = Path("/mnt/workspace/sjc/DATA")
allowed = set(json.loads(__import__("sys").argv[1]))
aliases = {
    "h2o": ("h2o",), "hot3d": ("hot3d",), "arctic": ("arctic",),
    "oakink_v2": ("oakink", "oaklink"), "taco": ("taco",), "hoi4d": ("hoi4d",),
}
contact_methods = {"contactopt", "s2contact"}

def datasets_for(text):
    text = text.lower().replace("-", "_")
    return [dataset for dataset, words in aliases.items() if any(word in text for word in words)]

def has_contact_method(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    found = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                if str(key).lower().replace("-", "_") in contact_methods:
                    found.add(str(key))
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
    return sorted(found)

rows = defaultdict(lambda: {"prediction_pairs": 0, "reports": []})
for top in sorted(base.iterdir() if base.is_dir() else []):
    if not top.is_dir():
        continue
    datasets = datasets_for(top.name)
    datasets = [dataset for dataset in datasets if dataset in allowed]
    if not datasets:
        continue
    for current, dirs, files in os.walk(top):
        current_path = Path(current)
        depth = len(current_path.relative_to(top).parts)
        if depth >= 7:
            dirs[:] = []
        contact_path = "contact" in str(current_path).lower()
        if contact_path and {"predictions.npz", "metadata.json"} <= set(files) and current_path.parent.name == "formal":
            lower_path = str(current_path).lower().replace("-", "_")
            method = "contactopt" if "contactopt" in lower_path else "s2contact" if "s2contact" in lower_path or "/s2_" in lower_path else "unknown"
            for dataset in datasets:
                row = rows[(dataset, str(top))]
                row["prediction_pairs"] += 1
                row.setdefault("prediction_counts", defaultdict(int))[method] += 1
                examples = row.setdefault("prediction_examples", defaultdict(list))[method]
                if len(examples) < 2:
                    examples.append(str(current_path))
            dirs[:] = []
        for name in ("report.json", "hand6_report.json"):
            if name not in files:
                continue
            methods = has_contact_method(current_path / name)
            if methods:
                for dataset in datasets:
                    row = rows[(dataset, str(top))]
                    if len(row["reports"]) < 8:
                        row["reports"].append({"path": str(current_path / name), "methods": methods})

caches = defaultdict(list)
for dataset, words in aliases.items():
    if dataset not in allowed:
        continue
    for word in words[:1]:
        cache_dir = data_base / f"{word}_contact_baseline" / "cache"
        try:
            for path in sorted(cache_dir.iterdir()):
                lower_name = path.name.lower().replace("-", "_")
                if path.is_file() and ("contactopt" in lower_name or "s2contact" in lower_name or lower_name.startswith("s2_")):
                    caches[dataset].append({"path": str(path), "bytes": path.stat().st_size})
        except OSError:
            continue

all_rows = [
    {"dataset": dataset, "candidate_root": root,
     **{key: (dict(counts) if key in {"prediction_counts", "prediction_examples"} else counts) for key, counts in value.items()}}
    for (dataset, root), value in sorted(rows.items())
]
result_rows = all_rows[:24]
print(json.dumps({"status": "ok", "scope": "six datasets contact artifacts, read-only", "rows": result_rows,
                  "row_count": len(all_rows), "truncated": len(all_rows) > len(result_rows),
                  "geometry_caches": dict(caches)}, sort_keys=True))
"""


CONTACT_PREDICTION_COVERAGE = {
    "arctic": {"target": 434, "methods": {
        "contactopt": [
            "/mnt/workspace/sjc/eval_artifacts/formal_arctic_contact_retry_400_20260823T231000Z/contactopt",
            "/mnt/workspace/sjc/eval_artifacts/formal_arctic_contact_tail34_20260823T151000Z/contactopt"],
        "s2contact": [
            "/mnt/workspace/sjc/eval_artifacts/formal_arctic_contact_retry_400_20260823T231000Z/s2contact",
            "/mnt/workspace/sjc/eval_artifacts/formal_arctic_contact_tail34_20260823T151000Z/s2contact"]}},
    "oakink_v2": {"target": 400, "methods": {
        "contactopt": ["/mnt/workspace/sjc/eval_artifacts/formal_oakink_v2_contact_400_20260823T234000Z/contactopt"],
        "s2contact": ["/mnt/workspace/sjc/eval_artifacts/formal_oakink_v2_contact_400_20260823T234000Z/s2contact"]}},
}


CONTACT_PREDICTION_COVERAGE_PROBE = r"""
import json
import sys
from pathlib import Path

spec = json.loads(sys.argv[1])
def ids(root):
    root = Path(root)
    found = set()
    for formal in (root / "formal", root / root.name / "formal"):
        try:
            found.update(child.name for child in formal.iterdir()
                         if child.is_dir() and (child / "predictions.npz").is_file()
                         and (child / "metadata.json").is_file())
        except OSError:
            continue
    return found
result = {}
for dataset, dataset_spec in spec.items():
    result[dataset] = {}
    for method, roots in dataset_spec["methods"].items():
        sets = [ids(root) for root in roots]
        union = set().union(*sets)
        overlap = sum(len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))
        result[dataset][method] = {"counts": [len(value) for value in sets], "union_count": len(union),
                                   "overlap_count": overlap, "target": dataset_spec["target"],
                                   "verified": not overlap and len(union) == dataset_spec["target"]}
print(json.dumps(result, sort_keys=True))
"""


class TaskError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TaskError(f"NOT_FOUND:{path}") from error
    if not isinstance(value, dict):
        raise TaskError(f"INVALID_JSON_OBJECT:{path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_registry(path: Path) -> dict[str, Any]:
    registry = read_json(path)
    if registry.get("schema_version") != "evaluation_task_registry_v1":
        raise TaskError("UNSUPPORTED_REGISTRY")
    return registry


def load_instance_config(path: Path) -> dict[str, Any]:
    """Read one active DSW allocation; never persist its SSH key path in the registry."""
    value = read_json(path)
    required = ("instance_id", "host", "port", "key", "entrance")
    if not isinstance(value, dict) or any(not value.get(name) for name in required):
        raise TaskError("INSTANCE_CONFIG_INCOMPLETE")
    try:
        port = int(value["port"])
    except (TypeError, ValueError) as error:
        raise TaskError("INSTANCE_PORT_INVALID") from error
    if not 1 <= port <= 65535 or value["entrance"] not in ("/mnt/workspace", "/mnt/cpfs"):
        raise TaskError("INSTANCE_CONFIG_INVALID")
    key = Path(os.path.expanduser(str(value["key"])))
    if not key.is_file():
        raise TaskError(f"INSTANCE_KEY_MISSING:{key}")
    if key.stat().st_mode & 0o077:
        raise TaskError(f"INSTANCE_KEY_PERMISSIONS:{key}")
    return {"instance_id": str(value["instance_id"]), "host": str(value["host"]),
            "port": port, "key": str(key), "entrance": str(value["entrance"]),
            "remote_python": str(value.get("remote_python", "python3"))}


def apply_instance_config(registry: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    """Apply a connection overlay in memory; historical run records stay unchanged."""
    registry["_current_instance"] = instance
    template = registry.setdefault("run_template", {})
    template["ssh"] = {"host": instance["host"], "key": instance["key"]}
    template["remote_python"] = instance["remote_python"]
    template["storage_probe"] = {"node": instance["port"], "mount": instance["entrance"]}
    template["_current_instance"] = instance
    policy = registry.setdefault("policy", {})
    policy["resource_nodes"] = [instance["port"]]
    policy["resource_node_entrances"] = {str(instance["port"]): instance["entrance"]}
    return registry


def run_jobs(registry: dict[str, Any], run: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    section = str(run.get("state_section", "jobs"))
    keys = run.get("job_keys")
    if keys is None:
        keys = ([str(run["dataset"])] if section != "jobs"
                else [f"{run['dataset']}::{method}" for method in run["methods"]])
    labels = run.get("job_labels", {})
    jobs = []
    for key in keys:
        key = str(key)
        default_label = key.split("::", 1)[-1] if section == "jobs" else key
        jobs.append((str(labels.get(key, default_label)), key))
    return section, jobs


def registration_identity(registry: dict[str, Any], logical_task_id: str,
                          run_record: dict[str, Any]) -> dict[str, Any]:
    logical = registry["logical_tasks"][logical_task_id]
    methods = registry["method_sets"][logical["method_set"]]
    run = {**registry.get("run_template", {}), **run_record, **logical, "methods": methods}
    section, jobs = run_jobs(registry, run)
    return {
        "logical_task_id": logical_task_id,
        "task_type": run.get("task_type", "evaluation"),
        "scheduler_id": run.get("scheduler_id"),
        "state_file": run.get("state_file"),
        "state_section": section,
        "job_keys": [key for _, key in jobs],
        "output_root": run.get("output_root"),
        "output_root_overrides": run.get("output_root_overrides", {}),
        "method_root_output": bool(run.get("method_root_output")),
        "target_windows_per_method": run.get("target_windows_per_method"),
        "identity": run.get("identity", {}),
    }


def registration_key(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def register_run(registry_path: Path, *, logical_task_id: str, dataset: str,
                 method_set: str, phase: str, protocol: str, run_record: dict[str, Any],
                 run_id: str | None = None, methods: list[str] | None = None) -> dict[str, Any]:
    """Atomically create one run per stable submission identity, or return the existing run."""
    lock_path = (DEFAULT_CONTROLS.parent / "task_registry.lock"
                 if registry_path.resolve() == DEFAULT_REGISTRY.resolve()
                 else registry_path.with_suffix(registry_path.suffix + ".lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = load_registry(registry_path)
        method_sets = registry.setdefault("method_sets", {})
        if method_set not in method_sets:
            if not methods:
                raise TaskError(f"METHOD_SET_NOT_REGISTERED:{method_set}")
            method_sets[method_set] = methods
        elif methods and method_sets[method_set] != methods:
            raise TaskError(f"METHOD_SET_CONFLICT:{method_set}")

        logical_spec = {
            "dataset": dataset,
            "method_set": method_set,
            "phase": phase,
            "protocol": protocol,
        }
        logical_tasks = registry.setdefault("logical_tasks", {})
        existing_logical = logical_tasks.get(logical_task_id)
        if existing_logical is not None:
            observed = {key: existing_logical.get(key) for key in logical_spec}
            if observed != logical_spec:
                raise TaskError(f"LOGICAL_TASK_CONFLICT:{logical_task_id}")
        else:
            logical_tasks[logical_task_id] = logical_spec

        candidate = {**run_record, "logical_task_id": logical_task_id}
        identity = registration_identity(registry, logical_task_id, candidate)
        key = registration_key(identity)
        runs = registry.setdefault("runs", {})
        for existing_id, existing_run in runs.items():
            if existing_run.get("logical_task_id") != logical_task_id:
                continue
            existing_identity = registration_identity(registry, logical_task_id, existing_run)
            existing_key = existing_run.get("registration_key")
            if existing_key is None:
                existing_key = registration_key(existing_identity)
            if existing_key == key:
                return {"run_id": existing_id, "task_id": logical_task_id, "created": False,
                        "registration_key": key}
            if existing_identity["output_root"] == identity["output_root"]:
                raise TaskError(f"OUTPUT_ROOT_ALREADY_REGISTERED:{existing_id}")

        if run_id is None:
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", logical_task_id).strip("-").lower()
            run_id = f"{slug}-{key[:12]}"
        if run_id in runs:
            raise TaskError(f"RUN_ID_COLLISION:{run_id}")
        candidate["registration_key"] = key
        candidate.setdefault("created_at_epoch", int(time.time()))
        runs[run_id] = candidate
        logical_tasks[logical_task_id]["latest"] = run_id
        atomic_json(registry_path, registry)
        # A newly registered run must have a valid pending state before any
        # start/inspect operation.  Older registrations predated this
        # initialization and could therefore fail with NOT_FOUND on start.
        state_file = candidate.get("state_file")
        if state_file:
            state_path = PROJECT_ROOT / str(state_file)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            section = str(candidate.get("state_section", "jobs"))
            keys = candidate.get("job_keys", [])
            if isinstance(keys, str):
                keys = [keys]
            state = read_json(state_path) if state_path.exists() else {}
            jobs = state.setdefault(section, {})
            if not isinstance(jobs, dict):
                raise TaskError(f"INVALID_STATE_SECTION:{section}")
            for job in (keys or [method_set]):
                jobs.setdefault(str(job), {"status": "pending", "method": str(job)})
            atomic_json(state_path, state)
        return {"run_id": run_id, "task_id": logical_task_id, "created": True,
                "registration_key": key}


def handoff_pad_bimanual_resume(registry_path: Path, instance: dict[str, Any],
                                run_id: str, gpu: int) -> dict[str, Any]:
    """Move one failed partial PAD lane to a shared-mount node with explicit resume."""
    if instance["port"] != 5000 or instance["entrance"] != "/mnt/workspace":
        raise TaskError("PAD_RESUME_HANDOFF_REQUIRES_5000_WORKSPACE")
    lock_path = DEFAULT_CONTROLS.parent / "task_registry.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = load_registry(registry_path)
        run = registry.get("runs", {}).get(run_id)
        if not run or not str(run.get("logical_task_id", "")).startswith("evaluation:pad_hand:bimanual:lane:"):
            raise TaskError("PAD_BIMANUAL_LANE_RUN_REQUIRED")
        launch = run.get("launch", {})
        state_path = PROJECT_ROOT / str(run["state_file"])
        state = read_json(state_path)
        keys = run.get("job_keys", [])
        if len(keys) != 1 or not isinstance(state.get(run.get("state_section", "jobs"), {}).get(keys[0]), dict):
            raise TaskError("PAD_RESUME_EXACT_JOB_REQUIRED")
        job = state[run.get("state_section", "jobs")][keys[0]]
        attempt = int(job.get("launch_attempt", 0))
        log_path = Path(str(launch["log_path"]))
        failed_log = str(log_path.with_name(f"{log_path.stem}.attempt{attempt}{log_path.suffix}"))
        script = ("import json,sys;from pathlib import Path;"
                  "root=Path(sys.argv[1]);log=Path(sys.argv[2]);"
                  "progress=list(root.glob('*/progress.jsonl'));"
                  "text=log.read_text(errors='replace') if log.is_file() else '';"
                  "print(json.dumps({'ok':bool(progress) and not (root/'COMPLETE').exists() and "
                  "'refusing implicit partial rerun' in text,'progress_files':[str(p) for p in progress]}))")
        probe = {**registry.get("run_template", {}), **run,
                 "ssh": {"host": instance["host"], "key": instance["key"]},
                 "remote_python": instance["remote_python"], "_current_instance": instance}
        completed = _ssh(probe, instance["port"], "python3 -c " + shlex.quote(script) + " "
                         + shlex.quote(str(run["output_root"])) + " " + shlex.quote(failed_log))
        try:
            evidence = json.loads(completed.stdout)
        except ValueError as error:
            raise TaskError("PAD_RESUME_HANDOFF_INVALID_AUDIT") from error
        if completed.returncode or not evidence.get("ok"):
            raise TaskError("PAD_RESUME_HANDOFF_AUDIT_FAILED")
        support = "formal_evaluation/run_pad_bimanual_lane.py"
        worker_sha = hashlib.sha256((PROJECT_ROOT / support).read_bytes()).hexdigest()
        previous = {name: job.get(name) for name in ("node", "pid", "pgid", "handle_path", "status")}
        launch.update({
            "node_candidates": [instance["port"]], "gpu_candidates": [gpu],
            "deploy_before_preflight": True,
            "support_files": list(dict.fromkeys([*launch.get("support_files", []), support])),
            "command": ("set -euo pipefail\nexport CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH='{worktree}' "
                        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
                        "exec '{python}' '{runtime_root}/formal_evaluation/run_pad_bimanual_lane.py' "
                        "--resume --spec '{runtime_root}/.auto_scheduler/pad_bimanual_20260914/"
                        "lane_g6_hot3d_arctic_v2_spec.json' --output-root '{output_root}'"),
        })
        run["identity"] = {**run.get("identity", {}), "instance_id": instance["instance_id"],
                           "resume_worker_sha256": worker_sha, "handoff_from_instance":
                           run.get("identity", {}).get("instance_id")}
        run["registration_key"] = registration_key(registration_identity(registry, run["logical_task_id"], run))
        history = list(job.get("handoff_history", []))
        history.append({"at_epoch": int(time.time()), "from": previous, "to_node": instance["port"],
                        "reason": "explicit_partial_resume_after_5001_disconnect"})
        job.update({"status": "queue_exited_needs_audit", "handoff_history": history,
                    "node": instance["port"], "gpu": gpu})
        for name in ("pid", "pgid", "handle_path"):
            job.pop(name, None)
        atomic_json(registry_path, registry)
        atomic_json(state_path, state)
        return {"run_id": run_id, "node": instance["port"], "gpu": gpu,
                "resume_worker_sha256": worker_sha, "progress_files": evidence["progress_files"],
                "status": "handoff_registered"}


def handoff_endpoint_gallery(registry_path: Path, instance: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Continue the exact incomplete endpoint gallery on the shared 5000 workspace."""
    if instance["port"] != 5000 or instance["entrance"] != "/mnt/workspace":
        raise TaskError("ENDPOINT_GALLERY_HANDOFF_REQUIRES_5000_WORKSPACE")
    lock_path = DEFAULT_CONTROLS.parent / "task_registry.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = load_registry(registry_path)
        run = registry.get("runs", {}).get(run_id)
        if not run or run.get("identity", {}).get("ego_completion_commit") != "8fc061a615895bd3b5a556f7387bae306e32d9db":
            raise TaskError("ENDPOINT_GALLERY_RUN_REQUIRED")
        state_path = PROJECT_ROOT / str(run["state_file"])
        state = read_json(state_path)
        keys = run.get("job_keys", [])
        if len(keys) != 1 or not isinstance(state.get(run.get("state_section", "jobs"), {}).get(keys[0]), dict):
            raise TaskError("ENDPOINT_GALLERY_EXACT_JOB_REQUIRED")
        job = state[run.get("state_section", "jobs")][keys[0]]
        repair_id = "evaluation-pad-hand-bimanual-gallery-missing40-hot3d-oakink-v2-20260915"
        repair = registry.get("runs", {}).get(repair_id)
        if not repair or repair.get("output_root") != "/mnt/workspace/sjc/DATA/eval_artifacts/pad_hand_bimanual_gallery_missing40_20260915":
            raise TaskError("ENDPOINT_GALLERY_PAD_REPAIR_NOT_REGISTERED")
        repair_state = read_json(PROJECT_ROOT / repair["state_file"])
        repair_job = repair_state.get(repair.get("state_section", "jobs"), {}).get("hot3d_oakink_v2::pad_hand")
        if not isinstance(repair_job, dict) or repair_job.get("status") not in {"running", "queue_exited_needs_audit", "done"}:
            raise TaskError("ENDPOINT_GALLERY_PAD_REPAIR_NOT_ACTIVE")
        script = ("import json,sys;from pathlib import Path;root=Path(sys.argv[1]);repair=Path(sys.argv[2]);"
                  "png={p.stem for p in (root/'png_gallery').glob('*.png')};"
                  "mp4={p.stem for p in (root/'video_gallery').glob('*.mp4') if '.partial.' not in p.name};"
                  "partial=list((root/'video_gallery').glob('*partial*'));summary=root/'summary.json';"
                  "data=json.loads(summary.read_text()) if summary.is_file() else {};"
                  "repair_summary=repair/'summary.json';"
                  "repair_data=json.loads(repair_summary.read_text()) if repair_summary.is_file() else {};"
                  "hot3d=repair/'hot3d'/'summary.json';oak=repair/'oakink_v2'/'summary.json';"
                  "hot3d_data=json.loads(hot3d.read_text()) if hot3d.is_file() else {};"
                  "oak_data=json.loads(oak.read_text()) if oak.is_file() else {};"
                  "repair_ok=(repair/'COMPLETE').is_file() and repair_data.get('status')=='complete' and "
                  "repair_data.get('completed_windows')==200 and hot3d_data.get('completed_windows')==165 and "
                  "oak_data.get('completed_windows')==35;"
                  "print(json.dumps({'ok':repair_ok and len(png)==64 and png==mp4 and not partial and "
                  "not (root/'COMPLETE').exists() and data.get('status')=='incomplete' and "
                  "data.get('target')==104 and data.get('completed')==64 and len(data.get('failures',[]))==40,"
                  "'pad_repair_complete':repair_ok,'pad_repair_windows':repair_data.get('completed_windows'),"
                  "'png_count':len(png),'mp4_count':len(mp4),'failure_count':len(data.get('failures',[]))}))")
        probe = {**registry.get("run_template", {}), **run,
                 "ssh": {"host": instance["host"], "key": instance["key"]},
                 "remote_python": instance["remote_python"], "_current_instance": instance}
        completed = _ssh(probe, instance["port"], "python3 -c " + shlex.quote(script) + " "
                         + shlex.quote(str(run["output_root"])) + " "
                         + shlex.quote(str(repair["output_root"])))
        try:
            evidence = json.loads(completed.stdout)
        except ValueError as error:
            raise TaskError("ENDPOINT_GALLERY_HANDOFF_INVALID_AUDIT") from error
        if completed.returncode or not evidence.get("ok"):
            raise TaskError("ENDPOINT_GALLERY_HANDOFF_AUDIT_FAILED")
        launch = run["launch"]
        previous = {name: job.get(name) for name in ("node", "pid", "pgid", "handle_path", "status")}
        launch["node_candidates"] = [instance["port"]]
        if " --resume" not in launch["command"]:
            launch["command"] += " --resume"
        run["identity"] = {**run.get("identity", {}), "instance_id": instance["instance_id"],
                           "reader_node": instance["port"], "handoff_from_instance":
                           run.get("identity", {}).get("instance_id"),
                           "source_alignment_sha256": hashlib.sha256(
                               (PROJECT_ROOT / launch["source_alignment_relative"]).read_bytes()).hexdigest(),
                           "pad_repair_run_id": repair_id}
        run["registration_key"] = registration_key(registration_identity(registry, run["logical_task_id"], run))
        history = list(job.get("handoff_history", []))
        history.append({"at_epoch": int(time.time()), "from": previous, "to_node": instance["port"],
                        "reason": "resume_verified_64_pairs_and_render_missing40_after_pad_repair"})
        job.update({"status": "queue_exited_needs_audit", "handoff_history": history,
                    "node": instance["port"]})
        for name in ("pid", "pgid", "handle_path"):
            job.pop(name, None)
        atomic_json(registry_path, registry)
        atomic_json(state_path, state)
        return {"run_id": run_id, "node": instance["port"], "status": "resume_registered",
                "existing_pairs": evidence["png_count"], "remaining_segments": 40,
                "pad_repair_complete": evidence["pad_repair_complete"],
                "pad_repair_run_id": repair_id}


def register_10s_gallery(registry_path: Path, instance: dict[str, Any], *, pilot: bool = False,
                         shard: int | None = None) -> dict[str, Any]:
    """Register one exact 5000 gallery run with a frozen source identity."""
    if instance["port"] != 5000 or instance["entrance"] != "/mnt/workspace":
        raise TaskError("VISUALIZATION_REQUIRES_5000_WORKSPACE_INSTANCE")
    relative = "visualization/batch_10s_177_p95_wmpjpe_20260912"
    manifest = PROJECT_ROOT / relative / "selected_manifest.jsonl"
    alignment = PROJECT_ROOT / relative / "source_alignment_5000.json"
    spec = json.loads(alignment.read_text())
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if spec.get("manifest_sha256") != digest or spec.get("status") != "audited_all_177":
        raise TaskError("VISUALIZATION_SOURCE_ALIGNMENT_NOT_VERIFIED")
    selected = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(selected) != 177 or len({row["segment_id"] for row in selected}) != 177:
        raise TaskError("VISUALIZATION_MANIFEST_NOT_177_UNIQUE_SEGMENTS")
    if pilot and shard is not None:
        raise TaskError("VISUALIZATION_PILOT_AND_SHARD_EXCLUSIVE")
    shard_ranges = {2: (45, 89), 3: (89, 133), 4: (133, 177)}
    if shard is not None and shard not in shard_ranges:
        raise TaskError("VISUALIZATION_SHARD_MUST_BE_2_3_OR_4")
    segment = "oakink_v2__0219c2db05f61b5f" if pilot else None
    if segment and sum(row["segment_id"] == segment for row in selected) != 1:
        raise TaskError("VISUALIZATION_PILOT_SEGMENT_MISSING")
    suffix = f"parallel4_shard{shard}" if shard else "pilot_oakink" if pilot else "full177"
    output_root = f"/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_joint8_p95_wmpjpe_20260913_{suffix}"
    runtime = output_root + "/runtime"
    task_id = f"visualization:five:10s-gallery:joint8-p95-wmpjpe:5000:{suffix}:20260913"
    run_id = f"visualization-five-10s-gallery-5000-{suffix}-20260913"
    command = f"exec python3 {{runtime_root}}/formal_evaluation/render_10s_gallery_worker.py --runtime {{runtime_root}} --output {{output_root}}"
    if segment:
        command += " --segment " + shlex.quote(segment)
    if shard is not None:
        start_index, end_index = shard_ranges[shard]
        command += f" --start-index {start_index} --end-index {end_index}"
    support_files = [
        "formal_evaluation/remote_task_control.py",
        "formal_evaluation/render_10s_gallery_worker.py",
        relative + "/selected_manifest.jsonl.gz",
        relative + "/source_alignment_5000.json",
        relative + "/render_full.py",
        relative + "/video_io.py",
        "visualization/batch_10s_41_p95_wmpjpe_20260910/render_batch.py",
        "visualization/six_same_mask_v7_p95_bestego_20260909/render_examples.py",
        "visualization/six_same_mask_v7_p95_bestego_20260909/render_config.json",
        "visualization/six_same_mask_v7_p95_bestego_20260909/mano_195_to_778.npz",
        "visualization/h2o_result3_stride5_minmpjpe_5frames_20260909/render.py",
    ]
    result = register_run(
        registry_path, logical_task_id=task_id, dataset="five",
        method_set="10s_gallery_six_rows", methods=["egofound3r", "wilor", "hawor", "reviv4d", "pad_hand", "gt"],
        phase="visualization", protocol="300sample-30fps-joint8-p95-wmpjpe", run_id=run_id,
        run_record={
            "task_type": "visualization", "scheduler_id": task_id,
            "state_file": f".auto_scheduler/10s_gallery_5000_{suffix}_20260913.json",
            "state_section": "jobs", "job_keys": ["five::gallery"],
            "output_root": output_root, "target_windows_per_method":
                1 if pilot else shard_ranges[shard][1] - shard_ranges[shard][0] if shard else 177,
            "identity": {"instance_id": instance["instance_id"], "manifest_sha256": digest,
                         "source_alignment_sha256": hashlib.sha256(alignment.read_bytes()).hexdigest(),
                         "pilot_segment": segment, "reader_node": 5000,
                         **({"shard_start": shard_ranges[shard][0], "shard_end": shard_ranges[shard][1],
                             "original_run_id": "visualization-five-10s-gallery-5000-full177-20260913"}
                            if shard else {})},
            "launch": {"node_candidates": [5000], "resource": "visualization", "method": "gallery",
                       "runtime_root": runtime, "manifest_sha256": digest,
                       "support_files": support_files, "deploy_before_preflight": True,
                       "command": command,
                       "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                       "handle_path": output_root + "/control/handle.json",
                       "log_path": output_root + "/logs/worker.log",
                       "values": {"runtime_root": runtime, "output_root": output_root}},
        })
    if shard is not None:
        return {**result, "output_root": output_root, "manifest_sha256": digest,
                "start_index": shard_ranges[shard][0], "end_index": shard_ranges[shard][1]}
    # A pending run may predate the compressed deployment list.  Keep its
    # stable registration identity and output root; never edit a started run.
    lock_path = (DEFAULT_CONTROLS.parent / "task_registry.lock"
                 if registry_path.resolve() == DEFAULT_REGISTRY.resolve()
                 else registry_path.with_suffix(registry_path.suffix + ".lock"))
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = load_registry(registry_path)
        recorded = registry["runs"][run_id]
        old = recorded["launch"]["support_files"]
        if old != support_files:
            expected_old = [item.replace("selected_manifest.jsonl.gz", "selected_manifest.jsonl")
                            for item in support_files]
            state = read_json(PROJECT_ROOT / recorded["state_file"])
            if old != expected_old or state["jobs"]["five::gallery"]["status"] != "pending":
                raise TaskError("VISUALIZATION_DEPLOYMENT_REVISION_NOT_SAFE")
            recorded["launch"]["support_files"] = support_files
            atomic_json(registry_path, registry)
            result["deployment_updated"] = True
    return {**result, "output_root": output_root, "manifest_sha256": digest,
            "segments": 1 if pilot else 177}


def register_10s_endpoint_gallery(registry_path: Path, instance: dict[str, Any], *, pilot: bool = False,
                                  hawor_native: bool = False, hand_zoom: bool = False,
                                  aux_methods: bool = False) -> dict[str, Any]:
    """Register the frozen endpoint-renderable 104-segment gallery."""
    # The auxiliary pilot only reads shared /mnt/workspace and authorized OSS
    # inputs, so it can move to the live 5001 allocation when 5000 expires.
    required_port = 5001 if aux_methods else 5000 if hawor_native else 5001
    if instance["port"] != required_port or instance["entrance"] != "/mnt/workspace":
        raise TaskError(f"ENDPOINT_VISUALIZATION_REQUIRES_{required_port}_WORKSPACE_INSTANCE")
    relative = "visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914"
    manifest_gz = PROJECT_ROOT / relative / "selected_manifest_hydrated.jsonl.gz"
    manifest_bytes = gzip.decompress(manifest_gz.read_bytes())
    rows = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    alignment_name = (("source_alignment_auxmethods_pilot_5001.json" if pilot else
                       "source_alignment_auxmethods_full104_5001.json") if aux_methods else
                      "source_alignment_hawor_native_5000.json" if hawor_native else
                      "source_alignment_5001.json")
    alignment = PROJECT_ROOT / relative / alignment_name
    spec = json.loads(alignment.read_text())
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if (len(rows) != 104 or len({row["segment_id"] for row in rows}) != 104
            or spec.get("manifest_sha256") != digest
            or spec.get("ego_interpolation", {}).get("source_commit") != "8fc061a615895bd3b5a556f7387bae306e32d9db"):
        raise TaskError("ENDPOINT_VISUALIZATION_SOURCE_IDENTITY_INVALID")
    if aux_methods and not (hand_zoom and hawor_native):
        raise TaskError("AUX_METHODS_REQUIRES_HAWOR_NATIVE_ZOOM")
    if hand_zoom and not hawor_native:
        raise TaskError("HAND_ZOOM_REQUIRES_HAWOR_NATIVE")
    segment = ("arctic__574b75d44d53e3ee" if hand_zoom and pilot else
               "h2o__131e283e117e01c6" if pilot else None)
    if segment and sum(row["segment_id"] == segment for row in rows) != 1:
        raise TaskError("ENDPOINT_VISUALIZATION_PILOT_MISSING")
    suffix = (("pilot_arctic_auxmethods_handzoom_panels_v5_5001" if pilot else
               "full104_auxmethods_handzoom_panels_v5_5001_r1") if aux_methods else
              "pilot_arctic_handzoom_v2" if hand_zoom else
              "pilot_h2o_v3" if hawor_native and pilot else
              "pilot_h2o" if pilot else "full104_parallel16")
    if hawor_native:
        output_root = ("/mnt/workspace/sjc/DATA/eval_artifacts/"
                       f"gallery_10s_endpoint104_hawor_native_unmasked_20260915_{suffix}")
    else:
        suffix = "pilot_h2o_v2" if pilot else suffix
        output_root = f"/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_endpoint104_ego_bracketed_8fc061a_20260914_{suffix}"
    runtime = output_root + "/runtime"
    source_tag = ("auxmethods-hawor-native-handzoom-panels" if aux_methods else
                  "hawor-native-unmasked-handzoom" if hand_zoom else
                  "hawor-native-unmasked" if hawor_native else "legacy-hawor")
    date_tag = "20260916" if aux_methods else "20260915" if hawor_native else "20260914"
    task_id = f"visualization:four:10s-gallery:endpoint104:{required_port}:{source_tag}:{suffix}:{date_tag}"
    run_id = f"visualization-four-10s-gallery-endpoint104-{required_port}-{source_tag}-{suffix}-{date_tag}"
    command = ("exec /mnt/workspace/sjc/envs/egofound3r/bin/python -u "
               "{runtime_root}/formal_evaluation/render_10s_endpoint_gallery_worker.py"
               " --runtime {runtime_root} --output {output_root}"
               f" --workers {1 if pilot else 16} --source-alignment {alignment_name}")
    if segment:
        command += " --segment " + shlex.quote(segment)
    else:
        command += " --wait-pad-seconds 7200"
    support_files = [
        "formal_evaluation/remote_task_control.py",
        "formal_evaluation/render_10s_endpoint_gallery_worker.py",
        relative + "/selected_manifest_hydrated.jsonl.gz",
        relative + "/" + alignment_name,
        relative + "/render_full.py",
        relative + "/video_io.py",
        relative + "/ego_bracketed_fill.py",
        relative + "/runtime_source/batch_10s_41_p95_wmpjpe_20260910/render_batch.py",
        relative + "/runtime_source/six_same_mask_v7_p95_bestego_20260909/render_examples.py",
        relative + "/runtime_source/six_same_mask_v7_p95_bestego_20260909/render_config.json",
        relative + "/runtime_source/six_same_mask_v7_p95_bestego_20260909/mano_195_to_778.npz",
        relative + "/runtime_source/h2o_result3_stride5_minmpjpe_5frames_20260909/render.py",
    ]
    if aux_methods:
        support_files.append(relative + "/render_aux_pilot.py")
    registered_methods = (["egofound3r", "egofound3r_gt", "wilor", "hawor", "reviv4d",
                           "pad_hand", "egoforce", "dyn_hamr", "gt"] if aux_methods else
                          ["egofound3r", "wilor", "hawor", "reviv4d", "pad_hand", "gt"])
    result = register_run(
        registry_path, logical_task_id=task_id, dataset="four",
        method_set=(("10s_gallery_auxmethods_endpoint104_pilot" if pilot else
                     "10s_gallery_auxmethods_endpoint104_full") if aux_methods
                    else "10s_gallery_six_rows_endpoint104"),
        methods=registered_methods,
        phase="visualization",
        protocol=("300sample-30fps-joint8-p95-candidate-endpoint-renderable-fullframes-"
                  + ("auxmethods-native-and-gt-extrinsics-near-hand-panels" if aux_methods else
                     "hawor-native-camera-unmasked-slam-hand-focused-fixed-crop" if hand_zoom else
                     "hawor-native-camera-unmasked-slam" if hawor_native else "ego-bracketed")),
        run_id=run_id,
        run_record={
            "task_type": "visualization", "scheduler_id": task_id,
            "state_file": (f".auto_scheduler/10s_endpoint_gallery_{required_port}_{source_tag}_"
                           f"{suffix}_{date_tag}.json"),
            "state_section": "jobs", "job_keys": ["four::gallery"],
            "output_root": output_root, "target_windows_per_method": 1 if pilot else 104,
            "progress_target": 1 if pilot else 104,
            "identity": {
                "instance_id": instance["instance_id"], "manifest_sha256": digest,
                "source_alignment_sha256": hashlib.sha256(alignment.read_bytes()).hexdigest(),
                "reader_node": required_port, "pilot_segment": segment,
                "ego_completion_commit": "8fc061a615895bd3b5a556f7387bae306e32d9db",
                "visualization_mask": "none",
                "hawor_label": (spec.get("hawor_visualization_label") if hawor_native else "HaWoR"),
                "visualization_layout": ("auxiliary methods fixed near-hand crop with individual 2048 panels" if aux_methods else
                                         "hand-focused fixed per-method/view crop" if hand_zoom else
                                         "shared full-scene bounds"),
                **({"egoforce_run_ids": {
                        dataset: spec[dataset].get(
                            "egoforce_registered_audit_run_ids",
                            [spec[dataset].get("egoforce_registered_audit_run_id")])
                        for dataset in ("h2o", "hot3d", "arctic", "oakink_v2")},
                    "dyn_hamr_run_id": spec["arctic"]["dyn_hamr_registered_audit_run_id"],
                    "external_pose_labels": spec["external_pose_labels"],
                    "expected_panel_pngs": (54 if pilot else 5136),
                    "conditional_dyn_hamr_row": not pilot,
                    "expected_dyn_hamr_segments": (1 if pilot else 24),
                    "expected_dyn_hamr_windows": (1 if pilot else 27)}
                   if aux_methods else {}),
                "hawor_run_ids": ({dataset: spec[dataset]["hawor_run_id"]
                                   for dataset in ("h2o", "hot3d", "arctic", "oakink_v2")}
                                  if hawor_native else {}),
            },
            "launch": {
                "node_candidates": [required_port], "resource": "visualization", "method": "gallery",
                "runtime_root": runtime, "manifest_sha256": digest,
                "manifest_relative": relative + "/selected_manifest_hydrated.jsonl.gz",
                "source_alignment_relative": relative + "/" + alignment_name,
                "expected_segments": 104,
                "preflight_python": "/mnt/workspace/sjc/envs/egofound3r/bin/python",
                "support_files": support_files, "deploy_before_preflight": True,
                "command": command,
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "handle_path": output_root + "/control/handle.json",
                "log_path": output_root + "/logs/worker.log",
                "values": {"runtime_root": runtime, "output_root": output_root},
            },
        })
    if not result["created"]:
        lock_path = DEFAULT_CONTROLS.parent / "task_registry.lock"
        with lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            registry = load_registry(registry_path)
            recorded = registry["runs"][run_id]
            state = read_json(PROJECT_ROOT / recorded["state_file"])
            if state["jobs"]["four::gallery"]["status"] != "pending":
                raise TaskError("ENDPOINT_VISUALIZATION_DEPLOYMENT_REVISION_NOT_SAFE")
            recorded["launch"].update({
                "manifest_relative": relative + "/selected_manifest_hydrated.jsonl.gz",
                "source_alignment_relative": relative + "/" + alignment_name,
                "expected_segments": 104,
                "preflight_python": "/mnt/workspace/sjc/envs/egofound3r/bin/python",
                "command": command,
            })
            atomic_json(registry_path, registry)
            result["deployment_updated"] = True
    return {**result, "output_root": output_root, "manifest_sha256": digest,
            "segments": 1 if pilot else 104, "workers": 1 if pilot else 16}


def register_10s_pad_repair(registry_path: Path, instance: dict[str, Any]) -> dict[str, Any]:
    """Register the exact 200-window PAD-Hand repair needed by the missing 40 segments."""
    if instance["port"] != 5000 or instance["entrance"] != "/mnt/workspace":
        raise TaskError("PAD_REPAIR_REQUIRES_5000_WORKSPACE_INSTANCE")
    relative = ".auto_scheduler/pad_repair_missing40_20260915"
    spec_path = PROJECT_ROOT / relative / "spec.json"
    selected_path = PROJECT_ROOT / relative / "selected_segments.jsonl"
    spec = read_json(spec_path)
    selected = [json.loads(line) for line in selected_path.read_text().splitlines() if line.strip()]
    counts = {dataset: len(source.get("selected_window_ids", []))
              for dataset, source in spec.get("datasets", {}).items()}
    if (len(selected) != 40 or counts != {"hot3d": 165, "oakink_v2": 35}
            or spec.get("selected_segments") != 40):
        raise TaskError("PAD_REPAIR_SELECTION_INVALID")
    output_root = "/mnt/workspace/sjc/DATA/eval_artifacts/pad_hand_bimanual_gallery_missing40_20260915"
    runtime = output_root + "/runtime"
    worktree = "/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R-baselines_pad_repair_missing40_20260915"
    bundle_relative = ".auto_scheduler/pad_bimanual_20260914/pad_bimanual_6e443a67_thin.bundle"
    bundle_remote = runtime + "/" + bundle_relative
    task_id = "evaluation:pad_hand:bimanual:gallery-missing40:hot3d_oakink_v2:20260915"
    run_id = "evaluation-pad-hand-bimanual-gallery-missing40-hot3d-oakink-v2-20260915"
    command = ("set -euo pipefail\nexport CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH='{worktree}' "
               "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
               "exec '{python}' '{runtime_root}/formal_evaluation/run_pad_bimanual_lane.py' "
               "--spec '{runtime_root}/" + relative + "/spec.json' --output-root '{output_root}'")
    preflight_paths = [path for source in spec["datasets"].values()
                       for path in source.get("input_indices", [source.get("input_index")])]
    preflight_paths += [source["gt_index"] for source in spec["datasets"].values()]
    preflight_paths += [spec["runtime"]["checkpoint"], spec["runtime"]["source_root"],
                        spec["runtime"]["python"], spec["runtime"]["wilor_python"]]
    registry = load_registry(registry_path)
    existing = registry.get("runs", {}).get(run_id)
    if existing is not None:
        state_path = PROJECT_ROOT / existing["state_file"]
        state = read_json(state_path)
        job = state.get(existing.get("state_section", "jobs"), {}).get("hot3d_oakink_v2::pad_hand")
        if (existing.get("logical_task_id") != task_id or existing.get("output_root") != output_root
                or not isinstance(job, dict) or job.get("status") != "pending"):
            raise TaskError("PAD_REPAIR_REGISTERED_RUN_NOT_SAFE_TO_UPDATE")
        existing["identity"].update({
            "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "selected_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        })
        existing["launch"].update({
            "gpu_candidates": [2, 3, 7],
            "support_files": ["formal_evaluation/remote_task_control.py",
                              "formal_evaluation/run_pad_bimanual_lane.py",
                              relative + "/spec.json", relative + "/selected_segments.jsonl",
                              bundle_relative],
            "preflight_paths": preflight_paths,
        })
        existing["registration_key"] = registration_key(
            registration_identity(registry, existing["logical_task_id"], existing))
        atomic_json(registry_path, registry)
        return {"created": False, "updated_pending_registration": True, "run_id": run_id,
                "task_id": task_id, "output_root": output_root, "windows": 200,
                "segments": 40, "datasets": counts}
    result = register_run(
        registry_path, logical_task_id=task_id, dataset="hot3d_oakink_v2",
        method_set="pad_hand_bimanual_gallery_missing40", methods=["pad_hand"],
        phase="formal", protocol="60f-selected200-dual-hand-no-p95-filter", run_id=run_id,
        run_record={
            "task_type": "evaluation", "scheduler_id": task_id,
            "state_file": relative + "/state.json", "state_section": "jobs",
            "job_keys": ["hot3d_oakink_v2::pad_hand"], "output_root": output_root,
            "target_windows_per_method": 200,
            "identity": {"instance_id": instance["instance_id"],
                         "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                         "selected_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
                         "source_manifest_sha256": spec["source_manifest_sha256"],
                         "code_commit": spec["code_commit"]},
            "launch": {
                "node_candidates": [5000], "gpu_candidates": [2, 3, 7], "required_gpu_count": 1,
                "resource": "gpu", "method": "pad_hand", "runtime_root": runtime,
                "support_files": ["formal_evaluation/remote_task_control.py",
                                  "formal_evaluation/run_pad_bimanual_lane.py",
                                  relative + "/spec.json", relative + "/selected_segments.jsonl",
                                  bundle_relative],
                "deploy_before_preflight": True, "command": command,
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "handle_path": output_root + "/control/handle.json",
                "log_path": output_root + "/logs/worker.log",
                "audit_path": output_root + "/summary.json",
                "worktree": worktree,
                "worktree_source": "/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R-baselines_traincrop512_1078f15_20260908",
                "worktree_commit": spec["code_commit"],
                "worktree_base_commit": "1078f154d9f84367f0ae03027b2f53d2af46d8b6",
                "worktree_bundle": bundle_remote,
                "runtime_registry": worktree + "/formal_evaluation/config/baseline_runtime_registry_dsw.json",
                "preflight_paths": preflight_paths,
                "values": {"runtime_root": runtime, "output_root": output_root,
                           "worktree": worktree, "python": spec["runtime"]["python"]},
            },
        })
    return {**result, "output_root": output_root, "windows": 200,
            "segments": 40, "datasets": counts}


def register_10s_gallery_merge(registry_path: Path, instance: dict[str, Any]) -> dict[str, Any]:
    """Register the read-only four-source assembler for the frozen 177 pairs."""
    if instance["port"] != 5000 or instance["entrance"] != "/mnt/workspace":
        raise TaskError("VISUALIZATION_REQUIRES_5000_WORKSPACE_INSTANCE")
    registry = load_registry(registry_path)
    run_ids = ["visualization-five-10s-gallery-5000-full177-20260913"] + [
        f"visualization-five-10s-gallery-5000-parallel4_shard{index}-20260913"
        for index in (2, 3, 4)]
    sources = []
    for run_id in run_ids:
        run = registry["runs"].get(run_id)
        if not run or run.get("identity", {}).get("reader_node") != 5000:
            raise TaskError(f"VISUALIZATION_SOURCE_RUN_NOT_REGISTERED:{run_id}")
        sources.append(run["output_root"])
    relative = "visualization/batch_10s_177_p95_wmpjpe_20260912"
    manifest = PROJECT_ROOT / relative / "selected_manifest.jsonl"
    alignment = PROJECT_ROOT / relative / "source_alignment_5000.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if json.loads(alignment.read_text()).get("manifest_sha256") != digest:
        raise TaskError("VISUALIZATION_SOURCE_ALIGNMENT_NOT_VERIFIED")
    output_root = "/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_joint8_p95_wmpjpe_20260913_parallel4_gallery"
    runtime = output_root + "/runtime"
    task_id = "visualization:five:10s-gallery:joint8-p95-wmpjpe:5000:parallel4_gallery:20260913"
    run_id = "visualization-five-10s-gallery-5000-parallel4_gallery-20260913"
    command = ("exec python3 {runtime_root}/formal_evaluation/merge_10s_gallery_shards.py"
               " --runtime {runtime_root} --output {output_root}" +
               "".join(" --source " + shlex.quote(source) for source in sources))
    result = register_run(
        registry_path, logical_task_id=task_id, dataset="five",
        method_set="10s_gallery_six_rows", methods=["egofound3r", "wilor", "hawor", "reviv4d", "pad_hand", "gt"],
        phase="visualization", protocol="300sample-30fps-joint8-p95-wmpjpe", run_id=run_id,
        run_record={
            "task_type": "visualization", "scheduler_id": task_id,
            "state_file": ".auto_scheduler/10s_gallery_5000_parallel4_gallery_20260913.json",
            "state_section": "jobs", "job_keys": ["five::gallery"],
            "output_root": output_root, "target_windows_per_method": 177,
            "identity": {"instance_id": instance["instance_id"], "manifest_sha256": digest,
                         "source_alignment_sha256": hashlib.sha256(alignment.read_bytes()).hexdigest(),
                         "reader_node": 5000, "source_run_ids": run_ids, "source_bounds": [0, 45, 89, 133, 177]},
            "launch": {"node_candidates": [5000], "resource": "visualization", "method": "gallery",
                       "runtime_root": runtime, "manifest_sha256": digest,
                       "support_files": ["formal_evaluation/remote_task_control.py",
                                         "formal_evaluation/merge_10s_gallery_shards.py",
                                         relative + "/selected_manifest.jsonl.gz",
                                         relative + "/source_alignment_5000.json"],
                       "deploy_before_preflight": True, "command": command,
                       "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                       "handle_path": output_root + "/control/handle.json",
                       "log_path": output_root + "/logs/worker.log",
                       "values": {"runtime_root": runtime, "output_root": output_root}},
        })
    return {**result, "output_root": output_root, "manifest_sha256": digest,
            "source_run_ids": run_ids}


def register_10s_gallery_handoff(registry_path: Path, instance: dict[str, Any],
                                 *, shard_number: int | None = None, merge: bool = False,
                                 parallelism: int = 8) -> dict[str, Any]:
    """Register a frozen continuation renderer or its assembler."""
    if instance["port"] != 5000 or instance["entrance"] != "/mnt/workspace":
        raise TaskError("VISUALIZATION_REQUIRES_5000_WORKSPACE_INSTANCE")
    if (shard_number is None) == (not merge):
        raise TaskError("VISUALIZATION_HANDOFF_REQUIRES_SHARD_OR_MERGE")
    if parallelism not in (8, 16):
        raise TaskError("VISUALIZATION_HANDOFF_PARALLELISM_MUST_BE_8_OR_16")
    if shard_number is not None and not 1 <= shard_number <= parallelism:
        raise TaskError(f"VISUALIZATION_HANDOFF_SHARD_MUST_BE_1_TO_{parallelism}")
    relative = "visualization/batch_10s_177_p95_wmpjpe_20260912"
    plan_relative = relative + f"/parallel{parallelism}_handoff_20260913/plan.json"
    plan_bytes = (PROJECT_ROOT / plan_relative).read_bytes()
    plan = json.loads(plan_bytes)
    plan_digest = hashlib.sha256(plan_bytes).hexdigest()
    manifest = PROJECT_ROOT / relative / "selected_manifest.jsonl"
    alignment = PROJECT_ROOT / relative / "source_alignment_5000.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if (plan.get("schema") != f"10s_gallery_parallel{parallelism}_handoff_v1"
            or plan.get("manifest_sha256") != digest
            or json.loads(alignment.read_text()).get("manifest_sha256") != digest
            or plan.get("completed_count") != len(plan.get("completed", []))
            or plan.get("remaining_count") != sum(len(shard["indices"]) for shard in plan.get("shards", []))
            or len(plan.get("shards", [])) != parallelism):
        raise TaskError("VISUALIZATION_HANDOFF_PLAN_NOT_VERIFIED")
    completed_indices = [entry["index"] for entry in plan["completed"]]
    shard_indices = [index for shard in plan["shards"] for index in shard["indices"]]
    if sorted(completed_indices + shard_indices) != list(range(177)):
        raise TaskError("VISUALIZATION_HANDOFF_PARTITION_INVALID")
    registry = load_registry(registry_path)
    old_ids = sorted({entry["run_id"] for entry in plan["completed"]})
    if parallelism == 16:
        old_ids = sorted(set(old_ids) | {
            f"visualization-five-10s-gallery-5000-parallel8_handoff_shard{number}-20260913"
            for number in range(1, 9)})
    for run_id in old_ids:
        old = registry["runs"].get(run_id)
        if not old or old.get("identity", {}).get("manifest_sha256") != digest:
            raise TaskError(f"VISUALIZATION_HANDOFF_SOURCE_NOT_REGISTERED:{run_id}")
        state = read_json(PROJECT_ROOT / old["state_file"])
        if state.get("jobs", {}).get("five::gallery", {}).get("status") != "paused":
            raise TaskError(f"VISUALIZATION_HANDOFF_SOURCE_NOT_PAUSED:{run_id}")
    for entry in plan["completed"]:
        old = registry["runs"].get(entry["run_id"])
        if not old or old["output_root"] != entry["source_root"]:
            raise TaskError("VISUALIZATION_HANDOFF_COMPLETED_SOURCE_MISMATCH")
    if merge:
        suffix = f"parallel{parallelism}_handoff_gallery"
        output_root = plan["aggregate_output_root"]
        target = 177
        command = ("exec python3 {runtime_root}/formal_evaluation/merge_10s_gallery_handoff.py"
                   " --runtime {runtime_root} --output {output_root} --plan-sha256 " + plan_digest
                   + " --plan-relative " + shlex.quote(plan_relative))
        support_files = [
            "formal_evaluation/remote_task_control.py",
            "formal_evaluation/merge_10s_gallery_handoff.py",
            "formal_evaluation/merge_10s_gallery_shards.py",
            relative + "/selected_manifest.jsonl.gz",
            relative + "/source_alignment_5000.json", plan_relative,
        ]
        selected_indices = None
    else:
        suffix = f"parallel{parallelism}_handoff_shard{shard_number}"
        shard = plan["shards"][shard_number - 1]
        if shard["number"] != shard_number:
            raise TaskError("VISUALIZATION_HANDOFF_SHARD_IDENTITY_MISMATCH")
        output_root = shard["output_root"]
        selected_indices = shard["indices"]
        target = len(selected_indices)
        command = ("exec python3 {runtime_root}/formal_evaluation/render_10s_gallery_worker.py"
                   " --runtime {runtime_root} --output {output_root}"
                   f" --shard-number {shard_number} --plan-sha256 {plan_digest}"
                   + " --plan-relative " + shlex.quote(plan_relative))
        support_files = [
            "formal_evaluation/remote_task_control.py",
            "formal_evaluation/render_10s_gallery_worker.py",
            relative + "/selected_manifest.jsonl.gz",
            relative + "/source_alignment_5000.json",
            relative + "/render_full.py", relative + "/video_io.py",
            "visualization/batch_10s_41_p95_wmpjpe_20260910/render_batch.py",
            "visualization/six_same_mask_v7_p95_bestego_20260909/render_examples.py",
            "visualization/six_same_mask_v7_p95_bestego_20260909/render_config.json",
            "visualization/six_same_mask_v7_p95_bestego_20260909/mano_195_to_778.npz",
            "visualization/h2o_result3_stride5_minmpjpe_5frames_20260909/render.py",
            plan_relative,
        ]
    runtime = output_root + "/runtime"
    task_id = f"visualization:five:10s-gallery:joint8-p95-wmpjpe:5000:{suffix}:20260913"
    run_id = f"visualization-five-10s-gallery-5000-{suffix}-20260913"
    result = register_run(
        registry_path, logical_task_id=task_id, dataset="five",
        method_set="10s_gallery_six_rows", methods=["egofound3r", "wilor", "hawor", "reviv4d", "pad_hand", "gt"],
        phase="visualization", protocol="300sample-30fps-joint8-p95-wmpjpe", run_id=run_id,
        run_record={
            "task_type": "visualization", "scheduler_id": task_id,
            "state_file": f".auto_scheduler/10s_gallery_5000_{suffix}_20260913.json",
            "state_section": "jobs", "job_keys": ["five::gallery"],
            "output_root": output_root, "target_windows_per_method": target,
            "identity": {"instance_id": instance["instance_id"], "manifest_sha256": digest,
                         "source_alignment_sha256": hashlib.sha256(alignment.read_bytes()).hexdigest(),
                         "plan_sha256": plan_digest, "reader_node": 5000,
                         "source_run_ids": old_ids, "handoff_shard_number": shard_number,
                         **({"selected_indices": selected_indices} if selected_indices is not None else {})},
            "launch": {"node_candidates": [5000], "resource": "visualization", "method": "gallery",
                       "runtime_root": runtime, "manifest_sha256": digest,
                       "plan_sha256": plan_digest, "plan_relative": plan_relative,
                       "support_files": support_files,
                       "deploy_before_preflight": True, "command": command,
                       "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                       "handle_path": output_root + "/control/handle.json",
                       "log_path": output_root + "/logs/worker.log",
                       "values": {"runtime_root": runtime, "output_root": output_root}},
        })
    return {**result, "output_root": output_root, "manifest_sha256": digest,
            "plan_sha256": plan_digest, "selected_indices": selected_indices,
            "target_segments": target}


def register_egofound3r_final_smoke(registry_path: Path) -> dict[str, Any]:
    """Register the one fixed final-checkpoint smoke and initialize it as pending."""
    result = register_run(
        registry_path,
        logical_task_id=EGOFOUND3R_FINAL_SMOKE_TASK_ID,
        dataset="h2o",
        method_set=EGOFOUND3R_FINAL_SMOKE_METHOD_SET,
        methods=["egofound3r"],
        phase="smoke",
        protocol="60f-single-window-v1",
        run_id=EGOFOUND3R_FINAL_SMOKE_RUN_ID,
        run_record={
            "task_type": "evaluation",
            "scheduler_id": "manual:egofound3r-final-rootfusionv2-step1599-smoke",
            "state_file": EGOFOUND3R_FINAL_SMOKE_STATE,
            "state_section": "jobs",
            "job_keys": [EGOFOUND3R_FINAL_SMOKE_JOB],
            "output_root": str(EGOFOUND3R_FINAL_SMOKE_OUTPUT),
            "target_windows_per_method": 1,
            "launch_profile": "egofound3r_final_smoke_v1",
            "storage_probe": {"mount": "/mnt/cpfs", "node": 5001},
            "identity": {
                "pipeline": "egofound3r_final_smoke_v1",
                "baseline_commit": EGOFOUND3R_BASELINES_COMMIT,
                "training_commit": EGOFOUND3R_TRAINING_COMMIT,
                "inference_commit": EGOFOUND3R_INFERENCE_COMMIT,
                "checkpoint_sha256": EGOFOUND3R_CHECKPOINT_SHA256,
                "checkpoint_step": 1599,
            },
        },
    )
    state_path = PROJECT_ROOT / EGOFOUND3R_FINAL_SMOKE_STATE
    state = read_json(state_path) if state_path.is_file() else {"jobs": {}}
    jobs = state.setdefault("jobs", {})
    jobs.setdefault(EGOFOUND3R_FINAL_SMOKE_JOB, {
        "status": "pending",
        "count": 0,
        "output_root": str(EGOFOUND3R_FINAL_SMOKE_OUTPUT),
    })
    atomic_json(state_path, state)
    return {
        **result,
        "status": jobs[EGOFOUND3R_FINAL_SMOKE_JOB].get("status"),
        "output_root": str(EGOFOUND3R_FINAL_SMOKE_OUTPUT),
        "state_file": str(state_path),
    }


def register_pad_bimanual_p95_manifest(registry_path: Path, instance: dict[str, Any]) -> dict[str, Any]:
    """Register the dependent CPU recompute for corrected PAD-Hand predictions."""
    if instance["port"] != 5001 or instance["entrance"] != "/mnt/workspace":
        raise TaskError("PAD_BIMANUAL_P95_REQUIRES_5001_WORKSPACE_INSTANCE")
    relative = ".auto_scheduler/pad_bimanual_20260914/p95_joint8_manifest_v2"
    support_root = PROJECT_ROOT / relative
    spec_path = support_root / "spec.json"
    source_manifest = support_root / "source_selected_manifest.jsonl.gz"
    source_alignment = support_root / "source_alignment_5000.json"
    spec = json.loads(spec_path.read_text())
    if (hashlib.sha256(source_manifest.read_bytes()).hexdigest() != spec["source_manifest_archive_sha256"]
            or hashlib.sha256(gzip.decompress(source_manifest.read_bytes())).hexdigest() != spec["source_manifest_sha256"]
            or hashlib.sha256(source_alignment.read_bytes()).hexdigest() != spec["source_alignment_sha256"]):
        raise TaskError("PAD_BIMANUAL_P95_FROZEN_SOURCE_HASH_MISMATCH")
    root = "/mnt/workspace/sjc/DATA/eval_artifacts/pad_hand_bimanual_6e443a67_20260914/p95_joint8_and_177_manifest_v2"
    runtime = root + "_runtime"
    worktree = "/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R-baselines_pad_bimanual_6e443a67_20260914"
    python = "/mnt/workspace/sjc/envs/egofound3r/bin/python"
    run_ids = sorted({source["prediction_run_id"] for source in spec["datasets"].values()})
    support_files = [
        relative + "/run_pad_bimanual_fixed_p95.py",
        relative + "/recompute_same_mask_all_methods.py",
        relative + "/runtime_registry.json",
        relative + "/spec.json",
        relative + "/source_selected_manifest.jsonl.gz",
        relative + "/source_alignment_5000.json",
        relative + "/remote_task_control.py",
    ]
    preflight = [python, spec["hand_script"]]
    preflight.extend(source["gt_index"] for source in spec["datasets"].values())
    preflight.extend(source["prediction_index"] for source in spec["datasets"].values())
    preflight.extend(source["path"] for source in spec["fixed_masks"].values())
    task_id = "result3:p95:pad_hand:bimanual:joint8-and-177-manifest:20260914:v2"
    run_id = "result3-p95-pad-hand-bimanual-joint8-and-177-manifest-20260914-v2"
    result = register_run(
        registry_path,
        logical_task_id=task_id,
        dataset="six",
        method_set="pad_hand_bimanual_p95_manifest",
        methods=["pad_hand"],
        phase="metrics",
        protocol="full60-fixed-joint8-p95-plus-frozen177-v2",
        run_id=run_id,
        run_record={
            "task_type": "metrics-recompute",
            "scheduler_id": "manual:pad-hand-bimanual-p95-manifest-v2",
            "state_file": ".auto_scheduler/pad_bimanual_20260914/p95_joint8_manifest_v2/state.json",
            "state_section": "jobs",
            "job_keys": ["pad_hand_bimanual_p95_manifest"],
            "output_root": root,
            "target_windows_per_method": 2378,
            "method_root_output": True,
            "dependency_run_ids": run_ids,
            "completion_artifacts": [
                {"path": root + "/report.json", "min_bytes": 1},
                {"path": root + "/summary.json", "min_bytes": 1},
                {"path": root + "/predictions.jsonl", "min_lines": 2378, "min_bytes": 1},
                {"path": root + "/selected_manifest.jsonl", "min_lines": 177, "min_bytes": 1},
                {"path": root + "/source_alignment_5001_pad_bimanual.json", "min_bytes": 1},
                {"path": root + "/COMPLETE", "min_bytes": 1},
            ],
            "identity": {
                "instance_id": instance["instance_id"],
                "pipeline": "pad-hand-bimanual-fixed-p95-manifest-v2",
                "code_commit": "6e443a67c7d9a83111be43be27ee9495e4fc9627",
                "fixed_selection_sha256": spec["fixed_selection_sha256"],
                "source_manifest_sha256": spec["source_manifest_sha256"],
                "source_alignment_sha256": spec["source_alignment_sha256"],
                "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            },
            "launch": {
                "command": (
                    "exec env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                    "PYTHONPATH='{worktree}' '{python}' -u "
                    "'{runtime_root}/" + relative + "/run_pad_bimanual_fixed_p95.py' "
                    "--spec '{runtime_root}/" + relative + "/spec.json' --output-root '{output_root}'"
                ),
                "controller_path": runtime + "/" + relative + "/remote_task_control.py",
                "handle_path": runtime + "/handle.json",
                "log_path": runtime + "/run.log",
                "node_candidates": [5001],
                "resource": "cpu",
                "method": "pad_hand",
                "runtime_root": runtime,
                "support_files": support_files,
                "deploy_before_preflight": True,
                "preflight_paths": preflight,
                "worktree": worktree,
                "worktree_commit": "6e443a67c7d9a83111be43be27ee9495e4fc9627",
                "values": {"worktree": worktree, "python": python, "runtime_root": runtime, "output_root": root},
            },
        },
    )
    old_state_path = PROJECT_ROOT / ".auto_scheduler/pad_bimanual_20260914/p95_joint8_manifest_v1/state.json"
    if old_state_path.is_file():
        old_state = read_json(old_state_path)
        old_job = old_state.get("jobs", {}).get("pad_hand_bimanual_p95_manifest")
        if isinstance(old_job, dict) and old_job.get("status") == "pending":
            old_job.update({
                "status": "blocked",
                "blocked_reason": "RUNTIME_DEPLOY_COPY_TIMEOUT:source_selected_manifest.jsonl:5001",
                "superseded_by": run_id,
            })
            atomic_json(old_state_path, old_state)
    return {**result, "output_root": root, "dependency_run_ids": run_ids,
            "fixed_selection_sha256": spec["fixed_selection_sha256"]}


def merged_run(registry: dict[str, Any], run_id: str, *, allow_shared_reader: bool = False) -> dict[str, Any]:
    try:
        run = {**registry.get("run_template", {}), **registry["runs"][run_id]}
        logical = registry["logical_tasks"][run["logical_task_id"]]
        methods = registry["method_sets"][logical["method_set"]]
    except KeyError as error:
        raise TaskError(f"NOT_REGISTERED:{run_id}") from error
    merged = {**run, **logical, "run_id": run_id, "methods": methods}
    instance = registry.get("_current_instance")
    if instance:
        bound = merged.get("identity", {}).get("instance_id")
        if bound and bound != instance["instance_id"] and not allow_shared_reader:
            raise TaskError(f"INSTANCE_ID_MISMATCH:{run_id}:{bound}")
        merged["ssh"] = {"host": instance["host"], "key": instance["key"]}
        merged["remote_python"] = instance["remote_python"]
        merged["storage_probe"] = {"node": instance["port"], "mount": instance["entrance"]}
        merged["_current_instance"] = instance
    identity = merged.get("identity", {})
    if identity.get('pipeline') == 'result3-completion-cpu-v1':
        kind = identity['kind']
        if kind not in {'pi3', 'restore'}:
            raise TaskError('UNKNOWN_RESULT3_CPU_KIND')
        if hashlib.sha256((PROJECT_ROOT / 'formal_evaluation/result3_cpu_completion.py').read_bytes()).hexdigest() != identity['worker_sha256']:
            raise TaskError('REGISTERED_WORKER_SHA_CHANGED')
        root = merged['output_root']; runtime = root + '/runtime'; worktree = identity['worktree']
        merged.update({
            'artifact_audit': False, 'progress_log': True,
            'completion_artifacts': [{'path': root + '/' + name, 'min_bytes': 1} for name in ['report.json', 'summary.json', 'COMPLETE']],
            'storage_probe': {'mount': '/mnt/cpfs', 'node': 5000},
            'launch': {
                'command': 'cd {worktree} && exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH={worktree} {python} -u {runtime}/formal_evaluation/result3_cpu_completion.py --kind {kind} --spec {runtime}/{spec} --output-root {output_root}',
                'controller_path': worktree + '/formal_evaluation/remote_task_control.py',
                'handle_path': runtime + '/handle.json', 'log_path': runtime + '/run.log',
                'node_candidates': [5000], 'resource': 'cpu', 'method': kind,
                'worktree': worktree, 'worktree_commit': identity['commit'],
                'runtime_root': runtime, 'deploy_before_preflight': False,
                'support_files': ['formal_evaluation/result3_cpu_completion.py', identity['spec_file']] + json.loads(identity.get('extra_support', '[]')),
                'preflight_paths': json.loads(identity['preflight_paths']),
                'values': {'worktree': worktree, 'python': identity['python'], 'runtime': runtime,
                           'kind': kind, 'spec': identity['spec_file'], 'output_root': root},
            },
        })
        if kind == 'pi3':
            merged['launch']['command'] = ('{python} {worktree}/formal_evaluation/validate_runtime_registry.py --method pi3 --strict && ' + merged['launch']['command'])
    if identity.get("pipeline") == "registered-report-read-v1":
        report_paths = [value for value in str(identity.get("report_paths", "")).split(",") if value]
        merged.update({
            "completion_artifacts": [
                {"path": path, "min_bytes": 1} for path in report_paths
            ],
            "report_paths": report_paths,
            "storage_probe": {
                "mount": str(identity.get("storage_mount", "/mnt/cpfs")),
                "node": int(identity.get("node", 5001)),
            },
        })
    if identity.get("pipeline") == "derive-v7-visibility-manifest-v1":
        root = str(merged["output_root"])
        runtime = root + "/runtime"
        worktree = str(identity["worktree"])
        spec = json.loads(base64.urlsafe_b64decode(identity["spec_b64"]))
        merged.update({
            "artifact_audit": False, "progress_log": True,
            "completion_artifacts": [
                {"path": root + "/windows.jsonl", "min_lines": 2378, "min_bytes": 1},
                {"path": root + "/summary.json", "min_bytes": 1},
                {"path": root + "/COMPLETE", "min_bytes": 1},
            ],
            "storage_probe": {"mount": "/mnt/cpfs", "node": 5000},
            "launch": {
                "command": "exec python3 '{runtime}/formal_evaluation/derive_v7_visibility_manifest.py' --spec-b64 '{spec_b64}' --output-root '{output_root}'",
                "controller_path": worktree + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": False,
                "handle_path": runtime + "/handle.json", "log_path": runtime + "/run.log",
                "node_candidates": [5000], "resource": "cpu", "runtime_root": runtime, "method": "manifest",
                "support_files": ["formal_evaluation/derive_v7_visibility_manifest.py", identity["source_relative"]],
                "preflight_paths": [item["gt_index"] for item in spec["catalogs"]],
                "worktree": worktree, "worktree_commit": identity["commit"],
                "values": {"runtime": runtime, "spec_b64": identity["spec_b64"], "output_root": root},
            },
        })
    if identity.get("pipeline") == "visibility-gt-cache-v3":
        root = str(merged["output_root"])
        runtime = root + "/runtime"
        worktree = str(identity["worktree"])
        dataset = str(merged["dataset"])
        data_roots = {
            "h2o": "/mnt/workspace/sjc/DATA/H2O/h2o_data",
            "hot3d": "/mnt/workspace/sjc/DATA/HOT3D/hot3d/hot3d/dataset",
            "arctic": "/mnt/workspace/sjc/DATA/EgoForce/ARCTIC",
            "oakink_v2": "/mnt/workspace/sjc/DATA/OakInk-v2",
            "taco": "/mnt/cpfs/sjc/DATA/TACO_resized",
            "hoi4d": "/mnt/workspace/sjc/DATA/mnt-1/HOI4D",
        }
        if identity.get('data_root'):
            data_roots[dataset] = str(identity['data_root'])
        root_args = " ".join(
            f"--root {name}={shlex.quote(path)}" for name, path in data_roots.items()
        )
        windows = str(identity.get("windows", worktree + "/formal_evaluation/datasets/manifests/evaluation_test_windows_60f_strict_20260819T023430Z_d9d2963f.jsonl"))
        python = str(identity.get("python", EGOFOUND3R_MODEL_PYTHON))
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [{
                "path": root + "/index_shard_000_of_001.jsonl",
                "min_bytes": 1,
                "min_lines": int(merged["target_windows_per_method"]),
            }],
            "progress_log": True,
            "storage_probe": {"mount": "/mnt/cpfs", "node": 5000},
            "launch": {
                "command": (
                    "set -euo pipefail\n"
                    "PYTHONPATH='{worktree}' exec env CUDA_VISIBLE_DEVICES={gpu} '{python}' "
                    "'{worktree}/formal_evaluation/build_six_dataset_gt_cache.py' "
                    "--windows '{windows}' {root_args} --mano-dir '/mnt/workspace/sjc/models/human' "
                    "--output-root '{output_root}' --datasets '{dataset}' "
                    "--cache-version six_dataset_window_gt_v3 --scene-visibility-device cuda:0 "
                    "--interhand-contact-compute-device cuda:0"
                ),
                "controller_path": worktree + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": True,
                "gpu_candidates": list(range(8)),
                "handle_path": root + "/handle.json",
                "log_path": root + "/run.log",
                "method": "egofound3r",
                "node_candidates": [5000, 5001, 6001],
                "preflight_paths": [
                    python, data_roots[dataset], "/mnt/workspace/sjc/models/human/mano",
                    windows, "/mnt/cpfs/sjc/eval_artifacts",
                ],
                "required_gpu_count": 1,
                "resource": "gpu",
                "runtime_root": runtime,
                "support_files": [
                    ".auto_scheduler/visibility_gt_v3_20260909/f8332ee.bundle",
                ],
                "validate_runtime_registry": False,
                "values": {
                    "dataset": dataset,
                    "output_root": root,
                    "python": python,
                    "root_args": root_args,
                    "windows": windows,
                    "worktree": worktree,
                },
                "worktree": worktree,
                "worktree_base_commit": "96679a183d077c7f153399485aa0372145f7c451",
                "worktree_bundle": runtime + "/.auto_scheduler/visibility_gt_v3_20260909/f8332ee.bundle",
                "worktree_commit": str(identity["commit"]),
                "worktree_fetch_ref": "formal-hand-fix",
                "worktree_source": "/mnt/workspace/sjc/EgoFound3R-baselines",
            },
        })
    if identity.get("pipeline") == "visibility-gt-cache-v3" and identity.get("prepared_worktree"):
        merged["launch"].update({
            "support_files": [], "deploy_before_preflight": False,
            "worktree_source": "", "worktree_bundle": "",
            "worktree_base_commit": "", "worktree_fetch_ref": "",
            "gpu_candidates": list(range(1, 8)),
        })
    if identity.get("pipeline") == "visibility-gt-cache-v3" and identity.get("manifest_sha256"):
        launch = merged["launch"]
        checksum_gate = (
            "'{python}' -c 'import hashlib,sys; from pathlib import Path; "
            "assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()==sys.argv[2]' "
            "'{windows}' '{manifest_sha256}'\n"
        )
        launch["command"] = launch["command"].replace("PYTHONPATH=", checksum_gate + "PYTHONPATH=", 1).replace("exec env", "env") + (
            "\nexec '{python}' '{runtime}/formal_evaluation/verify_visibility_gt.py' "
            "--manifest '{windows}' --sha256 '{manifest_sha256}' "
            "--dataset '{dataset}' --output-root '{output_root}'"
        )
        launch["support_files"] = ["formal_evaluation/verify_visibility_gt.py"]
        launch["values"].update({"runtime": runtime, "manifest_sha256": identity["manifest_sha256"]})
        merged["completion_artifacts"].extend([
            {"path": root + "/summary.json", "min_bytes": 1},
            {"path": root + "/COMPLETE", "min_bytes": 1},
        ])
    if identity.get("pipeline") == "baseline-clean-worktree-v1":
        root = str(merged["output_root"])
        runtime = root + "/runtime"
        worktree = str(identity["worktree"])
        commit = str(identity["commit"])
        bundle_support = str(identity.get("worktree_bundle_support", ""))
        bundle = str(identity.get("worktree_bundle", ""))
        bundle_base = str(identity.get("worktree_base_commit", ""))
        bundle_source = str(identity.get("worktree_source", ""))
        support_files = ["formal_evaluation/verify_clean_worktree.py"]
        if bundle_support:
            support_files.append(bundle_support)
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [
                {"path": root + "/summary.json", "min_bytes": 1},
                {"path": root + "/COMPLETE", "min_bytes": 1},
            ],
            "progress_log": True,
            "storage_probe": {"mount": "/mnt/cpfs", "node": 5000},
            "launch": {
                "command": (
                    "set -euo pipefail\n"
                    "exec python3 '{runtime}/formal_evaluation/verify_clean_worktree.py' "
                    "--worktree '{worktree}' --expected-commit '{commit}' "
                    "--output-root '{output_root}'"
                ),
                "controller_path": worktree + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": True,
                "handle_path": root + "/handle.json",
                "log_path": root + "/run.log",
                "method": "bundle",
                "node_candidates": [5000],
                "preflight_paths": ["/mnt/cpfs/sjc/eval_artifacts"],
                "resource": "cpu",
                "runtime_root": runtime,
                "support_files": support_files,
                "values": {
                    "commit": commit,
                    "output_root": root,
                    "runtime": runtime,
                    "worktree": worktree,
                },
                "worktree": worktree,
                "worktree_commit": commit,
            },
        })
        if bundle:
            merged["launch"].update({
                "worktree_base_commit": bundle_base,
                "worktree_bundle": bundle,
                "worktree_source": bundle_source,
            })
        else:
            merged["launch"].update({
                "worktree_clone_ref": "formal-hand-fix",
                "worktree_clone_url": "https://github.com/jingchengsimon/EgoFound3R-baselines.git",
            })
    if identity.get("pipeline") == "egofound3r_final_smoke_v1":
        root = str(merged["output_root"])
        runtime = root + "/runtime"
        worktree = (
            "/mnt/workspace/sjc/DATA/runtime_worktrees/"
            "EgoFound3R-baselines_egofound3r_final_839020c_20260905_retry1"
        )
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [
                {"path": root + "/COMPLETE", "min_bytes": 1},
                {"path": root + "/smoke_summary.json", "min_bytes": 1},
            ],
            "progress_log": True,
            "launch": {
                "command": (
                    "set -euo pipefail\n"
                    "export PYTHONPATH='{model_source}:{worktree}'\n"
                    "exec env TASKCTL_GPU={gpu} '{python}' "
                    "'{runtime}/formal_evaluation/run_egofound3r_smoke.py' "
                    "--input-index '{input_index}' --model-python '{python}' "
                    "--runner '{runner}' --methods-config '{methods_config}' "
                    "--config '{config}' --checkpoint '{checkpoint}' "
                    "--backbone-checkpoint '{backbone}' --output-root '{output_root}' "
                    "--source-commit '{training_commit}' --inference-commit '{inference_commit}' "
                    "--checkpoint-sha256 '{checkpoint_sha256}'"
                ),
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": True,
                "gpu_candidates": list(range(8)),
                "handle_path": root + "/handle.json",
                "log_path": root + "/run.log",
                "method": "egofound3r",
                "node_candidates": [6001, 5001, 5000],
                "preflight_paths": [
                    str(EGOFOUND3R_FINAL_SMOKE_INPUT_INDEX),
                    str(EGOFOUND3R_FORMAL_CONFIG),
                    str(EGOFOUND3R_FORMAL_CHECKPOINT),
                    str(EGOFOUND3R_BACKBONE),
                    str(EGOFOUND3R_INFERENCE_ROOT),
                ],
                "required_gpu_count": 1,
                "resource": "gpu",
                "runtime_registry": worktree + "/formal_evaluation/config/baseline_runtime_registry_dsw.json",
                "runtime_root": runtime,
                "support_files": [
                    ".auto_scheduler/egofound3r_final_839020c_from_cbdb6ce.bundle",
                    "formal_evaluation/remote_task_control.py",
                    "formal_evaluation/run_egofound3r_smoke.py",
                ],
                "validate_runtime_registry": True,
                "values": {
                    "backbone": str(EGOFOUND3R_BACKBONE),
                    "checkpoint": str(EGOFOUND3R_FORMAL_CHECKPOINT),
                    "checkpoint_sha256": EGOFOUND3R_CHECKPOINT_SHA256,
                    "config": str(EGOFOUND3R_FORMAL_CONFIG),
                    "inference_commit": EGOFOUND3R_INFERENCE_COMMIT,
                    "input_index": str(EGOFOUND3R_FINAL_SMOKE_INPUT_INDEX),
                    "methods_config": worktree + "/formal_evaluation/config/methods_v1.json",
                    "model_source": str(EGOFOUND3R_INFERENCE_ROOT),
                    "output_root": root,
                    "python": EGOFOUND3R_MODEL_PYTHON,
                    "runner": worktree + "/formal_evaluation/scene/adapters/run_egofound3r_baseline.py",
                    "runtime": runtime,
                    "training_commit": EGOFOUND3R_TRAINING_COMMIT,
                    "worktree": worktree,
                },
                "wait_for_idle_gpu": True,
                "worktree": worktree,
                "worktree_bundle": (
                    runtime + "/.auto_scheduler/egofound3r_final_839020c_from_cbdb6ce.bundle"
                ),
                "worktree_commit": EGOFOUND3R_BASELINES_COMMIT,
                "worktree_source": "/mnt/workspace/sjc/EgoFound3R-baselines",
            },
        })
    if merged.get("task_type") == "artifact-migration" and identity.get("direction") == "target":
        root = str(merged["output_root"])
        runtime = str(identity["runtime_root"])
        dataset = str(merged["dataset"])
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [{"path": root + "/COMPLETE", "min_bytes": 1}],
            "progress_log": True,
            "launch": {
                "command": "set -euo pipefail\nexec python3 '{runtime_root}/formal_evaluation/artifact_migration_worker.py' --spec '{runtime_root}/formal_evaluation/config/contact_artifact_migration_3dataset_20260901.json' --dataset '{dataset}' --destination-root '{output_root}' --lock '{lock_path}'",
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "handle_path": runtime + "/handle.json",
                "log_path": runtime + "/run.log",
                "method": "bundle",
                "node_candidates": [int(identity.get("node", 5000))],
                "resource": "io",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/artifact_migration_worker.py",
                    "formal_evaluation/remote_task_control.py",
                    "formal_evaluation/config/contact_artifact_migration_3dataset_20260901.json",
                ],
                "values": {
                    "dataset": dataset,
                    "lock_path": str(identity["lock_path"]),
                    "output_root": root,
                    "runtime_root": runtime,
                },
            },
        })
    if merged.get("task_type") == "artifact-migration" and identity.get("direction") == "contact-cache-rehome-target":
        root = str(merged["output_root"])
        runtime = str(identity["runtime_root"])
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [
                {"path": root + "/rehome_manifest.json", "min_bytes": 1},
                {"path": root + "/REHOME_COMPLETE", "min_bytes": 1},
            ],
            "progress_log": True,
            "launch": {
                "command": (
                    "set -euo pipefail\nexec python3 '{runtime}/formal_evaluation/rehome_restored_contact_cache.py' "
                    "stage --source-root '{source_root}' --target-root '{output_root}' --lock '{lock_path}'"
                ),
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": True,
                "handle_path": runtime + "/handle.json",
                "log_path": runtime + "/run.log",
                "method": "bundle",
                "node_candidates": [5000],
                "preflight_paths": [str(identity["source_root"]), "/mnt/oss/pre-train/ego/eval_artifacts"],
                "resource": "io",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/rehome_restored_contact_cache.py",
                    "formal_evaluation/remote_task_control.py",
                ],
                "values": {
                    "lock_path": str(identity["lock_path"]),
                    "output_root": root,
                    "runtime": runtime,
                    "source_root": str(identity["source_root"]),
                },
            },
        })
    if merged.get("task_type") == "artifact-migration" and identity.get("direction") == "contact-cache-rehome-release":
        root = str(merged["output_root"])
        runtime = str(identity["runtime_root"])
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [
                {"path": root + "/summary.json", "min_bytes": 1},
                {"path": root + "/COMPLETE", "min_bytes": 1},
            ],
            "progress_log": True,
            "launch": {
                "command": (
                    "set -euo pipefail\nexec python3 '{runtime}/formal_evaluation/rehome_restored_contact_cache.py' "
                    "release --source-root '{source_root}' --target-root '{target_root}' "
                    "--report-root '{output_root}' --lock '{lock_path}'"
                ),
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": True,
                "handle_path": runtime + "/handle.json",
                "log_path": runtime + "/run.log",
                "method": "bundle",
                "node_candidates": [5000],
                "preflight_paths": [str(identity["source_root"]), str(identity["target_root"]) + "/REHOME_COMPLETE"],
                "resource": "io",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/rehome_restored_contact_cache.py",
                    "formal_evaluation/remote_task_control.py",
                ],
                "values": {
                    "lock_path": str(identity["lock_path"]),
                    "output_root": root,
                    "runtime": runtime,
                    "source_root": str(identity["source_root"]),
                    "target_root": str(identity["target_root"]),
                },
            },
        })
    if merged.get("task_type") == "artifact-migration" and identity.get("direction") == "result-tree-rehome-target":
        root = str(merged["output_root"])
        runtime = str(identity["runtime_root"])
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [
                {"path": root + "/migration_manifest.json", "min_bytes": 1},
                {"path": root + "/MIGRATION_COMPLETE", "min_bytes": 1},
            ],
            "progress_log": True,
            "launch": {
                "command": (
                    "set -euo pipefail\nexec python3 '{runtime}/formal_evaluation/rehome_result_tree.py' "
                    "stage --source-root '{source_root}' --target-root '{output_root}' "
                    "--expected-files '{expected_files}' --expected-bytes '{expected_bytes}' --lock '{lock_path}'"
                ),
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": True,
                "handle_path": runtime + "/handle.json",
                "log_path": runtime + "/run.log",
                "method": "bundle",
                "node_candidates": [5000],
                "preflight_paths": [str(identity["source_root"]), "/mnt/oss/pre-train/ego/eval_artifacts"],
                "resource": "io",
                "runtime_root": runtime,
                "support_files": ["formal_evaluation/rehome_result_tree.py", "formal_evaluation/remote_task_control.py"],
                "values": {
                    "expected_bytes": str(identity["expected_bytes"]),
                    "expected_files": str(identity["expected_files"]),
                    "lock_path": str(identity["lock_path"]),
                    "output_root": root,
                    "runtime": runtime,
                    "source_root": str(identity["source_root"]),
                },
            },
        })
    if merged.get("task_type") == "artifact-migration" and identity.get("direction") == "result-tree-rehome-release":
        root = str(merged["output_root"])
        runtime = str(identity["runtime_root"])
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [
                {"path": root + "/summary.json", "min_bytes": 1},
                {"path": root + "/COMPLETE", "min_bytes": 1},
            ],
            "progress_log": True,
            "launch": {
                "command": (
                    "set -euo pipefail\nexec python3 '{runtime}/formal_evaluation/rehome_result_tree.py' "
                    "release --source-root '{source_root}' --target-root '{target_root}' "
                    "--expected-files '{expected_files}' --expected-bytes '{expected_bytes}' "
                    "--report-root '{output_root}' --lock '{lock_path}'"
                ),
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": True,
                "handle_path": runtime + "/handle.json",
                "log_path": runtime + "/run.log",
                "method": "bundle",
                "node_candidates": [5000],
                "preflight_paths": [str(identity["source_root"]), str(identity["target_root"]) + "/MIGRATION_COMPLETE"],
                "resource": "io",
                "runtime_root": runtime,
                "support_files": ["formal_evaluation/rehome_result_tree.py", "formal_evaluation/remote_task_control.py"],
                "values": {
                    "expected_bytes": str(identity["expected_bytes"]),
                    "expected_files": str(identity["expected_files"]),
                    "lock_path": str(identity["lock_path"]),
                    "output_root": root,
                    "runtime": runtime,
                    "source_root": str(identity["source_root"]),
                    "target_root": str(identity["target_root"]),
                },
            },
        })
    if merged.get("task_type") == "artifact-migration" and identity.get("direction") == "oss-to-cpfs":
        root = str(merged["output_root"])
        runtime = root + "/runtime"
        catalogs = [value for value in str(identity["catalog_run_ids"]).split(",") if value]
        copy_methods = [value for value in str(identity["copy_methods"]).split(",") if value]
        sources = []
        for catalog_run_id in catalogs:
            source_run = merged_run(registry, catalog_run_id)
            catalog = source_run.get("artifact_catalog")
            if not isinstance(catalog, dict):
                raise TaskError("ARTIFACT_CATALOG_NOT_REGISTERED:" + catalog_run_id)
            dataset = str(catalog["dataset"])
            sources.append({"dataset": dataset, "role": "gt_cache",
                            "source": str(Path(catalog["gt_index"]).parent)})
            for method in copy_methods:
                if method not in catalog["predictions"]:
                    raise TaskError("METHOD_NOT_REGISTERED:" + method)
                sources.extend({"dataset": dataset, "role": method, "source": str(path)}
                               for path in catalog["predictions"][method].get("formal_roots", []))
        sources_b64 = base64.urlsafe_b64encode(json.dumps({
            "catalog_run_ids": catalogs, "methods": copy_methods, "sources": sources,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).decode("ascii")
        merged.update({
            "artifact_audit": False,
            "partial_output_stats": True,
            "completion_artifacts": [
                {"path": root + "/migration_manifest.json", "min_bytes": 1},
                {"path": root + "/COMPLETE", "min_bytes": 1},
            ],
            "progress_log": True,
            "launch": {
                "command": (
                    "set -euo pipefail\nexec python3 '{runtime}/formal_evaluation/artifact_copy_to_cpfs.py' "
                    "--sources-b64 '{sources_b64}' --output-root '{output_root}' --lock '{lock_path}'"
                ),
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": True,
                "handle_path": runtime + "/handle.json",
                "log_path": runtime + "/run.log",
                "method": "bundle",
                "node_candidates": [int(identity.get("node", 5000))],
                "preflight_paths": ["/mnt/oss/pre-train/ego/eval_artifacts",
                                    "/mnt/cpfs/sjc/eval_artifacts"],
                "resource": "io",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/artifact_copy_to_cpfs.py",
                    "formal_evaluation/remote_task_control.py",
                ],
                "values": {
                    "lock_path": str(identity["lock_path"]),
                    "output_root": root,
                    "runtime": runtime,
                    "sources_b64": sources_b64,
                },
            },
        })
    if merged.get("task_type") == "metrics-recompute" and identity.get("protocol") == "same-mask-frozen-v1":
        root = str(merged["output_root"])
        runtime = root + "/runtime"
        catalogs = [value for value in str(identity["catalog_run_ids"]).split(",") if value]
        methods = list(merged.get("methods", []))
        catalog_specs = []
        for catalog_run_id in catalogs:
            source_run = merged_run(registry, catalog_run_id)
            catalog = source_run.get("artifact_catalog")
            if not isinstance(catalog, dict):
                raise TaskError("ARTIFACT_CATALOG_NOT_REGISTERED:" + catalog_run_id)
            predictions = {}
            for method in methods:
                entry = catalog.get("predictions", {}).get(method)
                if not isinstance(entry, dict) or entry.get("verification", {}).get("status") != "verified":
                    raise TaskError(f"PREDICTIONS_NOT_VERIFIED:{catalog_run_id}:{method}")
                predictions[method] = {
                    "formal_roots": entry.get("formal_roots", []),
                }
            catalog_specs.append({
                "catalog_run_id": catalog_run_id,
                "dataset": catalog["dataset"],
                "expected_windows": catalog["expected_windows"],
                "gt_index": catalog["gt_index"],
                "predictions": predictions,
            })
        code_root = str(identity["code_root"])
        source_mode = str(identity.get("source_mode", "cpfs_mirror"))
        spec_b64 = base64.urlsafe_b64encode(json.dumps({
            "catalogs": catalog_specs,
            "methods": methods,
            "methods_config": code_root + "/formal_evaluation/config/methods_v1.json",
            "migration_root": str(identity.get("migration_root", "")),
            "source_mode": source_mode,
            "result3_catalog": str(identity["result3_catalog"]),
            "mask_roots": {
                "p97_5": str(identity["p97_5_root"]),
                "variants": str(identity["variants_root"]),
            },
            "hand_script": runtime + "/.auto_scheduler/ego_stride5_joint8_frozen_q975_q99_20260907/analyze.py",
            "scene_script": runtime + "/.auto_scheduler/result3_filtered_scene_contact_20260909/analyze.py",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).decode("ascii")
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [
                {"path": root + "/summary.json", "min_bytes": 1},
                {"path": root + "/COMPLETE", "min_bytes": 1},
            ],
            "progress_log": True,
            "launch": {
                "command": (
                    "set -euo pipefail\n"
                    "PYTHONPATH='{code_root}' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                    "exec '{python}' '{runtime}/formal_evaluation/recompute_same_mask_all_methods.py' "
                    "--spec-b64 '{spec_b64}' --output-root '{output_root}' --workers '{workers}'"
                ),
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "deploy_before_preflight": True,
                "handle_path": runtime + "/handle.json",
                "log_path": runtime + "/run.log",
                "method": "bundle",
                "node_candidates": [int(identity.get("node", 5000))],
                "preflight_paths": (
                    (["/mnt/oss/pre-train/ego/eval_artifacts"] if source_mode == "direct_oss" else
                     [str(identity["migration_root"]) + "/COMPLETE"])
                    + [str(identity["p97_5_root"]), str(identity["variants_root"]),
                       str(identity["result3_catalog"]), code_root]
                ),
                "resource": "io",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/recompute_same_mask_all_methods.py",
                    "formal_evaluation/remote_task_control.py",
                    ".auto_scheduler/ego_stride5_joint8_frozen_q975_q99_20260907/analyze.py",
                    ".auto_scheduler/ego_stride5_filter_gtcamera_distribution_20260907/analyze.py",
                    ".auto_scheduler/result3_filtered_scene_contact_20260909/analyze.py",
                ],
                "values": {
                    "code_root": code_root,
                    "output_root": root,
                    "python": str(identity["python"]),
                    "runtime": runtime,
                    "spec_b64": spec_b64,
                    "workers": str(identity.get("workers", 8)),
                },
            },
        })
    if merged.get("task_type") == "artifact-migration" and identity.get("direction") == "cleanup":
        root = str(merged["output_root"])
        runtime = str(identity["runtime_root"])
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [{"path": root + "/COMPLETE", "min_bytes": 1}],
            "progress_log": True,
            "launch": {
                "command": "set -euo pipefail\nexec python3 '{runtime_root}/formal_evaluation/artifact_cleanup_worker.py' --spec '{runtime_root}/formal_evaluation/config/contact_artifact_cleanup_3dataset_20260901.json' --output-root '{output_root}' --lock '{lock_path}'",
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "handle_path": runtime + "/handle.json",
                "log_path": runtime + "/run.log",
                "method": "bundle",
                "node_candidates": [int(identity.get("node", 5000))],
                "resource": "io",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/artifact_cleanup_worker.py",
                    "formal_evaluation/remote_task_control.py",
                    "formal_evaluation/config/contact_artifact_cleanup_3dataset_20260901.json",
                ],
                "values": {
                    "lock_path": str(identity["lock_path"]),
                    "output_root": root,
                    "runtime_root": runtime,
                },
            },
        })
    if merged.get("launch_profile") == "h2o_contact_3shard_v1":
        method = str(methods[0])
        shard = int(str(merged["identity"]["shard"]).split("/", 1)[0])
        gpu = int(merged["identity"]["physical_gpu"])
        root = str(merged["output_root"])
        worktree = "/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R-baselines_contact_h2o_87e8f03_20260831T043548Z_retry3"
        cache_name = "s2_right_h2o_283" if method == "s2contact" else "contactopt_right_h2o_283"
        source = "S2Contact" if method == "s2contact" else "ContactOpt"
        checkpoint = "20211027-212322.pt" if method == "s2contact" else "deepcontact_checkpoint.pt"
        cache_root = f"/mnt/workspace/sjc/DATA/h2o_contact_baseline/cache_v2_20260827T180241Z/{method}"
        runtime = root + "/runtime"
        retry = int(merged.get("launch_retry", 0))
        retry_suffix = f"_retry{retry}" if retry else ""
        merged.update({
            "artifact_audit": True,
            "prediction_root": f"{root}/{method}",
            "progress_log": True,
            "launch": {
                "command": "set -euo pipefail\nexport PYTHONPATH='{runtime_root}:{worktree}'\nexec env CUDA_VISIBLE_DEVICES={gpu} '{python}' '{runtime_root}/formal_evaluation/contact/adapters/run_s2_contactopt.py' --baseline '{method_name}' --source-root '{source_root}' --cache '{cache}' --cache-index '{cache_index}' --window-input-index '{input_index}' --checkpoint '{checkpoint}' --mano-right '{mano_right}' --methods-config '{methods_config}' --output-root '{output_root}' --phase formal --batch-size 32 --device cuda:0 --num-shards 3 --shard-index {shard_index}",
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "gpu_candidates": [gpu],
                "handle_path": root + f"/handle{retry_suffix}.json",
                "log_path": root + f"/run{retry_suffix}.log",
                "method": method,
                "node_candidates": [5001],
                "required_gpu_count": 1,
                "resource": "gpu",
                "deploy_before_preflight": True,
                "runtime_registry": runtime + "/formal_evaluation/config/baseline_runtime_registry_dsw.json",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/remote_task_control.py",
                    "formal_evaluation/contact/adapters/run_s2_contactopt.py",
                    "formal_evaluation/contact/sharding.py",
                    "formal_evaluation/config/baseline_runtime_registry_dsw.json",
                ],
                "values": {
                    "cache": f"{cache_root}/{cache_name}.pkl",
                    "cache_index": f"{cache_root}/{cache_name}_index.jsonl",
                    "checkpoint": f"/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/{source}/checkpoints/{checkpoint}",
                    "input_index": "/mnt/workspace/sjc/DATA/eval_artifacts/contact_jobs_20260827T180241Z/h2o_geometry/h2o_first283.jsonl",
                    "mano_right": "/mnt/workspace/sjc/models/human/mano/MANO_RIGHT.pkl",
                    "method_name": method,
                    "methods_config": f"{worktree}/formal_evaluation/config/methods_v1.json",
                    "output_root": root,
                    "python": "/mnt/workspace/sjc/envs/contactopt/bin/python",
                    "runtime_root": runtime,
                    "shard_index": shard,
                    "source_root": f"/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/{source}",
                    "worktree": worktree,
                },
                "worktree": worktree,
                "worktree_source": "/mnt/workspace/sjc/EgoFound3R-baselines",
                "worktree_commit": "87e8f03",
            },
        })
    contact_pipeline = merged.get("identity", {}).get("pipeline")
    if contact_pipeline in {"taco_contact_3shard_v1", "hoi4d_contact_6shard_v1"}:
        shard = int(str(merged["identity"]["shard"]).split("/", 1)[0])
        gpu = int(merged["identity"]["physical_gpu"])
        node = int(merged["identity"].get("physical_node", 5001))
        root = str(merged["output_root"])
        runtime = root + "/runtime"
        worktree = "/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R-baselines_contact_h2o_87e8f03_20260831T043548Z_retry3"
        if contact_pipeline == "taco_contact_3shard_v1":
            dataset, total, num_shards = "taco", 400, 3
            cache_root = "/mnt/workspace/sjc/DATA/taco_contact_baseline/cache_v2_20260827T180241Z"
            input_index = "/mnt/workspace/sjc/DATA/eval_artifacts/contact_jobs_20260827T180241Z/taco_geometry/taco_first400.jsonl"
        else:
            dataset, total, num_shards = "hoi4d", 461, 6
            cache_root = "/mnt/workspace/sjc/DATA/hoi4d_contact_baseline/cache_v2_20260827T180241Z"
            input_index = "/mnt/workspace/sjc/DATA/eval_artifacts/contact_jobs_20260827T180241Z/hoi4d_geometry/hoi4d_first461.jsonl"
        merged.update({
            "artifact_audit": False,
            "completion_artifacts": [{"path": root + "/COMPLETE", "min_bytes": 0}],
            "progress_log": True,
            "launch": {
                "command": "set -euo pipefail\nexport PYTHONPATH='{runtime_root}:{worktree}'\nenv CUDA_VISIBLE_DEVICES={gpu} '{python}' '{runtime_root}/formal_evaluation/contact/adapters/run_s2_contactopt.py' --baseline s2contact --source-root '{s2_source}' --cache '{s2_cache}' --cache-index '{s2_index}' --window-input-index '{input_index}' --checkpoint '{s2_checkpoint}' --mano-right '{mano_right}' --methods-config '{methods_config}' --output-root '{output_root}' --phase formal --batch-size 32 --device cuda:0 --num-shards {num_shards} --shard-index {shard_index}\nenv CUDA_VISIBLE_DEVICES={gpu} '{python}' '{runtime_root}/formal_evaluation/contact/adapters/run_s2_contactopt.py' --baseline contactopt --source-root '{contactopt_source}' --cache '{contactopt_cache}' --cache-index '{contactopt_index}' --window-input-index '{input_index}' --checkpoint '{contactopt_checkpoint}' --mano-right '{mano_right}' --methods-config '{methods_config}' --output-root '{output_root}' --phase formal --batch-size 32 --device cuda:0 --num-shards {num_shards} --shard-index {shard_index}\ntouch '{output_root}/COMPLETE'",
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "gpu_candidates": [gpu],
                "handle_path": root + "/handle.json",
                "log_path": root + "/run.log",
                "method": "s2contact",
                "node_candidates": [node],
                "required_gpu_count": 1,
                "resource": "gpu",
                "deploy_before_preflight": True,
                "runtime_registry": runtime + "/formal_evaluation/config/baseline_runtime_registry_dsw.json",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/remote_task_control.py",
                    "formal_evaluation/contact/adapters/run_s2_contactopt.py",
                    "formal_evaluation/contact/sharding.py",
                    "formal_evaluation/config/baseline_runtime_registry_dsw.json",
                ],
                "values": {
                    "contactopt_cache": f"{cache_root}/contactopt/contactopt_right_{dataset}_{total}.pkl",
                    "contactopt_checkpoint": "/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/ContactOpt/checkpoints/deepcontact_checkpoint.pt",
                    "contactopt_index": f"{cache_root}/contactopt/contactopt_right_{dataset}_{total}_index.jsonl",
                    "contactopt_source": "/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/ContactOpt",
                    "input_index": input_index,
                    "mano_right": "/mnt/workspace/sjc/models/human/mano/MANO_RIGHT.pkl",
                    "methods_config": worktree + "/formal_evaluation/config/methods_v1.json",
                    "num_shards": num_shards,
                    "output_root": root,
                    "python": "/mnt/workspace/sjc/envs/contactopt/bin/python",
                    "runtime_root": runtime,
                    "s2_cache": f"{cache_root}/s2contact/s2_right_{dataset}_{total}.pkl",
                    "s2_checkpoint": "/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/S2Contact/checkpoints/20211027-212322.pt",
                    "s2_index": f"{cache_root}/s2contact/s2_right_{dataset}_{total}_index.jsonl",
                    "s2_source": "/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/S2Contact",
                    "shard_index": shard,
                    "worktree": worktree,
                },
                "worktree": worktree,
            },
        })
    metrics_pipeline = merged.get("identity", {}).get("pipeline")
    if metrics_pipeline == "metric_recompute_rot_hawor_worlddiag_v1":
        root = str(merged["output_root"])
        retry = int(merged.get("launch_retry", 0))
        retry_suffix = f".retry{retry}" if retry else ""
        runtime = root + (f"/runtime_retry{retry}" if retry else "/runtime")
        worktree = str(merged["identity"]["worktree"])
        spec_file = str(merged["identity"].get(
            "spec", "formal_evaluation/config/metric_recompute_rot_hawor_worlddiag_20260902.json"
        ))
        rot_only = "rot-only" in str(merged["logical_task_id"])
        merged.update({
            "progress_log": True,
            "completion_artifacts": [{"path": root + "/COMPLETE", "min_bytes": 1}],
            "report_path": root + f"/{merged['dataset']}/report.json",
            "report_metric_tokens": (["camera_rot_error_deg"] if rot_only else
                                     ["_w_mpjpe", "_wa_mpjpe", "_w_mpmpe", "_wa_mpmpe",
                                      "_w_mpvpe", "_wa_mpvpe"]),
            "report_units": "degrees" if rot_only else "mm",
            "launch": {
                "command": "set -euo pipefail\nexport PYTHONPATH='{runtime}:{worktree}'\nexec '{python}' '{runtime}/formal_evaluation/run_metric_recompute_six.py' --spec '{runtime}/" + spec_file + "' --runtime '{runtime}' --worktree '{worktree}' --output-root '{output_root}'",
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "handle_path": root + f"/handle{retry_suffix}.json",
                "log_path": root + f"/run{retry_suffix}.log",
                "method": "egofound3r",
                "node_candidates": [int(value) for value in merged["identity"].get(
                    "node_candidates", [5000, 5001, 6001]
                )],
                "preflight_paths": [
                    "/mnt/oss/pre-train/ego/eval_artifacts/formal_h2o_60f_20260822T112000Z_continuation",
                    "/mnt/oss/pre-train/ego/eval_artifacts/formal_standard_reports_10methods_20260824/h2o/gt_cache/index_shard_000_of_001.jsonl",
                ],
                "resource": "cpu",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/__init__.py",
                    "formal_evaluation/remote_task_control.py",
                    "formal_evaluation/run_metric_recompute_six.py",
                    "formal_evaluation/run_metrics_v2_aggregation.py",
                    "formal_evaluation/evaluate_six_dataset.py",
                    "formal_evaluation/hand/__init__.py",
                    "formal_evaluation/hand/metrics.py",
                    "formal_evaluation/scene/__init__.py",
                    "formal_evaluation/scene/metrics.py",
                    spec_file,
                ],
                "values": {
                    "output_root": root,
                    "python": "/mnt/workspace/sjc/envs/egofound3r/bin/python",
                    "runtime": runtime,
                    "worktree": worktree,
                },
                "worktree": worktree,
            },
        })
    if metrics_pipeline in {"h2o_contact_metrics_v1", "taco_contact_metrics_v1", "hoi4d_contact_metrics_v1", "hot3d_contact_metrics_v1", "arctic_contact_metrics_v1", "oakink_v2_contact_metrics_v1"}:
        root = str(merged["output_root"])
        runtime = root + "/runtime"
        retry = int(merged.get("launch_retry", 0))
        retry_suffix = f"_retry{retry}" if retry else ""
        worktree = "/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R-baselines_contact_h2o_87e8f03_20260831T043548Z_retry3"
        if metrics_pipeline == "h2o_contact_metrics_v1":
            dataset, expected_windows = "h2o", 283
            input_index = "/mnt/workspace/sjc/DATA/eval_artifacts/contact_jobs_20260827T180241Z/h2o_geometry/h2o_first283.jsonl"
            prediction_base = "/mnt/workspace/sjc/DATA/eval_artifacts/formal_contact_h2o_283_sixgpu_20260831T043548Z"
            prediction_roots = ((method, f"{prediction_base}/{method}_shard{shard}")
                                for method in ("s2contact", "contactopt") for shard in range(3))
            existing_gt_args = ""
            node_candidates = [6001, 5001]
        elif metrics_pipeline == "taco_contact_metrics_v1":
            dataset, expected_windows = "taco", 400
            input_index = "/mnt/workspace/sjc/DATA/eval_artifacts/contact_jobs_20260827T180241Z/taco_geometry/taco_first400.jsonl"
            prediction_base = "/mnt/workspace/sjc/DATA/eval_artifacts/formal_contact_taco_400_threegpu_20260831T150558Z"
            prediction_roots = ((method, f"{prediction_base}/shard{shard}")
                                for method in ("s2contact", "contactopt") for shard in range(3))
            existing_gt_args = ""
            node_candidates = [6001, 5001]
        elif metrics_pipeline == "hoi4d_contact_metrics_v1":
            dataset, expected_windows = "hoi4d", 461
            input_index = "/mnt/workspace/sjc/DATA/eval_artifacts/contact_jobs_20260827T180241Z/hoi4d_geometry/hoi4d_first461.jsonl"
            prediction_base = "/mnt/workspace/sjc/DATA/eval_artifacts/formal_contact_hoi4d_461_sixgpu_20260901T040257Z"
            prediction_roots = ((method, f"{prediction_base}/shard{shard}")
                                for method in ("s2contact", "contactopt") for shard in range(6))
            existing_gt_args = ""
            node_candidates = [6001, 5001, 5000]
        else:
            existing_dataset = metrics_pipeline.removesuffix("_contact_metrics_v1")
            dataset = existing_dataset
            expected_windows = 434 if dataset == "arctic" else 400
            input_index = f"/mnt/workspace/sjc/DATA/eval_artifacts/auto_eval_scheduler/combined_index/{dataset}_first400.jsonl"
            if dataset == "hot3d":
                prediction_base = "/mnt/workspace/sjc/DATA/eval_artifacts/formal_contact_hot3d_400_20260827T234639Z"
                prediction_roots = ((method, f"{prediction_base}/{method}")
                                    for method in ("s2contact", "contactopt"))
            elif dataset == "arctic":
                prediction_roots = (
                    (method, f"/mnt/workspace/sjc/eval_artifacts/{root}/{method}")
                    for method in ("s2contact", "contactopt")
                    for root in ("formal_arctic_contact_retry_400_20260823T231000Z", "formal_arctic_contact_tail34_20260823T151000Z")
                )
            else:
                prediction_base = "/mnt/workspace/sjc/eval_artifacts/formal_oakink_v2_contact_400_20260823T234000Z"
                prediction_roots = ((method, f"{prediction_base}/{method}")
                                    for method in ("s2contact", "contactopt"))
            existing_gt_args = " --existing-gt-index '{existing_gt_index}' --existing-gt-root '{existing_gt_root}'"
            node_candidates = [5001]
            existing_gt_base = f"/mnt/oss/pre-train/ego/eval_artifacts/formal_standard_reports_10methods_20260824/{dataset}/gt_cache"
        prediction_args = " ".join(
            f"--prediction-root '{method}={path}'" for method, path in prediction_roots
        )
        merged.update({
            "progress_log": True,
            "completion_artifacts": [{"path": root + "/report.json", "min_bytes": 1}],
            "launch": {
                "command": "set -euo pipefail\nexport PYTHONPATH='{runtime_root}:{worktree}'\nexec '{python}' '{runtime_root}/formal_evaluation/run_h2o_contact_metrics.py' --worktree '{worktree}' --runtime-root '{runtime_root}' --dataset '{dataset}' --expected-windows {expected_windows} --input-index '{input_index}' " + prediction_args + " --output-root '{output_root}' --data-root '{data_root}' --mano-dir '{mano_dir}'" + existing_gt_args,
                "controller_path": runtime + "/formal_evaluation/remote_task_control.py",
                "handle_path": root + f"/handle{retry_suffix}.json",
                "log_path": root + f"/run{retry_suffix}.log",
                "method": "s2contact",
                "node_candidates": node_candidates,
                "resource": "cpu",
                "runtime_root": runtime,
                "support_files": [
                    "formal_evaluation/remote_task_control.py",
                    "formal_evaluation/run_h2o_contact_metrics.py",
                    "formal_evaluation/build_six_dataset_gt_cache.py",
                    "formal_evaluation/evaluate_six_dataset.py",
                ],
                "values": {
                    "data_root": "/mnt/workspace/sjc/DATA",
                    "dataset": dataset,
                    "expected_windows": expected_windows,
                    "existing_gt_index": (existing_gt_base + "/index_shard_000_of_001.jsonl") if existing_gt_args else "",
                    "existing_gt_root": existing_gt_base if existing_gt_args else "",
                    "input_index": input_index,
                    "mano_dir": "/mnt/workspace/sjc/models/human",
                    "output_root": root,
                    "python": "/mnt/workspace/sjc/envs/egofound3r/bin/python",
                    "runtime_root": runtime,
                    "worktree": worktree,
                },
                "worktree": worktree,
            },
        })
    return merged


def resolve(registry: dict[str, Any], task_id: str | None, dataset: str | None,
            method_set: str | None, phase: str, protocol: str) -> dict[str, Any]:
    if task_id is None:
        if not dataset or not method_set:
            raise TaskError("RESOLVE_REQUIRES_TASK_ID_OR_DATASET_AND_METHOD_SET")
        task_id = f"{phase}:{dataset}:{method_set}:{protocol}"
    logical = registry.get("logical_tasks", {}).get(task_id)
    if logical is None:
        raise TaskError(f"NOT_REGISTERED:{task_id}")
    run = merged_run(registry, str(logical["latest"]))
    return {
        "artifact_catalog_run_id": run.get("artifact_catalog_run_id"),
        "path_validation_required": True,
        "task_id": task_id,
        "run_id": run["run_id"],
        "state_file": str(PROJECT_ROOT / run["state_file"]),
        "output_root": run["output_root"],
        "output_root_overrides": run.get("output_root_overrides", {}),
        "node_path_layout": run.get("node_path_layout", {}),
    }


def catalog_paths(run: dict[str, Any], artifact: str | None = None, method: str | None = None, audit_fields: bool = False) -> dict[str, Any]:
    catalog = run.get("artifact_catalog")
    if not catalog:
        raise TaskError("ARTIFACT_CATALOG_NOT_REGISTERED:" + run["run_id"])
    selected = json.loads(json.dumps(catalog))
    selected["audit_fields"] = audit_fields
    if artifact == "reports":
        selected["probe_kind"] = "reports"
    if artifact == "gt-cache":
        selected["predictions"] = {}
    elif method:
        if method not in selected["predictions"]:
            raise TaskError("METHOD_NOT_REGISTERED:" + method)
        selected["predictions"] = {method: selected["predictions"][method]}
    source = (PROJECT_ROOT / "formal_evaluation/registered_artifact_paths.py").read_text()
    remote = " ".join(("python3", "-c", shlex.quote(source), shlex.quote(json.dumps(selected))))
    try:
        completed = _ssh(run, int(catalog["node"]), remote, timeout=1200)
    except subprocess.TimeoutExpired as error:
        raise TaskError("ARTIFACT_PROBE_TIMEOUT:" + run["run_id"]) from error
    if completed.returncode:
        raise TaskError("ARTIFACT_PROBE_FAILED:" + str(completed.returncode))
    try:
        result = json.loads(completed.stdout)
    except ValueError as error:
        raise TaskError("ARTIFACT_PROBE_INVALID_RESPONSE") from error
    result["run_id"] = run["run_id"]
    result["selection_policy"] = catalog.get("selection_policy", catalog.get("scope"))
    for name, value in result.get("predictions", {}).items():
        value["newer_registered_runs"] = selected["predictions"][name].get("newer_registered_runs", [])
        value["source_reports"] = selected["predictions"][name].get("source_reports", [])
        value["source_prediction_index"] = selected["predictions"][name].get("source_prediction_index")
    result["node"] = catalog["node"]
    if artifact == "reports":
        wanted = [method] if method else list(selected["predictions"])
        verified = {}
        for report_path, report in result.get("reports", {}).items():
            if not isinstance(report, dict) or report.get("gt_windows") != catalog["expected_windows"]:
                continue
            for name, entry in report.get("methods", {}).items():
                if name in wanted and entry.get("missing_prediction_windows") == 0 and entry.get("datasets", {}).get(catalog["dataset"], {}).get("n_windows") == catalog["expected_windows"]:
                    verified.setdefault(name, []).append(report_path)
        result["verified_report_paths"] = verified
        result["unresolved_report_methods"] = [name for name in wanted if name not in verified]
        result["all_verified"] = not result["unresolved_report_methods"]
    if artifact != "reports":
        result["all_verified"] = result["gt_cache"].get("status") == "verified" and all(v.get("status") == "verified" for v in result["predictions"].values())
    return result


def audit_pi3_depth_sample(run):
    catalog = run.get('artifact_catalog', {})
    if catalog.get('dataset') not in {'h2o', 'taco', 'hoi4d'}:
        raise TaskError('PI3_DEPTH_CATALOG_REQUIRED')
    runtime = json.loads((PROJECT_ROOT/'formal_evaluation/config/baseline_runtime_registry_dsw.json').read_text())['methods']['pi3']
    script = r'''import json,sys,zipfile
from pathlib import Path
import numpy as np
def first(path,key):
    with zipfile.ZipFile(path) as z, z.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f)
        shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        assert not fortran and not dtype.hasobject
        n=int(np.prod(shape[1:])); raw=f.read(n*dtype.itemsize)
        return np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
c=json.loads(sys.argv[1]); index=Path(c['gt_index'])
gt=[json.loads(l) for l in index.read_text().splitlines() if l.strip()]
wanted={r['window_id']:r for r in gt}; ids={r['cache_id']:r['window_id'] for r in gt}
samples=[];records=[];seen=set()
def shapes(path):
    result={}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('.npy'):continue
            with z.open(name) as f:
                result[name[:-4]]=list(np.lib.format._read_array_header(f,np.lib.format.read_magic(f))[0])
    return result
for root in c['predictions']['pi3']['formal_roots']:
    sampled=False
    for p in sorted(Path(root).iterdir()):
        if not (p/'metadata.json').is_file():continue
        meta=json.loads((p/'metadata.json').read_text()); key=ids.get(meta['window_id'],meta['window_id'])
        if key not in wanted:continue
        row=wanted[key]; assert meta['frame_ids']==row['frame_ids']
        assert key not in seen;seen.add(key)
        assert meta['dataset']==c['dataset'] and meta['camera_convention'].startswith('OpenCV x-right y-down z-forward')
        assert meta['runner_detail']['preprocess']=='official Pi3 pixel-limit resize (255000 pixels), divisible by 14'
        sh,sw=meta['source_resolution_hw'];scale=(255000/(sh*sw))**.5
        tw,th=sw*scale,sh*scale;gw,gh=round(tw/14),round(th/14)
        while gw*14*gh*14>255000:
            if gw/gh>tw/th:gw-=1
            else:gh-=1
        expected=[max(1,gh)*14,max(1,gw)*14]
        assert meta['processed_resolution_hw']==expected
        raw=row['array_path'];target=index.parent/raw.split('/gt_cache/',1)[1] if '/gt_cache/' in raw else Path(raw)
        ph=shapes(p/'predictions.npz'); gth=shapes(target)
        assert ph['camera_points']==[len(row['frame_ids']),*expected,3]
        assert ph['camera_points_valid']==ph['camera_points'][:-1]
        assert gth['depth']==gth['depth_valid'] and gth['depth'][0]==len(row['frame_ids'])
        dh,dw=gth['depth'][1:];assert abs(sw/sh-dw/dh)<1e-6
        records.append({'dataset':c['dataset'],'window_id':key,'frame_ids':row['frame_ids'],'prediction_dir':str(p),'gt_array_path':str(target),'source_resolution_hw':[sh,sw],'processed_resolution_hw':expected,'gt_resolution_hw':[dh,dw],'depth_source':'camera_points[...,2]','depth_scale_alignment':'none','grid_mapping':'full_image_resize','verification':'identity, metadata provenance, shapes; numeric camera/world check sampled only'})
        if sampled:continue
        sampled=True
        cp=first(p/'predictions.npz','camera_points').astype('float64')
        wp=first(p/'predictions.npz','world_points').astype('float64')
        pose=first(p/'predictions.npz','camera_c2w').astype('float64')
        valid=first(p/'predictions.npz','camera_points_valid').astype(bool)
        transformed=cp@pose[:3,:3].T+pose[:3,3]
        err=np.linalg.norm(transformed-wp,axis=-1); ok=valid & np.isfinite(err)
        raw=row['array_path']; target=index.parent/raw.split('/gt_cache/',1)[1] if '/gt_cache/' in raw else Path(raw)
        depth=first(target,'depth'); dv=first(target,'depth_valid').astype(bool)
        ratio=float(np.median(err[ok])/max(np.median(np.linalg.norm(wp[ok],axis=-1)),1e-8))
        samples.append({'window_id':key,'prediction':str(p),'gt':str(target),'camera_grid':list(cp.shape),'gt_depth_grid':list(depth.shape),'valid_pixels':int(ok.sum()),'valid_nonpositive_z':int((valid & (cp[...,2]<=0)).sum()),'world_transform_relative_error':ratio,'gt_positive_valid_pixels':int((dv & np.isfinite(depth) & (depth>0)).sum()),'prediction_metadata':meta,'gt_metadata':json.loads((index.parent/row['metadata_path'].split('/gt_cache/',1)[1] if '/gt_cache/' in row['metadata_path'] else Path(row['metadata_path'])).read_text()),'passed':bool(ok.any() and ratio<0.01 and not (valid & (cp[...,2]<=0)).any())})
print(json.dumps({'dataset':c['dataset'],'samples':samples,'records':records,'verified_windows':len(seen),'expected_windows':len(wanted),'ok':seen==set(wanted) and bool(samples) and all(s['passed'] for s in samples),'scope':'all-window identity, grid and preprocessing provenance; first-frame numeric sample per root; no scale fitting or forward'}))
'''
    completed = _ssh(run, int(catalog['node']), shlex.join([runtime['python'], '-c', script, json.dumps(catalog)]), timeout=1200)
    if completed.returncode:
        raise TaskError('PI3_SAMPLE_AUDIT_FAILED:' + completed.stderr[-1500:])
    return {'run_id': run['run_id'], **json.loads(completed.stdout)}


REGISTERED_PREDICTION_INDEX_AUDIT = r'''import json,sys
from collections import Counter
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);alias=Path(sys.argv[2])
if not root.is_absolute():raise ValueError('OUTPUT_ROOT_NOT_ABSOLUTE')
index=root/'predictions.jsonl'
records=[]
if index.is_file():
    for line in index.read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line);path=Path(row['prediction_dir'])
        if not path.is_relative_to(root) and path.is_relative_to(alias):path=root/path.relative_to(alias)
        records.append((row,path))
else:
    for prediction in sorted(root.rglob('predictions.npz')):
        path=prediction.parent;metadata=path/'metadata.json'
        if metadata.is_file() and prediction.is_file():records.append((json.loads(metadata.read_text()),path))
if not records:raise ValueError('NO_PREDICTION_RECORDS:'+str(root)+':CHILDREN:'+','.join(p.name for p in list(root.iterdir())[:20]) if root.is_dir() else ':ROOT_MISSING')
rows=[];errors=[];signatures=Counter()
for row,path in records:
    try:
        if not path.is_relative_to(root):raise ValueError('PATH_OUTSIDE_ROOT')
        metadata=json.loads((path/'metadata.json').read_text())
        prediction=path/'predictions.npz'
        if not prediction.is_file():raise ValueError('PREDICTION_MISSING')
        if row.get('window_id') and metadata.get('window_id')!=row.get('window_id'):
            raise ValueError('WINDOW_ID_MISMATCH')
        with np.load(prediction,allow_pickle=False) as source:
            signature=tuple(sorted((key,tuple(source[key].shape),str(source[key].dtype)) for key in source.files))
        signatures[signature]+=1
        frame_ids=metadata.get('frame_ids') or []
        rows.append({'window_id':metadata.get('window_id'),'cache_id':path.name,
                     'dataset':metadata.get('dataset'),'method':metadata.get('method'),
                     'frame_count':len(frame_ids),'frame_start':frame_ids[0] if frame_ids else None,
                     'frame_end':frame_ids[-1] if frame_ids else None})
    except Exception as error:errors.append({'path':str(path),'error':str(error)})
report={}
for name in ('report.json','summary.json'):
    path=root/name
    if path.is_file():
        try:
            value=json.loads(path.read_text());report[name]={key:value.get(key) for key in ('status','windows','metric_count','gt_windows')}
        except Exception as error:report[name]={'error':str(error)}
print(json.dumps({'root':str(root),'complete_exists':(root/'COMPLETE').is_file(),
 'index_path':str(index) if index.is_file() else None,'prediction_records':len(records),
 'verified_records':len(rows),'errors':errors,'rows':rows,'array_signatures':[
 {'count':count,'arrays':[{'key':key,'shape':shape,'dtype':dtype} for key,shape,dtype in signature]}
 for signature,count in signatures.items()],'reports':report},sort_keys=True))'''


def audit_registered_prediction_index(run, dataset=None):
    root = str(run.get("output_root", ""))
    probe = run.get("storage_probe", {})
    node = int(probe.get("node", run.get("identity", {}).get("reader_node", 5000)))
    if not root.startswith("/mnt/") or node not in (5000, 5001):
        raise TaskError("REGISTERED_PREDICTION_ROOT_REQUIRED")
    bases = [(Path(root), Path(root))]
    freeze = run.get("identity", {}).get("freeze_manifest")
    if freeze:
        freeze_path = PROJECT_ROOT / freeze
        manifest = read_json(freeze_path)
        matches = [entry for entry in manifest.get("entries", []) if entry.get("cpfs_path") == root]
        if len(matches) == 1 and matches[0].get("proposed_oss_path"):
            bases.append((Path(matches[0]["proposed_oss_path"]), Path(root)))
    candidates = []
    for base, alias_base in bases:
        suffixes = ((Path(dataset), Path("egoforce") / dataset, Path()) if dataset else (Path(),))
        for suffix in suffixes:
            pair = (base / suffix, alias_base / suffix)
            if pair not in candidates:
                candidates.append(pair)
    failures = []
    result = None
    for candidate, alias in candidates:
        command = ("python3 -c " + shlex.quote(REGISTERED_PREDICTION_INDEX_AUDIT)
                   + " " + shlex.quote(str(candidate)) + " " + shlex.quote(str(alias)))
        completed = _ssh(run, node, command, timeout=300)
        if completed.returncode:
            failures.append({"root": str(candidate), "error": completed.stderr[-500:]})
            continue
        try:
            result = json.loads(completed.stdout)
        except ValueError as error:
            raise TaskError("REGISTERED_PREDICTION_INDEX_AUDIT_INVALID") from error
        break
    if result is None:
        raise TaskError("REGISTERED_PREDICTION_INDEX_AUDIT_FAILED:" + json.dumps(failures)[-1200:])
    result.update(run_id=run["run_id"], node=node, ok=not result["errors"])
    result["candidate_failures"] = failures
    return result


def audit_inference_inputs(run, geometry=False, reader_node=5000):
    inference = str(run.get('identity',{}).get('inference_commit',''))
    if not inference.startswith(('3e533881','2b9c180')) or run.get('task_type') != 'evaluation':
        raise TaskError('INFERENCE_INPUT_PROFILE_NOT_REGISTERED:' + run['run_id'])
    script = r'''import json,sys,time
from pathlib import Path
geometry_mode = sys.argv[2] == 'geometry'
geometry_frames=0;object_frames=0;geometry_fields={}
root=Path(sys.argv[1]); result={'root':str(root),'node':5000}
errors=[];seen=set();rgb_seen=set();verified=0;pad_ready=0
def read_nonempty(path):
    with Path(path).open('rb') as f:
        if not f.read(1):raise ValueError('EMPTY_FILE:'+str(path))
try:
    gt=[json.loads(l) for l in (root/'inputs/gt_index.jsonl').read_text().splitlines() if l.strip()]
    wanted={r['window_id']:r for r in gt}
    rows=[json.loads(l) for l in (root/'inputs/inputs.jsonl').read_text().splitlines() if l.strip()]
    result.update(expected_windows=len(gt),input_rows=len(rows))
    for row in rows:
        path=Path(row['window_input'])
        try:
            record=json.loads(path.read_text());key=record['window_id'];target=wanted[key]
            if key in seen:raise ValueError('DUPLICATE_WINDOW')
            seen.add(key)
            if record['dataset']!=target['dataset'] or record['frame_ids']!=target['frame_ids']:raise ValueError('IDENTITY_MISMATCH')
            if len(record['rgb_paths'])!=60:raise ValueError('NON_60F_RGB')
            if geometry_mode:
                if len(record['geometry_paths'])!=60:raise ValueError('NON_60F_GEOMETRY')
                for raw in record['geometry_paths']:
                    headers=npz_headers(Path(raw))
                    for key,shape in {'hand_vertices':[2,778,3],'hand_joints':[2,21,3],'hand_valid':[2]}.items():
                        if headers.get(key)!=shape:raise ValueError('GEOMETRY_SHAPE:'+str(raw)+':'+key)
                    for key in ['object_vertices','object_faces']:
                        shape=headers.get(key)
                        if not shape or len(shape)!=2 or shape[1]!=3:raise ValueError('GEOMETRY_SHAPE:'+str(raw)+':'+key)
                    geometry_frames+=1
                    object_frames+=int(headers['object_vertices'][0]>0 and headers['object_faces'][0]>0)
                verified+=1
                continue
            for rgb in record['rgb_paths']:
                if rgb not in rgb_seen:read_nonempty(rgb);rgb_seen.add(rgb)
            verified+=1
            try:
                mapping=json.loads((path.parent/'mapping.json').read_text())
                if mapping.get('window_id')!=key or mapping.get('frame_ids')!=record['frame_ids'] or mapping.get('sequence')!=record['sequence_id']:raise ValueError('PAD_MAPPING_IDENTITY_MISMATCH')
                if int(mapping.get('context_frames',0))<16:raise ValueError('PAD_CONTEXT_LT16')
                read_nonempty(path.parent/'input.mp4');pad_ready+=1
            except Exception as e:
                errors.append({'window_input':str(path),'scope':'pad_prepared','error':str(e)})
        except Exception as e:
            errors.append({'window_input':str(path),'scope':'rgb_input','error':str(e)})
    result.update(rgb_verified_windows=verified,pad_ready_windows=pad_ready,unique_rgb_files=len(rgb_seen),missing_window_ids=sorted(set(wanted)-seen)[:10],error_count=len(errors),errors=errors[:12],ok=verified==len(gt) and (geometry_mode or pad_ready==len(gt)) and not errors)
    if geometry_mode:result.update(geometry_frames=geometry_frames,nonempty_object_frames=object_frames,empty_object_frames=geometry_frames-object_frames,verification='NPZ fields/shapes/readable headers, not full numeric validation')
except Exception as e:result.update(ok=False,error=type(e).__name__+': '+str(e))
print(json.dumps(result))
'''
    script = (PROJECT_ROOT/'formal_evaluation/registered_artifact_paths.py').read_text().split("if __name__ == '__main__':")[0] + script
    remote=' '.join(('python3','-c',shlex.quote(script),shlex.quote(run['output_root']), 'geometry' if geometry else 'rgb'))
    completed=_ssh(run,reader_node,remote,timeout=1200)
    if completed.returncode:raise TaskError('INFERENCE_INPUT_AUDIT_FAILED:'+completed.stderr[-1000:])
    result=json.loads(completed.stdout);result['run_id']=run['run_id'];result['node']=reader_node;return result


def indexed_prediction_fields(run):
    task = run.get('logical_task_id', run.get('task_id', ''))
    if task.startswith('evaluation:dyn_hamr:'):
        root = run['output_root']
    elif task == 'evaluation:interactvlm:six:100x60f:named:20260906':
        root = run['identity']['oss_output_root']
    else:
        raise TaskError('INDEX_FIELD_AUDIT_PROFILE_NOT_REGISTERED:' + str(task))
    source = (PROJECT_ROOT / 'formal_evaluation/registered_artifact_paths.py').read_text().split("if __name__ == '__main__':")[0]
    source += r'''
import sys
root = Path(sys.argv[1])
result = {'root':str(root),'index':str(root/'predictions.jsonl'),'fields':{},'native_fields':{}}
try:
    rows = [json.loads(line) for line in (root/'predictions.jsonl').read_text().splitlines() if line.strip()]
    fields, native = Counter(), Counter()
    identities = set()
    for row in rows:
        identity = (row['dataset'],row['window_id'])
        if identity in identities: raise ValueError('DUPLICATE_IDENTITY')
        identities.add(identity)
        directory = Path(row['prediction_dir'])
        metadata = json.loads((directory/'metadata.json').read_text())
        if metadata['dataset'] != row['dataset'] or metadata['window_id'] != row['window_id']: raise ValueError('IDENTITY_MISMATCH')
        count_headers(fields,directory/'predictions.npz')
        native_path = directory/'native/predictions.npz'
        if native_path.is_file(): count_headers(native,native_path)
        native_meta = directory/'native/metadata.json'
        if native_meta.is_file():
            details = json.loads(native_meta.read_text())
            frames = details.get('frames',[])
            if frames and len(frames) != len(metadata['frame_ids']): raise ValueError('NATIVE_FRAME_COUNT_MISMATCH')
            for frame in frames: count_headers(native,Path(frame['prediction_path']))
    result.update(ok=True,windows=len(rows),fields=dict(fields),native_fields=dict(native))
except Exception as e:
    result.update(ok=False,error=type(e).__name__+': '+str(e))
print(json.dumps(result))
'''
    remote = ' '.join(('python3','-c',shlex.quote(source),shlex.quote(root)))
    completed = _ssh(run,5000,remote,timeout=600)
    if completed.returncode: raise TaskError('INDEX_FIELD_AUDIT_FAILED:'+completed.stderr[-500:])
    value = json.loads(completed.stdout)
    value['run_id'] = run['run_id']
    return value


def audit_dyn_camera_alignment(run, cache_id):
    """Inspect one exact registered Dyn-HaMR prediction after a metric-gate failure."""
    if not re.fullmatch(r'[0-9a-f]{24}', cache_id):
        raise TaskError('INVALID_DYN_CACHE_ID')
    task = run.get('logical_task_id', '')
    if not (task.startswith('evaluation:dyn_hamr:') and ':full-hand24:' in task):
        raise TaskError('DYN_CAMERA_AUDIT_PROFILE_NOT_REGISTERED:' + str(task))
    dataset = task.split(':')[2]
    spec_path = PROJECT_ROOT / 'formal_evaluation/config/dyn_full_hand_20260912' / (dataset + '.json')
    spec = read_json(spec_path)
    script = r'''import hashlib,json,sys
from pathlib import Path
import numpy as np
from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
from formal_evaluation.evaluate_six_dataset import _camera_trajectory_alignments
from formal_evaluation.recompute_same_mask_all_methods import remap_gt_row
root=Path(sys.argv[1]);cache=sys.argv[2];spec=json.loads(sys.argv[3]);directory=root/'dyn_hamr'/'formal'/cache
result={'prediction_dir':str(directory),'cache_id':cache}
try:
 metadata=json.loads((directory/'metadata.json').read_text())
 index=Path(spec['gt_index']);raw=index.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=spec['gt_index_sha256']:raise ValueError('GT_INDEX_SHA256_MISMATCH')
 rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
 row=next((r for r in rows if r['window_id']==metadata['window_id']),None)
 if row is None:raise ValueError('WINDOW_NOT_IN_GT_INDEX')
 gm,target=load_window_cache(remap_gt_row(row,Path('/unused'),index.parent,direct_oss=True))
 if gm['frame_ids']!=metadata['frame_ids']:raise ValueError('FRAME_ID_MISMATCH')
 with np.load(directory/'predictions.npz',allow_pickle=False) as archive:
  names=('camera_c2w','camera_valid','hand_valid')
  pred={name:archive[name] for name in names if name in archive.files}
 result.update(dataset=metadata['dataset'],window_id=metadata['window_id'],prediction_fields=sorted(archive.files))
 poses=np.asarray(pred.get('camera_c2w',np.empty((0,))),dtype=float)
 if poses.shape==(len(metadata['frame_ids']),4,4):
  valid=np.isfinite(poses).all(axis=(1,2))
  if 'camera_valid' in pred:valid &= pred['camera_valid'].astype(bool)
  target_pose=np.asarray(target['camera_c2w'],dtype=float)
  target_valid=np.isfinite(target_pose).all(axis=(1,2)) & np.asarray(target['camera_valid'],dtype=bool)
  combined=valid & target_valid
  result.update(pred_camera_valid=int(valid.sum()),gt_camera_valid=int(target_valid.sum()),paired_camera_valid=int(combined.sum()),pred_hand_valid=int(np.asarray(pred['hand_valid'],dtype=bool).sum()))
  if combined.any():
   p=poses[combined,:3,3];g=target_pose[combined,:3,3]
   result.update(pred_center_span=np.ptp(p,axis=0).tolist(),gt_center_span=np.ptp(g,axis=0).tolist(),pred_center_variance=float(np.sum(np.var(p,axis=0))))
  transforms,used=_camera_trajectory_alignments(pred,target)
  result.update(transform_types=sorted(transforms),alignment_valid_frames=int(used.sum()))
  if 'sim3' in transforms:result['sim3_scale']=float(transforms['sim3'].scale)
 else:result['camera_shape']=list(poses.shape)
 result['ok']=True
except Exception as error:result.update(ok=False,error=type(error).__name__+': '+str(error))
print(json.dumps(result))'''
    worktree = str(spec['metric_worktree'])
    python = str(spec['metric_python'])
    remote = ' '.join(('env', shlex.quote('PYTHONPATH=' + worktree), shlex.quote(python), '-c',
                       shlex.quote(script), shlex.quote(run['output_root']), shlex.quote(cache_id),
                       shlex.quote(json.dumps(spec))))
    completed = _ssh(run, 5000, remote, timeout=300)
    if completed.returncode:
        raise TaskError('DYN_CAMERA_AUDIT_FAILED:' + completed.stderr[-500:])
    result = json.loads(completed.stdout)
    result['run_id'] = run['run_id']
    return result


def audit_dyn_hand_report_values(run):
    """Read verified complete24/complete6 from one exact registered output root."""
    task = run.get('logical_task_id', '')
    if not (task.startswith('evaluation:dyn_hamr:') and ':full-hand24:' in task
            or task.startswith('result3:p95:completion:dyn_hamr_hand_sim3:')):
        raise TaskError('DYN_REPORT_AUDIT_PROFILE_NOT_REGISTERED:' + str(task))
    script = r'''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);expected=int(sys.argv[2]);kind=sys.argv[3]
result={'root':str(root)}
try:
 report_path=root/'report.json';raw=report_path.read_bytes();report=json.loads(raw)
 summary=json.loads((root/'summary.json').read_text());complete=(root/'COMPLETE').is_file()
 rows=[line for line in (root/'metrics.jsonl').read_text().splitlines() if line.strip()]
 values=report.get(kind)
 result.update(ok=bool(complete and report.get('status')=='complete' and summary.get('status')=='complete' and report.get('windows')==expected and summary.get('windows')==expected and len(rows)==expected and summary.get('report_sha256')==hashlib.sha256(raw).hexdigest() and isinstance(values,dict) and len(values)==summary.get('metric_count')),complete=complete,report_sha256=hashlib.sha256(raw).hexdigest(),windows=report.get('windows'),metric_rows=len(rows),metric_count=summary.get('metric_count'),values=values,source=report.get('source'),protocol=report.get('protocol'))
except Exception as error:result.update(ok=False,error=type(error).__name__+': '+str(error))
print(json.dumps(result))'''
    kind = 'complete24' if task.startswith('evaluation:dyn_hamr:') else 'complete6'
    remote = ' '.join(('python3', '-c', shlex.quote(script), shlex.quote(run['output_root']),
                       str(run['target_windows_per_method']), shlex.quote(kind)))
    completed = _ssh(run, 5000, remote, timeout=300)
    if completed.returncode:
        raise TaskError('DYN_REPORT_AUDIT_FAILED:' + completed.stderr[-500:])
    result = json.loads(completed.stdout)
    result['run_id'] = run['run_id']
    return result


def audit_result3_camera_report(run):
    """Read the exact completed Result3 camera-completion report."""
    task = str(run.get('logical_task_id', ''))
    if not task.startswith('result3:p95:completion:camera-hawor-dyn-ego-dyn100:'):
        raise TaskError('RESULT3_CAMERA_REPORT_PROFILE_NOT_REGISTERED:' + task)
    script = r'''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);result={'root':str(root)}
try:
 raw=(root/'report.json').read_bytes();report=json.loads(raw)
 summary=json.loads((root/'summary.json').read_text())
 digest=hashlib.sha256(raw).hexdigest()
 expected={'hawor_all8_p95':2378,'dyn_unfiltered':100,'ego_matched_dyn_unfiltered':100}
 methods={}
 for method,datasets in report.get('methods',{}).items():
  methods[method]={}
  for dataset,values in datasets.items():
   methods[method][dataset]={key:values.get(key) for key in (
    'n_windows','camera_ate_aligned_mean','camera_ate_aligned_count','camera_ate_aligned_undefined_window_count',
    'camera_rot_error_deg_mean','camera_rot_error_deg_count','camera_rot_error_deg_undefined_window_count',
    'camera_pose_auc_30_mean','camera_pose_auc_30_count','camera_pose_auc_30_undefined_window_count',
    'depth_abs_rel_mean','depth_abs_rel_count','depth_abs_rel_undefined_window_count',
    'depth_rmse_mean','depth_rmse_count','depth_rmse_undefined_window_count',
    'depth_delta1_mean','depth_delta1_count','depth_delta1_undefined_window_count')}
 result.update(ok=bool((root/'COMPLETE').is_file() and report.get('status')=='complete' and
  summary.get('status')=='complete' and report.get('windows')==expected and
  summary.get('windows')==expected and summary.get('report_sha256')==digest),
  complete=(root/'COMPLETE').is_file(),report_sha256=digest,windows=report.get('windows'),
  protocol=report.get('protocol'),methods=methods,audit=report.get('audit'))
except Exception as error:result.update(ok=False,error=type(error).__name__+': '+str(error))
print(json.dumps(result))'''
    node = int(run.get('launch', {}).get('node_candidates', [5000])[0])
    completed = _ssh(run, node, ' '.join(('python3', '-c', shlex.quote(script),
                                          shlex.quote(run['output_root']))), timeout=300)
    if completed.returncode:
        raise TaskError('RESULT3_CAMERA_REPORT_AUDIT_FAILED:' + completed.stderr[-500:])
    result = json.loads(completed.stdout)
    result['run_id'] = run['run_id']
    return result


def audit_dyn_reuse_index(run, trailing_cache_id):
    """Verify indexed predictions and one completed but unindexed trailing window."""
    if not re.fullmatch(r'[0-9a-f]{24}', trailing_cache_id):
        raise TaskError('INVALID_DYN_CACHE_ID')
    task = run.get('logical_task_id', '')
    if not (task.startswith('evaluation:dyn_hamr:') and ':full-hand24:' in task):
        raise TaskError('DYN_REUSE_PROFILE_NOT_REGISTERED:' + str(task))
    dataset = task.split(':')[2]
    spec = read_json(PROJECT_ROOT / 'formal_evaluation/config/dyn_full_hand_20260912' / (dataset + '.json'))
    script = r'''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);spec=json.loads(sys.argv[2]);trailing=sys.argv[3]
result={'root':str(root)}
try:
 raw=(root/'predictions.jsonl').read_bytes();prior=[json.loads(line) for line in raw.splitlines() if line.strip()]
 inputs=[json.loads(line) for line in Path(spec['prepared_index']).read_text().splitlines() if line.strip()]
 if not (len(prior)<len(inputs)==spec['expected_windows']):raise ValueError('INDEX_COUNT_MISMATCH')
 for i in range(len(prior)+1):
  record=json.loads(Path(inputs[i]['window_input']).read_text())
  target=root/'dyn_hamr/formal'/record['cache_id']
  meta=json.loads((target/'metadata.json').read_text());status=json.loads((target/'run.json').read_text())['status']
  if (meta['dataset'],meta['window_id'],meta['frame_ids'])!=(record['dataset'],record['window_id'],record['frame_ids']):raise ValueError('METADATA_MISMATCH:'+str(i))
  if status not in ('success','blocked_no_track_over_min_track_len') or not (target/'predictions.npz').is_file():raise ValueError('PREDICTION_INCOMPLETE:'+str(i))
  if i<len(prior) and (prior[i]['prediction_dir']!=str(target) or prior[i]['window_id']!=record['window_id']):raise ValueError('INDEX_IDENTITY_MISMATCH:'+str(i))
  if i==len(prior) and record['cache_id']!=trailing:raise ValueError('TRAILING_CACHE_MISMATCH')
 result.update(ok=True,prediction_index_sha256=hashlib.sha256(raw).hexdigest(),indexed_windows=len(prior),reusable_windows=len(prior)+1,target_windows=len(inputs),trailing_cache_id=trailing)
except Exception as error:result.update(ok=False,error=type(error).__name__+': '+str(error))
print(json.dumps(result))'''
    remote = ' '.join(('python3', '-c', shlex.quote(script), shlex.quote(run['output_root']),
                       shlex.quote(json.dumps(spec)), shlex.quote(trailing_cache_id)))
    completed = _ssh(run, 5000, remote, timeout=300)
    if completed.returncode:
        raise TaskError('DYN_REUSE_AUDIT_FAILED:' + completed.stderr[-500:])
    result = json.loads(completed.stdout)
    result['run_id'] = run['run_id']
    return result


def bundle_archive_paths(run, reader_node):
    if str(run.get('identity',{}).get('asset_spec')) != 'formal_evaluation/config/contact_artifact_migration_3dataset_20260901.json' or run['identity'].get('direction') != 'target':
        raise TaskError('ARCHIVE_CATALOG_NOT_REGISTERED:'+run['run_id'])
    script=r'''import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1]);path=root/'migration_manifest.json'
raw=path.read_bytes();manifest=json.loads(raw);errors=[];files=[];geometry_bytes=0
assert manifest['status']=='complete' and (root/'COMPLETE').is_file()
for row in manifest['files']:
    stored=row['parts'] if row.get('storage')=='parts' else [{'path':row['path'],'bytes':row['bytes']}]
    total=0
    for part in stored:
        p=Path(part['path'])
        assert root in p.parents
        if part.get('offset',total)!=total:errors.append('PART_OFFSET:'+str(p))
        try:
            if p.stat().st_size!=part['bytes']:errors.append('PART_SIZE:'+str(p))
            with p.open('rb') as f:
                if part['bytes'] and not f.read(1):errors.append('EMPTY_PART:'+str(p))
        except OSError as e:errors.append(str(e))
        total+=part['bytes']
    if total!=row['bytes']:errors.append('TOTAL_SIZE:'+row['path'])
    if row['relative_destination'].startswith('geometry_cache/'):
        geometry_bytes+=row['bytes'];files.append({k:row.get(k) for k in ['relative_destination','bytes','storage','sha256']})
external=[]
for ref in manifest.get('external_refs',[]):
    if ref['role']=='geometry_cache_tail34':
        d=Path(ref['destination']); observed=[]
        for p in d.iterdir():
            if p.is_file():
                with p.open('rb') as f:
                    if not f.read(1):errors.append('EMPTY_TAIL:'+str(p))
                observed.append({'path':str(p),'bytes':p.stat().st_size})
        total=sum(v['bytes'] for v in observed)
        expected=ref.get('destination_stats',{})
        if expected and (total!=expected['bytes'] or len(observed)!=expected['files']):errors.append('TAIL_GEOMETRY_SIZE_COUNT')
        geometry_bytes+=total;external.append({'role':ref['role'],'files':observed,'expected':expected})
print(json.dumps({'root':str(root),'manifest':str(path),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'all_verified':not errors,'verification':'manifest structure, offsets, file readability and sizes; no full content rehash','geometry_bytes':geometry_bytes,'geometry_files':files,'external':external,'errors':errors}))
'''
    remote=' '.join(('python3','-c',shlex.quote(script),shlex.quote(run['output_root'])))
    c=_ssh(run,reader_node or 5000,remote,timeout=180)
    if c.returncode:raise TaskError('BUNDLE_ARCHIVE_AUDIT_FAILED:'+c.stderr[-1000:])
    return {'run_id':run['run_id'],**json.loads(c.stdout)}


def archive_paths(run: dict[str, Any], reader_node: int | None = None) -> dict[str, Any]:
    if not run.get('archive_catalog'):
        return bundle_archive_paths(run, reader_node)
    source = (PROJECT_ROOT / 'formal_evaluation/cpfs_release_audit.py').read_text()
    completed = _ssh(run, reader_node if reader_node is not None else run['storage_probe']['node'], shlex.join([
        'python3', '-c', source, json.dumps({'archive_catalog': run['archive_catalog']})]), timeout=180)
    if completed.returncode:
        raise TaskError('ARCHIVE_VERIFICATION_FAILED:' + str(completed.returncode) + ':' + completed.stderr[-2000:])
    return {'run_id': run['run_id'], **json.loads(completed.stdout)}


def output_paths(run: dict[str, Any]) -> dict[str, str]:
    if run.get("method_root_output"):
        if len(run["methods"]) != 1:
            raise TaskError("METHOD_ROOT_OUTPUT_REQUIRES_SINGLE_METHOD")
        return {run["methods"][0]: str(run["output_root"])}
    overrides = run.get("output_root_overrides", {})
    return {method: f"{overrides.get(method, run['output_root'])}/{method}" for method in run["methods"]}


def remote_progress(run: dict[str, Any], node: int, probes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Read progress for exact registered paths on one registered node."""
    ssh = run.get("ssh")
    if not isinstance(ssh, dict) or not ssh.get("host") or not ssh.get("key"):
        return {probe["key"]: {"error": "REMOTE_PROBE_NOT_CONFIGURED"} for probe in probes}
    remote = " ".join((
        "python3", "-c", shlex.quote(REMOTE_PROGRESS_PROBE),
        shlex.quote(json.dumps(probes, sort_keys=True)),
    ))
    completed = _ssh(run, node, remote, timeout=30)
    if completed.returncode:
        detail = completed.stderr.strip() or "no stderr"
        return {probe["key"]: {"error": f"REMOTE_PROBE_SSH_EXIT:{completed.returncode}:{detail[-1500:]}"}
                for probe in probes}
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        return {probe["key"]: {"error": "REMOTE_PROBE_INVALID_RESPONSE"} for probe in probes}
    if not isinstance(payload, dict):
        return {probe["key"]: {"error": "REMOTE_PROBE_INVALID_RESPONSE"} for probe in probes}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def audit_historical_backfills(registry: dict[str, Any], *, verify_coverage: bool = False) -> dict[str, Any]:
    """List verified legacy prediction roots for the narrowly authorized datasets.

    This deliberately accepts no path, host, node, method, or process selector.
    The sole search scope and node set come from the registry and the project
    policy's historical-backfill exception.
    """
    allowed = {"h2o", "taco", "hoi4d"}
    template_ssh = registry.get("run_template", {}).get("ssh")
    if not isinstance(template_ssh, dict) or not template_ssh.get("host") or not template_ssh.get("key"):
        raise TaskError("REMOTE_PROBE_NOT_CONFIGURED")
    nodes = set()
    template_probe = registry.get("run_template", {}).get("storage_probe", {})
    for run in registry.get("runs", {}).values():
        if not isinstance(run, dict):
            continue
        logical = registry.get("logical_tasks", {}).get(run.get("logical_task_id"), {})
        if logical.get("dataset") not in allowed:
            continue
        probe = run.get("storage_probe", template_probe)
        if isinstance(probe, dict) and probe.get("node") is not None:
            nodes.add(int(probe["node"]))
    if not nodes:
        raise TaskError("HISTORICAL_AUDIT_NODE_NOT_REGISTERED")

    remote = " ".join((
        "python3", "-c", shlex.quote(HISTORICAL_BACKFILL_AUDIT_PROBE),
        shlex.quote(json.dumps(sorted(allowed))),
    ))
    per_node: dict[str, Any] = {}
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for node in sorted(nodes):
        completed = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-i", os.path.expanduser(str(template_ssh["key"])), "-p", str(node),
            f"root@{template_ssh['host']}", remote,
        ], text=True, capture_output=True, timeout=120, check=False)
        if completed.returncode:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no stderr"
            per_node[str(node)] = {
                "status": "unavailable",
                "error": f"SSH_EXIT:{completed.returncode}",
                "detail": detail[-500:],
            }
            continue
        try:
            observed = json.loads(completed.stdout)
            if not isinstance(observed, dict) or observed.get("status") != "ok":
                raise ValueError("invalid payload")
        except ValueError:
            per_node[str(node)] = {"status": "unavailable", "error": "INVALID_AUDIT_RESPONSE"}
            continue
        per_node[str(node)] = {"status": "ok", "scanned_candidate_roots": observed.get("scanned_candidate_roots", [])}
        for row in observed.get("rows", []):
            if not isinstance(row, dict):
                continue
            key = (str(row.get("dataset")), str(row.get("candidate_root")), str(row.get("method")))
            merged = rows_by_key.setdefault(key, {**row, "observed_nodes": []})
            merged["observed_nodes"].append(node)
            merged["valid_output_count"] = max(int(merged["valid_output_count"]), int(row.get("valid_output_count", 0)))
    result = {
        "status": "ok",
        "authorized_scope": "/mnt/workspace/sjc/eval_artifacts; H2O,TACO,HOI4D; read-only",
        "nodes": per_node,
        "candidates": sorted(rows_by_key.values(), key=lambda row: (row["dataset"], row["candidate_root"], row["method"])),
        "next_step": "verify candidate coverage before idempotent registration",
    }
    if verify_coverage:
        node = min(nodes)
        coverage_remote = " ".join((
            "python3", "-c", shlex.quote(HISTORICAL_BACKFILL_COVERAGE_PROBE),
            shlex.quote(json.dumps(HISTORICAL_BACKFILL_COVERAGE, sort_keys=True)),
        ))
        completed = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-i", os.path.expanduser(str(template_ssh["key"])), "-p", str(node),
            f"root@{template_ssh['host']}", coverage_remote,
        ], text=True, capture_output=True, timeout=60, check=False)
        try:
            coverage = json.loads(completed.stdout) if not completed.returncode else None
        except ValueError:
            coverage = None
        result["coverage"] = {
            "node": node,
            "status": "ok" if isinstance(coverage, dict) else "unavailable",
            "methods": coverage if isinstance(coverage, dict) else {},
        }
        if isinstance(coverage, dict):
            result["next_step"] = (
                "register only methods whose coverage.verified is true"
                if all(row.get("verified") for dataset in coverage.values() for row in dataset.values())
                else "do not register: one or more historical coverage checks failed"
            )
    return result


def audit_contact_artifacts(registry: dict[str, Any], datasets: list[str] | None = None,
                            *, verify_coverage: bool = False) -> dict[str, Any]:
    """Run the fixed-scope, read-only contact artifact audit on registered nodes."""
    ssh = registry.get("run_template", {}).get("ssh")
    if not isinstance(ssh, dict) or not ssh.get("host") or not ssh.get("key"):
        raise TaskError("REMOTE_PROBE_NOT_CONFIGURED")
    template_probe = registry.get("run_template", {}).get("storage_probe", {})
    registered_nodes = {
        int(run.get("storage_probe", template_probe).get("node", 5001))
        for run in registry.get("runs", {}).values()
        if isinstance(run, dict) and isinstance(run.get("storage_probe", template_probe), dict)
    }
    # CPFS is shared across registered DSW nodes. Probe in order until one is
    # reachable instead of assuming the numerically smallest port is online.
    nodes = sorted(registered_nodes)
    selected = datasets or sorted(SIX_DATASETS)
    remote = " ".join(("python3", "-c", shlex.quote(CONTACT_ARTIFACT_AUDIT_PROBE),
                         shlex.quote(json.dumps(selected))))
    observed = {}
    successful_node = None
    for node in nodes:
        completed = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-i", os.path.expanduser(str(ssh["key"])), "-p", str(node),
            f"root@{ssh['host']}", remote,
        ], text=True, capture_output=True, timeout=120, check=False)
        try:
            payload = json.loads(completed.stdout) if not completed.returncode else None
        except ValueError:
            payload = None
        observed[str(node)] = payload if isinstance(payload, dict) else {
            "status": "unavailable", "error": f"SSH_EXIT:{completed.returncode}",
        }
        if isinstance(payload, dict):
            successful_node = node
            break
    result = {"status": "ok", "nodes": observed,
              "next_step": "verify candidate counts and report content before idempotent registration"}
    if verify_coverage:
        spec = {dataset: CONTACT_PREDICTION_COVERAGE[dataset]
                for dataset in selected if dataset in CONTACT_PREDICTION_COVERAGE}
        if spec:
            coverage_remote = " ".join(("python3", "-c", shlex.quote(CONTACT_PREDICTION_COVERAGE_PROBE),
                                          shlex.quote(json.dumps(spec, sort_keys=True))))
            if successful_node is None:
                result["coverage"] = {"status": "unavailable", "error": "NO_REACHABLE_REGISTERED_NODE"}
                result.pop("nodes", None)
                return result
            node = successful_node
            completed = subprocess.run([
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-i", os.path.expanduser(str(ssh["key"])), "-p", str(node),
                f"root@{ssh['host']}", coverage_remote,
            ], text=True, capture_output=True, timeout=60, check=False)
            try:
                coverage = json.loads(completed.stdout) if not completed.returncode else None
            except ValueError:
                coverage = None
            result["coverage"] = coverage if isinstance(coverage, dict) else {"status": "unavailable"}
        # Coverage mode is a second-stage check; avoid re-emitting a large
        # candidate inventory that was already returned by the first-stage audit.
        result.pop("nodes", None)
    return result


def _audit_cpfs_eval_roots(registry: dict[str, Any], names: list[str]) -> dict[str, Any]:
    ssh = registry.get("run_template", {}).get("ssh")
    if not isinstance(ssh, dict) or not ssh.get("host") or not ssh.get("key"):
        raise TaskError("REMOTE_PROBE_NOT_CONFIGURED")
    remote = " ".join(("python3", "-c", shlex.quote(STANDARD10_CPFS_STORAGE_PROBE),
                       shlex.quote(json.dumps(names))))
    completed = subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-i", os.path.expanduser(str(ssh["key"])), "-p", "5000",
        f"root@{ssh['host']}", remote,
    ], text=True, capture_output=True, timeout=1800, check=False)
    if completed.returncode:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no stderr"
        raise TaskError(f"STANDARD10_STORAGE_AUDIT_FAILED:{completed.returncode}:{detail[-500:]}")
    try:
        payload = json.loads(completed.stdout)
    except ValueError as error:
        raise TaskError("STANDARD10_STORAGE_AUDIT_INVALID_RESPONSE") from error
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise TaskError("STANDARD10_STORAGE_AUDIT_INVALID_RESPONSE")
    payload["observed_node"] = 5000
    return payload


def audit_standard10_cpfs_storage(registry: dict[str, Any]) -> dict[str, Any]:
    """Measure the seven fixed standard10 CPFS roots without following links."""
    return _audit_cpfs_eval_roots(registry, STANDARD10_CPFS_ROOTS)


def audit_egofound3r_formal_model(registry: dict[str, Any]) -> dict[str, Any]:
    """Inspect the one authorized EgoFound3R release checkpoint and nearby provenance."""
    ssh = registry.get("run_template", {}).get("ssh")
    if not isinstance(ssh, dict) or not ssh.get("host") or not ssh.get("key"):
        raise TaskError("REMOTE_PROBE_NOT_CONFIGURED")
    remote = " ".join((
        "env",
        shlex.quote(f"PYTHONPATH={EGOFOUND3R_INFERENCE_ROOT}"),
        shlex.quote(EGOFOUND3R_MODEL_PYTHON),
        "-c",
        shlex.quote(REMOTE_EGOFOUND3R_FORMAL_MODEL_AUDIT),
        shlex.quote(str(EGOFOUND3R_FORMAL_MODEL_ROOT)),
        shlex.quote(str(EGOFOUND3R_FORMAL_CHECKPOINT)),
        shlex.quote(str(EGOFOUND3R_INFERENCE_ROOT)),
        shlex.quote(str(EGOFOUND3R_TRAINING_SOURCE_ROOT)),
    ))
    errors: dict[str, str] = {}
    missing: dict[str, Any] | None = None
    for node in (5000, 5001, 6001):
        completed = _ssh({"ssh": ssh}, node, remote, timeout=300)
        if completed.returncode:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no stderr"
            errors[str(node)] = f"SSH_EXIT:{completed.returncode}:{detail[-500:]}"
            continue
        try:
            payload = json.loads(completed.stdout)
        except ValueError:
            errors[str(node)] = "INVALID_AUDIT_RESPONSE"
            continue
        if not isinstance(payload, dict):
            errors[str(node)] = "INVALID_AUDIT_RESPONSE"
            continue
        payload["observed_node"] = node
        if payload.get("status") == "ok":
            return payload
        missing = missing or payload
    if missing is not None:
        missing["node_errors"] = errors
        return missing
    raise TaskError("EGOFOUND3R_FORMAL_MODEL_AUDIT_FAILED:" + json.dumps(errors, sort_keys=True))


def audit_cpfs_eval_top_level(registry: dict[str, Any], names: list[str] | None = None) -> dict[str, Any]:
    """Inventory every direct child of the fixed CPFS eval artifact root."""
    return _audit_cpfs_eval_roots(registry, names or [])


def list_cpfs_eval_top_level(registry: dict[str, Any], contains: str | None = None) -> dict[str, Any]:
    """List direct children without recursively measuring them."""
    root = "/mnt/workspace/sjc/DATA/eval_artifacts"
    script = """import json,os,sys\nfrom pathlib import Path\nr=Path(sys.argv[1]); rows=[]\nfor p in r.iterdir():\n s=p.lstat(); rows.append({'name':p.name,'path':str(p),'symlink':p.is_symlink(),'target':os.readlink(p) if p.is_symlink() else None,'mtime_ns':s.st_mtime_ns})\nprint(json.dumps({'root':str(r),'entries':sorted(rows,key=lambda x:x['name'])}))\n"""
    completed = _ssh({"ssh": registry["run_template"]["ssh"]}, 5000,
                     shlex.join(["python3", "-c", script, root]), timeout=180)
    if completed.returncode:
        raise TaskError("CPFS_TOP_LEVEL_LIST_FAILED:" + str(completed.returncode))
    result = json.loads(completed.stdout)
    if contains:
        result["entries"] = [row for row in result["entries"] if contains in row["name"]]
    return {"observed_node": 5000, **result}


def audit_cpfs_eval_storage(registry: dict[str, Any], *, cached: bool = False) -> dict[str, Any]:
    from formal_evaluation.cpfs_eval_storage import registered_roots
    cache = PROJECT_ROOT / '.auto_scheduler/cpfs_eval_storage_snapshot.json'
    roots = registered_roots(registry)
    if cached:
        if not cache.is_file():
            raise TaskError('NO_STORAGE_SNAPSHOT:run audit-cpfs-eval-storage first')
        result = read_json(cache)
        if result.get('registered_roots') != roots:
            raise TaskError('STORAGE_SNAPSHOT_SCOPE_CHANGED:run audit-cpfs-eval-storage again')
        return {**result, 'cached': True, 'age_seconds': int(time.time()) - result['checked_at_epoch']}
    source = (PROJECT_ROOT / 'formal_evaluation/cpfs_eval_storage.py').read_text()
    run = {'ssh': registry.get('run_template', {}).get('ssh')}
    completed = _ssh(run, 5000, shlex.join(['python3', '-c', source, json.dumps(roots)]), timeout=1800)
    if completed.returncode:
        raise TaskError('CPFS_EVAL_STORAGE_AUDIT_FAILED:' + str(completed.returncode))
    result = json.loads(completed.stdout)
    result.update(registered_roots=roots, observed_node=5000, cached=False)
    atomic_json(cache, result)
    return result


def audit_run_storage(registry: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Measure and check process references for one exact registered CPFS output."""
    run = merged_run(registry, run_id)
    root = str(run.get("output_root", ""))
    if not root.startswith(("/mnt/workspace/", "/mnt/cpfs/")):
        raise TaskError("RUN_OUTPUT_NOT_ON_CPFS:" + run_id)
    ssh = {"ssh": registry["run_template"]["ssh"]}
    storage_source = (PROJECT_ROOT / "formal_evaluation/cpfs_eval_storage.py").read_text()
    completed = _ssh(ssh, 5000, shlex.join([
        "python3", "-c", storage_source, json.dumps([root]),
    ]), timeout=1800)
    if completed.returncode:
        raise TaskError("RUN_STORAGE_AUDIT_FAILED:" + str(completed.returncode))
    dependencies_source = (PROJECT_ROOT / "formal_evaluation/cpfs_release_audit.py").read_text()
    dependencies = _ssh(ssh, 5000, shlex.join([
        "python3", "-c", dependencies_source, json.dumps({"roots": [root]}),
    ]), timeout=180)
    if dependencies.returncode:
        raise TaskError("RUN_DEPENDENCY_AUDIT_FAILED:" + str(dependencies.returncode))
    return {
        "run_id": run_id,
        "logical_task_id": run.get("logical_task_id"),
        "output_root": root,
        "observed_node": 5000,
        "storage": json.loads(completed.stdout),
        "dependencies": json.loads(dependencies.stdout),
    }


def audit_cpfs_release_dependencies(registry: dict[str, Any], *, mirrors: bool = False, mirror_dependencies: bool = False, drift: bool = False) -> dict[str, Any]:
    spec = registry.get('storage_release_audit')
    if not spec:
        raise TaskError('STORAGE_RELEASE_AUDIT_NOT_REGISTERED')
    source = (PROJECT_ROOT / 'formal_evaluation/cpfs_release_audit.py').read_text()
    if mirrors:
        completed = _ssh({'ssh': registry['run_template']['ssh']}, 5000,
                         shlex.join(['python3', '-c', source, json.dumps({'drift': spec['drift']} if drift else {'mirrors': spec['mirrors']})]), timeout=1800)
        if completed.returncode:
            raise TaskError('RELEASE_MIRROR_AUDIT_FAILED:' + str(completed.returncode))
        return json.loads(completed.stdout)
    result = {}
    for node in spec['nodes']:
        roots = ([p['source'] for p in spec['mirrors']] if mirror_dependencies else
                 spec.get('node_sources', {}).get(str(node), spec['sources']))
        if mirror_dependencies and node == 4091:
            roots = [p.replace('/mnt/workspace/', '/mnt/cpfs/') for p in roots]
        payload = {'roots': roots} if mirror_dependencies else roots
        completed = _ssh({'ssh': registry['run_template']['ssh']}, node,
                         shlex.join(['python3', '-c', source, json.dumps(payload)]), timeout=180)
        result[str(node)] = (json.loads(completed.stdout) if not completed.returncode else
                             {'safe_from_observed_processes': False, 'error': 'SSH_EXIT:' + str(completed.returncode)})
    return {'nodes': result, 'all_safe_from_observed_processes': all(x['safe_from_observed_processes'] for x in result.values()),
            'sources': spec['sources'], 'inactive_nodes': spec.get('inactive_nodes', {}), 'checked_at_epoch': int(time.time())}


def compare_migrated_eval_artifacts(registry: dict[str, Any]) -> dict[str, Any]:
    """Compare the fixed CPFS and OSSFS eval artifact roots without hashes."""
    ssh = registry.get("run_template", {}).get("ssh")
    if not isinstance(ssh, dict) or not ssh.get("host") or not ssh.get("key"):
        raise TaskError("REMOTE_PROBE_NOT_CONFIGURED")
    remote = " ".join(("python3", "-c", shlex.quote(EVAL_ARTIFACT_TREE_COMPARE_PROBE)))
    nodes = []
    for run_id in registry.get("runs", {}):
        try:
            nodes.append(int(merged_run(registry, run_id).get("storage_probe", {}).get("node", 5001)))
        except (KeyError, TaskError):
            continue
    nodes = sorted(set(nodes) | {5000})
    errors = {}
    for node in nodes:
        completed = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-i", os.path.expanduser(str(ssh["key"])), "-p", str(node),
            f"root@{ssh['host']}", remote,
        ], text=True, capture_output=True, timeout=1800, check=False)
        if completed.returncode:
            errors[str(node)] = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no stderr"
            continue
        try:
            payload = json.loads(completed.stdout)
        except ValueError:
            errors[str(node)] = "invalid response"
            continue
        if isinstance(payload, dict) and payload.get("status") == "ok":
            payload["observed_node"] = node
            return payload
        errors[str(node)] = "invalid response"
    raise TaskError(f"ARTIFACT_COMPARE_FAILED:{json.dumps(errors, sort_keys=True)}")


def audit_migrated_eval_symlinks(registry: dict[str, Any]) -> dict[str, Any]:
    """Audit fixed CPFS migrated artifact links and their rebased targets."""
    ssh = registry.get("run_template", {}).get("ssh")
    if not isinstance(ssh, dict) or not ssh.get("host") or not ssh.get("key"):
        raise TaskError("REMOTE_PROBE_NOT_CONFIGURED")
    remote = " ".join(("python3", "-c", shlex.quote(EVAL_ARTIFACT_SYMLINK_AUDIT_PROBE)))
    try:
        completed = subprocess.run([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-i", os.path.expanduser(str(ssh["key"])), "-p", "5000",
            f"root@{ssh['host']}", remote,
        ], text=True, capture_output=True, timeout=1200, check=False)
    except subprocess.TimeoutExpired as error:
        raise TaskError("ARTIFACT_SYMLINK_AUDIT_TIMEOUT") from error
    if completed.returncode:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no stderr"
        raise TaskError(f"ARTIFACT_SYMLINK_AUDIT_FAILED:{completed.returncode}:{detail[-500:]}")
    try:
        payload = json.loads(completed.stdout)
    except ValueError as error:
        raise TaskError("ARTIFACT_SYMLINK_AUDIT_INVALID_RESPONSE") from error
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise TaskError("ARTIFACT_SYMLINK_AUDIT_INVALID_RESPONSE")
    return payload


def refresh_active_run(registry: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Synchronize active jobs from their registered remote artifacts.

    This never discovers a path or process: every probe derives from the exact
    state job, its registered output root, and its registered node.
    """
    run = merged_run(registry, run_id)
    state_path = PROJECT_ROOT / run["state_file"]
    state = read_json(state_path)
    section, registered_jobs = run_jobs(registry, run)
    paths = output_paths(run)
    by_node: dict[int, list[dict[str, Any]]] = {}
    for label, key in registered_jobs:
        job = state.get(section, {}).get(key)
        status = str(job.get("status")) if isinstance(job, dict) else ""
        audit_path = (job.get("audit_path") if isinstance(job, dict) else None) or run.get("launch", {}).get("audit_path")
        progress_log = bool(run.get("progress_log")) and job.get("handle_path")
        audit_log = progress_log or (status == "queue_exited_needs_audit" and job.get("handle_path") and not job.get("audit_log_tail"))
        audit_summary = status == "queue_exited_needs_audit" and audit_path and not job.get("audit_summary")
        report_check = bool(run.get("report_path") or run.get("report_paths"))
        artifact_audit = bool(run.get("artifact_audit", run.get("task_type", "evaluation") == "evaluation")) and str(run.get("phase", "")).startswith("formal")
        completion_check = status == "queue_exited_needs_audit" and bool(run.get("completion_artifacts"))
        if not isinstance(job, dict) or (status not in ACTIVE_STATES and not audit_log and not audit_summary and not report_check and not artifact_audit and not completion_check):
            continue
        node = job.get("node") or run.get("storage_probe", {}).get("node")
        if node is None:
            continue
        output_root = str(paths.get(label, job.get("output_root") or run["output_root"])) if artifact_audit else str(job.get("output_root") or run["output_root"])
        prediction_root = run.get("prediction_root")
        if prediction_root is None and (str(run.get("phase", "")).startswith("formal") or str(run.get("phase", "")).startswith("pad-hand")):
            prediction_root = (
                output_root if artifact_audit else str(job.get("output_root") or paths.get(label, output_root)) + "/formal"
            )
        by_node.setdefault(int(node), []).append({
            "key": key,
            "pid": job.get("pid"),
            "pgid": job.get("pgid"),
            "handle_path": job.get("handle_path"),
            "output_root": output_root,
            "prediction_root": prediction_root,
            "completion_path": job.get("completion_path"),
            "completion_artifacts": run.get("completion_artifacts"),
            "distance_window_progress": run.get("distance_window_progress", False),
            "target_windows": int(run.get("progress_target", run["target_windows_per_method"])),
            "audit_log": bool(audit_log),
            "audit_path": audit_path,
            "report_path": run.get("report_path"),
            "report_paths": run.get("report_paths", []),
            "report_dataset": run.get("dataset"),
            "report_metric_tokens": run.get("report_metric_tokens", []),
            "report_units": run.get("report_units"),
            "runtime_sources": {
                str(Path(run["launch"]["runtime_root"]) / name): hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                for name in run.get("launch", {}).get("support_files", [])
                if name in {"formal_evaluation/evaluate_six_dataset.py", "formal_evaluation/hand/metrics.py", "formal_evaluation/scene/metrics.py"}
                and (PROJECT_ROOT / name).is_file()
            },
            "task_type": run.get("task_type"),
            "partial_output_stats": bool(run.get("partial_output_stats")),
            "accept_report": str(run.get("phase", "")).startswith("metrics"),
        })

    observed: dict[str, dict[str, Any]] = {}
    for node, probes in by_node.items():
        observed.update(remote_progress(run, node, probes))

    updated = []
    for _, key in registered_jobs:
        job = state.get(section, {}).get(key)
        remote = observed.get(key)
        if not isinstance(job, dict) or not remote or remote.get("error"):
            continue
        patch: dict[str, Any] = {"observed_at_epoch": int(time.time())}
        status = remote.get("status")
        process_status = remote.get("process_status")
        if status == "done":
            patch["status"] = "done"
        elif process_status == "running":
            patch["status"] = "running"
        elif process_status == "paused":
            patch["status"] = "paused"
        elif process_status == "exited":
            patch["status"] = "queue_exited_needs_audit"
        elif process_status in {"missing_handle", "stale_handle"}:
            patch["status"] = "control_handle_error"
        if isinstance(remote.get("count"), int):
            progress_cap = int(run.get("progress_target", run["target_windows_per_method"]))
            patch["count"] = min(progress_cap, max(0, int(remote["count"])))
            patch["progress_source"] = remote.get("source")
        if isinstance(remote.get("audit_log_tail"), str):
            patch["audit_log_tail"] = remote["audit_log_tail"]
        if isinstance(remote.get("audit_summary"), dict):
            patch["audit_summary"] = remote["audit_summary"]
        if isinstance(remote.get("report_summary"), dict):
            patch["report_summary"] = remote["report_summary"]
        if isinstance(remote.get("completion_evidence"), dict):
            patch["completion_evidence"] = remote["completion_evidence"]
        if any(job.get(field) != value for field, value in patch.items()):
            job.update(patch)
            updated.append(key)
    if updated:
        atomic_json(state_path, state)
    return {
        "checked_jobs": sum(len(probes) for probes in by_node.values()),
        "updated_jobs": updated,
        "errors": {key: value["error"] for key, value in observed.items() if value.get("error")},
    }


def inspect_storage(run: dict[str, Any]) -> dict[str, Any]:
    probe = run.get("storage_probe")
    if not probe:
        return {"status": "not_registered"}
    node, mount = int(probe["node"]), str(probe["mount"])
    remote = " ".join(shlex.quote(value) for value in (
        "df", "-B1", "--output=size,used,avail,pcent,target", "--", mount,
    ))
    completed = _ssh(run, node, remote, timeout=30)
    if completed.returncode:
        return {"status": "unavailable", "node": node, "mount": mount,
                "error": f"SSH_EXIT:{completed.returncode}"}
    try:
        total, used, available, percent, observed_mount = completed.stdout.strip().splitlines()[-1].split(maxsplit=4)
        if observed_mount != mount:
            raise ValueError("mount mismatch")
        return {
            "status": "ok", "node": node, "mount": mount,
            "total_bytes": int(total), "used_bytes": int(used), "available_bytes": int(available),
            "used_percent": int(percent.removesuffix("%")), "checked_at_epoch": int(time.time()),
        }
    except (IndexError, ValueError):
        return {"status": "unavailable", "node": node, "mount": mount, "error": "INVALID_DF_RESPONSE"}


def inspect_run(registry: dict[str, Any], run_id: str, *, include_storage: bool = False) -> dict[str, Any]:
    run = merged_run(registry, run_id)
    if run.get("artifact_catalog"):
        result = catalog_paths(run)
        if include_storage:
            result["storage"] = inspect_storage(run)
        return result

    remote_sync = refresh_active_run(registry, run_id)
    state = read_json(PROJECT_ROOT / run["state_file"])
    section, registered_jobs = run_jobs(registry, run)
    rows = {}
    for label, key in registered_jobs:
        job = state.get(section, {}).get(key)
        if job is None:
            rows[label] = {"status": "not_registered_in_state"}
            continue
        rows[label] = {
            field: job[field]
            for field in ("status", "desired_state", "node", "gpu", "pid", "pgid", "handle_path", "count", "progress_source", "observed_at_epoch", "audit_log_tail", "audit_summary", "report_summary", "completion_evidence")
            if job.get(field) is not None
        }
    counts = Counter(str(row["status"]) for row in rows.values())
    target = int(run["target_windows_per_method"])
    total_target = target * len(run["methods"])
    known = total_target if rows and all(row.get("status") == "done" for row in rows.values()) else sum(
        target if row.get("status") == "done" else int(row.get("count", 0)) for row in rows.values()
    )
    result = {
        "run_id": run_id,
        "task_id": run["logical_task_id"],
        "task_type": run.get("task_type", "evaluation"),
        "node_path_layout": run.get("node_path_layout", {}),
        "state_counts": dict(sorted(counts.items())),
        "progress": {"known_complete_windows": known, "target_windows": total_target},
        "method_states": {method: row["status"] for method, row in rows.items()},
        "method_counts": {method: row.get("count") for method, row in rows.items() if "count" in row},
        "active_locations": {
            method: {field: row[field] for field in ("node", "gpu", "pid", "pgid") if field in row}
            for method, row in rows.items()
            if row.get("status") in {"running", "paused", "pausing", "resuming"}
        },
        "audit_log_tails": {
            method: row["audit_log_tail"]
            for method, row in rows.items()
            if row.get("audit_log_tail")
        },
        "audit_summaries": {
            method: row["audit_summary"]
            for method, row in rows.items()
            if row.get("audit_summary")
        },
        "report_summaries": {
            method: row["report_summary"]
            for method, row in rows.items()
            if row.get("report_summary")
        },
        "state_file": str(PROJECT_ROOT / run["state_file"]),
        "completion_evidence": {
            method: row["completion_evidence"] for method, row in rows.items()
            if row.get("completion_evidence")
        },
        "state_mtime": int((PROJECT_ROOT / run["state_file"]).stat().st_mtime),
        "remote_sync": remote_sync,
    }
    if include_storage:
        result["storage"] = inspect_storage(run)
        if run.get("readability_probe"):
            result["readability"] = inspect_readability(run)
    return result


def overview_six_datasets(registry: dict[str, Any]) -> dict[str, Any]:
    """Synchronize and compactly list the registered six-dataset formal runs."""
    rows = []
    for task_id, logical in sorted(registry.get("logical_tasks", {}).items()):
        if logical.get("dataset") not in SIX_DATASETS or logical.get("phase") not in OVERVIEW_PHASES:
            continue
        run_id = str(logical.get("latest"))
        inspection = inspect_run(registry, run_id)
        rows.append({
            "dataset": logical["dataset"], "phase": logical["phase"],
            "task_id": task_id, "run_id": run_id,
            "states": inspection["state_counts"], "progress": inspection["progress"],
            "output_root": merged_run(registry, run_id)["output_root"],
        })
    return {"status": "ok", "rows": rows}


def set_desired_state(registry: dict[str, Any], run_id: str, desired: str, controls_path: Path) -> dict[str, Any]:
    run = merged_run(registry, run_id)
    inspection = inspect_run(registry, run_id)
    if desired == "paused" and not any(
        status in {"running", "pending", "resuming", "pausing"}
        for status in inspection["method_states"].values()
    ):
        raise TaskError(f"NO_ACTIVE_JOB:{run_id}")
    state = read_json(PROJECT_ROOT / run["state_file"])
    section, registered_jobs = run_jobs(registry, run)
    active = [
        state.get(section, {}).get(key, {})
        for label, key in registered_jobs
        if inspection["method_states"].get(label) in {"running", "paused", "resuming", "pausing"}
    ]
    if desired == "paused" and any(not job.get("handle_path") or not job.get("node") for job in active):
        raise TaskError(f"UNCONTROLLED_LEGACY_JOB:{run_id}")
    controls = read_json(controls_path) if controls_path.exists() else {"schema_version": "task_controls_v1", "jobs": {}}
    if controls.get("schema_version") != "task_controls_v1":
        raise TaskError("UNSUPPORTED_CONTROLS")
    jobs = controls.setdefault("jobs", {})
    for _, key in registered_jobs:
        jobs[key if section == "jobs" else f"{section}::{key}"] = desired
    controls["updated_at_epoch"] = int(time.time())
    controls["requested_run_id"] = run_id
    atomic_json(controls_path, controls)
    return {
        "run_id": run_id,
        "requested_state": desired,
        "job_count": len(registered_jobs),
        "control_file": str(controls_path),
        "accepted": True,
    }


def signal_registered_jobs(registry: dict[str, Any], run_id: str, signal_name: str, drain_relay: bool = False) -> dict[str, Any]:
    run = merged_run(registry, run_id)
    state = read_json(PROJECT_ROOT / run["state_file"])
    section, registered_jobs = run_jobs(registry, run)
    wanted = {"STOP": {"running", "pausing"}, "CONT": {"paused", "resuming"}}[signal_name]
    grouped: dict[int, list[str]] = {}
    for label, key in registered_jobs:
        job = state.get(section, {}).get(key, {})
        if job.get("status") not in wanted:
            continue
        if not job.get("node") or not job.get("handle_path"):
            raise TaskError(f"UNCONTROLLED_LEGACY_JOB:{run_id}:{label}")
        grouped.setdefault(int(job["node"]), []).append(str(job["handle_path"]))
    if not grouped:
        raise TaskError(f"NO_SIGNALABLE_JOB:{run_id}:{signal_name}")
    results = {}
    for port, handles in grouped.items():
        controller = run.get("launch", {}).get("controller_path") or run["remote_controller"]
        remote = " ".join([
            shlex.quote(str(run["remote_python"])),
            shlex.quote(str(controller)),
            "signal", "--signal", signal_name,
            *(["--drain-relay"] if drain_relay else []),
            *[f"--handle {shlex.quote(handle)}" for handle in handles],
        ])
        if run.get("coordinator_only_pause"):
            if signal_name != "STOP":
                raise TaskError("SPLIT_COORDINATOR_RESUME_FORBIDDEN")
            probe = r"""
import json, os, signal, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
from remote_task_control import checked_handle, proc_identity
result = {}
for path in json.loads(sys.argv[2]):
    handle, current = checked_handle(Path(path))
    assert current is not None and current['state'] != 'T', 'COORDINATOR_NOT_RUNNING'
    pid = int(handle['pid'])
    children = [int(v) for v in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    assert len(children) == 1, ('EXPECTED_CURRENT_DATASET_CHILD', children)
    argv = Path(f'/proc/{children[0]}/cmdline').read_bytes().decode().split('\0')
    assert any(a.endswith('/run_egofound3r_stride_evaluation.py') for a in argv)
    assert sys.argv[3] in argv, 'CURRENT_DATASET_CHANGED'
    os.kill(pid, signal.SIGSTOP)
    time.sleep(0.1)
    assert proc_identity(pid)['state'] == 'T'
    child = proc_identity(children[0])
    assert child and child['state'] != 'T', 'CURRENT_DATASET_NOT_RUNNING'
    result[path] = dict(status='signal_sent', signal='STOP', observed_state='T', preserved_child=child)
print(json.dumps(dict(ok=True, handles=result)))
"""
            remote = " ".join([shlex.quote(str(run["remote_python"])), "-c", shlex.quote(probe),
                               shlex.quote(str(controller)), shlex.quote(json.dumps(handles)),
                               shlex.quote(run["coordinator_only_pause"]["current_dataset_spec"])])
        completed = _ssh(run, port, remote, timeout=30)
        lines = completed.stdout.strip().splitlines()
        if not lines:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "empty output"
            raise TaskError(f"REMOTE_CONTROL_INVALID_RESPONSE:{port}:{completed.returncode}:{detail[-300:]}")
        try:
            payload = json.loads(lines[-1])
        except ValueError as error:
            raise TaskError(f"REMOTE_CONTROL_INVALID_RESPONSE:{port}") from error
        if completed.returncode or not payload.get("ok"):
            raise TaskError(f"REMOTE_CONTROL_FAILED:{port}:{payload.get('error', completed.returncode)}")
        results[str(port)] = payload["handles"]
    return {"signal": signal_name, "nodes": results, "verified": True}


def _node_text(run: dict[str, Any], node: int, value: Any) -> str:
    """Translate a registered canonical CPFS path for one node's mount layout."""
    text = str(value)
    layout = run.get("node_path_layout", {})
    canonical = str(layout.get("canonical_prefix", ""))
    node_prefix = str(layout.get("node_prefixes", {}).get(str(node), canonical))
    return text.replace(canonical, node_prefix) if canonical and node_prefix != canonical else text


def _ssh(run: dict[str, Any], node: int, remote: str, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    instance = run.get("_current_instance")
    if instance and node != instance["port"]:
        raise TaskError(f"INSTANCE_PORT_MISMATCH:{node}:{instance['port']}")
    ssh = run["ssh"]
    return subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-i", os.path.expanduser(str(ssh["key"])), "-p", str(node),
        f"root@{ssh['host']}", _node_text(run, node, remote),
    ], text=True, capture_output=True, timeout=timeout, check=False)


def inspect_resources(registry: dict[str, Any], nodes_to_probe=None) -> dict[str, Any]:
    """Return one bounded CPU/GPU/memory/CPFS snapshot per registered DSW node."""
    run = registry.get("run_template", {})
    if not isinstance(run.get("ssh"), dict):
        raise TaskError("REMOTE_PROBE_NOT_CONFIGURED")
    allowed_nodes = tuple(int(node) for node in registry.get("policy", {}).get("resource_nodes", RESOURCE_NODES))
    nodes_to_probe = allowed_nodes if nodes_to_probe is None else nodes_to_probe
    def probe(node: int) -> tuple[int, dict[str, Any]]:
        if node not in allowed_nodes:
            raise TaskError(f"UNREGISTERED_RESOURCE_NODE:{node}")
        mount = registry.get("policy", {}).get("resource_node_entrances", {}).get(str(node))
        if not mount:
            raise TaskError(f"RESOURCE_NODE_ENTRANCE_MISSING:{node}")
        remote = " ".join((
            "python3", "-c", shlex.quote(REMOTE_RESOURCE_PROBE), shlex.quote(mount),
        ))
        completed = _ssh(run, node, remote)
        if completed.returncode:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no stderr"
            return node, {"status": "unavailable", "error": f"SSH_EXIT:{completed.returncode}",
                          "detail": detail[-300:]}
        try:
            payload = json.loads(completed.stdout)
        except ValueError:
            payload = {"ok": False, "error": "INVALID_RESOURCE_RESPONSE"}
        storage = payload.get("storage")
        if isinstance(storage, dict) and storage.get("total_bytes", 0) > 0:
            used_percent = 100 * storage["used_bytes"] / storage["total_bytes"]
            filled = min(20, int(20 * used_percent / 100))
            available_gib = storage["available_bytes"] / 1024**3
            storage.update({
                "used_percent": round(used_percent, 2),
                "available_percent": round(100 - used_percent, 2),
                "available_gib": round(available_gib, 2),
                "progress_bar": f"[{'█' * filled}{'░' * (20 - filled)}] "
                                f"{used_percent:.2f}% used | {available_gib:.2f} GiB free",
            })
        return node, ({"status": "ok", **payload} if payload.get("ok") else
                      {"status": "unavailable", "error": payload.get("error", "RESOURCE_PROBE_FAILED")})
    nodes: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(nodes_to_probe) or 1)) as pool:
        futures = [pool.submit(probe, node) for node in nodes_to_probe]
        for future in as_completed(futures):
            node, value = future.result()
            nodes[str(node)] = value
    return {"nodes": nodes}


def audit_baseline_git(registry: dict[str, Any], nodes_to_probe: list[int]) -> dict[str, Any]:
    """Read the fixed baseline checkout branch, HEAD, and dirty state via taskctl."""
    run = registry.get("run_template", {})
    if not isinstance(run.get("ssh"), dict):
        raise TaskError("REMOTE_PROBE_NOT_CONFIGURED")
    nodes = {}
    for node in nodes_to_probe:
        completed = _ssh(run, node, "python3 -c " + shlex.quote(REMOTE_BASELINE_GIT_AUDIT), timeout=120)
        if completed.returncode:
            nodes[str(node)] = {"ok": False, "error": f"SSH_EXIT:{completed.returncode}"}
            continue
        try:
            nodes[str(node)] = json.loads(completed.stdout)
        except ValueError:
            nodes[str(node)] = {"ok": False, "error": "INVALID_GIT_AUDIT_RESPONSE"}
    return {"repo": "/mnt/workspace/sjc/EgoFound3R-baselines", "nodes": nodes}


def inspect_readability(run: dict[str, Any]) -> dict[str, Any]:
    """Probe only the exact registered cache paths on registered reader nodes."""
    probe = run.get("readability_probe")
    if not probe:
        return {}
    nodes: dict[str, Any] = {}
    for node in probe["nodes"]:
        remote = " ".join((
            "python3", "-c", shlex.quote(REMOTE_READABILITY_PROBE),
            shlex.quote(str(run["output_root"])),
            shlex.quote(str(probe["dataset_subdir"])),
            shlex.quote(str(probe["index_file"])),
        ))
        completed = _ssh(run, int(node), remote)
        try:
            payload = json.loads(completed.stdout)
        except ValueError:
            payload = {"ok": False, "error": f"SSH_EXIT:{completed.returncode}"}
        nodes[str(node)] = payload
    return nodes



def audit_p95_runtime_inputs(registry):
    runtime = json.loads((PROJECT_ROOT/'formal_evaluation/config/baseline_runtime_registry_dsw.json').read_text())
    wanted = ['egofound3r','pad_hand','s2contact','contactopt']
    entries = []
    for m in wanted:
        spec = runtime['methods'][m]
        items = list(runtime.get('shared_required_paths',[])) + list(spec.get('required_paths',[]))
        for d, values in spec.get('dataset_required_paths',{}).items():
            items.extend(dict(v, dataset=d) for v in values)
        for key in ('source_root','python','conda_executable'):
            if spec.get(key):items.append({'role':key,'path':spec[key],'state':'present'})
        for item in items:
            entries.append(dict(item,method=m))
    script = r'''import sys,json,subprocess
from pathlib import Path
entries=json.loads(sys.argv[1]);results=[]
for item in entries:
    p=Path(item['path'])
    try:
        stat=p.stat();size_match='bytes' not in item or stat.st_size==item['bytes']
        results.append(dict(item,exists=True,actual_bytes=stat.st_size,size_matches=size_match))
    except OSError as e:results.append(dict(item,exists=False,error=str(e)))
failed=[v for v in results if v.get('state','present')=='present' and (not v['exists'] or not v.get('size_matches',True))]
code=[]
for root,expected in [('/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R_pair_new_3e533881_20260907','3e533881af4cd98642fc4bf7e06e7212e723bded'),('/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R_hand_depth_scale_2b9c180_20260908','2b9c1806149398d3c2df0e36487ef20dd14703bc'),('/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R-baselines_f8332ee_20260910_clean_v2','f8332ee5b86222c8fe4b6beabfc56c5d692bda87')]:
    head=subprocess.run(['git','-C',root,'rev-parse','HEAD'],capture_output=True,text=True,timeout=15)
    state=subprocess.run(['git','-C',root,'status','--porcelain'],capture_output=True,text=True,timeout=15)
    code.append({'root':root,'expected':expected,'head':head.stdout.strip(),'clean':state.returncode==0 and not state.stdout.strip(),'ok':head.returncode==0 and head.stdout.strip()==expected and state.returncode==0 and not state.stdout.strip()})
print(json.dumps({'node':5000,'ok':not failed and all(v['ok'] for v in code),'checked':len(results),'failures':failed,'paths':results,'code':code}))
'''
    remote=' '.join(('python3','-c',shlex.quote(script),shlex.quote(json.dumps(entries))))
    result=_ssh(registry['run_template'],5000,remote,timeout=90)
    if result.returncode:raise TaskError('RUNTIME_AUDIT_FAILED:'+result.stderr[-500:])
    return json.loads(result.stdout)


def audit_v7_corrected_inputs(registry):
    """Read-only loader audit for the three user-specified roots on node 5000."""
    script = r'''import sys, os, json, hashlib, time, faulthandler
faulthandler.dump_traceback_later(45, exit=True)
started = time.monotonic()
def stage(name):
    print(json.dumps({'stage':name,'elapsed_seconds':round(time.monotonic()-started,3)}),flush=True)
stage('python_started')
from pathlib import Path
sys.dont_write_bytecode = True
worktree = Path('/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R-baselines_f8332ee_20260910_clean_v2')
sys.path.insert(0, str(worktree))
sys.path.insert(0, str(worktree / 'formal_evaluation/vendor/egofound3r_dataloader'))
manifest = Path('/mnt/cpfs/sjc/eval_artifacts/visibility_v7_derived_manifest_20260910_v2/windows.jsonl')
stage('manifest_read_start')
raw = manifest.read_bytes()
stage('manifest_read_done')
assert hashlib.sha256(raw).hexdigest() == '23f36cbd82f38ee4cd729f0295aa4a607b5ec3ea6cffed316ec333e75764ea0c', 'MANIFEST_SHA_MISMATCH'
rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
stage('loader_import_start')
from egohandmetric_prompt.data import datasets
from egohandmetric_prompt.data.stages import build_named_frame_dataset
from formal_evaluation.datasets.egofound3r_gt import source_indices_for_window
stage('loader_import_done')
# Preserve the official read-only cache validation; never force a full rebuild.
def memory_index(cache_path, source_paths, build_index, **kwargs):
    stage('index_cache_read_start:' + str(cache_path))
    signatures = [datasets._light_index_source_signature(path) for path in source_paths]
    cached = datasets._read_light_index_cache(cache_path, signatures, **kwargs)
    if cached is None:
        raise RuntimeError('VALID_INDEX_CACHE_UNAVAILABLE: ' + str(cache_path))
    stage('index_cache_read_done:' + str(cache_path))
    return cached
datasets._load_or_build_light_index_cache = memory_index
def readonly(event, args):
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise PermissionError('READ_ONLY_AUDIT_WRITE_BLOCKED: ' + str(args[0]))
    if event in ('os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.system', 'subprocess.Popen'):
        raise PermissionError('READ_ONLY_AUDIT_OPERATION_BLOCKED: ' + event)
sys.addaudithook(readonly)
result = {}
for name, loader, root, count in [
 ('hot3d', 'hot3d_aria', '/mnt/workspace/sjc/DATA/mnt-1/HOT3D/hot3d/hot3d/dataset', 400),
 ('arctic', 'egoforce_arctic', '/mnt/workspace/sjc/DATA/mnt-1/EgoForce/ARCTIC', 434),
 ('oakink_v2', 'oakink_v2', '/mnt/workspace/sjc/DATA/mnt-1/OakInk-v2', 400)]:
    record = {'root': root, 'expected_windows': count}
    result[name] = record
    try:
        selected = [r for r in rows if r['dataset'] == name]
        assert len(selected) == count
        record['required_sequences'] = len(set(r['sequence_id'] for r in selected))
        if name == 'arctic':
            h5 = Path(root) / 'cam0_hand_arm_annotations_v4.h5'
            record['h5_path'] = str(h5)
            with h5.open('rb') as f: record['h5_signature_valid'] = f.read(8) == b'\x89HDF\r\n\x1a\n'
            assert record['h5_signature_valid']
            record['ok'] = True
            record['scope'] = 'exact_h5_readability_only'
            continue
        stage('dataset_build_start:' + name)
        parent = build_named_frame_dataset(loader, root_override=root, split='all', load_rgb=True, load_depth=True)
        stage('dataset_build_done:' + name)
        record['missing_sequences'] = sorted(set(r['sequence_id'] for r in selected) - set(parent.sequence_to_indices))
        if name == 'oakink_v2':
            record['ok'] = not record['missing_sequences']
            record['scope'] = 'all_v7_sequence_identities_only'
            record['verified_sequences'] = record['required_sequences'] - len(record['missing_sequences'])
            continue
        if name == 'hot3d': record['P0010_1c9fe708_resolvable'] = 'P0010_1c9fe708' in parent.sequence_to_indices
        errors = []
        valid = 0
        for row in selected:
            try:
                source_indices_for_window(parent, row['sequence_id'], row['frame_ids'])
                valid += 1
            except Exception as e:
                errors.append({'window_id': row.get('window_id'), 'error': str(e)})
        record.update(verified_windows=valid, errors=errors[:10], error_count=len(errors), ok=valid == count)
    except Exception as e:
        record.update(ok=False, error=type(e).__name__ + ': ' + str(e))
faulthandler.cancel_dump_traceback_later()
print(json.dumps({'node':5000, 'manifest':str(manifest), 'datasets':result, 'ok':all(r.get('ok') for r in result.values())}))
'''
    remote = ' '.join((EGOFOUND3R_MODEL_PYTHON, '-B', '-c', shlex.quote(script)))
    try:
        completed = _ssh(registry['run_template'], 5000, remote, timeout=65)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b''
        if isinstance(output, bytes):
            output = output.decode(errors='replace')
        return {'ok':False, 'node':5000, 'error':'V7_INPUT_AUDIT_TIMEOUT', 'timeout_seconds':65, 'partial_output':output[-2000:]}
    if completed.returncode:
        return {'ok':False, 'error':f'REMOTE_AUDIT_EXIT:{completed.returncode}', 'detail':completed.stderr[-4000:], 'stages':completed.stdout[-4000:]}
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {'ok':False, 'error':'INVALID_AUDIT_RESPONSE', 'detail':completed.stdout[-2000:]}

def audit_hot3d_contact_inputs(registry: dict[str, Any]) -> dict[str, Any]:
    run = registry.get("run_template", {})
    if not isinstance(run.get("ssh"), dict):
        raise TaskError("REMOTE_PROBE_NOT_CONFIGURED")
    remote = " ".join((
        "/mnt/workspace/sjc/envs/egofound3r/bin/python", "-c",
        shlex.quote(REMOTE_HOT3D_CONTACT_INPUT_AUDIT),
    ))
    nodes: dict[str, Any] = {}
    for node in (5001, 5000, 6001):
        completed = _ssh(run, node, remote, timeout=300)
        if completed.returncode:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no stderr"
            nodes[str(node)] = {"status": "unavailable", "error": f"SSH_EXIT:{completed.returncode}",
                                "detail": detail[-500:]}
            continue
        try:
            payload = json.loads(completed.stdout)
        except ValueError:
            payload = {"ok": False, "errors": ["INVALID_AUDIT_RESPONSE"]}
        return {"observed_node": node, **payload, "nodes": nodes}
    return {"ok": False, "nodes": nodes, "errors": ["NO_REACHABLE_NODE"]}


def _choose_launch_target(run: dict[str, Any], launch: dict[str, Any]) -> tuple[int, int | str]:
    """Validate the exact registered worktree and select an idle physical GPU."""
    allowed = [int(item) for item in launch["node_candidates"]]
    if launch.get("resource") == "visualization":
        endpoint_5001 = (allowed == [5001]
                         and run.get("identity", {}).get("ego_completion_commit")
                         == "8fc061a615895bd3b5a556f7387bae306e32d9db")
        if allowed != [5000] and not endpoint_5001:
            raise TaskError("VISUALIZATION_REQUIRES_EXACT_REGISTERED_NODE")
        script = r'''import gzip,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
root,runtime,digest,plan_digest,plan_relative,manifest_relative,spec_relative,expected_count=sys.argv[1:]
root=Path(root);runtime=Path(runtime)
manifest=runtime/manifest_relative
spec=runtime/spec_relative
plan=runtime/plan_relative if plan_relative else None
raw=gzip.decompress(manifest.read_bytes()) if manifest.is_file() else b''
good=(root.is_dir() and manifest.is_file() and spec.is_file()
      and hashlib.sha256(raw).hexdigest()==digest
      and len([line for line in raw.splitlines() if line.strip()])==int(expected_count)
      and json.loads(spec.read_text())['manifest_sha256']==digest
      and (not plan_digest or (plan and plan.is_file() and hashlib.sha256(plan.read_bytes()).hexdigest()==plan_digest))
      and shutil.disk_usage(root).free>30_000_000_000
      and all(importlib.util.find_spec(m) for m in ('numpy','matplotlib','PIL'))
      and all(shutil.which(c) for c in ('ffmpeg','ffprobe'))
      and not (root/'COMPLETE').exists())
print(json.dumps({'ok':bool(good),'error':None if good else 'VISUALIZATION_PREFLIGHT_FAILED'}))'''
        default_relative = "visualization/batch_10s_177_p95_wmpjpe_20260912"
        remote = str(launch.get("preflight_python", "python3")) + " -c " + shlex.quote(script) + " " + shlex.join([
            run["output_root"], launch["runtime_root"], launch["manifest_sha256"],
            launch.get("plan_sha256", ""), launch.get("plan_relative", default_relative + "/parallel8_handoff_20260913/plan.json"),
            launch.get("manifest_relative", default_relative + "/selected_manifest.jsonl.gz"),
            launch.get("source_alignment_relative", default_relative + "/source_alignment_5000.json"),
            str(launch.get("expected_segments", 177))])
        completed = _ssh(run, allowed[0], remote, timeout=90)
        try:
            payload = json.loads(completed.stdout)
        except ValueError:
            payload = {}
        if completed.returncode or not payload.get("ok"):
            raise TaskError("VISUALIZATION_PREFLIGHT_FAILED:" + str(payload.get("error", completed.stderr[-300:])))
        return allowed[0], -1
    if launch.get("resource") == "coverage":
        if len(allowed) != 1 or allowed[0] not in (5000, 5001):
            raise TaskError("COVERAGE_REQUIRES_ONE_CURRENT_NODE")
        script = r'''import hashlib,json,os,shutil,sys
from pathlib import Path
root,manifest,digest,spec,worker=sys.argv[1:]
root=Path(root)
checks={"output_writable":root.is_dir() and os.access(root,os.W_OK),
        "manifest_hash":Path(manifest).is_file() and hashlib.sha256(Path(manifest).read_bytes()).hexdigest()==digest,
        "source_spec":Path(spec).is_file(),"worker":Path(worker).is_file(),
        "new_output":not (root/'summary.json').exists(),
        "free_space":root.is_dir() and shutil.disk_usage(root).free>1_000_000_000}
print(json.dumps({'ok':all(checks.values()),'checks':checks}))'''
        remote = "python3 -c " + shlex.quote(script) + " " + shlex.join([
            run["output_root"], launch["manifest_path"], launch["manifest_sha256"],
            launch["source_spec_path"], launch["worker_path"]])
        completed = _ssh(run, allowed[0], remote, timeout=90)
        try:
            payload = json.loads(completed.stdout)
        except ValueError:
            payload = {}
        if completed.returncode or not payload.get("ok"):
            raise TaskError("COVERAGE_PREFLIGHT_FAILED:" + json.dumps(payload or {"stderr": completed.stderr[-300:]}))
        return allowed[0], -1
    if launch.get("resource") == "io":
        script = r'''import json, sys
from pathlib import Path
destination = Path(sys.argv[1])
direction = sys.argv[2]
allowed = Path("/mnt/cpfs/sjc/eval_artifacts" if direction in {"oss-to-cpfs", "direct-oss-metrics"} else "/mnt/oss/pre-train/ego/eval_artifacts")
source = Path("/mnt/oss/pre-train/ego/eval_artifacts")
try:
    destination.relative_to(allowed)
    allowed.stat()
    if direction in {"oss-to-cpfs", "direct-oss-metrics"}:
        next(source.iterdir(), None)
except (OSError, ValueError) as error:
    print(json.dumps({"ok": False, "error": type(error).__name__})); raise SystemExit(0)
print(json.dumps({"ok": True}))'''
        direction = str(run.get("identity", {}).get("direction", "target"))
        if run.get("task_type") == "metrics-recompute" and run.get("identity", {}).get("source_mode") == "direct_oss":
            direction = "direct-oss-metrics"
        for node in [int(value) for value in launch["node_candidates"]]:
            completed = _ssh(run, node, " ".join(("python3", "-c", shlex.quote(script),
                                                   shlex.quote(str(run["output_root"])),
                                                   shlex.quote(direction))))
            try:
                payload = json.loads(completed.stdout)
            except ValueError:
                payload = {}
            if completed.returncode == 0 and payload.get("ok"):
                return node, -1
        raise TaskError(f"NO_HEALTHY_IO_NODE:{run['run_id']}")
    worktree = str(launch["worktree"])
    method = str(launch["method"])
    if launch.get("resource") == "cpu":
        script = r'''import json, subprocess, sys, time
from pathlib import Path
worktree, paths_text, clone_url, clone_ref, worktree_commit, bundle, base_commit, source = sys.argv[1:]
paths = json.loads(paths_text)
target = Path(worktree)
if bundle and base_commit and source:
    try:
        target.relative_to("/mnt/workspace/sjc/DATA/runtime_worktrees")
    except ValueError:
        print(json.dumps({"ok": False, "error": "INVALID_WORKTREE_TARGET"})); raise SystemExit(0)
    if not target.exists():
        known = subprocess.run(["git", "-C", source, "cat-file", "-e", base_commit + "^{commit}"],
                               text=True, capture_output=True)
        if known.returncode:
            print(json.dumps({"ok": False, "error": "WORKTREE_BASE_COMMIT_MISSING"})); raise SystemExit(0)
        target.parent.mkdir(parents=True, exist_ok=True)
        created = subprocess.run(["git", "-C", source, "worktree", "add", "--detach", worktree, base_commit],
                                 text=True, capture_output=True)
        if created.returncode:
            detail = " | ".join((created.stdout + created.stderr).splitlines()[-8:])
            print(json.dumps({"ok": False, "error": "WORKTREE_CREATE_FAILED:" + detail})); raise SystemExit(0)
    current = subprocess.run(["git", "-C", worktree, "rev-parse", "HEAD"], text=True, capture_output=True)
    if current.returncode or current.stdout.strip() != worktree_commit:
        existing_clean = subprocess.run(["git", "-C", worktree, "status", "--porcelain"],
                                        text=True, capture_output=True)
        if existing_clean.returncode or existing_clean.stdout.strip():
            print(json.dumps({"ok": False, "error": "WORKTREE_BUNDLE_TARGET_NOT_CLEAN"})); raise SystemExit(0)
        fetched = subprocess.run(["git", "-C", worktree, "fetch", bundle, "formal-hand-fix"],
                                 text=True, capture_output=True)
        if fetched.returncode:
            detail = " | ".join((fetched.stdout + fetched.stderr).splitlines()[-8:])
            print(json.dumps({"ok": False, "error": "WORKTREE_BUNDLE_FETCH_FAILED:" + detail})); raise SystemExit(0)
        checked_out = subprocess.run(["git", "-C", worktree, "checkout", "--detach", worktree_commit],
                                     text=True, capture_output=True)
        if checked_out.returncode:
            detail = " | ".join((checked_out.stdout + checked_out.stderr).splitlines()[-8:])
            print(json.dumps({"ok": False, "error": "WORKTREE_CHECKOUT_FAILED:" + detail})); raise SystemExit(0)
if target.exists() and clone_url and worktree_commit:
    valid_repo = subprocess.run(["git", "-C", worktree, "rev-parse", "--is-inside-work-tree"],
                                text=True, capture_output=True)
    if valid_repo.returncode:
        backup = target.with_name(target.name + f".failed_clone_{int(time.time())}")
        if backup.exists():
            print(json.dumps({"ok": False, "error": "FAILED_CLONE_BACKUP_EXISTS"})); raise SystemExit(0)
        target.rename(backup)
if not target.exists() and clone_url and worktree_commit:
    try:
        target.relative_to("/mnt/workspace/sjc/DATA/runtime_worktrees")
    except ValueError:
        print(json.dumps({"ok": False, "error": "INVALID_WORKTREE_TARGET"})); raise SystemExit(0)
    target.parent.mkdir(parents=True, exist_ok=True)
    clone_command = ["git", "clone", "--no-local"]
    if clone_ref:
        clone_command.extend(["--single-branch", "--branch", clone_ref])
    cloned = subprocess.run([*clone_command, clone_url, worktree], text=True, capture_output=True)
    if cloned.returncode:
        detail = " | ".join((cloned.stdout + cloned.stderr).splitlines()[-8:])
        print(json.dumps({"ok": False, "error": "WORKTREE_CLONE_FAILED:" + detail})); raise SystemExit(0)
    checked_out = subprocess.run(["git", "-C", worktree, "checkout", "--detach", worktree_commit],
                                 text=True, capture_output=True)
    if checked_out.returncode:
        detail = " | ".join((checked_out.stdout + checked_out.stderr).splitlines()[-8:])
        print(json.dumps({"ok": False, "error": "WORKTREE_CHECKOUT_FAILED:" + detail})); raise SystemExit(0)
clean = subprocess.run(["git", "-C", worktree, "status", "--porcelain"], text=True, capture_output=True)
if clean.returncode or clean.stdout.strip():
    detail = " | ".join((clean.stdout + clean.stderr).splitlines()[:12])
    print(json.dumps({"ok": False, "error": "WORKTREE_NOT_CLEAN:" + detail})); raise SystemExit(0)
if worktree_commit:
    head = subprocess.run(["git", "-C", worktree, "rev-parse", "HEAD"], text=True, capture_output=True)
    if head.returncode or head.stdout.strip() != worktree_commit:
        print(json.dumps({"ok": False, "error": "WORKTREE_COMMIT_MISMATCH:" + head.stdout.strip()})); raise SystemExit(0)
try:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            next(path.iterdir(), None)
        else:
            with path.open("rb") as handle:
                handle.read(1)
except OSError as error:
    print(json.dumps({"ok": False, "error": f"INPUT_UNREADABLE:{type(error).__name__}:{raw}"})); raise SystemExit(0)
print(json.dumps({"ok": True}))'''
        failures: dict[int, str] = {}
        for node in allowed:
            completed = _ssh(run, node, " ".join((
                "python3", "-c", shlex.quote(script), shlex.quote(worktree),
                shlex.quote(json.dumps(launch.get("preflight_paths", []))),
                shlex.quote(str(launch.get("worktree_clone_url", ""))),
                shlex.quote(str(launch.get("worktree_clone_ref", ""))),
                shlex.quote(str(launch.get("worktree_commit", ""))),
                shlex.quote(str(launch.get("worktree_bundle", ""))),
                shlex.quote(str(launch.get("worktree_base_commit", ""))),
                shlex.quote(str(launch.get("worktree_source", ""))),
            )), timeout=600 if launch.get("worktree_clone_url") else 120)
            if completed.returncode:
                failures[node] = f"SSH_EXIT:{completed.returncode}"
                continue
            try:
                payload = json.loads(completed.stdout)
            except ValueError:
                payload = {}
            if completed.returncode == 0 and payload.get("ok"):
                return node, -1
            failures[node] = str(payload.get("error") or "WORKTREE_NOT_CLEAN")
        raise TaskError(f"NO_VALID_CPU_NODE:{run['run_id']}:{json.dumps(failures, sort_keys=True)}")
    script = r'''import json, os, subprocess, sys
from pathlib import Path
worktree, method, dataset, requested_text, worktree_source, worktree_commit, worktree_fetch_ref, worktree_bundle, worktree_base_commit, runtime_registry, validate_runtime, preflight_text, runtime_python = sys.argv[1:]
requested = {int(value) for value in requested_text.split(",") if value}
if worktree_source and worktree_commit and not worktree_bundle:
    source = Path(worktree_source)
    if not source.is_dir():
        print(json.dumps({"ok": False, "error": "WORKTREE_SOURCE_MISSING"})); raise SystemExit(0)
    source_clean = subprocess.run(["git", "-C", worktree_source, "status", "--porcelain"], text=True, capture_output=True)
    if source_clean.returncode or source_clean.stdout.strip():
        print(json.dumps({"ok": False, "error": "WORKTREE_SOURCE_NOT_CLEAN"})); raise SystemExit(0)
    known = subprocess.run(["git", "-C", worktree_source, "cat-file", "-e", worktree_commit + "^{commit}"], text=True, capture_output=True)
    if known.returncode and worktree_fetch_ref:
        fetched = subprocess.run(["git", "-C", worktree_source, "fetch", "origin", worktree_fetch_ref], text=True, capture_output=True)
        if fetched.returncode:
            detail = " | ".join((fetched.stdout + fetched.stderr).splitlines()[-8:])
            print(json.dumps({"ok": False, "error": "WORKTREE_FETCH_FAILED:" + detail})); raise SystemExit(0)
        known = subprocess.run(["git", "-C", worktree_source, "cat-file", "-e", worktree_commit + "^{commit}"], text=True, capture_output=True)
    if known.returncode:
        print(json.dumps({"ok": False, "error": "WORKTREE_COMMIT_MISSING"})); raise SystemExit(0)
if worktree_bundle and worktree_source and worktree_commit:
    if not worktree.startswith("/mnt/workspace/sjc/DATA/runtime_worktrees/"):
        print(json.dumps({"ok": False, "error": "INVALID_WORKTREE_TARGET"})); raise SystemExit(0)
    if not Path(worktree).exists():
        Path(worktree).parent.mkdir(parents=True, exist_ok=True)
        if worktree_base_commit:
            source = Path(worktree_source)
            if not source.is_dir():
                print(json.dumps({"ok": False, "error": "WORKTREE_SOURCE_MISSING"})); raise SystemExit(0)
            known_base = subprocess.run(["git", "-C", worktree_source, "cat-file", "-e", worktree_base_commit + "^{commit}"], text=True, capture_output=True)
            if known_base.returncode:
                print(json.dumps({"ok": False, "error": "WORKTREE_BASE_COMMIT_MISSING"})); raise SystemExit(0)
            created = subprocess.run(["git", "-C", worktree_source, "worktree", "add", "--detach", worktree, worktree_base_commit], text=True, capture_output=True)
            if created.returncode:
                detail = " | ".join((created.stdout + created.stderr).splitlines()[-8:])
                print(json.dumps({"ok": False, "error": "WORKTREE_CREATE_FAILED:" + detail})); raise SystemExit(0)
        else:
            cloned = subprocess.run(["git", "clone", "--no-local", worktree_bundle, worktree], text=True, capture_output=True)
            if cloned.returncode:
                detail = " | ".join((cloned.stdout + cloned.stderr).splitlines()[-8:])
                print(json.dumps({"ok": False, "error": "WORKTREE_CLONE_FAILED:" + detail})); raise SystemExit(0)
    current = subprocess.run(["git", "-C", worktree, "rev-parse", "HEAD"], text=True, capture_output=True)
    if current.returncode or current.stdout.strip() != worktree_commit:
        existing_clean = subprocess.run(["git", "-C", worktree, "status", "--porcelain"], text=True, capture_output=True)
        if existing_clean.returncode or existing_clean.stdout.strip():
            print(json.dumps({"ok": False, "error": "WORKTREE_BUNDLE_TARGET_NOT_CLEAN"})); raise SystemExit(0)
        fetched = subprocess.run(["git", "-C", worktree, "fetch", worktree_bundle, "formal-hand-fix"], text=True, capture_output=True)
        if fetched.returncode:
            detail = " | ".join((fetched.stdout + fetched.stderr).splitlines()[-8:])
            print(json.dumps({"ok": False, "error": "WORKTREE_BUNDLE_FETCH_FAILED:" + detail})); raise SystemExit(0)
        checked_out = subprocess.run(["git", "-C", worktree, "checkout", "--detach", worktree_commit], text=True, capture_output=True)
        if checked_out.returncode:
            detail = " | ".join((checked_out.stdout + checked_out.stderr).splitlines()[-8:])
            print(json.dumps({"ok": False, "error": "WORKTREE_CHECKOUT_FAILED:" + detail})); raise SystemExit(0)
elif not Path(worktree).exists() and worktree_source and worktree_commit:
    if not worktree.startswith("/mnt/workspace/sjc/DATA/runtime_worktrees/"):
        print(json.dumps({"ok": False, "error": "INVALID_WORKTREE_TARGET"})); raise SystemExit(0)
    Path(worktree).parent.mkdir(parents=True, exist_ok=True)
    created = subprocess.run(["git", "-C", worktree_source, "worktree", "add", "--detach", worktree, worktree_commit], text=True, capture_output=True)
    if created.returncode:
        detail = " | ".join((created.stdout + created.stderr).splitlines()[-8:])
        print(json.dumps({"ok": False, "error": "WORKTREE_CREATE_FAILED:" + detail})); raise SystemExit(0)
clean = subprocess.run(["git", "-C", worktree, "status", "--porcelain"], text=True, capture_output=True)
if clean.returncode or clean.stdout.strip():
    detail = " | ".join((clean.stdout + clean.stderr).splitlines()[:12])
    print(json.dumps({"ok": False, "error": "WORKTREE_NOT_CLEAN:" + detail})); raise SystemExit(0)
if worktree_commit:
    head = subprocess.run(["git", "-C", worktree, "rev-parse", "HEAD"], text=True, capture_output=True)
    if head.returncode or head.stdout.strip() != worktree_commit:
        print(json.dumps({"ok": False, "error": "WORKTREE_COMMIT_MISMATCH:" + head.stdout.strip()})); raise SystemExit(0)
failed_path = runtime_python
try:
    for raw in json.loads(preflight_text):
        failed_path = raw
        path = Path(raw)
        if path.is_dir():
            next(path.iterdir(), None)
        else:
            with path.open("rb") as handle:
                handle.read(1)
    if runtime_python and (not Path(runtime_python).is_file() or not os.access(runtime_python, os.X_OK)):
        raise OSError("runtime python is not executable")
except (OSError, ValueError) as error:
    print(json.dumps({"ok": False, "error": f"INPUT_UNREADABLE:{type(error).__name__}:{failed_path}"})); raise SystemExit(0)
if validate_runtime == "1":
    validation = [runtime_python, "formal_evaluation/validate_runtime_registry.py"]
    if runtime_registry:
        validation.extend(["--registry", runtime_registry])
    validation.extend(["--method", method, "--dataset", dataset, "--strict"])
    valid = subprocess.run(validation, cwd=worktree, text=True, capture_output=True)
    if valid.returncode:
        detail = " | ".join((valid.stdout + valid.stderr).splitlines()[-8:])
        print(json.dumps({"ok": False, "error": "RUNTIME_REGISTRY_INVALID:" + detail})); raise SystemExit(0)
gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True, capture_output=True)
apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"], text=True, capture_output=True)
if gpu.returncode or apps.returncode:
    print(json.dumps({"ok": False, "error": "GPU_QUERY_FAILED"})); raise SystemExit(0)
occupied = {row.split(",", 1)[0].strip() for row in apps.stdout.splitlines() if row.strip()}
idle = []
for row in gpu.stdout.splitlines():
    index, uuid, memory = [value.strip() for value in row.split(",")]
    if int(memory) <= 10 and uuid not in occupied and (not requested or int(index) in requested):
        idle.append(int(index))
print(json.dumps({"ok": True, "idle_gpus": idle}))'''
    failures: dict[int, str] = {}
    required_gpu_count = int(launch.get("required_gpu_count", 1))
    if required_gpu_count < 1:
        raise TaskError(f"INVALID_REQUIRED_GPU_COUNT:{required_gpu_count}")
    for node in allowed:
        candidates_by_node = launch.get("gpu_candidates_by_node", {})
        node_candidates = candidates_by_node.get(str(node), launch.get("gpu_candidates", []))
        requested = ",".join(str(value) for value in node_candidates)
        remote = " ".join(("python3", "-c", shlex.quote(script), shlex.quote(worktree), shlex.quote(method),
                           shlex.quote(str(run["dataset"])), shlex.quote(requested),
                           shlex.quote(str(launch.get("worktree_source", ""))),
                           shlex.quote(str(launch.get("worktree_commit", ""))),
                           shlex.quote(str(launch.get("worktree_fetch_ref", ""))),
                           shlex.quote(str(launch.get("worktree_bundle", ""))),
                           shlex.quote(str(launch.get("worktree_base_commit", ""))),
                           shlex.quote(str(launch.get("runtime_registry", ""))),
                           "1" if launch.get("validate_runtime_registry", True) else "0",
                           shlex.quote(json.dumps(launch.get("preflight_paths", []))),
                           shlex.quote(str(launch.get("values", {}).get("python", "")))))
        completed = _ssh(run, node, remote, timeout=600 if launch.get("worktree_source") else 60)
        if completed.returncode:
            failures[node] = f"SSH_EXIT:{completed.returncode}"
            continue
        try:
            payload = json.loads(completed.stdout)
        except ValueError:
            failures[node] = "INVALID_PREFLIGHT_RESPONSE"
            continue
        idle_gpus = payload.get("idle_gpus", []) if payload.get("ok") else []
        if len(idle_gpus) >= required_gpu_count:
            selected = [int(value) for value in idle_gpus[:required_gpu_count]]
            return node, selected[0] if required_gpu_count == 1 else ",".join(map(str, selected))
        if payload.get("ok") and launch.get("wait_for_idle_gpu"):
            return node, -1
        failures[node] = str(payload.get("error") or "NO_IDLE_GPU")
    raise TaskError(f"NO_VALID_IDLE_GPU:{run['run_id']}:{json.dumps(failures, sort_keys=True)}")


def _deploy_launch_support(run: dict[str, Any], node: int, launch: dict[str, Any]) -> None:
    """Copy only listed local support scripts to this run's unique result runtime."""
    instance = run.get("_current_instance")
    if instance and node != instance["port"]:
        raise TaskError(f"INSTANCE_PORT_MISMATCH:{node}:{instance['port']}")
    runtime = _node_text(run, node, launch["runtime_root"])
    support = [str(item) for item in launch.get("support_files", [])]
    if not support:
        return
    directories = {runtime + "/formal_evaluation"}
    directories.update((runtime + "/" + relative).rsplit("/", 1)[0] for relative in support)
    setup = _ssh(run, node, "mkdir -p " + " ".join(shlex.quote(path) for path in sorted(directories)))
    if setup.returncode:
        raise TaskError(f"RUNTIME_DEPLOY_MKDIR_FAILED:{node}")
    for relative in support:
        source = PROJECT_ROOT / relative
        if not source.is_file() or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise TaskError(f"INVALID_SUPPORT_FILE:{relative}")
        destination = f"root@{run['ssh']['host']}:{runtime}/{relative}"
        try:
            completed = subprocess.run([
                "scp", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15",
                "-i", os.path.expanduser(str(run["ssh"]["key"])), "-P", str(node), str(source), destination,
            ], text=True, capture_output=True, timeout=300, check=False)
        except subprocess.TimeoutExpired as error:
            raise TaskError(f"RUNTIME_DEPLOY_COPY_TIMEOUT:{relative}:{node}") from error
        if completed.returncode:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no stderr"
            raise TaskError(f"RUNTIME_DEPLOY_COPY_FAILED:{relative}:{node}:{detail[-500:]}")


def launch_registered_run(registry: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Start a pre-registered launch spec; no paths or commands are caller supplied."""
    run = merged_run(registry, run_id)
    launch = run.get("launch")
    if not isinstance(launch, dict):
        raise TaskError(f"LAUNCH_NOT_CONFIGURED:{run_id}")
    section, jobs = run_jobs(registry, run)
    if len(jobs) != 1:
        raise TaskError(f"LAUNCH_REQUIRES_ONE_EXACT_JOB:{run_id}")
    state_path = PROJECT_ROOT / run["state_file"]
    state = read_json(state_path)
    label, key = jobs[0]
    job = state.get(section, {}).get(key)
    if not isinstance(job, dict) or job.get("status") not in {"pending", "queue_exited_needs_audit", "blocked"}:
        raise TaskError(f"RUN_NOT_STARTABLE:{run_id}:{label}")
    if launch.get("deploy_before_preflight"):
        for candidate in launch["node_candidates"]:
            _deploy_launch_support(run, int(candidate), launch)
    # Visibility initialization can run on CPU before CUDA appears in nvidia-smi.
    # Reserve GPUs held by exact registered visibility handles during that phase.
    if run.get("identity", {}).get("pipeline") == "visibility-gt-cache-v3":
        reserved = {}
        for other_id, other in registry["runs"].items():
            if other_id == run_id or other.get("identity", {}).get("pipeline") != "visibility-gt-cache-v3":
                continue
            other_state = read_json(PROJECT_ROOT / other["state_file"])
            for other_job in other_state.get(other.get("state_section", "jobs"), {}).values():
                if other_job.get("status") in {"running", "paused", "pausing", "resuming"} and other_job.get("node") and other_job.get("gpu") is not None:
                    reserved.setdefault(str(other_job["node"]), set()).add(int(other_job["gpu"]))
        launch = dict(launch)
        launch["gpu_candidates_by_node"] = {
            str(node): [gpu for gpu in launch.get("gpu_candidates", []) if gpu not in reserved.get(str(node), set())]
            for node in launch["node_candidates"]
        }
        launch["node_candidates"] = [node for node in launch["node_candidates"] if launch["gpu_candidates_by_node"][str(node)]]
    # Coverage audits are CPU-only and their preflight already selected the
    # exact current node; do not pass them through the GPU idle probe.
    if launch.get("resource") == "coverage":
        node, gpu = int(launch["node_candidates"][0]), -1
    else:
        node, gpu = _choose_launch_target(run, launch)
    if not launch.get("deploy_before_preflight"):
        _deploy_launch_support(run, node, launch)
    values = {"node": node, "gpu": gpu, **{str(name): str(value) for name, value in launch.get("values", {}).items()}}
    if launch.get("worker_args"):
        raw_arguments = [str(value).format(**values) for value in launch["worker_args"]]
        arguments = []
        iterator = iter(raw_arguments)
        for value in iterator:
            if value == "--runner-arg":
                arguments.append("--runner-arg=" + next(iterator))
            else:
                arguments.append(value)
        command = "TASKCTL_GPU=" + shlex.quote(str(gpu)) + " " + shlex.join([
            str(values["python"]), str(values["runtime"]) + "/formal_evaluation/registered_backfill_worker.py", *arguments,
        ])
    else:
        command = str(launch["command"]).format(**values)
    command = _node_text(run, node, command)
    handle = str(launch["handle_path"])
    log = str(launch["log_path"])
    if run.get("task_type") == "artifact-migration" or job.get("status") == "queue_exited_needs_audit":
        attempt = int(job.get("launch_attempt", 0)) + 1
        handle_path, log_path = Path(handle), Path(log)
        handle = str(handle_path.with_name(f"{handle_path.stem}.attempt{attempt}{handle_path.suffix}"))
        log = str(log_path.with_name(f"{log_path.stem}.attempt{attempt}{log_path.suffix}"))
        job["launch_attempt"] = attempt
        atomic_json(state_path, state)
    controller = str(launch["controller_path"])
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    digest = hashlib.sha256(command.encode("utf-8")).hexdigest()
    remote = " ".join((
        shlex.quote(str(run["remote_python"])), shlex.quote(controller), "launch",
        "--handle", shlex.quote(handle), "--log", shlex.quote(log),
        "--shell-b64", shlex.quote(encoded), "--command-sha256", shlex.quote(digest),
    ))
    completed = _ssh(run, node, remote)
    try:
        payload = json.loads(completed.stdout)
    except ValueError as error:
        raise TaskError(f"REMOTE_LAUNCH_INVALID_RESPONSE:{node}") from error
    if completed.returncode or not payload.get("ok"):
        launch_error = str(payload.get("error", completed.returncode))
        if launch_error != f"HANDLE_ALREADY_LIVE:{handle}":
            raise TaskError(f"REMOTE_LAUNCH_FAILED:{node}:{launch_error}")
        status_command = " ".join((
            shlex.quote(str(run["remote_python"])), shlex.quote(controller), "status",
            "--handle", shlex.quote(handle),
        ))
        status_completed = _ssh(run, node, status_command)
        try:
            status_payload = json.loads(status_completed.stdout)
            live = status_payload["handles"][handle]
        except (ValueError, KeyError, TypeError) as error:
            raise TaskError(f"LIVE_HANDLE_RECOVERY_INVALID_RESPONSE:{node}") from error
        if (status_completed.returncode or not status_payload.get("ok")
                or live.get("status") != "running"
                or live.get("command_sha256") != digest):
            raise TaskError(f"LIVE_HANDLE_RECOVERY_MISMATCH:{node}:{live}")
        payload = {"pid": live["pid"], "pgid": live["pgid"]}
    job.update({
        "status": "running", "node": node,
        "pid": payload["pid"], "pgid": payload["pgid"], "handle_path": handle,
        "output_root": run["output_root"], "launched_at_epoch": int(time.time()),
        "command_sha256": digest,
    })
    job.pop("audit_log_tail", None)
    job.pop("audit_summary", None)
    if str(gpu) != "-1":
        job["gpu"] = gpu
    else:
        job.pop("gpu", None)
        job["resource_state"] = "waiting_for_idle_gpu"
    atomic_json(state_path, state)
    return {"run_id": run_id, "status": "running", "node": node,
            "gpu": None if str(gpu) == "-1" else gpu,
            "pid": payload["pid"], "handle_path": handle, "verified": True}


def emit(payload: dict[str, Any], ok: bool = True) -> None:
    print(json.dumps({"ok": ok, **payload}, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    default_instance_config = PROJECT_ROOT / ".auto_scheduler/current_dsw_instance.json"
    parser.add_argument("--instance-config", type=Path,
                        default=(Path(os.environ["TASKCTL_INSTANCE_CONFIG"])
                                 if os.environ.get("TASKCTL_INSTANCE_CONFIG") else
                                 default_instance_config if default_instance_config.is_file() else None),
                        help="local current-instance SSH and entrance configuration")
    subparsers = parser.add_subparsers(dest="action", required=True)
    register_parser = subparsers.add_parser("register", help="idempotently register before submission")
    gallery_parser = subparsers.add_parser("register-10s-gallery", help="register the frozen 5000 five-dataset gallery")
    gallery_parser.add_argument("--pilot", action="store_true", help="register the exact uncached OakInk-v2 pilot")
    gallery_parser.add_argument("--shard", type=int, choices=(2, 3, 4), help="register one disjoint parallel4 continuation shard")
    endpoint_gallery_parser = subparsers.add_parser(
        "register-10s-endpoint-gallery",
        help="register the frozen 5001 endpoint-renderable 104-segment gallery",
    )
    endpoint_gallery_parser.add_argument("--pilot", action="store_true", help="register the exact H2O pilot")
    endpoint_hawor_parser = subparsers.add_parser(
        "register-10s-endpoint-gallery-hawor-native",
        help="register a new 5000 full rerender with native-camera unmasked-SLAM HaWoR",
    )
    endpoint_hawor_parser.add_argument("--pilot", action="store_true", help="register the exact H2O pilot")
    subparsers.add_parser(
        "register-10s-endpoint-gallery-hawor-native-zoom-pilot",
        help="register the exact ARCTIC hand-focused fixed-crop HaWoR-native pilot",
    )
    subparsers.add_parser(
        "register-10s-endpoint-gallery-auxmethods-zoom-pilot",
        help="register the exact ARCTIC auxiliary-method near-hand pilot and panel gallery",
    )
    subparsers.add_parser(
        "register-10s-endpoint-gallery-auxmethods-zoom",
        help="register the full 104-segment v5 auxiliary-method gallery with conditional Dyn-HaMR rows",
    )
    subparsers.add_parser(
        "register-10s-pad-repair",
        help="register the exact PAD-Hand bimanual repair for the missing 40 endpoint segments",
    )
    subparsers.add_parser("register-10s-gallery-merge", help="register the parallel4 177-pair gallery assembler")
    handoff_parser = subparsers.add_parser("register-10s-gallery-handoff", help="register one frozen parallel8/16 continuation shard or assembler")
    handoff_parser.add_argument("--parallelism", type=int, choices=(8, 16), default=8)
    handoff_choice = handoff_parser.add_mutually_exclusive_group(required=True)
    handoff_choice.add_argument("--shard-number", type=int, choices=range(1, 17))
    handoff_choice.add_argument("--merge", action="store_true")
    register_parser.add_argument("--task-id", required=True)
    register_parser.add_argument("--dataset", required=True)
    register_parser.add_argument("--method-set", required=True)
    register_parser.add_argument("--methods", nargs="+")
    register_parser.add_argument("--phase", default="formal")
    register_parser.add_argument("--protocol", default="60f")
    register_parser.add_argument("--run-id")
    register_parser.add_argument("--task-type", default="evaluation")
    register_parser.add_argument("--scheduler-id", required=True)
    register_parser.add_argument("--state-file", required=True)
    register_parser.add_argument("--state-section", default="jobs")
    register_parser.add_argument("--job-key", action="append")
    register_parser.add_argument("--output-root", required=True)
    register_parser.add_argument("--method-root-output", action="store_true")
    register_parser.add_argument("--target-windows", type=int, required=True)
    register_parser.add_argument("--identity", action="append", default=[], metavar="KEY=VALUE")
    subparsers.add_parser(
        "register-egofound3r-final-smoke",
        help="register the fixed final Root Fusion V2 single-window smoke",
    )
    subparsers.add_parser(
        "register-pad-bimanual-p95-manifest",
        help="register the dependent corrected PAD-Hand P95 and frozen-177 manifest recompute",
    )
    pad_resume_parser = subparsers.add_parser(
        "handoff-pad-bimanual-resume",
        help="handoff one failed partial PAD-Hand lane to 5000 with verified append-only resume",
    )
    pad_resume_parser.add_argument("--run-id", required=True)
    pad_resume_parser.add_argument("--gpu", type=int, default=6)
    gallery_handoff_parser = subparsers.add_parser(
        "handoff-10s-endpoint-gallery",
        help="handoff one empty endpoint gallery after PAD wait timeout to 5000",
    )
    gallery_handoff_parser.add_argument("--run-id", required=True)
    resolve_parser = subparsers.add_parser("resolve", help="step 1: exact task -> run ID and paths")
    resolve_parser.add_argument("--task-id")
    resolve_parser.add_argument("--dataset")
    resolve_parser.add_argument("--method-set")
    resolve_parser.add_argument("--phase", default="formal")
    resolve_parser.add_argument("--protocol", default="60f")
    audit_parser = subparsers.add_parser(
        "audit-history",
        help="authorized read-only audit of H2O/TACO/HOI4D historical backfill artifacts",
    )
    audit_parser.add_argument("--verify-coverage", action="store_true")
    contact_audit_parser = subparsers.add_parser("audit-contact-artifacts", help="authorized read-only six-dataset contact artifact audit")
    contact_audit_parser.add_argument("--dataset", choices=sorted(SIX_DATASETS))
    contact_audit_parser.add_argument("--verify-coverage", action="store_true")
    subparsers.add_parser("audit-standard10-cpfs-storage", help="read-only size audit of seven fixed standard10 CPFS roots")
    subparsers.add_parser(
        "audit-egofound3r-formal-model",
        help="read-only metadata audit of the fixed EgoFound3R e73dcd8 checkpoint",
    )
    top_level_parser = subparsers.add_parser("audit-cpfs-eval-top-level", help="read-only inventory of the fixed CPFS eval artifact root")
    top_level_parser.add_argument("--name", action="append", default=[])
    shallow_parser = subparsers.add_parser("list-cpfs-eval-top-level", help="read-only shallow listing of the fixed CPFS eval artifact root")
    shallow_parser.add_argument("--contains")
    storage_parser = subparsers.add_parser('audit-cpfs-eval-storage', help='read-only deduplicated CPFS evaluation size, including registered caches')
    storage_parser.add_argument('--cached', action='store_true', help='return last snapshot with its age; no remote scan')
    storage_parser.add_argument('--summary', action='store_true', help='compact GiB totals instead of per-root details')
    run_storage_parser = subparsers.add_parser('audit-run-storage', help='read-only size and process-reference audit for one exact registered CPFS output')
    run_storage_parser.add_argument('--run-id', required=True)
    subparsers.add_parser('audit-cpfs-release-dependencies', help='read-only process references to exact registered release candidates').add_argument('--mirrors', action='store_true')
    subparsers.add_parser('audit-cpfs-release-mirrors', help='read-only comparison of exact registered release source/OSS pairs').add_argument('--drift', action='store_true')
    subparsers.add_parser("compare-migrated-eval-artifacts", help="read-only size-only CPFS/OSSFS artifact comparison")
    subparsers.add_parser("audit-migrated-eval-symlinks", help="read-only fixed migrated artifact symlink audit")
    subparsers.add_parser("overview-six", help="synchronize registered six-dataset formal task status")
    resources_parser = subparsers.add_parser("resources", help="read-only GPU and CPFS snapshot for registered DSW nodes")
    resources_parser.add_argument("--nodes", type=int, nargs="+", help="default: active instance, or legacy registry nodes")
    git_audit_parser = subparsers.add_parser(
        "audit-baseline-git", help="read-only fixed baseline checkout branch/HEAD/dirty audit"
    )
    git_audit_parser.add_argument(
        "--nodes", type=int, nargs="+", help="default: active instance, or legacy compute nodes"
    )
    subparsers.add_parser("audit-p95-runtime-inputs", help="read-only exact runtime dependency audit on 5000")
    subparsers.add_parser("audit-v7-corrected-inputs", help="read-only fixed v7 loader audit on node 5000")
    subparsers.add_parser("audit-hot3d-contact-inputs", help="read-only HOT3D contact prediction and GT audit")
    for action in ("inspect", "path", "watch", "pause", "resume", "start"):
        child = subparsers.add_parser(action, help="step 2: exact run-ID operation")
        if action == "pause":
            child.add_argument("--drain-relay", action="store_true", help="pause relay coordinator only; let active inference release CUDA")
        child.add_argument("--run-id", required=True)
        if action == "path":
            child.add_argument("--export-10s-gallery-local", type=Path,
                               help="copy an exact completed registered 10s gallery into a new local visualization directory")
            child.add_argument("--export-10s-gallery-sample-local", type=Path,
                               help="copy one verified completed PNG/MP4 pair from a registered endpoint104 gallery")
            child.add_argument("--audit-runtime", action="store_true", help="read exact selected method runtime assets on registered reader node")
            child.add_argument("--audit-prediction-index", action="store_true", help="verify exact registered prediction records and array signatures")
            child.add_argument("--prediction-dataset", choices=sorted(SIX_DATASETS), help="dataset below a registered multi-dataset artifact root")
            child.add_argument("--export-p95-mask", type=Path, help="export exact registered P95 mask and frame selection locally")
            child.add_argument("--audit-2d-overlap", type=Path, help="audit exact registered P95 projection inputs into a new local directory")
            child.add_argument("--select-2d-overlap", action="store_true", help="compute cached silhouettes for a registered local 2D selection task")
            child.add_argument("--audit-geometry", action="store_true", help="read per-frame geometry array headers from registered input records")
            child.add_argument("--audit-pi3-depth", action="store_true", help="bounded CPU camera/depth convention sample from the exact catalog")
            child.add_argument("--audit-inputs", action="store_true", help="read exact resident input indices and RGB/prepared files")
            child.add_argument("--audit-10s-visualization-inputs", type=Path, help="read-only paired PAD-Hand/ReViV4D and RGB audit for a frozen local 10s manifest")
            child.add_argument("--audit-10s-common-all", action="store_true", help="also verify registered GT/WiLoR/HaWoR inputs for every selected segment")
            child.add_argument("--rgb-source-run-id", help="exact registered evaluation run providing the RGB input indices")
            child.add_argument("--export-10s-visualization-segment", help="exact selected segment to extract from registered OSS/workspace sources locally")
            child.add_argument("--export-output-dir", type=Path, help="new local input directory for exported visualization files")
            child.add_argument("--export-include-common", action="store_true", help="also extract exact registered GT/WiLoR/HaWoR files for an uncached segment")
            child.add_argument("--export-10s-ego-segment", help="read-only local extraction of one exact stride5 segment on a registered CPFS reader")
            child.add_argument("--audit-10s-ego-segment", help="read-only exact source-file existence check for one stride5 segment")
            child.add_argument("--audit-10s-ego-all", action="store_true", help="read-only identity and file audit of every selected stride5 segment for this registered run")
            child.add_argument("--audit-10s-render-host", action="store_true", help="read-only 5000 visualization dependency and output-capacity probe")
            child.add_argument("--audit-10s-gallery-output", action="store_true", help="read-only registered PNG/MP4 gallery counts and worker log tail")
            child.add_argument("--verify-10s-gallery-output", action="store_true", help="verify registered PNG/MP4 pairs against the frozen manifest")
            child.add_argument("--audit-10s-pad-join", action="store_true", help="verify endpoint gallery PAD-Hand inputs by exact window_id join")
            child.add_argument("--audit-hawor-native-segment", help="quantify native HaWoR camera drift for one staged endpoint104 segment")
            child.add_argument("--reader-node", type=int, help="read-only archive reader node")
            child.add_argument("--audit-fields", action="store_true", help="read NPZ headers at exact registered paths")
            child.add_argument("--audit-pad-bimanual-pilot", action="store_true", help="read exact registered PAD-Hand pilot dual-hand counts")
            child.add_argument("--audit-pad-bimanual-progress", action="store_true", help="read exact registered PAD-Hand lane progress")
            child.add_argument("--audit-dyn-camera-cache-id", help="audit one exact registered Dyn-HaMR prediction camera alignment")
            child.add_argument("--audit-dyn-hand-report", action="store_true", help="verify one registered Dyn-HaMR hand report and return final metric values")
            child.add_argument("--audit-result3-camera-report", action="store_true", help="verify and return the exact Result3 HaWoR/Dyn/Ego camera metrics")
            child.add_argument("--audit-dyn-reuse-index", help="verify indexed and trailing Dyn-HaMR predictions for an exact registered run")
            child.add_argument("--artifact", choices=("gt-cache", "predictions", "reports", "archive"))
            child.add_argument("--method")
        if action == "watch":
            child.add_argument("--interval", type=float, default=10.0)
            child.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        instance = load_instance_config(args.instance_config) if args.instance_config else None
        if args.action == "register":
            identity = dict(value.split("=", 1) for value in args.identity)
            if instance:
                identity.setdefault("instance_id", instance["instance_id"])
            result = register_run(
                args.registry, logical_task_id=args.task_id, dataset=args.dataset,
                method_set=args.method_set, methods=args.methods, phase=args.phase,
                protocol=args.protocol, run_id=args.run_id,
                run_record={
                    "task_type": args.task_type,
                    "scheduler_id": args.scheduler_id,
                    "state_file": args.state_file,
                    "state_section": args.state_section,
                    **({"job_keys": args.job_key} if args.job_key else {}),
                    "output_root": args.output_root,
                    "method_root_output": args.method_root_output,
                    "target_windows_per_method": args.target_windows,
                    "identity": identity,
                },
            )
            emit(result)
            return
        if args.action == "register-egofound3r-final-smoke":
            emit(register_egofound3r_final_smoke(args.registry))
            return
        if args.action == "register-pad-bimanual-p95-manifest":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_pad_bimanual_p95_manifest(args.registry, instance))
            return
        if args.action == "handoff-pad-bimanual-resume":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(handoff_pad_bimanual_resume(args.registry, instance, args.run_id, args.gpu))
            return
        if args.action == "handoff-10s-endpoint-gallery":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(handoff_endpoint_gallery(args.registry, instance, args.run_id))
            return
        if args.action == "register-10s-gallery":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_10s_gallery(args.registry, instance, pilot=args.pilot, shard=args.shard))
            return
        if args.action == "register-10s-endpoint-gallery":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_10s_endpoint_gallery(args.registry, instance, pilot=args.pilot))
            return
        if args.action == "register-10s-endpoint-gallery-hawor-native":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_10s_endpoint_gallery(
                args.registry, instance, pilot=args.pilot, hawor_native=True))
            return
        if args.action == "register-10s-endpoint-gallery-hawor-native-zoom-pilot":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_10s_endpoint_gallery(
                args.registry, instance, pilot=True, hawor_native=True, hand_zoom=True))
            return
        if args.action == "register-10s-endpoint-gallery-auxmethods-zoom-pilot":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_10s_endpoint_gallery(
                args.registry, instance, pilot=True, hawor_native=True, hand_zoom=True,
                aux_methods=True))
            return
        if args.action == "register-10s-endpoint-gallery-auxmethods-zoom":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_10s_endpoint_gallery(
                args.registry, instance, pilot=False, hawor_native=True, hand_zoom=True,
                aux_methods=True))
            return
        if args.action == "register-10s-pad-repair":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_10s_pad_repair(args.registry, instance))
            return
        if args.action == "register-10s-gallery-merge":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_10s_gallery_merge(args.registry, instance))
            return
        if args.action == "register-10s-gallery-handoff":
            if not instance:
                raise TaskError("CURRENT_INSTANCE_CONFIG_REQUIRED")
            emit(register_10s_gallery_handoff(args.registry, instance,
                                              shard_number=args.shard_number, merge=args.merge,
                                              parallelism=args.parallelism))
            return
        registry = load_registry(args.registry)
        if instance:
            apply_instance_config(registry, instance)
        if args.action == "resolve":
            emit(resolve(registry, args.task_id, args.dataset, args.method_set, args.phase, args.protocol))
        elif args.action == "audit-history":
            emit(audit_historical_backfills(registry, verify_coverage=args.verify_coverage))
        elif args.action == "audit-contact-artifacts":
            emit(audit_contact_artifacts(registry, [args.dataset] if args.dataset else None,
                                         verify_coverage=args.verify_coverage))
        elif args.action == "audit-standard10-cpfs-storage":
            emit(audit_standard10_cpfs_storage(registry))
        elif args.action == "audit-egofound3r-formal-model":
            emit(audit_egofound3r_formal_model(registry))
        elif args.action == "audit-cpfs-eval-top-level":
            emit(audit_cpfs_eval_top_level(registry, args.name))
        elif args.action == "list-cpfs-eval-top-level":
            emit(list_cpfs_eval_top_level(registry, args.contains))
        elif args.action == 'audit-cpfs-eval-storage':
            storage = audit_cpfs_eval_storage(registry, cached=args.cached)
            if args.summary:
                storage = {key: storage.get(key) for key in ('status', 'cached', 'checked_at_epoch', 'age_seconds', 'error_count')} | {
                    'evaluation_logical_GiB': round(storage['totals']['logical_bytes'] / 1024**3, 3),
                    'evaluation_allocated_GiB': round(storage['totals']['allocated_bytes'] / 1024**3, 3),
                    'cpfs_available_GiB': round(storage['cpfs']['available_bytes'] / 1024**3, 3),
                    'regular_files': storage['totals']['regular_files'],
                }
            emit(storage)
        elif args.action == 'audit-run-storage':
            emit(audit_run_storage(registry, args.run_id))
        elif args.action == 'audit-cpfs-release-dependencies':
            emit(audit_cpfs_release_dependencies(registry, mirror_dependencies=args.mirrors))
        elif args.action == 'audit-cpfs-release-mirrors':
            emit(audit_cpfs_release_dependencies(registry, mirrors=True, drift=args.drift))
        elif args.action == "compare-migrated-eval-artifacts":
            emit(compare_migrated_eval_artifacts(registry))
        elif args.action == "audit-migrated-eval-symlinks":
            emit(audit_migrated_eval_symlinks(registry))
        elif args.action == "overview-six":
            emit(overview_six_datasets(registry))
        elif args.action == "resources":
            emit(inspect_resources(registry, args.nodes))
        elif args.action == "audit-baseline-git":
            emit(audit_baseline_git(registry, args.nodes or ([instance["port"]] if instance else [5000, 5001, 6001])))
        elif args.action == "audit-p95-runtime-inputs":
            emit(audit_p95_runtime_inputs(registry))
        elif args.action == "audit-v7-corrected-inputs":
            emit(audit_v7_corrected_inputs(registry))
        elif args.action == "audit-hot3d-contact-inputs":
            emit(audit_hot3d_contact_inputs(registry))
        elif args.action == "inspect":
            emit(inspect_run(registry, args.run_id, include_storage=True))
        elif args.action == "path":
            allow_shared_reader = bool(
                instance and args.reader_node == instance["port"]
                and instance["entrance"] == "/mnt/workspace"
            )
            run = merged_run(registry, args.run_id, allow_shared_reader=allow_shared_reader)
            if allow_shared_reader:
                original_reader = int(run.get("identity", {}).get("reader_node", -1))
                if original_reader not in (5000, 5001) or args.reader_node not in (5000, 5001):
                    raise TaskError("SHARED_WORKSPACE_READER_OVERRIDE_NOT_ALLOWED")
            if args.audit_prediction_index:
                emit(audit_registered_prediction_index(run, args.prediction_dataset))
                return
            if args.export_10s_gallery_local:
                from formal_evaluation.export_10s_gallery import export
                result = export(run, PROJECT_ROOT, args.export_10s_gallery_local, _ssh)
                emit({"run_id": args.run_id, **result})
                return
            if args.export_10s_gallery_sample_local:
                from formal_evaluation.export_10s_gallery import export_sample
                result = export_sample(run, PROJECT_ROOT, args.export_10s_gallery_sample_local, _ssh)
                emit({"run_id": args.run_id, **result})
                return
            if args.select_2d_overlap:
                from formal_evaluation.audit_2d_overlap import select
                try:
                    result = select(run, registry, _ssh)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    raise TaskError("OVERLAP_SELECTION_FAILED:" + str(error)) from error
                emit(result, ok=result["ok"])
                return
            if args.audit_10s_pad_join:
                reader_node = int(run.get("identity", {}).get("reader_node", -1))
                if (run.get("task_type") != "visualization" or reader_node not in (5000, 5001)
                        or run.get("identity", {}).get("ego_completion_commit")
                        != "8fc061a615895bd3b5a556f7387bae306e32d9db"):
                    raise TaskError("REGISTERED_ENDPOINT_GALLERY_RUN_REQUIRED")
                local_spec = PROJECT_ROOT / str(run["launch"]["source_alignment_relative"])
                spec = read_json(local_spec)
                progress = {dataset: spec[dataset]["pad_progress"] for dataset in
                            ("h2o", "hot3d", "arctic", "oakink_v2")}
                roots = {dataset: spec[dataset]["method_roots"]["pad_hand"] for dataset in progress}
                rewrites = {dataset: spec[dataset].get("pad_path_rewrites", []) for dataset in progress}
                script = r'''import gzip,json,sys
from pathlib import Path
root=Path(sys.argv[1]);relative=sys.argv[2];progress=json.loads(sys.argv[3]);roots=json.loads(sys.argv[4]);rewrites=json.loads(sys.argv[5])
rows=[json.loads(line) for line in gzip.decompress((root/'runtime'/relative/'selected_manifest_hydrated.jsonl.gz').read_bytes()).decode().splitlines() if line.strip()]
result={};all_ok=True
for dataset in sorted(progress):
    index={};frame_index={};duplicates=[];progress_exists=Path(progress[dataset]).is_file()
    if progress_exists:
        for line in Path(progress[dataset]).read_text().splitlines():
            if not line.strip():continue
            item=json.loads(line);key=item.get('window_id')
            if not key or key in index:duplicates.append(key)
            else:index[key]=item
    for root_path in roots[dataset]:
        for path in Path(root_path).iterdir() if Path(root_path).is_dir() else []:
            metadata=path/'metadata.json'
            if not metadata.is_file() or not (path/'predictions.npz').is_file():continue
            try:item=json.loads(metadata.read_text());key=tuple(item.get('frame_ids',[]))
            except Exception:continue
            if not key:continue
            frame_index.setdefault(key,[]).append({'prediction_dir':str(path),'cache_id':path.name})
    selected=[window for row in rows if row['dataset']==dataset for window in row['windows']]
    missing=[];errors=[];cache_equal=0;via_progress=0;via_frame_ids=0
    for window in selected:
        item=index.get(window['window_id'])
        if item is not None:via_progress+=1
        else:
            matches=frame_index.get(tuple(window['gt']['frame_ids']),[])
            if len(matches)!=1:missing.append(window['window_id']);continue
            item=matches[0];via_frame_ids+=1
        path=Path(item['prediction_dir'])
        candidates=[path]
        for rewrite in rewrites[dataset]:
            source=Path(rewrite['source'])
            if path.is_relative_to(source):candidates.append(Path(rewrite['target'])/path.relative_to(source))
        found=[candidate for candidate in candidates if any(candidate.is_relative_to(Path(value)) for value in roots[dataset]) and (candidate/'predictions.npz').is_file() and (candidate/'metadata.json').is_file()]
        if len(found)!=1:errors.append({'window_id':window['window_id'],'error':'source_match_count:'+str(len(found))});continue
        path=found[0]
        allowed=any(path.is_relative_to(Path(value)) for value in roots[dataset])
        try:meta=json.loads((path/'metadata.json').read_text())
        except Exception as error:errors.append({'window_id':window['window_id'],'error':'metadata:'+str(error)});continue
        if not allowed or not (path/'predictions.npz').is_file() or meta.get('dataset')!=dataset or meta.get('method')!='pad_hand' or meta.get('frame_ids')!=window['gt']['frame_ids']:
            errors.append({'window_id':window['window_id'],'error':'identity_or_path'})
        cache_equal+=int(item.get('cache_id')==window['gt']['cache_id'])
    ok=not duplicates and not missing and not errors and len(selected)==len({x['window_id'] for x in selected})
    all_ok&=ok
    result[dataset]={'selected_windows':len(selected),'resolved_windows':len(selected)-len(missing)-len(errors),'via_progress':via_progress,'via_frame_ids':via_frame_ids,'progress_exists':progress_exists,'cache_id_equal':cache_equal,'duplicates':duplicates[:5],'missing':missing[:5],'errors':errors[:5],'ok':ok}
print(json.dumps({'datasets':result,'selected_segments':len(rows),'selected_windows':sum(len(row['windows']) for row in rows),'ok':all_ok}))'''
                relative = "visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914"
                command = "python3 -c " + shlex.quote(script) + " " + shlex.join([
                    run["output_root"], relative, json.dumps(progress), json.dumps(roots),
                    json.dumps(rewrites)])
                completed = _ssh(run, reader_node, command, timeout=1200)
                if completed.returncode:
                    raise TaskError("PAD_JOIN_AUDIT_FAILED:" + completed.stderr[-500:])
                result = json.loads(completed.stdout)
                emit({"run_id": args.run_id, **result}, ok=result["ok"])
                return
            if args.audit_hawor_native_segment:
                reader_node = int(run.get("identity", {}).get("reader_node", -1))
                if (run.get("task_type") != "visualization" or reader_node not in (5000, 5001)
                        or run.get("identity", {}).get("hawor_label")
                        != "HaWoR native camera (unmasked SLAM)"):
                    raise TaskError("REGISTERED_HAWOR_NATIVE_GALLERY_REQUIRED")
                script = r'''import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);segment=sys.argv[2]
matches=list((root/'input_workspace').glob('*/segments/'+segment+'/inputs'))
if len(matches)!=1:raise ValueError('STAGED_SEGMENT_MATCH_COUNT:'+str(len(matches)))
inputs=matches[0];selection=json.loads((inputs/'selection.json').read_text())
def concat(method):
    parts=[]
    for index in range(5):
        with np.load(inputs/f'{index}_{method}.npz',allow_pickle=False) as source:
            parts.append({key:source[key] for key in source.files})
    return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
gt=concat('gt');hawor=concat('hawor');g=gt['camera_c2w'].astype(float);h=hawor['camera_c2w'].astype(float)
def angle(a,b):
    r=a[:3,:3].T@b[:3,:3];return float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))
aligned=np.empty_like(h);windows=[];transforms=[]
for start in range(0,300,60):
    stop=start+60;transform=g[start]@np.linalg.inv(h[start]);aligned[start:stop]=np.einsum('ij,tjk->tik',transform,h[start:stop]);transforms.append(transform)
    gc=g[start:stop,:3,3];hc=aligned[start:stop,:3,3]
    gt_steps=np.linalg.norm(np.diff(gc,axis=0),axis=1);hawor_steps=np.linalg.norm(np.diff(hc,axis=0),axis=1)
    windows.append({'window_index':start//60,'window_id':selection['windows'][start//60]['window_id'],
      'anchor_translation_error_m':float(np.linalg.norm(hc[0]-gc[0])),
      'anchor_rotation_error_deg':angle(aligned[start],g[start]),
      'end_translation_error_m':float(np.linalg.norm(hc[-1]-gc[-1])),
      'end_rotation_error_deg':angle(aligned[stop-1],g[stop-1]),
      'max_translation_error_m':float(np.linalg.norm(hc-gc,axis=1).max()),
      'gt_path_length_m':float(gt_steps.sum()),'hawor_path_length_m':float(hawor_steps.sum()),
      'path_length_ratio':float(hawor_steps.sum()/max(gt_steps.sum(),1e-12))})
gc=g[:,:3,3];hc=aligned[:,:3,3];err=np.linalg.norm(hc-gc,axis=1)
native_world=hawor['hand_vertices_world'].astype(float);world=np.empty_like(native_world)
for index,start in enumerate(range(0,300,60)):
    stop=start+60;t=transforms[index];world[start:stop]=native_world[start:stop]@t[:3,:3].T+t[:3,3]
valid=hawor['hand_valid'].astype(bool);hand=world[valid]
print(json.dumps({'segment_id':segment,'dataset':selection['dataset'],'sequence_id':selection['sequence_id'],
 'windows':windows,'camera_error_m':{'median':float(np.median(err)),'p95':float(np.percentile(err,95)),'max':float(err.max())},
 'camera_span_m':{'gt_axis':np.ptp(gc,axis=0).tolist(),'hawor_aligned_axis':np.ptp(hc,axis=0).tolist(),
                  'gt_diagonal':float(np.linalg.norm(np.ptp(gc,axis=0))),
                  'hawor_aligned_diagonal':float(np.linalg.norm(np.ptp(hc,axis=0)))},
 'hawor_hand_span_m':np.ptp(hand.reshape(-1,3),axis=0).tolist(),
 'anchor_contract_ok':all(x['anchor_translation_error_m']<1e-6 and x['anchor_rotation_error_deg']<1e-4 for x in windows),
 'interpretation':'Each independent 60-frame native SLAM world is rigidly anchored at its first camera; scale and within-window drift are preserved.'}))'''
                command = "/mnt/workspace/sjc/envs/egofound3r/bin/python -c " + shlex.quote(script) + " " + shlex.join([
                    run["output_root"], args.audit_hawor_native_segment])
                completed = _ssh(run, reader_node, command, timeout=1200)
                if completed.returncode:
                    raise TaskError("HAWOR_NATIVE_SEGMENT_AUDIT_FAILED:" + completed.stderr[-800:])
                result = json.loads(completed.stdout)
                emit({"run_id": args.run_id, **result}, ok=result["anchor_contract_ok"])
                return
            if args.audit_2d_overlap:
                from formal_evaluation.audit_2d_overlap import audit
                try:
                    result = audit(run, registry, args.audit_2d_overlap, _ssh)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    raise TaskError("OVERLAP_AUDIT_FAILED:" + str(error)) from error
                emit(result, ok=result["ok"])
                return
            if args.verify_10s_gallery_output:
                reader_node = int(run.get("identity", {}).get("reader_node", -1))
                endpoint_gallery = (run.get("identity", {}).get("ego_completion_commit")
                                    == "8fc061a615895bd3b5a556f7387bae306e32d9db")
                if run.get("task_type") != "visualization" or reader_node not in (5000, 5001):
                    raise TaskError("REGISTERED_10S_GALLERY_RUN_REQUIRED")
                script = r'''import gzip,json,subprocess,sys
from pathlib import Path
from PIL import Image
root=Path(sys.argv[1]);pilot=sys.argv[2];start=int(sys.argv[3]);end=int(sys.argv[4]);selected=json.loads(sys.argv[5])
relative=sys.argv[6];manifest_name=sys.argv[7]
png_sizes={tuple(value) for value in json.loads(sys.argv[8])};video_sizes={tuple(value) for value in json.loads(sys.argv[9])}
runtime=root/'runtime'/relative
rows=[json.loads(line) for line in gzip.decompress((runtime/manifest_name).read_bytes()).decode().splitlines() if line.strip()]
rows=[rows[index] for index in selected] if selected is not None else rows[start:end]
if pilot:rows=[row for row in rows if row['segment_id']==pilot]
wanted={row['gallery_stem']:row for row in rows}
png={p.stem:p for p in (root/'png_gallery').glob('*.png')}
mp4={p.stem:p for p in (root/'video_gallery').glob('*.mp4') if '.partial.' not in p.name}
paired=set(png)&set(mp4)&set(wanted);errors=[]
for stem in sorted(paired):
    try:
        with Image.open(png[stem]) as im:
            if im.size not in png_sizes:raise ValueError('PNG_SIZE:'+str(im.size))
            im.verify()
        proc=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(mp4[stem])],capture_output=True,text=True,check=True)
        stream=json.loads(proc.stdout)['streams'][0]
        if stream['codec_name']!='h264' or (stream['width'],stream['height']) not in video_sizes or stream['r_frame_rate']!='30/1' or int(stream['nb_frames'])!=300 or abs(float(stream['duration'])-10)>0.02:
            raise ValueError('MP4_CONTRACT:'+str(stream))
    except Exception as error:errors.append({'stem':stem,'error':str(error)})
missing=sorted(set(wanted)-paired)
extra=sorted((set(png)|set(mp4))-set(wanted))
unpaired=sorted((set(png)^set(mp4))&set(wanted))
complete=(root/'COMPLETE').is_file()
print(json.dumps({'expected':len(wanted),'paired_verified':len(paired)-len(errors),
 'paired_verified_stems':sorted(paired-{error['stem'] for error in errors}),
 'missing_count':len(missing),'missing_stems':missing,'unpaired':unpaired,'unexpected':extra,'errors':errors,
 'complete_sentinel':complete,'ok':not (errors or extra or unpaired or missing) and complete}))'''
                relative = ("visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914"
                            if endpoint_gallery else "visualization/batch_10s_177_p95_wmpjpe_20260912")
                manifest_name = "selected_manifest_hydrated.jsonl.gz" if endpoint_gallery else "selected_manifest.jsonl.gz"
                auxiliary_layout = bool(run.get("identity", {}).get("expected_panel_pngs"))
                conditional_dyn = bool(run.get("identity", {}).get("conditional_dyn_hamr_row"))
                png_sizes = ([(3140, 3800), (3140, 4240)] if auxiliary_layout and conditional_dyn else
                             [(3140, 4240)] if auxiliary_layout else [(4450, 4240)]
                             if endpoint_gallery else [(4620, 4240)])
                video_sizes = ([(2090, 2340), (2090, 2620)] if auxiliary_layout and conditional_dyn else
                               [(2090, 2620)] if auxiliary_layout else [(2000, 1780)]
                               if endpoint_gallery else [(1788, 1524)])
                remote_python = "/mnt/workspace/sjc/envs/egofound3r/bin/python" if endpoint_gallery else "python3"
                remote = remote_python + " -c " + shlex.quote(script) + " " + shlex.join([
                    run["output_root"], run.get("identity", {}).get("pilot_segment") or "",
                    str(run.get("identity", {}).get("shard_start", 0)),
                    str(run.get("identity", {}).get("shard_end", 104 if endpoint_gallery else 177)),
                    json.dumps(run.get("identity", {}).get("selected_indices")), relative, manifest_name,
                    json.dumps(png_sizes), json.dumps(video_sizes)])
                completed = _ssh(run, reader_node, remote, timeout=1200)
                if completed.returncode:
                    raise TaskError("GALLERY_VERIFY_FAILED:" + completed.stderr[-500:])
                result = json.loads(completed.stdout)
                emit({"run_id": args.run_id, **result}, ok=result["ok"])
                return
            if args.audit_10s_gallery_output:
                reader_node = int(args.reader_node or run.get("identity", {}).get("reader_node", -1))
                if run.get("task_type") != "visualization" or reader_node not in (5000, 5001):
                    raise TaskError("REGISTERED_10S_GALLERY_RUN_REQUIRED")
                script = r'''import gzip,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);logs=[p for p in (root/'logs').glob('worker*.log') if p.is_file()]
log=max(logs,key=lambda p:p.stat().st_mtime) if logs else root/'logs/worker.log'
png=list((root/'png_gallery').glob('*.png'))
mp4=[p for p in (root/'video_gallery').glob('*.mp4') if '.partial.' not in p.name]
partial=[p for p in (root/'video_gallery').glob('*partial*')]
panels=list((root/'panel_gallery').glob('*/*.png'))
summary=root/'summary.json'
summary_data=json.loads(summary.read_text()) if summary.is_file() else None
preflight=root/'preflight.json'
preflight_data=json.loads(preflight.read_text()) if preflight.is_file() else None
tail=log.read_text(errors='replace').splitlines()[-12:] if log.is_file() else []
handle=root/'control/handle.json';process=[]
if handle.is_file():
 try:
  pid=int(json.loads(handle.read_text())['pid'])
  process=subprocess.run(['ps','-o','pid=,ppid=,etime=,stat=,%cpu=,%mem=,command=','-p',str(pid)],capture_output=True,text=True).stdout.splitlines()
 except Exception as error:process=['process_audit_error:'+str(error)]
relative=Path('visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914')
alignment=root/'runtime'/relative/'source_alignment_5001.json'
manifest=root/'runtime'/relative/'selected_manifest_hydrated.jsonl.gz'
ego_candidates={}
if alignment.is_file() and manifest.is_file():
 spec=json.loads(alignment.read_text());alias=Path(spec['cpfs_alias'])
 rows=[json.loads(line) for line in gzip.decompress(manifest.read_bytes()).decode().splitlines() if line.strip()]
 pending=[row for row in rows if row['segment_id'] not in {p.stem for p in png}]
 paths=[Path(win['pred']['prediction_dir']) for row in pending for win in row['windows']]
 roots={'frozen':None,'registered_alias_target':Path(spec['cpfs_alias_target']),
        'authorized_workspace_history':Path('/mnt/workspace/sjc/eval_artifacts'),
        'verified_p95_freeze_archive':Path('/mnt/oss/pre-train/ego/eval_artifacts/p95_table_freeze_20260915/recompute_dependency')}
 for name,candidate_root in roots.items():
  resolved=[path if candidate_root is None else candidate_root/path.relative_to(alias) for path in paths]
  ok=[(path/'predictions.npz').is_file() and (path/'metadata.json').is_file() for path in resolved]
  ego_candidates[name]={'pairs':sum(ok),'expected':len(ok),
   'first_missing':next((str(path) for path,valid in zip(resolved,ok) if not valid),None)}
print(json.dumps({'output_root':str(root),'png_count':len(png),'mp4_count':len(mp4),
 'paired_count':len({x.stem for x in png}&{x.stem for x in mp4}),
 'partial_video_count':len(partial),'partial_video_bytes':sum(p.stat().st_size for p in partial),
 'panel_png_count':len(panels),'process':process,'log_mtime':log.stat().st_mtime if log.is_file() else None,
 'result_bytes':sum(p.stat().st_size for p in [*png,*mp4] if p.is_file()),
 'summary':({key:summary_data.get(key) for key in ('status','completed','target')}|
            {'failure_count':len(summary_data.get('failures',[]))}) if summary_data else None,
 'preflight':preflight_data,
 'complete_sentinel':(root/'COMPLETE').is_file(),
 'runtime_file_count':sum(1 for x in (root/'runtime').rglob('*') if x.is_file()) if (root/'runtime').is_dir() else 0,
 'ego_source_candidates':ego_candidates,'log_path':str(log),'log_tail':tail}))'''
                completed = _ssh(run, reader_node, "python3 -c " + shlex.quote(script) + " " + shlex.quote(run["output_root"]), timeout=90)
                if completed.returncode:
                    raise TaskError("GALLERY_OUTPUT_AUDIT_FAILED:" + completed.stderr[-500:])
                emit({"run_id": args.run_id, **json.loads(completed.stdout)})
                return
            if args.audit_10s_render_host:
                if args.reader_node != 5000 or run.get("task_type") != "evaluation":
                    raise TaskError("REGISTERED_5000_EVALUATION_READER_REQUIRED")
                script = r'''import importlib.util,json,shutil,subprocess
from pathlib import Path
root=Path('/mnt/workspace/sjc/DATA/eval_artifacts')
mods={name:importlib.util.find_spec(name) is not None for name in ('numpy','matplotlib','PIL')}
commands={name:shutil.which(name) for name in ('ffmpeg','ffprobe')}
print(json.dumps({'python_modules':mods,'commands':commands,'output_parent_exists':root.is_dir(),
 'output_parent_writable':bool(root.stat().st_mode&0o200) if root.is_dir() else False,
 'free_bytes':shutil.disk_usage(root).free if root.is_dir() else None}))'''
                completed = _ssh(run, 5000, "python3 -c " + shlex.quote(script), timeout=60)
                if completed.returncode:
                    raise TaskError("RENDER_HOST_AUDIT_FAILED:" + completed.stderr[-500:])
                emit({"run_id": args.run_id, "node": 5000, **json.loads(completed.stdout)})
                return
            if args.audit_10s_ego_all:
                if not args.audit_10s_visualization_inputs:
                    raise TaskError("EGO_AUDIT_REQUIRES_MANIFEST")
                if args.reader_node not in registry.get("policy", {}).get("resource_nodes", []) or registry.get("policy", {}).get("resource_node_entrances", {}).get(str(args.reader_node)) not in ("/mnt/cpfs", "/mnt/workspace"):
                    raise TaskError("REGISTERED_EGO_READER_REQUIRED")
                from formal_evaluation.export_10s_ego import audit_all
                result = audit_all(run, args.audit_10s_visualization_inputs, args.reader_node, _ssh)
                emit(result, ok=result["ok"])
                return
            if args.audit_10s_ego_segment:
                if not args.audit_10s_visualization_inputs:
                    raise TaskError("EGO_AUDIT_REQUIRES_MANIFEST")
                if args.reader_node not in registry.get("policy", {}).get("resource_nodes", []) or registry.get("policy", {}).get("resource_node_entrances", {}).get(str(args.reader_node)) not in ("/mnt/cpfs", "/mnt/workspace"):
                    raise TaskError("REGISTERED_EGO_READER_REQUIRED")
                from formal_evaluation.export_10s_ego import audit
                result = audit(run, args.audit_10s_visualization_inputs, args.audit_10s_ego_segment,
                               args.reader_node, _ssh)
                emit(result, ok=result["ok"])
                return
            if args.export_10s_ego_segment:
                if not args.audit_10s_visualization_inputs or not args.export_output_dir:
                    raise TaskError("EGO_EXPORT_REQUIRES_MANIFEST_AND_LOCAL_INPUT_DIR")
                if args.reader_node not in registry.get("policy", {}).get("resource_nodes", []) or registry.get("policy", {}).get("resource_node_entrances", {}).get(str(args.reader_node)) not in ("/mnt/cpfs", "/mnt/workspace"):
                    raise TaskError("REGISTERED_EGO_READER_REQUIRED")
                from formal_evaluation.export_10s_ego import export
                result = export(run, args.audit_10s_visualization_inputs, args.export_10s_ego_segment,
                                args.export_output_dir, args.reader_node, _ssh)
                emit(result, ok=result["ok"])
                return
            if args.audit_10s_visualization_inputs:
                if not args.rgb_source_run_id:
                    raise TaskError("RGB_SOURCE_RUN_ID_REQUIRED")
                from formal_evaluation.audit_10s_visualization_inputs import audit
                result = audit(run, merged_run(registry, args.rgb_source_run_id), args.audit_10s_visualization_inputs, _ssh,
                               export_segment_id=args.export_10s_visualization_segment, output_dir=args.export_output_dir,
                               include_common=args.export_include_common, check_common_all=args.audit_10s_common_all)
                emit(result, ok=result["ok"])
                return
            if args.audit_runtime:
                from formal_evaluation.audit_registered_method_runtime import audit
                result = audit(run, args.method, args.reader_node or 5000, _ssh)
                emit(result, ok=result["ok"])
                return
            if args.audit_dyn_camera_cache_id:
                result = audit_dyn_camera_alignment(run, args.audit_dyn_camera_cache_id)
                emit(result, ok=result["ok"])
                return
            if args.audit_dyn_hand_report:
                result = audit_dyn_hand_report_values(run)
                emit(result, ok=result["ok"])
                return
            if args.audit_result3_camera_report:
                result = audit_result3_camera_report(run)
                emit(result, ok=result["ok"])
                return
            if args.audit_pad_bimanual_pilot or args.audit_pad_bimanual_progress:
                if not str(run.get("logical_task_id", "")).startswith("evaluation:pad_hand:bimanual:"):
                    raise TaskError("PAD_BIMANUAL_ID_REQUIRED")
                node = int(run["launch"]["node_candidates"][0])
                support = [name for name in run["launch"]["support_files"] if name.endswith("_spec.json")]
                if len(support) != 1:
                    raise TaskError("PAD_BIMANUAL_SPEC_NOT_UNIQUE")
                spec_path = str(Path(run["launch"]["runtime_root"]) / support[0])
                script = """import json,sys
from pathlib import Path
root=Path(sys.argv[1]); spec=json.loads(Path(sys.argv[2]).read_text()); result={}
for dataset,source in spec['datasets'].items():
    progress=root/dataset/'progress.jsonl'; summary=root/dataset/'summary.json'
    rows=[json.loads(line) for line in progress.read_text().splitlines() if line] if progress.exists() else []
    done=json.loads(summary.read_text()) if summary.exists() else None
    result[dataset]={'completed_records':len(rows),'expected_windows':source['expected_windows'],
                     'valid_side_frames':done.get('valid_side_frames') if done else
                         [sum(row['valid_side_frames'][side] for row in rows) for side in range(2)],
                     'status':done.get('status') if done else 'running_or_pending'}
print(json.dumps({'datasets':result,'complete':(root/'COMPLETE').is_file()}))"""
                command = "python3 -c " + shlex.quote(script) + " " + shlex.quote(str(run["output_root"])) + " " + shlex.quote(spec_path)
                completed = _ssh(run, node, command)
                if completed.returncode:
                    raise TaskError("PAD_BIMANUAL_PILOT_AUDIT_FAILED")
                result = json.loads(completed.stdout)
                emit({"run_id": args.run_id, "node": node, **result})
                return
            if args.audit_dyn_reuse_index:
                result = audit_dyn_reuse_index(run, args.audit_dyn_reuse_index)
                emit(result, ok=result["ok"])
                return
            if args.export_p95_mask:
                from formal_evaluation.export_registered_p95_mask import export
                emit(export(run, registry, args.export_p95_mask, _ssh))
                return
            if args.audit_pi3_depth:
                result = audit_pi3_depth_sample(run)
                emit(result, ok=result['ok'])
                return
            if args.audit_inputs or args.audit_geometry:
                result = audit_inference_inputs(run, args.audit_geometry, args.reader_node or 5000)
                emit(result, ok=result['ok'])
                return
            if args.artifact == 'archive':
                result = archive_paths(run, args.reader_node)
                emit(result, ok=result['all_verified'])
                return
            if run.get("superseded_artifact_run_id"):
                raise TaskError("ARTIFACT_SUPERSEDED:" + run["superseded_artifact_run_id"])
            if run.get("artifact_catalog_run_id"):
                raise TaskError("ARTIFACT_CATALOG_REQUIRED:" + run["artifact_catalog_run_id"])
            if run.get("artifact_catalog"):
                result = catalog_paths(run, args.artifact, args.method, args.audit_fields)
                emit(result, ok=result["all_verified"])
                if not result["all_verified"]:
                    raise SystemExit(2)
            elif args.audit_fields:
                result = indexed_prediction_fields(run)
                emit(result, ok=result['ok'])
            elif args.artifact or args.method:
                raise TaskError("ARTIFACT_CATALOG_NOT_REGISTERED:" + args.run_id)
            else:
                emit({"run_id": args.run_id, "output_root": run["output_root"], "output_paths": output_paths(run), "path_status": "historical_unverified"})
        elif args.action == "watch":
            previous = None
            while True:
                current = inspect_run(registry, args.run_id)
                signature = json.dumps(current, sort_keys=True)
                if signature != previous:
                    emit(current)
                    previous = signature
                if args.once or all(state in TERMINAL_STATES for state in current["state_counts"]):
                    break
                time.sleep(max(args.interval, 1.0))
        else:
            if args.action == "start":
                result = launch_registered_run(registry, args.run_id)
            else:
                desired = "paused" if args.action == "pause" else "running"
                result = set_desired_state(registry, args.run_id, desired, args.controls)
                result.update(signal_registered_jobs(registry, args.run_id, "STOP" if args.action == "pause" else "CONT", getattr(args, "drain_relay", False)))
            emit(result)
    except (TaskError, ValueError) as error:
        emit({"error": str(error)}, ok=False)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
