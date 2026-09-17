from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from formal_evaluation import auto_scheduler, remote_task_control, taskctl
from formal_evaluation.run_h2o_contact_metrics import _canonical_prediction_window_id


def fixture_registry(tmp_path: Path, *, legacy: bool = False) -> tuple[dict, Path, Path]:
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "jobs": {
            "h2o::wilor": {
                "status": "running",
                "node": 5001,
                **({} if legacy else {"handle_path": "/registered/h2o-wilor.json"}),
            },
            "h2o::hawor": {"status": "done"},
        }
    }), encoding="utf-8")
    registry = {
        "schema_version": "evaluation_task_registry_v1",
        "method_sets": {"standard2": ["wilor", "hawor"]},
        "logical_tasks": {
            "formal:h2o:standard2:60f": {
                "dataset": "h2o", "method_set": "standard2", "phase": "formal", "protocol": "60f",
                "latest": "run-exact-1",
            }
        },
        "run_template": {"state_file": str(state_path)},
        "runs": {
            "run-exact-1": {
                "logical_task_id": "formal:h2o:standard2:60f",
                "target_windows_per_method": 3,
                "output_root": "/results/h2o",
            }
        },
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return registry, registry_path, state_path


class TaskControlTest(unittest.TestCase):
    def test_contact_metrics_maps_cache_id_to_canonical_window_id(self) -> None:
        self.assertEqual(
            _canonical_prediction_window_id("cache-hash", {"cache-hash": "sequence:000-059"}),
            "sequence:000-059",
        )

    def test_prediction_target_is_a_completion_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                output = root / "s2contact" / "formal" / str(index)
                output.mkdir(parents=True)
                (output / "metadata.json").write_text("{}", encoding="utf-8")
                (output / "predictions.npz").write_bytes(b"npz")
            probe = [{"key": "h2o::s2contact", "output_root": str(root),
                      "prediction_root": str(root / "s2contact"), "target_windows": 3}]
            completed = subprocess.run(
                [sys.executable, "-c", taskctl.REMOTE_PROGRESS_PROBE, json.dumps(probe)],
                text=True, capture_output=True, check=True,
            )
            observed = json.loads(completed.stdout)["h2o::s2contact"]
            self.assertEqual(observed["count"], 3)
            self.assertEqual(observed["status"], "done")
            self.assertFalse(observed["completion_evidence"]["complete_exists"])
            (root / "COMPLETE").write_text("complete\n")
            (root / "smoke_summary.json").write_text('{"status":"complete","global_stride":5}')
            completed = subprocess.run(
                [sys.executable, "-c", taskctl.REMOTE_PROGRESS_PROBE, json.dumps(probe)],
                text=True, capture_output=True, check=True,
            )
            evidence = json.loads(completed.stdout)["h2o::s2contact"]["completion_evidence"]
            self.assertTrue(evidence["complete_exists"])
            self.assertEqual(evidence["smoke_summary.json"]["global_stride"], 5)

    def test_gpu_availability_depends_on_presence_and_processes(self) -> None:
        # Execute the actual GPU-query portion without Linux /proc or remote access.
        probe = taskctl.REMOTE_RESOURCE_PROBE.split("gpu = subprocess.run(", 1)[1]
        probe = "gpu = subprocess.run(" + probe.split("usage = shutil.disk_usage", 1)[0]
        gpu = mock.Mock(returncode=0, stdout="0, GPU-free, 100, 1000\n8, GPU-busy, 1, 81000\n")
        apps = mock.Mock(returncode=0, stdout="GPU-busy, 123, python, 1\n")
        namespace = {"subprocess": subprocess, "json": json, "sys": sys}
        with mock.patch.object(subprocess, "run", side_effect=[gpu, apps]):
            exec(probe, namespace)
        rows = {row["index"]: row for row in namespace["rows"]}
        self.assertTrue(rows[0]["idle"])
        self.assertEqual(rows[0]["memory_free_mb"], 1000)
        self.assertFalse(rows[8]["idle"])
        self.assertNotIn(1, rows)  # A nonexistent GPU is never offered.

    def test_resource_snapshot_has_no_memory_threshold(self) -> None:
        registry = {
            "run_template": {"ssh": {"host": "example.invalid", "key": "~/.ssh/test"}},
            "policy": {
                "resource_nodes": list(taskctl.RESOURCE_NODES),
                "resource_node_entrances": {
                    str(node): "/mnt/workspace" for node in taskctl.RESOURCE_NODES
                },
            },
        }
        payload = {"ok": True,
                   "cpu": {"count": 128, "usage_percent": 12.5, "load_1m_per_cpu": 0.1},
                   "memory": {"total_bytes": 100, "available_bytes": 80},
                   "oss": {"root": "/mnt/oss/pre-train/ego/eval_artifacts", "readable": True},
                   "gpus": [{"index": 0, "allowed": True, "idle": True}],
                   "storage": {"total_bytes": 100 * 1024**3, "used_bytes": 75 * 1024**3,
                               "available_bytes": 25 * 1024**3}}
        response = mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with mock.patch.object(taskctl, "_ssh", return_value=response) as ssh:
            result = taskctl.inspect_resources(registry)
        self.assertEqual(ssh.call_count, len(taskctl.RESOURCE_NODES))
        self.assertTrue(result["nodes"]["5000"]["gpus"][0]["idle"])
        self.assertEqual(result["nodes"]["5001"]["cpu"]["count"], 128)
        self.assertEqual(result["nodes"]["6001"]["memory"]["available_bytes"], 80)
        self.assertTrue(result["nodes"]["5000"]["oss"]["readable"])
        self.assertEqual(result["nodes"]["5001"]["storage"]["used_percent"], 75.0)
        self.assertEqual(result["nodes"]["5001"]["storage"]["available_percent"], 25.0)
        self.assertEqual(result["nodes"]["5001"]["storage"]["available_gib"], 25.0)
        self.assertEqual(
            result["nodes"]["5001"]["storage"]["progress_bar"],
            "[███████████████░░░░░] 75.00% used | 25.00 GiB free",
        )
        self.assertNotIn("minimum_free_mb", result)
        with mock.patch.object(taskctl, "_ssh", return_value=response) as ssh:
            result = taskctl.inspect_resources(registry, (5000, 5001, 6001))
        self.assertEqual([call.args[1] for call in ssh.call_args_list], [5000, 5001, 6001])
        self.assertNotIn("4093", result["nodes"])

    def test_cache_readability_reports_duplicate_rows_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"cache_id": "duplicate", "array_path": str(root / "missing.npz"),
                   "metadata_path": str(root / "missing.json")}
            (root / "index.jsonl").write_text((json.dumps(row) + "\n") * 2)
            completed = subprocess.run(
                [sys.executable, "-c", taskctl.REMOTE_READABILITY_PROBE,
                 str(root), "hoi4d", "index.jsonl"],
                text=True, capture_output=True, check=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["index_lines"], 2)
            self.assertEqual(result["unique_cache_ids"], 1)
            self.assertEqual(result["missing_index_artifacts"], 4)
            self.assertFalse(result["complete_exists"])

    def test_metric_recompute_uses_registered_node_and_retry_handle(self) -> None:
        registry = {
            "method_sets": {"metrics": ["egofound3r"]},
            "logical_tasks": {"metrics": {
                "dataset": "taco", "method_set": "metrics", "phase": "formal",
                "protocol": "60f", "latest": "run",
            }},
            "runs": {"run": {
                "logical_task_id": "metrics", "output_root": "/results/metrics",
                "launch_retry": 1,
                "identity": {"pipeline": "metric_recompute_rot_hawor_worlddiag_v1",
                             "node_candidates": [5001], "worktree": "/work"},
            }},
        }
        run = taskctl.merged_run(registry, "run")
        self.assertEqual(run["launch"]["node_candidates"], [5001])
        self.assertEqual(run["launch"]["handle_path"], "/results/metrics/handle.retry1.json")
        self.assertEqual(run["launch"]["runtime_root"], "/results/metrics/runtime_retry1")

    def test_readability_probe_uses_only_registered_nodes_and_paths(self) -> None:
        run = {
            "output_root": "/mnt/workspace/sjc/cache",
            "readability_probe": {
                "nodes": [5001, 6001],
                "dataset_subdir": "hoi4d",
                "index_file": "index.jsonl",
            },
        }
        response = mock.Mock(returncode=0, stdout=json.dumps({"ok": True, "npz_files": 3}), stderr="")
        with mock.patch.object(taskctl, "_ssh", return_value=response) as ssh:
            result = taskctl.inspect_readability(run)
        self.assertEqual(result["5001"]["npz_files"], 3)
        self.assertEqual(result["6001"]["npz_files"], 3)
        self.assertEqual([call.args[1] for call in ssh.call_args_list], [5001, 6001])
        self.assertTrue(all("/mnt/workspace/sjc/cache" in call.args[2] for call in ssh.call_args_list))

    def test_node_path_layout_translates_only_registered_node(self) -> None:
        run = {"node_path_layout": {
            "canonical_prefix": "/mnt/workspace/sjc",
            "node_prefixes": {"4093": "/mnt/cpfs/sjc"},
        }}
        path = "/mnt/workspace/sjc/DATA/HOI4D"
        self.assertEqual(taskctl._node_text(run, 4093, path), "/mnt/cpfs/sjc/DATA/HOI4D")
        command = "exec /mnt/workspace/sjc/envs/egofound3r/bin/python /mnt/workspace/sjc/job.py"
        self.assertNotIn("/mnt/workspace/sjc", taskctl._node_text(run, 4093, command))
        self.assertEqual(taskctl._node_text(run, 5000, path), path)

    def test_hot3d_contact_audit_uses_first_reachable_node(self) -> None:
        registry = {"run_template": {"ssh": {"host": "example.invalid", "key": "~/.ssh/test"}}}
        unavailable = mock.Mock(returncode=255, stdout="", stderr="")
        healthy = mock.Mock(returncode=0, stdout=json.dumps({"ok": True, "gt_count": 400}), stderr="")
        with mock.patch.object(taskctl, "_ssh", side_effect=[unavailable, healthy]) as ssh:
            result = taskctl.audit_hot3d_contact_inputs(registry)
        self.assertEqual(ssh.call_count, 2)
        self.assertEqual(result["observed_node"], 5000)
        self.assertEqual(result["gt_count"], 400)

    def test_standard10_storage_audit_returns_fixed_rows(self) -> None:
        registry = {"run_template": {"ssh": {"host": "example.invalid", "key": "~/.ssh/test"}}}
        payload = {"status": "ok", "rows": [{"name": "formal_arctic", "regular_file_bytes": 3}]}
        response = mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with mock.patch.object(taskctl.subprocess, "run", return_value=response):
            result = taskctl.audit_standard10_cpfs_storage(registry)
        self.assertEqual(result["rows"], payload["rows"])
        self.assertEqual(result["observed_node"], 5000)

    def test_egofound3r_model_audit_uses_only_fixed_release_paths(self) -> None:
        registry = {"run_template": {"ssh": {"host": "example.invalid", "key": "~/.ssh/test"}}}
        payload = {"status": "ok", "checkpoint": {"path": str(taskctl.EGOFOUND3R_FORMAL_CHECKPOINT)}}
        response = mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with mock.patch.object(taskctl, "_ssh", return_value=response) as ssh:
            result = taskctl.audit_egofound3r_formal_model(registry)
        self.assertEqual(result["observed_node"], 5000)
        remote = ssh.call_args.args[2]
        self.assertIn(str(taskctl.EGOFOUND3R_FORMAL_CHECKPOINT), remote)
        self.assertIn(str(taskctl.EGOFOUND3R_INFERENCE_ROOT), remote)
        self.assertIn(str(taskctl.EGOFOUND3R_TRAINING_SOURCE_ROOT), remote)
        self.assertIn("validate_marker_checkpoint_contract", remote)
        self.assertNotIn("--root", remote)

    def test_egofound3r_final_smoke_registration_is_fixed_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, registry_path, _ = fixture_registry(root)
            state_path = root / "smoke-state.json"
            with (
                mock.patch.object(taskctl, "EGOFOUND3R_FINAL_SMOKE_TASK_ID", "smoke:h2o:final:test"),
                mock.patch.object(taskctl, "EGOFOUND3R_FINAL_SMOKE_METHOD_SET", "egofound3r_final_test"),
                mock.patch.object(taskctl, "EGOFOUND3R_FINAL_SMOKE_RUN_ID", "smoke-final-test"),
                mock.patch.object(taskctl, "EGOFOUND3R_FINAL_SMOKE_STATE", str(state_path)),
                mock.patch.object(taskctl, "EGOFOUND3R_FINAL_SMOKE_JOB", "h2o::egofound3r_final_test"),
                mock.patch.object(taskctl, "EGOFOUND3R_FINAL_SMOKE_OUTPUT", root / "remote-output"),
            ):
                first = taskctl.register_egofound3r_final_smoke(registry_path)
                second = taskctl.register_egofound3r_final_smoke(registry_path)
                run = taskctl.merged_run(taskctl.load_registry(registry_path), first["run_id"])
            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            self.assertEqual(json.loads(state_path.read_text())["jobs"]["h2o::egofound3r_final_test"]["status"], "pending")
            self.assertEqual(run["identity"]["checkpoint_sha256"], taskctl.EGOFOUND3R_CHECKPOINT_SHA256)
            self.assertEqual(run["launch"]["worktree_commit"], taskctl.EGOFOUND3R_BASELINES_COMMIT)
            self.assertTrue(run["launch"]["worktree_bundle"].endswith("_from_cbdb6ce.bundle"))
            self.assertTrue(run["launch"]["wait_for_idle_gpu"])

    def test_eval_top_level_audit_uses_empty_root_selector(self) -> None:
        registry = {"run_template": {"ssh": {"host": "example.invalid", "key": "~/.ssh/test"}}}
        response = mock.Mock(returncode=0, stdout=json.dumps({"status": "ok", "rows": []}), stderr="")
        with mock.patch.object(taskctl.subprocess, "run", return_value=response) as run:
            taskctl.audit_cpfs_eval_top_level(registry)
        self.assertIn("'[]'", run.call_args.args[0][-1])

    def test_compare_migrated_artifacts_falls_back_to_healthy_node(self) -> None:
        registry = {
            "run_template": {"ssh": {"host": "example.invalid", "key": "~/.ssh/test"}},
            "method_sets": {"noop": []},
            "logical_tasks": {
                "task:first": {"method_set": "noop", "storage_probe": {"node": 5000}},
                "task:second": {"method_set": "noop", "storage_probe": {"node": 5001}},
            },
            "runs": {
                "first": {"logical_task_id": "task:first"},
                "second": {"logical_task_id": "task:second"},
            },
        }
        unavailable = mock.Mock(returncode=1, stdout="", stderr="INVALID_OSSFS_DESTINATION\n")
        healthy = mock.Mock(returncode=0, stdout=json.dumps({"status": "ok", "differences": 0}), stderr="")
        with mock.patch.object(taskctl.subprocess, "run", side_effect=[unavailable, healthy]) as run:
            result = taskctl.compare_migrated_eval_artifacts(registry)
        self.assertEqual(result["observed_node"], 5001)
        self.assertEqual([call.args[0][8] for call in run.call_args_list], ["5000", "5001"])

    def test_launch_target_requires_both_requested_gpus(self) -> None:
        run = {"run_id": "pad", "dataset": "h2o", "ssh": {}, "remote_python": "python3"}
        launch = {"worktree": "/work", "method": "pad_hand", "node_candidates": [5000],
                  "gpu_candidates": [2, 3], "required_gpu_count": 2}
        response = mock.Mock(returncode=0, stdout=json.dumps({"ok": True, "idle_gpus": [2, 3]}))
        with mock.patch.object(taskctl, "_ssh", return_value=response):
            self.assertEqual(taskctl._choose_launch_target(run, launch), (5000, "2,3"))

    def test_gpu_launch_bootstraps_missing_worktree_from_registered_bundle(self) -> None:
        run = {"run_id": "ego", "dataset": "h2o", "ssh": {}, "remote_python": "python3"}
        launch = {
            "worktree": "/mnt/workspace/sjc/DATA/runtime_worktrees/ego",
            "worktree_source": "/mnt/workspace/sjc/DATA/runtime_worktrees/source",
            "worktree_bundle": "/mnt/cpfs/run/full.bundle",
            "worktree_commit": "b1fab90771656bfaf1905cfe2383728ddf039845",
            "worktree_fetch_ref": "formal-hand-fix",
            "method": "egofound3r",
            "node_candidates": [5000],
            "gpu_candidates": [0],
            "required_gpu_count": 1,
            "preflight_paths": [],
            "values": {"python": "/opt/python"},
        }
        response = mock.Mock(returncode=0, stdout=json.dumps({"ok": True, "idle_gpus": [0]}))
        with mock.patch.object(taskctl, "_ssh", return_value=response) as ssh:
            self.assertEqual(taskctl._choose_launch_target(run, launch), (5000, 0))
        remote = ssh.call_args.args[2]
        self.assertIn('["git", "clone", "--no-local", worktree_bundle, worktree]', remote)

    def test_gpu_launch_can_apply_thin_bundle_over_registered_base_commit(self) -> None:
        run = {"run_id": "ego", "dataset": "h2o", "ssh": {}, "remote_python": "python3"}
        launch = {
            "worktree": "/mnt/workspace/sjc/DATA/runtime_worktrees/ego",
            "worktree_source": "/mnt/workspace/sjc/DATA/runtime_worktrees/source",
            "worktree_bundle": "/mnt/cpfs/run/thin.bundle",
            "worktree_base_commit": "5db37bb2ea5bb240bb5420f52807dcdc5db40e1c",
            "worktree_commit": "b1fab90771656bfaf1905cfe2383728ddf039845",
            "worktree_fetch_ref": "formal-hand-fix",
            "method": "egofound3r",
            "node_candidates": [5000],
            "gpu_candidates": [0],
            "required_gpu_count": 1,
            "preflight_paths": [],
            "values": {"python": "/opt/python"},
        }
        response = mock.Mock(returncode=0, stdout=json.dumps({"ok": True, "idle_gpus": [0]}))
        with mock.patch.object(taskctl, "_ssh", return_value=response) as ssh:
            self.assertEqual(taskctl._choose_launch_target(run, launch), (5000, 0))
        remote = ssh.call_args.args[2]
        self.assertIn('["git", "-C", worktree_source, "worktree", "add", "--detach", worktree, worktree_base_commit]', remote)
        self.assertEqual(shlex.split(remote)[11], launch["worktree_base_commit"])

    def test_gpu_launch_uses_node_specific_fallback_candidates(self) -> None:
        run = {"run_id": "ego", "dataset": "h2o", "ssh": {}, "remote_python": "python3"}
        launch = {
            "worktree": "/work",
            "method": "egofound3r",
            "node_candidates": [5000, 5001],
            "gpu_candidates": [0],
            "gpu_candidates_by_node": {"5000": [3], "5001": [5]},
            "required_gpu_count": 1,
        }
        unavailable = mock.Mock(returncode=0, stdout=json.dumps({"ok": True, "idle_gpus": []}))
        available = mock.Mock(returncode=0, stdout=json.dumps({"ok": True, "idle_gpus": [5]}))
        with mock.patch.object(taskctl, "_ssh", side_effect=[unavailable, available]) as ssh:
            self.assertEqual(taskctl._choose_launch_target(run, launch), (5001, 5))
        self.assertEqual(shlex.split(ssh.call_args_list[0].args[2])[6], "3")
        self.assertEqual(shlex.split(ssh.call_args_list[1].args[2])[6], "5")

    def test_cpu_launch_passes_exact_clone_identity_to_preflight(self) -> None:
        run = {"run_id": "metrics", "dataset": "h2o", "ssh": {}, "remote_python": "python3"}
        launch = {"worktree": "/mnt/workspace/sjc/DATA/runtime_worktrees/metrics", "method": "egofound3r",
                  "node_candidates": [5001], "resource": "cpu", "preflight_paths": ["/gt/index.jsonl"],
                  "worktree_clone_url": "https://example.invalid/repo.git", "worktree_clone_ref": "formal-hand-fix",
                  "worktree_commit": "b1fab90771656bfaf1905cfe2383728ddf039845"}
        response = mock.Mock(returncode=0, stdout=json.dumps({"ok": True}))
        with mock.patch.object(taskctl, "_ssh", return_value=response) as ssh:
            self.assertEqual(taskctl._choose_launch_target(run, launch), (5001, -1))
        remote = ssh.call_args.args[2]
        self.assertIn(launch["worktree_clone_url"], remote)
        self.assertIn(launch["worktree_clone_ref"], remote)
        self.assertIn(launch["worktree_commit"], remote)

    def test_current_scheduler_registration_is_idempotent_with_existing_nine_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(__file__).parents[1] / "config/evaluation_task_registry.json"
            registry_path = Path(directory) / "registry.json"
            registry_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            original_count = len(taskctl.load_registry(registry_path)["runs"])
            state_path = Path(directory) / "state.json"
            with (mock.patch.object(auto_scheduler, "REGISTRY_PATH", str(registry_path)),
                  mock.patch.object(auto_scheduler, "STATE_PATH", str(state_path))):
                auto_scheduler.register_scheduler_runs()
                count_after_first_registration = len(taskctl.load_registry(registry_path)["runs"])
                auto_scheduler.register_scheduler_runs()
            saved = taskctl.load_registry(registry_path)
            self.assertEqual(count_after_first_registration, original_count + 1)
            self.assertEqual(len(saved["runs"]), count_after_first_registration)

    def test_registration_is_idempotent_and_only_new_run_becomes_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, registry_path, _ = fixture_registry(root)
            common = {
                "logical_task_id": "formal:h2o:standard2:60f", "dataset": "h2o",
                "method_set": "standard2", "phase": "formal", "protocol": "60f",
            }
            first = taskctl.register_run(
                registry_path, **common,
                run_record={"scheduler_id": "next", "state_file": str(root / "state.json"),
                            "output_root": "/results/h2o-next", "target_windows_per_method": 3},
            )
            duplicate = taskctl.register_run(
                registry_path, **common,
                run_record={"scheduler_id": "next", "state_file": str(root / "state.json"),
                            "output_root": "/results/h2o-next", "target_windows_per_method": 3},
            )
            with self.assertRaisesRegex(taskctl.TaskError, "OUTPUT_ROOT_ALREADY_REGISTERED"):
                taskctl.register_run(
                    registry_path, **common,
                    run_record={"scheduler_id": "changed", "state_file": str(root / "state.json"),
                                "output_root": "/results/h2o-next", "target_windows_per_method": 99},
                )
            newer = taskctl.register_run(
                registry_path, **common,
                run_record={"scheduler_id": "newer", "state_file": str(root / "state.json"),
                            "output_root": "/results/h2o-newer", "target_windows_per_method": 3},
            )
            retry_old = taskctl.register_run(
                registry_path, **common,
                run_record={"scheduler_id": "next", "state_file": str(root / "state.json"),
                            "output_root": "/results/h2o-next", "target_windows_per_method": 3},
            )
            saved = taskctl.load_registry(registry_path)
            self.assertTrue(first["created"])
            self.assertFalse(duplicate["created"])
            self.assertEqual(first["run_id"], duplicate["run_id"])
            self.assertEqual(retry_old["run_id"], first["run_id"])
            self.assertEqual(saved["logical_tasks"][common["logical_task_id"]]["latest"], newer["run_id"])
            self.assertEqual(len(saved["runs"]), 3)

    def test_gt_cache_registration_uses_exact_state_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, registry_path, state_path = fixture_registry(root)
            state_path.write_text(json.dumps({"cache_jobs": {"h2o": {
                "status": "running", "count": 2, "node": 5001, "gpu": 3,
                "handle_path": "/registered/cache.json",
            }}}), encoding="utf-8")
            result = taskctl.register_run(
                registry_path, logical_task_id="gt-cache:h2o:cache:60f", dataset="h2o",
                method_set="cache", methods=["gt_cache"], phase="gt-cache", protocol="60f",
                run_record={"task_type": "gt-cache", "scheduler_id": "cache-1",
                            "state_file": str(state_path), "state_section": "cache_jobs",
                            "job_keys": ["h2o"], "output_root": "/cache/h2o",
                            "target_windows_per_method": 3},
            )
            inspected = taskctl.inspect_run(taskctl.load_registry(registry_path), result["run_id"])
            self.assertEqual(inspected["task_type"], "gt-cache")
            self.assertEqual(inspected["active_locations"]["h2o"]["gpu"], 3)

    def test_exact_resolve_and_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _, _ = fixture_registry(Path(directory))
            resolved = taskctl.resolve(registry, None, "h2o", "standard2", "formal", "60f")
            self.assertEqual(resolved["run_id"], "run-exact-1")
            self.assertEqual(resolved["output_root"], "/results/h2o")
            inspected = taskctl.inspect_run(registry, "run-exact-1")
            self.assertEqual(inspected["state_counts"], {"done": 1, "running": 1})
            self.assertEqual(inspected["progress"], {"known_complete_windows": 3, "target_windows": 6})
            with self.assertRaisesRegex(taskctl.TaskError, "NOT_REGISTERED"):
                taskctl.resolve(registry, None, "h2", "standard2", "formal", "60f")

    def test_inspect_refreshes_only_active_registered_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _, state_path = fixture_registry(Path(directory))
            registry["run_template"]["ssh"] = {"host": "example.invalid", "key": "~/.ssh/test"}
            with mock.patch.object(taskctl, "remote_progress", return_value={
                "h2o::wilor": {"process_status": "running", "count": 2, "source": "prediction_dirs"}
            }) as probe:
                inspected = taskctl.inspect_run(registry, "run-exact-1")
            self.assertEqual(inspected["progress"], {"known_complete_windows": 5, "target_windows": 6})
            self.assertEqual(inspected["remote_sync"]["updated_jobs"], ["h2o::wilor"])
            self.assertEqual(probe.call_count, 1)
            probes = probe.call_args.args[2]
            self.assertEqual([item["key"] for item in probes], ["h2o::wilor"])
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["jobs"]["h2o::wilor"]["count"], 2)
            self.assertEqual(saved["jobs"]["h2o::wilor"]["progress_source"], "prediction_dirs")
            self.assertNotIn("count", saved["jobs"]["h2o::hawor"])

    def test_refresh_marks_registered_exited_job_for_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _, state_path = fixture_registry(Path(directory))
            registry["run_template"]["ssh"] = {"host": "example.invalid", "key": "~/.ssh/test"}
            with mock.patch.object(taskctl, "remote_progress", return_value={
                "h2o::wilor": {"process_status": "exited"}
            }):
                inspected = taskctl.inspect_run(registry, "run-exact-1")
            self.assertEqual(inspected["method_states"]["wilor"], "queue_exited_needs_audit")
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["jobs"]["h2o::wilor"]["status"], "queue_exited_needs_audit")

    def test_inspect_promotes_audited_completion_artifacts_to_done(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _, state_path = fixture_registry(Path(directory))
            registry["run_template"]["ssh"] = {"host": "example.invalid", "key": "~/.ssh/test"}
            registry["runs"]["run-exact-1"]["completion_artifacts"] = [
                {"path": "/cache/index.jsonl", "min_lines": 3}
            ]
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["jobs"]["h2o::wilor"]["status"] = "queue_exited_needs_audit"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with mock.patch.object(taskctl, "remote_progress", return_value={
                "h2o::wilor": {"process_status": "exited", "status": "done"}
            }) as probe:
                inspected = taskctl.inspect_run(registry, "run-exact-1")
            self.assertEqual(inspected["method_states"]["wilor"], "done")
            self.assertEqual(probe.call_args.args[2][0]["completion_artifacts"][0]["min_lines"], 3)

    def test_formal_contact_audit_uses_registered_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _, state_path = fixture_registry(Path(directory))
            registry["run_template"]["ssh"] = {"host": "example.invalid", "key": "~/.ssh/test"}
            registry["logical_tasks"]["formal:h2o:standard2:60f"]["phase"] = "formal-contact"
            registry["runs"]["run-exact-1"]["prediction_root"] = "/results/h2o"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["jobs"]["h2o::wilor"]["status"] = "queue_exited_needs_audit"
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with mock.patch.object(taskctl, "remote_progress", return_value={
                "h2o::wilor": {"process_status": "exited", "status": "done", "count": 3}
            }) as probe:
                inspected = taskctl.inspect_run(registry, "run-exact-1")
            self.assertEqual(inspected["method_states"]["wilor"], "done")
            self.assertEqual(probe.call_args.args[2][0]["target_windows"], 3)

    def test_taco_contact_profile_pairs_methods_on_one_gpu_shard(self) -> None:
        registry = {
            "method_sets": {"contact2": ["s2contact", "contactopt"]},
            "logical_tasks": {"taco-contact": {
                "dataset": "taco", "method_set": "contact2", "phase": "formal-contact",
                "protocol": "60f", "latest": "run",
            }},
            "runs": {"run": {
                "logical_task_id": "taco-contact", "output_root": "/results/shard0",
                "target_windows_per_method": 134,
                "identity": {"pipeline": "taco_contact_3shard_v1", "physical_gpu": "4", "shard": "0/3"},
            }},
        }
        run = taskctl.merged_run(registry, "run")
        self.assertEqual(run["launch"]["gpu_candidates"], [4])
        self.assertIn("--baseline s2contact", run["launch"]["command"])
        self.assertIn("--baseline contactopt", run["launch"]["command"])
        self.assertEqual(run["completion_artifacts"][0]["path"], "/results/shard0/COMPLETE")

    def test_hoi4d_contact_profile_uses_six_shards_on_registered_node(self) -> None:
        registry = {
            "method_sets": {"contact2": ["s2contact", "contactopt"]},
            "logical_tasks": {"hoi4d-contact": {
                "dataset": "hoi4d", "method_set": "contact2", "phase": "formal-contact",
                "protocol": "60f-461", "latest": "run",
            }},
            "runs": {"run": {
                "logical_task_id": "hoi4d-contact", "output_root": "/results/shard0",
                "target_windows_per_method": 77,
                "identity": {"pipeline": "hoi4d_contact_6shard_v1", "physical_gpu": "2",
                             "physical_node": "5001", "shard": "0/6"},
            }},
        }
        run = taskctl.merged_run(registry, "run")
        self.assertEqual(run["launch"]["node_candidates"], [5001])
        self.assertEqual(run["launch"]["gpu_candidates"], [2])
        self.assertEqual(run["launch"]["values"]["num_shards"], 6)
        self.assertIn("s2_right_hoi4d_461.pkl", run["launch"]["values"]["s2_cache"])

    def test_taco_contact_metrics_profile_uses_all_three_prediction_shards(self) -> None:
        registry = {
            "method_sets": {"contact2_metrics": ["s2contact", "contactopt"]},
            "logical_tasks": {"taco-metrics": {
                "dataset": "taco", "method_set": "contact2_metrics",
                "phase": "metrics-contact-v1", "protocol": "60f-400-v1", "latest": "run",
            }},
            "runs": {"run": {
                "logical_task_id": "taco-metrics", "output_root": "/results/taco-metrics",
                "target_windows_per_method": 400,
                "identity": {"pipeline": "taco_contact_metrics_v1"},
            }},
        }
        run = taskctl.merged_run(registry, "run")
        self.assertEqual(run["launch"]["values"]["dataset"], "taco")
        self.assertEqual(run["launch"]["values"]["expected_windows"], 400)
        self.assertEqual(run["launch"]["command"].count("--prediction-root"), 6)
        self.assertEqual(run["completion_artifacts"][0]["path"], "/results/taco-metrics/report.json")

    def test_hoi4d_contact_metrics_profile_uses_all_six_prediction_shards(self) -> None:
        registry = {
            "method_sets": {"contact2_metrics": ["s2contact", "contactopt"]},
            "logical_tasks": {"hoi4d-metrics": {
                "dataset": "hoi4d", "method_set": "contact2_metrics",
                "phase": "metrics-contact-v1", "protocol": "60f-461-v1", "latest": "run",
            }},
            "runs": {"run": {
                "logical_task_id": "hoi4d-metrics", "output_root": "/results/hoi4d-metrics",
                "target_windows_per_method": 461,
                "identity": {"pipeline": "hoi4d_contact_metrics_v1"},
            }},
        }
        run = taskctl.merged_run(registry, "run")
        self.assertEqual(run["launch"]["values"]["dataset"], "hoi4d")
        self.assertEqual(run["launch"]["values"]["expected_windows"], 461)
        self.assertEqual(run["launch"]["command"].count("--prediction-root"), 12)
        self.assertEqual(run["completion_artifacts"][0]["path"], "/results/hoi4d-metrics/report.json")

    def test_hot3d_contact_metrics_profile_rewrites_existing_oss_gt_index(self) -> None:
        registry = {
            "method_sets": {"contact2_metrics": ["s2contact", "contactopt"]},
            "logical_tasks": {"hot3d-metrics": {
                "dataset": "hot3d", "method_set": "contact2_metrics",
                "phase": "metrics-contact-v1", "protocol": "60f-400-v1", "latest": "run",
            }},
            "runs": {"run": {
                "logical_task_id": "hot3d-metrics", "output_root": "/results/hot3d-metrics",
                "target_windows_per_method": 400,
                "identity": {"pipeline": "hot3d_contact_metrics_v1"},
            }},
        }
        run = taskctl.merged_run(registry, "run")
        self.assertEqual(run["launch"]["node_candidates"], [5001])
        self.assertEqual(run["launch"]["command"].count("--prediction-root"), 2)
        self.assertIn("--existing-gt-index", run["launch"]["command"])

    def test_arctic_contact_metrics_profile_combines_main_and_tail_predictions(self) -> None:
        registry = {
            "method_sets": {"contact2_metrics": ["s2contact", "contactopt"]},
            "logical_tasks": {"arctic-metrics": {
                "dataset": "arctic", "method_set": "contact2_metrics",
                "phase": "metrics-contact-v1", "protocol": "60f-434-v1", "latest": "run",
            }},
            "runs": {"run": {
                "logical_task_id": "arctic-metrics", "output_root": "/results/arctic-metrics",
                "target_windows_per_method": 434,
                "identity": {"pipeline": "arctic_contact_metrics_v1"},
            }},
        }
        run = taskctl.merged_run(registry, "run")
        self.assertEqual(run["launch"]["command"].count("--prediction-root"), 4)
        self.assertEqual(run["launch"]["values"]["expected_windows"], 434)
        self.assertIn("/arctic/gt_cache", run["launch"]["values"]["existing_gt_root"])


    def test_all_done_aggregate_job_reports_all_method_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _, state_path = fixture_registry(Path(directory))
            registry["runs"]["run-exact-1"]["job_keys"] = ["h2o::combined"]
            state_path.write_text(json.dumps({"jobs": {"h2o::combined": {"status": "done"}}}), encoding="utf-8")
            inspected = taskctl.inspect_run(registry, "run-exact-1")
            self.assertEqual(inspected["progress"], {"known_complete_windows": 6, "target_windows": 6})

    def test_inspect_reads_exact_handle_log_once_for_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _, state_path = fixture_registry(Path(directory))
            registry["run_template"]["ssh"] = {"host": "example.invalid", "key": "~/.ssh/test"}
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["jobs"]["h2o::wilor"].update({"status": "queue_exited_needs_audit"})
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with mock.patch.object(taskctl, "remote_progress", return_value={
                "h2o::wilor": {"process_status": "exited", "audit_log_tail": "ValueError: exact failure"}
            }) as probe:
                inspected = taskctl.inspect_run(registry, "run-exact-1")
            self.assertEqual(probe.call_count, 1)
            self.assertTrue(probe.call_args.args[2][0]["audit_log"])
            self.assertEqual(inspected["method_states"]["wilor"], "queue_exited_needs_audit")
            self.assertEqual(inspected["remote_sync"]["updated_jobs"], ["h2o::wilor"])
            self.assertIn("ValueError", inspected["audit_log_tails"]["wilor"])

    def test_inspect_includes_only_registered_storage_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _, _ = fixture_registry(Path(directory))
            registry["run_template"].update({
                "ssh": {"host": "example.invalid", "key": "~/.ssh/test"},
                "storage_probe": {"node": 5001, "mount": "/mnt/cpfs"},
            })
            response = mock.Mock(returncode=0, stdout="1B-blocks Used Available Use% Mounted on\n1000 750 250 75% /mnt/cpfs\n")
            with mock.patch.object(taskctl.subprocess, "run", return_value=response) as run:
                inspected = taskctl.inspect_run(registry, "run-exact-1", include_storage=True)
            self.assertEqual(inspected["storage"]["available_bytes"], 250)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-p") + 1], "5001")
            self.assertTrue(command[-1].endswith("-- /mnt/cpfs"))

    def test_pause_writes_only_exact_registered_job_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, _, _ = fixture_registry(root)
            controls = root / "controls.json"
            result = taskctl.set_desired_state(registry, "run-exact-1", "paused", controls)
            self.assertTrue(result["accepted"])
            payload = json.loads(controls.read_text(encoding="utf-8"))
            self.assertEqual(payload["jobs"], {"h2o::hawor": "paused", "h2o::wilor": "paused"})

    def test_pause_rejects_unregistered_legacy_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, _, _ = fixture_registry(root, legacy=True)
            with self.assertRaisesRegex(taskctl.TaskError, "UNCONTROLLED_LEGACY_JOB"):
                taskctl.set_desired_state(registry, "run-exact-1", "paused", root / "controls.json")

    def test_signal_ignores_ssh_banner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry, _, _ = fixture_registry(Path(directory))
            registry["run_template"].update({
                "ssh": {"host": "example.invalid", "key": "~/.ssh/test"},
                "remote_python": "python3",
                "remote_controller": "/deleted/control.py",
            })
            registry["runs"]["run-exact-1"]["launch"] = {"controller_path": "/registered/control.py"}
            payload = {"ok": True, "handles": {"/registered/h2o-wilor.json": {"status": "signal_sent"}}}
            response = mock.Mock(returncode=0, stdout="Welcome to DSW\n" + json.dumps(payload) + "\n")
            with mock.patch.object(taskctl.subprocess, "run", return_value=response) as run:
                result = taskctl.signal_registered_jobs(registry, "run-exact-1", "STOP")
            self.assertTrue(result["verified"])
            self.assertIn("/registered/control.py", run.call_args.args[0][-1])

    def test_scheduler_contains_no_process_or_filesystem_discovery(self) -> None:
        source = (Path(__file__).parents[1] / "auto_scheduler.py").read_text(encoding="utf-8")
        for forbidden in ("ps -eo", "pgrep", "find_live_pid", "fetch_process_listing", "find ", "awk ", "rg "):
            self.assertNotIn(forbidden, source)

    def test_remote_signal_uses_exact_registered_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handle = Path(directory) / "handle.json"
            handle.write_text(json.dumps({"pid": 17, "pgid": 17, "start_ticks": 91}), encoding="utf-8")
            identity = {"pid": 17, "pgid": 17, "start_ticks": 91, "state": "R"}
            with mock.patch.object(remote_task_control, "proc_identity", return_value=identity), \
                    mock.patch.object(remote_task_control.os, "killpg") as killpg:
                result = remote_task_control.send_signal([str(handle)], "STOP")
            killpg.assert_called_once_with(17, remote_task_control.signal.SIGSTOP)
            self.assertEqual(result["handles"][str(handle)]["status"], "signal_sent")

    def test_remote_status_exposes_registered_command_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handle = Path(directory) / "handle.json"
            handle.write_text(json.dumps({
                "pid": 17, "pgid": 17, "start_ticks": 91, "command_sha256": "abc"
            }), encoding="utf-8")
            identity = {"pid": 17, "pgid": 17, "start_ticks": 91, "state": "R"}
            with mock.patch.object(remote_task_control, "proc_identity", return_value=identity):
                result = remote_task_control.statuses([str(handle)])
            self.assertEqual(result["handles"][str(handle)]["command_sha256"], "abc")

    def test_stale_handle_is_rejected_without_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handle = Path(directory) / "handle.json"
            handle.write_text(json.dumps({"pid": 17, "pgid": 17, "start_ticks": 91}), encoding="utf-8")
            identity = {"pid": 17, "pgid": 17, "start_ticks": 92, "state": "R"}
            with mock.patch.object(remote_task_control, "proc_identity", return_value=identity), \
                    mock.patch.object(remote_task_control.os, "killpg") as killpg:
                with self.assertRaisesRegex(RuntimeError, "STALE_HANDLE"):
                    remote_task_control.send_signal([str(handle)], "STOP")
            killpg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
