import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from formal_evaluation import run_egofound3r_stride_evaluation as runner
from formal_evaluation import run_ablation_dataset_smoke as smoke_runner


class SingleDatasetEvaluationTest(unittest.TestCase):
    def test_metric_smoke_does_not_complete_with_missing_predictions(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            folder = Path(directory)
            spec = {"datasets": {"h2o": {}}, **{key: key for key in (
                "methods_config", "config", "checkpoint", "backbone", "source_commit",
                "inference_commit", "checkpoint_sha256")}}
            gt = folder / "gt.jsonl"
            gt.write_text(json.dumps({"window_id": "w"}) + "\n")
            records = [{"window_id": "w", "cache_id": "c"}]
            stack.enter_context(patch.dict("os.environ", {"TASKCTL_GPU": "2"}))
            stack.enter_context(patch.object(smoke_runner, "prepare_inputs", return_value=(folder / "inputs", gt, records)))
            missing = [1]

            def execute(command, **kwargs):
                if "--runner" in command:
                    out = Path(command[command.index("--output-root") + 1])
                    out.mkdir()
                    (out / "smoke_summary.json").write_text(json.dumps({"status": "complete", "output": "prediction"}))
                    (out / "COMPLETE").write_text("complete\n")
                else:
                    out = Path(command[command.index("--report-path") + 1])
                    out.write_text(json.dumps({"gt_windows": 1, "methods": {"egofound3r": {
                        "missing_prediction_windows": missing[0], "datasets": {"h2o": {"n_windows": 1}}}}}))

            stack.enter_context(patch.object(smoke_runner.subprocess, "run", side_effect=execute))
            failed = folder / "failed"
            with self.assertRaisesRegex(ValueError, "metric coverage"):
                smoke_runner.run(spec, failed)
            self.assertFalse((failed / "COMPLETE").exists())
            missing[0] = 0
            good = folder / "good"
            smoke_runner.run(spec, good)
            self.assertTrue((good / "COMPLETE").exists())
            summary = json.loads((good / "smoke_summary.json").read_text())
            self.assertEqual(summary["dataset"], "h2o")
            self.assertEqual(summary["metric_validation"], "complete")
            with self.assertRaises(FileExistsError):
                smoke_runner.run(spec, good)

    def test_ablation_identity_single_gpu_and_legacy_smoke_gate(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            tmp_path = Path(directory)
            identity = {"source_commit": "ablation-training", "inference_commit": "ablation-inference",
                        "checkpoint_sha256": "ablation-checkpoint"}
            smoke = tmp_path / "smoke"
            smoke.mkdir()
            (smoke / "COMPLETE").write_text("complete\n")
            summary = {**identity, "status": "complete", "global_stride": 5, "global_anchor_phase": 2}
            (smoke / "smoke_summary.json").write_text(json.dumps(summary))
            methods = tmp_path / "ablation_methods.json"
            methods.write_text(json.dumps({"methods": {"egofound3r": identity}}))
            spec = {**identity, "methods_config": str(methods), "required_smoke_strides": [5],
                    "smoke_roots": [str(smoke)], "datasets": {"h2o": {}},
                    "config": "ablation.toml", "checkpoint": "step_004999.pt", "backbone": "backbone.pt"}
            spec_path = tmp_path / "spec.json"
            work = tmp_path / "work"
            (work / "h2o").mkdir(parents=True)
            output = tmp_path / "output"
            stack.enter_context(patch.dict("os.environ", {"TASKCTL_GPU": "3"}))
            stack.enter_context(patch("sys.argv", ["runner", "--spec", str(spec_path), "--output-root", str(output),
                                            "--work-root", str(work), "--global-stride", "5"]))
            record = {"window_id": "w", "cache_id": "c"}
            stack.enter_context(patch.object(runner, "prepare_inputs", return_value=(tmp_path / "inputs", tmp_path / "gt", [record])))
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                if "--runner" in command:
                    assert command[command.index("--gpus") + 1] == "3"
                    assert "--runner-arg=step_004999.pt" in command
                    root = output / "h2o/egofound3r/formal/c"
                    root.mkdir(parents=True)
                    (root / "metadata.json").write_text(json.dumps({**identity, "global_stride": 5, "global_anchor_phase": 2}))
                else:
                    (output / "h2o/report.json").write_text("{}")

            stack.enter_context(patch.object(runner.subprocess, "run", side_effect=run))
            # A stride-5 receipt must not satisfy the legacy default requiring strides 1 and 5.
            spec_path.write_text(json.dumps({k: v for k, v in spec.items() if k != "required_smoke_strides"}))
            with self.assertRaisesRegex(ValueError, "required smoke strides missing"):
                runner.main()
            assert not calls and not output.exists()
            spec_path.write_text(json.dumps(spec))
            methods.write_text(json.dumps({"methods": {"egofound3r": {**identity, "source_commit": "formal-model"}}}))
            with self.assertRaisesRegex(ValueError, "model identity mismatch"):
                runner.main()
            assert not calls and not output.exists()
            methods.write_text(json.dumps({"methods": {"egofound3r": identity}}))
            spec["require_metric_smoke"] = True
            spec_path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "dataset metric smoke not complete"):
                runner.main()
            metric_report = smoke / "metric_report.json"
            metric_report.write_text(json.dumps({"gt_windows": 1, "methods": {"egofound3r": {
                "missing_prediction_windows": 0, "datasets": {"h2o": {"n_windows": 1}}}}}))
            summary.update(dataset="h2o", metric_validation="complete", metric_report=str(metric_report))
            (smoke / "smoke_summary.json").write_text(json.dumps(summary))
            runner.main()
            assert len(calls) == 2  # One single-GPU queue, then the existing CPU metric evaluator.
            assert (output / "COMPLETE").is_file()
            assert list(json.loads((output / "summary.json").read_text())["reports"]) == ["h2o"]
            assert json.loads((work / "methods.json").read_text())["methods"]["egofound3r"] == identity

if __name__ == "__main__":
    unittest.main()
