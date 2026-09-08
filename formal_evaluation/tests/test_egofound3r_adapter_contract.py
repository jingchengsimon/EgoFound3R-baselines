import argparse
import ast
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import numpy as np

from formal_evaluation.run_egofound3r_smoke import _idle_gpu


def test_multirate_scene_uses_native_anchors_without_double_scaling():
    tree, filename = _adapter_ast()
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_multirate_scene_arrays")
    namespace = {"np": np, "_as_numpy": np.asarray}
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    camera = np.tile(np.eye(4), (1, 6, 1, 1))
    camera[..., 0, 3] = 7
    output = {"camera_pose_refined_high": camera,
              "camera_refined_valid_high": np.ones((1, 6), dtype=bool),
              "interpolation_scene_metric_scale_valid": np.array([True]),
              "interpolation_scene_metric_scale_factor": np.array([3.0]),
              "intrinsics_global": np.tile(np.eye(3), (1, 2, 1, 1)),
              # Official inference reconstruction has already applied the factor.
              "depth_global": np.full((1, 2, 2, 2, 1), 6.0),
              "depth_conf_global": np.ones((1, 2, 2, 2, 1))}
    frame_map = SimpleNamespace(global_anchor_indices=np.array([[1, 4]]),
                                global_frame_present=np.array([[True, True]]))
    poses, intrinsics, depth, confidence = namespace["_multirate_scene_arrays"](output, frame_map)
    assert np.all(poses[:, 0, 3] == 7)
    assert np.all(depth[[1, 4]] == 6)
    assert np.isnan(depth[[0, 2, 3, 5]]).all()
    assert np.isnan(intrinsics[[0, 2, 3, 5]]).all()
    assert np.all(confidence[[1, 4]] == 1)
    output["interpolation_scene_metric_scale_valid"][:] = False
    assert np.isnan(namespace["_multirate_scene_arrays"](output, frame_map)[2]).all()


def _adapter_ast():
    # Exercise the adapter without importing the deployment-only model runtime.
    path = Path(__file__).parents[1] / "scene/adapters/run_egofound3r_baseline.py"
    return ast.parse(path.read_text()), str(path)


def test_adapter_stride_parser_and_checkpoint_hash(tmp_path) -> None:
    tree, filename = _adapter_ast()
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"parse_args", "_verified_checkpoint_sha256"}]
    namespace = {
        "argparse": argparse,
        "Path": Path,
        "hashlib": hashlib,
        "INPUT_RESOLUTIONS_HW": {
            "384x512": (384, 512),
            "448x448": (448, 448),
            "512x512": (512, 512),
        },
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    required = ["adapter", "--phase", "formal"]
    for flag in ("methods-config", "config", "checkpoint", "backbone-checkpoint", "output-root"):
        required.extend([f"--{flag}", "unused"])
    required.extend(["--input-resolution", "448x448"])
    for stride in (None, 1, 2, 3, 4, 5):
        argv = required + ([] if stride is None else ["--global-stride", str(stride)])
        with mock.patch.object(sys, "argv", argv):
            assert namespace["parse_args"]().global_stride == (5 if stride is None else stride)
    for invalid in ("0", "6", "mixed"):
        with mock.patch.object(sys, "argv", required + ["--global-stride", invalid]):
            with pytest.raises(SystemExit):
                namespace["parse_args"]()
    for resolution in ("384x512", "448x448", "512x512"):
        argv = [*required[:-1], resolution]
        with mock.patch.object(sys, "argv", argv):
            assert namespace["parse_args"]().input_resolution == resolution
    with mock.patch.object(sys, "argv", [*required[:-1], "256x256"]):
        with pytest.raises(SystemExit):
            namespace["parse_args"]()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    verify = namespace["_verified_checkpoint_sha256"]
    assert verify(checkpoint, digest) == digest
    for expected in (None, "0" * 64):
        with pytest.raises(ValueError):
            verify(checkpoint, expected)


@pytest.mark.parametrize("stride", range(1, 6))
def test_stride_reaches_forward_contract_and_saved_provenance(stride) -> None:
    tree, filename = _adapter_ast()
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignments = {node.targets[0].id: node for node in main.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    contract_call = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)
                         and node.func.id == "_build_prediction_marker_forward_contract")
    forward = mock.Mock(return_value=({}, None))
    method = {"source_commit": "training", "source_tag": "tag", "inference_commit": "inference"}
    namespace = {"args": SimpleNamespace(global_stride=stride, checkpoint=Path("checkpoint.pt"),
                                           input_resolution="448x448"),
                 "project_config": object(), "frames": SimpleNamespace(dtype="torch.bfloat16"),
                 "_build_prediction_marker_forward_contract": forward,
                 "method_config": method, "checkpoint_sha256": "verified-sha",
                 "input_height": 448, "input_width": 448,
                 "model": object(), "marker_model_floating_dtype": mock.Mock(side_effect=ValueError("mixed after forward"))}
    setup = [assignments["global_stride"], assignments["global_anchor_phase"]]
    exec(compile(ast.Module(body=setup, type_ignores=[]), filename, "exec"), namespace)
    eval(compile(ast.Expression(contract_call), filename, "eval"), namespace)
    assert forward.call_args.kwargs["global_stride"] == stride
    assert forward.call_args.kwargs["global_anchor_phase"] == stride // 2
    assert forward.call_args.kwargs["batch"] == {}
    expected = {**method, "checkpoint": "checkpoint.pt", "checkpoint_sha256": "verified-sha",
                "model_compute_dtype": "torch.bfloat16", "global_stride": stride,
                "global_anchor_phase": stride // 2, "input_resolution": "448x448",
                "processed_resolution_hw": [448, 448],
                "image_preprocessing": "training_marker_runtime_collator_label_independent_center_crop"}
    metadata = assignments["metadata"].value
    namespace["metadata"] = {key.value: eval(compile(ast.Expression(value), filename, "eval"), namespace)
                             for key, value in zip(metadata.keys, metadata.values)
                             if isinstance(key, ast.Constant) and key.value in expected}
    assert namespace["metadata"] == expected
    run = assignments["run"].value
    provenance = next(value for key, value in zip(run.keys, run.values) if key.value == "provenance")
    assert eval(compile(ast.Expression(provenance), filename, "eval"), namespace) == expected


def test_adapter_uses_dynamic_multirate_bf16_contract() -> None:
    source = (
        Path(__file__).parents[1] / "scene/adapters/run_egofound3r_baseline.py"
    ).read_text(encoding="utf-8")
    for required in (
        "align_model_floating_dtype=True",
        "marker_model_floating_dtype",
        "_build_prediction_marker_forward_contract",
        "_call_marker_model",
        "reconstruct_multirate_metric_scene_outputs",
        "apply_hand_depth_scale",
        "HandDepthScaleOptions()",
        'parser.add_argument("--global-stride", type=int, choices=range(1, 6), default=5)',
        'parser.add_argument("--input-resolution", choices=tuple(INPUT_RESOLUTIONS_HW), required=True)',
        "global_stride=global_stride",
        "global_anchor_phase=global_anchor_phase",
        '"global_stride": global_stride',
        '"global_anchor_phase": global_anchor_phase',
        'outputs["in_view_probability"]',
        'outputs.get("root_translation_valid")',
        "load_media_ref",
        "MarkerRuntimeCollator",
        '"hand_annos": []',
        "crop_probability=1.0",
        'hasattr(MarkerRuntimeCollator, "_should_apply_center_crop")',
        '"preprocessing_uses_hand_annotations": False',
    ):
        assert required in source
    assert "cv2.resize" not in source
    assert "inference_egocentric=" not in source
    assert 'outputs["presence_mask"]' not in source
    assert 'outputs["presence_logits"]' not in source
    assert 'factor if key == "depth_global"' not in source


def test_label_independent_training_crop_and_eval_contract() -> None:
    adapter_source = (
        Path(__file__).parents[1] / "scene/adapters/run_egofound3r_baseline.py"
    ).read_text(encoding="utf-8")
    for resolution in ('"384x512": (384, 512)', '"448x448": (448, 448)', '"512x512": (512, 512)'):
        assert resolution in adapter_source
    assert '"hand_annos": []' in adapter_source
    assert "crop_probability=1.0" in adapter_source


def test_smoke_forwards_and_validates_stride_identity() -> None:
    source = (Path(__file__).parents[1] / "run_egofound3r_smoke.py").read_text(encoding="utf-8")
    for required in (
        'parser.add_argument("--global-stride", type=int, choices=range(1, 6), default=5)',
        '"--input-resolution", args.input_resolution',
        '"--global-stride", str(args.global_stride)',
        '"global_stride": args.global_stride',
        '"global_anchor_phase": args.global_stride // 2',
        '"input_resolution": args.input_resolution',
        '"--query-compute-apps=gpu_uuid"',
        "uuid not in busy",
    ):
        assert required in source


def test_smoke_accepts_driver_baseline_memory_when_no_compute_process_exists() -> None:
    responses = (
        SimpleNamespace(returncode=0, stdout="4, GPU-idle, 337\n5, GPU-busy, 400\n"),
        SimpleNamespace(returncode=0, stdout="GPU-busy\n"),
    )
    with mock.patch("formal_evaluation.run_egofound3r_smoke.subprocess.run", side_effect=responses):
        assert _idle_gpu("-1") == "4"
