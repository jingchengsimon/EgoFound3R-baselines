from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from formal_evaluation.run_egofound3r_smoke import _idle_gpu


def test_adapter_uses_dynamic_multirate_bf16_contract() -> None:
    source = (
        Path(__file__).parents[1] / "scene/adapters/run_egofound3r_baseline.py"
    ).read_text(encoding="utf-8")
    for required in (
        "align_model_floating_dtype=True",
        "marker_model_floating_dtype",
        "_build_prediction_marker_forward_contract",
        "_call_marker_model",
        'parser.add_argument("--global-stride", type=int, choices=range(1, 6))',
        "global_stride=global_stride",
        "global_anchor_phase=global_anchor_phase",
        '"global_stride": global_stride',
        '"global_anchor_phase": global_anchor_phase',
        'outputs["in_view_probability"]',
        'outputs.get("root_translation_valid")',
    ):
        assert required in source
    assert "inference_egocentric=" not in source
    assert 'outputs["presence_mask"]' not in source
    assert 'outputs["presence_logits"]' not in source


def test_smoke_forwards_and_validates_stride_identity() -> None:
    source = (Path(__file__).parents[1] / "run_egofound3r_smoke.py").read_text(encoding="utf-8")
    for required in (
        'parser.add_argument("--global-stride", type=int, choices=range(1, 6), default=5)',
        '"--global-stride", str(args.global_stride)',
        '"global_stride": args.global_stride',
        '"global_anchor_phase": args.global_stride // 2',
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
