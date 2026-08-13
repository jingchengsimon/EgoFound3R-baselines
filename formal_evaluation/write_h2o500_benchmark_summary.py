#!/usr/bin/env python3
"""Merge isolated H2O-500 benchmark outputs into the required JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    "egofound3r", "wilor", "hawor", "dyn_hamr", "pad_hand", "reviv4d",
    "s2contact", "contactopt", "vggt", "pi3", "da3_large_1_1",
    "lingbot_map_long", "vggt_omega", "interactvlm",
)


AUDIT = {
    "egofound3r": ("EgoFound3R", "infer_marker_video.py-compatible runtime", "step_006099_newflow.pt", "egofound3r", "success: real neural-network forward"),
    "wilor": ("hand", "formal_evaluation/hand/adapters/run_wilor_baseline.py", "wilor_final.ckpt + detector.pt", "egofound3r", "blocked: WiLoR/mano_data/MANO_RIGHT.pkl missing"),
    "hawor": ("hand pipeline", "formal_evaluation/hand/adapters/run_hawor_baseline.py", "hawor.ckpt + infiller.pt + DROID/Metric3D", "egofound3r", "blocked: source-local DROID and Metric3D assets missing"),
    "dyn_hamr": ("hand pipeline", "formal_evaluation/benchmark_dyn_hamr_500.py", "HaMeR + YOLO + DROID-SLAM + MANO", "dyn_hamr", "full RGB pipeline runner implemented; DSW 500-frame validation pending"),
    "pad_hand": ("hand pipeline", "formal_evaluation/hand/adapters/run_pad_hand_baseline.py", "checkpoints/pad_hand.pt", "pad_hand_h20", "blocked: no verified pad_hand_h20 runtime"),
    "reviv4d": ("scene + hand", "formal_evaluation/scene/adapters/run_reviv4d_baseline.py", "ReViV + Cosmos DV8", "unverified", "blocked: Cosmos decoder.jit missing"),
    "s2contact": ("contact", "formal_evaluation/contact/adapters/run_s2_contactopt.py", "S2Contact-20211027-212322.pt", "contactopt", "blocked for RGB speed: adapter consumes H2O cache, not raw RGB"),
    "contactopt": ("contact", "formal_evaluation/contact/adapters/run_s2_contactopt.py", "ContactOpt-deepcontact_checkpoint.pt", "contactopt", "blocked for RGB speed: adapter consumes H2O cache, not raw RGB"),
    "vggt": ("scene", "formal_evaluation/scene/adapters/run_scene_baseline.py", "VGGT-1B/model.pt", "egofound3r", "success: real neural-network forward"),
    "pi3": ("scene", "formal_evaluation/scene/adapters/run_scene_baseline.py", "Pi3/model.safetensors", "egofound3r", "success: real neural-network forward"),
    "da3_large_1_1": ("scene", "formal_evaluation/scene/adapters/run_scene_baseline.py", "DA3-LARGE-1.1/model.safetensors", "egofound3r", "success: real neural-network forward"),
    "lingbot_map_long": ("scene", "formal_evaluation/scene/adapters/run_scene_baseline.py", "lingbot-map-long.pt", "egofound3r", "blocked: PyTorch 2.4.1 lacks torch.nn.attention.flex_attention"),
    "vggt_omega": ("scene", "formal_evaluation/scene/adapters/run_scene_baseline.py", "vggt_omega_1b_512.pt", "egofound3r", "success: real neural-network forward"),
    "interactvlm": ("contact", "scripts/eval_h2o_interactvlm.py", "N/A", "interactvlm_eval", "blocked: cached predictions only; no executable inference entrypoint verified"),
}

PARAMETERS = {
    "s2contact": {"modules": {"DeepContactNet": 587658}, "unique_total": 587658, "rule": "CPU checkpoint loaded; requires_grad unique storage"},
    "contactopt": {"modules": {"DeepContactNet": 1424138}, "unique_total": 1424138, "rule": "CPU checkpoint loaded; requires_grad unique storage"},
}

BLOCKERS = {
    "wilor": "`/mnt/workspace/sjc/EgoFound3R-baselines/WiLoR/mano_data/MANO_RIGHT.pkl` absent; official loader raised `AssertionError`.",
    "hawor": "`HaWoR/thirdparty/DROID-SLAM/droid.pth` and `HaWoR/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth` absent; central copies were not linked into source.",
    "dyn_hamr": "This legacy three-JSON summary does not ingest the dedicated Dyn-HaMR result yet; use benchmark_dyn_hamr_500.py and never substitute optimization-stage FPS.",
    "pad_hand": "`pad_hand_h20` was requested by the adapter but no corresponding verified interpreter/environment exists under `/mnt/workspace/sjc/envs`.",
    "reviv4d": "`/mnt/workspace/sjc/external/reviv4d/Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit` is absent.",
    "s2contact": "The only verified H2O adapter requires a prebuilt `s2_right_h2o_30724.pkl` contact cache, not raw RGB; cache throughput was deliberately not measured.",
    "contactopt": "The only verified H2O adapter requires a prebuilt `contactopt_right_h2o_30724.pkl` contact cache, not raw RGB; cache throughput was deliberately not measured.",
    "lingbot_map_long": "Import failed with `ModuleNotFoundError: torch.nn.attention.flex_attention` under the available PyTorch 2.4.1; FlashInfer is also unavailable.",
    "interactvlm": "No `/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/InteractVLM` or executable checkpoint/entrypoint was present; cached prediction evaluation is not inference.",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--wilor-json", type=Path, required=True)
    parser.add_argument("--egofound3r-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scene, wilor, ego = (_load(path) for path in (args.scene_json, args.wilor_json, args.egofound3r_json))
    measurements = {**scene["methods"], **wilor["methods"], **ego["methods"]}
    rows = {}
    for method in METHODS:
        category, runner, checkpoint, environment, status = AUDIT[method]
        row = {"category": category, "runner": runner, "checkpoint": checkpoint, "environment": environment, "status": status}
        if measurements.get(method, {}).get("status", "").startswith("success"):
            row["measurement"] = measurements[method]
        if method in PARAMETERS:
            row["parameter_count"] = PARAMETERS[method]
        elif "measurement" in row:
            row["parameter_count"] = row["measurement"]["parameter_count"]
        else:
            row["parameter_count"] = {"modules": "N/A", "unique_total": "N/A", "rule": BLOCKERS.get(method, "N/A")}
        if method in BLOCKERS:
            row["blocker"] = BLOCKERS[method]
        rows[method] = row
    result = {
        "protocol": {
            "data_root": "/mnt/workspace/sjc/DATA/H2O/h2o_data",
            "split": "test (subject4_ego)",
            "selection": "sorted eligible sequence IDs; random.Random(0).choice; first 500 sorted original RGB frame IDs",
            "seed": 0,
            "sequence": ego["sequence"],
            "frame_ids": ego["frame_ids"],
            "timing_boundary": scene["timing_boundary"],
            "trials": "2 warm-ups plus 5 CUDA-synchronized formal trials; P90 is nearest-rank (max for n=5)",
            "gpu": "NVIDIA H20, node 6001, physical GPU7 (CUDA_VISIBLE_DEVICES=7)",
        },
        "methods": rows,
        "artifact_sources": {
            "scene": str(args.scene_json), "wilor": str(args.wilor_json), "egofound3r": str(args.egofound3r_json),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "h2o_test500_benchmark.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# H2O test 500-frame online-inference benchmark", "",
        f"- Sequence: `{result['protocol']['sequence']}`, seed=0, 500 original frames `{result['protocol']['frame_ids'][0]}`–`{result['protocol']['frame_ids'][-1]}`.",
        f"- GPU: {result['protocol']['gpu']}.",
        "- Timing: models and official-form in-memory inputs were ready before timing; image read/decode, preprocessing, checkpoint load, metrics, saving, and visualization are excluded. No cached-prediction throughput appears in the FPS table.",
        "", "## Runability", "", "| Method | Category | Runner | Checkpoint | Environment | Status |", "|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        row = rows[method]
        lines.append("| " + " | ".join([method, *(_cell(row[key]) for key in ("category", "runner", "checkpoint", "environment", "status"))]) + " |")
    lines += ["", "## Main speed", "", "| Method | Median s | Mean s | P90 s | 500-output FPS | Model input frames | Windows/chunks | Peak GPU bytes |", "|---|---:|---:|---:|---:|---:|---|---:|"]
    for method in METHODS:
        measurement = rows[method].get("measurement")
        if measurement:
            lines.append(f"| {method} | {measurement['median_seconds']:.6f} | {measurement['mean_seconds']:.6f} | {measurement['p90_seconds']:.6f} | {measurement['output_fps']:.6f} | {measurement['model_input_frames']} | {_cell(measurement['strategy'])} | {measurement['peak_gpu_memory_bytes']} |")
        else:
            lines.append(f"| {method} | N/A | N/A | N/A | N/A | N/A | {_cell(rows[method]['status'])} | N/A |")
    lines += ["", "## Parameters", "", "| Method | Modules | Unique total | Counting rule |", "|---|---|---:|---|"]
    for method in METHODS:
        parameter = rows[method]["parameter_count"]
        modules = parameter["modules"]
        if isinstance(modules, dict):
            modules = "; ".join(f"{name}={count}" for name, count in modules.items())
        lines.append(f"| {method} | {_cell(modules)} | {_cell(parameter['unique_total'])} | {_cell(parameter['rule'])} |")
    lines += ["", "## Five formal trials", ""]
    for method in METHODS:
        measurement = rows[method].get("measurement")
        if measurement:
            lines.append(f"- `{method}`: " + ", ".join(f"{seconds:.6f}" for seconds in measurement["trial_seconds"]) + " s")
            if "stage_trial_seconds" in measurement:
                for stage, trials in measurement["stage_trial_seconds"].items():
                    lines.append(f"  - `{stage}`: " + ", ".join(f"{seconds:.6f}" for seconds in trials) + " s")
    lines += ["", "## Blocked methods", ""]
    lines += [f"- `{method}`: {reason}" for method, reason in BLOCKERS.items()]
    lines += ["", "CHOI is intentionally excluded from this fixed 14-method list.", ""]
    (args.output_dir / "h2o_test500_benchmark.md").write_text("\n".join(lines))
    print(json.dumps({"json": str(json_path), "markdown": str(args.output_dir / "h2o_test500_benchmark.md")}, indent=2))


if __name__ == "__main__":
    main()
