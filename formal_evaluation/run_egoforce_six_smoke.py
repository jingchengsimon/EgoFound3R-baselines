#!/usr/bin/env python3
"""Prepare and optionally run one strict 60-frame EgoForce smoke window per dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from formal_evaluation.common.schema import validate_comparison_output
from formal_evaluation.datasets.egofound3r_gt import DATASET_LOADERS, VENDORED_DATALOADER_ROOT, canonical_sequence_id, source_indices_for_window, validate_window_row
from formal_evaluation.datasets.six_dataset_gt_cache import window_cache_id
from formal_evaluation.datasets.window_inputs import WINDOW_INPUT_VERSION

EXPECTED_WINDOWS = {"h2o": 283, "taco": 400, "hot3d": 400, "oakink_v2": 400, "arctic": 434, "hoi4d": 461}


def _roots(values: list[str]) -> dict[str, str]:
    roots = dict(value.split("=", 1) for value in values)
    if set(roots) != set(DATASET_LOADERS):
        raise ValueError(f"--root must name exactly {sorted(DATASET_LOADERS)}")
    return roots


def _selected_rows(path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    counts = Counter(row.get("dataset") for row in rows)
    selected = []
    for dataset in DATASET_LOADERS:
        row = next(row for row in rows if row["dataset"] == dataset)
        _, _, frame_ids = validate_window_row(row)
        if len(frame_ids) != 60 or row.get("window_stride") != 60 or row.get("window_overlap") != 0:
            raise ValueError(f"{dataset}: selected row is not a strict 60-frame window")
        selected.append(row)
    return selected, dict(counts)


def _as_rgb(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"loader RGB shape is not HxWx3: {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0, 1) * 255)
    return image.astype(np.uint8, copy=False)


def _materialize_rgb_input(dataset: object, row: dict[str, object], root: Path) -> Path:
    name, sequence, frame_ids = validate_window_row(row)
    indices = source_indices_for_window(dataset, canonical_sequence_id(name, sequence), frame_ids)
    directory = root / "inputs" / name / window_cache_id(row)
    rgb_paths, intrinsics = [], []
    for index, (expected_id, source_index) in enumerate(zip(frame_ids, indices, strict=True)):
        sample = dataset[source_index]
        if str(sample["frame_id"]) != expected_id:
            raise RuntimeError(f"{name}: frame identity drift at {index}")
        K = sample.get("intrinsics")
        K = K.detach().cpu().numpy() if hasattr(K, "detach") else np.asarray(K)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"{name}/{expected_id}: missing usable rectified pinhole intrinsics")
        path = directory / "rgb" / f"{index:03d}_{expected_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_as_rgb(sample["rgb"]), mode="RGB").save(path)
        rgb_paths.append(str(path)); intrinsics.append(np.asarray(K, dtype=np.float64).tolist())
    record = {"window_input_version": WINDOW_INPUT_VERSION, "dataset": name, "sequence_id": sequence,
              "window_id": row["window_id"], "cache_id": window_cache_id(row), "frame_ids": frame_ids,
              "rgb_paths": rgb_paths, "intrinsics": intrinsics, "intrinsics_valid": [True] * len(frame_ids),
              "input_kind": "rgb_intrinsics_only_no_geometry"}
    target = directory / "window_input.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _redirect_loader_caches(output_root: Path) -> None:
    """Keep index/rectification caches out of read-only dataset roots."""
    from egohandmetric_prompt.data import datasets
    def cache(root: Path, name: str) -> Path:
        key = hashlib.sha256(f"{root}:{name}".encode()).hexdigest()[:16]
        return output_root / "loader_cache" / f"{name}_{key}.pkl"
    datasets._light_index_cache_path = cache
    datasets._hot3d_rectified_rgb_cache_path = lambda *args, **kwargs: None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); roots = _roots(args.root); rows, manifest_counts = _selected_rows(args.windows)
    if args.output_root.exists() and any(path.name not in {"runtime", "handle.json", "run.log"} for path in args.output_root.iterdir()):
        raise FileExistsError(f"output root must be new and empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True); sys.path.insert(0, str(VENDORED_DATALOADER_ROOT))
    _redirect_loader_caches(args.output_root)
    from egohandmetric_prompt.data.stages import build_named_frame_dataset
    prepared, summary = [], {"status": "prepared", "datasets": {}, "manifest_counts": manifest_counts,
                              "expected_manifest_counts": EXPECTED_WINDOWS,
                              "manifest_counts_match_expected": manifest_counts == EXPECTED_WINDOWS}
    for row in rows:
        dataset_name = str(row["dataset"])
        try:
            dataset = build_named_frame_dataset(DATASET_LOADERS[dataset_name], root_override=roots[dataset_name], split="all", load_rgb=True, load_depth=False)
            input_path = _materialize_rgb_input(dataset, row, args.output_root)
            prepared.append({"dataset": dataset_name, "window_input": str(input_path), "window_id": row["window_id"]})
            summary["datasets"][dataset_name] = {"status": "prepared", "window_input": str(input_path), "frame_count": 60}
        except Exception as error:
            summary["datasets"][dataset_name] = {"status": "failed_prepare", "error": repr(error), "traceback": traceback.format_exc()}
    (args.output_root / "prepared_inputs.json").write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n")
    if not args.prepare_only:
        if args.source_root is None or args.checkpoint is None:
            raise ValueError("inference requires --source-root and --checkpoint")
        adapter = Path(__file__).parent / "hand/adapters/run_egoforce_baseline.py"
        for item in prepared:
            name = item["dataset"]
            try:
                log = args.output_root / "logs" / f"{name}.log"
                log.parent.mkdir(exist_ok=True)
                with log.open("x") as stream:
                    subprocess.run([sys.executable, str(adapter), "--phase", "smoke", "--window-input", item["window_input"], "--source-root", str(args.source_root), "--checkpoint", str(args.checkpoint), "--output-root", str(args.output_root), "--device", args.device], check=True, stdout=stream, stderr=subprocess.STDOUT)
                target = args.output_root / "egoforce" / "smoke" / Path(item["window_input"]).parent.name
                metadata = json.loads((target / "metadata.json").read_text()); arrays = dict(np.load(target / "predictions.npz"))
                validate_comparison_output(metadata, arrays)
                if len(metadata["frame_ids"]) != 60 or metadata.get("processed_resolution_hw") != [256, 256] or not arrays["hand_valid"].any():
                    raise ValueError("smoke output lacks 60 frames, 256 metadata, or a finite detected hand")
                summary["datasets"][name] = {"status": "success", "prediction_dir": str(target), "valid_hands": int(arrays["hand_valid"].sum())}
            except Exception as error:
                summary["datasets"][name] = {"status": "failed_inference", "error": repr(error), "traceback": traceback.format_exc()}
        summary["status"] = "complete" if all(row.get("status") == "success" for row in summary["datasets"].values()) else "incomplete"
    (args.output_root / "smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["status"] == "complete":
        (args.output_root / "COMPLETE").write_text("six EgoForce RGB-only smoke windows passed\n")
    print(json.dumps(summary, ensure_ascii=False))
    if any(value.get("status", "").startswith("failed") for value in summary["datasets"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
