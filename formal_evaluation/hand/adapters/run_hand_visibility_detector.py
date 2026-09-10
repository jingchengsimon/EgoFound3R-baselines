#!/usr/bin/env python3
"""Canonical joint-visibility adapter for hand-visibility-detector."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from formal_evaluation.common.io import write_comparison_output
from formal_evaluation.common.schema import SCHEMA_VERSION
from formal_evaluation.datasets.window_inputs import WINDOW_INPUT_VERSION

OFFICIAL_SOURCE_COMMIT = "6321d62fb7617cf504d1de6e4d9ecda7aadfb989"
OFFICIAL_CHECKPOINT_SHA256 = "c26712319e82fd5701d4eb1dd597d69469b770140e5fbf59252c6f08863b44cb"
INPUT_HW = (256, 256)


def _verified_sha256(path: Path, expected: str) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise ValueError(f"checkpoint SHA-256 mismatch: expected={expected}, actual={actual}")
    return actual


def _resize_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image.resize((INPUT_HW[1], INPUT_HW[0]), Image.Resampling.BILINEAR))


def _select_sides(results: list[Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """Keep the highest-confidence official detection for each canonical side."""
    selected: list[Any | None] = [None, None]  # left, right
    counts = [0, 0]
    for result in results:
        slot = 1 if bool(result.is_right) else 0
        counts[slot] += 1
        if selected[slot] is None or float(result.bbox_conf) > float(selected[slot].bbox_conf):
            selected[slot] = result
    values = np.full((2, 21), np.nan, dtype=np.float32)
    valid = np.zeros(2, dtype=bool)
    detail = []
    for slot, result in enumerate(selected):
        if result is None:
            detail.append({"slot": slot, "detections": counts[slot]})
            continue
        probability = np.asarray(result.visibility, dtype=np.float32)
        if probability.shape != (21,) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError(f"official visibility must be finite (21,) probabilities, got {probability.shape}")
        values[slot], valid[slot] = probability, True
        detail.append({"slot": slot, "detections": counts[slot], "selected_bbox_confidence": float(result.bbox_conf)})
    return values, valid, detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--window-input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=OFFICIAL_CHECKPOINT_SHA256)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hand-conf", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = json.loads(args.window_input.read_text())
    if record.get("window_input_version") != WINDOW_INPUT_VERSION:
        raise ValueError("Unsupported window input version")
    frame_ids, rgb_paths = record.get("frame_ids"), record.get("rgb_paths")
    if not isinstance(frame_ids, list) or not isinstance(rgb_paths, list) or not frame_ids or len(frame_ids) != len(rgb_paths):
        raise ValueError("Expected matching nonempty frame_ids and rgb_paths")
    for path in rgb_paths:
        if not Path(str(path)).is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_root / "hand_visibility_detector" / args.phase / str(record["cache_id"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    checkpoint_sha256 = _verified_sha256(args.checkpoint, args.checkpoint_sha256)
    from hand_visibility_detector import HandVisibilityPipeline
    pipeline = HandVisibilityPipeline(device=args.device, vis_checkpoint=str(args.checkpoint), hand_conf=args.hand_conf, crop_size=256)
    frame_ids = [str(value) for value in record["frame_ids"]]
    visibility = np.full((len(frame_ids), 2, 21), np.nan, dtype=np.float32)
    hand_valid = np.zeros((len(frame_ids), 2), dtype=bool)
    detection_log = []
    started = time.perf_counter()
    for index, path in enumerate(record["rgb_paths"]):
        values, valid, detail = _select_sides(pipeline.predict(_resize_rgb(Path(str(path)))))
        visibility[index], hand_valid[index] = values, valid
        detection_log.append(detail)
    metadata = {
        "schema_version": SCHEMA_VERSION, "method": "hand_visibility_detector", "phase": args.phase,
        "dataset": str(record["dataset"]), "sequence": str(record["sequence_id"]), "window_id": str(record["window_id"]),
        "frame_ids": frame_ids, "capabilities": {"hand_visibility": True, "hand_valid": True},
        "processed_resolution_hw": list(INPUT_HW),
        "runner_detail": {"official_source_commit": OFFICIAL_SOURCE_COMMIT, "checkpoint_sha256": checkpoint_sha256,
                          "image_preprocessing": "PIL_bilinear_full_frame_256x256_then_official_256_hand_crop",
                          "visibility": "official_probabilities_not_thresholded", "hand_slot_convention": "slot 0=left, slot 1=right",
                          "multi_detection_rule": "highest_bbox_confidence_per_side", "per_frame_detection_log": detection_log},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays = {"hand_visibility": visibility, "hand_valid": hand_valid}
    write_comparison_output(output_dir, metadata=metadata, arrays=arrays,
                            run={"status": "success", "elapsed_seconds": time.perf_counter() - started, "frame_count": len(frame_ids), "device": args.device},
                            native_metadata={"official_package": "hand_visibility_detector", "native_output": "HandResult.visibility"})
    print(json.dumps({"output_dir": str(output_dir), "valid_hands": int(hand_valid.sum())}))


if __name__ == "__main__":
    main()
