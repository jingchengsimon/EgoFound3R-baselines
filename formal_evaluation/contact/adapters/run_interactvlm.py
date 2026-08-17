#!/usr/bin/env python3
"""Run official InteractVLM 3D human-contact inference from a dataset manifest."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _safe(value: object) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True,
                        help="JSONL rows with rgb_path/input_path, object_name, sequence and frame_id")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--vision-tower",
        type=Path,
        help="local CLIP vision-tower snapshot passed to the official runner for offline execution",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    entrypoint = source_root / "run_demo.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"official InteractVLM entrypoint missing: {entrypoint}")
    if args.work_dir.exists():
        raise FileExistsError(f"refusing to reuse work directory: {args.work_dir}")

    input_dir = args.work_dir / "inputs"
    input_dir.mkdir(parents=True)
    rows = []
    seen = set()
    with args.input_manifest.open() as handle:
        for order, line in enumerate(handle):
            row = json.loads(line)
            rgb_value = row.get("rgb_path", row.get("input_path"))
            if not rgb_value or not row.get("object_name"):
                raise ValueError(f"manifest row {order} lacks RGB path or object_name")
            rgb = Path(rgb_value).resolve(strict=True)
            identity = (str(row.get("sequence", "")), str(row.get("frame_id", order)))
            if identity in seen:
                raise ValueError(f"duplicate sequence/frame in manifest: {identity}")
            seen.add(identity)
            stem = (f"{_safe(row['object_name']).lower()}__{order:06d}_"
                    f"{_safe(identity[0])}_{_safe(identity[1])}")
            link = input_dir / f"{stem}{rgb.suffix.lower()}"
            link.symlink_to(rgb)
            rows.append({**row, "adapter_order": order, "adapter_stem": stem,
                         "rgb_path": str(rgb), "prepared_input": str(link)})
    if not rows:
        raise ValueError("input manifest is empty")

    command = [
        args.python, str(entrypoint), "--version", str(checkpoint),
        "--img_folder", str(input_dir), "--contact_type", "hcontact",
        "--input_mode", "file", "--precision", args.precision,
    ]
    if args.vision_tower is not None:
        command.extend(("--vision-tower", str(args.vision_tower.resolve(strict=True))))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    subprocess.run(command, cwd=source_root, env=env, check=True)

    native_dir = input_dir / "contact_output"
    output_rows = []
    for row in rows:
        prediction = native_dir / f"{row['adapter_stem']}_hcontact_vertices.npz"
        if not prediction.is_file():
            raise FileNotFoundError(f"InteractVLM prediction missing: {prediction}")
        output_rows.append({**row, "prediction_path": str(prediction),
                            "prediction_field": "pred_contact_3d_smplh",
                            "prediction_vertices": 6890,
                            "checkpoint": str(checkpoint)})
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w") as handle:
        for row in output_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"status": "success: official InteractVLM inference",
                      "frames": len(output_rows),
                      "output_manifest": str(args.output_manifest)}))


if __name__ == "__main__":
    main()
