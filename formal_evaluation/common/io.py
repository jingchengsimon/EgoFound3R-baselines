from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .schema import validate_comparison_output


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("comparison manifest 顶层必须是 object")
    return manifest


def select_manifest_window(
    manifest: Mapping[str, object],
    *,
    phase: str,
    sequence: str | None = None,
    window_id: str | None = None,
) -> tuple[str, str, list[str]]:
    if phase == "smoke":
        smoke = manifest.get("smoke")
        if not isinstance(smoke, Mapping):
            raise ValueError("manifest 缺少 smoke")
        smoke_sequence = smoke.get("sequence")
        frame_ids = smoke.get("frame_ids")
        if not isinstance(smoke_sequence, str) or not isinstance(frame_ids, list):
            raise ValueError("manifest.smoke 格式错误")
        return smoke_sequence, f"{smoke_sequence.replace('/', '_')}_{frame_ids[0]}_{frame_ids[-1]}", list(frame_ids)

    if phase not in {"pilot", "formal"}:
        raise ValueError(f"未知 phase: {phase}")
    if sequence is None or window_id is None:
        raise ValueError(f"{phase} 必须同时指定 sequence 和 window_id")
    phase_key = "pilot" if phase == "pilot" else "formal_test"
    phase_manifest = manifest.get(phase_key)
    sequence_entries = phase_manifest.get("sequences") if isinstance(phase_manifest, Mapping) else None
    if not isinstance(sequence_entries, list):
        raise ValueError(f"manifest.{phase_key}.sequences 格式错误")
    for sequence_entry in sequence_entries:
        if not isinstance(sequence_entry, Mapping) or sequence_entry.get("sequence") != sequence:
            continue
        windows = sequence_entry.get("windows")
        if not isinstance(windows, list):
            break
        for window in windows:
            if isinstance(window, Mapping) and window.get("window_id") == window_id:
                frame_ids = window.get("frame_ids")
                if not isinstance(frame_ids, list) or not all(isinstance(item, str) for item in frame_ids):
                    raise ValueError(f"window frame_ids 格式错误: {window_id}")
                return sequence, window_id, list(frame_ids)
        break
    raise KeyError(f"manifest 中不存在 {phase} window: sequence={sequence}, window_id={window_id}")


def resolve_rgb_paths(data_root: Path, sequence: str, frame_ids: list[str]) -> list[Path]:
    rgb_dir = data_root / sequence / "cam4" / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    by_stem = {
        path.stem: path
        for path in rgb_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    missing = [frame_id for frame_id in frame_ids if frame_id not in by_stem]
    if missing:
        raise FileNotFoundError(f"RGB 帧不存在: sequence={sequence}, frame_ids={missing}")
    return [by_stem[frame_id] for frame_id in frame_ids]


def write_comparison_output(
    output_dir: Path,
    *,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    run: Mapping[str, object],
    native_metadata: Mapping[str, object] | None = None,
    native_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    validate_comparison_output(metadata, arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "predictions.npz", **arrays)
    (output_dir / "run.json").write_text(
        json.dumps(dict(run), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    native_dir = output_dir / "native"
    native_dir.mkdir(exist_ok=True)
    (native_dir / "metadata.json").write_text(
        json.dumps(dict(native_metadata or {}), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if native_arrays:
        np.savez_compressed(native_dir / "predictions.npz", **native_arrays)
