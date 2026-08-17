from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


DEFAULT_SMOKE_SEQUENCE = "subject1_ego/o1/7"
DEFAULT_PILOT_SEQUENCES = (
    "subject3_ego/k2/0",
    "subject3_ego/o1/0",
    "subject3_ego/o2/0",
)
REQUIRED_H2O_ENTRIES = (
    "subject1_ego",
    "subject2_ego",
    "subject3_ego",
    "subject4_ego",
    "object",
    "label_split",
)


def validate_h2o_root(data_root: Path) -> Path:
    data_root = data_root.resolve()
    missing = [name for name in REQUIRED_H2O_ENTRIES if not (data_root / name).exists()]
    if missing:
        raise ValueError(f"H2O 数据根目录不完整: {data_root}; missing={missing}")
    if any(data_root.parent.glob("*.download-state")) or any(data_root.glob("*.download-state")):
        raise ValueError(f"H2O 数据根目录带未完成下载标记: {data_root}")
    return data_root


def sequence_rgb_paths(data_root: Path, sequence: str) -> list[Path]:
    rgb_dir = data_root / sequence / "cam4" / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"H2O RGB 目录不存在: {rgb_dir}")
    paths = sorted(path for path in rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise ValueError(f"H2O RGB 目录为空: {rgb_dir}")
    return paths


def build_windows(
    frame_ids: Iterable[str],
    *,
    window_size: int = 12,
    stride: int = 12,
    include_final_window: bool = True,
) -> list[list[str]]:
    ids = list(frame_ids)
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size 和 stride 必须为正整数")
    if len(ids) < window_size:
        raise ValueError(f"帧数 {len(ids)} 小于窗口长度 {window_size}")
    windows = [ids[start : start + window_size] for start in range(0, len(ids) - window_size + 1, stride)]
    if include_final_window:
        final_window = ids[-window_size:]
        if not windows or windows[-1] != final_window:
            windows.append(final_window)
    return windows


def _frame_list_digest(relative_paths: list[str]) -> str:
    payload = "\n".join(relative_paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sequence_manifest(data_root: Path, sequence: str, *, window_size: int, stride: int) -> dict[str, object]:
    paths = sequence_rgb_paths(data_root, sequence)
    relative_paths = [path.relative_to(data_root).as_posix() for path in paths]
    frame_ids = [path.stem for path in paths]
    windows = build_windows(frame_ids, window_size=window_size, stride=stride)
    return {
        "sequence": sequence,
        "frame_count": len(frame_ids),
        "first_frame_id": frame_ids[0],
        "last_frame_id": frame_ids[-1],
        "frame_list_sha256": _frame_list_digest(relative_paths),
        "windows": [
            {
                "window_id": f"{sequence.replace('/', '_')}_{window[0]}_{window[-1]}",
                "frame_ids": window,
            }
            for window in windows
        ],
    }


def formal_test_sequences(data_root: Path) -> tuple[str, ...]:
    """Return every complete Subject4 RGB episode in stable order."""
    subject_root = data_root / "subject4_ego"
    sequences = sorted(
        rgb_dir.parent.parent.relative_to(data_root).as_posix()
        for rgb_dir in subject_root.glob("*/*/cam4/rgb")
        if rgb_dir.is_dir()
    )
    if not sequences:
        raise ValueError(f"Subject4 test 序列为空: {subject_root}")
    return tuple(sequences)


def build_h2o_comparison_manifest(
    data_root: Path,
    *,
    window_size: int = 12,
    stride: int = 12,
) -> dict[str, object]:
    data_root = validate_h2o_root(data_root)
    smoke_paths = sequence_rgb_paths(data_root, DEFAULT_SMOKE_SEQUENCE)[:window_size]
    if len(smoke_paths) != window_size:
        raise ValueError("native smoke 无法取得连续 12 帧")
    return {
        "protocol_version": "h2o_comparison_v1",
        "schema_version": "egofound3r_comparison_output_v1",
        "dataset": "H2O",
        "input_modality": "RGB only",
        "source_resolution_hw": [720, 1280],
        "fps": 30.0,
        "window_size": window_size,
        "window_stride": stride,
        "include_final_window": True,
        "aggregation_unit": "sequence",
        "smoke": {
            "partition": "train",
            "sequence": DEFAULT_SMOKE_SEQUENCE,
            "frame_ids": [path.stem for path in smoke_paths],
        },
        "pilot": {
            "partition": "validation",
            "sequences": [
                _sequence_manifest(data_root, sequence, window_size=window_size, stride=stride)
                for sequence in DEFAULT_PILOT_SEQUENCES
            ],
        },
        "formal_test_gate": {
            "partition": "test",
            "subject": "subject4_ego",
            "requires_user_confirmation": True,
        },
        "formal_test": {
            "partition": "test",
            "sequences": [
                _sequence_manifest(data_root, sequence, window_size=window_size, stride=stride)
                for sequence in formal_test_sequences(data_root)
            ],
        },
        "rules": {
            "unsupported_outputs": "N/A",
            "missing_outputs_must_not_be_filled": True,
            "native_preprocessing_allowed": True,
            "test_gt_scale_recovery_for_raw_metric_forbidden": True,
            "alignment_scope": "window for window metrics; sequence for sequence metrics",
            "camera_convention": "OpenCV x-right y-down z-forward; camera_c2w maps camera to world",
            "contact_scope": "EgoFound3R only",
        },
    }


def write_manifest(manifest: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
