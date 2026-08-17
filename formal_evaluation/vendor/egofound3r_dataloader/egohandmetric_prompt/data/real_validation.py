from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from egohandmetric_prompt.configs import load_project_config
from egohandmetric_prompt.data.base import TemporalChunkDataset
from egohandmetric_prompt.data.stages import build_named_frame_dataset
from egohandmetric_prompt.data.training import (
    DEFAULT_HUMAN_MODEL_ROOT,
    ManoVertexBuilder,
    MarkerBatchCollator,
    MIDTRAIN_DATASET_NAMES,
    normalize_hand_target,
)


MARKER_SIGNALS = ("bbox", "joints_2d", "joints_3d", "mano")
THREE_R_SIGNALS = ("depth", "intrinsics", "camera_pose")


@dataclass(slots=True)
class DatasetValidationResult:
    dataset_name: str
    num_samples_checked: int = 0
    num_chunks_checked: int = 0
    loader_ok: bool = False
    sample_decode_ok: bool = False
    chunk_decode_ok: bool = False
    mano_vertices_ok: bool = False
    marker_signal_counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in MARKER_SIGNALS})
    three_r_signal_counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in THREE_R_SIGNALS})
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_failure(result: DatasetValidationResult, stage: str, error: Exception) -> None:
    result.failures.append(f"{stage}: {type(error).__name__}: {error}")


def _count_sample_signals(result: DatasetValidationResult, sample: dict[str, Any]) -> None:
    if sample.get("depth") is not None or sample.get("has_depth_gt"):
        result.three_r_signal_counts["depth"] += 1
    if sample.get("intrinsics") is not None:
        result.three_r_signal_counts["intrinsics"] += 1
    if sample.get("camera_pose") is not None:
        result.three_r_signal_counts["camera_pose"] += 1

    for hand in sample.get("hand_annos", []):
        normalized_hand = normalize_hand_target(hand)
        if normalized_hand is None:
            continue
        if normalized_hand.get("bbox_xyxy") is not None:
            result.marker_signal_counts["bbox"] += 1
        if normalized_hand.get("joints_2d") is not None:
            result.marker_signal_counts["joints_2d"] += 1
        if normalized_hand.get("joints_3d") is not None:
            result.marker_signal_counts["joints_3d"] += 1
        if normalized_hand.get("mano_valid"):
            result.marker_signal_counts["mano"] += 1


def _build_mano_builder(
    human_model_root: str | Path | None = None,
) -> tuple[ManoVertexBuilder | None, Exception | None]:
    try:
        model_root = DEFAULT_HUMAN_MODEL_ROOT if human_model_root is None else Path(human_model_root)
        return ManoVertexBuilder(model_root), None
    except Exception as error:
        return None, error


def _try_validate_mano_vertices(
    result: DatasetValidationResult,
    sample: dict[str, Any],
    mano_builder: ManoVertexBuilder | None,
) -> None:
    if mano_builder is None or result.mano_vertices_ok:
        return
    if not sample.get("has_mano"):
        return
    for hand in sample.get("hand_annos", []):
        normalized_hand = normalize_hand_target(hand)
        if normalized_hand is None or not normalized_hand.get("mano_valid"):
            continue
        vertices = mano_builder.build_vertices(normalized_hand, marker_vertex_ids=[0, 1, 2])
        if vertices is not None:
            result.mano_vertices_ok = True
            return


def validate_named_dataset(
    name: str,
    *,
    num_samples: int,
    num_frames: int,
    output_dir: str | Path,
    split: str = "train",
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> DatasetValidationResult:
    del output_dir
    result = DatasetValidationResult(dataset_name=name)
    try:
        config = load_project_config(config_path)
        root_override = config.paths.dataset_root_overrides.get(name)
        dataset = build_named_frame_dataset(
            name,
            split=split,
            load_rgb=True,
            load_depth=True,
            data_root=data_root or (config.paths.data_root or None),
            root_override=root_override,
        )
        result.loader_ok = True
    except Exception as error:
        _record_failure(result, "loader", error)
        return result

    dataset_length = len(dataset)
    if dataset_length == 0:
        _record_failure(result, "sample", ValueError("empty dataset"))
        result.sample_decode_ok = False
        result.chunk_decode_ok = False
        return result

    mano_builder, mano_builder_error = _build_mano_builder(config.paths.human_model_root or None)
    sample_count = min(dataset_length, num_samples)
    sample_ok = True
    for index in range(sample_count):
        try:
            sample = dataset[index]
        except Exception as error:
            sample_ok = False
            _record_failure(result, f"sample[{index}]", error)
            continue
        result.num_samples_checked += 1
        try:
            _count_sample_signals(result, sample)
            _try_validate_mano_vertices(result, sample, mano_builder)
        except Exception as error:
            sample_ok = False
            _record_failure(result, f"sample[{index}]", error)
            continue
    result.sample_decode_ok = sample_ok and result.num_samples_checked == sample_count
    if (
        mano_builder is None
        and mano_builder_error is not None
        and (
            result.marker_signal_counts["mano"] > 0
            or not isinstance(mano_builder_error, FileNotFoundError)
        )
    ):
        _record_failure(result, "mano_builder", mano_builder_error)

    chunk_ok = True
    try:
        chunk_dataset = TemporalChunkDataset(dataset, num_frames=num_frames, drop_last=False)
    except Exception as error:
        _record_failure(result, "chunk_loader", error)
        result.chunk_decode_ok = False
        return result

    chunk_count = min(len(chunk_dataset), num_samples)
    if chunk_count == 0:
        _record_failure(result, "chunk", ValueError("empty chunk dataset"))
        result.chunk_decode_ok = False
        return result
    collator = MarkerBatchCollator(stage="midtrain")
    for index in range(chunk_count):
        try:
            chunk = chunk_dataset[index]
            collator([chunk])
        except Exception as error:
            chunk_ok = False
            _record_failure(result, f"chunk[{index}]", error)
            continue
        result.num_chunks_checked += 1
    result.chunk_decode_ok = chunk_ok and result.num_chunks_checked == chunk_count
    return result


def write_validation_reports(
    results: list[DatasetValidationResult],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_json = output_path / "summary.json"
    summary_json.write_text(json.dumps([result.to_dict() for result in results], indent=2), encoding="utf-8")

    lines = [
        "# Dataset Validation Summary",
        "",
        "| dataset | loader_ok | sample_decode_ok | chunk_decode_ok | mano_vertices_ok | failures |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| {dataset} | {loader_ok} | {sample_decode_ok} | {chunk_decode_ok} | {mano_vertices_ok} | {failures} |".format(
                dataset=result.dataset_name,
                loader_ok=result.loader_ok,
                sample_decode_ok=result.sample_decode_ok,
                chunk_decode_ok=result.chunk_decode_ok,
                mano_vertices_ok=result.mano_vertices_ok,
                failures=len(result.failures),
            )
        )
        lines.extend(
            [
                "",
                f"## {result.dataset_name}",
                "",
                f"- loader_ok: {result.loader_ok}",
                f"- sample_decode_ok: {result.sample_decode_ok}",
                f"- chunk_decode_ok: {result.chunk_decode_ok}",
                f"- mano_vertices_ok: {result.mano_vertices_ok}",
                f"- failures: {len(result.failures)}",
            ]
        )
    (output_path / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json, output_path / "summary.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(MIDTRAIN_DATASET_NAMES))
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/real_dataset_validation"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)

    results = [
        validate_named_dataset(
            dataset_name,
            num_samples=args.num_samples,
            num_frames=args.num_frames,
            output_dir=args.output_dir,
            config_path=args.config,
            data_root=args.data_root,
        )
        for dataset_name in args.datasets
    ]
    write_validation_reports(results, args.output_dir)
    return 1 if any(result.failures for result in results) else 0
