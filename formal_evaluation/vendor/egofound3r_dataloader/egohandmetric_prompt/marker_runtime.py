from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import math
import os
import pickle
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

from egohandmetric_prompt.configs import (
    MarkerRuntimeConfig,
    ProjectConfig,
    RandomCropResizeConfig,
    default_human_model_root_path,
)
from egohandmetric_prompt.data import (
    BaseFrameDataset,
    CombinedFrameDataset,
    DualStreamBatchMixer,
    FilteredRandomLengthTemporalChunkDataset,
    FilteredStageDataset,
    ManoVertexBuilder,
    MANAGED_SEQUENCE_SPLIT_DATASETS,
    MarkerBatchCollator,
    MissingSampleCollator,
    MissingSampleRetryDataset,
    MultiSourceRandomLengthTemporalChunkDataset,
    RandomLengthBatchSampler,
    RandomLengthEpochBatchSampler,
    RandomLengthTemporalChunkDataset,
    RankShardedBatchSampler,
    StageDatasetBundle,
    TemporalChunkDataset,
    WeightedRandomLengthBatchSampler,
    build_stage_chunk_bundle,
    build_named_frame_dataset,
    chunk_allows_fixed_query_marker_supervision,
)
from egohandmetric_prompt.data.marker_mesh import marker_faces_for_count
from egohandmetric_prompt.data.marker_vertices import marker_vertex_ids_for_count
from egohandmetric_prompt.data.flow_pseudo_labels import FlowPseudoLabelStore, transport_flow_pseudo_label
from egohandmetric_prompt.heatmap_targets import build_keypoint_heatmap_targets
from egohandmetric_prompt.losses.contact_losses import (
    contact_classification_metrics,
    masked_contact_confidence_bce_with_logits,
)
from egohandmetric_prompt.losses.marker_losses import (
    hand_confidence_weighted_loss,
    vertex_depth_consistency_loss,
)
from egohandmetric_prompt.losses.metric_scale_losses import (
    depth_anchored_hand_scale_target,
    log_scale_loss,
    masked_average_distance,
    scene_average_distance_from_depth,
    transform_camera_points_to_first_frame,
    unproject_depth_to_camera_points,
)
from egohandmetric_prompt.losses.three_r_losses import depth_confidence_loss, point_confidence_loss
from egohandmetric_prompt.models import WiLorTeacherWrapper
from egohandmetric_prompt.vendor import ensure_vendor_paths


@dataclass(slots=True)
class MarkerStageLoaders:
    stage: str
    bundle: StageDatasetBundle
    marker_loader: DataLoader
    three_r_loader: DataLoader
    posttrain_three_r_stream_ratio: float = 1.0
    posttrain_stream_schedule: tuple[str, ...] | None = None
    marker_epoch_controller: Any | None = None
    three_r_epoch_controller: Any | None = None
    rank_sharded: bool = False


def _serialize_raw_chunks(chunks: list[dict[str, Any]]) -> bytes:
    """Worker-side identity collate that avoids tensor shared-memory IPC."""
    return pickle.dumps(chunks, protocol=pickle.HIGHEST_PROTOCOL)


def _pin_cpu_tensors(value: Any) -> Any:
    """Recursively pin CPU tensors after main-process lazy supervision."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            return value
        try:
            return value.pin_memory()
        except RuntimeError:
            # PyTorch may expose pin_memory=True on a CPU-only host.
            return value
    if isinstance(value, dict):
        return {key: _pin_cpu_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pin_cpu_tensors(item) for item in value]
    if isinstance(value, tuple):
        values = tuple(_pin_cpu_tensors(item) for item in value)
        return type(value)(*values) if hasattr(value, "_fields") else values
    return value


class _MainProcessCollatingLoader:
    """Apply lazy MANO/visibility/contact supervision only in the train process."""
    def __init__(self, raw_loader: DataLoader, collator: Any) -> None:
        self.raw_loader = raw_loader
        self.collator = collator
        self._pin_memory = bool(raw_loader.pin_memory)

    def __iter__(self):
        for payload in self.raw_loader:
            batch = self.collator(pickle.loads(payload))
            yield _pin_cpu_tensors(batch) if self._pin_memory else batch

    def __len__(self) -> int:
        return len(self.raw_loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw_loader, name)


def _main_process_collating_loader(raw_loader: DataLoader, collator: Any, common: Any) -> Any:
    if int(common.num_workers) <= 0:
        return raw_loader
    return _MainProcessCollatingLoader(raw_loader, collator)


_LOADER_DETAIL_LOG_CACHE: set[str] = set()
_FRAME_SIGNAL_KEYS = (
    "has_mano",
    "has_3r_gt",
    "has_bbox_gt",
    "has_joint_3d_gt",
    "has_depth_gt",
    "has_intrinsics_gt",
    "has_camera_pose_gt",
)


def _signal_summary(**flags: bool) -> dict[str, bool]:
    summary = {key: False for key in _FRAME_SIGNAL_KEYS}
    summary.update(flags)
    if "has_3r_gt" not in flags:
        summary["has_3r_gt"] = (
            summary["has_depth_gt"] or summary["has_intrinsics_gt"] or summary["has_camera_pose_gt"]
        )
    return summary


_STATIC_DATASET_SIGNAL_SUMMARIES: dict[str, dict[str, bool]] = {
    "h2o": _signal_summary(
        has_mano=True,
        has_bbox_gt=True,
        has_joint_3d_gt=True,
        has_depth_gt=True,
        has_intrinsics_gt=True,
        has_camera_pose_gt=True,
    ),
}


@dataclass(slots=True)
class EpochGeneratorController:
    generator: torch.Generator
    base_seed: int

    def set_epoch(self, epoch: int) -> None:
        self.generator.manual_seed(self.base_seed + int(epoch))


def _is_primary_process_from_env() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _distributed_world_size_from_env() -> int:
    return max(int(os.environ.get("WORLD_SIZE", "1")), 1)


def _distributed_process_index_from_env() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _rank_sharded_batch_sampler_from_env(batch_sampler: Any) -> Any:
    world_size = _distributed_world_size_from_env()
    if world_size <= 1:
        return batch_sampler
    return RankShardedBatchSampler(
        batch_sampler,
        rank=_distributed_process_index_from_env(),
        world_size=world_size,
    )


def _log_loader_stage(message: str, *args: Any) -> None:
    if _is_primary_process_from_env():
        logger.info(message, *args)


def _missing_sample_retry_loader_parts(
    *,
    dataset: Any,
    collator: Any,
    project_config: ProjectConfig,
    for_training: bool,
    stream_name: str,
    seed_offset: int,
    registry_dir: str | Path | None,
) -> tuple[Any, Any]:
    attempts = int(project_config.marker_data.common.missing_sample_replacement_attempts)
    if not for_training or attempts <= 0:
        return dataset, collator
    configured_registry_dir = project_config.marker_data.common.missing_sample_registry_dir.strip()
    resolved_registry_dir = (
        Path(configured_registry_dir)
        if configured_registry_dir
        else Path(registry_dir)
        if registry_dir is not None
        else Path(project_config.marker_train.output_dir) / "missing_samples"
    )
    return (
        MissingSampleRetryDataset(
            dataset,
            replacement_attempts=attempts,
            stream_name=stream_name,
            registry_dir=resolved_registry_dir,
            seed=project_config.project.seed + seed_offset,
        ),
        MissingSampleCollator(collator),
    )


def set_marker_stage_loaders_epoch(loaders: MarkerStageLoaders, epoch_index: int) -> None:
    for controller in (loaders.marker_epoch_controller, loaders.three_r_epoch_controller):
        if controller is not None and hasattr(controller, "set_epoch"):
            controller.set_epoch(epoch_index)


def _validated_dataset_sampling_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    validated = {name: float(weight) for name, weight in raw_weights.items()}
    invalid = {name: weight for name, weight in validated.items() if not math.isfinite(weight) or weight < 0.0}
    if invalid:
        raise ValueError(f"dataset_sampling_weights 必须是有限的非负数，当前非法项: {invalid}")
    return validated


def _is_random_length_dataset(dataset: Any) -> bool:
    return isinstance(dataset, (RandomLengthTemporalChunkDataset, FilteredRandomLengthTemporalChunkDataset))


def _cached_frame_records(frame_dataset: BaseFrameDataset) -> list[Any]:
    materialized = getattr(frame_dataset, "_records_materialized", None)
    if materialized is not None:
        return list(materialized)
    record_cache = getattr(frame_dataset, "_record_cache", None)
    if record_cache:
        return list(record_cache.values())
    return []


def _merge_record_signals(summary: dict[str, bool], record: Any) -> None:
    for key in _FRAME_SIGNAL_KEYS:
        summary[key] |= bool(getattr(record, key))


def _probe_frame_dataset_signals(frame_dataset: BaseFrameDataset, *, max_probe_records: int = 4) -> dict[str, bool]:
    total_records = len(frame_dataset)
    if total_records == 0:
        return _signal_summary()
    static_summary = _STATIC_DATASET_SIGNAL_SUMMARIES.get(frame_dataset.dataset_name)
    if static_summary is not None:
        return dict(static_summary)
    summary = _signal_summary()
    for record in _cached_frame_records(frame_dataset)[:max_probe_records]:
        _merge_record_signals(summary, record)
    return summary


def _frame_dataset_supervision_lists(signal_summary: dict[str, bool]) -> tuple[list[str], list[str]]:
    marker_supervisions: list[str] = []
    three_r_supervisions: list[str] = []
    if signal_summary["has_bbox_gt"]:
        marker_supervisions.append("bbox/heatmap")
    if signal_summary["has_joint_3d_gt"]:
        marker_supervisions.append("joint_xyz/joint_visibility")
    if signal_summary["has_mano"]:
        marker_supervisions.append("mano_pose/mano_shape/mano_trans")
        marker_supervisions.append("vertex_xyz/vertex_visibility")
    if signal_summary["has_depth_gt"]:
        three_r_supervisions.append("depth")
    if signal_summary["has_intrinsics_gt"]:
        three_r_supervisions.append("intrinsics")
    if signal_summary["has_camera_pose_gt"]:
        three_r_supervisions.append("camera_pose")
    if signal_summary["has_mano"] and signal_summary["has_depth_gt"] and signal_summary["has_intrinsics_gt"]:
        three_r_supervisions.append("vertex_depth_consistency")
    return marker_supervisions, three_r_supervisions


def _iter_frame_datasets(dataset: Dataset) -> list[BaseFrameDataset]:
    if isinstance(dataset, BaseFrameDataset):
        return [dataset]
    if isinstance(dataset, CombinedFrameDataset):
        outputs: list[BaseFrameDataset] = []
        for child in dataset.datasets:
            outputs.extend(_iter_frame_datasets(child))
        return outputs
    if isinstance(dataset, MultiSourceRandomLengthTemporalChunkDataset):
        outputs: list[BaseFrameDataset] = []
        for child in dataset.datasets:
            outputs.extend(_iter_frame_datasets(child))
        return outputs
    if isinstance(dataset, TemporalChunkDataset):
        return _iter_frame_datasets(dataset.frame_dataset)
    if _is_random_length_dataset(dataset):
        return _iter_frame_datasets(dataset.frame_dataset)
    if isinstance(dataset, ConcatDataset):
        outputs: list[BaseFrameDataset] = []
        for child in dataset.datasets:
            outputs.extend(_iter_frame_datasets(child))
        return outputs
    if isinstance(dataset, Subset):
        return _iter_frame_datasets(dataset.dataset)
    if hasattr(dataset, "dataset"):
        return _iter_frame_datasets(dataset.dataset)
    return []


def _loader_batch_count(loader: DataLoader) -> int:
    try:
        return len(loader)
    except TypeError:
        return -1


def _dataset_ratio_entries(
    *,
    dataset: Dataset,
    stream_name: str,
    control_mode: str,
    dataset_sampling_weights: dict[str, float],
) -> list[dict[str, Any]]:
    if isinstance(dataset, MultiSourceRandomLengthTemporalChunkDataset):
        raw_epoch_counts = dataset.dataset_epoch_sample_counts()
        total_raw_epoch_count = float(sum(raw_epoch_counts.values()))
        weighted_epoch_counts = dataset.weighted_epoch_sample_counts(dataset_sampling_weights)
        total_weighted_epoch_count = float(sum(weighted_epoch_counts.values()))
        sampling_probabilities = dataset.sampling_probabilities(dataset_sampling_weights)
        entries: list[dict[str, Any]] = []
        for dataset_name in dataset.dataset_names:
            raw_epoch_count = raw_epoch_counts.get(dataset_name, 0)
            weighted_epoch_count = weighted_epoch_counts.get(dataset_name, 0)
            weighted_epoch_ratio = 0.0 if total_weighted_epoch_count <= 0 else float(weighted_epoch_count) / total_weighted_epoch_count
            entry = {
                "stream_name": stream_name,
                "dataset_name": dataset_name,
                "weight": float(dataset_sampling_weights.get(dataset_name, 1.0)),
                "raw_epoch_sample_count": raw_epoch_count,
                "raw_epoch_ratio": 0.0 if total_raw_epoch_count <= 0 else float(raw_epoch_count) / total_raw_epoch_count,
                "weighted_epoch_sample_count": weighted_epoch_count,
                "weighted_epoch_ratio": weighted_epoch_ratio,
                "sampling_ratio": weighted_epoch_ratio if control_mode == "epochs" else sampling_probabilities.get(dataset_name, 0.0),
                "weight_applied": True,
            }
            entries.append(entry)
        return entries
    if isinstance(dataset, ConcatDataset):
        total_count = float(len(dataset))
        entries = []
        for child in dataset.datasets:
            child_name = getattr(getattr(child, "frame_dataset", None), "dataset_name", type(child).__name__)
            child_count = len(child)
            entries.append(
                {
                    "stream_name": stream_name,
                    "dataset_name": child_name,
                    "weight": float(dataset_sampling_weights.get(child_name, 1.0)),
                    "raw_epoch_sample_count": child_count,
                    "raw_epoch_ratio": 0.0 if total_count <= 0 else float(child_count) / total_count,
                    "weighted_epoch_sample_count": child_count,
                    "weighted_epoch_ratio": 0.0 if total_count <= 0 else float(child_count) / total_count,
                    "sampling_ratio": 0.0 if total_count <= 0 else float(child_count) / total_count,
                    "weight_applied": False,
                }
            )
        return entries
    child_name = getattr(getattr(dataset, "frame_dataset", None), "dataset_name", type(dataset).__name__)
    total_count = len(dataset)
    return [
        {
            "stream_name": stream_name,
            "dataset_name": child_name,
            "weight": float(dataset_sampling_weights.get(child_name, 1.0)),
            "raw_epoch_sample_count": total_count,
            "raw_epoch_ratio": 1.0 if total_count > 0 else 0.0,
            "weighted_epoch_sample_count": total_count,
            "weighted_epoch_ratio": 1.0 if total_count > 0 else 0.0,
            "sampling_ratio": 1.0 if total_count > 0 else 0.0,
            "weight_applied": False,
        }
    ]


def _log_loader_details(
    *,
    stage: str,
    marker_loader: DataLoader,
    three_r_loader: DataLoader,
    bundle: StageDatasetBundle,
    build_time_sec: float,
    collator_time_sec: float,
    dataset_time_sec: float,
    epoch_index: int | None,
    control_mode: str,
    dataset_sampling_weights: dict[str, float],
) -> None:
    if not _is_primary_process_from_env():
        return
    marker_batch_count = _loader_batch_count(marker_loader)
    three_r_batch_count = _loader_batch_count(three_r_loader)
    random_length_mode = _is_random_length_dataset(bundle.marker_dataset) or isinstance(
        bundle.marker_dataset,
        MultiSourceRandomLengthTemporalChunkDataset,
    ) or _is_random_length_dataset(bundle.three_r_dataset) or isinstance(
        bundle.three_r_dataset,
        MultiSourceRandomLengthTemporalChunkDataset,
    )
    logger.info(
        "dataloader实例化完成: stage={} epoch_index={} build_time={:.2f}s collator_time={:.2f}s dataset_time={:.2f}s marker_samples={} marker_batches={} three_r_samples={} three_r_batches={} random_length_mode={}",
        stage,
        epoch_index,
        build_time_sec,
        collator_time_sec,
        dataset_time_sec,
        bundle.marker_sample_count,
        marker_batch_count,
        bundle.three_r_sample_count,
        three_r_batch_count,
        random_length_mode,
    )

    detail_cache_key = "|".join(
        [
            stage,
            ",".join(bundle.marker_dataset_names),
            ",".join(bundle.three_r_dataset_names),
            str(epoch_index is not None),
            repr(sorted(dataset_sampling_weights.items())),
        ]
    )
    if detail_cache_key in _LOADER_DETAIL_LOG_CACHE:
        return
    _LOADER_DETAIL_LOG_CACHE.add(detail_cache_key)

    if dataset_sampling_weights:
        logger.info(
            "dataset sampling weights: control_mode={} weights={} note={}",
            control_mode,
            dataset_sampling_weights,
            "epochs 模式下通过重复样本提高数据集占比，同时保持原始样本全集覆盖。" if control_mode == "epochs" else "steps 模式下权重会影响随机采样比例。",
        )

    for entry in _dataset_ratio_entries(
        dataset=bundle.marker_dataset,
        stream_name="marker",
        control_mode=control_mode,
        dataset_sampling_weights=dataset_sampling_weights,
    ):
        logger.info(
            "dataset占比: stream={} dataset={} raw_epoch_samples={} raw_ratio={:.4f} weighted_epoch_samples={} weighted_ratio={:.4f} sampling_ratio={:.4f} weight={} weight_applied={}",
            entry["stream_name"],
            entry["dataset_name"],
            entry["raw_epoch_sample_count"],
            entry["raw_epoch_ratio"],
            entry["weighted_epoch_sample_count"],
            entry["weighted_epoch_ratio"],
            entry["sampling_ratio"],
            entry["weight"],
            entry["weight_applied"],
        )
    for entry in _dataset_ratio_entries(
        dataset=bundle.three_r_dataset,
        stream_name="three_r",
        control_mode=control_mode,
        dataset_sampling_weights=dataset_sampling_weights,
    ):
        logger.info(
            "dataset占比: stream={} dataset={} raw_epoch_samples={} raw_ratio={:.4f} weighted_epoch_samples={} weighted_ratio={:.4f} sampling_ratio={:.4f} weight={} weight_applied={}",
            entry["stream_name"],
            entry["dataset_name"],
            entry["raw_epoch_sample_count"],
            entry["raw_epoch_ratio"],
            entry["weighted_epoch_sample_count"],
            entry["weighted_epoch_ratio"],
            entry["sampling_ratio"],
            entry["weight"],
            entry["weight_applied"],
        )

    dataset_usage: dict[int, dict[str, Any]] = {}
    for stream_name, stream_dataset in (("marker", bundle.marker_dataset), ("three_r", bundle.three_r_dataset)):
        for frame_dataset in _iter_frame_datasets(stream_dataset):
            key = id(frame_dataset)
            entry = dataset_usage.setdefault(
                key,
                {
                    "dataset": frame_dataset,
                    "streams": set(),
                },
            )
            entry["streams"].add(stream_name)

    for entry in dataset_usage.values():
        frame_dataset = entry["dataset"]
        signal_summary = _probe_frame_dataset_signals(frame_dataset)
        marker_supervisions, three_r_supervisions = _frame_dataset_supervision_lists(signal_summary)
        logger.info(
            "dataloader数据集摘要: dataset={} split={} frames={} sequences={} streams={} has_mano={} has_3r={} marker_supervisions={} three_r_supervisions={}",
            frame_dataset.dataset_name,
            frame_dataset.split,
            len(frame_dataset),
            len(frame_dataset.sequence_to_indices),
            sorted(entry["streams"]),
            signal_summary["has_mano"],
            signal_summary["has_3r_gt"],
            marker_supervisions,
            three_r_supervisions,
        )


class MarkerRuntimeCollator:
    def __init__(
        self,
        base_collator: MarkerBatchCollator,
        *,
        target_height: int,
        target_width: int,
        random_crop_resize: RandomCropResizeConfig | None = None,
        flow_pseudo_store: FlowPseudoLabelStore | None = None,
        flow_fingertip_sigma_px: float = 12.0,
        seed: int = 0,
    ) -> None:
        self.base_collator = base_collator
        self.target_height = target_height
        self.target_width = target_width
        self.random_crop_resize = random_crop_resize or RandomCropResizeConfig()
        self.flow_pseudo_store = flow_pseudo_store
        self.flow_fingertip_sigma_px = max(float(flow_fingertip_sigma_px), 1e-3)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def __call__(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        target_height, target_width = self._batch_target_size(chunks)
        return self.base_collator(
            [
                self._resize_chunk(
                    chunk,
                    target_height=target_height,
                    target_width=target_width,
                )
                for chunk in chunks
            ]
        )

    def _batch_target_size(self, chunks: list[dict[str, Any]]) -> tuple[int, int]:
        target_sizes = []
        for chunk in chunks:
            target_size = chunk.get("target_image_size_hw")
            if target_size is None:
                continue
            target_height, target_width = target_size
            target_sizes.append((int(target_height), int(target_width)))
        if not target_sizes:
            return self.target_height, self.target_width
        first = target_sizes[0]
        if any(target_size != first for target_size in target_sizes):
            raise ValueError(f"同一个 batch 内 target_image_size_hw 必须一致，当前为 {target_sizes}")
        return first

    def _resize_chunk(
        self,
        chunk: dict[str, Any],
        *,
        target_height: int,
        target_width: int,
    ) -> dict[str, Any]:
        resized = deepcopy(chunk)
        crop_box = self._select_crop_box(resized, target_height=target_height, target_width=target_width)
        resized["samples"] = [
            self._resize_sample(
                sample,
                target_height=target_height,
                target_width=target_width,
                crop_box=crop_box,
            )
            for sample in resized["samples"]
        ]
        rgb_items = [sample["rgb"] for sample in resized["samples"] if sample.get("rgb") is not None]
        if len(rgb_items) == len(resized["samples"]) and rgb_items:
            resized["rgb"] = torch.stack(rgb_items)
        else:
            resized["rgb"] = None
        depth_items = [sample["depth"] for sample in resized["samples"] if sample.get("depth") is not None]
        if len(depth_items) == len(resized["samples"]) and depth_items:
            resized["depth"] = torch.stack(depth_items)
        else:
            resized["depth"] = None
        if self.flow_pseudo_store is not None:
            self._attach_flow_pseudo_labels(
                resized,
                source_samples=chunk["samples"],
                mask_samples=resized["samples"],
                crop_box=crop_box,
                target_height=target_height,
                target_width=target_width,
            )
        return resized

    def _attach_flow_pseudo_labels(
        self,
        chunk: dict[str, Any],
        *,
        source_samples: list[dict[str, Any]],
        mask_samples: list[dict[str, Any]],
        crop_box: tuple[int, int, int, int] | None,
        target_height: int,
        target_width: int,
    ) -> None:
        samples = source_samples
        pair_count = max(len(samples) - 1, 0)
        flow_target = torch.zeros(pair_count, 2, target_height, target_width, dtype=torch.float32)
        flow_valid = torch.zeros(pair_count, target_height, target_width, dtype=torch.bool)
        flow_hand_region_mask = torch.zeros(pair_count, target_height, target_width, dtype=torch.bool)
        flow_fingertip_weight = torch.zeros(pair_count, target_height, target_width, dtype=torch.float32)
        pair_mask = torch.zeros(pair_count, dtype=torch.bool)
        for pair_index, (source, target) in enumerate(zip(samples[:-1], samples[1:], strict=True)):
            source_index = int(source["temporal_index"])
            target_index = int(target["temporal_index"])
            same_source = (
                str(source["dataset_name"]) == str(target["dataset_name"])
                and str(source["sequence_id"]) == str(target["sequence_id"])
                and str(source.get("view_name", "")) == str(target.get("view_name", ""))
            )
            if not same_source or target_index - source_index != 1:
                continue
            label = self.flow_pseudo_store.load(
                dataset_name=str(source["dataset_name"]),
                sequence_id=str(source["sequence_id"]),
                view_name=str(source.get("view_name", "")),
                source_temporal_index=source_index,
                target_temporal_index=target_index,
            )
            if label is None:
                continue
            source_rgb = source.get("rgb")
            if source_rgb is None or tuple(source_rgb.shape[-2:]) != label.source_image_size_hw:
                continue
            transported = transport_flow_pseudo_label(
                label,
                crop_box=crop_box,
                target_height=target_height,
                target_width=target_width,
            )
            flow_target[pair_index] = transported.flow
            flow_valid[pair_index] = transported.valid
            hand_mask, fingertip_weight = self._flow_hand_maps_for_sample(
                mask_samples[pair_index],
                target_height=target_height,
                target_width=target_width,
            )
            flow_hand_region_mask[pair_index] = hand_mask
            flow_fingertip_weight[pair_index] = fingertip_weight
            pair_mask[pair_index] = True
        chunk["flow_pseudo_target"] = flow_target
        chunk["flow_pseudo_valid"] = flow_valid
        chunk["flow_hand_region_mask"] = flow_hand_region_mask
        chunk["flow_fingertip_weight"] = flow_fingertip_weight
        chunk["flow_pair_mask"] = pair_mask

    def _flow_hand_maps_for_sample(
        self,
        sample: dict[str, Any],
        *,
        target_height: int,
        target_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hand_mask = torch.zeros(target_height, target_width, dtype=torch.bool)
        fingertip_weight = torch.zeros(target_height, target_width, dtype=torch.float32)
        joints_for_tips: list[torch.Tensor] = []
        for hand in sample.get("hand_annos", []):
            if hand is None or not bool(hand.get("visible", True)):
                continue
            bbox = hand.get("bbox_xyxy")
            joints = hand.get("joints_2d")
            if bbox is not None:
                bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32).reshape(-1)
            else:
                bbox_tensor = torch.empty(0, dtype=torch.float32)
            if (bbox_tensor.numel() != 4 or not torch.isfinite(bbox_tensor).all()) and joints is not None:
                joints_tensor = torch.as_tensor(joints, dtype=torch.float32).reshape(-1, 2)
                valid_joints = joints_tensor[torch.isfinite(joints_tensor).all(dim=-1)]
                if valid_joints.numel() > 0:
                    bbox_tensor = torch.stack(
                        (
                            valid_joints[:, 0].min(),
                            valid_joints[:, 1].min(),
                            valid_joints[:, 0].max(),
                            valid_joints[:, 1].max(),
                        )
                    )
            if bbox_tensor.numel() == 4 and torch.isfinite(bbox_tensor).all():
                left, top, right, bottom = bbox_tensor.tolist()
                x0 = max(0, min(target_width, int(torch.floor(torch.tensor(left)).item())))
                y0 = max(0, min(target_height, int(torch.floor(torch.tensor(top)).item())))
                x1 = max(0, min(target_width, int(torch.ceil(torch.tensor(right)).item())))
                y1 = max(0, min(target_height, int(torch.ceil(torch.tensor(bottom)).item())))
                if x1 > x0 and y1 > y0:
                    hand_mask[y0:y1, x0:x1] = True
            if joints is not None:
                joints_tensor = torch.as_tensor(joints, dtype=torch.float32).reshape(-1, 2)
                if joints_tensor.shape[0] >= 21:
                    tips = joints_tensor[[4, 8, 12, 16, 20]]
                    tips = tips[torch.isfinite(tips).all(dim=-1)]
                    tips = tips[
                        (tips[:, 0] >= 0.0)
                        & (tips[:, 0] < float(target_width))
                        & (tips[:, 1] >= 0.0)
                        & (tips[:, 1] < float(target_height))
                    ]
                    if tips.numel() > 0:
                        joints_for_tips.append(tips)
        if joints_for_tips:
            yy, xx = torch.meshgrid(
                torch.arange(target_height, dtype=torch.float32),
                torch.arange(target_width, dtype=torch.float32),
                indexing="ij",
            )
            sigma = self.flow_fingertip_sigma_px
            for tips in joints_for_tips:
                for x_coord, y_coord in tips:
                    distance_sq = (xx - x_coord).square() + (yy - y_coord).square()
                    fingertip_weight = torch.maximum(
                        fingertip_weight,
                        torch.exp(-distance_sq / (2.0 * sigma * sigma)),
                    )
        return hand_mask, fingertip_weight

    def _resize_sample(
        self,
        sample: dict[str, Any],
        *,
        target_height: int,
        target_width: int,
        crop_box: tuple[int, int, int, int] | None,
    ) -> dict[str, Any]:
        resized = deepcopy(sample)
        rgb = resized.get("rgb")
        if rgb is None:
            raise ValueError("真实 marker runtime 需要 sample['rgb'] 已加载")
        source_height, source_width = rgb.shape[-2:]
        crop_left, crop_top, crop_right, crop_bottom = crop_box or (0, 0, source_width, source_height)
        crop_left = max(min(int(crop_left), source_width - 1), 0)
        crop_top = max(min(int(crop_top), source_height - 1), 0)
        crop_right = max(min(int(crop_right), source_width), crop_left + 1)
        crop_bottom = max(min(int(crop_bottom), source_height), crop_top + 1)
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        scale_x = float(target_width) / max(crop_width, 1)
        scale_y = float(target_height) / max(crop_height, 1)
        resized["rgb"] = F.interpolate(
            rgb.to(dtype=torch.float32)[:, crop_top:crop_bottom, crop_left:crop_right].unsqueeze(0),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        depth = resized.get("depth")
        if depth is not None:
            depth_height, depth_width = depth.shape[-2:]
            depth_scale_x = float(depth_width) / max(float(source_width), 1.0)
            depth_scale_y = float(depth_height) / max(float(source_height), 1.0)
            depth_crop_left = max(min(int(round(float(crop_left) * depth_scale_x)), depth_width - 1), 0)
            depth_crop_top = max(min(int(round(float(crop_top) * depth_scale_y)), depth_height - 1), 0)
            depth_crop_right = max(
                min(int(round(float(crop_right) * depth_scale_x)), depth_width),
                depth_crop_left + 1,
            )
            depth_crop_bottom = max(
                min(int(round(float(crop_bottom) * depth_scale_y)), depth_height),
                depth_crop_top + 1,
            )
            resized["depth"] = F.interpolate(
                depth.to(dtype=torch.float32)[
                    depth_crop_top:depth_crop_bottom,
                    depth_crop_left:depth_crop_right,
                ].unsqueeze(0).unsqueeze(0),
                size=(target_height, target_width),
                mode="nearest",
            ).squeeze(0).squeeze(0)
        intrinsics = resized.get("intrinsics")
        if intrinsics is not None:
            intrinsics = intrinsics.clone().to(dtype=torch.float32)
            intrinsics[0, 2] -= float(crop_left)
            intrinsics[1, 2] -= float(crop_top)
            intrinsics[0, 0] *= scale_x
            intrinsics[0, 2] *= scale_x
            intrinsics[1, 1] *= scale_y
            intrinsics[1, 2] *= scale_y
            resized["intrinsics"] = intrinsics
        resized["hand_annos"] = [
            self._resize_hand(
                hand,
                crop_left=crop_left,
                crop_top=crop_top,
                crop_right=crop_right,
                crop_bottom=crop_bottom,
                scale_x=scale_x,
                scale_y=scale_y,
                target_height=target_height,
                target_width=target_width,
            )
            for hand in resized["hand_annos"]
        ]
        if resized.get("left_hand") is not None:
            resized["left_hand"] = self._resize_hand(
                resized["left_hand"],
                crop_left=crop_left,
                crop_top=crop_top,
                crop_right=crop_right,
                crop_bottom=crop_bottom,
                scale_x=scale_x,
                scale_y=scale_y,
                target_height=target_height,
                target_width=target_width,
            )
        if resized.get("right_hand") is not None:
            resized["right_hand"] = self._resize_hand(
                resized["right_hand"],
                crop_left=crop_left,
                crop_top=crop_top,
                crop_right=crop_right,
                crop_bottom=crop_bottom,
                scale_x=scale_x,
                scale_y=scale_y,
                target_height=target_height,
                target_width=target_width,
            )
        return resized

    def _resize_hand(
        self,
        hand: dict[str, Any],
        *,
        crop_left: int,
        crop_top: int,
        crop_right: int,
        crop_bottom: int,
        scale_x: float,
        scale_y: float,
        target_height: int,
        target_width: int,
    ) -> dict[str, Any]:
        resized = deepcopy(hand)
        bbox = resized.get("bbox_xyxy")
        if bbox is not None:
            bbox = bbox.clone().to(dtype=torch.float32)
            bbox[0] = bbox[0].clamp(float(crop_left), float(crop_right))
            bbox[2] = bbox[2].clamp(float(crop_left), float(crop_right))
            bbox[1] = bbox[1].clamp(float(crop_top), float(crop_bottom))
            bbox[3] = bbox[3].clamp(float(crop_top), float(crop_bottom))
            if bool(((bbox[2] > bbox[0]) & (bbox[3] > bbox[1])).item()):
                bbox[0] = (bbox[0] - float(crop_left)) * scale_x
                bbox[2] = (bbox[2] - float(crop_left)) * scale_x
                bbox[1] = (bbox[1] - float(crop_top)) * scale_y
                bbox[3] = (bbox[3] - float(crop_top)) * scale_y
                resized["bbox_xyxy"] = bbox
            else:
                resized["bbox_xyxy"] = None
        joints_2d = resized.get("joints_2d")
        if joints_2d is not None:
            joints_2d = joints_2d.clone().to(dtype=torch.float32)
            joints_2d[..., 0] = (joints_2d[..., 0] - float(crop_left)) * scale_x
            joints_2d[..., 1] = (joints_2d[..., 1] - float(crop_top)) * scale_y
            resized["joints_2d"] = joints_2d
        resized["visible"] = self._hand_has_visible_2d_evidence(resized, target_height=target_height, target_width=target_width)
        return resized

    def _select_crop_box(
        self,
        chunk: dict[str, Any],
        *,
        target_height: int,
        target_width: int,
    ) -> tuple[int, int, int, int] | None:
        if not self.random_crop_resize.enabled:
            return None
        rgb = chunk["samples"][0].get("rgb")
        if rgb is None:
            raise ValueError("真实 marker runtime 需要 sample['rgb'] 已加载")
        source_height, source_width = rgb.shape[-2:]
        target_aspect = float(target_width) / max(float(target_height), 1.0)
        source_aspect = float(source_width) / max(float(source_height), 1.0)
        if abs(source_aspect - target_aspect) < 1e-6:
            crop_width = source_width
            crop_height = source_height
        elif source_aspect > target_aspect:
            crop_height = source_height
            crop_width = max(min(int(round(float(source_height) * target_aspect)), source_width), 1)
        else:
            crop_width = source_width
            crop_height = max(min(int(round(float(source_width) / target_aspect)), source_height), 1)
        evidence_bbox = self._chunk_2d_evidence_bbox(chunk)
        randomize = self._should_randomize_crop()
        crop_left = self._axis_crop_start(
            source_size=source_width,
            crop_size=crop_width,
            evidence_min=None if evidence_bbox is None else float(evidence_bbox[0]),
            evidence_max=None if evidence_bbox is None else float(evidence_bbox[2]),
            randomize=randomize,
        )
        crop_top = self._axis_crop_start(
            source_size=source_height,
            crop_size=crop_height,
            evidence_min=None if evidence_bbox is None else float(evidence_bbox[1]),
            evidence_max=None if evidence_bbox is None else float(evidence_bbox[3]),
            randomize=randomize,
        )
        return (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)

    def _should_randomize_crop(self) -> bool:
        probability = float(self.random_crop_resize.crop_probability)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool((torch.rand((), generator=self.generator).item() < probability))

    def _axis_crop_start(
        self,
        *,
        source_size: int,
        crop_size: int,
        evidence_min: float | None,
        evidence_max: float | None,
        randomize: bool,
    ) -> int:
        max_start = int(source_size) - int(crop_size)
        if max_start <= 0:
            return 0
        if evidence_min is None or evidence_max is None:
            if randomize:
                return int(torch.randint(max_start + 1, (1,), generator=self.generator).item())
            return max_start // 2
        lower = max(0, int(round(evidence_max - float(crop_size))))
        upper = min(max_start, int(round(evidence_min)))
        if lower <= upper:
            if randomize:
                return int(torch.randint(lower, upper + 1, (1,), generator=self.generator).item())
            return (lower + upper) // 2
        center = 0.5 * (float(evidence_min) + float(evidence_max))
        return max(min(int(round(center - 0.5 * float(crop_size))), max_start), 0)

    def _chunk_2d_evidence_bbox(self, chunk: dict[str, Any]) -> torch.Tensor | None:
        points: list[torch.Tensor] = []
        for sample in chunk["samples"]:
            for hand in sample.get("hand_annos", []):
                bbox = hand.get("bbox_xyxy")
                if bbox is not None:
                    bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32)
                    if bbox_tensor.shape == (4,) and bool(torch.isfinite(bbox_tensor).all().item()):
                        points.append(bbox_tensor[[0, 1]])
                        points.append(bbox_tensor[[2, 3]])
                joints_2d = hand.get("joints_2d")
                if joints_2d is None:
                    continue
                joint_tensor = torch.as_tensor(joints_2d, dtype=torch.float32)
                if joint_tensor.ndim != 2 or joint_tensor.shape[-1] != 2:
                    continue
                finite = torch.isfinite(joint_tensor).all(dim=-1)
                if bool(finite.any().item()):
                    points.append(joint_tensor[finite])
        if not points:
            return None
        stacked = torch.cat([point.reshape(-1, 2) for point in points], dim=0)
        return torch.stack([stacked[:, 0].min(), stacked[:, 1].min(), stacked[:, 0].max(), stacked[:, 1].max()])

    def _hand_has_visible_2d_evidence(self, hand: dict[str, Any], *, target_height: int, target_width: int) -> bool:
        bbox = hand.get("bbox_xyxy")
        if bbox is not None:
            bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32)
            if bbox_tensor.shape == (4,) and bool(torch.isfinite(bbox_tensor).all().item()):
                if bool(((bbox_tensor[2] > 0) & (bbox_tensor[0] < target_width) & (bbox_tensor[3] > 0) & (bbox_tensor[1] < target_height)).item()):
                    return True
        joints_2d = hand.get("joints_2d")
        if joints_2d is None:
            return False
        joint_tensor = torch.as_tensor(joints_2d, dtype=torch.float32)
        if joint_tensor.ndim != 2 or joint_tensor.shape[-1] != 2:
            return False
        finite = torch.isfinite(joint_tensor).all(dim=-1)
        inside = (
            (joint_tensor[..., 0] >= 0)
            & (joint_tensor[..., 0] <= target_width - 1)
            & (joint_tensor[..., 1] >= 0)
            & (joint_tensor[..., 1] <= target_height - 1)
        )
        return bool((finite & inside).any().item())


def _stage_data_config(project_config: ProjectConfig, stage: str):
    if stage == "midtrain":
        return project_config.marker_data.midtrain
    if stage == "posttrain":
        return project_config.marker_data.posttrain
    raise ValueError(f"未知 stage: {stage}")


def _stage_marker_dataset_names(project_config: ProjectConfig, stage: str) -> list[str]:
    return list(_stage_data_config(project_config, stage).marker_dataset_names)


def _stage_three_r_dataset_names(project_config: ProjectConfig, stage: str) -> list[str]:
    if stage == "midtrain":
        return []
    return list(_stage_data_config(project_config, stage).three_r_dataset_names)


def _stage_dataset_sampling_weights(
    project_config: ProjectConfig,
    stage: str,
    stream_name: str,
) -> dict[str, float]:
    """Resolve stream-specific weights, with common weights as a legacy fallback."""
    common = dict(project_config.marker_data.common.dataset_sampling_weights)
    stage_config = _stage_data_config(project_config, stage)
    selected = (
        stage_config.marker_dataset_sampling_weights
        if stream_name == "marker"
        else stage_config.three_r_dataset_sampling_weights
    )
    resolved = dict(common)
    resolved.update(selected)
    if stream_name not in {"marker", "three_r"}:
        raise ValueError(f"未知 stream: {stream_name}")
    allowed_names = set(
        _stage_marker_dataset_names(project_config, stage)
        if stream_name == "marker"
        else _stage_three_r_dataset_names(project_config, stage)
    )
    unknown_names = set(resolved) - allowed_names
    if unknown_names:
        raise ValueError(
            f"{stage}.{stream_name} stream 的 dataset_sampling_weights 包含未注册数据集: {sorted(unknown_names)}"
        )
    if resolved and not allowed_names:
        raise ValueError(f"{stage}.{stream_name} stream 没有可采样数据集")
    if allowed_names and resolved and not any(float(resolved.get(name, 1.0)) > 0.0 for name in allowed_names):
        raise ValueError(f"{stage}.{stream_name} stream 的 dataset sampling weights 不能全部为 0")
    return _validated_dataset_sampling_weights(resolved)


def _fixed_length_weighted_sampler(
    dataset: Any,
    dataset_names: Sequence[str],
    weights: dict[str, float],
    *,
    seed: int,
) -> WeightedRandomSampler | None:
    if not isinstance(dataset, ConcatDataset) or len(dataset.datasets) != len(dataset_names):
        return None
    if not weights or all(float(value) == 1.0 for value in weights.values()):
        return None
    sample_weights: list[float] = []
    for name, child in zip(dataset_names, dataset.datasets, strict=True):
        sample_weights.extend([max(float(weights.get(name, 1.0)), 0.0)] * len(child))
    weight_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    if weight_tensor.numel() == 0 or float(weight_tensor.sum()) <= 0.0:
        raise ValueError("固定长度 stream 的 dataset sampling weights 不能全部为 0。")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weight_tensor,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def _posttrain_three_r_stream_ratio(project_config: ProjectConfig) -> float:
    ratio = float(project_config.marker_train.posttrain_three_r_stream_ratio)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("marker_train.posttrain_three_r_stream_ratio 必须在 [0, 1] 范围内。")
    return ratio


def _posttrain_stream_schedule(project_config: ProjectConfig) -> tuple[str, ...] | None:
    """Return an explicit finetune cycle, or None to use the legacy ratio."""
    train = project_config.marker_train
    raw = dict(train.posttrain_stream_schedule)
    if raw:
        marker_steps = int(raw.get("marker", 0))
        three_r_steps = int(raw.get("three_r", 0))
    elif int(train.posttrain_marker_stream_steps) != 1 or int(train.posttrain_three_r_stream_steps) != 1:
        marker_steps = int(train.posttrain_marker_stream_steps)
        three_r_steps = int(train.posttrain_three_r_stream_steps)
    else:
        return None
    if marker_steps < 0 or three_r_steps < 0 or marker_steps + three_r_steps <= 0:
        raise ValueError("finetune stream schedule 至少需要一个正周期，且周期不能为负数。")
    return ("marker",) * marker_steps + ("three_r",) * three_r_steps


def _dataset_split(project_config: ProjectConfig, dataset_name: str, *, for_training: bool) -> str:
    split = project_config.marker_data.common.split
    if not for_training:
        return split
    if dataset_name in MANAGED_SEQUENCE_SPLIT_DATASETS:
        return "trainval" if dataset_name == "h2o" else "train"
    if project_config.marker_data.common.exclude_h2o_test_split_from_training and split == "all" and dataset_name == "h2o":
        return "trainval"
    return split


def _needs_fixed_query_marker_chunk_filter(dataset: Dataset) -> bool:
    return False


def _fixed_query_marker_chunk_filter(dataset: Dataset, *, stage: str, for_training: bool) -> Dataset:
    if not for_training:
        return dataset
    if not _needs_fixed_query_marker_chunk_filter(dataset):
        return dataset
    return FilteredStageDataset(
        dataset,
        predicate=lambda chunk: chunk_allows_fixed_query_marker_supervision(chunk, stage=stage),
    )


def _fixed_query_random_marker_chunk_filter(dataset: Any, *, stage: str, for_training: bool) -> Any:
    if not for_training:
        return dataset
    if not _needs_fixed_query_marker_chunk_filter(dataset):
        return dataset
    return FilteredRandomLengthTemporalChunkDataset(
        dataset,
        predicate=lambda chunk: chunk_allows_fixed_query_marker_supervision(chunk, stage=stage),
        eager=False,
    )


def _dataset_kwargs_by_name_from_config(project_config: ProjectConfig) -> dict[str, dict[str, Any]]:
    kwargs_by_name: dict[str, dict[str, Any]] = {
        "h2o": {"include_scene_occlusion_in_visibility": True}
    }
    common = project_config.marker_data.common
    if common.hot3d_rectified_rgb_cache_root:
        kwargs_by_name["hot3d_aria"] = {
            "rgb_cache_root": common.hot3d_rectified_rgb_cache_root,
            "rgb_cache_max_bytes": common.hot3d_rectified_rgb_cache_max_bytes,
            "rgb_cache_worker_count": common.hot3d_rectified_rgb_cache_worker_count,
        }
    for dataset_name, root_override in project_config.paths.dataset_root_overrides.items():
        kwargs_by_name.setdefault(dataset_name, {})["root_override"] = root_override
    return kwargs_by_name


def _dataloader_worker_kwargs(common: Any) -> dict[str, Any]:
    """Keep worker prefetch settings inert for PyTorch's single-process loader."""
    if int(common.num_workers) <= 0:
        return {"persistent_workers": False}
    return {
        "persistent_workers": True,
        "pin_memory": bool(common.data_loader_pin_memory),
        "prefetch_factor": int(common.data_loader_prefetch_factor),
        "worker_init_fn": partial(_initialize_marker_data_worker, worker_threads=int(common.data_loader_worker_threads)),
    }


def _initialize_marker_data_worker(worker_id: int, *, worker_threads: int) -> None:
    """Bound CPU parallelism and prohibit accidental CUDA work in data workers."""
    del worker_id
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(int(worker_threads))
    try:
        import cv2

        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        # OpenCV is optional and some builds omit the OpenCL controls.
        pass


def build_marker_stage_chunk_bundle_from_config(
    project_config: ProjectConfig,
    *,
    stage: str,
    data_root: str | None = None,
    for_training: bool = False,
) -> StageDatasetBundle:
    common = project_config.marker_data.common
    resolved_data_root = data_root if data_root is not None else project_config.paths.data_root or None
    marker_dataset_names = _stage_marker_dataset_names(project_config, stage)
    three_r_dataset_names = _stage_three_r_dataset_names(project_config, stage)
    split_overrides = {
        name: _dataset_split(project_config, name, for_training=for_training)
        for name in {*marker_dataset_names, *three_r_dataset_names}
    }
    return build_stage_chunk_bundle(
        stage=stage,
        marker_dataset_names=marker_dataset_names,
        three_r_dataset_names=three_r_dataset_names,
        num_frames=common.num_frames,
        window_stride=common.window_stride,
        data_root=resolved_data_root,
        split=common.split,
        split_overrides=split_overrides,
        dataset_kwargs_by_name=_dataset_kwargs_by_name_from_config(project_config),
        load_rgb=True,
        load_depth=True,
    )


def build_marker_batch_collator_from_config(
    project_config: ProjectConfig,
    *,
    stage: str,
    stream_name: str | None = None,
) -> MarkerRuntimeCollator:
    model_root = (
        Path(project_config.paths.human_model_root)
        if project_config.paths.human_model_root
        else default_human_model_root_path()
    )
    mano_builder = None
    if model_root.exists():
        try:
            mano_builder = ManoVertexBuilder(model_root)
        except (FileNotFoundError, RuntimeError, ValueError):
            mano_builder = None
    model_config = (
        project_config.smoke_marker_model
        if project_config.marker_runtime.backend == "smoke"
        else project_config.marker_model
    )
    marker_vertex_ids = marker_vertex_ids_for_count(model_config.num_markers)
    vertex_visibility_mesh_mode = project_config.marker_runtime.vertex_visibility_mesh_mode
    if vertex_visibility_mesh_mode == "marker" and marker_faces_for_count(len(marker_vertex_ids)).numel() == 0:
        vertex_visibility_mesh_mode = "full"
    base_collator = MarkerBatchCollator(
        stage=stage,
        marker_vertex_ids=marker_vertex_ids,
        mano_vertex_builder=mano_builder,
        vertex_visibility_mesh_mode=vertex_visibility_mesh_mode,
        include_scene_occlusion_in_visibility=True,
        contact_supervision_by_dataset=project_config.marker_runtime.contact_supervision,
        stream_name=stream_name,
    )
    flow_pseudo_store = None
    if model_config.enable_flow_guidance:
        if not project_config.marker_runtime.flow_pseudo_cache_root:
            raise ValueError("启用 flow guidance 时 marker_runtime.flow_pseudo_cache_root 不能为空")
        flow_pseudo_store = FlowPseudoLabelStore(project_config.marker_runtime.flow_pseudo_cache_root)
    return MarkerRuntimeCollator(
        base_collator,
        target_height=project_config.marker_runtime.image_height,
        target_width=project_config.marker_runtime.image_width,
        random_crop_resize=project_config.marker_runtime.random_crop_resize,
        flow_pseudo_store=flow_pseudo_store,
        flow_fingertip_sigma_px=project_config.marker_runtime.flow_fingertip_sigma_px,
        seed=project_config.project.seed,
    )


def _random_crop_target_shapes(runtime: MarkerRuntimeConfig) -> list[list[int]] | None:
    random_crop_resize = runtime.random_crop_resize
    if not random_crop_resize.enabled:
        return None
    if random_crop_resize.target_shapes:
        return random_crop_resize.target_shapes
    return [[runtime.image_height, runtime.image_width]]


def build_marker_stage_loaders_from_config(
    project_config: ProjectConfig,
    *,
    stage: str,
    data_root: str | None = None,
    shuffle: bool,
    for_training: bool = False,
    max_samples: int | None = None,
    sample_seed: int | None = None,
    epoch_index: int | None = None,
    missing_sample_registry_dir: str | Path | None = None,
) -> MarkerStageLoaders:
    common = project_config.marker_data.common
    resolved_data_root = data_root if data_root is not None else project_config.paths.data_root or None
    build_start_time = time.perf_counter()
    _log_loader_stage(
        "开始实例化dataloader: stage={} shuffle={} for_training={} epoch_index={} batch_size={} num_workers={} window_stride={} frame_range=[{}, {}]",
        stage,
        shuffle,
        for_training,
        epoch_index,
        common.batch_size,
        common.num_workers,
        common.window_stride,
        common.min_num_frames if common.min_num_frames is not None else common.num_frames,
        common.max_num_frames if common.max_num_frames is not None else common.num_frames,
    )
    collator_start_time = time.perf_counter()
    _log_loader_stage("正在构建 MarkerRuntimeCollator: stage={} stream=marker", stage)
    marker_collator = build_marker_batch_collator_from_config(
        project_config,
        stage=stage,
        stream_name="marker",
    )
    three_r_collator = (
        marker_collator
        if stage == "midtrain"
        else build_marker_batch_collator_from_config(
            project_config,
            stage=stage,
            stream_name="three_r",
        )
    )
    collator_time_sec = time.perf_counter() - collator_start_time
    _log_loader_stage("MarkerRuntimeCollator 构建完成: stage={} elapsed={:.2f}s", stage, collator_time_sec)
    min_num_frames = common.min_num_frames if common.min_num_frames is not None else common.num_frames
    max_num_frames = common.max_num_frames if common.max_num_frames is not None else common.num_frames
    target_shapes = _random_crop_target_shapes(project_config.marker_runtime)
    temporal_stride_sampling = (
        int(common.min_frame_stride) != 1
        or int(common.max_frame_stride) != 1
        or bool(common.group_frame_stride_in_batch)
    )
    sampler_target_shapes = (
        target_shapes
        if target_shapes is not None
        else [[project_config.marker_runtime.image_height, project_config.marker_runtime.image_width]]
        if temporal_stride_sampling
        else None
    )
    augmented_random_sampling = sampler_target_shapes is not None
    use_random_length = shuffle and (
        min_num_frames != max_num_frames
        or augmented_random_sampling
        or bool(common.container_local_sampling_enabled)
    )
    control_mode = project_config.marker_train.control_mode
    marker_dataset_sampling_weights = _stage_dataset_sampling_weights(project_config, stage, "marker")
    three_r_dataset_sampling_weights = _stage_dataset_sampling_weights(project_config, stage, "three_r")
    streaming_epoch_sampler = bool(common.streaming_epoch_sampler)
    ddp_batch_group_size = _distributed_world_size_from_env()
    rank_sharded_batch_samplers = ddp_batch_group_size > 1 and use_random_length
    sticky_sequence_slots = max(common.num_workers, ddp_batch_group_size, 1)
    container_local_sampling_kwargs = {
        "container_local_sampling_enabled": common.container_local_sampling_enabled,
        "container_local_sampling_dataset_names": common.container_local_sampling_dataset_names,
        "container_local_sampling_block_batches": common.container_local_sampling_block_batches,
        "container_local_sampling_start_order": common.container_local_sampling_start_order,
        "container_local_prefetch_enabled": common.container_local_prefetch_enabled,
        "container_local_prefetch_margin_batches": common.container_local_prefetch_margin_batches,
    }
    epoch_control_enabled = (
        shuffle
        and for_training
        and epoch_index is not None
        and not streaming_epoch_sampler
    )
    marker_sampler_seed = project_config.project.seed if epoch_index is None else project_config.project.seed + int(epoch_index)
    three_r_sampler_seed = (
        project_config.project.seed + 1
        if epoch_index is None
        else project_config.project.seed + 1 + int(epoch_index)
    )
    dataset_start_time = time.perf_counter()
    if use_random_length:
        marker_dataset_names = _stage_marker_dataset_names(project_config, stage)
        three_r_dataset_names = _stage_three_r_dataset_names(project_config, stage)
        _log_loader_stage(
            "随机长度模式: stage={} marker_datasets={} three_r_datasets={} frame_range=[{}, {}]",
            stage,
            marker_dataset_names,
            three_r_dataset_names,
            min_num_frames,
            max_num_frames,
        )

        frame_dataset_cache: dict[tuple[str, str], Any] = {}
        dataset_kwargs_by_name = _dataset_kwargs_by_name_from_config(project_config)

        def _build_logged_frame_dataset(name: str) -> Any:
            dataset_split = _dataset_split(project_config, name, for_training=for_training)
            cache_key = (name, dataset_split)
            cached = frame_dataset_cache.get(cache_key)
            if cached is not None:
                _log_loader_stage("复用 frame dataset: name={} split={} stage={}", name, dataset_split, stage)
                return cached
            start_time = time.perf_counter()
            _log_loader_stage("正在构建 frame dataset: name={} split={} stage={}", name, dataset_split, stage)
            dataset = build_named_frame_dataset(
                name,
                data_root=resolved_data_root,
                split=dataset_split,
                load_rgb=True,
                load_depth=True,
                **dataset_kwargs_by_name.get(name, {}),
            )
            _log_loader_stage(
                "frame dataset 构建完成: name={} split={} frames={} sequences={} elapsed={:.2f}s",
                name,
                dataset_split,
                len(dataset),
                len(dataset.sequence_to_indices),
                time.perf_counter() - start_time,
            )
            frame_dataset_cache[cache_key] = dataset
            return dataset

        marker_frame_datasets = [_build_logged_frame_dataset(name) for name in marker_dataset_names]
        three_r_frame_datasets = [_build_logged_frame_dataset(name) for name in three_r_dataset_names]
        marker_frame_dataset = (
            marker_frame_datasets[0]
            if len(marker_frame_datasets) == 1
            else CombinedFrameDataset(marker_frame_datasets)
        )
        if len(three_r_frame_datasets) == 1:
            three_r_frame_dataset = three_r_frame_datasets[0]
        elif len(three_r_frame_datasets) > 1:
            three_r_frame_dataset = CombinedFrameDataset(three_r_frame_datasets)
        else:
            three_r_frame_dataset = None
        marker_dataset_start_time = time.perf_counter()
        _log_loader_stage("正在构建 marker_dataset(random_length): stage={}", stage)

        def _marker_random_dataset(dataset: Any) -> Any:
            return _fixed_query_random_marker_chunk_filter(
                RandomLengthTemporalChunkDataset(
                    dataset,
                    min_num_frames=min_num_frames,
                    max_num_frames=max_num_frames,
                    window_stride=common.window_stride,
                    drop_last=True,
                ),
                stage=stage,
                for_training=for_training,
            )

        marker_dataset = (
            MultiSourceRandomLengthTemporalChunkDataset(
                [_marker_random_dataset(dataset) for dataset in marker_frame_datasets],
                dataset_names=marker_dataset_names,
            )
            if len(marker_frame_datasets) > 1
            else _marker_random_dataset(marker_frame_dataset)
        )
        _log_loader_stage(
            "marker_dataset(random_length) 构建完成: stage={} anchors_at_min_len={} elapsed={:.2f}s",
            stage,
            len(marker_dataset),
            time.perf_counter() - marker_dataset_start_time,
        )
        three_r_dataset_start_time = time.perf_counter()
        _log_loader_stage("正在构建 three_r_dataset(random_length): stage={}", stage)
        if three_r_frame_datasets:
            three_r_dataset = (
                MultiSourceRandomLengthTemporalChunkDataset(
                    [
                        RandomLengthTemporalChunkDataset(
                            dataset,
                            min_num_frames=min_num_frames,
                            max_num_frames=max_num_frames,
                            window_stride=common.window_stride,
                            drop_last=True,
                        )
                        for dataset in three_r_frame_datasets
                    ],
                    dataset_names=three_r_dataset_names,
                )
                if len(three_r_frame_datasets) > 1
                else RandomLengthTemporalChunkDataset(
                    three_r_frame_dataset,
                    min_num_frames=min_num_frames,
                    max_num_frames=max_num_frames,
                    window_stride=common.window_stride,
                    drop_last=True,
                )
            )
        else:
            three_r_dataset = []
        _log_loader_stage(
            "three_r_dataset(random_length) 构建完成: stage={} anchors_at_min_len={} elapsed={:.2f}s",
            stage,
            len(three_r_dataset),
            time.perf_counter() - three_r_dataset_start_time,
        )
        bundle = StageDatasetBundle(
            stage=stage,
            marker_dataset=marker_dataset,
            three_r_dataset=three_r_dataset,
            marker_dataset_names=marker_dataset_names,
            three_r_dataset_names=three_r_dataset_names,
        )
        marker_epoch_controller = (
            RandomLengthEpochBatchSampler(
                bundle.marker_dataset,
                batch_size=common.batch_size,
                seed=project_config.project.seed,
                epoch=0 if epoch_index is None else epoch_index,
                dataset_sampling_weights=marker_dataset_sampling_weights if isinstance(bundle.marker_dataset, MultiSourceRandomLengthTemporalChunkDataset) else None,
                target_shapes=sampler_target_shapes,
                min_frame_stride=common.min_frame_stride,
                max_frame_stride=common.max_frame_stride,
                group_frame_stride_in_batch=common.group_frame_stride_in_batch,
                **container_local_sampling_kwargs,
            )
            if epoch_control_enabled
            else None
        )
        marker_base_batch_sampler = (
            marker_epoch_controller
            if epoch_control_enabled
            else RandomLengthBatchSampler(
                bundle.marker_dataset,
                batch_size=common.batch_size,
                seed=marker_sampler_seed,
                ddp_batch_group_size=ddp_batch_group_size,
                target_shapes=sampler_target_shapes,
                min_frame_stride=common.min_frame_stride,
                max_frame_stride=common.max_frame_stride,
                group_frame_stride_in_batch=common.group_frame_stride_in_batch,
                sequence_sampling_mode=common.sequence_sampling_mode,
                **container_local_sampling_kwargs,
            )
            if _is_random_length_dataset(bundle.marker_dataset)
            else WeightedRandomLengthBatchSampler(
                bundle.marker_dataset,
                batch_size=common.batch_size,
                seed=marker_sampler_seed,
                dataset_sampling_weights=marker_dataset_sampling_weights,
                sticky_sequence_slots=sticky_sequence_slots,
                ddp_batch_group_size=ddp_batch_group_size,
                target_shapes=sampler_target_shapes,
                min_frame_stride=common.min_frame_stride,
                max_frame_stride=common.max_frame_stride,
                group_frame_stride_in_batch=common.group_frame_stride_in_batch,
                sequence_sampling_mode=common.sequence_sampling_mode,
                **container_local_sampling_kwargs,
            )
        )
        marker_loader_start_time = time.perf_counter()
        _log_loader_stage("正在构建 marker_loader: stage={} epoch_index={}", stage, epoch_index)
        marker_loader_dataset, marker_collator = _missing_sample_retry_loader_parts(
            dataset=bundle.marker_dataset,
            collator=marker_collator,
            project_config=project_config,
            for_training=for_training,
            stream_name="marker",
            seed_offset=0,
            registry_dir=missing_sample_registry_dir,
        )
        marker_loader = DataLoader(
            marker_loader_dataset,
            batch_sampler=_rank_sharded_batch_sampler_from_env(marker_base_batch_sampler),
            num_workers=common.num_workers,
            collate_fn=_serialize_raw_chunks if int(common.num_workers) > 0 else marker_collator,
            **_dataloader_worker_kwargs(common),
        )
        marker_loader = _main_process_collating_loader(marker_loader, marker_collator, common)
        _log_loader_stage(
            "marker_loader 构建完成: stage={} batches={} elapsed={:.2f}s",
            stage,
            _loader_batch_count(marker_loader),
            time.perf_counter() - marker_loader_start_time,
        )
        three_r_epoch_controller = None
        three_r_loader_start_time = time.perf_counter()
        _log_loader_stage("正在构建 three_r_loader: stage={} epoch_index={}", stage, epoch_index)
        if len(bundle.three_r_dataset) == 0:
            three_r_loader = DataLoader([], batch_size=None)
        else:
            three_r_epoch_controller = (
                RandomLengthEpochBatchSampler(
                    bundle.three_r_dataset,
                    batch_size=common.batch_size,
                    seed=project_config.project.seed + 1,
                    epoch=0 if epoch_index is None else epoch_index,
                    dataset_sampling_weights=three_r_dataset_sampling_weights if isinstance(bundle.three_r_dataset, MultiSourceRandomLengthTemporalChunkDataset) else None,
                    target_shapes=sampler_target_shapes,
                    min_frame_stride=common.min_frame_stride,
                    max_frame_stride=common.max_frame_stride,
                    group_frame_stride_in_batch=common.group_frame_stride_in_batch,
                    **container_local_sampling_kwargs,
                )
                if epoch_control_enabled
                else None
            )
            three_r_base_batch_sampler = (
                three_r_epoch_controller
                if epoch_control_enabled
                else RandomLengthBatchSampler(
                    bundle.three_r_dataset,
                    batch_size=common.batch_size,
                    seed=three_r_sampler_seed,
                    ddp_batch_group_size=ddp_batch_group_size,
                    target_shapes=sampler_target_shapes,
                    min_frame_stride=common.min_frame_stride,
                    max_frame_stride=common.max_frame_stride,
                    group_frame_stride_in_batch=common.group_frame_stride_in_batch,
                    sequence_sampling_mode=common.sequence_sampling_mode,
                    **container_local_sampling_kwargs,
                )
                if _is_random_length_dataset(bundle.three_r_dataset)
                else WeightedRandomLengthBatchSampler(
                    bundle.three_r_dataset,
                    batch_size=common.batch_size,
                    seed=three_r_sampler_seed,
                    dataset_sampling_weights=three_r_dataset_sampling_weights,
                    sticky_sequence_slots=sticky_sequence_slots,
                    ddp_batch_group_size=ddp_batch_group_size,
                    target_shapes=sampler_target_shapes,
                    min_frame_stride=common.min_frame_stride,
                    max_frame_stride=common.max_frame_stride,
                    group_frame_stride_in_batch=common.group_frame_stride_in_batch,
                    sequence_sampling_mode=common.sequence_sampling_mode,
                    **container_local_sampling_kwargs,
                )
            )
            three_r_loader_dataset, three_r_collator = _missing_sample_retry_loader_parts(
                dataset=bundle.three_r_dataset,
                collator=three_r_collator,
                project_config=project_config,
                for_training=for_training,
                stream_name="three_r",
                seed_offset=1,
                registry_dir=missing_sample_registry_dir,
            )
            three_r_loader = DataLoader(
                three_r_loader_dataset,
                batch_sampler=_rank_sharded_batch_sampler_from_env(three_r_base_batch_sampler),
                num_workers=common.num_workers,
                collate_fn=_serialize_raw_chunks if int(common.num_workers) > 0 else three_r_collator,
                **_dataloader_worker_kwargs(common),
            )
            three_r_loader = _main_process_collating_loader(three_r_loader, three_r_collator, common)
        _log_loader_stage(
            "three_r_loader 构建完成: stage={} batches={} elapsed={:.2f}s",
            stage,
            _loader_batch_count(three_r_loader),
            time.perf_counter() - three_r_loader_start_time,
        )
    else:
        _log_loader_stage("固定长度模式: stage={} num_frames={}", stage, common.num_frames)
        bundle_build_start_time = time.perf_counter()
        _log_loader_stage("正在构建 stage chunk bundle: stage={}", stage)
        bundle = build_marker_stage_chunk_bundle_from_config(project_config, stage=stage, data_root=data_root, for_training=for_training)
        bundle = StageDatasetBundle(
            stage=bundle.stage,
            marker_dataset=_fixed_query_marker_chunk_filter(
                bundle.marker_dataset,
                stage=stage,
                for_training=for_training,
            ),
            three_r_dataset=bundle.three_r_dataset,
            marker_dataset_names=bundle.marker_dataset_names,
            three_r_dataset_names=bundle.three_r_dataset_names,
        )
        _log_loader_stage(
            "stage chunk bundle 构建完成: stage={} marker_samples={} three_r_samples={} elapsed={:.2f}s",
            stage,
            bundle.marker_sample_count,
            bundle.three_r_sample_count,
            time.perf_counter() - bundle_build_start_time,
        )
        if max_samples is not None and max_samples > 0:
            def _subset_dataset(dataset: Any, *, seed_offset: int) -> Any:
                if len(dataset) <= max_samples:
                    return dataset
                generator = torch.Generator()
                generator.manual_seed(
                    project_config.project.seed + seed_offset if sample_seed is None else sample_seed + seed_offset
                )
                indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
                return Subset(dataset, indices)

            marker_dataset = _subset_dataset(bundle.marker_dataset, seed_offset=0)
            three_r_dataset = _subset_dataset(bundle.three_r_dataset, seed_offset=1)
            bundle = StageDatasetBundle(
                stage=bundle.stage,
                marker_dataset=marker_dataset,
                three_r_dataset=three_r_dataset,
                marker_dataset_names=bundle.marker_dataset_names,
                three_r_dataset_names=bundle.three_r_dataset_names,
            )
        generator = None
        marker_epoch_controller = None
        three_r_epoch_controller = None
        marker_generator = None
        if shuffle:
            marker_base_seed = project_config.project.seed if sample_seed is None else int(sample_seed)
            if epoch_control_enabled:
                marker_generator = torch.Generator()
                marker_epoch_controller = EpochGeneratorController(marker_generator, marker_base_seed)
                marker_epoch_controller.set_epoch(0 if epoch_index is None else epoch_index)
            else:
                marker_generator = torch.Generator()
                marker_generator.manual_seed(marker_base_seed if epoch_index is None else marker_base_seed + int(epoch_index))
        marker_loader_start_time = time.perf_counter()
        _log_loader_stage("正在构建 marker_loader: stage={} epoch_index={}", stage, epoch_index)
        marker_loader_dataset, marker_collator = _missing_sample_retry_loader_parts(
            dataset=bundle.marker_dataset,
            collator=marker_collator,
            project_config=project_config,
            for_training=for_training,
            stream_name="marker",
            seed_offset=0,
            registry_dir=missing_sample_registry_dir,
        )
        marker_weighted_sampler = _fixed_length_weighted_sampler(
            bundle.marker_dataset,
            bundle.marker_dataset_names,
            marker_dataset_sampling_weights,
            seed=marker_base_seed,
        ) if shuffle else None
        marker_loader = DataLoader(
            marker_loader_dataset,
            batch_size=common.batch_size,
            shuffle=shuffle and marker_weighted_sampler is None,
            sampler=marker_weighted_sampler,
            num_workers=common.num_workers,
            collate_fn=_serialize_raw_chunks if int(common.num_workers) > 0 else marker_collator,
            generator=marker_generator,
            **_dataloader_worker_kwargs(common),
        )
        marker_loader = _main_process_collating_loader(marker_loader, marker_collator, common)
        _log_loader_stage(
            "marker_loader 构建完成: stage={} batches={} elapsed={:.2f}s",
            stage,
            _loader_batch_count(marker_loader),
            time.perf_counter() - marker_loader_start_time,
        )
        three_r_generator = None
        if shuffle:
            three_r_base_seed = project_config.project.seed + 1 if sample_seed is None else int(sample_seed) + 1
            if epoch_control_enabled:
                three_r_generator = torch.Generator()
                three_r_epoch_controller = EpochGeneratorController(three_r_generator, three_r_base_seed)
                three_r_epoch_controller.set_epoch(0 if epoch_index is None else epoch_index)
            else:
                three_r_generator = torch.Generator()
                three_r_generator.manual_seed(
                    three_r_base_seed if epoch_index is None else three_r_base_seed + int(epoch_index)
                )
        three_r_loader_start_time = time.perf_counter()
        _log_loader_stage("正在构建 three_r_loader: stage={} epoch_index={}", stage, epoch_index)
        if len(bundle.three_r_dataset) == 0:
            three_r_loader = DataLoader([], batch_size=None)
        else:
            three_r_loader_dataset, three_r_collator = _missing_sample_retry_loader_parts(
                dataset=bundle.three_r_dataset,
                collator=three_r_collator,
                project_config=project_config,
                for_training=for_training,
                stream_name="three_r",
                seed_offset=1,
                registry_dir=missing_sample_registry_dir,
            )
            three_r_weighted_sampler = _fixed_length_weighted_sampler(
                bundle.three_r_dataset,
                bundle.three_r_dataset_names,
                three_r_dataset_sampling_weights,
                seed=three_r_base_seed if shuffle else project_config.project.seed + 1,
            ) if shuffle else None
            three_r_loader = DataLoader(
                three_r_loader_dataset,
                batch_size=common.batch_size,
                shuffle=shuffle and three_r_weighted_sampler is None,
                sampler=three_r_weighted_sampler,
                num_workers=common.num_workers,
                collate_fn=_serialize_raw_chunks if int(common.num_workers) > 0 else three_r_collator,
                generator=three_r_generator,
                **_dataloader_worker_kwargs(common),
            )
            three_r_loader = _main_process_collating_loader(three_r_loader, three_r_collator, common)
        _log_loader_stage(
            "three_r_loader 构建完成: stage={} batches={} elapsed={:.2f}s",
            stage,
            _loader_batch_count(three_r_loader),
            time.perf_counter() - three_r_loader_start_time,
        )
    dataset_time_sec = time.perf_counter() - dataset_start_time
    build_time_sec = time.perf_counter() - build_start_time
    _log_loader_details(
        stage=stage,
        marker_loader=marker_loader,
        three_r_loader=three_r_loader,
        bundle=bundle,
        build_time_sec=build_time_sec,
        collator_time_sec=collator_time_sec,
        dataset_time_sec=dataset_time_sec,
        epoch_index=epoch_index,
        control_mode=control_mode,
        dataset_sampling_weights=marker_dataset_sampling_weights,
    )
    return MarkerStageLoaders(
        stage=stage,
        bundle=bundle,
        marker_loader=marker_loader,
        three_r_loader=three_r_loader,
        posttrain_three_r_stream_ratio=_posttrain_three_r_stream_ratio(project_config),
        posttrain_stream_schedule=_posttrain_stream_schedule(project_config),
        marker_epoch_controller=marker_epoch_controller,
        three_r_epoch_controller=three_r_epoch_controller,
        rank_sharded=rank_sharded_batch_samplers,
    )


def build_wilor_teacher_from_config(project_config: ProjectConfig) -> WiLorTeacherWrapper | None:
    runtime = project_config.marker_runtime
    if not runtime.enable_wilor_teacher:
        return None
    checkpoint_path = Path(runtime.wilor_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"WiLoR checkpoint 不存在: {checkpoint_path}")
    cfg_path = None if not runtime.wilor_config_path else Path(runtime.wilor_config_path)
    return WiLorTeacherWrapper.from_vendor(
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        pooled_dim=runtime.wilor_teacher_feature_dim,
    )


def iter_mixed_stage_batches(loaders: MarkerStageLoaders) -> DualStreamBatchMixer:
    return DualStreamBatchMixer(loaders.marker_loader, loaders.three_r_loader)


class _JointStageBatchIterator:
    """Stateful iterator; schedule_position is checkpointable by the trainer."""

    def __init__(self, loaders: MarkerStageLoaders, *, start_schedule_position: int = 0) -> None:
        self.loaders = loaders
        self.marker_iter = iter(loaders.marker_loader)
        self.three_r_iter = iter(loaders.three_r_loader)
        self.marker_done = _loader_batch_count(loaders.marker_loader) == 0
        self.three_r_done = _loader_batch_count(loaders.three_r_loader) == 0
        self.schedule = loaders.posttrain_stream_schedule
        self.schedule_position: int | None = int(start_schedule_position) if self.schedule else None
        self.marker_index = 0

    def __iter__(self) -> "_JointStageBatchIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self.loaders.stage == "midtrain":
            if self.marker_done:
                raise StopIteration
            try:
                return {"marker": next(self.marker_iter)}
            except StopIteration:
                self.marker_done = True
                raise
        if self.loaders.stage == "posttrain" and self.schedule:
            assert self.schedule_position is not None
            while not (self.marker_done and self.three_r_done):
                stream_name = self.schedule[self.schedule_position % len(self.schedule)]
                self.schedule_position += 1
                if stream_name == "marker":
                    if self.marker_done:
                        continue
                    try:
                        return {"marker": next(self.marker_iter)}
                    except StopIteration:
                        self.marker_done = True
                else:
                    if self.three_r_done:
                        continue
                    try:
                        return {"three_r": next(self.three_r_iter)}
                    except StopIteration:
                        self.three_r_done = True
            raise StopIteration
        if self.loaders.stage == "posttrain":
            ratio = float(self.loaders.posttrain_three_r_stream_ratio)
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError("posttrain_three_r_stream_ratio 必须在 [0, 1] 范围内。")
            if self.marker_done:
                raise StopIteration
            try:
                marker_batch = next(self.marker_iter)
            except StopIteration:
                self.marker_done = True
                raise
            result: dict[str, Any] = {"marker": marker_batch}
            include_three_r = (
                not self.three_r_done
                and ratio > 0.0
                and int(float(self.marker_index + 1) * ratio) > int(float(self.marker_index) * ratio)
            )
            self.marker_index += 1
            if include_three_r:
                try:
                    result["three_r"] = next(self.three_r_iter)
                except StopIteration:
                    self.three_r_iter = iter(self.loaders.three_r_loader)
                    result["three_r"] = next(self.three_r_iter)
            return result
        while not (self.marker_done and self.three_r_done):
            result: dict[str, Any] = {}
            if not self.marker_done:
                try:
                    result["marker"] = next(self.marker_iter)
                except StopIteration:
                    self.marker_done = True
            if not self.three_r_done:
                try:
                    result["three_r"] = next(self.three_r_iter)
                except StopIteration:
                    self.three_r_done = True
            if result:
                return result
        raise StopIteration


def iter_joint_stage_batches(
    loaders: MarkerStageLoaders,
    *,
    start_schedule_position: int = 0,
) -> _JointStageBatchIterator:
    return _JointStageBatchIterator(loaders, start_schedule_position=start_schedule_position)


def average_stream_losses(stream_losses: dict[str, torch.Tensor]) -> torch.Tensor:
    if not stream_losses:
        raise ValueError("stream_losses 不能为空。")
    total_loss: torch.Tensor | None = None
    for loss in stream_losses.values():
        total_loss = loss if total_loss is None else total_loss + loss
    assert total_loss is not None
    return total_loss / float(len(stream_losses))


def save_marker_inference_output(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_marker_inference_output(path: str | Path) -> dict[str, Any]:
    output_path = Path(path)
    if not output_path.exists():
        raise FileNotFoundError(f"推理结果不存在: {output_path}")
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"推理结果格式错误: {output_path}")
    return payload


def _build_vertex_targets(
    batch: dict[str, torch.Tensor],
    *,
    num_vertices: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    vertex_xyz_targets = _fixed_side_tensor(batch, "vertex_xyz_targets").to(device=device, dtype=dtype)
    vertex_xyz_mask = _fixed_side_tensor(batch, "vertex_xyz_supervision_mask").to(device=device)
    vertex_visibility_targets = _fixed_side_tensor(batch, "vertex_visibility_targets").to(device=device, dtype=dtype)
    vertex_visibility_mask = _fixed_side_tensor(batch, "vertex_visibility_supervision_mask").to(device=device)
    vertex_count = min(num_vertices, vertex_xyz_targets.shape[3])
    return (
        vertex_xyz_targets[..., :vertex_count, :],
        vertex_xyz_mask[..., :vertex_count],
        vertex_visibility_targets[..., :vertex_count],
        vertex_visibility_mask[..., :vertex_count],
    )


def _build_joint_targets(
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_xyz_targets = _fixed_side_tensor(batch, "joints_3d_targets").to(device=device, dtype=dtype)
    joint_xyz_mask = _fixed_side_tensor(batch, "raw_joint_supervision_mask").to(device=device).unsqueeze(-1).expand(
        joint_xyz_targets.shape[0],
        joint_xyz_targets.shape[1],
        joint_xyz_targets.shape[2],
        joint_xyz_targets.shape[3],
    )
    joint_xyz_mask = joint_xyz_mask & torch.isfinite(joint_xyz_targets).all(dim=-1)
    joint_visibility_targets = _fixed_side_tensor(batch, "joint_visibility_targets").to(device=device, dtype=dtype)
    joint_visibility_mask = _fixed_side_tensor(batch, "joint_visibility_supervision_mask").to(device=device)
    return (
        torch.nan_to_num(joint_xyz_targets),
        joint_visibility_targets,
        joint_xyz_mask,
        joint_xyz_mask & joint_visibility_mask,
    )


def _hand_slot_average_distance(
    *,
    vertex_xyz: torch.Tensor,
    vertex_mask: torch.Tensor,
    joint_xyz: torch.Tensor | None = None,
    joint_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    slot_shape = vertex_xyz.shape[:3]
    point_chunks = [vertex_xyz.reshape(-1, vertex_xyz.shape[3], 3)]
    mask_chunks = [vertex_mask.reshape(-1, vertex_mask.shape[3])]
    if joint_xyz is not None and joint_mask is not None:
        point_chunks.append(joint_xyz.reshape(-1, joint_xyz.shape[3], 3))
        mask_chunks.append(joint_mask.reshape(-1, joint_mask.shape[3]))
    flat_points = torch.cat(point_chunks, dim=1)
    flat_mask = torch.cat(mask_chunks, dim=1)
    return masked_average_distance(flat_points, flat_mask).reshape(slot_shape)


def _normalize_slot_points(points: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return points / scale.clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)


def _effective_after_warmup_weight(weight: float, warmup_steps: int, global_step: int | None) -> float:
    if float(weight) == 0.0:
        return 0.0
    if int(warmup_steps) <= 0 or global_step is None:
        return float(weight)
    return 0.0 if int(global_step) < int(warmup_steps) else float(weight)


def _record_loss_metrics(
    metrics: dict[str, float],
    key: str,
    loss: torch.Tensor,
    weight: float,
    valid_mask: torch.Tensor | None = None,
) -> None:
    metrics[key] = float(loss.detach().item())
    metrics[f"{key}_weighted"] = float((loss.detach() * float(weight)).item())
    if valid_mask is not None:
        count_key = f"{key[:-5]}_valid_count" if key.endswith("_loss") else f"{key}_valid_count"
        metrics[count_key] = float(valid_mask.to(dtype=torch.float32).sum().detach().item())


def _masked_log_distance_huber_loss(
    prediction: torch.Tensor,
    target_meters: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor | None:
    if prediction.shape != target_meters.shape or valid_mask.shape != prediction.shape:
        raise ValueError("contact distance prediction/target/mask 形状必须一致。")
    if not torch.any(valid_mask):
        return None
    # 毫米尺度的 log1p 保留近接触区域的分辨率，并避免远距离主导辅助任务。
    target_log_distance = torch.log1p(target_meters.clamp_min(0.0) * 1000.0)
    return F.smooth_l1_loss(prediction[valid_mask], target_log_distance[valid_mask])


def _record_contact_confidence_metrics(
    metrics: dict[str, float],
    prefix: str,
    stats: dict[str, float],
    *,
    enabled: bool,
) -> None:
    metrics[f"{prefix}_confidence_enabled"] = 1.0 if enabled else 0.0
    if not stats:
        return
    metrics[f"{prefix}_loss_raw"] = stats["raw"]
    metrics[f"{prefix}_loss_conf"] = stats["conf"]
    metrics[f"{prefix}_conf_mean"] = stats["conf_mean"]
    metrics[f"{prefix}_conf_p10"] = stats["conf_p10"]
    metrics[f"{prefix}_conf_p50"] = stats["conf_p50"]
    metrics[f"{prefix}_conf_p90"] = stats["conf_p90"]
    metrics[f"{prefix}_log_conf_mean"] = stats["log_conf_mean"]


def _hand_scene_abs_conf_loss(
    pred_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    point_mask: torch.Tensor,
    log_conf: torch.Tensor,
    pred_scene_scale: torch.Tensor,
    gt_scene_scale: torch.Tensor,
    *,
    alpha: float,
    log_min: float,
    log_max: float,
    use_confidence: bool,
    max_sample_loss: float,
) -> tuple[torch.Tensor, dict[str, float], int]:
    pred_norm = pred_xyz / pred_scene_scale.view(-1, 1, 1, 1, 1).clamp_min(1e-6)
    target_norm = target_xyz / gt_scene_scale.view(-1, 1, 1, 1, 1).clamp_min(1e-6)
    error = F.smooth_l1_loss(pred_norm, torch.nan_to_num(target_norm), reduction="none").mean(dim=-1)
    point_mask, dropped = _drop_sequence_samples_by_loss(
        error,
        point_mask,
        max_sample_loss=max_sample_loss,
    )
    loss, stats = hand_confidence_weighted_loss(
        error,
        log_conf,
        point_mask,
        alpha=alpha,
        log_min=log_min,
        log_max=log_max,
        use_confidence=use_confidence,
    )
    return loss, stats, dropped


def _hand_metric_abs_conf_loss(
    pred_xyz: torch.Tensor,
    target_metric_xyz: torch.Tensor,
    point_mask: torch.Tensor,
    log_conf: torch.Tensor,
    pred_scene_scale: torch.Tensor,
    metric_value: torch.Tensor,
    *,
    alpha: float,
    log_min: float,
    log_max: float,
    use_confidence: bool,
    max_sample_loss: float,
) -> tuple[torch.Tensor, dict[str, float], int]:
    scale_factor = (metric_value.detach() / pred_scene_scale.clamp_min(1e-6)).view(-1, 1, 1, 1, 1)
    pred_metric = pred_xyz * scale_factor
    error = F.smooth_l1_loss(pred_metric, torch.nan_to_num(target_metric_xyz), reduction="none").mean(dim=-1)
    point_mask, dropped = _drop_sequence_samples_by_loss(
        error,
        point_mask,
        max_sample_loss=max_sample_loss,
    )
    loss, stats = hand_confidence_weighted_loss(
        error,
        log_conf,
        point_mask,
        alpha=alpha,
        log_min=log_min,
        log_max=log_max,
        use_confidence=use_confidence,
    )
    return loss, stats, dropped


def _select_hand_metric_value(
    log_metric_value: Any,
    batch_metric_value: Any,
) -> torch.Tensor | None:
    if not isinstance(log_metric_value, torch.Tensor):
        return None
    pred_metric_value = torch.exp(log_metric_value)
    if not isinstance(batch_metric_value, torch.Tensor):
        return pred_metric_value
    gt_metric_value = batch_metric_value.to(
        device=pred_metric_value.device,
        dtype=pred_metric_value.dtype,
    )
    if gt_metric_value.ndim != 1 or gt_metric_value.shape[0] != pred_metric_value.shape[0]:
        raise ValueError("batch['metric_value'] must have shape [B].")
    valid_gt = torch.isfinite(gt_metric_value) & (gt_metric_value > 0)
    return torch.where(valid_gt, gt_metric_value, pred_metric_value)


def _normalized_intrinsics_vector(
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    fov_h = 2.0 * torch.atan((float(image_height) / 2.0) / fy.clamp_min(1e-6))
    fov_w = 2.0 * torch.atan((float(image_width) / 2.0) / fx.clamp_min(1e-6))
    return torch.stack(
        [
            fov_h,
            fov_w,
            cx / max(float(image_width), 1.0),
            cy / max(float(image_height), 1.0),
        ],
        dim=-1,
    )


def _normalize_sequence_points(points: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return points / scale.clamp_min(1e-6).view(scale.shape[0], 1, 1, 1, 1)


def _scene_scale_sequence_mask(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    depth_for_mask = depth.squeeze(-1) if depth.ndim == 5 and depth.shape[-1] == 1 else depth
    valid = torch.isfinite(depth_for_mask) & (depth_for_mask > 0)
    valid &= torch.isfinite(intrinsics).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    valid &= torch.isfinite(camera_pose).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    pose_det = torch.linalg.det(camera_pose.float()).abs().to(device=depth.device)
    valid &= (pose_det > 1e-6).unsqueeze(-1).unsqueeze(-1)
    if valid_mask is not None:
        valid &= valid_mask.to(device=depth.device, dtype=torch.bool)
    return valid.reshape(valid.shape[0], -1).any(dim=1)


def _valid_scene_scale_mask(scene_scale: torch.Tensor, *, min_value: float) -> torch.Tensor:
    valid = torch.isfinite(scene_scale)
    if float(min_value) > 0.0:
        return valid & (scene_scale >= float(min_value))
    return valid & (scene_scale > 0)


def _drop_sequence_samples_by_loss(
    loss_values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_sample_loss: float,
) -> tuple[torch.Tensor, int]:
    valid = valid_mask.to(device=loss_values.device, dtype=torch.bool)
    valid &= torch.isfinite(loss_values)
    if float(max_sample_loss) <= 0.0 or not torch.any(valid):
        return valid, 0
    batch_size = loss_values.shape[0]
    flat_valid = valid.reshape(batch_size, -1)
    flat_loss = torch.where(valid, torch.nan_to_num(loss_values), torch.zeros_like(loss_values)).reshape(batch_size, -1)
    counts = flat_valid.sum(dim=1)
    per_sample_valid = counts > 0
    per_sample_loss = flat_loss.sum(dim=1) / counts.clamp_min(1).to(dtype=loss_values.dtype)
    keep = per_sample_valid & torch.isfinite(per_sample_loss) & (
        per_sample_loss.detach() <= float(max_sample_loss)
    )
    dropped = int((per_sample_valid & ~keep).sum().item())
    keep_view = keep.view(batch_size, *((1,) * (valid.ndim - 1)))
    return valid & keep_view, dropped


def _record_scene_scale_guard_metrics(
    metrics: dict[str, float],
    prefix: str,
    scene_scale: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    with torch.no_grad():
        invalid = (~valid).to(dtype=torch.float32).sum()
        metrics[f"{prefix}_invalid_samples"] = float(invalid.item())
        finite = torch.isfinite(scene_scale)
        if torch.any(finite):
            values = scene_scale[finite].detach().float()
            metrics[f"{prefix}_min"] = float(values.min().item())
            metrics[f"{prefix}_p50"] = float(torch.quantile(values, 0.50).item())


def compute_pred_scene_scale(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
) -> torch.Tensor:
    return scene_average_distance_from_depth(
        depth.detach(),
        intrinsics.detach(),
        camera_pose.detach(),
    )


def _apply_batch_scale(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] == scale.shape[0]:
        tensor_scale = scale
    else:
        if tensor.shape[0] % scale.shape[0] != 0:
            raise ValueError(
                f"Cannot apply batch scale with shape {tuple(scale.shape)} to tensor shape {tuple(tensor.shape)}"
            )
        tensor_scale = scale.repeat_interleave(tensor.shape[0] // scale.shape[0])
    view_shape = (tensor_scale.shape[0],) + (1,) * (tensor.ndim - 1)
    return tensor * tensor_scale.view(view_shape).to(device=tensor.device, dtype=tensor.dtype)


_METRIC_SCALE_HAND_KEYS = (
    "dense_vertex_xyz",
    "dense_joint_xyz",
    "dense_hand_root_xyz",
    "prompt_vertex_xyz",
    "prompt_joint_xyz",
    "prompt_hand_root_xyz",
)


def reconstruct_metric_scale_outputs(
    outputs: dict[str, Any],
    *,
    eps: float = 1e-6,
) -> dict[str, Any]:
    reconstructed = dict(outputs)
    depth = outputs.get("depth")
    intrinsics = outputs.get("intrinsics")
    camera_pose = outputs.get("camera_pose")
    metric_value = outputs.get("metric_value")
    can_scale = (
        isinstance(depth, torch.Tensor)
        and depth.ndim in (4, 5)
        and isinstance(intrinsics, torch.Tensor)
        and intrinsics.ndim == 4
        and intrinsics.shape[-2:] == (3, 3)
        and isinstance(camera_pose, torch.Tensor)
        and camera_pose.ndim == 4
        and camera_pose.shape[-2:] == (4, 4)
        and isinstance(metric_value, torch.Tensor)
        and metric_value.ndim == 1
        and metric_value.shape[0] == depth.shape[0]
    )
    if not can_scale:
        reconstructed["s_pred_scene"] = None
        reconstructed["metric_scale_factor"] = None
        return reconstructed

    s_pred_scene = compute_pred_scene_scale(depth, intrinsics, camera_pose)
    metric_value_detached = metric_value.detach().to(device=s_pred_scene.device, dtype=s_pred_scene.dtype)
    scale_factor = metric_value_detached / s_pred_scene.clamp_min(eps)
    reconstructed["s_pred_scene"] = s_pred_scene
    reconstructed["metric_scale_factor"] = scale_factor

    reconstructed["depth"] = _apply_batch_scale(depth, scale_factor)
    camera_pose_metric = camera_pose.clone()
    camera_pose_metric[..., :3, 3] = _apply_batch_scale(camera_pose[..., :3, 3], scale_factor)
    reconstructed["camera_pose"] = camera_pose_metric
    for key in _METRIC_SCALE_HAND_KEYS:
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            reconstructed[key] = _apply_batch_scale(value, scale_factor)
    return reconstructed


def _joint_depth_anchor_mask(batch: dict[str, Any], joint_xyz_mask: torch.Tensor) -> torch.Tensor | None:
    if "joints_2d_targets" not in batch or "joints_2d_supervision_mask" not in batch:
        return None
    device = joint_xyz_mask.device
    joint_2d_mask = _fixed_side_tensor(batch, "joints_2d_supervision_mask").to(device=device).unsqueeze(-1)
    anchor_mask = joint_xyz_mask & joint_2d_mask
    if "joint_visibility_targets" in batch and "joint_visibility_supervision_mask" in batch:
        visibility_targets = _fixed_side_tensor(batch, "joint_visibility_targets").to(
            device=device,
            dtype=torch.bool,
        )
        visibility_mask = _fixed_side_tensor(batch, "joint_visibility_supervision_mask").to(
            device=device,
            dtype=torch.bool,
        )
        anchor_mask = anchor_mask & (~visibility_mask | visibility_targets)
    return anchor_mask


def _hand_2d_reprojection_loss(
    points_xyz: torch.Tensor,
    target_uv: torch.Tensor,
    supervision_mask: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    min_valid_z: float = 1e-4,
    normalized_error_max: float = 0.0,
) -> torch.Tensor:
    if target_uv.shape[:-1] != points_xyz.shape[:-1] or target_uv.shape[-1] != 2:
        raise ValueError("target_uv must match points_xyz except for uv/xyz dimensions")
    if supervision_mask.shape not in (points_xyz.shape[:3], points_xyz.shape[:-1]):
        raise ValueError("supervision_mask must match points_xyz batch/frame/slot or point dimensions")
    if intrinsics.shape[:2] != points_xyz.shape[:2] or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B, T, 3, 3]")

    intrinsics = intrinsics.detach().to(device=points_xyz.device, dtype=points_xyz.dtype)
    finite_intrinsics = torch.isfinite(intrinsics).all(dim=(-2, -1))
    safe_intrinsics = torch.nan_to_num(intrinsics)
    finite_points = torch.isfinite(points_xyz).all(dim=-1)
    safe_points_xyz = torch.where(finite_points.unsqueeze(-1), points_xyz, torch.zeros_like(points_xyz))
    z = safe_points_xyz[..., 2]
    fx = safe_intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = safe_intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = safe_intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = safe_intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
    safe_z = z.clamp_min(min_valid_z)
    u = fx * (safe_points_xyz[..., 0] / safe_z) + cx
    v = fy * (safe_points_xyz[..., 1] / safe_z) + cy
    pred_uv_norm = torch.stack(
        [
            u / max(float(image_width), 1.0),
            v / max(float(image_height), 1.0),
        ],
        dim=-1,
    )
    target_scale = torch.tensor(
        [max(float(image_width), 1.0), max(float(image_height), 1.0)],
        device=points_xyz.device,
        dtype=points_xyz.dtype,
    )
    target_uv_norm = target_uv.to(device=points_xyz.device, dtype=points_xyz.dtype) / target_scale
    if supervision_mask.shape == points_xyz.shape[:3]:
        valid = supervision_mask.to(device=points_xyz.device, dtype=torch.bool).unsqueeze(-1).expand(points_xyz.shape[:-1]).clone()
    else:
        valid = supervision_mask.to(device=points_xyz.device, dtype=torch.bool).clone()
    valid &= finite_intrinsics.unsqueeze(-1).unsqueeze(-1)
    valid &= finite_points
    valid &= torch.isfinite(target_uv_norm).all(dim=-1)
    valid &= z > min_valid_z
    valid &= torch.isfinite(pred_uv_norm).all(dim=-1)
    if not torch.any(valid):
        return points_xyz.sum() * 0.0
    uv_error = pred_uv_norm[valid] - target_uv_norm[valid]
    if float(normalized_error_max) > 0.0:
        uv_error = uv_error.clamp(
            min=-float(normalized_error_max),
            max=float(normalized_error_max),
        )
    return F.smooth_l1_loss(uv_error, torch.zeros_like(uv_error))


MANO_JOINT_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def _project_pred_hand_points_to_uv(
    points_xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    min_valid_z: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if intrinsics.shape[:2] != points_xyz.shape[:2] or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B, T, 3, 3]")

    intrinsics = intrinsics.detach().to(device=points_xyz.device, dtype=points_xyz.dtype)
    finite_intrinsics = torch.isfinite(intrinsics).all(dim=(-2, -1))
    safe_intrinsics = torch.nan_to_num(intrinsics)
    finite_points = torch.isfinite(points_xyz).all(dim=-1)
    safe_points_xyz = torch.where(finite_points.unsqueeze(-1), points_xyz, torch.zeros_like(points_xyz))
    z = safe_points_xyz[..., 2]
    safe_z = z.clamp_min(min_valid_z)
    fx = safe_intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = safe_intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = safe_intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = safe_intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
    uv = torch.stack(
        [
            fx * (safe_points_xyz[..., 0] / safe_z) + cx,
            fy * (safe_points_xyz[..., 1] / safe_z) + cy,
        ],
        dim=-1,
    )
    valid = finite_intrinsics.unsqueeze(-1).unsqueeze(-1)
    valid = valid & finite_points
    valid = valid & (z > min_valid_z)
    valid = valid & torch.isfinite(uv).all(dim=-1)
    return uv, valid


def _hand_joint_2d_bone_length_loss(
    pred_uv: torch.Tensor,
    target_uv: torch.Tensor,
    point_mask: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    min_target_bone_length_px: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pred_uv.shape != target_uv.shape or pred_uv.shape[-1] != 2:
        raise ValueError("pred_uv and target_uv must have the same [..., 2] shape")
    if point_mask.shape != pred_uv.shape[:-1]:
        raise ValueError("point_mask must match pred_uv except for uv dimension")
    if pred_uv.shape[-2] < 21:
        raise ValueError("joint bone length loss requires at least 21 joints")

    pred_uv_for_loss = pred_uv.to(dtype=torch.float32)
    target_uv = target_uv.to(device=pred_uv.device, dtype=pred_uv.dtype)
    target_uv_for_loss = target_uv.to(dtype=torch.float32)
    point_mask = point_mask.to(device=pred_uv.device, dtype=torch.bool)
    edges = torch.as_tensor(MANO_JOINT_EDGES, device=pred_uv.device, dtype=torch.long)
    start = edges[:, 0]
    end = edges[:, 1]
    pred_len = torch.linalg.norm(pred_uv_for_loss[..., start, :] - pred_uv_for_loss[..., end, :], dim=-1)
    target_len = torch.linalg.norm(target_uv_for_loss[..., start, :] - target_uv_for_loss[..., end, :], dim=-1)
    valid = point_mask[..., start] & point_mask[..., end]
    valid = valid & torch.isfinite(pred_len) & torch.isfinite(target_len)
    valid = valid & (target_len >= float(min_target_bone_length_px))
    if not torch.any(valid):
        return pred_uv.sum() * 0.0, {"valid_count": 0.0}

    image_scale = float(max(int(image_height), int(image_width), 1))
    error_px = (pred_len[valid] - target_len[valid]).abs()
    loss = (error_px / image_scale).mean()
    ratio = pred_len[valid].detach() / target_len[valid].detach().clamp_min(1.0)
    error_px_float = error_px.detach().float()
    stats = {
        "valid_count": float(valid.to(dtype=torch.float32).sum().detach().item()),
        "error_px_mean": float(error_px_float.mean().item()),
        "error_px_p90": float(torch.quantile(error_px_float, 0.90).item()),
        "ratio_mean": float(ratio.float().mean().item()),
    }
    return loss, stats


def _hand_2d_bbox_size_loss(
    pred_uv: torch.Tensor,
    target_uv: torch.Tensor,
    point_mask: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    min_points: int,
    min_target_size_px: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pred_uv.shape != target_uv.shape or pred_uv.shape[-1] != 2:
        raise ValueError("pred_uv and target_uv must have the same [..., 2] shape")
    if point_mask.shape != pred_uv.shape[:-1]:
        raise ValueError("point_mask must match pred_uv except for uv dimension")

    pred_uv_for_loss = pred_uv.to(dtype=torch.float32)
    target_uv = target_uv.to(device=pred_uv.device, dtype=pred_uv.dtype)
    target_uv_for_loss = target_uv.to(dtype=torch.float32)
    point_mask = point_mask.to(device=pred_uv.device, dtype=torch.bool)
    valid = point_mask & torch.isfinite(pred_uv_for_loss).all(dim=-1) & torch.isfinite(target_uv_for_loss).all(dim=-1)
    count = valid.sum(dim=-1)
    fill_high = torch.full_like(pred_uv_for_loss, float("inf"))
    fill_low = torch.full_like(pred_uv_for_loss, float("-inf"))
    pred_min = torch.where(valid.unsqueeze(-1), pred_uv_for_loss, fill_high).amin(dim=-2)
    pred_max = torch.where(valid.unsqueeze(-1), pred_uv_for_loss, fill_low).amax(dim=-2)
    target_min = torch.where(valid.unsqueeze(-1), target_uv_for_loss, fill_high).amin(dim=-2)
    target_max = torch.where(valid.unsqueeze(-1), target_uv_for_loss, fill_low).amax(dim=-2)
    pred_size = pred_max - pred_min
    target_size = target_max - target_min
    slot_valid = count >= int(min_points)
    slot_valid = slot_valid & torch.isfinite(pred_size).all(dim=-1)
    slot_valid = slot_valid & torch.isfinite(target_size).all(dim=-1)
    slot_valid = slot_valid & (target_size[..., 0] >= float(min_target_size_px))
    slot_valid = slot_valid & (target_size[..., 1] >= float(min_target_size_px))
    if not torch.any(slot_valid):
        return pred_uv.sum() * 0.0, {"valid_count": 0.0}

    norm = pred_size.new_tensor([max(float(image_width), 1.0), max(float(image_height), 1.0)])
    error_px = (pred_size[slot_valid] - target_size[slot_valid]).abs()
    loss = (error_px / norm).mean()
    ratio = pred_size[slot_valid].detach() / target_size[slot_valid].detach().clamp_min(1.0)
    error_px_float = error_px.detach().float()
    stats = {
        "valid_count": float(slot_valid.to(dtype=torch.float32).sum().detach().item()),
        "width_error_px_mean": float(error_px_float[:, 0].mean().item()),
        "height_error_px_mean": float(error_px_float[:, 1].mean().item()),
        "width_ratio_mean": float(ratio[:, 0].float().mean().item()),
        "height_ratio_mean": float(ratio[:, 1].float().mean().item()),
    }
    return loss, stats


def _hand_reprojection_intrinsics(
    *,
    batch: dict[str, Any],
    pred_intrinsics: torch.Tensor | None,
    gt_intrinsics: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    reprojection_intrinsics = pred_intrinsics if isinstance(pred_intrinsics, torch.Tensor) else None
    required_frame_mask = None
    gt_intrinsics_mask = batch.get("intrinsics_supervision_mask")
    if gt_intrinsics is not None and isinstance(gt_intrinsics_mask, torch.Tensor):
        gt_intrinsics = gt_intrinsics.to(device=device, dtype=dtype)
        gt_intrinsics_mask = gt_intrinsics_mask.to(device=device, dtype=torch.bool)
        gt_intrinsics_mask = gt_intrinsics_mask & torch.isfinite(gt_intrinsics).all(dim=(-2, -1))
        if reprojection_intrinsics is None:
            reprojection_intrinsics = gt_intrinsics
            required_frame_mask = gt_intrinsics_mask
        else:
            gt_intrinsics_mask_view = gt_intrinsics_mask.unsqueeze(-1).unsqueeze(-1)
            reprojection_intrinsics = torch.where(gt_intrinsics_mask_view, gt_intrinsics, reprojection_intrinsics)
    return reprojection_intrinsics, required_frame_mask


def _project_hand_targets_to_uv(
    points_xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    point_mask: torch.Tensor | None = None,
    min_valid_z: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    points_xyz = points_xyz.detach().to(device=intrinsics.device, dtype=intrinsics.dtype)
    intrinsics = intrinsics.detach().to(device=points_xyz.device, dtype=points_xyz.dtype)
    finite_intrinsics = torch.isfinite(intrinsics).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    finite_points = torch.isfinite(points_xyz).all(dim=-1)
    safe_points = torch.where(finite_points.unsqueeze(-1), points_xyz, torch.zeros_like(points_xyz))
    z = safe_points[..., 2]
    safe_z = z.clamp_min(min_valid_z)
    fx = intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
    uv = torch.stack(
        [
            fx * (safe_points[..., 0] / safe_z) + cx,
            fy * (safe_points[..., 1] / safe_z) + cy,
        ],
        dim=-1,
    )
    valid = finite_intrinsics & finite_points & (z > min_valid_z) & torch.isfinite(uv).all(dim=-1)
    if point_mask is not None:
        valid = valid & point_mask.to(device=points_xyz.device, dtype=torch.bool)
    uv = torch.where(valid.unsqueeze(-1), uv, torch.full_like(uv, float("nan")))
    return uv, valid


def _fixed_side_tensor(batch: dict[str, Any], key: str) -> torch.Tensor:
    tensor = batch[key]
    if tensor.shape[2] == 2:
        return tensor
    slots_per_side = int(batch.get("hand_slots_per_side", max(tensor.shape[2] // 2, 1)))
    indices = torch.tensor([0, slots_per_side], dtype=torch.long, device=tensor.device)
    return tensor.index_select(2, indices)


def _presence_targets_and_mask(
    batch: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    targets = batch.get("presence_targets")
    mask = batch.get("presence_supervision_mask")
    if targets is None or mask is None:
        return None, None
    return targets.to(device=device, dtype=dtype), mask.to(device=device)


def _keypoint_heatmap_channel_mask(
    batch: dict[str, Any],
    visible_keypoint_mask: torch.Tensor,
) -> torch.Tensor:
    device = visible_keypoint_mask.device
    side_mask = None
    if "joints_2d_supervision_mask" in batch:
        side_mask = _fixed_side_tensor(batch, "joints_2d_supervision_mask").to(
            device=device,
            dtype=torch.bool,
        )
    presence_targets = batch.get("presence_targets")
    presence_mask = batch.get("presence_supervision_mask")
    if presence_targets is not None and presence_mask is not None:
        presence_targets = _fixed_side_tensor(batch, "presence_targets").to(device=device)
        presence_mask = _fixed_side_tensor(batch, "presence_supervision_mask").to(
            device=device,
            dtype=torch.bool,
        )
        absent_side_mask = presence_mask & (presence_targets <= 0.5)
        side_mask = absent_side_mask if side_mask is None else (side_mask | absent_side_mask)
    if side_mask is None:
        return visible_keypoint_mask
    invalid_mask = batch.get("fixed_side_invalid_mask")
    if invalid_mask is not None:
        side_mask = side_mask & ~_fixed_side_tensor(batch, "fixed_side_invalid_mask").to(
            device=device,
            dtype=torch.bool,
        )
    return side_mask.unsqueeze(-1).expand_as(visible_keypoint_mask)


def _log_invalid_fixed_query_sides(batch: dict[str, Any], invalid_mask: torch.Tensor) -> None:
    if not torch.any(invalid_mask):
        return
    batch_sources = batch.get("batch_sources", [])
    invalid_indices = invalid_mask.nonzero(as_tuple=False).detach().cpu().tolist()
    side_names = ("left", "right")
    for batch_index, frame_index, side_index in invalid_indices[:16]:
        source = batch_sources[batch_index] if batch_index < len(batch_sources) else {}
        frame_ids = source.get("frame_ids", [])
        frame_id = frame_ids[frame_index] if frame_index < len(frame_ids) else ""
        logger.warning(
            "fixed-query hand supervision skipped: dataset={} sequence={} frame={} side={} reason=multi_same_side_hand",
            source.get("dataset_name", ""),
            source.get("sequence_id", ""),
            frame_id,
            side_names[int(side_index)],
        )


def _resize_depth_and_intrinsics(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    depth_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch_size, num_frames, source_height, source_width = depth.shape
    resized_depth = F.interpolate(
        depth.reshape(batch_size * num_frames, 1, source_height, source_width),
        size=(target_height, target_width),
        mode="nearest",
    ).reshape(batch_size, num_frames, target_height, target_width)
    resized_depth_valid_mask = None
    if depth_valid_mask is not None:
        resized_depth_valid_mask = F.interpolate(
            depth_valid_mask.to(dtype=torch.float32).reshape(batch_size * num_frames, 1, source_height, source_width),
            size=(target_height, target_width),
            mode="nearest",
        ).reshape(batch_size, num_frames, target_height, target_width) > 0.5
    scaled_intrinsics = _scale_intrinsics_to_size(
        intrinsics,
        source_height=source_height,
        source_width=source_width,
        target_height=target_height,
        target_width=target_width,
    )
    return resized_depth, scaled_intrinsics, resized_depth_valid_mask


def _scale_intrinsics_to_size(
    intrinsics: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    scaled_intrinsics = intrinsics.clone()
    scale_x = float(target_width) / max(source_width, 1)
    scale_y = float(target_height) / max(source_height, 1)
    scaled_intrinsics[..., 0, 0] *= scale_x
    scaled_intrinsics[..., 0, 2] *= scale_x
    scaled_intrinsics[..., 1, 1] *= scale_y
    scaled_intrinsics[..., 1, 2] *= scale_y
    return scaled_intrinsics


def _vggt_fov_from_intrinsics(
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    fy = intrinsics[..., 1, 1]
    fx = intrinsics[..., 0, 0]
    fov_h = 2.0 * torch.atan((float(image_height) / 2.0) / fy)
    fov_w = 2.0 * torch.atan((float(image_width) / 2.0) / fx)
    return torch.stack([fov_h, fov_w], dim=-1)


def _vggt_quaternion_from_rotation(rotation: torch.Tensor) -> torch.Tensor:
    ensure_vendor_paths()
    from vggt_omega.utils.rotation import mat_to_quat

    return mat_to_quat(rotation)


def _vggt_rotation_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    ensure_vendor_paths()
    from vggt_omega.utils.rotation import quat_to_mat

    return quat_to_mat(_standardize_unit_quaternion(quaternion))


def _standardize_unit_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(quaternion, dim=-1, keepdim=True)
    identity = torch.zeros_like(quaternion)
    identity[..., 3] = 1.0
    quaternion = torch.where(norm > 1e-6, quaternion / norm.clamp_min(1e-6), identity)
    return torch.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)


def _quaternion_sign_invariant_l1(pred_quaternion: torch.Tensor, target_quaternion: torch.Tensor) -> torch.Tensor:
    pred_quaternion = _standardize_unit_quaternion(pred_quaternion)
    target_quaternion = _standardize_unit_quaternion(target_quaternion.to(device=pred_quaternion.device, dtype=pred_quaternion.dtype))
    direct = torch.abs(pred_quaternion - target_quaternion).mean(dim=-1)
    flipped = torch.abs(pred_quaternion + target_quaternion).mean(dim=-1)
    return torch.minimum(direct, flipped).mean()


def _quaternion_dot_geodesic_loss(pred_quaternion: torch.Tensor, target_quaternion: torch.Tensor) -> torch.Tensor:
    pred_quaternion = _standardize_unit_quaternion(pred_quaternion)
    target_quaternion = _standardize_unit_quaternion(target_quaternion.to(device=pred_quaternion.device, dtype=pred_quaternion.dtype))
    dot = torch.abs((pred_quaternion * target_quaternion).sum(dim=-1)).clamp(0.0, 1.0)
    return (1.0 - dot).mean()


def _vggt_pose_matrix_from_encoding(pose_encoding: torch.Tensor) -> torch.Tensor:
    rotation = _vggt_rotation_from_quaternion(pose_encoding[..., 3:7])
    pose = torch.eye(4, dtype=pose_encoding.dtype, device=pose_encoding.device)
    pose = (
        pose.view(*((1,) * (pose_encoding.ndim - 1)), 4, 4)
        .expand(*pose_encoding.shape[:-1], 4, 4)
        .clone()
    )
    pose[..., :3, :3] = rotation
    pose[..., :3, 3] = pose_encoding[..., :3]
    return pose


def _relative_pose_to_first_frame(pose: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=pose.device.type, enabled=False):
        pose_f = pose.float()
        return torch.matmul(pose_f, torch.linalg.pinv(pose_f[:, :1]))


def _relative_pose_supervision_mask(camera_pose_mask: torch.Tensor) -> torch.Tensor:
    relative_mask = camera_pose_mask & camera_pose_mask[:, :1]
    if relative_mask.shape[1] > 0:
        relative_mask = relative_mask.clone()
        relative_mask[:, 0] = False
    return relative_mask


def _crop_single_hand(
    image: torch.Tensor,
    bbox_xyxy: torch.Tensor,
    *,
    side_index: int,
) -> torch.Tensor:
    _, height, width = image.shape
    x_min, y_min, x_max, y_max = bbox_xyxy.tolist()
    x0 = max(int(x_min), 0)
    y0 = max(int(y_min), 0)
    x1 = min(int(x_max) + 1, width)
    y1 = min(int(y_max) + 1, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("无效 bbox crop。")
    crop = image[:, y0:y1, x0:x1]
    crop = F.interpolate(crop.unsqueeze(0), size=(256, 192), mode="bilinear", align_corners=False).squeeze(0)
    if side_index == 0:
        crop = torch.flip(crop, dims=[2])
    return crop


def _bbox_allows_teacher_crop(image: torch.Tensor, bbox_xyxy: torch.Tensor) -> bool:
    if not torch.isfinite(bbox_xyxy).all():
        return False
    _, height, width = image.shape
    x_min, y_min, x_max, y_max = bbox_xyxy.tolist()
    if x_max <= x_min or y_max <= y_min:
        return False
    if x_max < 0 or y_max < 0 or x_min >= width or y_min >= height:
        return False
    return True


def build_wilor_teacher_targets(
    batch: dict[str, torch.Tensor],
    *,
    outputs: dict[str, torch.Tensor | list[Any]],
    teacher: WiLorTeacherWrapper | None,
    device: torch.device,
    dtype: torch.dtype,
    loss_weight: float,
) -> dict[str, torch.Tensor] | None:
    if teacher is None:
        return None
    hand_feature_teacher_features = outputs.get("hand_feature_teacher_features")
    if hand_feature_teacher_features is None:
        return None
    bbox_targets = batch.get("bbox_targets")
    bbox_mask = batch.get("bbox_supervision_mask")
    if bbox_targets is None or bbox_mask is None:
        return None
    presence_targets = batch.get("presence_targets")
    presence_mask = batch.get("presence_supervision_mask")
    if presence_targets is None or presence_mask is None:
        return None

    batch_size, num_frames, slot_count = hand_feature_teacher_features.shape[:3]
    teacher_targets = torch.zeros_like(hand_feature_teacher_features)
    teacher_mask = torch.zeros(batch_size, num_frames, slot_count, dtype=torch.bool, device=device)
    crops = []
    assignments: list[tuple[int, int, int]] = []
    images = batch["images"].to(device=device, dtype=dtype)
    bbox_targets = _fixed_side_tensor(batch, "bbox_targets").to(device=device, dtype=dtype)
    bbox_mask = _fixed_side_tensor(batch, "bbox_supervision_mask").to(device=device)
    presence_targets = presence_targets.to(device=device, dtype=dtype)
    presence_mask = presence_mask.to(device=device)
    for batch_index in range(batch_size):
        for frame_index in range(num_frames):
            for side_index in range(min(2, slot_count)):
                if not bool(presence_mask[batch_index, frame_index, side_index].item()):
                    continue
                if float(presence_targets[batch_index, frame_index, side_index].item()) <= 0.5:
                    continue
                if not bool(bbox_mask[batch_index, frame_index, side_index].item()):
                    continue
                bbox = bbox_targets[batch_index, frame_index, side_index]
                if not _bbox_allows_teacher_crop(images[batch_index, frame_index], bbox):
                    continue
                crop = _crop_single_hand(
                    images[batch_index, frame_index],
                    bbox,
                    side_index=side_index,
                )
                crops.append(crop)
                assignments.append((batch_index, frame_index, side_index))
    if not crops:
        return None
    with torch.no_grad():
        teacher_features = teacher(torch.stack(crops, dim=0))
    for teacher_feature, (batch_index, frame_index, slot_index) in zip(teacher_features, assignments, strict=True):
        teacher_targets[batch_index, frame_index, slot_index] = teacher_feature.to(device=device, dtype=dtype)
        teacher_mask[batch_index, frame_index, slot_index] = True
    return {
        "teacher_features": teacher_targets,
        "teacher_mask": teacher_mask,
        "weight": torch.tensor(loss_weight, device=device, dtype=dtype),
    }


_ZERO_LOSS_ANCHOR_OUTPUT_KEYS = (
    "heatmap_logits",
    "presence_logits",
    "depth",
    "intrinsics",
    "camera_pose",
    "camera_pose_encoding",
    "prompt_vertex_xyz",
    "prompt_vertex_offset",
    "prompt_vertex_visibility_logits",
    "prompt_joint_xyz",
    "prompt_joint_visibility_logits",
    "log_metric_value",
    "metric_value",
    "prompt_teacher_features",
    "hand_feature_teacher_features",
)


def _zero_loss_anchor_from_outputs(outputs: dict[str, torch.Tensor | list[Any]]) -> torch.Tensor | None:
    anchor: torch.Tensor | None = None
    for key in _ZERO_LOSS_ANCHOR_OUTPUT_KEYS:
        value = outputs.get(key)
        if not isinstance(value, torch.Tensor) or not value.requires_grad or value.numel() == 0:
            continue
        scalar = value.reshape(-1)[0]
        finite_scalar = torch.where(torch.isfinite(scalar), scalar, torch.zeros_like(scalar))
        term = finite_scalar * 0.0
        anchor = term if anchor is None else anchor + term
    return anchor


def _output_flag_enabled(outputs: dict[str, Any], key: str, default: bool) -> bool:
    value = outputs.get(key)
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return bool((value.detach().reshape(-1)[0] > 0.5).item())
    return bool(value)


def _log_depth_conf_stats(
    metrics: dict[str, float],
    prefix: str,
    depth_conf: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    valid = mask.to(device=depth_conf.device, dtype=torch.bool)
    valid &= torch.isfinite(depth_conf)
    if not torch.any(valid):
        return
    values = depth_conf[valid].detach().float()
    metrics[f"{prefix}_depth_conf_mean"] = float(values.mean().item())
    metrics[f"{prefix}_depth_conf_p50"] = float(torch.quantile(values, 0.50).item())
    metrics[f"{prefix}_depth_conf_p90"] = float(torch.quantile(values, 0.90).item())
    metrics[f"{prefix}_depth_conf_p99"] = float(torch.quantile(values, 0.99).item())


def _robust_metric_mask(error: torch.Tensor, valid: torch.Tensor, robust_quantile: float) -> torch.Tensor:
    valid = valid.to(device=error.device, dtype=torch.bool) & torch.isfinite(error)
    if not torch.any(valid) or robust_quantile <= 0.0 or robust_quantile >= 1.0:
        return valid
    threshold = torch.quantile(error[valid].detach(), float(robust_quantile))
    return valid & (error <= threshold)


def _log_depth_error_metrics(
    metrics: dict[str, float],
    prefix: str,
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    depth_conf: torch.Tensor,
    mask: torch.Tensor,
    *,
    suffix: str,
    alpha: float,
    robust_quantile: float,
    log_reward: bool,
    eps: float = 1e-6,
) -> None:
    with torch.no_grad():
        pred_depth = pred_depth.detach()
        target_depth = target_depth.detach().to(device=pred_depth.device, dtype=pred_depth.dtype)
        depth_conf = depth_conf.detach().to(device=pred_depth.device, dtype=pred_depth.dtype)
        valid = mask.to(device=pred_depth.device, dtype=torch.bool)
        valid &= torch.isfinite(pred_depth) & torch.isfinite(target_depth) & torch.isfinite(depth_conf)
        error = torch.abs(pred_depth - target_depth)
        valid = _robust_metric_mask(error, valid, robust_quantile)
        if not torch.any(valid):
            return
        metrics[f"{prefix}_depth_abs_error_{suffix}"] = float(error[valid].float().mean().item())
        if log_reward:
            conf = depth_conf.clamp_min(1.0 + eps)
            reward = -float(alpha) * torch.log(conf[valid])
            metrics[f"{prefix}_depth_conf_reward"] = float(reward.float().mean().item())


def _log_point_error_metrics(
    metrics: dict[str, float],
    prefix: str,
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    pred_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    pred_camera_pose: torch.Tensor,
    target_camera_pose: torch.Tensor,
    mask: torch.Tensor,
    *,
    suffix: str,
    robust_quantile: float,
    pred_scene_scale: torch.Tensor | None = None,
    target_scene_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> None:
    with torch.no_grad():
        pred_depth = pred_depth.detach()
        target_depth = target_depth.detach().to(device=pred_depth.device, dtype=pred_depth.dtype)
        pred_intrinsics = pred_intrinsics.detach().to(device=pred_depth.device, dtype=pred_depth.dtype)
        target_intrinsics = target_intrinsics.detach().to(device=pred_depth.device, dtype=pred_depth.dtype)
        pred_camera_pose = pred_camera_pose.detach().to(device=pred_depth.device, dtype=pred_depth.dtype)
        target_camera_pose = target_camera_pose.detach().to(device=pred_depth.device, dtype=pred_depth.dtype)
        valid = mask.to(device=pred_depth.device, dtype=torch.bool)
        valid &= torch.isfinite(pred_depth) & torch.isfinite(target_depth)
        valid &= torch.isfinite(pred_intrinsics).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        valid &= torch.isfinite(target_intrinsics).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        valid &= torch.isfinite(pred_camera_pose).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        valid &= torch.isfinite(target_camera_pose).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)

        pred_points = unproject_depth_to_camera_points(pred_depth, pred_intrinsics)
        target_points = unproject_depth_to_camera_points(target_depth, target_intrinsics)
        pred_points = transform_camera_points_to_first_frame(pred_points, pred_camera_pose)
        target_points = transform_camera_points_to_first_frame(target_points, target_camera_pose)
        if pred_scene_scale is not None and target_scene_scale is not None:
            pred_scene_scale = pred_scene_scale.detach().to(device=pred_depth.device, dtype=pred_depth.dtype)
            target_scene_scale = target_scene_scale.detach().to(device=pred_depth.device, dtype=pred_depth.dtype)
            valid &= torch.isfinite(pred_scene_scale).view(-1, 1, 1, 1)
            valid &= torch.isfinite(target_scene_scale).view(-1, 1, 1, 1)
            pred_points = pred_points / pred_scene_scale.clamp_min(eps).view(-1, 1, 1, 1, 1)
            target_points = target_points / target_scene_scale.clamp_min(eps).view(-1, 1, 1, 1, 1)

        error = torch.linalg.norm(pred_points - target_points, dim=-1)
        valid = _robust_metric_mask(error, valid, robust_quantile)
        if torch.any(valid):
            metrics[f"{prefix}_point_error_{suffix}"] = float(error[valid].float().mean().item())


def _masked_translation_loss(
    normalized_pred: torch.Tensor,
    normalized_target: torch.Tensor,
    translation_mask: torch.Tensor,
    *,
    max_sample_loss: float,
) -> tuple[torch.Tensor, int]:
    diff = normalized_pred - normalized_target
    sample_mask = translation_mask.to(dtype=torch.bool)
    safe_diff = torch.where(sample_mask.unsqueeze(-1), diff, torch.zeros_like(diff))
    diff_sq = safe_diff.pow(2).sum(dim=-1)
    per_sample_count = sample_mask.sum(dim=1)
    per_sample_valid = per_sample_count > 0
    per_sample_loss = diff_sq.sum(dim=1) / per_sample_count.clamp_min(1).to(dtype=diff_sq.dtype)

    keep = per_sample_valid
    if max_sample_loss > 0.0:
        keep = keep & (per_sample_loss.detach() <= float(max_sample_loss))
    dropped = int((per_sample_valid & ~keep).sum().item())
    if not torch.any(keep):
        return normalized_pred.sum() * 0.0, dropped
    return per_sample_loss[keep].mean(), dropped


def _hand_abs_geometry_supervision(
    *,
    metrics: dict[str, float],
    runtime_config: MarkerRuntimeConfig,
    stream_name: str,
    block_hand_gt: bool,
    use_hand_conf: bool,
    global_step: int | None,
    zero_loss: torch.Tensor,
    metric_value,
    dense_joint_xyz,
    dense_vertex_xyz,
    dense_joint_log_conf,
    dense_vertex_log_conf,
    joint_targets,
    joint_xyz_mask,
    vertex_xyz_targets,
    vertex_xyz_mask,
    resized_depth,
    resized_intrinsics,
    resized_depth_valid_mask,
    camera_pose_target_for_scale,
    valid_depth_frames,
    valid_intrinsics_frames,
    valid_camera_pose_frames,
    pred_depth,
    pred_intrinsics,
    pred_camera_pose,
) -> torch.Tensor:
    total = zero_loss
    if block_hand_gt:
        return total
    pred_ready = (
        pred_depth is not None
        and pred_intrinsics is not None
        and isinstance(pred_camera_pose, torch.Tensor)
        and pred_camera_pose.ndim >= 4
        and pred_camera_pose.shape[-2:] == (4, 4)
    )
    if not pred_ready:
        return total
    pred_scene_scale = scene_average_distance_from_depth(
        pred_depth.detach(),
        pred_intrinsics.detach(),
        pred_camera_pose.detach(),
    )
    pred_scene_mask = _scene_scale_sequence_mask(pred_depth, pred_intrinsics, pred_camera_pose)
    pred_scene_valid = _valid_scene_scale_mask(
        pred_scene_scale,
        min_value=runtime_config.scene_scale_min_value,
    )
    pred_scene_mask &= pred_scene_valid
    _record_scene_scale_guard_metrics(
        metrics,
        f"{stream_name}_hand_pred_scene_scale",
        pred_scene_scale,
        pred_scene_valid,
    )

    gt_scene_scale = None
    scene_seq_mask = None
    scene_ready = (
        resized_depth is not None
        and resized_intrinsics is not None
        and valid_depth_frames is not None
        and valid_intrinsics_frames is not None
        and valid_camera_pose_frames is not None
        and camera_pose_target_for_scale is not None
    )
    if scene_ready:
        scale_frames = valid_depth_frames & valid_intrinsics_frames & valid_camera_pose_frames
        if torch.any(scale_frames):
            valid_depth_for_scale = (
                torch.ones_like(resized_depth, dtype=torch.bool)
                if resized_depth_valid_mask is None
                else resized_depth_valid_mask
            )
            gt_depth_scale_mask = valid_depth_for_scale & scale_frames.unsqueeze(-1).unsqueeze(-1)
            gt_scene_scale = scene_average_distance_from_depth(
                resized_depth.detach(),
                resized_intrinsics.detach(),
                camera_pose_target_for_scale.detach(),
                gt_depth_scale_mask,
            )
            scene_seq_mask = (
                _scene_scale_sequence_mask(
                    resized_depth,
                    resized_intrinsics,
                    camera_pose_target_for_scale,
                    gt_depth_scale_mask,
                )
                & pred_scene_mask
                & _valid_scene_scale_mask(
                    gt_scene_scale,
                    min_value=runtime_config.scene_scale_min_value,
                )
            )
            _record_scene_scale_guard_metrics(
                metrics,
                f"{stream_name}_hand_gt_scene_scale",
                gt_scene_scale,
                _valid_scene_scale_mask(
                    gt_scene_scale,
                    min_value=runtime_config.scene_scale_min_value,
                ),
            )

    targets = (
        ("joint", dense_joint_xyz, joint_targets, joint_xyz_mask, dense_joint_log_conf),
        ("vertex", dense_vertex_xyz, vertex_xyz_targets, vertex_xyz_mask, dense_vertex_log_conf),
    )
    for name, pred_xyz, target_xyz, point_mask, log_conf in targets:
        if not (isinstance(pred_xyz, torch.Tensor) and isinstance(target_xyz, torch.Tensor)):
            continue
        if point_mask is None or not isinstance(log_conf, torch.Tensor):
            continue
        finite_target = torch.isfinite(target_xyz).all(dim=-1)
        scene_weight = getattr(runtime_config, f"hand_{name}_scene_loss_weight")
        if scene_weight > 0.0 and gt_scene_scale is not None and scene_seq_mask is not None:
            scene_mask = point_mask & finite_target & scene_seq_mask.view(-1, 1, 1, 1)
            if torch.any(scene_mask):
                loss, stats, dropped = _hand_scene_abs_conf_loss(
                    pred_xyz,
                    target_xyz,
                    scene_mask,
                    log_conf,
                    pred_scene_scale,
                    gt_scene_scale,
                    alpha=runtime_config.hand_conf_local_alpha,
                    log_min=runtime_config.hand_conf_log_min,
                    log_max=runtime_config.hand_conf_log_max,
                    use_confidence=use_hand_conf,
                    max_sample_loss=runtime_config.scale_normalized_max_sample_loss,
                )
                total = total + scene_weight * loss
                _record_loss_metrics(
                    metrics,
                    f"{stream_name}_hand_{name}_scene_loss",
                    loss,
                    scene_weight,
                    scene_mask,
                )
                if dropped > 0:
                    metrics[f"{stream_name}_hand_{name}_scene_dropped_samples"] = float(dropped)
                if stats:
                    metrics[f"{stream_name}_hand_{name}_scene_loss_raw"] = stats["raw"]
        metric_weight = _effective_after_warmup_weight(
            getattr(runtime_config, f"hand_{name}_metric_loss_weight"),
            runtime_config.hand_root_metric_warmup_steps,
            global_step,
        )
        if metric_weight > 0.0 and metric_value is not None:
            metric_mask = point_mask & finite_target & pred_scene_mask.view(-1, 1, 1, 1)
            if torch.any(metric_mask):
                loss, stats, dropped = _hand_metric_abs_conf_loss(
                    pred_xyz,
                    target_xyz,
                    metric_mask,
                    log_conf,
                    pred_scene_scale,
                    metric_value,
                    alpha=runtime_config.hand_conf_local_alpha,
                    log_min=runtime_config.hand_conf_log_min,
                    log_max=runtime_config.hand_conf_log_max,
                    use_confidence=use_hand_conf,
                    max_sample_loss=runtime_config.scale_normalized_max_sample_loss,
                )
                total = total + metric_weight * loss
                _record_loss_metrics(
                    metrics,
                    f"{stream_name}_hand_{name}_metric_loss",
                    loss,
                    metric_weight,
                    metric_mask,
                )
                if dropped > 0:
                    metrics[f"{stream_name}_hand_{name}_metric_dropped_samples"] = float(dropped)
                if stats:
                    metrics[f"{stream_name}_hand_{name}_metric_loss_raw"] = stats["raw"]
    return total



def _effective_flow_loss_weight(
    base_weight: float,
    peak_weight: float,
    warmup_steps: int,
    peak_steps: int,
    decay_steps: int,
    global_step: int | None,
) -> float:
    """Return the scheduled marker-stream flow loss multiplier.

    peak_steps and decay_steps are absolute global-step milestones:
    warmup ramps to warmup_steps, peak holds until peak_steps, and decay
    reaches the base weight at decay_steps. A zero base weight is a strict
    opt-out that never ramps back up.
    """
    base = float(base_weight)
    peak = float(peak_weight)
    if base == 0.0:
        return 0.0
    if global_step is None:
        return base
    step = max(int(global_step), 0)
    warmup_end = max(int(warmup_steps), 0)
    peak_end = max(int(peak_steps), warmup_end)
    decay_end = max(int(decay_steps), peak_end)
    if warmup_end > 0 and step < warmup_end:
        return base + (peak - base) * (float(step) / float(warmup_end))
    if step < peak_end:
        return peak
    if decay_end > peak_end and step < decay_end:
        progress = float(step - peak_end) / float(decay_end - peak_end)
        return peak + (base - peak) * progress
    return base


def _flow_boolean_mask(mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """Convert a mask to bool without treating non-finite numeric values as valid."""
    mask = mask.to(device=device)
    if mask.dtype == torch.bool:
        return mask
    return torch.isfinite(mask) & (mask > 0)


def _flow_smoothness_loss(flow_pred: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute a first-order, valid-neighbour flow smoothness penalty."""
    if flow_pred.ndim != 5 or flow_pred.shape[-3] != 2:
        raise ValueError("flow_pred must be (B, P, 2, H, W)")
    flow = flow_pred.to(dtype=torch.float32)
    valid = _flow_boolean_mask(valid, device=flow.device)
    valid = valid & torch.isfinite(flow).all(dim=2)
    safe_flow = torch.where(valid.unsqueeze(2), flow, torch.zeros_like(flow))
    terms: list[torch.Tensor] = []
    if flow.shape[-1] > 1:
        horizontal = (safe_flow[..., :, 1:] - safe_flow[..., :, :-1]).abs().sum(dim=2)
        horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
        if torch.any(horizontal_valid):
            terms.append(horizontal[horizontal_valid])
    if flow.shape[-2] > 1:
        vertical = (safe_flow[..., 1:, :] - safe_flow[..., :-1, :]).abs().sum(dim=2)
        vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
        if torch.any(vertical_valid):
            terms.append(vertical[vertical_valid])
    if not terms:
        return safe_flow.sum() * 0.0
    return torch.cat(terms).mean()


def _flow_downsample_valid_vectors(
    flow: torch.Tensor,
    valid: torch.Tensor,
    size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask-aware area pooling for vectors expressed in source pixel units."""
    flow = flow.to(dtype=torch.float32)
    batch, pairs, channels, height, width = flow.shape
    target_height, target_width = size
    valid = _flow_boolean_mask(valid, device=flow.device)
    valid = valid & torch.isfinite(flow).all(dim=2)
    valid_float = valid.to(dtype=torch.float32).reshape(
        batch * pairs,
        1,
        height,
        width,
    )
    valid_fraction = F.interpolate(valid_float, size=size, mode="area")
    safe_flow = torch.where(valid.unsqueeze(2), flow, torch.zeros_like(flow))
    pooled = F.interpolate(
        safe_flow.reshape(batch * pairs, channels, height, width),
        size=size,
        mode="area",
    ) / valid_fraction.clamp_min(1e-8)
    pooled = pooled.reshape(batch, pairs, channels, target_height, target_width)
    pooled = pooled.clone()
    pooled[:, :, 0] *= float(target_width) / max(float(width), 1.0)
    pooled[:, :, 1] *= float(target_height) / max(float(height), 1.0)
    return pooled, valid_fraction.reshape(batch, pairs, target_height, target_width)


def _flow_downsample_valid_map(
    values: torch.Tensor,
    valid: torch.Tensor,
    valid_fraction: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    batch, pairs, height, width = values.shape
    target_height, target_width = size
    valid = _flow_boolean_mask(valid, device=values.device)
    valid = valid & torch.isfinite(values)
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    pooled = F.interpolate(
        safe_values.reshape(batch * pairs, 1, height, width),
        size=size,
        mode="area",
    )
    safe_fraction = torch.nan_to_num(
        valid_fraction.to(device=values.device, dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return (pooled / safe_fraction.reshape(batch * pairs, 1, target_height, target_width).clamp_min(1e-8)).reshape(
        batch,
        pairs,
        target_height,
        target_width,
    )


def _flow_hand_weight_maps(
    batch: dict[str, torch.Tensor],
    flow_pred: torch.Tensor,
    runtime_config: MarkerRuntimeConfig,
    *,
    supervised_pairs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve precomputed maps and fill only supervised missing pairs from annotations."""
    batch_size, pair_count, _, height, width = flow_pred.shape
    device = flow_pred.device
    expected_map_shape = (batch_size, pair_count, height, width)
    expected_pair_shape = (batch_size, pair_count)
    if supervised_pairs is None:
        annotation_fallback_pairs = torch.ones(*expected_pair_shape, dtype=torch.bool, device=device)
    elif tuple(supervised_pairs.shape) != expected_pair_shape:
        raise ValueError("supervised_pairs 形状与 flow_pred 的 batch/pair 维度不一致")
    else:
        annotation_fallback_pairs = _flow_boolean_mask(supervised_pairs, device=device)

    hand_mask = torch.zeros(*expected_map_shape, dtype=torch.bool, device=device)
    hand_fallback_pairs = annotation_fallback_pairs
    hand_value = batch.get("flow_hand_region_mask")
    if isinstance(hand_value, torch.Tensor) and tuple(hand_value.shape) == expected_map_shape:
        hand_mask = hand_value.to(device=device, dtype=torch.bool).clone()
        hand_fallback_pairs = annotation_fallback_pairs & ~hand_mask.reshape(
            batch_size,
            pair_count,
            -1,
        ).any(dim=-1)

    bbox_targets = batch.get("bbox_targets")
    bbox_mask = batch.get("bbox_supervision_mask")
    if (
        isinstance(bbox_targets, torch.Tensor)
        and bbox_targets.ndim == 4
        and bool(hand_fallback_pairs.any().item())
    ):
        bbox_targets = bbox_targets.to(device=device, dtype=torch.float32)
        if isinstance(bbox_mask, torch.Tensor) and tuple(bbox_mask.shape) == tuple(bbox_targets.shape[:-1]):
            bbox_mask = bbox_mask.to(device=device, dtype=torch.bool)
        else:
            bbox_mask = torch.ones(bbox_targets.shape[:-1], dtype=torch.bool, device=device)
        batch_count = min(batch_size, int(bbox_targets.shape[0]))
        slot_count = int(bbox_targets.shape[2])
        for batch_index in range(batch_count):
            for frame_index in range(min(pair_count, int(bbox_targets.shape[1]) - 1)):
                if not bool(hand_fallback_pairs[batch_index, frame_index].item()):
                    continue
                for slot_index in range(slot_count):
                    if not bool(bbox_mask[batch_index, frame_index, slot_index].item()):
                        continue
                    bbox = bbox_targets[batch_index, frame_index, slot_index]
                    if bbox.numel() != 4 or not torch.isfinite(bbox).all():
                        continue
                    x0 = max(0, min(width, int(torch.floor(bbox[0]).item())))
                    y0 = max(0, min(height, int(torch.floor(bbox[1]).item())))
                    x1 = max(0, min(width, int(torch.ceil(bbox[2]).item())))
                    y1 = max(0, min(height, int(torch.ceil(bbox[3]).item())))
                    if x1 > x0 and y1 > y0:
                        hand_mask[batch_index, frame_index, y0:y1, x0:x1] = True

    fingertip_map = torch.zeros(*expected_map_shape, dtype=torch.float32, device=device)
    fingertip_fallback_pairs = annotation_fallback_pairs
    fingertip_value = batch.get("flow_fingertip_weight")
    if isinstance(fingertip_value, torch.Tensor) and tuple(fingertip_value.shape) == expected_map_shape:
        fingertip_map = torch.nan_to_num(
            fingertip_value.to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        fingertip_fallback_pairs = annotation_fallback_pairs & ~fingertip_map.gt(0.0).reshape(
            batch_size,
            pair_count,
            -1,
        ).any(dim=-1)

    joints = batch.get("joints_2d_targets")
    if (
        isinstance(joints, torch.Tensor)
        and joints.ndim == 5
        and bool(fingertip_fallback_pairs.any().item())
    ):
        joints = joints.to(device=device, dtype=torch.float32)
        finite = torch.isfinite(joints).all(dim=-1)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        sigma = max(float(runtime_config.flow_fingertip_sigma_px), 1e-3)
        batch_count = min(batch_size, int(joints.shape[0]))
        for batch_index in range(batch_count):
            for frame_index in range(min(pair_count, int(joints.shape[1]) - 1)):
                if not bool(fingertip_fallback_pairs[batch_index, frame_index].item()):
                    continue
                point_values = joints[batch_index, frame_index]
                point_valid = finite[batch_index, frame_index]
                if point_values.ndim != 3 or point_values.shape[-2] < 21:
                    continue
                for slot_index in range(int(point_values.shape[0])):
                    for keypoint_index in (4, 8, 12, 16, 20):
                        if not bool(point_valid[slot_index, keypoint_index].item()):
                            continue
                        x_coord, y_coord = point_values[slot_index, keypoint_index]
                        if not (
                            0.0 <= float(x_coord.item()) < float(width)
                            and 0.0 <= float(y_coord.item()) < float(height)
                        ):
                            continue
                        distance_sq = (xx - x_coord).square() + (yy - y_coord).square()
                        fingertip_map[batch_index, frame_index] = torch.maximum(
                            fingertip_map[batch_index, frame_index],
                            torch.exp(-distance_sq / (2.0 * sigma * sigma)),
                        )
    return hand_mask, fingertip_map


def _flow_weighted_epe(
    flow_pred: torch.Tensor,
    flow_target: torch.Tensor,
    valid: torch.Tensor,
    pixel_weight: torch.Tensor,
    *,
    motion_weight: float,
    motion_threshold_px: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = flow_pred.to(dtype=torch.float32)
    target = flow_target.to(device=pred.device, dtype=torch.float32)
    valid = _flow_boolean_mask(valid, device=pred.device)
    pixel_weight = pixel_weight.to(device=pred.device, dtype=torch.float32)
    finite_vectors = torch.isfinite(pred).all(dim=2) & torch.isfinite(target).all(dim=2)
    effective_valid = valid & finite_vectors & torch.isfinite(pixel_weight)
    safe_pred = torch.where(effective_valid.unsqueeze(2), pred, torch.zeros_like(pred))
    safe_target = torch.where(effective_valid.unsqueeze(2), target, torch.zeros_like(target))
    endpoint_error = torch.sqrt((safe_pred - safe_target).square().sum(dim=2) + 1e-12)
    target_magnitude = torch.sqrt(safe_target.square().sum(dim=2) + 1e-12)
    motion_scale = 1.0 + float(motion_weight) * (
        target_magnitude / max(float(motion_threshold_px), 1e-6)
    ).clamp(max=1.0)
    safe_weight = torch.where(effective_valid, pixel_weight, torch.zeros_like(pixel_weight))
    combined_weight = safe_weight * motion_scale
    selected_weight = combined_weight[effective_valid]
    if selected_weight.numel() == 0:
        return safe_pred.sum() * 0.0, endpoint_error, target_magnitude
    return (
        (selected_weight * endpoint_error[effective_valid]).sum() / selected_weight.sum().clamp_min(1e-8),
        endpoint_error,
        target_magnitude,
    )


def compute_marker_model_loss(
    outputs: dict[str, torch.Tensor | list[Any]],
    batch: dict[str, torch.Tensor],
    *,
    stream_name: str,
    runtime_config: MarkerRuntimeConfig | None = None,
    teacher_targets: dict[str, torch.Tensor] | None = None,
    global_step: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if runtime_config is None:
        runtime_config = MarkerRuntimeConfig()
    dense_vertex_xyz = outputs["dense_vertex_xyz"]
    dense_vertex_visibility_logits = outputs["dense_vertex_visibility_logits"]
    device = dense_vertex_xyz.device
    dtype = dense_vertex_xyz.dtype
    metrics: dict[str, float] = {}
    total_loss = _zero_loss_anchor_from_outputs(outputs)
    if total_loss is None:
        total_loss = dense_vertex_xyz.new_zeros(())
    flow_pred = outputs.get("flow_pred")
    if stream_name == "marker" and isinstance(flow_pred, torch.Tensor):
        flow_target = batch.get("flow_pseudo_target")
        flow_valid = batch.get("flow_pseudo_valid")
        flow_pair_mask = batch.get("flow_pair_mask")
        flow_gate = outputs.get("flow_gate")
        if not (
            isinstance(flow_target, torch.Tensor)
            and isinstance(flow_valid, torch.Tensor)
            and isinstance(flow_pair_mask, torch.Tensor)
        ):
            raise ValueError("启用 flow guidance 时 batch 必须包含 flow_pseudo_target/valid/pair_mask")
        if flow_pred.ndim != 5 or flow_pred.shape[2] != 2:
            raise ValueError("flow_pred 必须是 (B, P, 2, H, W)")
        if flow_target.shape != flow_pred.shape:
            raise ValueError(
                f"flow_pseudo_target 形状应为 {tuple(flow_pred.shape)}，当前为 {tuple(flow_target.shape)}"
            )
        if flow_valid.shape != flow_pred.shape[:2] + flow_pred.shape[-2:]:
            raise ValueError("flow_pseudo_valid 形状与 flow_pred 不一致")
        if flow_pair_mask.shape != flow_pred.shape[:2]:
            raise ValueError("flow_pair_mask 形状与 flow_pred 不一致")
        flow_pred_float = flow_pred.to(dtype=torch.float32)
        flow_target_float = flow_target.to(device=flow_pred.device, dtype=torch.float32)
        valid = _flow_boolean_mask(flow_valid, device=flow_pred.device)
        pair_mask = _flow_boolean_mask(flow_pair_mask, device=flow_pred.device)
        valid = valid & pair_mask.unsqueeze(-1).unsqueeze(-1)
        finite_flow_vectors = torch.isfinite(flow_pred_float).all(dim=2) & torch.isfinite(flow_target_float).all(dim=2)
        valid = valid & finite_flow_vectors
        flow_pred_safe = torch.where(
            finite_flow_vectors.unsqueeze(2),
            flow_pred_float,
            torch.zeros_like(flow_pred_float),
        )
        flow_target_safe = torch.where(
            finite_flow_vectors.unsqueeze(2),
            flow_target_float,
            torch.zeros_like(flow_target_float),
        )
        effective_weight = _effective_flow_loss_weight(
            runtime_config.flow_loss_weight,
            runtime_config.flow_loss_peak_weight,
            runtime_config.flow_loss_warmup_steps,
            runtime_config.flow_loss_peak_steps,
            runtime_config.flow_loss_decay_steps,
            global_step,
        )
        endpoint_error = torch.sqrt(
            (flow_pred_safe - flow_target_safe).square().sum(dim=2) + 1e-12
        )
        target_magnitude = torch.sqrt(flow_target_safe.square().sum(dim=2) + 1e-12)
        prediction_magnitude = torch.sqrt(flow_pred_safe.square().sum(dim=2) + 1e-12)
        valid_pixel_count = valid.sum()
        valid_pair_count = (valid.any(dim=(-2, -1)) & pair_mask).sum()
        pair_count = pair_mask.numel()
        metrics["flow_valid_pixel_count"] = float(valid_pixel_count.item())
        metrics["flow_valid_pair_count"] = float(valid_pair_count.item())
        metrics["flow_supervised_pair_fraction"] = (
            0.0 if pair_count == 0 else float(valid_pair_count.item()) / float(pair_count)
        )
        metrics["flow_gate"] = (
            float(flow_gate.detach().to(dtype=torch.float32).item())
            if isinstance(flow_gate, torch.Tensor)
            else 0.0
        )
        metrics["flow_loss_effective_weight"] = float(effective_weight)
        supervised_pairs = valid.any(dim=(-2, -1))
        if torch.any(valid):
            hand_mask, fingertip_map = _flow_hand_weight_maps(
                batch,
                flow_pred,
                runtime_config,
                supervised_pairs=supervised_pairs,
            )
            pixel_weight = (
                float(runtime_config.flow_background_weight)
                + (
                    float(runtime_config.flow_hand_region_weight)
                    - float(runtime_config.flow_background_weight)
                )
                * hand_mask.to(dtype=torch.float32)
            )
            pixel_weight = torch.nan_to_num(
                pixel_weight
                + fingertip_map.clamp(0.0, 1.0)
                * (float(runtime_config.flow_fingertip_weight) - pixel_weight),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            metrics["flow_hand_pixel_fraction"] = float(
                hand_mask[valid].to(dtype=torch.float32).mean().detach().item()
            )
            metrics["flow_fingertip_weight_mean"] = float(
                fingertip_map[valid].mean().detach().item()
            )
            motion_threshold = float(runtime_config.flow_motion_threshold_px)
            flow_loss, endpoint_error, target_magnitude = _flow_weighted_epe(
                flow_pred_safe,
                flow_target_safe,
                valid,
                pixel_weight,
                motion_weight=float(runtime_config.flow_motion_weight),
                motion_threshold_px=motion_threshold,
            )
            motion_pixels = valid & (target_magnitude >= motion_threshold)
            metrics["flow_loss"] = float(flow_loss.detach().item())
            metrics["flow_weighted_epe"] = metrics["flow_loss"]
            metrics["flow_epe"] = float(endpoint_error[valid].mean().detach().item())
            metrics["flow_zero_epe"] = float(target_magnitude[valid].mean().detach().item())
            metrics["flow_prediction_magnitude"] = float(prediction_magnitude[valid].mean().detach().item())
            metrics["flow_target_magnitude"] = metrics["flow_zero_epe"]
            metrics["flow_motion_pixel_fraction"] = float(motion_pixels.sum().item()) / float(valid_pixel_count.item())
            metrics["flow_motion_epe"] = (
                float(endpoint_error[motion_pixels].mean().detach().item()) if torch.any(motion_pixels) else 0.0
            )

            half_height = max(int(flow_pred.shape[-2]) // 2, 1)
            half_width = max(int(flow_pred.shape[-1]) // 2, 1)
            half_pred, half_valid_fraction = _flow_downsample_valid_vectors(
                flow_pred_safe,
                valid,
                (half_height, half_width),
            )
            half_target, _ = _flow_downsample_valid_vectors(
                flow_target_safe,
                valid,
                (half_height, half_width),
            )
            half_valid = half_valid_fraction > 0.0
            half_pixel_weight = _flow_downsample_valid_map(
                pixel_weight,
                valid,
                half_valid_fraction,
                (half_height, half_width),
            ) * half_valid_fraction
            half_motion_threshold = motion_threshold * (
                (
                    (float(half_height) / max(float(flow_pred.shape[-2]), 1.0)) ** 2
                    + (float(half_width) / max(float(flow_pred.shape[-1]), 1.0)) ** 2
                )
                / 2.0
            ) ** 0.5
            half_loss, _, _ = _flow_weighted_epe(
                half_pred,
                half_target,
                half_valid,
                half_pixel_weight,
                motion_weight=float(runtime_config.flow_motion_weight),
                motion_threshold_px=max(half_motion_threshold, 1e-6),
            )
            smoothness_loss = _flow_smoothness_loss(flow_pred_safe, valid)
            half_raw_epe, _, _ = _flow_weighted_epe(
                half_pred,
                half_target,
                half_valid,
                half_valid_fraction,
                motion_weight=0.0,
                motion_threshold_px=1.0,
            )
            metrics["flow_weighted_epe_half"] = float(half_loss.detach().item())
            metrics["flow_epe_half"] = float(half_raw_epe.detach().item())
            metrics["flow_smoothness_loss"] = float(smoothness_loss.detach().item())
            if effective_weight > 0.0:
                total_loss = total_loss + effective_weight * (
                    flow_loss
                    + float(runtime_config.flow_multiscale_half_weight) * half_loss
                    + float(runtime_config.flow_smoothness_weight) * smoothness_loss
                )
        else:
            metrics["flow_hand_pixel_fraction"] = 0.0
            metrics["flow_fingertip_weight_mean"] = 0.0
            metrics["flow_loss"] = 0.0
            metrics["flow_weighted_epe"] = 0.0
            metrics["flow_epe"] = 0.0
            metrics["flow_weighted_epe_half"] = 0.0
            metrics["flow_epe_half"] = 0.0
            metrics["flow_smoothness_loss"] = 0.0
    is_marker_stream = stream_name == "marker"
    block_hand_gt = (not is_marker_stream) and (str(batch.get("stage", "")) == "posttrain")
    use_hand_conf = (global_step is None) or (
        int(global_step) >= int(runtime_config.hand_confidence_warmup_steps)
    )
    use_contact_conf = (global_step is None) or (
        int(global_step) >= int(runtime_config.contact_confidence_warmup_steps)
    )
    presence_targets, presence_mask = _presence_targets_and_mask(batch, device=device, dtype=dtype)
    if presence_mask is not None:
        invalid_mask = batch.get("fixed_side_invalid_mask")
        if invalid_mask is not None:
            _log_invalid_fixed_query_sides(batch, invalid_mask.to(device=device) & presence_mask)
    three_r_loss_mask = batch.get("three_r_supervision_mask")
    if three_r_loss_mask is not None:
        three_r_loss_mask = three_r_loss_mask.to(device=device)

    presence_logits = outputs.get("presence_logits")
    presence_loss_enabled = _output_flag_enabled(outputs, "presence_loss_enabled", True)
    positive_presence_mask = None
    absent_presence_mask = None
    if is_marker_stream and presence_targets is not None and presence_mask is not None:
        positive_presence_mask = presence_mask & (presence_targets > 0.5)
        absent_presence_mask = presence_mask & (presence_targets <= 0.5)
        if presence_logits is not None and torch.any(presence_mask):
            if presence_loss_enabled:
                presence_loss = F.binary_cross_entropy_with_logits(
                    presence_logits[presence_mask],
                    presence_targets[presence_mask],
                )
                total_loss = total_loss + presence_loss
                metrics["presence_loss"] = float(presence_loss.item())
            pred_presence = presence_logits[presence_mask] > 0
            target_presence = presence_targets[presence_mask] > 0.5
            metrics["presence_accuracy"] = float((pred_presence == target_presence).to(dtype=torch.float32).mean().item())
            true_positive = (pred_presence & target_presence).sum().to(dtype=torch.float32)
            false_positive = (pred_presence & ~target_presence).sum().to(dtype=torch.float32)
            false_negative = (~pred_presence & target_presence).sum().to(dtype=torch.float32)
            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative
            metrics["presence_precision"] = 0.0 if float(precision_denominator.item()) == 0.0 else float((true_positive / precision_denominator).item())
            metrics["presence_recall"] = 0.0 if float(recall_denominator.item()) == 0.0 else float((true_positive / recall_denominator).item())

    contact_logits = outputs.get("dense_joint_contact_logits")
    contact_log_conf = outputs.get("dense_joint_contact_log_conf")
    contact_targets = batch.get("contact_targets")
    contact_mask = batch.get("contact_supervision_mask")
    if (
        is_marker_stream
        and isinstance(contact_logits, torch.Tensor)
        and isinstance(contact_targets, torch.Tensor)
        and isinstance(contact_mask, torch.Tensor)
    ):
        if not isinstance(contact_log_conf, torch.Tensor):
            raise ValueError("contact confidence requires outputs['dense_joint_contact_log_conf']")
        contact_targets = contact_targets.to(device=contact_logits.device, dtype=contact_logits.dtype)
        contact_mask = contact_mask.to(device=contact_logits.device, dtype=torch.bool)
        contact_loss, contact_conf_stats = masked_contact_confidence_bce_with_logits(
            contact_logits,
            contact_targets,
            contact_mask,
            contact_log_conf,
            alpha=runtime_config.contact_confidence_alpha,
            use_confidence=use_contact_conf,
        )
        total_loss = total_loss + runtime_config.contact_loss_weight * contact_loss
        _record_loss_metrics(
            metrics,
            "contact_loss",
            contact_loss,
            runtime_config.contact_loss_weight,
        )
        _record_contact_confidence_metrics(
            metrics,
            "contact",
            contact_conf_stats,
            enabled=use_contact_conf,
        )
        metrics.update(contact_classification_metrics(contact_logits, contact_targets, contact_mask))

    marker_contact_logits = outputs.get("dense_vertex_contact_logits")
    marker_contact_log_conf = outputs.get("dense_vertex_contact_log_conf")
    marker_contact_targets = batch.get("marker_contact_targets")
    marker_contact_mask = batch.get("marker_contact_supervision_mask")
    if (
        is_marker_stream
        and isinstance(marker_contact_logits, torch.Tensor)
        and isinstance(marker_contact_targets, torch.Tensor)
        and isinstance(marker_contact_mask, torch.Tensor)
    ):
        if not isinstance(marker_contact_log_conf, torch.Tensor):
            raise ValueError("marker contact confidence requires outputs['dense_vertex_contact_log_conf']")
        marker_contact_targets = marker_contact_targets.to(
            device=marker_contact_logits.device,
            dtype=marker_contact_logits.dtype,
        )
        marker_contact_mask = marker_contact_mask.to(device=marker_contact_logits.device, dtype=torch.bool)
        marker_contact_loss, marker_contact_conf_stats = masked_contact_confidence_bce_with_logits(
            marker_contact_logits,
            marker_contact_targets,
            marker_contact_mask,
            marker_contact_log_conf,
            alpha=runtime_config.contact_confidence_alpha,
            use_confidence=use_contact_conf,
        )
        total_loss = total_loss + runtime_config.marker_contact_loss_weight * marker_contact_loss
        _record_loss_metrics(
            metrics,
            "marker_contact_loss",
            marker_contact_loss,
            runtime_config.marker_contact_loss_weight,
        )
        _record_contact_confidence_metrics(
            metrics,
            "marker_contact",
            marker_contact_conf_stats,
            enabled=use_contact_conf,
        )
        metrics.update(
            {
                f"marker_{name}": value
                for name, value in contact_classification_metrics(
                    marker_contact_logits,
                    marker_contact_targets,
                    marker_contact_mask,
                ).items()
            }
        )

    distance_specs = (
        ("contact_distance", "dense_joint_contact_log_distance", "contact_distance_targets", "contact_distance_supervision_mask", runtime_config.contact_distance_loss_weight),
        ("marker_contact_distance", "dense_vertex_contact_log_distance", "marker_contact_distance_targets", "marker_contact_distance_supervision_mask", runtime_config.marker_contact_distance_loss_weight),
    )
    if is_marker_stream:
        for metric_name, output_key, target_key, mask_key, weight in distance_specs:
            prediction = outputs.get(output_key)
            target = batch.get(target_key)
            mask = batch.get(mask_key)
            if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor) or not isinstance(mask, torch.Tensor):
                continue
            target = target.to(device=prediction.device, dtype=prediction.dtype)
            mask = mask.to(device=prediction.device, dtype=torch.bool)
            distance_loss = _masked_log_distance_huber_loss(prediction, target, mask)
            if distance_loss is not None:
                total_loss = total_loss + float(weight) * distance_loss
                _record_loss_metrics(metrics, f"{metric_name}_loss", distance_loss, float(weight), mask)

    keypoint_heatmap_logits = outputs.get("keypoint_heatmap_logits")
    if not isinstance(keypoint_heatmap_logits, torch.Tensor):
        heatmap_logits = outputs.get("heatmap_logits")
        if isinstance(heatmap_logits, torch.Tensor) and heatmap_logits.ndim == 6:
            keypoint_heatmap_logits = heatmap_logits
    if (
        is_marker_stream
        and runtime_config.keypoint_heatmap_loss_weight > 0.0
        and isinstance(keypoint_heatmap_logits, torch.Tensor)
        and keypoint_heatmap_logits.ndim == 6
    ):
        keypoint_targets = outputs.get("keypoint_heatmap_targets")
        keypoint_mask = outputs.get("keypoint_heatmap_mask")
        if not (
            isinstance(keypoint_targets, torch.Tensor)
            and isinstance(keypoint_mask, torch.Tensor)
            and keypoint_targets.shape == keypoint_heatmap_logits.shape
            and keypoint_mask.shape == keypoint_heatmap_logits.shape[:4]
        ):
            keypoint_target_bundle = build_keypoint_heatmap_targets(
                batch,
                heatmap_height=keypoint_heatmap_logits.shape[-2],
                heatmap_width=keypoint_heatmap_logits.shape[-1],
                device=keypoint_heatmap_logits.device,
                dtype=keypoint_heatmap_logits.dtype,
            )
            keypoint_targets = keypoint_target_bundle.heatmaps
            keypoint_mask = keypoint_target_bundle.mask
        keypoint_targets = keypoint_targets.to(device=keypoint_heatmap_logits.device, dtype=keypoint_heatmap_logits.dtype)
        keypoint_mask = keypoint_mask.to(device=keypoint_heatmap_logits.device, dtype=torch.bool)
        keypoint_channel_mask = _keypoint_heatmap_channel_mask(batch, keypoint_mask)
        if torch.any(keypoint_channel_mask):
            valid_cells = keypoint_channel_mask.unsqueeze(-1).unsqueeze(-1).expand_as(keypoint_heatmap_logits)
            bce = F.binary_cross_entropy_with_logits(
                keypoint_heatmap_logits,
                keypoint_targets,
                reduction="none",
            )
            positive_cells = valid_cells & (keypoint_targets > 0.5)
            negative_cells = valid_cells & ~positive_cells
            positive_loss = (
                bce[positive_cells].mean()
                if torch.any(positive_cells)
                else keypoint_heatmap_logits.sum() * 0.0
            )
            negative_loss = (
                bce[negative_cells].mean()
                if torch.any(negative_cells)
                else keypoint_heatmap_logits.sum() * 0.0
            )
            positive_weight = float(runtime_config.keypoint_heatmap_positive_weight)
            if positive_weight <= 0.0:
                positive_weight = float(max(keypoint_heatmap_logits.shape[-2] * keypoint_heatmap_logits.shape[-1] - 1, 1))
            keypoint_heatmap_loss = positive_weight * positive_loss + negative_loss
            total_loss = total_loss + runtime_config.keypoint_heatmap_loss_weight * keypoint_heatmap_loss
            _record_loss_metrics(
                metrics,
                "keypoint_heatmap_loss",
                keypoint_heatmap_loss,
                runtime_config.keypoint_heatmap_loss_weight,
                keypoint_channel_mask,
            )
            metrics["keypoint_heatmap_positive_count"] = float(positive_cells.to(dtype=torch.float32).sum().item())
            metrics["keypoint_heatmap_negative_count"] = float(negative_cells.to(dtype=torch.float32).sum().item())

    keypoint_offset = outputs.get("keypoint_offset")
    if (
        is_marker_stream
        and runtime_config.keypoint_offset_loss_weight > 0.0
        and isinstance(keypoint_offset, torch.Tensor)
        and keypoint_offset.ndim == 7
        and keypoint_offset.shape[4] == 2
    ):
        keypoint_offset_targets = outputs.get("keypoint_offset_targets")
        keypoint_offset_mask = outputs.get("keypoint_offset_mask")
        keypoint_cell_indices = outputs.get("keypoint_cell_indices")
        if not (
            isinstance(keypoint_offset_targets, torch.Tensor)
            and isinstance(keypoint_offset_mask, torch.Tensor)
            and isinstance(keypoint_cell_indices, torch.Tensor)
            and keypoint_offset_targets.shape == (*keypoint_offset.shape[:4], 2)
            and keypoint_offset_mask.shape == keypoint_offset.shape[:4]
            and keypoint_cell_indices.shape == keypoint_offset.shape[:4]
        ):
            keypoint_target_bundle = build_keypoint_heatmap_targets(
                batch,
                heatmap_height=keypoint_offset.shape[-2],
                heatmap_width=keypoint_offset.shape[-1],
                device=keypoint_offset.device,
                dtype=keypoint_offset.dtype,
            )
            keypoint_offset_targets = keypoint_target_bundle.offsets
            keypoint_offset_mask = keypoint_target_bundle.mask
            keypoint_cell_indices = keypoint_target_bundle.cell_indices
        keypoint_offset_targets = keypoint_offset_targets.to(device=keypoint_offset.device, dtype=keypoint_offset.dtype)
        keypoint_offset_mask = keypoint_offset_mask.to(device=keypoint_offset.device, dtype=torch.bool)
        keypoint_cell_indices = keypoint_cell_indices.to(device=keypoint_offset.device, dtype=torch.long).clamp_min(0)
        if torch.any(keypoint_offset_mask):
            height, width = keypoint_offset.shape[-2:]
            flat_offset = keypoint_offset.reshape(*keypoint_offset.shape[:4], 2, height * width)
            offset_index = keypoint_cell_indices.unsqueeze(-1).unsqueeze(-1).expand(*keypoint_offset.shape[:4], 2, 1)
            pred_offsets = 0.5 * torch.tanh(flat_offset.gather(-1, offset_index).squeeze(-1))
            keypoint_offset_loss = F.smooth_l1_loss(
                pred_offsets.to(dtype=torch.float32)[keypoint_offset_mask],
                keypoint_offset_targets.to(dtype=torch.float32)[keypoint_offset_mask],
            )
            total_loss = total_loss + runtime_config.keypoint_offset_loss_weight * keypoint_offset_loss
            _record_loss_metrics(
                metrics,
                "keypoint_offset_loss",
                keypoint_offset_loss,
                runtime_config.keypoint_offset_loss_weight,
                keypoint_offset_mask,
            )

    vertex_xyz_targets, vertex_xyz_mask, vertex_visibility_targets, vertex_visibility_mask = _build_vertex_targets(
        batch,
        num_vertices=dense_vertex_xyz.shape[3],
        device=device,
        dtype=dtype,
    )
    depth_consistency_vertex_visibility_targets = vertex_visibility_targets
    depth_consistency_vertex_visibility_mask = vertex_visibility_mask
    marker_loss_mask = batch.get("marker_supervision_mask")
    if marker_loss_mask is not None:
        marker_loss_mask = _fixed_side_tensor(batch, "marker_supervision_mask").to(device=device)
    elif not is_marker_stream:
        marker_loss_mask = torch.zeros(dense_vertex_xyz.shape[:3], dtype=torch.bool, device=device)
    if positive_presence_mask is not None:
        vertex_xyz_mask = vertex_xyz_mask & positive_presence_mask.unsqueeze(-1)
        vertex_visibility_mask = vertex_visibility_mask & positive_presence_mask.unsqueeze(-1)
    if absent_presence_mask is not None:
        absent_vertex_mask = absent_presence_mask.unsqueeze(-1).expand_as(dense_vertex_visibility_logits)
        vertex_visibility_mask = vertex_visibility_mask | absent_vertex_mask
        vertex_visibility_targets = torch.where(absent_vertex_mask, torch.zeros_like(vertex_visibility_targets), vertex_visibility_targets)
    if marker_loss_mask is not None:
        marker_or_absent_mask = marker_loss_mask if absent_presence_mask is None else marker_loss_mask | absent_presence_mask
        vertex_xyz_mask = vertex_xyz_mask & marker_loss_mask.unsqueeze(-1)
        vertex_visibility_mask = vertex_visibility_mask & marker_or_absent_mask.unsqueeze(-1)
    if block_hand_gt:
        vertex_visibility_mask = torch.zeros_like(vertex_visibility_mask)

    dense_joint_xyz = outputs.get("dense_joint_xyz")
    dense_joint_visibility_logits = outputs.get("dense_joint_visibility_logits")
    joint_targets = None
    joint_visibility_targets = None
    joint_xyz_mask = None
    joint_visibility_mask = None
    if dense_joint_xyz is not None and dense_joint_visibility_logits is not None:
        joint_targets, joint_visibility_targets, joint_xyz_mask, joint_visibility_mask = _build_joint_targets(
            batch,
            device=device,
            dtype=dtype,
        )
        if positive_presence_mask is not None:
            joint_xyz_mask = joint_xyz_mask & positive_presence_mask.unsqueeze(-1)
            joint_visibility_mask = joint_visibility_mask & positive_presence_mask.unsqueeze(-1)
        if absent_presence_mask is not None:
            absent_joint_mask = absent_presence_mask.unsqueeze(-1).expand_as(dense_joint_visibility_logits)
            joint_visibility_mask = joint_visibility_mask | absent_joint_mask
            joint_visibility_targets = torch.where(absent_joint_mask, torch.zeros_like(joint_visibility_targets), joint_visibility_targets)
        if marker_loss_mask is not None:
            marker_or_absent_mask = marker_loss_mask if absent_presence_mask is None else marker_loss_mask | absent_presence_mask
            joint_xyz_mask = joint_xyz_mask & marker_loss_mask.unsqueeze(-1)
            joint_visibility_mask = joint_visibility_mask & marker_or_absent_mask.unsqueeze(-1)
        if block_hand_gt:
            joint_visibility_mask = torch.zeros_like(joint_visibility_mask)

    dense_vertex_offset = outputs.get("dense_vertex_offset")
    dense_joint_offset = outputs.get("dense_joint_offset")
    root_joint_mask = None
    joint_offset_mask = None
    target_joint_offset = None
    vertex_offset_mask = None
    target_vertex_offset = None
    if joint_targets is not None and joint_xyz_mask is not None:
        root_joint_mask = joint_xyz_mask[..., :1]
        joint_offset_mask = joint_xyz_mask & root_joint_mask.expand_as(joint_xyz_mask)
        target_joint_offset = joint_targets - joint_targets[..., :1, :]
        vertex_offset_mask = vertex_xyz_mask & root_joint_mask.expand_as(vertex_xyz_mask)
        target_vertex_offset = vertex_xyz_targets - joint_targets[..., :1, :]
    needs_joint_offsets = joint_offset_mask is not None and torch.any(joint_offset_mask)
    needs_vertex_offsets = vertex_offset_mask is not None and torch.any(vertex_offset_mask)
    if block_hand_gt:
        needs_joint_offsets = False
        needs_vertex_offsets = False
    if needs_joint_offsets and not isinstance(dense_joint_offset, torch.Tensor):
        raise ValueError("root-offset hand representation requires outputs['dense_joint_offset']")
    if needs_vertex_offsets and not isinstance(dense_vertex_offset, torch.Tensor):
        raise ValueError("root-offset hand representation requires outputs['dense_vertex_offset']")
    if needs_joint_offsets or needs_vertex_offsets:
        scale_vertex_offset = (
            dense_vertex_offset
            if isinstance(dense_vertex_offset, torch.Tensor)
            else dense_joint_offset[..., :0, :]
        )
        scale_vertex_mask = (
            vertex_offset_mask
            if vertex_offset_mask is not None
            else joint_offset_mask[..., :0]
        )
        target_scale_vertex_offset = (
            target_vertex_offset
            if target_vertex_offset is not None
            else target_joint_offset[..., :0, :]
        )
        pred_hand_scale = _hand_slot_average_distance(
            vertex_xyz=scale_vertex_offset,
            vertex_mask=scale_vertex_mask,
            joint_xyz=dense_joint_offset if isinstance(dense_joint_offset, torch.Tensor) else None,
            joint_mask=joint_offset_mask,
        )
        target_hand_scale = _hand_slot_average_distance(
            vertex_xyz=target_scale_vertex_offset,
            vertex_mask=scale_vertex_mask,
            joint_xyz=target_joint_offset,
            joint_mask=joint_offset_mask,
        )
        if needs_vertex_offsets:
            dense_vertex_log_conf = outputs.get("dense_vertex_log_conf")
            if not isinstance(dense_vertex_log_conf, torch.Tensor):
                raise ValueError("hand confidence requires outputs['dense_vertex_log_conf']")
            normalized_dense_vertex_offset = _normalize_slot_points(dense_vertex_offset, pred_hand_scale)
            normalized_target_vertex_offset = _normalize_slot_points(target_vertex_offset, target_hand_scale)
            vertex_offset_error = (normalized_dense_vertex_offset - normalized_target_vertex_offset).abs().mean(dim=-1)
            vertex_local_loss, vertex_stats = hand_confidence_weighted_loss(
                vertex_offset_error,
                dense_vertex_log_conf,
                vertex_offset_mask,
                alpha=runtime_config.hand_conf_local_alpha,
                log_min=runtime_config.hand_conf_log_min,
                log_max=runtime_config.hand_conf_log_max,
                use_confidence=use_hand_conf,
            )
            total_loss = total_loss + runtime_config.hand_vertex_offset_loss_weight * vertex_local_loss
            _record_loss_metrics(
                metrics,
                "vertex_xyz_loss",
                vertex_local_loss,
                runtime_config.hand_vertex_offset_loss_weight,
                vertex_offset_mask,
            )
            _record_loss_metrics(
                metrics,
                "vertex_offset_loss",
                vertex_local_loss,
                runtime_config.hand_vertex_offset_loss_weight,
                vertex_offset_mask,
            )
            if vertex_stats:
                metrics["vertex_xyz_loss_raw"] = vertex_stats["raw"]
                metrics["vertex_xyz_loss_conf"] = vertex_stats["conf"]
                metrics["vertex_conf_mean"] = vertex_stats["conf_mean"]
                metrics["vertex_conf_p10"] = vertex_stats["conf_p10"]
                metrics["vertex_conf_p50"] = vertex_stats["conf_p50"]
                metrics["vertex_conf_p90"] = vertex_stats["conf_p90"]
                metrics["vertex_log_conf_mean"] = vertex_stats["log_conf_mean"]
        if needs_joint_offsets:
            dense_joint_log_conf = outputs.get("dense_joint_log_conf")
            if not isinstance(dense_joint_log_conf, torch.Tensor):
                raise ValueError("hand confidence requires outputs['dense_joint_log_conf']")
            normalized_dense_joint_offset = _normalize_slot_points(dense_joint_offset, pred_hand_scale)
            normalized_target_joint_offset = _normalize_slot_points(target_joint_offset, target_hand_scale)
            joint_offset_error = (normalized_dense_joint_offset - normalized_target_joint_offset).abs().mean(dim=-1)
            joint_xyz_loss, joint_stats = hand_confidence_weighted_loss(
                joint_offset_error[..., 1:],
                dense_joint_log_conf[..., 1:],
                joint_offset_mask[..., 1:],
                alpha=runtime_config.hand_conf_local_alpha,
                log_min=runtime_config.hand_conf_log_min,
                log_max=runtime_config.hand_conf_log_max,
                use_confidence=use_hand_conf,
            )
            total_loss = total_loss + runtime_config.hand_joint_offset_loss_weight * joint_xyz_loss
            _record_loss_metrics(
                metrics,
                "joint_xyz_loss",
                joint_xyz_loss,
                runtime_config.hand_joint_offset_loss_weight,
                joint_offset_mask[..., 1:],
            )
            if joint_stats:
                metrics["joint_xyz_loss_raw"] = joint_stats["raw"]
                metrics["joint_xyz_loss_conf"] = joint_stats["conf"]
                metrics["joint_conf_mean"] = joint_stats["conf_mean"]
                metrics["joint_conf_p10"] = joint_stats["conf_p10"]
                metrics["joint_conf_p50"] = joint_stats["conf_p50"]
                metrics["joint_conf_p90"] = joint_stats["conf_p90"]
                metrics["joint_log_conf_mean"] = joint_stats["log_conf_mean"]
    if torch.any(vertex_visibility_mask):
        vertex_visibility_loss = F.binary_cross_entropy_with_logits(
            dense_vertex_visibility_logits[vertex_visibility_mask],
            vertex_visibility_targets[vertex_visibility_mask],
        )
        total_loss = total_loss + runtime_config.marker_visibility_loss_weight * vertex_visibility_loss
        _record_loss_metrics(
            metrics,
            "vertex_visibility_loss",
            vertex_visibility_loss,
            runtime_config.marker_visibility_loss_weight,
            vertex_visibility_mask,
        )

    if (
        dense_joint_xyz is not None
        and dense_joint_visibility_logits is not None
        and joint_targets is not None
        and joint_visibility_targets is not None
        and joint_xyz_mask is not None
        and joint_visibility_mask is not None
    ):
        if torch.any(joint_visibility_mask):
            joint_visibility_loss = F.binary_cross_entropy_with_logits(
                dense_joint_visibility_logits[joint_visibility_mask],
                joint_visibility_targets[joint_visibility_mask],
            )
            total_loss = total_loss + runtime_config.joint_visibility_loss_weight * joint_visibility_loss
            _record_loss_metrics(
                metrics,
                "joint_visibility_loss",
                joint_visibility_loss,
                runtime_config.joint_visibility_loss_weight,
                joint_visibility_mask,
            )

    depth = batch.get("depth")
    intrinsics = batch.get("intrinsics")
    camera_pose_target = batch.get("camera_pose")
    pred_depth = outputs.get("depth")
    pred_depth_conf = outputs.get("depth_conf")
    pred_intrinsics = outputs.get("intrinsics")
    pred_camera_pose = outputs.get("camera_pose")
    pred_camera_pose_encoding = outputs.get("camera_pose_encoding")
    resized_depth = None
    resized_intrinsics = None
    resized_depth_valid_mask = None
    valid_depth_frames = None
    valid_intrinsics_frames = None
    valid_camera_pose_frames = None
    camera_pose_target_for_scale = None

    if depth is not None and pred_depth is not None:
        valid_depth_frames = batch["depth_supervision_mask"].to(device=device)
        if three_r_loss_mask is not None:
            valid_depth_frames = valid_depth_frames & three_r_loss_mask
        depth_valid_mask = batch.get("depth_valid_mask")
        if torch.any(valid_depth_frames):
            depth = depth.to(device=device, dtype=dtype)
            depth_valid_mask = None if depth_valid_mask is None else depth_valid_mask.to(device=device)
            resized_depth = F.interpolate(
                depth.reshape(depth.shape[0] * depth.shape[1], 1, depth.shape[-2], depth.shape[-1]),
                size=pred_depth.shape[-2:],
                mode="nearest",
            ).reshape(depth.shape[0], depth.shape[1], pred_depth.shape[-2], pred_depth.shape[-1])
            valid_depth_frames &= torch.isfinite(resized_depth).any(dim=(-2, -1))
            if depth_valid_mask is not None:
                resized_depth_valid_mask = F.interpolate(
                    depth_valid_mask.to(dtype=torch.float32).reshape(
                        depth.shape[0] * depth.shape[1], 1, depth.shape[-2], depth.shape[-1]
                    ),
                    size=pred_depth.shape[-2:],
                    mode="nearest",
                ).reshape(depth.shape[0], depth.shape[1], pred_depth.shape[-2], pred_depth.shape[-1]) > 0.5
                valid_depth_frames &= resized_depth_valid_mask.any(dim=(-2, -1))
            if torch.any(valid_depth_frames):
                depth_mask = valid_depth_frames.unsqueeze(-1).unsqueeze(-1).expand_as(pred_depth)
                depth_mask = depth_mask & torch.isfinite(resized_depth)
                if resized_depth_valid_mask is not None:
                    depth_mask = depth_mask & resized_depth_valid_mask
                if torch.any(depth_mask):
                    if not isinstance(pred_depth_conf, torch.Tensor):
                        raise ValueError("新版 3R depth loss 需要 outputs['depth_conf']")
                    if pred_depth_conf.shape != pred_depth.shape:
                        raise ValueError("outputs['depth_conf'] 必须和 outputs['depth'] 形状一致")
                    _log_depth_conf_stats(metrics, stream_name, pred_depth_conf, depth_mask)
                    _log_depth_error_metrics(
                        metrics,
                        stream_name,
                        pred_depth,
                        resized_depth,
                        pred_depth_conf,
                        depth_mask,
                        suffix="raw",
                        alpha=runtime_config.depth_confidence_alpha,
                        robust_quantile=runtime_config.depth_robust_quantile,
                        log_reward=False,
                    )
                    depth_conf_loss = (pred_depth.sum() + pred_depth_conf.sum()) * 0.0
                    if (
                        intrinsics is not None
                        and pred_intrinsics is not None
                        and pred_camera_pose is not None
                        and pred_camera_pose.ndim >= 4
                        and pred_camera_pose.shape[-2:] == (4, 4)
                        and camera_pose_target is not None
                        and "intrinsics_supervision_mask" in batch
                        and "camera_pose_supervision_mask" in batch
                    ):
                        intrinsics_for_depth_scale = intrinsics.to(device=device, dtype=dtype)
                        target_intrinsics_for_depth_scale = _scale_intrinsics_to_size(
                            intrinsics_for_depth_scale,
                            source_height=batch["images"].shape[-2],
                            source_width=batch["images"].shape[-1],
                            target_height=pred_depth.shape[-2],
                            target_width=pred_depth.shape[-1],
                        )
                        intrinsics_scale_mask = batch["intrinsics_supervision_mask"].to(device=device)
                        camera_scale_mask = batch["camera_pose_supervision_mask"].to(device=device)
                        if three_r_loss_mask is not None:
                            intrinsics_scale_mask = intrinsics_scale_mask & three_r_loss_mask
                            camera_scale_mask = camera_scale_mask & three_r_loss_mask
                        camera_pose_target_for_depth_scale = camera_pose_target.to(device=device, dtype=dtype)
                        scale_frames = valid_depth_frames & intrinsics_scale_mask & camera_scale_mask
                        scale_frames &= torch.isfinite(target_intrinsics_for_depth_scale).all(dim=(-2, -1))
                        scale_frames &= torch.isfinite(camera_pose_target_for_depth_scale).all(dim=(-2, -1))
                        if torch.any(scale_frames):
                            valid_depth_for_scale = (
                                torch.ones_like(resized_depth, dtype=torch.bool)
                                if resized_depth_valid_mask is None
                                else resized_depth_valid_mask
                            )
                            gt_depth_scale_mask = valid_depth_for_scale & scale_frames.unsqueeze(-1).unsqueeze(-1)
                            gt_depth_scene_scale = scene_average_distance_from_depth(
                                resized_depth.detach(),
                                target_intrinsics_for_depth_scale.detach(),
                                camera_pose_target_for_depth_scale.detach(),
                                gt_depth_scale_mask,
                            )
                            pred_depth_scene_scale = scene_average_distance_from_depth(
                                pred_depth.detach(),
                                pred_intrinsics.detach(),
                                pred_camera_pose.detach(),
                            )
                            gt_depth_scene_valid = _valid_scene_scale_mask(
                                gt_depth_scene_scale,
                                min_value=runtime_config.scene_scale_min_value,
                            )
                            pred_depth_scene_valid = _valid_scene_scale_mask(
                                pred_depth_scene_scale,
                                min_value=runtime_config.scene_scale_min_value,
                            )
                            _record_scene_scale_guard_metrics(
                                metrics,
                                f"{stream_name}_depth_gt_scene_scale",
                                gt_depth_scene_scale,
                                gt_depth_scene_valid,
                            )
                            _record_scene_scale_guard_metrics(
                                metrics,
                                f"{stream_name}_depth_pred_scene_scale",
                                pred_depth_scene_scale,
                                pred_depth_scene_valid,
                            )
                            scale_sequence_mask = (
                                _scene_scale_sequence_mask(
                                    resized_depth,
                                    target_intrinsics_for_depth_scale,
                                    camera_pose_target_for_depth_scale,
                                    gt_depth_scale_mask,
                                )
                                & _scene_scale_sequence_mask(pred_depth, pred_intrinsics, pred_camera_pose)
                                & gt_depth_scene_valid
                                & pred_depth_scene_valid
                            )
                            normalized_depth_mask = depth_mask & scale_sequence_mask.view(-1, 1, 1, 1)
                            if torch.any(normalized_depth_mask):
                                pred_depth_for_loss = pred_depth / pred_depth_scene_scale.view(-1, 1, 1, 1).clamp_min(1e-6)
                                resized_depth_for_loss = resized_depth / gt_depth_scene_scale.view(-1, 1, 1, 1).clamp_min(1e-6)
                                _log_depth_error_metrics(
                                    metrics,
                                    stream_name,
                                    pred_depth_for_loss,
                                    resized_depth_for_loss,
                                    pred_depth_conf,
                                    normalized_depth_mask,
                                    suffix="norm",
                                    alpha=runtime_config.depth_confidence_alpha,
                                    robust_quantile=runtime_config.depth_robust_quantile,
                                    log_reward=True,
                                )
                                depth_conf_loss, depth_dropped = depth_confidence_loss(
                                    pred_depth_for_loss,
                                    resized_depth_for_loss,
                                    pred_depth_conf,
                                    normalized_depth_mask,
                                    alpha=runtime_config.depth_confidence_alpha,
                                    relative_weight_max=runtime_config.depth_relative_weight_max,
                                    robust_quantile=runtime_config.depth_robust_quantile,
                                    max_sample_loss=runtime_config.scale_normalized_max_sample_loss,
                                    return_dropped=True,
                                )
                                if depth_dropped > 0:
                                    metrics[f"{stream_name}_depth_conf_dropped_samples"] = float(depth_dropped)
                    total_loss = total_loss + runtime_config.depth_supervision_loss_weight * depth_conf_loss
                    metrics[f"{stream_name}_depth_conf_loss"] = float(depth_conf_loss.item())

    has_vggt_fov_output = pred_camera_pose_encoding is not None and pred_camera_pose_encoding.shape[-1] >= 9
    if intrinsics is not None and (pred_intrinsics is not None or has_vggt_fov_output):
        valid_intrinsics_frames = batch["intrinsics_supervision_mask"].to(device=device)
        if three_r_loss_mask is not None:
            valid_intrinsics_frames = valid_intrinsics_frames & three_r_loss_mask
        if torch.any(valid_intrinsics_frames):
            intrinsics = intrinsics.to(device=device, dtype=dtype)
            source_height = batch["images"].shape[-2]
            source_width = batch["images"].shape[-1]
            target_height = pred_depth.shape[-2] if pred_depth is not None else source_height
            target_width = pred_depth.shape[-1] if pred_depth is not None else source_width
            resized_intrinsics = _scale_intrinsics_to_size(
                intrinsics,
                source_height=source_height,
                source_width=source_width,
                target_height=target_height,
                target_width=target_width,
            )
            valid_intrinsics_frames &= torch.isfinite(resized_intrinsics).all(dim=(-2, -1))
            if torch.any(valid_intrinsics_frames):
                intrinsics_loss = None
                if has_vggt_fov_output:
                    valid_fov_frames = valid_intrinsics_frames & (intrinsics[..., 0, 0] > 0) & (intrinsics[..., 1, 1] > 0)
                    if torch.any(valid_fov_frames):
                        target_fov = _vggt_fov_from_intrinsics(
                            intrinsics,
                            image_height=source_height,
                            image_width=source_width,
                        )
                        fov_mask = valid_fov_frames.unsqueeze(-1).expand_as(pred_camera_pose_encoding[..., 7:9])
                        intrinsics_loss = F.l1_loss(
                            pred_camera_pose_encoding[..., 7:9][fov_mask],
                            target_fov[fov_mask],
                        )
                elif pred_intrinsics is not None:
                    direct_intrinsics_mask = valid_intrinsics_frames & torch.isfinite(pred_intrinsics).all(dim=(-2, -1))
                    direct_intrinsics_mask &= (pred_intrinsics[..., 0, 0] > 0) & (pred_intrinsics[..., 1, 1] > 0)
                    direct_intrinsics_mask &= (resized_intrinsics[..., 0, 0] > 0) & (resized_intrinsics[..., 1, 1] > 0)
                    if torch.any(direct_intrinsics_mask):
                        pred_intrinsics_norm = _normalized_intrinsics_vector(
                            pred_intrinsics,
                            image_height=target_height,
                            image_width=target_width,
                        )
                        target_intrinsics_norm = _normalized_intrinsics_vector(
                            resized_intrinsics,
                            image_height=target_height,
                            image_width=target_width,
                        )
                        intrinsics_mask = direct_intrinsics_mask.unsqueeze(-1).expand_as(pred_intrinsics_norm)
                        intrinsics_loss = F.smooth_l1_loss(
                            pred_intrinsics_norm[intrinsics_mask],
                            target_intrinsics_norm[intrinsics_mask],
                        )
                if intrinsics_loss is not None:
                    total_loss = total_loss + runtime_config.intrinsics_supervision_loss_weight * intrinsics_loss
                    metrics[f"{stream_name}_intrinsics_loss"] = float(intrinsics_loss.item())

    if camera_pose_target is not None and (
        (pred_camera_pose_encoding is not None and pred_camera_pose_encoding.shape[-1] >= 7)
        or (pred_camera_pose is not None and pred_camera_pose.ndim >= 4 and pred_camera_pose.shape[-2:] == (4, 4))
    ):
        camera_pose_target = camera_pose_target.to(device=device, dtype=dtype)
        camera_pose_mask = batch["camera_pose_supervision_mask"].to(device=device)
        if three_r_loss_mask is not None:
            camera_pose_mask = camera_pose_mask & three_r_loss_mask
        camera_pose_mask &= torch.isfinite(camera_pose_target).all(dim=(-2, -1))
        valid_camera_pose_frames = camera_pose_mask
        camera_pose_target_for_scale = camera_pose_target
        if torch.any(camera_pose_mask):
            absolute_translation_weight = float(runtime_config.camera_absolute_translation_loss_weight)
            if (
                absolute_translation_weight > 0.0
                and pred_camera_pose is not None
                and pred_camera_pose.ndim >= 4
                and pred_camera_pose.shape[-2:] == (4, 4)
            ):
                metric_camera_pose = reconstruct_metric_scale_outputs(outputs)["camera_pose"]
                absolute_translation_pred = metric_camera_pose[..., :3, 3]
                absolute_translation_target = camera_pose_target[..., :3, 3]
                absolute_translation_mask = camera_pose_mask & torch.isfinite(absolute_translation_pred).all(dim=-1)
                if torch.any(absolute_translation_mask):
                    absolute_translation_diff = torch.where(
                        absolute_translation_mask.unsqueeze(-1),
                        absolute_translation_pred.float() - absolute_translation_target.float(),
                        torch.zeros_like(absolute_translation_pred, dtype=torch.float32),
                    )
                    absolute_translation_loss = torch.linalg.norm(absolute_translation_diff, dim=-1)[
                        absolute_translation_mask
                    ].mean().to(dtype=dtype)
                    total_loss = total_loss + absolute_translation_weight * absolute_translation_loss
                    metrics[f"{stream_name}_camera_absolute_translation_l2_loss"] = float(
                        absolute_translation_loss.item()
                    )
            if pred_camera_pose_encoding is not None and pred_camera_pose_encoding.shape[-1] >= 7:
                relative_pose_mask = _relative_pose_supervision_mask(camera_pose_mask)
                if not torch.any(relative_pose_mask):
                    relative_pose_mask = None
                if relative_pose_mask is None:
                    translation_loss = camera_pose_target.new_zeros(())
                    rotation_loss = camera_pose_target.new_zeros(())
                else:
                    relative_pose_pred = _relative_pose_to_first_frame(
                        _vggt_pose_matrix_from_encoding(pred_camera_pose_encoding)
                    )
                    relative_pose_target = _relative_pose_to_first_frame(camera_pose_target)
                    translation_pred = relative_pose_pred[..., :3, 3]
                    translation_target = relative_pose_target[..., :3, 3]
                    translation_loss = translation_pred.sum() * 0.0
                    if (
                        resized_depth is not None
                        and resized_intrinsics is not None
                        and valid_depth_frames is not None
                        and valid_intrinsics_frames is not None
                        and pred_depth is not None
                        and pred_intrinsics is not None
                        and pred_camera_pose is not None
                        and pred_camera_pose.ndim >= 4
                        and pred_camera_pose.shape[-2:] == (4, 4)
                    ):
                        scale_frames = valid_depth_frames & valid_intrinsics_frames & camera_pose_mask
                        if torch.any(scale_frames):
                            valid_depth_for_scale = (
                                torch.ones_like(resized_depth, dtype=torch.bool)
                                if resized_depth_valid_mask is None
                                else resized_depth_valid_mask
                            )
                            gt_depth_scale_mask = valid_depth_for_scale & scale_frames.unsqueeze(-1).unsqueeze(-1)
                            gt_scene_scale = scene_average_distance_from_depth(
                                resized_depth.detach(),
                                resized_intrinsics.detach(),
                                camera_pose_target.detach(),
                                gt_depth_scale_mask,
                            )
                            pred_scene_scale = scene_average_distance_from_depth(
                                pred_depth.detach(),
                                pred_intrinsics.detach(),
                                pred_camera_pose.detach(),
                            )
                            gt_scene_valid = _valid_scene_scale_mask(
                                gt_scene_scale,
                                min_value=runtime_config.scene_scale_min_value,
                            )
                            pred_scene_valid = _valid_scene_scale_mask(
                                pred_scene_scale,
                                min_value=runtime_config.scene_scale_min_value,
                            )
                            _record_scene_scale_guard_metrics(
                                metrics,
                                f"{stream_name}_camera_gt_scene_scale",
                                gt_scene_scale,
                                gt_scene_valid,
                            )
                            _record_scene_scale_guard_metrics(
                                metrics,
                                f"{stream_name}_camera_pred_scene_scale",
                                pred_scene_scale,
                                pred_scene_valid,
                            )
                            scale_sequence_mask = (
                                _scene_scale_sequence_mask(
                                    resized_depth,
                                    resized_intrinsics,
                                    camera_pose_target,
                                    gt_depth_scale_mask,
                                )
                                & _scene_scale_sequence_mask(pred_depth, pred_intrinsics, pred_camera_pose)
                                & gt_scene_valid
                                & pred_scene_valid
                            )
                            translation_mask = relative_pose_mask & scale_sequence_mask.view(-1, 1)
                            if torch.any(translation_mask):
                                normalized_translation_pred = translation_pred / pred_scene_scale.view(-1, 1, 1).clamp_min(1e-6)
                                normalized_translation_target = translation_target / gt_scene_scale.view(-1, 1, 1).clamp_min(1e-6)
                                translation_loss, translation_dropped = _masked_translation_loss(
                                    normalized_translation_pred,
                                    normalized_translation_target,
                                    translation_mask,
                                    max_sample_loss=runtime_config.camera_translation_max_sample_loss,
                                )
                                if translation_dropped > 0:
                                    metrics[f"{stream_name}_camera_translation_dropped_samples"] = float(translation_dropped)
                    rotation_pred = _vggt_quaternion_from_rotation(relative_pose_pred[..., :3, :3])
                    rotation_target = _vggt_quaternion_from_rotation(relative_pose_target[..., :3, :3])
                    expanded_rotation_mask = relative_pose_mask.unsqueeze(-1).expand_as(rotation_pred)
                    rotation_loss = _quaternion_dot_geodesic_loss(
                        rotation_pred[expanded_rotation_mask].reshape(-1, 4),
                        rotation_target[expanded_rotation_mask].reshape(-1, 4),
                    )
            else:
                relative_pose_mask = _relative_pose_supervision_mask(camera_pose_mask)
                if not torch.any(relative_pose_mask):
                    translation_loss = camera_pose_target.new_zeros(())
                    rotation_loss = camera_pose_target.new_zeros(())
                else:
                    relative_pose_pred = _relative_pose_to_first_frame(pred_camera_pose)
                    relative_pose_target = _relative_pose_to_first_frame(camera_pose_target)
                    translation_pred = relative_pose_pred[..., :3, 3]
                    translation_target = relative_pose_target[..., :3, 3]
                    translation_loss = translation_pred.sum() * 0.0
                    if (
                        resized_depth is not None
                        and resized_intrinsics is not None
                        and valid_depth_frames is not None
                        and valid_intrinsics_frames is not None
                        and pred_depth is not None
                        and pred_intrinsics is not None
                        and pred_camera_pose.ndim >= 4
                        and pred_camera_pose.shape[-2:] == (4, 4)
                    ):
                        scale_frames = valid_depth_frames & valid_intrinsics_frames & camera_pose_mask
                        if torch.any(scale_frames):
                            valid_depth_for_scale = (
                                torch.ones_like(resized_depth, dtype=torch.bool)
                                if resized_depth_valid_mask is None
                                else resized_depth_valid_mask
                            )
                            gt_depth_scale_mask = valid_depth_for_scale & scale_frames.unsqueeze(-1).unsqueeze(-1)
                            gt_scene_scale = scene_average_distance_from_depth(
                                resized_depth.detach(),
                                resized_intrinsics.detach(),
                                camera_pose_target.detach(),
                                gt_depth_scale_mask,
                            )
                            pred_scene_scale = scene_average_distance_from_depth(
                                pred_depth.detach(),
                                pred_intrinsics.detach(),
                                pred_camera_pose.detach(),
                            )
                            gt_scene_valid = _valid_scene_scale_mask(
                                gt_scene_scale,
                                min_value=runtime_config.scene_scale_min_value,
                            )
                            pred_scene_valid = _valid_scene_scale_mask(
                                pred_scene_scale,
                                min_value=runtime_config.scene_scale_min_value,
                            )
                            _record_scene_scale_guard_metrics(
                                metrics,
                                f"{stream_name}_camera_gt_scene_scale",
                                gt_scene_scale,
                                gt_scene_valid,
                            )
                            _record_scene_scale_guard_metrics(
                                metrics,
                                f"{stream_name}_camera_pred_scene_scale",
                                pred_scene_scale,
                                pred_scene_valid,
                            )
                            scale_sequence_mask = (
                                _scene_scale_sequence_mask(
                                    resized_depth,
                                    resized_intrinsics,
                                    camera_pose_target,
                                    gt_depth_scale_mask,
                                )
                                & _scene_scale_sequence_mask(pred_depth, pred_intrinsics, pred_camera_pose)
                                & gt_scene_valid
                                & pred_scene_valid
                            )
                            translation_mask = relative_pose_mask & scale_sequence_mask.view(-1, 1)
                            if torch.any(translation_mask):
                                normalized_translation_pred = translation_pred / pred_scene_scale.view(-1, 1, 1).clamp_min(1e-6)
                                normalized_translation_target = translation_target / gt_scene_scale.view(-1, 1, 1).clamp_min(1e-6)
                                translation_loss, translation_dropped = _masked_translation_loss(
                                    normalized_translation_pred,
                                    normalized_translation_target,
                                    translation_mask,
                                    max_sample_loss=runtime_config.camera_translation_max_sample_loss,
                                )
                                if translation_dropped > 0:
                                    metrics[f"{stream_name}_camera_translation_dropped_samples"] = float(translation_dropped)
                    rotation_pred = _vggt_quaternion_from_rotation(relative_pose_pred[..., :3, :3])
                    rotation_target = _vggt_quaternion_from_rotation(relative_pose_target[..., :3, :3])
                    expanded_rotation_mask = relative_pose_mask.unsqueeze(-1).expand_as(rotation_pred)
                    rotation_loss = _quaternion_dot_geodesic_loss(
                        rotation_pred[expanded_rotation_mask].reshape(-1, 4),
                        rotation_target[expanded_rotation_mask].reshape(-1, 4),
                    )
            total_loss = (
                total_loss
                + runtime_config.camera_translation_loss_weight * translation_loss
                + runtime_config.camera_rotation_loss_weight * rotation_loss
            )
            metrics[f"{stream_name}_camera_translation_loss"] = float(translation_loss.item())
            metrics[f"{stream_name}_camera_rotation_loss"] = float(rotation_loss.item())

    if (
        isinstance(pred_depth_conf, torch.Tensor)
        and pred_depth is not None
        and resized_depth is not None
        and resized_intrinsics is not None
        and pred_intrinsics is not None
        and pred_camera_pose is not None
        and pred_camera_pose.ndim >= 4
        and pred_camera_pose.shape[-2:] == (4, 4)
        and camera_pose_target_for_scale is not None
        and valid_depth_frames is not None
        and valid_intrinsics_frames is not None
        and valid_camera_pose_frames is not None
    ):
        point_frames = valid_depth_frames & valid_intrinsics_frames & valid_camera_pose_frames
        if torch.any(point_frames):
            point_depth_mask = point_frames.unsqueeze(-1).unsqueeze(-1).expand_as(pred_depth)
            point_depth_mask = point_depth_mask & torch.isfinite(resized_depth)
            if resized_depth_valid_mask is not None:
                point_depth_mask = point_depth_mask & resized_depth_valid_mask
            if torch.any(point_depth_mask):
                _log_point_error_metrics(
                    metrics,
                    stream_name,
                    pred_depth,
                    resized_depth,
                    pred_intrinsics,
                    resized_intrinsics,
                    pred_camera_pose,
                    camera_pose_target_for_scale,
                    point_depth_mask,
                    suffix="raw",
                    robust_quantile=runtime_config.point_robust_quantile,
                )
                gt_scene_scale = scene_average_distance_from_depth(
                    resized_depth.detach(),
                    resized_intrinsics.detach(),
                    camera_pose_target_for_scale.detach(),
                    point_depth_mask,
                )
                pred_scene_scale = scene_average_distance_from_depth(
                    pred_depth.detach(),
                    pred_intrinsics.detach(),
                    pred_camera_pose.detach(),
                )
                gt_scene_valid = _valid_scene_scale_mask(
                    gt_scene_scale,
                    min_value=runtime_config.scene_scale_min_value,
                )
                pred_scene_valid = _valid_scene_scale_mask(
                    pred_scene_scale,
                    min_value=runtime_config.scene_scale_min_value,
                )
                _record_scene_scale_guard_metrics(
                    metrics,
                    f"{stream_name}_point_gt_scene_scale",
                    gt_scene_scale,
                    gt_scene_valid,
                )
                _record_scene_scale_guard_metrics(
                    metrics,
                    f"{stream_name}_point_pred_scene_scale",
                    pred_scene_scale,
                    pred_scene_valid,
                )
                scale_sequence_mask = (
                    _scene_scale_sequence_mask(
                        resized_depth,
                        resized_intrinsics,
                        camera_pose_target_for_scale,
                        point_depth_mask,
                    )
                    & _scene_scale_sequence_mask(pred_depth, pred_intrinsics, pred_camera_pose)
                    & gt_scene_valid
                    & pred_scene_valid
                )
                point_depth_mask = point_depth_mask & scale_sequence_mask.view(-1, 1, 1, 1)
                point_loss = (pred_depth.sum() + pred_depth_conf.sum()) * 0.0
                if torch.any(point_depth_mask):
                    _log_point_error_metrics(
                        metrics,
                        stream_name,
                        pred_depth,
                        resized_depth,
                        pred_intrinsics,
                        resized_intrinsics,
                        pred_camera_pose,
                        camera_pose_target_for_scale,
                        point_depth_mask,
                        suffix="norm",
                        robust_quantile=runtime_config.point_robust_quantile,
                        pred_scene_scale=pred_scene_scale,
                        target_scene_scale=gt_scene_scale,
                    )
                    point_loss, point_dropped = point_confidence_loss(
                        pred_depth=pred_depth,
                        target_depth=resized_depth,
                        depth_conf=pred_depth_conf,
                        pred_intrinsics=pred_intrinsics,
                        target_intrinsics=resized_intrinsics,
                        pred_camera_pose=pred_camera_pose,
                        target_camera_pose=camera_pose_target_for_scale,
                        pred_scene_scale=pred_scene_scale,
                        target_scene_scale=gt_scene_scale,
                        mask=point_depth_mask,
                        alpha=runtime_config.depth_confidence_alpha,
                        relative_weight_max=runtime_config.depth_relative_weight_max,
                        robust_quantile=runtime_config.point_robust_quantile,
                        max_sample_loss=runtime_config.scale_normalized_max_sample_loss,
                        return_dropped=True,
                    )
                    if point_dropped > 0:
                        metrics[f"{stream_name}_point_supervision_dropped_samples"] = float(point_dropped)
                total_loss = total_loss + runtime_config.point_supervision_loss_weight * point_loss
                metrics[f"{stream_name}_point_supervision_loss"] = float(point_loss.item())

    dense_hand_root_xyz = outputs.get("dense_hand_root_xyz")
    if (
        runtime_config.hand_root_scene_loss_weight > 0.0
        and not block_hand_gt
        and isinstance(dense_hand_root_xyz, torch.Tensor)
        and joint_targets is not None
        and joint_xyz_mask is not None
        and resized_depth is not None
        and resized_intrinsics is not None
        and valid_depth_frames is not None
        and valid_intrinsics_frames is not None
        and valid_camera_pose_frames is not None
        and camera_pose_target_for_scale is not None
        and pred_depth is not None
        and pred_intrinsics is not None
        and pred_camera_pose is not None
        and pred_camera_pose.ndim >= 4
        and pred_camera_pose.shape[-2:] == (4, 4)
    ):
        root_scene_mask = joint_xyz_mask[..., 0]
        scale_frames = valid_depth_frames & valid_intrinsics_frames & valid_camera_pose_frames
        if torch.any(root_scene_mask) and torch.any(scale_frames):
            valid_depth_for_scale = (
                torch.ones_like(resized_depth, dtype=torch.bool)
                if resized_depth_valid_mask is None
                else resized_depth_valid_mask
            )
            gt_depth_scale_mask = valid_depth_for_scale & scale_frames.unsqueeze(-1).unsqueeze(-1)
            gt_scene_scale = scene_average_distance_from_depth(
                resized_depth.detach(),
                resized_intrinsics.detach(),
                camera_pose_target_for_scale.detach(),
                gt_depth_scale_mask,
            )
            pred_scene_scale = scene_average_distance_from_depth(
                pred_depth.detach(),
                pred_intrinsics.detach(),
                pred_camera_pose.detach(),
            )
            gt_scene_valid = _valid_scene_scale_mask(
                gt_scene_scale,
                min_value=runtime_config.scene_scale_min_value,
            )
            pred_scene_valid = _valid_scene_scale_mask(
                pred_scene_scale,
                min_value=runtime_config.scene_scale_min_value,
            )
            _record_scene_scale_guard_metrics(
                metrics,
                f"{stream_name}_hand_root_scene_gt_scene_scale",
                gt_scene_scale,
                gt_scene_valid,
            )
            _record_scene_scale_guard_metrics(
                metrics,
                f"{stream_name}_hand_root_scene_pred_scene_scale",
                pred_scene_scale,
                pred_scene_valid,
            )
            scale_sequence_mask = (
                _scene_scale_sequence_mask(
                    resized_depth,
                    resized_intrinsics,
                    camera_pose_target_for_scale,
                    gt_depth_scale_mask,
                )
                & _scene_scale_sequence_mask(pred_depth, pred_intrinsics, pred_camera_pose)
                & gt_scene_valid
                & pred_scene_valid
            )
            root_scene_mask = root_scene_mask & scale_sequence_mask.view(-1, 1, 1)
            if torch.any(root_scene_mask):
                dense_hand_root_log_conf = outputs.get("dense_hand_root_log_conf")
                if not isinstance(dense_hand_root_log_conf, torch.Tensor):
                    raise ValueError("hand confidence requires outputs['dense_hand_root_log_conf']")
                pred_root_norm = dense_hand_root_xyz / pred_scene_scale.view(-1, 1, 1, 1).clamp_min(1e-6)
                target_root_norm = joint_targets[..., 0, :] / gt_scene_scale.view(-1, 1, 1, 1).clamp_min(1e-6)
                root_scene_error = F.smooth_l1_loss(
                    pred_root_norm,
                    target_root_norm,
                    reduction="none",
                ).mean(dim=-1)
                root_scene_mask, root_scene_dropped = _drop_sequence_samples_by_loss(
                    root_scene_error,
                    root_scene_mask,
                    max_sample_loss=runtime_config.scale_normalized_max_sample_loss,
                )
                hand_root_scene_loss, root_stats = hand_confidence_weighted_loss(
                    root_scene_error,
                    dense_hand_root_log_conf[..., 0],
                    root_scene_mask,
                    alpha=runtime_config.hand_conf_root_alpha,
                    log_min=runtime_config.hand_conf_log_min,
                    log_max=runtime_config.hand_conf_log_max,
                    use_confidence=use_hand_conf,
                )
                total_loss = total_loss + runtime_config.hand_root_scene_loss_weight * hand_root_scene_loss
                _record_loss_metrics(
                    metrics,
                    f"{stream_name}_hand_root_scene_loss",
                    hand_root_scene_loss,
                    runtime_config.hand_root_scene_loss_weight,
                    root_scene_mask,
                )
                if root_scene_dropped > 0:
                    metrics[f"{stream_name}_hand_root_scene_dropped_samples"] = float(root_scene_dropped)
                if root_stats:
                    metrics[f"{stream_name}_hand_root_scene_loss_raw"] = root_stats["raw"]
                    metrics[f"{stream_name}_hand_root_scene_loss_conf"] = root_stats["conf"]
                    metrics[f"{stream_name}_hand_root_conf_mean"] = root_stats["conf_mean"]
                    metrics[f"{stream_name}_hand_root_conf_p10"] = root_stats["conf_p10"]
                    metrics[f"{stream_name}_hand_root_conf_p50"] = root_stats["conf_p50"]
                    metrics[f"{stream_name}_hand_root_conf_p90"] = root_stats["conf_p90"]
                    metrics[f"{stream_name}_hand_root_log_conf_mean"] = root_stats["log_conf_mean"]

    hand_metric_value = _select_hand_metric_value(outputs.get("log_metric_value"), batch.get("metric_value"))
    if (
        runtime_config.hand_root_metric_loss_weight > 0.0
        and not block_hand_gt
        and isinstance(dense_hand_root_xyz, torch.Tensor)
        and isinstance(hand_metric_value, torch.Tensor)
        and joint_targets is not None
        and joint_xyz_mask is not None
        and pred_depth is not None
        and pred_intrinsics is not None
        and pred_camera_pose is not None
        and pred_camera_pose.ndim >= 4
        and pred_camera_pose.shape[-2:] == (4, 4)
    ):
        root_metric_mask = joint_xyz_mask[..., 0]
        pred_scene_scale_for_root = scene_average_distance_from_depth(
            pred_depth.detach(),
            pred_intrinsics.detach(),
            pred_camera_pose.detach(),
        )
        pred_scene_mask_for_root = _scene_scale_sequence_mask(pred_depth, pred_intrinsics, pred_camera_pose)
        pred_scene_valid_for_root = _valid_scene_scale_mask(
            pred_scene_scale_for_root,
            min_value=runtime_config.scene_scale_min_value,
        )
        pred_scene_mask_for_root &= pred_scene_valid_for_root
        _record_scene_scale_guard_metrics(
            metrics,
            f"{stream_name}_hand_root_metric_pred_scene_scale",
            pred_scene_scale_for_root,
            pred_scene_valid_for_root,
        )
        root_metric_mask = root_metric_mask & pred_scene_mask_for_root.view(-1, 1, 1)
        root_metric_mask &= torch.isfinite(joint_targets[..., 0, :]).all(dim=-1)
        if torch.any(root_metric_mask):
            root_scale_factor = (
                hand_metric_value.detach() / pred_scene_scale_for_root.clamp_min(1e-6)
            ).view(-1, 1, 1, 1)
            pred_root_metric = dense_hand_root_xyz * root_scale_factor
            target_root_metric = torch.nan_to_num(joint_targets[..., 0, :])
            root_metric_error = F.smooth_l1_loss(
                pred_root_metric,
                target_root_metric,
                reduction="none",
            ).mean(dim=-1)
            root_metric_mask, root_metric_dropped = _drop_sequence_samples_by_loss(
                root_metric_error,
                root_metric_mask,
                max_sample_loss=runtime_config.scale_normalized_max_sample_loss,
            )
            hand_root_metric_weight = _effective_after_warmup_weight(
                runtime_config.hand_root_metric_loss_weight,
                runtime_config.hand_root_metric_warmup_steps,
                global_step,
            )
            if root_metric_dropped > 0:
                metrics[f"{stream_name}_hand_root_metric_dropped_samples"] = float(root_metric_dropped)
            if hand_root_metric_weight > 0.0 and torch.any(root_metric_mask):
                hand_root_metric_loss = root_metric_error[root_metric_mask].mean()
                total_loss = total_loss + hand_root_metric_weight * hand_root_metric_loss
                _record_loss_metrics(
                    metrics,
                    f"{stream_name}_hand_root_metric_loss",
                    hand_root_metric_loss,
                    hand_root_metric_weight,
                    root_metric_mask,
                )

    hand_abs_loss = _hand_abs_geometry_supervision(
        metrics=metrics,
        runtime_config=runtime_config,
        stream_name=stream_name,
        block_hand_gt=block_hand_gt,
        use_hand_conf=use_hand_conf,
        global_step=global_step,
        zero_loss=total_loss.new_zeros(()),
        metric_value=hand_metric_value,
        dense_joint_xyz=dense_joint_xyz,
        dense_vertex_xyz=dense_vertex_xyz,
        dense_joint_log_conf=outputs.get("dense_joint_log_conf"),
        dense_vertex_log_conf=outputs.get("dense_vertex_log_conf"),
        joint_targets=joint_targets,
        joint_xyz_mask=joint_xyz_mask,
        vertex_xyz_targets=vertex_xyz_targets,
        vertex_xyz_mask=vertex_xyz_mask,
        resized_depth=resized_depth,
        resized_intrinsics=resized_intrinsics,
        resized_depth_valid_mask=resized_depth_valid_mask,
        camera_pose_target_for_scale=camera_pose_target_for_scale,
        valid_depth_frames=valid_depth_frames,
        valid_intrinsics_frames=valid_intrinsics_frames,
        valid_camera_pose_frames=valid_camera_pose_frames,
        pred_depth=pred_depth,
        pred_intrinsics=pred_intrinsics,
        pred_camera_pose=pred_camera_pose,
    )
    total_loss = total_loss + hand_abs_loss

    image_height = int(batch["images"].shape[-2])
    image_width = int(batch["images"].shape[-1])
    reprojection_intrinsics, reprojection_intrinsics_frame_mask = _hand_reprojection_intrinsics(
        batch=batch,
        pred_intrinsics=pred_intrinsics if isinstance(pred_intrinsics, torch.Tensor) else None,
        gt_intrinsics=intrinsics if isinstance(intrinsics, torch.Tensor) else None,
        device=device,
        dtype=dtype,
    )
    hand_2d_weight = _effective_after_warmup_weight(
        runtime_config.hand_2d_reprojection_loss_weight,
        runtime_config.hand_2d_reprojection_warmup_steps,
        global_step,
    )
    hand_root_2d_weight = _effective_after_warmup_weight(
        runtime_config.hand_root_2d_reprojection_loss_weight,
        runtime_config.hand_2d_reprojection_warmup_steps,
        global_step,
    )
    hand_vertex_2d_weight = _effective_after_warmup_weight(
        runtime_config.hand_vertex_2d_reprojection_loss_weight,
        runtime_config.hand_2d_reprojection_warmup_steps,
        global_step,
    )
    hand_joint_2d_bone_weight = _effective_after_warmup_weight(
        runtime_config.hand_joint_2d_bone_length_loss_weight,
        runtime_config.hand_2d_reprojection_warmup_steps,
        global_step,
    )
    hand_joint_2d_bbox_weight = _effective_after_warmup_weight(
        runtime_config.hand_joint_2d_bbox_size_loss_weight,
        runtime_config.hand_2d_reprojection_warmup_steps,
        global_step,
    )
    hand_vertex_2d_bbox_weight = _effective_after_warmup_weight(
        runtime_config.hand_vertex_2d_bbox_size_loss_weight,
        runtime_config.hand_2d_reprojection_warmup_steps,
        global_step,
    )

    if (
        (hand_2d_weight > 0.0 or hand_joint_2d_bone_weight > 0.0 or hand_joint_2d_bbox_weight > 0.0)
        and not block_hand_gt
        and isinstance(dense_joint_xyz, torch.Tensor)
        and "joints_2d_targets" in batch
        and "joints_2d_supervision_mask" in batch
    ):
        joint_2d_targets = _fixed_side_tensor(batch, "joints_2d_targets").to(device=device, dtype=dtype)
        joint_2d_slot_mask = _fixed_side_tensor(batch, "joints_2d_supervision_mask").to(device=device)
        if positive_presence_mask is not None:
            joint_2d_slot_mask = joint_2d_slot_mask & positive_presence_mask
        if reprojection_intrinsics_frame_mask is not None:
            joint_2d_slot_mask = joint_2d_slot_mask & reprojection_intrinsics_frame_mask.unsqueeze(-1)
        joint_2d_point_mask = torch.isfinite(joint_2d_targets).all(dim=-1) & joint_2d_slot_mask.unsqueeze(-1)
        joint_2d_size_targets = joint_2d_targets
        joint_2d_size_point_mask = joint_2d_point_mask
        if "joints_3d_targets" in batch and "raw_joint_supervision_mask" in batch and isinstance(
            reprojection_intrinsics, torch.Tensor
        ):
            joint_3d_targets = _fixed_side_tensor(batch, "joints_3d_targets").to(device=device, dtype=dtype)
            joint_3d_slot_mask = _fixed_side_tensor(batch, "raw_joint_supervision_mask").to(device=device)
            if positive_presence_mask is not None:
                joint_3d_slot_mask = joint_3d_slot_mask & positive_presence_mask
            if reprojection_intrinsics_frame_mask is not None:
                joint_3d_slot_mask = joint_3d_slot_mask & reprojection_intrinsics_frame_mask.unsqueeze(-1)
            projected_joint_targets, projected_joint_mask = _project_hand_targets_to_uv(
                joint_3d_targets,
                reprojection_intrinsics,
                point_mask=joint_3d_slot_mask.unsqueeze(-1).expand_as(joint_2d_point_mask),
            )
            fill_joint_mask = (~joint_2d_point_mask) & projected_joint_mask
            joint_2d_targets = torch.where(fill_joint_mask.unsqueeze(-1), projected_joint_targets, joint_2d_targets)
            joint_2d_point_mask = joint_2d_point_mask | fill_joint_mask
        if (
            isinstance(reprojection_intrinsics, torch.Tensor)
            and (hand_joint_2d_bone_weight > 0.0 or hand_joint_2d_bbox_weight > 0.0)
            and torch.any(joint_2d_size_point_mask)
        ):
            pred_joint_uv, pred_joint_uv_mask = _project_pred_hand_points_to_uv(dense_joint_xyz, reprojection_intrinsics)
            joint_size_point_mask = joint_2d_size_point_mask & pred_joint_uv_mask
            if hand_joint_2d_bone_weight > 0.0 and torch.any(joint_size_point_mask):
                joint_bone_loss, joint_bone_stats = _hand_joint_2d_bone_length_loss(
                    pred_joint_uv,
                    joint_2d_size_targets,
                    joint_size_point_mask,
                    image_height=image_height,
                    image_width=image_width,
                )
                total_loss = total_loss + hand_joint_2d_bone_weight * joint_bone_loss
                _record_loss_metrics(
                    metrics,
                    f"{stream_name}_hand_joint_2d_bone_length_loss",
                    joint_bone_loss,
                    hand_joint_2d_bone_weight,
                    None,
                )
                for stat_name, stat_value in joint_bone_stats.items():
                    metrics[f"{stream_name}_hand_joint_2d_bone_length_{stat_name}"] = stat_value
            if hand_joint_2d_bbox_weight > 0.0 and torch.any(joint_size_point_mask):
                joint_bbox_loss, joint_bbox_stats = _hand_2d_bbox_size_loss(
                    pred_joint_uv,
                    joint_2d_size_targets,
                    joint_size_point_mask,
                    image_height=image_height,
                    image_width=image_width,
                    min_points=4,
                )
                total_loss = total_loss + hand_joint_2d_bbox_weight * joint_bbox_loss
                _record_loss_metrics(
                    metrics,
                    f"{stream_name}_hand_joint_2d_bbox_size_loss",
                    joint_bbox_loss,
                    hand_joint_2d_bbox_weight,
                    None,
                )
                for stat_name, stat_value in joint_bbox_stats.items():
                    metrics[f"{stream_name}_hand_joint_2d_bbox_size_{stat_name}"] = stat_value
        if hand_2d_weight > 0.0 and torch.any(joint_2d_point_mask) and isinstance(reprojection_intrinsics, torch.Tensor):
            hand_2d_reprojection_loss = _hand_2d_reprojection_loss(
                dense_joint_xyz,
                joint_2d_targets,
                joint_2d_point_mask,
                reprojection_intrinsics,
                image_height=image_height,
                image_width=image_width,
                normalized_error_max=runtime_config.hand_2d_reprojection_error_max,
            )
            total_loss = total_loss + hand_2d_weight * hand_2d_reprojection_loss
            _record_loss_metrics(
                metrics,
                f"{stream_name}_hand_2d_reprojection_loss",
                hand_2d_reprojection_loss,
                hand_2d_weight,
                joint_2d_point_mask,
            )

    if (
        hand_root_2d_weight > 0.0
        and not block_hand_gt
        and isinstance(dense_hand_root_xyz, torch.Tensor)
        and "wrist_uv_targets" in batch
    ):
        root_2d_targets = _fixed_side_tensor(batch, "wrist_uv_targets").to(device=device, dtype=dtype)
        root_mask_key = "wrist_uv_supervision_mask" if "wrist_uv_supervision_mask" in batch else "wrist_supervision_mask"
        root_2d_slot_mask = _fixed_side_tensor(batch, root_mask_key).to(device=device)
        if positive_presence_mask is not None:
            root_2d_slot_mask = root_2d_slot_mask & positive_presence_mask
        if reprojection_intrinsics_frame_mask is not None:
            root_2d_slot_mask = root_2d_slot_mask & reprojection_intrinsics_frame_mask.unsqueeze(-1)
        root_2d_point_mask = torch.isfinite(root_2d_targets).all(dim=-1) & root_2d_slot_mask
        if "joints_3d_targets" in batch and "raw_joint_supervision_mask" in batch and isinstance(
            reprojection_intrinsics, torch.Tensor
        ):
            root_3d_targets = _fixed_side_tensor(batch, "joints_3d_targets").to(device=device, dtype=dtype)[..., :1, :]
            root_3d_slot_mask = _fixed_side_tensor(batch, "raw_joint_supervision_mask").to(device=device)
            if positive_presence_mask is not None:
                root_3d_slot_mask = root_3d_slot_mask & positive_presence_mask
            if reprojection_intrinsics_frame_mask is not None:
                root_3d_slot_mask = root_3d_slot_mask & reprojection_intrinsics_frame_mask.unsqueeze(-1)
            projected_root_targets, projected_root_mask = _project_hand_targets_to_uv(
                root_3d_targets,
                reprojection_intrinsics,
                point_mask=root_3d_slot_mask.unsqueeze(-1),
            )
            projected_root_targets = projected_root_targets.squeeze(-2)
            projected_root_mask = projected_root_mask.squeeze(-1)
            fill_root_mask = (~root_2d_point_mask) & projected_root_mask
            root_2d_targets = torch.where(fill_root_mask.unsqueeze(-1), projected_root_targets, root_2d_targets)
            root_2d_point_mask = root_2d_point_mask | fill_root_mask
        if torch.any(root_2d_point_mask) and isinstance(reprojection_intrinsics, torch.Tensor):
            hand_root_2d_reprojection_loss = _hand_2d_reprojection_loss(
                dense_hand_root_xyz.unsqueeze(-2),
                root_2d_targets.unsqueeze(-2),
                root_2d_point_mask,
                reprojection_intrinsics,
                image_height=image_height,
                image_width=image_width,
                normalized_error_max=runtime_config.hand_2d_reprojection_error_max,
            )
            total_loss = total_loss + hand_root_2d_weight * hand_root_2d_reprojection_loss
            _record_loss_metrics(
                metrics,
                f"{stream_name}_hand_root_2d_reprojection_loss",
                hand_root_2d_reprojection_loss,
                hand_root_2d_weight,
                root_2d_point_mask,
            )

    if (
        (hand_vertex_2d_weight > 0.0 or hand_vertex_2d_bbox_weight > 0.0)
        and not block_hand_gt
        and isinstance(dense_vertex_xyz, torch.Tensor)
        and "vertex_2d_targets" in batch
        and "vertex_xyz_supervision_mask" in batch
    ):
        vertex_2d_targets = _fixed_side_tensor(batch, "vertex_2d_targets").to(device=device, dtype=dtype)
        vertex_2d_mask = _fixed_side_tensor(batch, "vertex_xyz_supervision_mask").to(device=device)
        vertex_xyz_targets = _fixed_side_tensor(batch, "vertex_xyz_targets").to(device=device, dtype=dtype)
        vertex_count = min(dense_vertex_xyz.shape[-2], vertex_2d_targets.shape[-2], vertex_2d_mask.shape[-1])
        vertex_2d_targets = vertex_2d_targets[..., :vertex_count, :]
        vertex_2d_mask = vertex_2d_mask[..., :vertex_count]
        vertex_xyz_targets = vertex_xyz_targets[..., :vertex_count, :]
        dense_vertex_for_2d = dense_vertex_xyz[..., :vertex_count, :]
        if positive_presence_mask is not None:
            vertex_2d_mask = vertex_2d_mask & positive_presence_mask.unsqueeze(-1)
        if reprojection_intrinsics_frame_mask is not None:
            vertex_2d_mask = vertex_2d_mask & reprojection_intrinsics_frame_mask.unsqueeze(-1).unsqueeze(-1)
        vertex_2d_point_mask = torch.isfinite(vertex_2d_targets).all(dim=-1) & vertex_2d_mask
        vertex_2d_size_targets = vertex_2d_targets
        vertex_2d_size_point_mask = vertex_2d_point_mask
        vertex_visibility_targets_for_2d = vertex_visibility_targets[..., :vertex_count].to(dtype=torch.bool)
        vertex_visibility_mask_for_2d = vertex_visibility_mask[..., :vertex_count].to(dtype=torch.bool)
        vertex_2d_size_point_mask = vertex_2d_size_point_mask & (
            ~vertex_visibility_mask_for_2d | vertex_visibility_targets_for_2d
        )
        if isinstance(reprojection_intrinsics, torch.Tensor):
            projected_vertex_targets, projected_vertex_mask = _project_hand_targets_to_uv(
                vertex_xyz_targets,
                reprojection_intrinsics,
                point_mask=vertex_2d_mask,
            )
            fill_vertex_mask = (~vertex_2d_point_mask) & projected_vertex_mask
            vertex_2d_targets = torch.where(fill_vertex_mask.unsqueeze(-1), projected_vertex_targets, vertex_2d_targets)
            vertex_2d_point_mask = vertex_2d_point_mask | fill_vertex_mask
        if (
            isinstance(reprojection_intrinsics, torch.Tensor)
            and hand_vertex_2d_bbox_weight > 0.0
            and torch.any(vertex_2d_size_point_mask)
        ):
            pred_vertex_uv, pred_vertex_uv_mask = _project_pred_hand_points_to_uv(dense_vertex_for_2d, reprojection_intrinsics)
            vertex_size_point_mask = vertex_2d_size_point_mask & pred_vertex_uv_mask
            if torch.any(vertex_size_point_mask):
                vertex_bbox_loss, vertex_bbox_stats = _hand_2d_bbox_size_loss(
                    pred_vertex_uv,
                    vertex_2d_size_targets,
                    vertex_size_point_mask,
                    image_height=image_height,
                    image_width=image_width,
                    min_points=8,
                )
                total_loss = total_loss + hand_vertex_2d_bbox_weight * vertex_bbox_loss
                _record_loss_metrics(
                    metrics,
                    f"{stream_name}_hand_vertex_2d_bbox_size_loss",
                    vertex_bbox_loss,
                    hand_vertex_2d_bbox_weight,
                    None,
                )
                for stat_name, stat_value in vertex_bbox_stats.items():
                    metrics[f"{stream_name}_hand_vertex_2d_bbox_size_{stat_name}"] = stat_value
        if hand_vertex_2d_weight > 0.0 and torch.any(vertex_2d_point_mask) and isinstance(reprojection_intrinsics, torch.Tensor):
            hand_vertex_2d_reprojection_loss = _hand_2d_reprojection_loss(
                dense_vertex_for_2d,
                vertex_2d_targets,
                vertex_2d_point_mask,
                reprojection_intrinsics,
                image_height=image_height,
                image_width=image_width,
                normalized_error_max=runtime_config.hand_2d_reprojection_error_max,
            )
            total_loss = total_loss + hand_vertex_2d_weight * hand_vertex_2d_reprojection_loss
            _record_loss_metrics(
                metrics,
                f"{stream_name}_hand_vertex_2d_reprojection_loss",
                hand_vertex_2d_reprojection_loss,
                hand_vertex_2d_weight,
                vertex_2d_point_mask,
            )

    if (
        runtime_config.vertex_depth_pred_self_loss_weight > 0.0
        and pred_depth is not None
        and pred_intrinsics is not None
    ):
        pred_self_visibility = (
            depth_consistency_vertex_visibility_targets
            * depth_consistency_vertex_visibility_mask.to(dtype=dtype)
        )
        if three_r_loss_mask is not None:
            pred_self_visibility = pred_self_visibility * three_r_loss_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=dtype)
        if torch.any(pred_self_visibility >= 0.5):
            pred_self_loss = vertex_depth_consistency_loss(
                vertex_xyz=dense_vertex_xyz,
                vertex_visibility=pred_self_visibility,
                intrinsics=pred_intrinsics.detach(),
                depth_map=pred_depth.detach(),
            )
            vertex_depth_pred_self_weight = _effective_after_warmup_weight(
                runtime_config.vertex_depth_pred_self_loss_weight,
                runtime_config.vertex_depth_pred_self_warmup_steps,
                global_step,
            )
            total_loss = total_loss + vertex_depth_pred_self_weight * pred_self_loss
            metrics[f"{stream_name}_vertex_depth_pred_self_loss"] = float(pred_self_loss.item())

    log_metric_value = outputs.get("log_metric_value")
    if (
        isinstance(log_metric_value, torch.Tensor)
        and pred_depth is not None
        and pred_intrinsics is not None
        and pred_camera_pose is not None
        and pred_camera_pose.ndim >= 4
        and pred_camera_pose.shape[-2:] == (4, 4)
    ):
        pred_scene_scale = scene_average_distance_from_depth(
            pred_depth.detach(),
            pred_intrinsics.detach(),
            pred_camera_pose.detach(),
        )
        pred_scene_mask = _scene_scale_sequence_mask(pred_depth, pred_intrinsics, pred_camera_pose)
        pred_scene_valid = _valid_scene_scale_mask(
            pred_scene_scale,
            min_value=runtime_config.scene_scale_min_value,
        )
        pred_scene_mask &= pred_scene_valid
        _record_scene_scale_guard_metrics(
            metrics,
            f"{stream_name}_metric_hand_pred_scene_scale",
            pred_scene_scale,
            pred_scene_valid,
        )
        image_height = int(batch["images"].shape[-2])
        image_width = int(batch["images"].shape[-1])
        if (
            not block_hand_gt
            and joint_targets is not None
            and joint_xyz_mask is not None
            and "joints_2d_targets" in batch
        ):
            metrics["metric_hand_joint_anchor_count"] = 0.0
            metrics["metric_hand_joint_valid_count"] = 0.0
            joint_anchor_mask = _joint_depth_anchor_mask(batch, joint_xyz_mask)
            if joint_anchor_mask is not None:
                metrics["metric_hand_joint_anchor_count"] = float(
                    joint_anchor_mask.to(dtype=torch.float32).sum().detach().item()
                )
            if joint_anchor_mask is not None and torch.any(joint_anchor_mask):
                joint_scale_target, joint_metric_mask = depth_anchored_hand_scale_target(
                    pred_depth.detach(),
                    pred_intrinsics.detach(),
                    pred_scene_scale.detach(),
                    _fixed_side_tensor(batch, "joints_2d_targets").to(device=device, dtype=dtype),
                    joint_targets,
                    joint_anchor_mask,
                    image_height=image_height,
                    image_width=image_width,
                    min_value=runtime_config.metric_scale_min_value,
                    max_value=runtime_config.metric_scale_max_value,
                    min_scene_scale=runtime_config.scene_scale_min_value,
                )
                joint_metric_mask = pred_scene_mask & joint_metric_mask
                metrics["metric_hand_joint_valid_count"] = float(
                    joint_metric_mask.to(dtype=torch.float32).sum().detach().item()
                )
                if torch.any(joint_metric_mask):
                    metric_hand_joint_loss = log_scale_loss(
                        log_metric_value,
                        joint_scale_target,
                        joint_metric_mask,
                    )
                    metric_hand_joint_weight = _effective_after_warmup_weight(
                        runtime_config.metric_hand_joint_loss_weight,
                        runtime_config.metric_hand_scale_warmup_steps,
                        global_step,
                    )
                    total_loss = (
                        total_loss
                        + runtime_config.metric_scale_loss_weight
                        * metric_hand_joint_weight
                        * metric_hand_joint_loss
                    )
                    _record_loss_metrics(
                        metrics,
                        "metric_hand_joint_loss",
                        metric_hand_joint_loss,
                        runtime_config.metric_scale_loss_weight * metric_hand_joint_weight,
                        joint_metric_mask,
                    )
        if not block_hand_gt and "vertex_2d_targets" in batch and torch.any(vertex_xyz_mask):
            metrics["metric_hand_vertex_anchor_count"] = 0.0
            metrics["metric_hand_vertex_valid_count"] = 0.0
            vertex_2d_targets = _fixed_side_tensor(batch, "vertex_2d_targets").to(device=device, dtype=dtype)
            vertex_count = min(vertex_xyz_targets.shape[3], vertex_2d_targets.shape[3])
            vertex_anchor_mask = (
                vertex_xyz_mask[..., :vertex_count]
                & vertex_visibility_targets[..., :vertex_count].to(dtype=torch.bool)
                & vertex_visibility_mask[..., :vertex_count].to(dtype=torch.bool)
            )
            metrics["metric_hand_vertex_anchor_count"] = float(
                vertex_anchor_mask.to(dtype=torch.float32).sum().detach().item()
            )
            if torch.any(vertex_anchor_mask):
                vertex_scale_target, vertex_metric_mask = depth_anchored_hand_scale_target(
                    pred_depth.detach(),
                    pred_intrinsics.detach(),
                    pred_scene_scale.detach(),
                    vertex_2d_targets[..., :vertex_count, :],
                    vertex_xyz_targets[..., :vertex_count, :],
                    vertex_anchor_mask,
                    image_height=image_height,
                    image_width=image_width,
                    min_value=runtime_config.metric_scale_min_value,
                    max_value=runtime_config.metric_scale_max_value,
                    min_scene_scale=runtime_config.scene_scale_min_value,
                )
                vertex_metric_mask = pred_scene_mask & vertex_metric_mask
                metrics["metric_hand_vertex_valid_count"] = float(
                    vertex_metric_mask.to(dtype=torch.float32).sum().detach().item()
                )
                if torch.any(vertex_metric_mask):
                    metric_hand_vertex_loss = log_scale_loss(
                        log_metric_value,
                        vertex_scale_target,
                        vertex_metric_mask,
                    )
                    metric_hand_vertex_weight = _effective_after_warmup_weight(
                        runtime_config.metric_hand_vertex_loss_weight,
                        runtime_config.metric_hand_scale_warmup_steps,
                        global_step,
                    )
                    total_loss = (
                        total_loss
                        + runtime_config.metric_scale_loss_weight
                        * metric_hand_vertex_weight
                        * metric_hand_vertex_loss
                    )
                    _record_loss_metrics(
                        metrics,
                        "metric_hand_vertex_loss",
                        metric_hand_vertex_loss,
                        runtime_config.metric_scale_loss_weight * metric_hand_vertex_weight,
                        vertex_metric_mask,
                    )

    if (
        isinstance(log_metric_value, torch.Tensor)
        and resized_depth is not None
        and resized_intrinsics is not None
        and valid_depth_frames is not None
        and valid_intrinsics_frames is not None
        and valid_camera_pose_frames is not None
        and camera_pose_target_for_scale is not None
    ):
        metric_3r_frames = valid_depth_frames & valid_intrinsics_frames & valid_camera_pose_frames
        if torch.any(metric_3r_frames):
            metric_3r_valid_depth = (
                resized_depth_valid_mask
                if resized_depth_valid_mask is not None
                else torch.isfinite(resized_depth) & (resized_depth > 0)
            )
            metric_3r_depth_mask = metric_3r_valid_depth & metric_3r_frames.unsqueeze(-1).unsqueeze(-1)
            gt_scene_scale = scene_average_distance_from_depth(
                resized_depth.detach(),
                resized_intrinsics.detach(),
                camera_pose_target_for_scale.detach(),
                metric_3r_depth_mask,
            )
            metric_3r_sequence_mask = _scene_scale_sequence_mask(
                resized_depth,
                resized_intrinsics,
                camera_pose_target_for_scale,
                metric_3r_depth_mask,
            )
            metric_3r_scene_valid = _valid_scene_scale_mask(
                gt_scene_scale,
                min_value=runtime_config.scene_scale_min_value,
            )
            metric_3r_sequence_mask &= metric_3r_scene_valid
            _record_scene_scale_guard_metrics(
                metrics,
                f"{stream_name}_metric_scale_gt_scene_scale",
                gt_scene_scale,
                metric_3r_scene_valid,
            )
            metric_3r_loss = log_scale_loss(log_metric_value, gt_scene_scale, metric_3r_sequence_mask)
            total_loss = total_loss + runtime_config.metric_scale_loss_weight * metric_3r_loss
            metrics[f"{stream_name}_metric_scale_loss"] = float(metric_3r_loss.item())

    if (
        resized_depth is not None
        and resized_intrinsics is not None
        and valid_depth_frames is not None
        and valid_intrinsics_frames is not None
    ):
        valid_frames = valid_depth_frames & valid_intrinsics_frames
        if torch.any(valid_frames):
            gt_vertex_visibility = depth_consistency_vertex_visibility_targets * depth_consistency_vertex_visibility_mask.to(dtype=dtype)
            visibility = gt_vertex_visibility * valid_frames.unsqueeze(-1).unsqueeze(-1)
            depth_consistency_loss = dense_vertex_xyz.sum() * 0.0
            if (
                pred_depth is not None
                and pred_intrinsics is not None
                and pred_camera_pose is not None
                and pred_camera_pose.ndim >= 4
                and pred_camera_pose.shape[-2:] == (4, 4)
                and camera_pose_target_for_scale is not None
                and valid_camera_pose_frames is not None
            ):
                scale_frames = valid_frames & valid_camera_pose_frames
                if torch.any(scale_frames):
                    valid_depth_for_scale = (
                        torch.ones_like(resized_depth, dtype=torch.bool)
                        if resized_depth_valid_mask is None
                        else resized_depth_valid_mask
                    )
                    gt_depth_scale_mask = valid_depth_for_scale & scale_frames.unsqueeze(-1).unsqueeze(-1)
                    gt_scene_scale = scene_average_distance_from_depth(
                        resized_depth.detach(),
                        resized_intrinsics.detach(),
                        camera_pose_target_for_scale.detach(),
                        gt_depth_scale_mask,
                    )
                    pred_scene_scale = scene_average_distance_from_depth(
                        pred_depth.detach(),
                        pred_intrinsics.detach(),
                        pred_camera_pose.detach(),
                    )
                    gt_scene_valid = _valid_scene_scale_mask(
                        gt_scene_scale,
                        min_value=runtime_config.scene_scale_min_value,
                    )
                    pred_scene_valid = _valid_scene_scale_mask(
                        pred_scene_scale,
                        min_value=runtime_config.scene_scale_min_value,
                    )
                    _record_scene_scale_guard_metrics(
                        metrics,
                        f"{stream_name}_depth_consistency_gt_scene_scale",
                        gt_scene_scale,
                        gt_scene_valid,
                    )
                    _record_scene_scale_guard_metrics(
                        metrics,
                        f"{stream_name}_depth_consistency_pred_scene_scale",
                        pred_scene_scale,
                        pred_scene_valid,
                    )
                    scale_sequence_mask = (
                        _scene_scale_sequence_mask(
                            resized_depth,
                            resized_intrinsics,
                            camera_pose_target_for_scale,
                            gt_depth_scale_mask,
                        )
                        & _scene_scale_sequence_mask(pred_depth, pred_intrinsics, pred_camera_pose)
                        & gt_scene_valid
                        & pred_scene_valid
                    )
                    consistency_visibility = visibility * scale_sequence_mask.view(-1, 1, 1, 1).to(dtype=dtype)
                    if torch.any(consistency_visibility >= 0.5):
                        depth_for_consistency = resized_depth / gt_scene_scale.view(-1, 1, 1, 1).clamp_min(1e-6)
                        vertex_for_consistency = dense_vertex_xyz / pred_scene_scale.view(-1, 1, 1, 1, 1).clamp_min(1e-6)
                        depth_consistency_loss, depth_consistency_dropped = vertex_depth_consistency_loss(
                            vertex_xyz=vertex_for_consistency,
                            vertex_visibility=consistency_visibility,
                            intrinsics=resized_intrinsics,
                            depth_map=depth_for_consistency,
                            depth_valid_mask=resized_depth_valid_mask,
                            max_sample_loss=runtime_config.scale_normalized_max_sample_loss,
                            return_dropped=True,
                        )
                        if depth_consistency_dropped > 0:
                            metrics[f"{stream_name}_depth_consistency_dropped_samples"] = float(depth_consistency_dropped)
            total_loss = total_loss + runtime_config.depth_consistency_loss_weight * depth_consistency_loss
            metrics[f"{stream_name}_depth_consistency_loss"] = float(depth_consistency_loss.item())

    teacher_features = outputs.get("hand_feature_teacher_features")
    if teacher_targets is not None and teacher_features is not None:
        teacher_mask = teacher_targets["teacher_mask"].to(device=device)
        if torch.any(teacher_mask):
            teacher_features_for_loss = teacher_features[teacher_mask].float()
            teacher_target_features = teacher_targets["teacher_features"].to(device=device, dtype=torch.float32)[teacher_mask]
            distill_loss = F.mse_loss(
                teacher_features_for_loss,
                teacher_target_features,
            )
            weight = float(teacher_targets["weight"].item())
            total_loss = total_loss + weight * distill_loss
            metrics["wilor_distill_loss"] = float(distill_loss.item())

    metrics["total_loss"] = float(total_loss.item())
    return total_loss, metrics
