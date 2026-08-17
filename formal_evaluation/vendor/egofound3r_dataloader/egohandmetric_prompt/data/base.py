from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, get_worker_info

from egohandmetric_prompt.data.media import load_media_ref
from egohandmetric_prompt.data.prefetch import (
    ContainerPrefetchHint,
    PrefetchIndex,
    get_container_prefetch_manager,
)
from egohandmetric_prompt.data.schema import FrameRecord, HandAnnotation, MediaRef


_DEPTH_MILLIMETER_MODES = {"png_uint16", "h5_uint16_mm", "tar_depth_png", "hoi4d_depth_video"}
_DEPTH_TACO_VIDEO_MODES = {"taco_depth_video"}
_STICKY_SEQUENCE_DATASET_NAMES: set[str] = set()
_CONTAINER_LOCAL_SAMPLING_START_ORDERS = {"sequential", "random_cursor"}


@dataclass(slots=True)
class _LocalSequenceState:
    dataset_index: int
    sequence_index: int
    remaining_batches: int


@dataclass(frozen=True, slots=True)
class _AffinePermutation:
    """Constant-memory bijection over ``range(count)`` for finite epochs."""

    count: int
    offset: int
    step: int

    def at(self, position: int) -> int:
        if position < 0 or position >= self.count:
            raise IndexError(position)
        return (self.offset + self.step * int(position)) % self.count


@dataclass(slots=True)
class _CompactEpochSampleSource:
    """A weighted source whose shuffled candidates are decoded only on demand."""

    base_count: int
    full_permutations: tuple[_AffinePermutation, ...]
    fractional_permutation: _AffinePermutation | None
    fractional_count: int
    resolve: Any

    @property
    def sample_count(self) -> int:
        return self.base_count * len(self.full_permutations) + self.fractional_count

    def sample_at(self, position: int) -> Any:
        if position < 0 or position >= self.sample_count:
            raise IndexError(position)
        full_count = self.base_count * len(self.full_permutations)
        if position < full_count:
            permutation_index, local_position = divmod(position, self.base_count)
            return self.resolve(self.full_permutations[permutation_index].at(local_position))
        assert self.fractional_permutation is not None
        return self.resolve(self.fractional_permutation.at(position - full_count))


@dataclass(slots=True)
class _CompactEpochSampleGroup:
    """A compactly shuffled union of one or more weighted candidate sources."""

    sources: tuple[_CompactEpochSampleSource, ...]
    source_offsets: tuple[int, ...]
    permutation: _AffinePermutation

    @property
    def sample_count(self) -> int:
        return self.permutation.count

    def sample_at(self, position: int) -> Any:
        source_position = self.permutation.at(position)
        source_index = bisect_right(self.source_offsets, source_position) - 1
        return self.sources[source_index].sample_at(source_position - self.source_offsets[source_index])


@dataclass(slots=True)
class _CompactEpochBatchSource:
    """Expose one independently shuffled sample group as one-batch units."""

    group: _CompactEpochSampleGroup
    batch_size: int

    @property
    def unit_count(self) -> int:
        return (self.group.sample_count + self.batch_size - 1) // self.batch_size

    @property
    def batch_count(self) -> int:
        return self.unit_count

    def iter_unit_batches(self, unit_index: int):
        if unit_index < 0 or unit_index >= self.unit_count:
            raise IndexError(unit_index)
        start = int(unit_index) * self.batch_size
        end = min(start + self.batch_size, self.group.sample_count)
        yield [self.group.sample_at(sample_index) for sample_index in range(start, end)]


@dataclass(slots=True)
class _CompactEpochLocalBlockBatchSource:
    """Emit consecutive batches from one sequence without storing its anchors."""

    sources: tuple[_CompactEpochSampleSource, ...]
    source_block_offsets: tuple[int, ...]
    block_batches: int
    batch_size: int
    permutation: _AffinePermutation

    @property
    def unit_count(self) -> int:
        return self.permutation.count

    @property
    def batch_count(self) -> int:
        return sum((source.sample_count + self.batch_size - 1) // self.batch_size for source in self.sources)

    def iter_unit_batches(self, unit_index: int):
        if unit_index < 0 or unit_index >= self.unit_count:
            raise IndexError(unit_index)
        source_block_index = self.permutation.at(unit_index)
        source_index = bisect_right(self.source_block_offsets, source_block_index) - 1
        source = self.sources[source_index]
        local_block_index = source_block_index - self.source_block_offsets[source_index]
        block_sample_start = local_block_index * self.block_batches * self.batch_size
        block_sample_end = min(
            block_sample_start + self.block_batches * self.batch_size,
            source.sample_count,
        )
        for batch_start in range(block_sample_start, block_sample_end, self.batch_size):
            batch_end = min(batch_start + self.batch_size, block_sample_end)
            yield [source.sample_at(sample_index) for sample_index in range(batch_start, batch_end)]


@dataclass(slots=True)
class _CompactEpochBatchPlan:
    """Randomized batch units; local units retain a sequence for a bounded span."""

    groups: tuple[_CompactEpochSampleGroup, ...]
    sources: tuple[Any, ...]
    source_unit_offsets: tuple[int, ...]
    unit_permutation: _AffinePermutation | None
    batch_count: int
    sample_count: int

    def iter_batches(self):
        if self.unit_permutation is None:
            return
        for ordered_unit_index in range(self.unit_permutation.count):
            source_unit_index = self.unit_permutation.at(ordered_unit_index)
            source_index = bisect_right(self.source_unit_offsets, source_unit_index) - 1
            source = self.sources[source_index]
            local_unit_index = source_unit_index - self.source_unit_offsets[source_index]
            yield from source.iter_unit_batches(local_unit_index)


def _normalize_dataset_name_set(dataset_names: Sequence[str] | None) -> set[str]:
    if dataset_names is None:
        return set()
    return {str(name) for name in dataset_names}


def _random_length_dataset_name(dataset: Dataset) -> str:
    frame_dataset = getattr(dataset, "frame_dataset", None)
    return str(getattr(frame_dataset, "dataset_name", ""))


def _validate_container_local_prefetch_margin(block_batches: int, margin_batches: int) -> None:
    if int(margin_batches) <= 0:
        raise ValueError("container_local_prefetch_margin_batches 必须为正整数。")
    if int(margin_batches) >= int(block_batches):
        raise ValueError("container_local_prefetch_margin_batches 必须小于 container_local_sampling_block_batches。")


def _wrap_batch_with_prefetch_hint(batch: list[Any], hint: ContainerPrefetchHint | None) -> list[Any]:
    if hint is None:
        return batch
    return [PrefetchIndex(index=item, hint=hint) for item in batch]


def _split_prefetch_index(index: Any) -> tuple[Any, ContainerPrefetchHint | None]:
    if isinstance(index, PrefetchIndex):
        return index.index, index.hint
    return index, None


@dataclass(frozen=True, slots=True)
class MissingSample:
    index_key: Any
    error_path: str
    attempts: int


def _jsonable_index(index: Any) -> Any:
    if isinstance(index, PrefetchIndex):
        return {
            "index": _jsonable_index(index.index),
            "prefetch_hint": _jsonable_index(index.hint.key()),
        }
    if isinstance(index, tuple):
        return [_jsonable_index(item) for item in index]
    if isinstance(index, list):
        return [_jsonable_index(item) for item in index]
    if isinstance(index, np.integer):
        return int(index)
    return index


def _index_key(index: Any) -> str:
    return json.dumps(_jsonable_index(index), sort_keys=True, separators=(",", ":"))


def _path_fingerprint(path: Any) -> str:
    return str(Path(path).resolve(strict=False))


def _dataset_fingerprint_payload(dataset: Dataset) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(dataset).__name__,
        "length": len(dataset),
    }
    for name in (
        "dataset_name",
        "base_dataset_name",
        "split",
        "dataset_names",
        "num_frames",
        "min_num_frames",
        "max_num_frames",
        "window_stride",
        "drop_last",
    ):
        if hasattr(dataset, name):
            payload[name] = _jsonable_index(getattr(dataset, name))
    for name in ("root", "manifest_path"):
        if hasattr(dataset, name):
            payload[name] = _path_fingerprint(getattr(dataset, name))
    if isinstance(dataset, ConcatDataset):
        payload["children"] = [_dataset_fingerprint_payload(child) for child in dataset.datasets]
    elif isinstance(dataset, CombinedFrameDataset):
        payload["children"] = [_dataset_fingerprint_payload(child) for child in dataset.datasets]
    elif isinstance(dataset, TemporalChunkDataset):
        payload["frame_dataset"] = _dataset_fingerprint_payload(dataset.frame_dataset)
    elif isinstance(dataset, RandomLengthTemporalChunkDataset):
        payload["frame_dataset"] = _dataset_fingerprint_payload(dataset.frame_dataset)
    elif isinstance(dataset, MultiSourceRandomLengthTemporalChunkDataset):
        payload["children"] = [_dataset_fingerprint_payload(child) for child in dataset.datasets]
    return payload


def _dataset_fingerprint(dataset: Dataset) -> str:
    payload = _dataset_fingerprint_payload(dataset)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class MissingSampleRetryDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        replacement_attempts: int,
        stream_name: str,
        registry_dir: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        if replacement_attempts < 0:
            raise ValueError("replacement_attempts 必须 >= 0。")
        self.dataset = dataset
        self.replacement_attempts = int(replacement_attempts)
        self.stream_name = stream_name
        self.registry_dir = Path(registry_dir) if registry_dir is not None else None
        self.seed = int(seed)
        self.fingerprint = _dataset_fingerprint(dataset)
        self._bad_keys: set[str] = set()
        self._worker_generators: dict[int, torch.Generator] = {}
        self._load_registry()

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_registry(self) -> None:
        if self.registry_dir is None or not self.registry_dir.exists():
            return
        for path in self.registry_dir.glob("*.jsonl"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if row.get("stream") != self.stream_name:
                            continue
                        if row.get("dataset_fingerprint") != self.fingerprint:
                            continue
                        key = row.get("index_key_string")
                        if isinstance(key, str):
                            self._bad_keys.add(key)
            except (OSError, json.JSONDecodeError):
                continue

    def _worker_id(self) -> int:
        worker = get_worker_info()
        return 0 if worker is None else int(worker.id)

    def _rank(self) -> int:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    def _generator(self) -> torch.Generator:
        worker_id = self._worker_id()
        cached = self._worker_generators.get(worker_id)
        if cached is not None:
            return cached
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1009 * worker_id + 9176 * self._rank())
        self._worker_generators[worker_id] = generator
        return generator

    def _registry_path(self) -> Path | None:
        if self.registry_dir is None:
            return None
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        return self.registry_dir / f"{self.stream_name}_rank{self._rank()}_worker{self._worker_id()}.jsonl"

    def _mark_missing(self, index: Any, exc: FileNotFoundError, attempts: int) -> MissingSample:
        key = _index_key(index)
        error_path = str(exc.args[0]) if exc.args else repr(exc)
        if key not in self._bad_keys:
            self._bad_keys.add(key)
            path = self._registry_path()
            if path is not None:
                row = {
                    "stream": self.stream_name,
                    "dataset_fingerprint": self.fingerprint,
                    "index_key": _jsonable_index(index),
                    "index_key_string": key,
                    "error_path": error_path,
                    "rank": self._rank(),
                    "worker": self._worker_id(),
                    "time": time.time(),
                }
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return MissingSample(index_key=_jsonable_index(index), error_path=error_path, attempts=attempts)

    def _sample_int_replacement(self, index: int) -> int:
        length = len(self.dataset)
        if length <= 0:
            raise IndexError("无法从空数据集中替换缺失样本。")
        start = 0
        end = length
        if isinstance(self.dataset, ConcatDataset):
            child_index = bisect_right(self.dataset.cumulative_sizes, int(index))
            start = 0 if child_index == 0 else int(self.dataset.cumulative_sizes[child_index - 1])
            end = int(self.dataset.cumulative_sizes[child_index])
        elif isinstance(self.dataset, CombinedFrameDataset):
            child_index = bisect_right(self.dataset._offsets, int(index)) - 1
            start = int(self.dataset._offsets[child_index])
            end = start + len(self.dataset.datasets[child_index])
        span = max(end - start, 1)
        offset = int(torch.randint(span, (1,), generator=self._generator()).item())
        return start + offset

    def _sample_replacement(self, original_index: Any, *, replacement_attempt: int = 1) -> Any:
        original_index, _ = _split_prefetch_index(original_index)
        if isinstance(original_index, tuple) and len(original_index) == 7 and hasattr(self.dataset, "datasets"):
            dataset_index, _, _, num_frames, frame_stride, target_height, target_width = original_index
            replacement_dataset_index, sequence_index, start, _, _, _, _ = self._sample_multisource_lazy_temporal_replacement(
                self.dataset,
                original_dataset_index=int(dataset_index),
                prefer_original_dataset=replacement_attempt <= 1,
                num_frames=int(num_frames),
                frame_stride=int(frame_stride),
                target_height=int(target_height),
                target_width=int(target_width),
            )
            return (
                replacement_dataset_index,
                sequence_index,
                start,
                int(num_frames),
                int(frame_stride),
                int(target_height),
                int(target_width),
            )
        if isinstance(original_index, tuple) and len(original_index) == 6 and hasattr(self.dataset, "valid_start_count"):
            _, _, num_frames, frame_stride, target_height, target_width = original_index
            return self._sample_lazy_temporal_replacement(
                self.dataset,
                num_frames=int(num_frames),
                frame_stride=int(frame_stride),
                target_height=int(target_height),
                target_width=int(target_width),
            )
        if isinstance(original_index, tuple) and len(original_index) == 3 and hasattr(self.dataset, "valid_anchor_count"):
            dataset_index, _, num_frames = original_index
            count = int(self.dataset.valid_anchor_count(int(dataset_index), int(num_frames)))
            if count <= 0:
                raise IndexError(original_index)
            anchor_index = int(torch.randint(count, (1,), generator=self._generator()).item())
            return (int(dataset_index), anchor_index, int(num_frames))
        if isinstance(original_index, tuple) and len(original_index) == 2 and hasattr(self.dataset, "valid_anchor_count"):
            _, num_frames = original_index
            count = int(self.dataset.valid_anchor_count(int(num_frames)))
            if count <= 0:
                raise IndexError(original_index)
            anchor_index = int(torch.randint(count, (1,), generator=self._generator()).item())
            return (anchor_index, int(num_frames))
        return self._sample_int_replacement(int(original_index))

    def _sample_multisource_lazy_temporal_replacement(
        self,
        dataset: Any,
        *,
        original_dataset_index: int,
        prefer_original_dataset: bool,
        num_frames: int,
        frame_stride: int,
        target_height: int,
        target_width: int,
    ) -> tuple[int, int, int, int, int, int, int]:
        candidates: list[tuple[int, int]] = []
        for dataset_index, child_dataset in enumerate(dataset.datasets):
            valid_count = sum(
                int(
                    child_dataset.valid_start_count(
                        sequence_index,
                        num_frames=int(num_frames),
                        frame_stride=int(frame_stride),
                    )
                )
                for sequence_index in range(int(child_dataset.sequence_count))
            )
            if valid_count > 0:
                candidates.append((dataset_index, valid_count))
        if not candidates:
            raise IndexError((num_frames, frame_stride, target_height, target_width))
        original_candidates = [candidate for candidate in candidates if candidate[0] == int(original_dataset_index)]
        alternatives = [candidate for candidate in candidates if candidate[0] != int(original_dataset_index)]
        if prefer_original_dataset and original_candidates:
            replacement_pool = original_candidates
        else:
            replacement_pool = alternatives if alternatives else candidates
        weights = torch.tensor([valid_count for _, valid_count in replacement_pool], dtype=torch.float64)
        position = int(torch.multinomial(weights, num_samples=1, replacement=True, generator=self._generator()).item())
        dataset_index = int(replacement_pool[position][0])
        sequence_index, start, _, _, _, _ = self._sample_lazy_temporal_replacement(
            dataset.datasets[dataset_index],
            num_frames=int(num_frames),
            frame_stride=int(frame_stride),
            target_height=int(target_height),
            target_width=int(target_width),
        )
        return (
            dataset_index,
            sequence_index,
            start,
            int(num_frames),
            int(frame_stride),
            int(target_height),
            int(target_width),
        )

    def _sample_lazy_temporal_replacement(
        self,
        dataset: Any,
        *,
        num_frames: int,
        frame_stride: int,
        target_height: int,
        target_width: int,
    ) -> tuple[int, int, int, int, int, int]:
        active_sequences: list[int] = []
        valid_counts: list[int] = []
        for sequence_index in range(int(dataset.sequence_count)):
            valid_count = int(
                dataset.valid_start_count(
                    sequence_index,
                    num_frames=int(num_frames),
                    frame_stride=int(frame_stride),
                )
            )
            if valid_count <= 0:
                continue
            active_sequences.append(sequence_index)
            valid_counts.append(valid_count)
        if not active_sequences:
            raise IndexError((num_frames, frame_stride, target_height, target_width))
        weights = torch.tensor(valid_counts, dtype=torch.float64)
        sequence_position = int(torch.multinomial(weights, num_samples=1, replacement=True, generator=self._generator()).item())
        sequence_index = active_sequences[sequence_position]
        start_slot = int(torch.randint(valid_counts[sequence_position], (1,), generator=self._generator()).item())
        start = start_slot * int(dataset.window_stride)
        return (
            int(sequence_index),
            int(start),
            int(num_frames),
            int(frame_stride),
            int(target_height),
            int(target_width),
        )

    def __getitem__(self, index: Any) -> Any:
        candidate = index
        attempts = 0
        last_missing = MissingSample(index_key=_jsonable_index(index), error_path="", attempts=0)
        while True:
            if _index_key(candidate) not in self._bad_keys:
                try:
                    return self.dataset[candidate]
                except FileNotFoundError as exc:
                    last_missing = self._mark_missing(candidate, exc, attempts)
            if attempts >= self.replacement_attempts:
                return MissingSample(
                    index_key=_jsonable_index(index),
                    error_path=last_missing.error_path,
                    attempts=attempts,
                )
            attempts += 1
            candidate = self._sample_replacement(index, replacement_attempt=attempts)


class MissingSampleCollator:
    def __init__(self, collator) -> None:
        self.collator = collator

    def __call__(self, samples: list[Any]) -> Any:
        missing = [sample for sample in samples if isinstance(sample, MissingSample)]
        kept = [sample for sample in samples if not isinstance(sample, MissingSample)]
        if not missing:
            return self.collator(samples)
        missing_payloads = [
            {
                "index_key": sample.index_key,
                "error_path": sample.error_path,
                "attempts": sample.attempts,
            }
            for sample in missing
        ]
        if not kept:
            return {
                "skip_batch": True,
                "dropped_missing_sample_count": len(missing),
                "missing_samples": missing_payloads,
            }
        batch = self.collator(kept)
        if isinstance(batch, dict):
            batch = dict(batch)
            batch["dropped_missing_sample_count"] = len(missing)
            batch["missing_samples"] = missing_payloads
        return batch


def _to_tensor(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: _to_tensor(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_tensor(item) for item in value]
    return value


def _hand_to_sample(hand: HandAnnotation) -> dict[str, Any]:
    return _to_tensor(hand.to_dict())


def _depth_to_vggt_scale(depth: torch.Tensor, depth_mode: str | None) -> torch.Tensor:
    source_dtype = depth.dtype
    depth = depth.to(dtype=torch.float32)
    if depth_mode in _DEPTH_MILLIMETER_MODES:
        return depth / 1000.0
    if depth_mode in _DEPTH_TACO_VIDEO_MODES:
        # Official TACO FFV1 stores raw uint16 with scale 4000. The resized
        # H.264 mirror retains only the high byte, so it is quantized by 256.
        if source_dtype == torch.uint8:
            return depth * (256.0 / 4000.0)
        return depth / 4000.0
    return depth


def record_to_sample(record: FrameRecord, load_rgb: bool, load_depth: bool) -> dict[str, Any]:
    sample = {
        "dataset_name": record.dataset_name,
        "base_dataset_name": record.base_dataset_name,
        "split": record.split,
        "sequence_id": record.sequence_id,
        "frame_id": record.frame_id,
        "temporal_index": record.temporal_index,
        "view_name": record.view_name,
        "is_egocentric": record.is_egocentric,
        "rgb_ref": record.rgb_ref.to_dict() if record.rgb_ref else None,
        "depth_ref": record.depth_ref.to_dict() if record.depth_ref else None,
        "depth_mode": record.depth_mode,
        "intrinsics": _to_tensor(record.intrinsics),
        "camera_pose": _to_tensor(record.camera_pose),
        "hand_annos": [_hand_to_sample(hand) for hand in record.hand_annos],
        "left_hand": _hand_to_sample(record.unique_side_hand("left")) if record.unique_side_hand("left") else None,
        "right_hand": _hand_to_sample(record.unique_side_hand("right")) if record.unique_side_hand("right") else None,
        "has_mano": record.has_mano,
        "has_3d_joints": record.has_3d_joints,
        "has_depth_gt": record.has_depth_gt,
        "has_intrinsics_gt": record.has_intrinsics_gt,
        "has_camera_pose_gt": record.has_camera_pose_gt,
        "has_bbox_gt": record.has_bbox_gt,
        "has_joint_3d_gt": record.has_joint_3d_gt,
        "has_3r_gt": record.has_3r_gt,
        "max_left_count": record.max_left_count,
        "max_right_count": record.max_right_count,
        "extras": _to_tensor(record.extras),
    }
    if load_rgb and record.rgb_ref:
        sample["rgb"] = load_media_ref(record.rgb_ref, is_rgb=True)
    else:
        sample["rgb"] = None
    if load_depth and record.depth_ref:
        depth = _depth_to_vggt_scale(load_media_ref(record.depth_ref, is_rgb=False), record.depth_mode)
        # The official TACO export is 1920x1080 while the RGB mirror is
        # 512x376. Keep the metric values intact and use nearest-neighbour
        # spatial resampling so no synthetic depth values are introduced.
        if record.depth_mode == "taco_depth_video" and sample.get("rgb") is not None:
            target_size = tuple(int(value) for value in sample["rgb"].shape[-2:])
            if tuple(depth.shape[-2:]) != target_size:
                depth = F.interpolate(
                    depth.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode="nearest",
                ).squeeze(0).squeeze(0)
        sample["depth"] = depth
    else:
        sample["depth"] = None
    return sample


def collate_frame_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {"samples": samples}
    rgb_items = [sample["rgb"] for sample in samples if sample.get("rgb") is not None]
    if len(rgb_items) == len(samples) and rgb_items:
        batch["rgb"] = torch.stack(rgb_items)
    depth_items = [sample["depth"] for sample in samples if sample.get("depth") is not None]
    if len(depth_items) == len(samples) and depth_items:
        batch["depth"] = torch.stack(depth_items)
    return batch


def collate_chunk_batch(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {"chunks": chunks}
    rgb_items = [chunk["rgb"] for chunk in chunks if chunk.get("rgb") is not None]
    if len(rgb_items) == len(chunks) and rgb_items:
        batch["rgb"] = torch.stack(rgb_items)
    depth_items = [chunk["depth"] for chunk in chunks if chunk.get("depth") is not None]
    if len(depth_items) == len(chunks) and depth_items:
        batch["depth"] = torch.stack(depth_items)
    return batch


def _chunk_from_sample_indices(frame_dataset: BaseFrameDataset, sample_indices: list[int] | tuple[int, ...]) -> dict[str, Any]:
    samples = [frame_dataset[idx] for idx in sample_indices]
    chunk = {
        "dataset_name": samples[0]["dataset_name"],
        "base_dataset_name": samples[0]["base_dataset_name"],
        "split": samples[0]["split"],
        "sequence_id": samples[0]["sequence_id"],
        "view_name": samples[0]["view_name"],
        "frame_ids": [sample["frame_id"] for sample in samples],
        "temporal_indices": [sample["temporal_index"] for sample in samples],
        "samples": samples,
    }
    rgb_items = [sample["rgb"] for sample in samples if sample.get("rgb") is not None]
    if len(rgb_items) == len(samples) and rgb_items:
        chunk["rgb"] = torch.stack(rgb_items)
    else:
        chunk["rgb"] = None
    depth_items = [sample["depth"] for sample in samples if sample.get("depth") is not None]
    if len(depth_items) == len(samples) and depth_items:
        chunk["depth"] = torch.stack(depth_items)
    else:
        chunk["depth"] = None
    return chunk


class BaseFrameDataset(Dataset):
    dataset_name: str = "base"
    base_dataset_name: str = "base"
    allowed_temporal_lengths: tuple[int, ...] | None = None

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        *,
        load_rgb: bool = False,
        load_depth: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self._record_cache: OrderedDict[int, FrameRecord] = OrderedDict()
        self._records_materialized: list[FrameRecord] | None = None
        self._record_index: list[dict[str, Any]] | None = None
        if self._uses_lazy_records():
            self._validate_lazy_record_contract()
            self._record_index = self._build_record_index()
            assert self._record_index is not None
            self._record_count = len(self._record_index)
            self.sequence_to_indices = self._group_sequence_indices_from_index(self._record_index)
        else:
            self._validate_eager_record_contract()
            self._initialize_eager_records(self._build_records())

    def _build_records(self) -> list[FrameRecord]:
        raise NotImplementedError

    def _uses_lazy_records(self) -> bool:
        return False

    def _build_record_index(self) -> list[dict[str, Any]] | None:
        return None

    def _materialize_record(self, index_entry: dict[str, Any]) -> FrameRecord:
        raise NotImplementedError

    def prefetch_sequence(self, sequence_index: int) -> None:
        return None

    def _record_cache_limit(self) -> int | None:
        return 64

    def _initialize_eager_records(self, records: list[FrameRecord]) -> None:
        self._records_materialized = records
        self._record_count = len(records)
        self._record_cache = OrderedDict((index, record) for index, record in enumerate(records))
        self.sequence_to_indices = self._group_sequence_indices_from_records(records)

    def _has_custom_lazy_index_builder(self) -> bool:
        return type(self)._build_record_index is not BaseFrameDataset._build_record_index

    def _has_custom_lazy_materializer(self) -> bool:
        return type(self)._materialize_record is not BaseFrameDataset._materialize_record

    def _validate_lazy_record_contract(self) -> None:
        has_index_builder = self._has_custom_lazy_index_builder()
        has_materializer = self._has_custom_lazy_materializer()
        if not has_index_builder or not has_materializer:
            raise TypeError(
                "lazy record datasets must implement both _build_record_index() and _materialize_record()."
            )

    def _validate_eager_record_contract(self) -> None:
        has_lazy_hook = self._has_custom_lazy_index_builder() or self._has_custom_lazy_materializer()
        if has_lazy_hook:
            raise TypeError(
                "Datasets with lazy record hooks must explicitly opt in via _uses_lazy_records()."
            )

    @property
    def records(self) -> tuple[FrameRecord, ...]:
        if self._records_materialized is None and not self._uses_lazy_records():
            self._records_materialized = [self._get_record(index) for index in range(self._record_count)]
        if self._records_materialized is not None:
            return tuple(self._records_materialized)
        return tuple(self._get_record(index) for index in range(self._record_count))

    def _group_sequence_indices_from_records(self, records: list[FrameRecord]) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            grouped[record.sequence_id].append(index)
        for indices in grouped.values():
            indices.sort(key=lambda idx: records[idx].temporal_index)
        return dict(grouped)

    def _group_sequence_indices_from_index(self, record_index: list[dict[str, Any]]) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, index_entry in enumerate(record_index):
            grouped[str(index_entry["sequence_id"])].append(index)
        for indices in grouped.values():
            indices.sort(key=lambda idx: int(record_index[idx]["temporal_index"]))
        return dict(grouped)

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += self._record_count
        if index < 0 or index >= self._record_count:
            raise IndexError(index)
        return index

    def _get_record(self, index: int) -> FrameRecord:
        index = self._normalize_index(index)
        cached = self._record_cache.get(index)
        if cached is not None:
            self._record_cache.move_to_end(index)
            return cached
        if self._record_index is None:
            raise IndexError(index)
        record = self._materialize_record(self._record_index[index])
        self._record_cache[index] = record
        self._record_cache.move_to_end(index)
        cache_limit = self._record_cache_limit()
        if cache_limit is not None and cache_limit > 0:
            while len(self._record_cache) > cache_limit:
                self._record_cache.popitem(last=False)
        return record

    def __len__(self) -> int:
        return self._record_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._get_record(index)
        return record_to_sample(record, load_rgb=self.load_rgb, load_depth=self.load_depth)


class CombinedFrameDataset(Dataset):
    def __init__(self, datasets: list[BaseFrameDataset]) -> None:
        self.datasets = datasets
        self._offsets: list[int] = []
        self.sequence_to_indices: dict[str, list[int]] = {}
        running_total = 0
        for dataset_index, dataset in enumerate(datasets):
            self._offsets.append(running_total)
            for sequence_id, indices in dataset.sequence_to_indices.items():
                combined_sequence_id = f"{dataset_index}:{sequence_id}"
                self.sequence_to_indices[combined_sequence_id] = [running_total + int(index) for index in indices]
            running_total += len(dataset)
        self._length = running_total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        dataset_index = bisect_right(self._offsets, index) - 1
        local_index = index - self._offsets[dataset_index]
        return self.datasets[dataset_index][local_index]


class TemporalChunkDataset(Dataset):
    def __init__(
        self,
        frame_dataset: BaseFrameDataset,
        *,
        num_frames: int,
        window_stride: int = 1,
        drop_last: bool = True,
    ) -> None:
        self.frame_dataset = frame_dataset
        self.num_frames = num_frames
        self.window_stride = window_stride
        self.drop_last = drop_last
        self._window_specs = self._build_window_specs()
        self._window_offsets: list[int] = []
        running_total = 0
        for spec in self._window_specs:
            self._window_offsets.append(running_total)
            running_total += int(spec["window_count"])
        self._total_windows = running_total

    def _build_window_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for indices in self.frame_dataset.sequence_to_indices.values():
            if len(indices) < self.num_frames:
                if not self.drop_last and indices:
                    specs.append(
                        {
                            "indices": tuple(indices),
                            "window_count": 1,
                            "short_window": True,
                        }
                    )
                continue
            window_count = 1 + (len(indices) - self.num_frames) // self.window_stride
            specs.append(
                {
                    "indices": tuple(indices),
                    "window_count": window_count,
                    "short_window": False,
                }
            )
        return specs

    def __len__(self) -> int:
        return self._total_windows

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += self._total_windows
        if index < 0 or index >= self._total_windows:
            raise IndexError(index)
        return index

    def _resolve_window_indices(self, index: int) -> tuple[int, ...]:
        index = self._normalize_index(index)
        spec_index = bisect_right(self._window_offsets, index) - 1
        spec = self._window_specs[spec_index]
        if bool(spec["short_window"]):
            return spec["indices"]
        local_window_index = index - self._window_offsets[spec_index]
        start = local_window_index * self.window_stride
        end = start + self.num_frames
        return spec["indices"][start:end]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_indices = self._resolve_window_indices(index)
        return _chunk_from_sample_indices(self.frame_dataset, sample_indices)


class RandomLengthTemporalChunkDataset(Dataset):
    def __init__(
        self,
        frame_dataset: BaseFrameDataset,
        *,
        min_num_frames: int,
        max_num_frames: int,
        window_stride: int = 1,
        drop_last: bool = True,
    ) -> None:
        if min_num_frames <= 0 or max_num_frames <= 0:
            raise ValueError("min_num_frames 和 max_num_frames 必须为正整数。")
        if min_num_frames > max_num_frames:
            raise ValueError("min_num_frames 不能大于 max_num_frames。")
        self.frame_dataset = frame_dataset
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.window_stride = window_stride
        self.drop_last = drop_last
        self._sequence_items = [tuple(indices) for indices in self.frame_dataset.sequence_to_indices.values()]
        self._anchor_cache: dict[tuple[int, int], tuple[list[int], list[int], int]] = {}

    def _supports_num_frames(self, num_frames: int) -> bool:
        allowed_lengths = self.frame_dataset.allowed_temporal_lengths
        return allowed_lengths is None or int(num_frames) in allowed_lengths

    def prefetch_sequence(self, sequence_index: int) -> None:
        prefetch_sequence = getattr(self.frame_dataset, "prefetch_sequence", None)
        if prefetch_sequence is not None:
            prefetch_sequence(int(sequence_index))

    @property
    def sequence_count(self) -> int:
        return len(self._sequence_items)

    def sequence_length(self, sequence_index: int) -> int:
        if sequence_index < 0:
            sequence_index += len(self._sequence_items)
        if sequence_index < 0 or sequence_index >= len(self._sequence_items):
            raise IndexError(sequence_index)
        return len(self._sequence_items[sequence_index])

    def valid_start_count(self, sequence_index: int, *, num_frames: int, frame_stride: int) -> int:
        if num_frames <= 0 or frame_stride <= 0 or not self._supports_num_frames(num_frames):
            return 0
        sequence_length = self.sequence_length(sequence_index)
        max_start = sequence_length - 1 - (int(num_frames) - 1) * int(frame_stride)
        if max_start < 0:
            return 0
        return int(max_start // self.window_stride) + 1

    def _anchor_specs(self, num_frames: int, frame_stride: int = 1) -> tuple[list[int], list[int], int]:
        cache_key = (int(num_frames), int(frame_stride))
        cached = self._anchor_cache.get(cache_key)
        if cached is not None:
            return cached
        if not self._supports_num_frames(num_frames):
            cached = ([0] * len(self._sequence_items), [0] * len(self._sequence_items), 0)
            self._anchor_cache[cache_key] = cached
            return cached
        offsets: list[int] = []
        counts: list[int] = []
        running_total = 0
        for indices in self._sequence_items:
            offsets.append(running_total)
            if len(indices) < num_frames:
                count = 1 if (frame_stride == 1 and not self.drop_last and len(indices) > 0) else 0
            else:
                max_start = len(indices) - 1 - (int(num_frames) - 1) * int(frame_stride)
                count = 0 if max_start < 0 else 1 + max_start // self.window_stride
            counts.append(count)
            running_total += count
        cached = (offsets, counts, running_total)
        self._anchor_cache[cache_key] = cached
        return cached

    def valid_anchor_count(self, num_frames: int, frame_stride: int = 1) -> int:
        return self._anchor_specs(num_frames, frame_stride)[2]

    def sequence_anchor_counts(self, num_frames: int, frame_stride: int = 1) -> list[int]:
        return list(self._anchor_specs(num_frames, frame_stride)[1])

    def global_anchor_index(self, sequence_index: int, local_anchor_index: int, num_frames: int, frame_stride: int = 1) -> int:
        offsets, counts, total_count = self._anchor_specs(num_frames, frame_stride)
        if total_count <= 0:
            raise IndexError("当前数据集没有可用随机长度窗口。")
        if sequence_index < 0 or sequence_index >= len(counts):
            raise IndexError(sequence_index)
        if local_anchor_index < 0 or local_anchor_index >= counts[sequence_index]:
            raise IndexError(local_anchor_index)
        return offsets[sequence_index] + local_anchor_index

    def __len__(self) -> int:
        return self.valid_anchor_count(self.min_num_frames)

    def _resolve_sequence_window_indices(
        self,
        sequence_index: int,
        start: int,
        num_frames: int,
        frame_stride: int,
    ) -> tuple[int, ...]:
        if num_frames < self.min_num_frames or num_frames > self.max_num_frames:
            raise ValueError(f"num_frames={num_frames} 超出允许区间 [{self.min_num_frames}, {self.max_num_frames}]")
        if not self._supports_num_frames(num_frames):
            raise ValueError(f"num_frames={num_frames} 不受数据集 {self.frame_dataset.dataset_name} 支持。")
        if frame_stride <= 0:
            raise ValueError("frame_stride 必须为正整数。")
        if sequence_index < 0:
            sequence_index += len(self._sequence_items)
        if sequence_index < 0 or sequence_index >= len(self._sequence_items):
            raise IndexError(sequence_index)
        indices = self._sequence_items[sequence_index]
        if start < 0:
            raise IndexError(start)
        final_index = int(start) + (int(num_frames) - 1) * int(frame_stride)
        if final_index >= len(indices):
            raise IndexError((sequence_index, start, num_frames, frame_stride))
        return tuple(indices[int(start) + frame_index * int(frame_stride)] for frame_index in range(int(num_frames)))

    def _resolve_window_indices(self, anchor_index: int, num_frames: int, frame_stride: int = 1) -> tuple[int, ...]:
        offsets, counts, total_count = self._anchor_specs(num_frames, frame_stride)
        if total_count <= 0:
            raise IndexError("当前数据集没有可用随机长度窗口。")
        if anchor_index < 0:
            anchor_index += total_count
        if anchor_index < 0 or anchor_index >= total_count:
            raise IndexError(anchor_index)
        sequence_index = bisect_right(offsets, anchor_index) - 1
        indices = self._sequence_items[sequence_index]
        local_index = anchor_index - offsets[sequence_index]
        if counts[sequence_index] <= 0:
            raise IndexError(anchor_index)
        if len(indices) < num_frames:
            return indices
        start = local_index * self.window_stride
        return self._resolve_sequence_window_indices(sequence_index, start, num_frames, frame_stride)

    def __getitem__(self, index: int | tuple[int, ...]) -> dict[str, Any]:
        index, prefetch_hint = _split_prefetch_index(index)
        if prefetch_hint is not None:
            get_container_prefetch_manager().submit(
                prefetch_hint,
                lambda: self.prefetch_sequence(prefetch_hint.sequence_index),
            )
        target_image_size_hw = None
        frame_stride = 1
        sequence_index = None
        sequence_start = None
        if isinstance(index, tuple):
            if len(index) == 6:
                sequence_index, sequence_start, num_frames, frame_stride, target_height, target_width = index
                sample_indices = self._resolve_sequence_window_indices(
                    int(sequence_index),
                    int(sequence_start),
                    int(num_frames),
                    int(frame_stride),
                )
                target_image_size_hw = (int(target_height), int(target_width))
                chunk = _chunk_from_sample_indices(self.frame_dataset, sample_indices)
                chunk["frame_stride"] = int(frame_stride)
                chunk["target_image_size_hw"] = target_image_size_hw
                chunk["sequence_index"] = int(sequence_index)
                chunk["sequence_start"] = int(sequence_start)
                return chunk
            anchor_index, num_frames = index
        else:
            anchor_index = index
            num_frames = self.min_num_frames
        if num_frames < self.min_num_frames or num_frames > self.max_num_frames:
            raise ValueError(f"num_frames={num_frames} 超出允许区间 [{self.min_num_frames}, {self.max_num_frames}]")
        sample_indices = self._resolve_window_indices(anchor_index, num_frames)
        return _chunk_from_sample_indices(self.frame_dataset, sample_indices)

    def epoch_sample_count(self) -> int:
        return sum(self.valid_anchor_count(num_frames) for num_frames in range(self.min_num_frames, self.max_num_frames + 1))

    def epoch_samples(self) -> list[tuple[int, int]]:
        samples: list[tuple[int, int]] = []
        for num_frames in range(self.min_num_frames, self.max_num_frames + 1):
            valid_count = self.valid_anchor_count(num_frames)
            samples.extend((anchor_index, num_frames) for anchor_index in range(valid_count))
        return samples


class MultiSourceRandomLengthTemporalChunkDataset(Dataset):
    def __init__(
        self,
        datasets: list[RandomLengthTemporalChunkDataset],
        *,
        dataset_names: list[str],
    ) -> None:
        if len(datasets) != len(dataset_names):
            raise ValueError("datasets 与 dataset_names 长度必须一致。")
        if not datasets:
            raise ValueError("datasets 不能为空。")
        min_num_frames = datasets[0].min_num_frames
        max_num_frames = datasets[0].max_num_frames
        for dataset in datasets[1:]:
            if dataset.min_num_frames != min_num_frames or dataset.max_num_frames != max_num_frames:
                raise ValueError("MultiSourceRandomLengthTemporalChunkDataset 要求所有子数据集 frame 范围一致。")
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def prefetch_sequence(self, dataset_index: int, sequence_index: int) -> None:
        self.datasets[int(dataset_index)].prefetch_sequence(int(sequence_index))

    def __getitem__(self, index: int | tuple[int, ...]) -> dict[str, Any]:
        index, prefetch_hint = _split_prefetch_index(index)
        if prefetch_hint is not None:
            get_container_prefetch_manager().submit(
                prefetch_hint,
                lambda: self.prefetch_sequence(prefetch_hint.dataset_index, prefetch_hint.sequence_index),
            )
        if isinstance(index, tuple):
            if len(index) == 7:
                dataset_index, sequence_index, start, num_frames, frame_stride, target_height, target_width = index
                return self.datasets[int(dataset_index)][
                    (
                        int(sequence_index),
                        int(start),
                        int(num_frames),
                        int(frame_stride),
                        int(target_height),
                        int(target_width),
                    )
                ]
            dataset_index, anchor_index, num_frames = index
            return self.datasets[dataset_index][(anchor_index, num_frames)]
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        running_total = 0
        for dataset in self.datasets:
            next_total = running_total + len(dataset)
            if index < next_total:
                return dataset[index - running_total]
            running_total = next_total
        raise IndexError(index)

    def valid_anchor_count(self, dataset_index: int, num_frames: int) -> int:
        return self.datasets[dataset_index].valid_anchor_count(num_frames)

    def epoch_sample_count(self) -> int:
        return sum(dataset.epoch_sample_count() for dataset in self.datasets)

    def _dataset_weight(self, dataset_name: str, dataset_sampling_weights: dict[str, float] | None) -> float:
        if dataset_sampling_weights is None:
            return 1.0
        weight = float(dataset_sampling_weights.get(dataset_name, 1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"dataset_sampling_weights[{dataset_name!r}] 必须是有限的非负数，当前为 {weight}.")
        return weight

    def epoch_samples(self) -> list[tuple[int, int, int]]:
        samples: list[tuple[int, int, int]] = []
        for dataset_index, dataset in enumerate(self.datasets):
            samples.extend((dataset_index, anchor_index, num_frames) for anchor_index, num_frames in dataset.epoch_samples())
        return samples

    def weighted_epoch_sample_counts(self, dataset_sampling_weights: dict[str, float] | None = None) -> dict[str, int]:
        outputs: dict[str, int] = {}
        for dataset_name, dataset in zip(self.dataset_names, self.datasets, strict=True):
            raw_count = dataset.epoch_sample_count()
            weight = self._dataset_weight(dataset_name, dataset_sampling_weights)
            full_repeats = int(weight)
            fractional_repeat_count = int(round((weight - float(full_repeats)) * float(raw_count)))
            outputs[dataset_name] = (full_repeats * raw_count) + fractional_repeat_count
        return outputs

    def weighted_epoch_samples(
        self,
        *,
        dataset_sampling_weights: dict[str, float] | None = None,
        seed: int,
        epoch: int,
    ) -> list[tuple[int, int, int]]:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + int(epoch))
        outputs: list[tuple[int, int, int]] = []
        for dataset_index, (dataset_name, dataset) in enumerate(zip(self.dataset_names, self.datasets, strict=True)):
            base_samples = [(dataset_index, anchor_index, num_frames) for anchor_index, num_frames in dataset.epoch_samples()]
            if not base_samples:
                continue
            weight = self._dataset_weight(dataset_name, dataset_sampling_weights)
            full_repeats = int(weight)
            fractional_repeat_count = int(round((weight - float(full_repeats)) * float(len(base_samples))))
            for _ in range(full_repeats):
                outputs.extend(base_samples)
            if fractional_repeat_count > 0:
                permutation = torch.randperm(len(base_samples), generator=generator).tolist()
                outputs.extend(base_samples[index] for index in permutation[:fractional_repeat_count])
        if not outputs:
            return outputs
        permutation = torch.randperm(len(outputs), generator=generator).tolist()
        return [outputs[index] for index in permutation]

    def dataset_epoch_sample_counts(self) -> dict[str, int]:
        return {
            dataset_name: dataset.epoch_sample_count()
            for dataset_name, dataset in zip(self.dataset_names, self.datasets, strict=True)
        }

    def sampling_probabilities(self, dataset_sampling_weights: dict[str, float] | None = None) -> dict[str, float]:
        if dataset_sampling_weights is None:
            dataset_sampling_weights = {}
        probabilities = {dataset_name: 0.0 for dataset_name in self.dataset_names}
        frame_lengths = list(range(self.min_num_frames, self.max_num_frames + 1))
        if not frame_lengths:
            return probabilities
        for num_frames in frame_lengths:
            weighted_counts = []
            total_weighted_count = 0.0
            for dataset_name, dataset in zip(self.dataset_names, self.datasets, strict=True):
                count = dataset.valid_anchor_count(num_frames)
                weight = self._dataset_weight(dataset_name, dataset_sampling_weights)
                weighted_count = weight * float(count)
                weighted_counts.append(weighted_count)
                total_weighted_count += weighted_count
            if total_weighted_count <= 0:
                continue
            for dataset_name, weighted_count in zip(self.dataset_names, weighted_counts, strict=True):
                probabilities[dataset_name] += weighted_count / total_weighted_count
        divisor = float(len(frame_lengths))
        return {dataset_name: probability / divisor for dataset_name, probability in probabilities.items()}


class RandomLengthBatchSampler:
    is_streaming_sampler = True

    def __init__(
        self,
        dataset: RandomLengthTemporalChunkDataset,
        *,
        batch_size: int,
        seed: int,
        ddp_batch_group_size: int = 1,
        target_shapes: Sequence[Sequence[int]] | None = None,
        min_frame_stride: int = 1,
        max_frame_stride: int = 1,
        group_frame_stride_in_batch: bool = False,
        sequence_sampling_mode: str = "window_proportional",
        container_local_sampling_enabled: bool = False,
        container_local_sampling_dataset_names: Sequence[str] | None = None,
        container_local_sampling_block_batches: int = 64,
        container_local_sampling_start_order: str = "random_cursor",
        container_local_prefetch_enabled: bool = False,
        container_local_prefetch_margin_batches: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        if ddp_batch_group_size <= 0:
            raise ValueError("ddp_batch_group_size 必须为正整数。")
        if min_frame_stride <= 0 or max_frame_stride <= 0:
            raise ValueError("frame_stride 必须为正整数。")
        if min_frame_stride > max_frame_stride:
            raise ValueError("min_frame_stride 不能大于 max_frame_stride。")
        if sequence_sampling_mode not in {"window_proportional", "uniform"}:
            raise ValueError("sequence_sampling_mode 必须是 'window_proportional' 或 'uniform'。")
        if container_local_sampling_block_batches <= 0:
            raise ValueError("container_local_sampling_block_batches 必须为正整数。")
        if container_local_sampling_start_order not in _CONTAINER_LOCAL_SAMPLING_START_ORDERS:
            raise ValueError("container_local_sampling_start_order 必须是 sequential 或 random_cursor。")
        if container_local_prefetch_enabled:
            _validate_container_local_prefetch_margin(
                container_local_sampling_block_batches,
                container_local_prefetch_margin_batches,
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.ddp_batch_group_size = int(ddp_batch_group_size)
        self.target_shapes = self._normalize_target_shapes(target_shapes)
        self.frame_strides = list(range(int(min_frame_stride), int(max_frame_stride) + 1))
        self.group_frame_stride_in_batch = bool(group_frame_stride_in_batch)
        self.sequence_sampling_mode = sequence_sampling_mode
        self.container_local_sampling_enabled = bool(container_local_sampling_enabled)
        self.container_local_sampling_dataset_names = _normalize_dataset_name_set(container_local_sampling_dataset_names)
        self.container_local_sampling_block_batches = int(container_local_sampling_block_batches)
        self.container_local_sampling_start_order = str(container_local_sampling_start_order)
        self.container_local_prefetch_enabled = bool(container_local_prefetch_enabled)
        self.container_local_prefetch_margin_batches = int(container_local_prefetch_margin_batches)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self._start_cursor_state: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        self._local_sequence_state: dict[int, _LocalSequenceState] = {}
        self._local_prefetch_sequence_state: dict[int, _LocalSequenceState] = {}
        self._local_start_cursor_state: dict[tuple[int, int, int, int], int] = {}
        self.valid_lengths = [
            num_frames
            for num_frames in range(self.dataset.min_num_frames, self.dataset.max_num_frames + 1)
            if self.dataset.valid_anchor_count(num_frames) > 0
        ]
        self.valid_buckets = self._build_valid_buckets() if self.target_shapes else []
        if not self.valid_lengths and not self.valid_buckets:
            raise ValueError("随机长度数据集没有有效窗口。")

    def _normalize_target_shapes(self, target_shapes: Sequence[Sequence[int]] | None) -> list[tuple[int, int]]:
        if target_shapes is None:
            return []
        normalized: list[tuple[int, int]] = []
        for shape in target_shapes:
            if len(shape) != 2:
                raise ValueError(f"target shape 必须是 [height, width]，当前为 {shape!r}")
            height, width = int(shape[0]), int(shape[1])
            if height <= 0 or width <= 0:
                raise ValueError(f"target shape 必须为正整数，当前为 {shape!r}")
            normalized.append((height, width))
        return normalized

    def _bucket_num_frames(self, bucket: tuple[int, ...]) -> int:
        return int(bucket[0])

    def _bucket_frame_stride(self, bucket: tuple[int, ...]) -> int | None:
        return int(bucket[1]) if self.group_frame_stride_in_batch else None

    def _bucket_shape(self, bucket: tuple[int, ...]) -> tuple[int, int]:
        if self.group_frame_stride_in_batch:
            return int(bucket[2]), int(bucket[3])
        return int(bucket[1]), int(bucket[2])

    def _sequence_valid_start_count(self, sequence_index: int, bucket: tuple[int, ...]) -> int:
        num_frames = self._bucket_num_frames(bucket)
        bucket_frame_stride = self._bucket_frame_stride(bucket)
        if bucket_frame_stride is not None:
            return self.dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=bucket_frame_stride)
        return sum(
            self.dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=frame_stride)
            for frame_stride in self.frame_strides
        )

    def _build_valid_buckets(self) -> list[tuple[int, ...]]:
        buckets: list[tuple[int, ...]] = []
        for num_frames in range(self.dataset.min_num_frames, self.dataset.max_num_frames + 1):
            for target_height, target_width in self.target_shapes:
                if self.group_frame_stride_in_batch:
                    for frame_stride in self.frame_strides:
                        if any(
                            self.dataset.valid_start_count(
                                sequence_index,
                                num_frames=num_frames,
                                frame_stride=frame_stride,
                            )
                            > 0
                            for sequence_index in range(self.dataset.sequence_count)
                        ):
                            buckets.append((num_frames, frame_stride, target_height, target_width))
                    continue
                bucket = (num_frames, target_height, target_width)
                if any(self._sequence_valid_start_count(sequence_index, bucket) > 0 for sequence_index in range(self.dataset.sequence_count)):
                    buckets.append(bucket)
        return buckets

    def __len__(self) -> int:
        if self.target_shapes:
            total_epoch_samples = sum(
                self._sequence_valid_start_count(sequence_index, bucket)
                for bucket in self.valid_buckets
                for sequence_index in range(self.dataset.sequence_count)
            )
            return total_epoch_samples // self.batch_size
        total_epoch_samples = self.dataset.epoch_sample_count()
        return total_epoch_samples // self.batch_size

    def _weighted_sample_positions(self, weights: torch.Tensor, count: int, *, replacement: bool) -> list[int]:
        if count <= 0:
            return []
        return torch.multinomial(weights, num_samples=count, replacement=replacement, generator=self.generator).tolist()

    def _sample_sequence_indices(self, bucket: tuple[int, ...]) -> list[int]:
        active_indices: list[int] = []
        weights: list[float] = []
        for sequence_index in range(self.dataset.sequence_count):
            valid_count = self._sequence_valid_start_count(sequence_index, bucket)
            if valid_count <= 0:
                continue
            active_indices.append(sequence_index)
            weights.append(1.0 if self.sequence_sampling_mode == "uniform" else float(valid_count))
        if not active_indices:
            raise IndexError("当前 bucket 没有 eligible sequence。")
        weight_tensor = torch.tensor(weights, dtype=torch.float64)
        if len(active_indices) >= self.batch_size:
            positions = self._weighted_sample_positions(weight_tensor, self.batch_size, replacement=False)
            return [active_indices[position] for position in positions]
        positions = self._weighted_sample_positions(weight_tensor, len(active_indices), replacement=False)
        sampled = [active_indices[position] for position in positions]
        extra_positions = self._weighted_sample_positions(weight_tensor, self.batch_size - len(sampled), replacement=True)
        sampled.extend(active_indices[position] for position in extra_positions)
        return sampled

    def _sample_frame_stride(self, sequence_index: int, bucket: tuple[int, ...]) -> int:
        bucket_frame_stride = self._bucket_frame_stride(bucket)
        if bucket_frame_stride is not None:
            return bucket_frame_stride
        num_frames = self._bucket_num_frames(bucket)
        active_strides: list[int] = []
        weights: list[float] = []
        for frame_stride in self.frame_strides:
            valid_count = self.dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=frame_stride)
            if valid_count <= 0:
                continue
            active_strides.append(frame_stride)
            weights.append(1.0 if self.sequence_sampling_mode == "uniform" else float(valid_count))
        if not active_strides:
            raise IndexError("当前 sequence 没有 eligible frame_stride。")
        weight_tensor = torch.tensor(weights, dtype=torch.float64)
        position = self._weighted_sample_positions(weight_tensor, 1, replacement=True)[0]
        return active_strides[position]

    def _sample_coprime_jump(self, valid_start_count: int) -> int:
        if valid_start_count <= 1:
            return 1
        start = int(torch.randint(valid_start_count, (1,), generator=self.generator).item())
        for offset in range(valid_start_count):
            candidate = 1 + ((start + offset) % valid_start_count)
            if math.gcd(candidate, valid_start_count) == 1:
                return candidate
        return 1

    def _next_start(self, sequence_index: int, num_frames: int, frame_stride: int) -> int:
        valid_start_count = self.dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=frame_stride)
        if valid_start_count <= 0:
            raise IndexError((sequence_index, num_frames, frame_stride))
        key = (int(sequence_index), int(num_frames), int(frame_stride))
        state = self._start_cursor_state.get(key)
        if state is None:
            offset = int(torch.randint(valid_start_count, (1,), generator=self.generator).item())
            jump = self._sample_coprime_jump(valid_start_count)
            cursor = 0
        else:
            offset, jump, cursor = state
        start_slot = (offset + cursor * jump) % valid_start_count
        self._start_cursor_state[key] = (offset, jump, cursor + 1)
        return int(start_slot) * int(self.dataset.window_stride)

    def _local_sampling_enabled_for_dataset(self) -> bool:
        if not self.container_local_sampling_enabled:
            return False
        dataset_name = _random_length_dataset_name(self.dataset)
        return dataset_name in self.container_local_sampling_dataset_names

    def _locality_slot(self, batch_index: int) -> int:
        return int(batch_index) % max(int(self.ddp_batch_group_size), 1)

    def _next_local_start(self, slot: int, sequence_index: int, num_frames: int, frame_stride: int) -> int:
        if self.container_local_sampling_start_order == "random_cursor":
            return self._next_start(sequence_index, num_frames, frame_stride)
        valid_start_count = self.dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=frame_stride)
        if valid_start_count <= 0:
            raise IndexError((sequence_index, num_frames, frame_stride))
        key = (int(slot), int(sequence_index), int(num_frames), int(frame_stride))
        start_slot = self._local_start_cursor_state.get(key)
        if start_slot is None:
            start_slot = int(torch.randint(valid_start_count, (1,), generator=self.generator).item())
        output_slot = int(start_slot) % int(valid_start_count)
        self._local_start_cursor_state[key] = (output_slot + 1) % int(valid_start_count)
        return output_slot * int(self.dataset.window_stride)

    def _sample_local_sequence_index(
        self,
        bucket: tuple[int, ...],
        slot: int,
        *,
        excluded_sequence_indices: set[int] | None = None,
    ) -> int:
        active_indices: list[int] = []
        weights: list[float] = []
        used_sequence_indices = {
            state.sequence_index
            for state_slot, state in self._local_sequence_state.items()
            if state_slot != int(slot)
            and state.remaining_batches > 0
            and self._sequence_valid_start_count(state.sequence_index, bucket) > 0
        }
        if excluded_sequence_indices:
            used_sequence_indices.update(int(index) for index in excluded_sequence_indices)
        for sequence_index in range(self.dataset.sequence_count):
            valid_count = self._sequence_valid_start_count(sequence_index, bucket)
            if valid_count <= 0:
                continue
            active_indices.append(sequence_index)
            weights.append(1.0 if self.sequence_sampling_mode == "uniform" else float(valid_count))
        if not active_indices:
            raise IndexError("当前 bucket 没有 eligible sequence。")
        available = [
            (sequence_index, weight)
            for sequence_index, weight in zip(active_indices, weights, strict=True)
            if sequence_index not in used_sequence_indices
        ]
        if available:
            active_indices = [sequence_index for sequence_index, _ in available]
            weights = [weight for _, weight in available]
        position = self._weighted_sample_positions(torch.tensor(weights, dtype=torch.float64), 1, replacement=True)[0]
        return active_indices[position]

    def _local_prefetch_hint(
        self,
        bucket: tuple[int, ...],
        slot: int,
        state: _LocalSequenceState,
    ) -> ContainerPrefetchHint | None:
        if not self.container_local_prefetch_enabled:
            return None
        if state.remaining_batches > self.container_local_prefetch_margin_batches:
            return None
        next_state = self._local_prefetch_sequence_state.get(slot)
        if next_state is None or next_state.remaining_batches <= 0 or self._sequence_valid_start_count(next_state.sequence_index, bucket) <= 0:
            next_state = _LocalSequenceState(
                dataset_index=0,
                sequence_index=self._sample_local_sequence_index(
                    bucket,
                    slot,
                    excluded_sequence_indices={state.sequence_index},
                ),
                remaining_batches=self.container_local_sampling_block_batches,
            )
            self._local_prefetch_sequence_state[slot] = next_state
        return ContainerPrefetchHint(dataset_index=0, sequence_index=next_state.sequence_index)

    def _sample_local_bucketed_batch(self, bucket: tuple[int, ...], batch_index: int) -> list[tuple[int, int, int, int, int, int]]:
        slot = self._locality_slot(batch_index)
        num_frames = self._bucket_num_frames(bucket)
        target_height, target_width = self._bucket_shape(bucket)
        state = self._local_sequence_state.get(slot)
        if (
            state is None
            or state.remaining_batches <= 0
            or self._sequence_valid_start_count(state.sequence_index, bucket) <= 0
        ):
            prefetched_state = self._local_prefetch_sequence_state.pop(slot, None)
            if prefetched_state is not None and prefetched_state.remaining_batches > 0 and self._sequence_valid_start_count(prefetched_state.sequence_index, bucket) > 0:
                state = prefetched_state
            else:
                state = _LocalSequenceState(
                    dataset_index=0,
                    sequence_index=self._sample_local_sequence_index(bucket, slot),
                    remaining_batches=self.container_local_sampling_block_batches,
                )
        batch: list[tuple[int, int, int, int, int, int]] = []
        for _ in range(self.batch_size):
            frame_stride = self._sample_frame_stride(state.sequence_index, bucket)
            start = self._next_local_start(slot, state.sequence_index, num_frames, frame_stride)
            batch.append((state.sequence_index, start, num_frames, frame_stride, target_height, target_width))
        prefetch_hint = self._local_prefetch_hint(bucket, slot, state)
        self._local_sequence_state[slot] = _LocalSequenceState(
            dataset_index=0,
            sequence_index=state.sequence_index,
            remaining_batches=state.remaining_batches - 1,
        )
        return _wrap_batch_with_prefetch_hint(batch, prefetch_hint)

    def _sample_bucketed_batch(self, bucket: tuple[int, ...], batch_index: int = 0) -> list[tuple[int, int, int, int, int, int]]:
        if self._local_sampling_enabled_for_dataset():
            return self._sample_local_bucketed_batch(bucket, batch_index)
        num_frames = self._bucket_num_frames(bucket)
        target_height, target_width = self._bucket_shape(bucket)
        sequence_indices = self._sample_sequence_indices(bucket)
        batch: list[tuple[int, int, int, int, int, int]] = []
        for sequence_index in sequence_indices:
            frame_stride = self._sample_frame_stride(sequence_index, bucket)
            start = self._next_start(sequence_index, num_frames, frame_stride)
            batch.append((sequence_index, start, num_frames, frame_stride, target_height, target_width))
        return batch

    def __iter__(self):
        batch_index = 0
        if self.target_shapes:
            bucket = self.valid_buckets[0]
            while True:
                if batch_index % self.ddp_batch_group_size == 0:
                    bucket_index = int(torch.randint(len(self.valid_buckets), (1,), generator=self.generator).item())
                    bucket = self.valid_buckets[bucket_index]
                batch_index += 1
                yield self._sample_bucketed_batch(bucket, batch_index - 1)
        else:
            num_frames = self.valid_lengths[0]
            while True:
                if batch_index % self.ddp_batch_group_size == 0:
                    length_index = int(torch.randint(len(self.valid_lengths), (1,), generator=self.generator).item())
                    num_frames = self.valid_lengths[length_index]
                valid_count = self.dataset.valid_anchor_count(num_frames)
                anchor_indices = torch.randint(valid_count, (self.batch_size,), generator=self.generator).tolist()
                batch_index += 1
                yield [(anchor_index, num_frames) for anchor_index in anchor_indices]


class RandomLengthEpochBatchSampler:
    def __init__(
        self,
        dataset: RandomLengthTemporalChunkDataset | MultiSourceRandomLengthTemporalChunkDataset,
        *,
        batch_size: int,
        seed: int,
        epoch: int,
        dataset_sampling_weights: dict[str, float] | None = None,
        target_shapes: Sequence[Sequence[int]] | None = None,
        min_frame_stride: int = 1,
        max_frame_stride: int = 1,
        group_frame_stride_in_batch: bool = False,
        container_local_sampling_enabled: bool = False,
        container_local_sampling_dataset_names: Sequence[str] | None = None,
        container_local_sampling_block_batches: int = 64,
        container_local_sampling_start_order: str = "random_cursor",
        container_local_prefetch_enabled: bool = False,
        container_local_prefetch_margin_batches: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        if min_frame_stride <= 0 or max_frame_stride <= 0:
            raise ValueError("frame_stride 必须为正整数。")
        if min_frame_stride > max_frame_stride:
            raise ValueError("min_frame_stride 不能大于 max_frame_stride。")
        if container_local_sampling_block_batches <= 0:
            raise ValueError("container_local_sampling_block_batches 必须为正整数。")
        if container_local_sampling_start_order not in _CONTAINER_LOCAL_SAMPLING_START_ORDERS:
            raise ValueError("container_local_sampling_start_order 必须是 sequential 或 random_cursor。")
        if container_local_prefetch_enabled:
            _validate_container_local_prefetch_margin(
                container_local_sampling_block_batches,
                container_local_prefetch_margin_batches,
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.dataset_sampling_weights = dataset_sampling_weights or {}
        self.target_shapes = self._normalize_target_shapes(target_shapes)
        self.frame_strides = list(range(int(min_frame_stride), int(max_frame_stride) + 1))
        self.group_frame_stride_in_batch = bool(group_frame_stride_in_batch)
        self.container_local_sampling_enabled = bool(container_local_sampling_enabled)
        self.container_local_sampling_dataset_names = _normalize_dataset_name_set(container_local_sampling_dataset_names)
        self.container_local_sampling_block_batches = int(container_local_sampling_block_batches)
        self.container_local_sampling_start_order = str(container_local_sampling_start_order)
        self.container_local_prefetch_enabled = bool(container_local_prefetch_enabled)
        self.container_local_prefetch_margin_batches = int(container_local_prefetch_margin_batches)
        self._epoch_plan = self._build_compact_epoch_plan()
        self._epoch_groups = list(self._epoch_plan.groups)
        self.raw_epoch_sample_count = self._epoch_plan.sample_count
        self.used_epoch_sample_count = self.raw_epoch_sample_count

    def _normalize_target_shapes(self, target_shapes: Sequence[Sequence[int]] | None) -> list[tuple[int, int]]:
        if target_shapes is None:
            return []
        normalized: list[tuple[int, int]] = []
        for shape in target_shapes:
            if len(shape) != 2:
                raise ValueError(f"target shape 必须是 [height, width]，当前为 {shape!r}")
            height, width = int(shape[0]), int(shape[1])
            if height <= 0 or width <= 0:
                raise ValueError(f"target shape 必须为正整数，当前为 {shape!r}")
            normalized.append((height, width))
        return normalized

    @staticmethod
    def _compact_permutation(count: int, generator: torch.Generator) -> _AffinePermutation:
        count = int(count)
        if count <= 0:
            raise ValueError("紧凑 permutation 的 count 必须为正整数。")
        if count == 1:
            return _AffinePermutation(count=1, offset=0, step=1)
        offset = int(torch.randint(count, (1,), generator=generator).item())
        step = int(torch.randint(1, count, (1,), generator=generator).item())
        while math.gcd(step, count) != 1:
            step += 1
            if step >= count:
                step = 1
        return _AffinePermutation(count=count, offset=offset, step=step)

    def _compact_source(
        self,
        base_count: int,
        resolve: Any,
        weight: float,
        generator: torch.Generator,
    ) -> _CompactEpochSampleSource | None:
        base_count = int(base_count)
        if base_count <= 0 or weight <= 0.0:
            return None
        full_repeats = int(weight)
        fractional_count = int(round((float(weight) - float(full_repeats)) * float(base_count)))
        return self._compact_source_from_repeat_counts(
            base_count,
            resolve,
            full_repeats=full_repeats,
            fractional_count=fractional_count,
            generator=generator,
        )

    def _compact_source_from_repeat_counts(
        self,
        base_count: int,
        resolve: Any,
        *,
        full_repeats: int,
        fractional_count: int,
        generator: torch.Generator,
    ) -> _CompactEpochSampleSource | None:
        base_count = int(base_count)
        full_repeats = int(full_repeats)
        fractional_count = int(fractional_count)
        if base_count <= 0 or full_repeats < 0 or fractional_count < 0 or fractional_count > base_count:
            raise ValueError("紧凑 epoch source 的 repeat 参数无效。")
        if full_repeats == 0 and fractional_count == 0:
            return None
        full_permutations = tuple(
            self._compact_permutation(base_count, generator)
            for _ in range(full_repeats)
        )
        fractional_permutation = (
            self._compact_permutation(base_count, generator)
            if fractional_count > 0
            else None
        )
        return _CompactEpochSampleSource(
            base_count=base_count,
            full_permutations=full_permutations,
            fractional_permutation=fractional_permutation,
            fractional_count=fractional_count,
            resolve=resolve,
        )

    def _local_sampling_enabled_for_dataset(self, dataset_index: int | None) -> bool:
        if not self.container_local_sampling_enabled:
            return False
        if isinstance(self.dataset, MultiSourceRandomLengthTemporalChunkDataset):
            assert dataset_index is not None
            dataset_name = self.dataset.dataset_names[int(dataset_index)]
        else:
            dataset_name = _random_length_dataset_name(self.dataset)
        return dataset_name in self.container_local_sampling_dataset_names

    def _compact_group(
        self,
        sources: list[_CompactEpochSampleSource],
        generator: torch.Generator,
    ) -> _CompactEpochSampleGroup | None:
        sources = [source for source in sources if source.sample_count > 0]
        if not sources:
            return None
        offsets: list[int] = []
        running_total = 0
        for source in sources:
            offsets.append(running_total)
            running_total += source.sample_count
        return _CompactEpochSampleGroup(
            sources=tuple(sources),
            source_offsets=tuple(offsets),
            permutation=self._compact_permutation(running_total, generator),
        )

    def _compact_sequence_window_source(
        self,
        child_dataset: RandomLengthTemporalChunkDataset,
        *,
        num_frames: int,
        frame_strides: Sequence[int],
        target_shape: tuple[int, int],
        dataset_index: int | None,
        weight: float,
        generator: torch.Generator,
    ) -> list[_CompactEpochSampleSource]:
        sources: list[_CompactEpochSampleSource] = []
        target_height, target_width = target_shape
        for frame_stride in frame_strides:
            counts: list[int] = []
            offsets: list[int] = []
            running_total = 0
            for sequence_index in range(child_dataset.sequence_count):
                offsets.append(running_total)
                count = child_dataset.valid_start_count(
                    sequence_index,
                    num_frames=num_frames,
                    frame_stride=frame_stride,
                )
                counts.append(count)
                running_total += count
            if running_total <= 0:
                continue
            offsets_tuple = tuple(offsets)
            counts_tuple = tuple(counts)
            window_stride = int(child_dataset.window_stride)

            def _resolve(
                local_index: int,
                *,
                _offsets=offsets_tuple,
                _counts=counts_tuple,
                _dataset_index=dataset_index,
                _frame_stride=int(frame_stride),
                _num_frames=int(num_frames),
                _target_height=int(target_height),
                _target_width=int(target_width),
                _window_stride=window_stride,
            ) -> tuple[int, ...]:
                sequence_index = bisect_right(_offsets, int(local_index)) - 1
                start_slot = int(local_index) - _offsets[sequence_index]
                if start_slot < 0 or start_slot >= _counts[sequence_index]:
                    raise IndexError(local_index)
                start = start_slot * _window_stride
                if _dataset_index is None:
                    return (sequence_index, start, _num_frames, _frame_stride, _target_height, _target_width)
                return (
                    _dataset_index,
                    sequence_index,
                    start,
                    _num_frames,
                    _frame_stride,
                    _target_height,
                    _target_width,
                )

            source = self._compact_source(running_total, _resolve, weight, generator)
            if source is not None:
                sources.append(source)
        return sources

    @staticmethod
    def _fractional_sequence_repeat_counts(counts: Sequence[int], weight: float) -> tuple[int, list[int]]:
        """Split a dataset-level fractional repeat exactly across sequences."""

        full_repeats = int(weight)
        fractional_weight = float(weight) - float(full_repeats)
        total_count = sum(int(count) for count in counts)
        total_fractional_count = int(round(fractional_weight * float(total_count)))
        allocations = [int(math.floor(fractional_weight * float(count))) for count in counts]
        remaining = total_fractional_count - sum(allocations)
        if remaining > 0:
            fractional_parts = sorted(
                (
                    (fractional_weight * float(count)) - math.floor(fractional_weight * float(count)),
                    index,
                )
                for index, count in enumerate(counts)
            )
            for _, index in reversed(fractional_parts[-remaining:]):
                allocations[index] += 1
        return full_repeats, allocations

    def _compact_local_sequence_window_sources(
        self,
        child_dataset: RandomLengthTemporalChunkDataset,
        *,
        num_frames: int,
        frame_strides: Sequence[int],
        target_shape: tuple[int, int],
        dataset_index: int | None,
        weight: float,
        generator: torch.Generator,
    ) -> list[_CompactEpochSampleSource]:
        sources: list[_CompactEpochSampleSource] = []
        target_height, target_width = target_shape
        for frame_stride in frame_strides:
            sequence_indices: list[int] = []
            counts: list[int] = []
            for sequence_index in range(child_dataset.sequence_count):
                count = child_dataset.valid_start_count(
                    sequence_index,
                    num_frames=num_frames,
                    frame_stride=frame_stride,
                )
                if count > 0:
                    sequence_indices.append(sequence_index)
                    counts.append(count)
            full_repeats, fractional_counts = self._fractional_sequence_repeat_counts(counts, weight)
            for sequence_index, count, fractional_count in zip(sequence_indices, counts, fractional_counts, strict=True):
                window_stride = int(child_dataset.window_stride)

                def _resolve(
                    local_index: int,
                    *,
                    _dataset_index=dataset_index,
                    _sequence_index=int(sequence_index),
                    _frame_stride=int(frame_stride),
                    _num_frames=int(num_frames),
                    _target_height=int(target_height),
                    _target_width=int(target_width),
                    _window_stride=window_stride,
                ) -> tuple[int, ...]:
                    start = int(local_index) * _window_stride
                    if _dataset_index is None:
                        return (_sequence_index, start, _num_frames, _frame_stride, _target_height, _target_width)
                    return (
                        _dataset_index,
                        _sequence_index,
                        start,
                        _num_frames,
                        _frame_stride,
                        _target_height,
                        _target_width,
                    )

                source = self._compact_source_from_repeat_counts(
                    count,
                    _resolve,
                    full_repeats=full_repeats,
                    fractional_count=fractional_count,
                    generator=generator,
                )
                if source is not None:
                    sources.append(source)
        return sources

    def _compact_local_sequence_anchor_sources(
        self,
        child_dataset: RandomLengthTemporalChunkDataset,
        *,
        num_frames: int,
        dataset_index: int | None,
        weight: float,
        generator: torch.Generator,
    ) -> list[_CompactEpochSampleSource]:
        sequence_indices: list[int] = []
        counts: list[int] = []
        for sequence_index in range(child_dataset.sequence_count):
            count = child_dataset.valid_start_count(
                sequence_index,
                num_frames=num_frames,
                frame_stride=1,
            )
            if count > 0:
                sequence_indices.append(sequence_index)
                counts.append(count)
        full_repeats, fractional_counts = self._fractional_sequence_repeat_counts(counts, weight)
        sources: list[_CompactEpochSampleSource] = []
        for sequence_index, count, fractional_count in zip(sequence_indices, counts, fractional_counts, strict=True):
            def _resolve(
                local_index: int,
                *,
                _dataset_index=dataset_index,
                _sequence_index=int(sequence_index),
                _num_frames=int(num_frames),
                _dataset=child_dataset,
            ) -> tuple[int, ...]:
                anchor_index = _dataset.global_anchor_index(_sequence_index, int(local_index), _num_frames)
                if _dataset_index is None:
                    return (anchor_index, _num_frames)
                return (_dataset_index, anchor_index, _num_frames)

            source = self._compact_source_from_repeat_counts(
                count,
                _resolve,
                full_repeats=full_repeats,
                fractional_count=fractional_count,
                generator=generator,
            )
            if source is not None:
                sources.append(source)
        return sources

    def _compact_local_block_batch_source(
        self,
        sources: list[_CompactEpochSampleSource],
        generator: torch.Generator,
    ) -> _CompactEpochLocalBlockBatchSource | None:
        sources = [source for source in sources if source.sample_count > 0]
        if not sources:
            return None
        source_block_offsets: list[int] = []
        block_sample_count = self.container_local_sampling_block_batches * self.batch_size
        running_total = 0
        for source in sources:
            source_block_offsets.append(running_total)
            running_total += (source.sample_count + block_sample_count - 1) // block_sample_count
        return _CompactEpochLocalBlockBatchSource(
            sources=tuple(sources),
            source_block_offsets=tuple(source_block_offsets),
            block_batches=self.container_local_sampling_block_batches,
            batch_size=self.batch_size,
            permutation=self._compact_permutation(running_total, generator),
        )

    def _build_compact_epoch_plan(
        self,
    ) -> _CompactEpochBatchPlan:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        groups: list[_CompactEpochSampleGroup] = []
        batch_sources: list[Any] = []
        total_sample_count = 0
        is_multi_source = isinstance(self.dataset, MultiSourceRandomLengthTemporalChunkDataset)
        if self.target_shapes:
            for shape_group in self._shape_frame_stride_groups():
                num_frames = self._shape_group_num_frames(shape_group)
                frame_strides = self._shape_group_frame_strides(shape_group)
                target_shape = self._shape_group_target_shape(shape_group)
                standard_sources: list[_CompactEpochSampleSource] = []
                local_sources: list[_CompactEpochSampleSource] = []
                children = self.dataset.datasets if is_multi_source else [self.dataset]
                for dataset_index, child_dataset in enumerate(children):
                    dataset_weight = (
                        self.dataset._dataset_weight(
                            self.dataset.dataset_names[dataset_index],
                            self.dataset_sampling_weights,
                        )
                        if is_multi_source
                        else 1.0
                    )
                    source_dataset_index = dataset_index if is_multi_source else None
                    if self._local_sampling_enabled_for_dataset(source_dataset_index):
                        local_sources.extend(
                            self._compact_local_sequence_window_sources(
                                child_dataset,
                                num_frames=num_frames,
                                frame_strides=frame_strides,
                                target_shape=target_shape,
                                dataset_index=source_dataset_index,
                                weight=dataset_weight,
                                generator=generator,
                            )
                        )
                    else:
                        standard_sources.extend(
                            self._compact_sequence_window_source(
                                child_dataset,
                                num_frames=num_frames,
                                frame_strides=frame_strides,
                                target_shape=target_shape,
                                dataset_index=source_dataset_index,
                                weight=dataset_weight,
                                generator=generator,
                            )
                        )
                group = self._compact_group(standard_sources, generator)
                if group is not None:
                    groups.append(group)
                    batch_sources.append(_CompactEpochBatchSource(group=group, batch_size=self.batch_size))
                    total_sample_count += group.sample_count
                local_source = self._compact_local_block_batch_source(local_sources, generator)
                if local_source is not None:
                    batch_sources.append(local_source)
                    total_sample_count += sum(source.sample_count for source in local_sources)
        else:
            for num_frames in range(self.dataset.min_num_frames, self.dataset.max_num_frames + 1):
                standard_sources: list[_CompactEpochSampleSource] = []
                local_sources: list[_CompactEpochSampleSource] = []
                children = self.dataset.datasets if is_multi_source else [self.dataset]
                for dataset_index, child_dataset in enumerate(children):
                    dataset_weight = (
                        self.dataset._dataset_weight(
                            self.dataset.dataset_names[dataset_index],
                            self.dataset_sampling_weights,
                        )
                        if is_multi_source
                        else 1.0
                    )
                    source_dataset_index = dataset_index if is_multi_source else None
                    if self._local_sampling_enabled_for_dataset(source_dataset_index):
                        local_sources.extend(
                            self._compact_local_sequence_anchor_sources(
                                child_dataset,
                                num_frames=num_frames,
                                dataset_index=source_dataset_index,
                                weight=dataset_weight,
                                generator=generator,
                            )
                        )
                        continue
                    base_count = child_dataset.valid_anchor_count(num_frames)
                    if base_count <= 0:
                        continue

                    def _resolve(
                        local_index: int,
                        *,
                        _dataset_index=source_dataset_index,
                        _num_frames=int(num_frames),
                    ) -> tuple[int, ...]:
                        if _dataset_index is None:
                            return (int(local_index), _num_frames)
                        return (_dataset_index, int(local_index), _num_frames)

                    source = self._compact_source(base_count, _resolve, dataset_weight, generator)
                    if source is not None:
                        standard_sources.append(source)
                group = self._compact_group(standard_sources, generator)
                if group is not None:
                    groups.append(group)
                    batch_sources.append(_CompactEpochBatchSource(group=group, batch_size=self.batch_size))
                    total_sample_count += group.sample_count
                local_source = self._compact_local_block_batch_source(local_sources, generator)
                if local_source is not None:
                    batch_sources.append(local_source)
                    total_sample_count += sum(source.sample_count for source in local_sources)
        if not batch_sources:
            return _CompactEpochBatchPlan(
                groups=tuple(groups),
                sources=(),
                source_unit_offsets=(),
                unit_permutation=None,
                batch_count=0,
                sample_count=0,
            )
        source_unit_offsets: list[int] = []
        running_units = 0
        batch_count = 0
        for source in batch_sources:
            source_unit_offsets.append(running_units)
            running_units += source.unit_count
            batch_count += source.batch_count
        return _CompactEpochBatchPlan(
            groups=tuple(groups),
            sources=tuple(batch_sources),
            source_unit_offsets=tuple(source_unit_offsets),
            unit_permutation=self._compact_permutation(running_units, generator),
            batch_count=batch_count,
            sample_count=total_sample_count,
        )

    def _shape_frame_stride_groups(self) -> list[tuple[int, ...]]:
        groups: list[tuple[int, ...]] = []
        for num_frames in range(self.dataset.min_num_frames, self.dataset.max_num_frames + 1):
            for target_height, target_width in self.target_shapes:
                if self.group_frame_stride_in_batch:
                    groups.extend(
                        (num_frames, frame_stride, target_height, target_width)
                        for frame_stride in self.frame_strides
                    )
                else:
                    groups.append((num_frames, target_height, target_width))
        return groups

    def _shape_group_num_frames(self, group: tuple[int, ...]) -> int:
        return int(group[0])

    def _shape_group_frame_strides(self, group: tuple[int, ...]) -> list[int]:
        if self.group_frame_stride_in_batch:
            return [int(group[1])]
        return list(self.frame_strides)

    def _shape_group_target_shape(self, group: tuple[int, ...]) -> tuple[int, int]:
        if self.group_frame_stride_in_batch:
            return int(group[2]), int(group[3])
        return int(group[1]), int(group[2])

    def _child_shape_samples(
        self,
        child_dataset: RandomLengthTemporalChunkDataset,
        group: tuple[int, ...],
    ) -> list[tuple[int, int, int, int, int, int]]:
        num_frames = self._shape_group_num_frames(group)
        target_height, target_width = self._shape_group_target_shape(group)
        samples: list[tuple[int, int, int, int, int, int]] = []
        for frame_stride in self._shape_group_frame_strides(group):
            for sequence_index in range(child_dataset.sequence_count):
                valid_start_count = child_dataset.valid_start_count(
                    sequence_index,
                    num_frames=num_frames,
                    frame_stride=frame_stride,
                )
                for start_slot in range(valid_start_count):
                    samples.append(
                        (
                            sequence_index,
                            start_slot * int(child_dataset.window_stride),
                            num_frames,
                            frame_stride,
                            target_height,
                            target_width,
                        )
                    )
        return samples

    def _weighted_child_shape_samples(
        self,
        *,
        dataset_index: int,
        dataset_name: str,
        child_dataset: RandomLengthTemporalChunkDataset,
        group: tuple[int, ...],
        generator: torch.Generator,
    ) -> list[tuple[int, int, int, int, int, int, int]]:
        base_samples = [
            (dataset_index, *sample)
            for sample in self._child_shape_samples(child_dataset, group)
        ]
        if not base_samples:
            return []
        weight = self.dataset._dataset_weight(dataset_name, self.dataset_sampling_weights)
        full_repeats = int(weight)
        fractional_repeat_count = int(round((weight - float(full_repeats)) * float(len(base_samples))))
        outputs: list[tuple[int, int, int, int, int, int, int]] = []
        for _ in range(full_repeats):
            outputs.extend(base_samples)
        if fractional_repeat_count > 0:
            permutation = torch.randperm(len(base_samples), generator=generator).tolist()
            outputs.extend(base_samples[index] for index in permutation[:fractional_repeat_count])
        return outputs

    def _sticky_sequence_epoch_samples_for_length(
        self,
        *,
        dataset_index: int,
        dataset_name: str,
        child_dataset: RandomLengthTemporalChunkDataset,
        num_frames: int,
        generator: torch.Generator,
    ) -> list[tuple[int, int, int]]:
        weight = self.dataset._dataset_weight(dataset_name, self.dataset_sampling_weights)
        sequence_counts = child_dataset.sequence_anchor_counts(num_frames)
        sequence_blocks: list[list[tuple[int, int, int]]] = []
        for sequence_index, anchor_count in enumerate(sequence_counts):
            if anchor_count <= 0:
                continue
            anchors = [
                (
                    dataset_index,
                    child_dataset.global_anchor_index(sequence_index, local_anchor_index, num_frames),
                    num_frames,
                )
                for local_anchor_index in range(anchor_count)
            ]
            permutation = torch.randperm(len(anchors), generator=generator).tolist()
            sequence_blocks.append([anchors[index] for index in permutation])
        if not sequence_blocks:
            return []

        outputs: list[tuple[int, int, int]] = []
        full_repeats = int(weight)
        fractional_repeat_count = int(round((weight - float(full_repeats)) * float(sum(len(block) for block in sequence_blocks))))

        for _ in range(full_repeats):
            block_permutation = torch.randperm(len(sequence_blocks), generator=generator).tolist()
            for block_index in block_permutation:
                outputs.extend(sequence_blocks[block_index])
        if fractional_repeat_count > 0:
            block_permutation = torch.randperm(len(sequence_blocks), generator=generator).tolist()
            flattened = [sample for block_index in block_permutation for sample in sequence_blocks[block_index]]
            outputs.extend(flattened[:fractional_repeat_count])
        return outputs

    def _build_epoch_batches(self) -> list[list[Any]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        grouped_samples: list[list[Any]] = []

        if self.target_shapes and isinstance(self.dataset, MultiSourceRandomLengthTemporalChunkDataset):
            for group in self._shape_frame_stride_groups():
                length_group: list[tuple[int, int, int, int, int, int, int]] = []
                for dataset_index, (dataset_name, child_dataset) in enumerate(
                    zip(self.dataset.dataset_names, self.dataset.datasets, strict=True)
                ):
                    length_group.extend(
                        self._weighted_child_shape_samples(
                            dataset_index=dataset_index,
                            dataset_name=dataset_name,
                            child_dataset=child_dataset,
                            group=group,
                            generator=generator,
                        )
                    )
                if length_group:
                    permutation = torch.randperm(len(length_group), generator=generator).tolist()
                    grouped_samples.append([length_group[index] for index in permutation])
        elif self.target_shapes:
            for group in self._shape_frame_stride_groups():
                length_group = self._child_shape_samples(self.dataset, group)
                if length_group:
                    permutation = torch.randperm(len(length_group), generator=generator).tolist()
                    grouped_samples.append([length_group[index] for index in permutation])
        elif isinstance(self.dataset, MultiSourceRandomLengthTemporalChunkDataset):
            for num_frames in range(self.dataset.min_num_frames, self.dataset.max_num_frames + 1):
                length_group: list[tuple[int, int, int]] = []
                for dataset_index, (dataset_name, child_dataset) in enumerate(
                    zip(self.dataset.dataset_names, self.dataset.datasets, strict=True)
                ):
                    valid_count = child_dataset.valid_anchor_count(num_frames)
                    if valid_count <= 0:
                        continue
                    if dataset_name in _STICKY_SEQUENCE_DATASET_NAMES:
                        length_group.extend(
                            self._sticky_sequence_epoch_samples_for_length(
                                dataset_index=dataset_index,
                                dataset_name=dataset_name,
                                child_dataset=child_dataset,
                                num_frames=num_frames,
                                generator=generator,
                            )
                        )
                        continue
                    weight = self.dataset._dataset_weight(dataset_name, self.dataset_sampling_weights)
                    base_samples = [(dataset_index, anchor_index, num_frames) for anchor_index in range(valid_count)]
                    full_repeats = int(weight)
                    fractional_repeat_count = int(round((weight - float(full_repeats)) * float(len(base_samples))))
                    for _ in range(full_repeats):
                        length_group.extend(base_samples)
                    if fractional_repeat_count > 0:
                        permutation = torch.randperm(len(base_samples), generator=generator).tolist()
                        length_group.extend(base_samples[index] for index in permutation[:fractional_repeat_count])
                if length_group:
                    permutation = torch.randperm(len(length_group), generator=generator).tolist()
                    shuffled_group = [length_group[index] for index in permutation]
                    grouped_samples.append(shuffled_group)
        else:
            for num_frames in range(self.dataset.min_num_frames, self.dataset.max_num_frames + 1):
                valid_count = self.dataset.valid_anchor_count(num_frames)
                if valid_count <= 0:
                    continue
                length_group = [(anchor_index, num_frames) for anchor_index in range(valid_count)]
                permutation = torch.randperm(len(length_group), generator=generator).tolist()
                shuffled_group = [length_group[index] for index in permutation]
                grouped_samples.append(shuffled_group)

        batches: list[list[Any]] = []
        for group in grouped_samples:
            for start in range(0, len(group), self.batch_size):
                batches.append(group[start : start + self.batch_size])
        if not batches:
            return batches
        permutation = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[index] for index in permutation]

    def __len__(self) -> int:
        return self._epoch_plan.batch_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._epoch_plan = self._build_compact_epoch_plan()
        self._epoch_groups = list(self._epoch_plan.groups)
        self.raw_epoch_sample_count = self._epoch_plan.sample_count
        self.used_epoch_sample_count = self.raw_epoch_sample_count

    def __iter__(self):
        yield from self._epoch_plan.iter_batches()


class WeightedRandomLengthBatchSampler:
    _STICKY_SEQUENCE_SPAN = 256
    is_streaming_sampler = True

    def __init__(
        self,
        dataset: MultiSourceRandomLengthTemporalChunkDataset,
        *,
        batch_size: int,
        seed: int,
        dataset_sampling_weights: dict[str, float] | None = None,
        sticky_sequence_slots: int = 1,
        ddp_batch_group_size: int = 1,
        target_shapes: Sequence[Sequence[int]] | None = None,
        min_frame_stride: int = 1,
        max_frame_stride: int = 1,
        group_frame_stride_in_batch: bool = False,
        sequence_sampling_mode: str = "window_proportional",
        container_local_sampling_enabled: bool = False,
        container_local_sampling_dataset_names: Sequence[str] | None = None,
        container_local_sampling_block_batches: int = 64,
        container_local_sampling_start_order: str = "random_cursor",
        container_local_prefetch_enabled: bool = False,
        container_local_prefetch_margin_batches: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        if sticky_sequence_slots <= 0:
            raise ValueError("sticky_sequence_slots 必须为正整数。")
        if ddp_batch_group_size <= 0:
            raise ValueError("ddp_batch_group_size 必须为正整数。")
        if min_frame_stride <= 0 or max_frame_stride <= 0:
            raise ValueError("frame_stride 必须为正整数。")
        if min_frame_stride > max_frame_stride:
            raise ValueError("min_frame_stride 不能大于 max_frame_stride。")
        if sequence_sampling_mode not in {"window_proportional", "uniform"}:
            raise ValueError("sequence_sampling_mode 必须是 'window_proportional' 或 'uniform'。")
        if container_local_sampling_block_batches <= 0:
            raise ValueError("container_local_sampling_block_batches 必须为正整数。")
        if container_local_sampling_start_order not in _CONTAINER_LOCAL_SAMPLING_START_ORDERS:
            raise ValueError("container_local_sampling_start_order 必须是 sequential 或 random_cursor。")
        if container_local_prefetch_enabled:
            _validate_container_local_prefetch_margin(
                container_local_sampling_block_batches,
                container_local_prefetch_margin_batches,
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_sampling_weights = dataset_sampling_weights or {}
        self.sticky_sequence_slots = int(sticky_sequence_slots)
        self.ddp_batch_group_size = int(ddp_batch_group_size)
        self.target_shapes = self._normalize_target_shapes(target_shapes)
        self.frame_strides = list(range(int(min_frame_stride), int(max_frame_stride) + 1))
        self.group_frame_stride_in_batch = bool(group_frame_stride_in_batch)
        self.sequence_sampling_mode = sequence_sampling_mode
        self.container_local_sampling_enabled = bool(container_local_sampling_enabled)
        self.container_local_sampling_dataset_names = _normalize_dataset_name_set(container_local_sampling_dataset_names)
        self.container_local_sampling_block_batches = int(container_local_sampling_block_batches)
        self.container_local_sampling_start_order = str(container_local_sampling_start_order)
        self.container_local_prefetch_enabled = bool(container_local_prefetch_enabled)
        self.container_local_prefetch_margin_batches = int(container_local_prefetch_margin_batches)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self._sticky_sequence_state: dict[tuple[int, int], tuple[int, int]] = {}
        self._start_cursor_state: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
        self._local_sequence_state: dict[int, _LocalSequenceState] = {}
        self._local_prefetch_sequence_state: dict[int, _LocalSequenceState] = {}
        self._local_start_cursor_state: dict[tuple[int, int, int, int, int], int] = {}
        self.valid_lengths = [
            num_frames
            for num_frames in range(self.dataset.min_num_frames, self.dataset.max_num_frames + 1)
            if self._dataset_probabilities_for_length(num_frames) is not None
        ]
        self.valid_buckets = self._build_valid_buckets() if self.target_shapes else []
        if not self.valid_lengths and not self.valid_buckets:
            raise ValueError("随机长度多源数据集没有有效窗口。")

    def __len__(self) -> int:
        if self.target_shapes:
            total_epoch_samples = 0.0
            for bucket in self.valid_buckets:
                for dataset_index, child_dataset in enumerate(self.dataset.datasets):
                    dataset_weight = self._dataset_weight(self.dataset.dataset_names[dataset_index])
                    for sequence_index in range(child_dataset.sequence_count):
                        total_epoch_samples += dataset_weight * float(
                            self._sequence_valid_start_count(dataset_index, sequence_index, bucket)
                        )
            if total_epoch_samples <= 0:
                return 0
            return max(1, int(total_epoch_samples) // self.batch_size)
        total_epoch_samples = sum(self.dataset.weighted_epoch_sample_counts(self.dataset_sampling_weights).values())
        if total_epoch_samples <= 0:
            return 0
        return max(1, total_epoch_samples // self.batch_size)

    def _normalize_target_shapes(self, target_shapes: Sequence[Sequence[int]] | None) -> list[tuple[int, int]]:
        if target_shapes is None:
            return []
        normalized: list[tuple[int, int]] = []
        for shape in target_shapes:
            if len(shape) != 2:
                raise ValueError(f"target shape 必须是 [height, width]，当前为 {shape!r}")
            height, width = int(shape[0]), int(shape[1])
            if height <= 0 or width <= 0:
                raise ValueError(f"target shape 必须为正整数，当前为 {shape!r}")
            normalized.append((height, width))
        return normalized

    def _dataset_weight(self, dataset_name: str) -> float:
        weight = float(self.dataset_sampling_weights.get(dataset_name, 1.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"dataset_sampling_weights[{dataset_name!r}] 必须是有限的非负数，当前为 {weight}.")
        return weight

    def _bucket_num_frames(self, bucket: tuple[int, ...]) -> int:
        return int(bucket[0])

    def _bucket_frame_stride(self, bucket: tuple[int, ...]) -> int | None:
        return int(bucket[1]) if self.group_frame_stride_in_batch else None

    def _bucket_shape(self, bucket: tuple[int, ...]) -> tuple[int, int]:
        if self.group_frame_stride_in_batch:
            return int(bucket[2]), int(bucket[3])
        return int(bucket[1]), int(bucket[2])

    def _sequence_valid_start_count(self, dataset_index: int, sequence_index: int, bucket: tuple[int, ...]) -> int:
        child_dataset = self.dataset.datasets[dataset_index]
        num_frames = self._bucket_num_frames(bucket)
        bucket_frame_stride = self._bucket_frame_stride(bucket)
        if bucket_frame_stride is not None:
            return child_dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=bucket_frame_stride)
        return sum(
            child_dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=frame_stride)
            for frame_stride in self.frame_strides
        )

    def _build_valid_buckets(self) -> list[tuple[int, ...]]:
        buckets: list[tuple[int, ...]] = []
        for num_frames in range(self.dataset.min_num_frames, self.dataset.max_num_frames + 1):
            for target_height, target_width in self.target_shapes:
                if self.group_frame_stride_in_batch:
                    for frame_stride in self.frame_strides:
                        if any(
                            child_dataset.valid_start_count(
                                sequence_index,
                                num_frames=num_frames,
                                frame_stride=frame_stride,
                            )
                            > 0
                            for child_dataset in self.dataset.datasets
                            for sequence_index in range(child_dataset.sequence_count)
                        ):
                            buckets.append((num_frames, frame_stride, target_height, target_width))
                    continue
                bucket = (num_frames, target_height, target_width)
                if any(
                    self._sequence_valid_start_count(dataset_index, sequence_index, bucket) > 0
                    for dataset_index, child_dataset in enumerate(self.dataset.datasets)
                    for sequence_index in range(child_dataset.sequence_count)
                ):
                    buckets.append(bucket)
        return buckets

    def _dataset_probabilities_for_length(self, num_frames: int) -> tuple[list[int], torch.Tensor] | None:
        active_indices: list[int] = []
        probabilities: list[float] = []
        for dataset_index, dataset_name in enumerate(self.dataset.dataset_names):
            count = self.dataset.valid_anchor_count(dataset_index, num_frames)
            if count <= 0:
                continue
            weight = self._dataset_weight(dataset_name)
            weighted_count = weight * float(count)
            active_indices.append(dataset_index)
            probabilities.append(weighted_count)
        if not active_indices:
            return None
        probability_tensor = torch.tensor(probabilities, dtype=torch.float64)
        probability_tensor /= probability_tensor.sum()
        return active_indices, probability_tensor

    def _sample_anchor_for_dataset(self, dataset_index: int, num_frames: int, *, sequence_slot: int = 0) -> tuple[int, int, int]:
        dataset_name = self.dataset.dataset_names[dataset_index]
        if dataset_name not in _STICKY_SEQUENCE_DATASET_NAMES:
            valid_count = self.dataset.valid_anchor_count(dataset_index, num_frames)
            anchor_index = int(torch.randint(valid_count, (1,), generator=self.generator).item())
            return (dataset_index, anchor_index, num_frames)

        child_dataset = self.dataset.datasets[dataset_index]
        sequence_counts = child_dataset.sequence_anchor_counts(num_frames)
        sticky_key = (dataset_index, int(sequence_slot))
        sticky_state = self._sticky_sequence_state.get(sticky_key)
        if sticky_state is not None:
            sticky_sequence_index, sticky_remaining = sticky_state
            if sticky_remaining > 0 and sticky_sequence_index < len(sequence_counts) and sequence_counts[sticky_sequence_index] > 0:
                local_anchor_index = int(torch.randint(sequence_counts[sticky_sequence_index], (1,), generator=self.generator).item())
                self._sticky_sequence_state[sticky_key] = (sticky_sequence_index, sticky_remaining - 1)
                return (
                    dataset_index,
                    child_dataset.global_anchor_index(sticky_sequence_index, local_anchor_index, num_frames),
                    num_frames,
                )

        active_sequence_indices = [index for index, count in enumerate(sequence_counts) if count > 0]
        if not active_sequence_indices:
            valid_count = self.dataset.valid_anchor_count(dataset_index, num_frames)
            anchor_index = int(torch.randint(valid_count, (1,), generator=self.generator).item())
            return (dataset_index, anchor_index, num_frames)
        active_sequence_set = set(active_sequence_indices)
        used_sequence_indices = {
            sequence_index
            for (state_dataset_index, _), (sequence_index, sticky_remaining) in self._sticky_sequence_state.items()
            if state_dataset_index == dataset_index
            and sticky_remaining > 0
            and sequence_index in active_sequence_set
            and sequence_counts[sequence_index] > 0
        }
        available_sequence_indices = [
            sequence_index for sequence_index in active_sequence_indices if sequence_index not in used_sequence_indices
        ]
        if available_sequence_indices:
            active_sequence_indices = available_sequence_indices
        probabilities = torch.tensor(
            [float(sequence_counts[index]) for index in active_sequence_indices],
            dtype=torch.float64,
        )
        probabilities /= probabilities.sum()
        sampled_sequence_position = int(torch.multinomial(probabilities, num_samples=1, replacement=True, generator=self.generator).item())
        sampled_sequence_index = active_sequence_indices[sampled_sequence_position]
        local_anchor_index = int(torch.randint(sequence_counts[sampled_sequence_index], (1,), generator=self.generator).item())
        self._sticky_sequence_state[sticky_key] = (sampled_sequence_index, self._STICKY_SEQUENCE_SPAN - 1)
        return (
            dataset_index,
            child_dataset.global_anchor_index(sampled_sequence_index, local_anchor_index, num_frames),
            num_frames,
        )

    def _weighted_sample_positions(self, weights: torch.Tensor, count: int, *, replacement: bool) -> list[int]:
        if count <= 0:
            return []
        return torch.multinomial(weights, num_samples=count, replacement=replacement, generator=self.generator).tolist()

    def _bucket_candidates_and_weights(
        self,
        bucket: tuple[int, ...],
        *,
        include_local: bool = True,
        include_nonlocal: bool = True,
    ) -> tuple[list[tuple[int, int]], list[float]]:
        active_candidates: list[tuple[int, int]] = []
        weights: list[float] = []
        for dataset_index, child_dataset in enumerate(self.dataset.datasets):
            is_local_dataset = self._container_local_sampling_enabled_for_dataset(dataset_index)
            if is_local_dataset and not include_local:
                continue
            if not is_local_dataset and not include_nonlocal:
                continue
            dataset_weight = self._dataset_weight(self.dataset.dataset_names[dataset_index])
            for sequence_index in range(child_dataset.sequence_count):
                valid_count = self._sequence_valid_start_count(dataset_index, sequence_index, bucket)
                if valid_count <= 0:
                    continue
                active_candidates.append((dataset_index, sequence_index))
                sequence_weight = 1.0 if self.sequence_sampling_mode == "uniform" else float(valid_count)
                weights.append(dataset_weight * sequence_weight)
        return active_candidates, weights

    def _sample_bucket_candidates(
        self,
        bucket: tuple[int, ...],
        *,
        include_local: bool = True,
        include_nonlocal: bool = True,
    ) -> list[tuple[int, int]]:
        active_candidates, weights = self._bucket_candidates_and_weights(
            bucket,
            include_local=include_local,
            include_nonlocal=include_nonlocal,
        )
        if not active_candidates:
            raise IndexError("当前 bucket 没有 eligible sequence。")
        weight_tensor = torch.tensor(weights, dtype=torch.float64)
        if len(active_candidates) >= self.batch_size:
            positions = self._weighted_sample_positions(weight_tensor, self.batch_size, replacement=False)
            return [active_candidates[position] for position in positions]
        positions = self._weighted_sample_positions(weight_tensor, len(active_candidates), replacement=False)
        sampled = [active_candidates[position] for position in positions]
        extra_positions = self._weighted_sample_positions(weight_tensor, self.batch_size - len(sampled), replacement=True)
        sampled.extend(active_candidates[position] for position in extra_positions)
        return sampled

    def _sample_frame_stride(self, dataset_index: int, sequence_index: int, bucket: tuple[int, ...]) -> int:
        bucket_frame_stride = self._bucket_frame_stride(bucket)
        if bucket_frame_stride is not None:
            return bucket_frame_stride
        child_dataset = self.dataset.datasets[dataset_index]
        num_frames = self._bucket_num_frames(bucket)
        active_strides: list[int] = []
        weights: list[float] = []
        for frame_stride in self.frame_strides:
            valid_count = child_dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=frame_stride)
            if valid_count <= 0:
                continue
            active_strides.append(frame_stride)
            weights.append(1.0 if self.sequence_sampling_mode == "uniform" else float(valid_count))
        if not active_strides:
            raise IndexError("当前 sequence 没有 eligible frame_stride。")
        weight_tensor = torch.tensor(weights, dtype=torch.float64)
        position = self._weighted_sample_positions(weight_tensor, 1, replacement=True)[0]
        return active_strides[position]

    def _sample_coprime_jump(self, valid_start_count: int) -> int:
        if valid_start_count <= 1:
            return 1
        start = int(torch.randint(valid_start_count, (1,), generator=self.generator).item())
        for offset in range(valid_start_count):
            candidate = 1 + ((start + offset) % valid_start_count)
            if math.gcd(candidate, valid_start_count) == 1:
                return candidate
        return 1

    def _next_start(self, dataset_index: int, sequence_index: int, num_frames: int, frame_stride: int) -> int:
        child_dataset = self.dataset.datasets[dataset_index]
        valid_start_count = child_dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=frame_stride)
        if valid_start_count <= 0:
            raise IndexError((dataset_index, sequence_index, num_frames, frame_stride))
        key = (int(dataset_index), int(sequence_index), int(num_frames), int(frame_stride))
        state = self._start_cursor_state.get(key)
        if state is None:
            offset = int(torch.randint(valid_start_count, (1,), generator=self.generator).item())
            jump = self._sample_coprime_jump(valid_start_count)
            cursor = 0
        else:
            offset, jump, cursor = state
        start_slot = (offset + cursor * jump) % valid_start_count
        self._start_cursor_state[key] = (offset, jump, cursor + 1)
        return int(start_slot) * int(child_dataset.window_stride)

    def _container_local_sampling_enabled_for_dataset(self, dataset_index: int) -> bool:
        if not self.container_local_sampling_enabled:
            return False
        dataset_name = self.dataset.dataset_names[int(dataset_index)]
        return dataset_name in self.container_local_sampling_dataset_names

    def _locality_slot(self, batch_index: int) -> int:
        return int(batch_index) % max(int(self.ddp_batch_group_size), 1)

    def _next_local_start(self, slot: int, dataset_index: int, sequence_index: int, num_frames: int, frame_stride: int) -> int:
        if self.container_local_sampling_start_order == "random_cursor":
            return self._next_start(dataset_index, sequence_index, num_frames, frame_stride)
        child_dataset = self.dataset.datasets[dataset_index]
        valid_start_count = child_dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=frame_stride)
        if valid_start_count <= 0:
            raise IndexError((dataset_index, sequence_index, num_frames, frame_stride))
        key = (int(slot), int(dataset_index), int(sequence_index), int(num_frames), int(frame_stride))
        start_slot = self._local_start_cursor_state.get(key)
        if start_slot is None:
            start_slot = int(torch.randint(valid_start_count, (1,), generator=self.generator).item())
        output_slot = int(start_slot) % int(valid_start_count)
        self._local_start_cursor_state[key] = (output_slot + 1) % int(valid_start_count)
        return output_slot * int(child_dataset.window_stride)

    def _local_state_valid(self, state: _LocalSequenceState, bucket: tuple[int, ...]) -> bool:
        if state.remaining_batches <= 0:
            return False
        if state.dataset_index < 0:
            return False
        if not self._container_local_sampling_enabled_for_dataset(state.dataset_index):
            return False
        return self._sequence_valid_start_count(state.dataset_index, state.sequence_index, bucket) > 0

    def _nonlocal_state_valid(self, state: _LocalSequenceState, bucket: tuple[int, ...]) -> bool:
        if state.dataset_index >= 0 or state.remaining_batches <= 0:
            return False
        nonlocal_candidates, _ = self._bucket_candidates_and_weights(
            bucket,
            include_local=False,
            include_nonlocal=True,
        )
        return bool(nonlocal_candidates)

    def _sample_local_sequence_state(
        self,
        bucket: tuple[int, ...],
        slot: int,
        *,
        excluded_candidates: set[tuple[int, int]] | None = None,
    ) -> _LocalSequenceState:
        local_candidates, local_weights = self._bucket_candidates_and_weights(
            bucket,
            include_local=True,
            include_nonlocal=False,
        )
        if not local_candidates:
            raise IndexError("当前 bucket 没有 eligible local sequence。")
        used_sequences = {
            (state.dataset_index, state.sequence_index)
            for state_slot, state in self._local_sequence_state.items()
            if state_slot != int(slot) and self._local_state_valid(state, bucket)
        }
        if excluded_candidates:
            used_sequences.update((int(dataset_index), int(sequence_index)) for dataset_index, sequence_index in excluded_candidates)
        available = [
            (candidate, weight)
            for candidate, weight in zip(local_candidates, local_weights, strict=True)
            if candidate not in used_sequences
        ]
        if available:
            local_candidates = [candidate for candidate, _ in available]
            local_weights = [weight for _, weight in available]
        position = self._weighted_sample_positions(
            torch.tensor(local_weights, dtype=torch.float64),
            1,
            replacement=True,
        )[0]
        dataset_index, sequence_index = local_candidates[position]
        return _LocalSequenceState(
            dataset_index=int(dataset_index),
            sequence_index=int(sequence_index),
            remaining_batches=self.container_local_sampling_block_batches,
        )

    def _local_prefetch_hint(
        self,
        bucket: tuple[int, ...],
        slot: int,
        state: _LocalSequenceState,
    ) -> ContainerPrefetchHint | None:
        if not self.container_local_prefetch_enabled:
            return None
        if state.remaining_batches > self.container_local_prefetch_margin_batches:
            return None
        next_state = self._local_prefetch_sequence_state.get(slot)
        if next_state is None or not self._local_state_valid(next_state, bucket):
            next_state = self._sample_local_sequence_state(
                bucket,
                slot,
                excluded_candidates={(state.dataset_index, state.sequence_index)},
            )
            self._local_prefetch_sequence_state[slot] = next_state
        return ContainerPrefetchHint(dataset_index=next_state.dataset_index, sequence_index=next_state.sequence_index)

    def _should_use_local_sampling_for_bucket(self, bucket: tuple[int, ...]) -> bool:
        if not self.container_local_sampling_enabled or not self.container_local_sampling_dataset_names:
            return False
        local_candidates, local_weights = self._bucket_candidates_and_weights(
            bucket,
            include_local=True,
            include_nonlocal=False,
        )
        if not local_candidates:
            return False
        nonlocal_candidates, nonlocal_weights = self._bucket_candidates_and_weights(
            bucket,
            include_local=False,
            include_nonlocal=True,
        )
        if not nonlocal_candidates:
            return True
        weights = torch.tensor([sum(local_weights), sum(nonlocal_weights)], dtype=torch.float64)
        position = self._weighted_sample_positions(weights, 1, replacement=True)[0]
        return position == 0

    def _sample_standard_bucketed_batch(
        self,
        bucket: tuple[int, ...],
        *,
        include_local: bool,
    ) -> list[tuple[int, int, int, int, int, int, int]]:
        num_frames = self._bucket_num_frames(bucket)
        target_height, target_width = self._bucket_shape(bucket)
        candidates = self._sample_bucket_candidates(bucket, include_local=include_local, include_nonlocal=True)
        batch: list[tuple[int, int, int, int, int, int, int]] = []
        for dataset_index, sequence_index in candidates:
            frame_stride = self._sample_frame_stride(dataset_index, sequence_index, bucket)
            start = self._next_start(dataset_index, sequence_index, num_frames, frame_stride)
            batch.append((dataset_index, sequence_index, start, num_frames, frame_stride, target_height, target_width))
        return batch

    def _sample_local_bucketed_batch(self, bucket: tuple[int, ...], batch_index: int) -> list[tuple[int, int, int, int, int, int, int]]:
        slot = self._locality_slot(batch_index)
        num_frames = self._bucket_num_frames(bucket)
        target_height, target_width = self._bucket_shape(bucket)
        state = self._local_sequence_state.get(slot)
        if state is None or not self._local_state_valid(state, bucket):
            prefetched_state = self._local_prefetch_sequence_state.pop(slot, None)
            if prefetched_state is not None and self._local_state_valid(prefetched_state, bucket):
                state = prefetched_state
            else:
                state = self._sample_local_sequence_state(bucket, slot)
        batch: list[tuple[int, int, int, int, int, int, int]] = []
        for _ in range(self.batch_size):
            frame_stride = self._sample_frame_stride(state.dataset_index, state.sequence_index, bucket)
            start = self._next_local_start(slot, state.dataset_index, state.sequence_index, num_frames, frame_stride)
            batch.append((state.dataset_index, state.sequence_index, start, num_frames, frame_stride, target_height, target_width))
        prefetch_hint = self._local_prefetch_hint(bucket, slot, state)
        self._local_sequence_state[slot] = _LocalSequenceState(
            dataset_index=state.dataset_index,
            sequence_index=state.sequence_index,
            remaining_batches=state.remaining_batches - 1,
        )
        return _wrap_batch_with_prefetch_hint(batch, prefetch_hint)

    def _sample_bucketed_batch(self, bucket: tuple[int, ...], batch_index: int = 0) -> list[tuple[int, int, int, int, int, int, int]]:
        if self.container_local_sampling_enabled and self.container_local_sampling_dataset_names:
            slot = self._locality_slot(batch_index)
            state = self._local_sequence_state.get(slot)
            if state is not None and self._local_state_valid(state, bucket):
                return self._sample_local_bucketed_batch(bucket, batch_index)
            if state is not None and self._nonlocal_state_valid(state, bucket):
                self._local_sequence_state[slot] = _LocalSequenceState(
                    dataset_index=-1,
                    sequence_index=-1,
                    remaining_batches=state.remaining_batches - 1,
                )
                return self._sample_standard_bucketed_batch(bucket, include_local=False)
            if self._should_use_local_sampling_for_bucket(bucket):
                return self._sample_local_bucketed_batch(bucket, batch_index)
            self._local_sequence_state[slot] = _LocalSequenceState(
                dataset_index=-1,
                sequence_index=-1,
                remaining_batches=self.container_local_sampling_block_batches - 1,
            )
            return self._sample_standard_bucketed_batch(bucket, include_local=False)
        return self._sample_standard_bucketed_batch(bucket, include_local=True)

    def __iter__(self):
        batch_index = 0
        if self.target_shapes:
            bucket = self.valid_buckets[0]
            while True:
                if batch_index % self.ddp_batch_group_size == 0:
                    bucket_index = int(torch.randint(len(self.valid_buckets), (1,), generator=self.generator).item())
                    bucket = self.valid_buckets[bucket_index]
                batch_index += 1
                yield self._sample_bucketed_batch(bucket, batch_index - 1)
        else:
            num_frames = self.valid_lengths[0]
            while True:
                if batch_index % self.ddp_batch_group_size == 0:
                    length_index = int(torch.randint(len(self.valid_lengths), (1,), generator=self.generator).item())
                    num_frames = self.valid_lengths[length_index]
                probability_spec = self._dataset_probabilities_for_length(num_frames)
                assert probability_spec is not None
                active_indices, probability_tensor = probability_spec
                sampled_dataset_positions = torch.multinomial(
                    probability_tensor,
                    num_samples=self.batch_size,
                    replacement=True,
                    generator=self.generator,
                ).tolist()
                batch: list[tuple[int, int, int]] = []
                sequence_slot = batch_index % self.sticky_sequence_slots
                for dataset_position in sampled_dataset_positions:
                    dataset_index = active_indices[dataset_position]
                    batch.append(self._sample_anchor_for_dataset(dataset_index, num_frames, sequence_slot=sequence_slot))
                batch_index += 1
                yield batch


class RankShardedBatchSampler:
    def __init__(self, batch_sampler: Any, *, rank: int, world_size: int) -> None:
        if world_size <= 0:
            raise ValueError("world_size 必须为正整数。")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank={rank} 超出 world_size={world_size}。")
        self.batch_sampler = batch_sampler
        self.rank = int(rank)
        self.world_size = int(world_size)

    def _usable_batch_count(self) -> int:
        batch_count = len(self.batch_sampler)
        return batch_count - (batch_count % self.world_size)

    def __iter__(self):
        usable_batch_count = None
        if not bool(getattr(self.batch_sampler, "is_streaming_sampler", False)):
            usable_batch_count = self._usable_batch_count()
        for batch_index, batch in enumerate(self.batch_sampler):
            if usable_batch_count is not None and batch_index >= usable_batch_count:
                break
            if batch_index % self.world_size == self.rank:
                yield batch

    def __len__(self) -> int:
        return self._usable_batch_count() // self.world_size

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.batch_sampler, name)
