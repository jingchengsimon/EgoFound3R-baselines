from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from torch.utils.data import Dataset


MANAGED_SEQUENCE_SPLIT_DATASETS: dict[str, str] = {
    "h2o": "h2o",
    "hot3d_aria": "hot3d",
    "hoi4d": "hoi4d",
    "oakink_v2": "oakink_v2",
    "egoforce_arctic": "arctic",
    "taco": "taco",
}
_MANIFEST_VERSION = "egofound3r_dataset_sequence_splits_v2"


def canonical_sequence_id(dataset_name: str, raw_sequence_id: str) -> str:
    """Normalize a loader sequence ID to its manifest identity."""
    value = str(raw_sequence_id).strip("/")
    if dataset_name != "h2o":
        return value
    prefix, separator, view = value.rpartition("/")
    if separator != "/" or not prefix or view != "cam4":
        raise ValueError(f"h2o sequence must end in /cam4: {raw_sequence_id!r}")
    return prefix


def _manifest_id_set(value: Any, *, label: str) -> frozenset[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip("/") for item in value):
        raise ValueError(f"{label} must contain non-empty string IDs")
    ids = [item.strip("/") for item in value]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {label}")
    return frozenset(ids)


def load_sequence_partitions(manifest_path: str | Path, manifest_key: str) -> tuple[frozenset[str], frozenset[str]]:
    """Load immutable train/test sequence sets after schema validation."""
    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read sequence split manifest: {path}") from error
    if not isinstance(payload, dict) or payload.get("manifest_version") != _MANIFEST_VERSION:
        raise ValueError(f"unsupported sequence split manifest version in {path}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get(manifest_key), dict):
        raise ValueError(f"missing manifest dataset entry: {manifest_key}")
    entry = datasets[manifest_key]
    train_key = "trainval_sequence_ids" if manifest_key == "h2o" else "train_sequence_ids"
    train_ids = _manifest_id_set(entry.get(train_key), label=f"{manifest_key}.{train_key}")
    test_ids = _manifest_id_set(entry.get("test_sequence_ids"), label=f"{manifest_key}.test_sequence_ids")
    if train_ids & test_ids:
        raise ValueError(f"overlap between train and test sequence IDs for {manifest_key}")
    return train_ids, test_ids


def summarize_sequence_split_coverage(
    *,
    parent_ids: set[str] | frozenset[str],
    train_ids: set[str] | frozenset[str],
    test_ids: set[str] | frozenset[str],
) -> dict[str, list[str] | bool]:
    """Return a deterministic, fail-closed coverage report for canonical IDs."""
    parent = frozenset(parent_ids)
    train = frozenset(train_ids)
    test = frozenset(test_ids)
    overlap = train & test
    missing_train = train - parent
    missing_test = test - parent
    unpartitioned_parent = parent - (train | test)
    return {
        "missing_train": sorted(missing_train),
        "missing_test": sorted(missing_test),
        "unpartitioned_parent": sorted(unpartitioned_parent),
        "overlap": sorted(overlap),
        "exact_coverage": not (missing_train or missing_test or unpartitioned_parent or overlap),
    }


class SequenceSplitFrameDataset(Dataset):
    """Expose one manifest partition through wrapper-local frame indices."""

    def __init__(
        self,
        parent: Dataset,
        *,
        dataset_name: str,
        manifest_key: str,
        split: str,
        manifest_path: str | Path,
    ) -> None:
        if split not in {"train", "trainval", "test"}:
            raise ValueError(f"unsupported logical sequence split: {split}")
        parent_sequences = getattr(parent, "sequence_to_indices", None)
        if not isinstance(parent_sequences, Mapping) or not parent_sequences:
            raise ValueError("parent dataset must provide non-empty sequence_to_indices")

        train_ids, test_ids = load_sequence_partitions(manifest_path, manifest_key)
        selected_ids = test_ids if split == "test" else train_ids
        known_ids = train_ids | test_ids
        canonical_dataset_name = "h2o" if manifest_key == "h2o" else dataset_name
        canonical_to_parent: dict[str, tuple[int, str, list[int]]] = {}
        for parent_sequence_index, (raw_sequence_id, source_indices) in enumerate(parent_sequences.items()):
            canonical_id = canonical_sequence_id(canonical_dataset_name, str(raw_sequence_id))
            if canonical_id in canonical_to_parent:
                previous_raw_sequence_id = canonical_to_parent[canonical_id][1]
                raise ValueError(
                    "duplicate canonical parent sequence ID "
                    f"{canonical_id!r}: {previous_raw_sequence_id!r}, {raw_sequence_id!r}"
                )
            if canonical_id not in known_ids:
                raise ValueError(f"unpartitioned unexpected parent sequence ID: {canonical_id!r}")
            canonical_to_parent[canonical_id] = (
                parent_sequence_index,
                str(raw_sequence_id),
                [int(index) for index in source_indices],
            )

        missing_selected_ids = selected_ids - canonical_to_parent.keys()
        if missing_selected_ids:
            raise ValueError(f"missing selected manifest sequence IDs: {sorted(missing_selected_ids)}")

        self.parent = parent
        self.dataset_name = getattr(parent, "dataset_name", dataset_name)
        self.base_dataset_name = getattr(parent, "base_dataset_name", dataset_name)
        self.root = getattr(parent, "root", None)
        self.load_rgb = getattr(parent, "load_rgb", None)
        self.load_depth = getattr(parent, "load_depth", None)
        self.allowed_temporal_lengths = getattr(parent, "allowed_temporal_lengths", None)
        self.manifest_key = manifest_key
        self.manifest_path = Path(manifest_path)
        self.split = split
        self._source_indices: list[int] = []
        self._parent_sequence_indices: list[int] = []
        self.sequence_to_indices: dict[str, list[int]] = {}

        for canonical_id, (parent_sequence_index, raw_sequence_id, source_indices) in canonical_to_parent.items():
            if canonical_id not in selected_ids:
                continue
            local_start = len(self._source_indices)
            self._source_indices.extend(source_indices)
            self.sequence_to_indices[raw_sequence_id] = list(range(local_start, len(self._source_indices)))
            self._parent_sequence_indices.append(parent_sequence_index)

        if not self._source_indices or not self.sequence_to_indices:
            raise ValueError(f"empty selected sequence partition: {split}")

    def __len__(self) -> int:
        return len(self._source_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self._source_indices)
        if index < 0 or index >= len(self._source_indices):
            raise IndexError(index)
        sample = dict(self.parent[self._source_indices[index]])
        sample["split"] = self.split
        return sample

    def prefetch_sequence(self, sequence_index: int) -> None:
        if sequence_index < 0:
            sequence_index += len(self._parent_sequence_indices)
        if sequence_index < 0 or sequence_index >= len(self._parent_sequence_indices):
            raise IndexError(sequence_index)
        callback = getattr(self.parent, "prefetch_sequence", None)
        if callback is not None:
            callback(self._parent_sequence_indices[sequence_index])
