"""Build deterministic sequence splits and all strict 60-frame evaluation clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Mapping

SEED = 0
WINDOW_SIZE = 60
TACO_EVALUATION_SEQUENCE_FRACTION = 0.5
CUSTOM_EVALUATION_SEQUENCE_FRACTION = 0.5
CUSTOM_TEST_FRACTIONS = {
    "hot3d": 0.10,
    "oakink_v2": 0.10,
    "arctic": 0.25,
    "hoi4d": 0.15,
}
DATASET_ORDER = ("h2o", "taco", "hot3d", "oakink_v2", "arctic", "hoi4d")
EXPECTED_COUNTS = {
    "h2o": (184, 138, 46),
    "taco": (2114, 839, 1275),
    "hot3d": (151, 136, 15),
    "oakink_v2": (627, 565, 62),
    "arctic": (301, 226, 75),
    "hoi4d": (1683, 1431, 252),
}
CAPABILITIES = {
    "h2o": {"hand_evaluation_status": "available", "depth_3r_status": "available"},
    "taco": {"hand_evaluation_status": "available", "depth_3r_status": "available"},
    "hot3d": {"hand_evaluation_status": "available", "depth_3r_status": "unavailable_no_depth_gt"},
    "oakink_v2": {"hand_evaluation_status": "available", "depth_3r_status": "unavailable_no_depth_gt"},
    "arctic": {"hand_evaluation_status": "available", "depth_3r_status": "unavailable_no_usable_depth_gt"},
    "hoi4d": {"hand_evaluation_status": "available", "depth_3r_status": "available"},
}


def _sequence_map(manifest: Mapping[str, object], dataset: str) -> dict[str, list[str]]:
    entries = manifest.get("sequences")
    if not isinstance(entries, list):
        raise ValueError(f"{dataset}: manifest.sequences must be a list")
    result: dict[str, list[str]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{dataset}: sequence entry must be an object")
        sequence_id, frame_ids = entry.get("sequence_id"), entry.get("frame_ids")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError(f"{dataset}: invalid sequence_id")
        if sequence_id in result:
            raise ValueError(f"{dataset}: duplicate sequence_id {sequence_id}")
        if not isinstance(frame_ids, list) or not all(isinstance(item, str) for item in frame_ids):
            raise ValueError(f"{dataset}/{sequence_id}: frame_ids must be a string list")
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError(f"{dataset}/{sequence_id}: duplicate frame_ids")
        result[sequence_id] = frame_ids
    return result


def _h2o_sequence_id(item: str) -> str:
    parts = item.split("/")
    if "cam4" in parts:
        parts = parts[:parts.index("cam4")]
    if len(parts) != 3:
        raise ValueError(f"h2o: cannot derive sequence ID from official item {item!r}")
    if parts[0].startswith("subject") and not parts[0].endswith("_ego"):
        parts[0] += "_ego"
    return "/".join(parts)


def _official_splits(manifest: Mapping[str, object], dataset: str) -> dict[str, set[str]]:
    raw = manifest.get("official_splits")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{dataset}: official_splits is required")
    result: dict[str, set[str]] = {}
    for split, sequence_ids in raw.items():
        if not isinstance(split, str) or not isinstance(sequence_ids, list) or not all(isinstance(item, str) for item in sequence_ids):
            raise ValueError(f"{dataset}: invalid official split {split!r}")
        result[split] = {
            _h2o_sequence_id(item) if dataset == "h2o" else item
            for item in sequence_ids
        }
    return result


def _official_partition(
    dataset: str,
    actual_ids: set[str],
    official: Mapping[str, set[str]],
) -> tuple[list[str], list[str], dict[str, list[str]], dict[str, int]]:
    if dataset == "h2o":
        required = {"train", "val", "test"}
        training_labels, test_labels = ("train", "val"), ("test",)
    else:
        required = {"train", "test_1", "test_2", "test_3", "test_4"}
        training_labels, test_labels = ("train",), ("test_1", "test_2", "test_3", "test_4")
    missing_labels = required - set(official)
    if missing_labels:
        raise ValueError(f"{dataset}: missing official split labels {sorted(missing_labels)}")

    memberships = Counter(sequence_id for label in required for sequence_id in official[label] if sequence_id in actual_ids)
    overlaps = sorted(sequence_id for sequence_id, count in memberships.items() if count != 1)
    if overlaps:
        raise ValueError(f"{dataset}: actual sequences occur in multiple official splits: {overlaps[:5]}")
    classified = set(memberships)
    unclassified = sorted(actual_ids - classified)
    if unclassified:
        raise ValueError(f"{dataset}: actual sequences missing from official splits: {unclassified[:5]}")

    training = sorted(actual_ids & set().union(*(official[label] for label in training_labels)))
    test = sorted(actual_ids & set().union(*(official[label] for label in test_labels)))
    missing = {label: sorted(official[label] - actual_ids) for label in sorted(required)}
    actual_counts = {label: len(actual_ids & official[label]) for label in sorted(required)}
    return training, test, missing, actual_counts


def _taco_evaluation_selection(
    official: Mapping[str, set[str]], actual_ids: set[str]
) -> tuple[dict[str, list[str]], list[str]]:
    selected_by_subset: dict[str, list[str]] = {}
    for subset in ("test_1", "test_2", "test_3", "test_4"):
        candidates = sorted(actual_ids & official[subset])
        selected_count = int(len(candidates) * TACO_EVALUATION_SEQUENCE_FRACTION)
        shuffled = list(candidates)
        random.Random(f"{SEED}:{subset}").shuffle(shuffled)
        selected_by_subset[subset] = sorted(shuffled[:selected_count])
    return selected_by_subset, sorted(
        sequence_id for subset_ids in selected_by_subset.values() for sequence_id in subset_ids
    )


def _custom_evaluation_selection(dataset: str, test_ids: list[str]) -> list[str]:
    shuffled = sorted(test_ids)
    random.Random(f"{SEED}:{dataset}:evaluation").shuffle(shuffled)
    return sorted(shuffled[:int(len(shuffled) * CUSTOM_EVALUATION_SEQUENCE_FRACTION)])


def _custom_partition(actual_ids: set[str], test_count: int) -> tuple[list[str], list[str]]:
    shuffled = sorted(actual_ids)
    random.Random(SEED).shuffle(shuffled)
    test = set(shuffled[:test_count])
    return sorted(actual_ids - test), sorted(test)


def _validate_expected_counts(dataset: str, available: int, training: list[str], test: list[str]) -> None:
    expected = EXPECTED_COUNTS[dataset]
    observed = (available, len(training), len(test))
    if observed != expected:
        raise ValueError(f"{dataset}: expected available/training/test={expected}, got {observed}")
    if set(training) & set(test) or len(set(training) | set(test)) != available:
        raise ValueError(f"{dataset}: training/test must be disjoint and exhaustive")


def build_sequence_splits(manifests: Mapping[str, Mapping[str, object]]) -> tuple[dict[str, object], dict[str, dict[str, list[str]]]]:
    output: dict[str, object] = {
        "manifest_version": "egofound3r_dataset_sequence_splits_v2",
        "seed": SEED,
        "datasets": {},
    }
    frames_by_dataset: dict[str, dict[str, list[str]]] = {}
    for dataset in DATASET_ORDER:
        manifest = manifests[dataset]
        sequence_frames = _sequence_map(manifest, dataset)
        frames_by_dataset[dataset] = sequence_frames
        actual_ids = set(sequence_frames)
        if dataset in {"h2o", "taco"}:
            official = _official_splits(manifest, dataset)
            training, test, missing, official_counts = _official_partition(
                dataset, actual_ids, official
            )
            split_type = "official_test_subset"
        else:
            training, test = _custom_partition(
                actual_ids, int(len(actual_ids) * CUSTOM_TEST_FRACTIONS[dataset])
            )
            missing, official_counts = {}, {}
            split_type = "custom_fraction_sequence"
        _validate_expected_counts(dataset, len(actual_ids), training, test)
        training_partition = "trainval" if dataset == "h2o" else "train"
        entry = {
            "split_type": split_type,
            "split_source": "official frame-level train+val/test collapsed to sequences" if dataset == "h2o" else (
                "official train/S1-S4 intersection" if dataset == "taco" else (
                    "sorted sequence IDs; random.Random(0); "
                    f"{CUSTOM_TEST_FRACTIONS[dataset]:.0%} test"
                )
            ),
            "partition_names": [training_partition, "test"],
            "available_sequence_count": len(actual_ids),
            f"{training_partition}_sequence_count": len(training),
            "test_sequence_count": len(test),
            f"{training_partition}_sequence_ids": training,
            "test_sequence_ids": test,
            "official_available_counts": official_counts,
            "missing_official_sequence_ids": missing,
            "filtered_sequence_ids": [],
            **CAPABILITIES[dataset],
        }
        if dataset == "taco":
            selected_by_subset, selected = _taco_evaluation_selection(official, actual_ids)
            entry.update({
                "official_test_subset_sequence_ids": {
                    subset: sorted(actual_ids & official[subset])
                    for subset in ("test_1", "test_2", "test_3", "test_4")
                },
                "evaluation_selection": {
                    "unit": "sequence",
                    "fraction_per_official_test_subset": TACO_EVALUATION_SEQUENCE_FRACTION,
                    "seed_scheme": "random.Random(f'{seed}:{official_test_subset}')",
                    "selected_sequence_counts": {subset: len(items) for subset, items in selected_by_subset.items()},
                    "selected_sequence_ids_by_official_test_subset": selected_by_subset,
                    "selected_sequence_count": len(selected),
                    "selected_sequence_ids": selected,
                },
            })
        elif dataset in CUSTOM_TEST_FRACTIONS:
            selected = _custom_evaluation_selection(dataset, test)
            entry["evaluation_selection"] = {
                "unit": "sequence",
                "fraction_of_custom_test": CUSTOM_EVALUATION_SEQUENCE_FRACTION,
                "seed_scheme": "random.Random(f'{seed}:{dataset}:evaluation')",
                "selected_sequence_count": len(selected),
                "selected_sequence_ids": selected,
            }
        output["datasets"][dataset] = entry
    return output, frames_by_dataset


def _generated_candidates(dataset: str, test_ids: list[str], sequence_frames: Mapping[str, list[str]]) -> list[dict[str, object]]:
    candidates = []
    for sequence_id in test_ids:
        frames = sequence_frames[sequence_id]
        for start in range(0, len(frames) - WINDOW_SIZE + 1, WINDOW_SIZE):
            frame_ids = frames[start : start + WINDOW_SIZE]
            candidates.append({
                "dataset": dataset,
                "window_id": f"{sequence_id}:{frame_ids[0]}-{frame_ids[-1]}",
                "sequence_id": sequence_id,
                "frame_ids": frame_ids,
            })
    return candidates


def build_evaluation_windows(
    manifests: Mapping[str, Mapping[str, object]],
    sequence_splits: Mapping[str, object],
    frames_by_dataset: Mapping[str, Mapping[str, list[str]]],
) -> list[dict[str, object]]:
    rows = []
    split_datasets = sequence_splits["datasets"]
    for dataset in DATASET_ORDER:
        selection = split_datasets[dataset].get("evaluation_selection", {})
        test_ids = list(selection.get("selected_sequence_ids", split_datasets[dataset]["test_sequence_ids"]))
        candidates = _generated_candidates(dataset, test_ids, frames_by_dataset[dataset])
        window_ids = [item["window_id"] for item in candidates]
        if len(window_ids) != len(set(window_ids)):
            raise ValueError(f"{dataset}: duplicate candidate window IDs")
        if not candidates:
            raise ValueError(f"{dataset}: no valid evaluation windows")
        for item in candidates:
            rows.append({
                **item,
                "split": "test",
                "window_size": WINDOW_SIZE,
                "window_stride": WINDOW_SIZE,
                "window_overlap": 0,
                "window_source": "sequence_start_strict_nonoverlap",
                **CAPABILITIES[dataset],
            })
    return rows


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()


def write_dataset_split_files(splits: Mapping[str, object], output_dir: Path, aggregate_bytes: bytes) -> list[Path]:
    split_dir = output_dir / "sequence_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    aggregate_sha256 = hashlib.sha256(aggregate_bytes).hexdigest()
    paths = []
    for dataset in DATASET_ORDER:
        path = split_dir / f"{dataset}.json"
        path.write_bytes(_json_bytes({
            "manifest_version": "egofound3r_per_dataset_sequence_split_v1",
            "dataset": dataset,
            "seed": splits["seed"],
            "aggregate_manifest": "../dataset_sequence_splits.json",
            "aggregate_sha256": aggregate_sha256,
            **splits["datasets"][dataset],
        }))
        paths.append(path)
    return paths


def write_outputs(manifests: Mapping[str, Mapping[str, object]], output_dir: Path) -> tuple[Path, Path]:
    splits, frames_by_dataset = build_sequence_splits(manifests)
    windows = build_evaluation_windows(manifests, splits, frames_by_dataset)
    window_bytes = _jsonl_bytes(windows)
    splits["evaluation"] = {
        "window_manifest": "evaluation_test_windows_60f_strict.jsonl",
        "window_count_per_dataset": dict(Counter(row["dataset"] for row in windows)),
        "total_window_count": len(windows),
        "window_size": WINDOW_SIZE,
        "window_stride": WINDOW_SIZE,
        "window_overlap": 0,
        "drop_incomplete_tail": True,
        "seed": SEED,
        "sha256": hashlib.sha256(window_bytes).hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "dataset_sequence_splits.json"
    window_path = output_dir / "evaluation_test_windows_60f_strict.jsonl"
    split_bytes = _json_bytes(splits)
    split_path.write_bytes(split_bytes)
    write_dataset_split_files(splits, output_dir, split_bytes)
    window_path.write_bytes(window_bytes)
    return split_path, window_path


def _load_manifests(values: list[str]) -> dict[str, Mapping[str, object]]:
    paths = {}
    for value in values:
        dataset, separator, path = value.partition("=")
        if not separator or dataset not in DATASET_ORDER or dataset in paths:
            raise ValueError(f"--manifest must be a unique DATASET=PATH for {DATASET_ORDER}: {value}")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{dataset}: manifest root must be an object")
        paths[dataset] = payload
    missing = set(DATASET_ORDER) - set(paths)
    if missing:
        raise ValueError(f"missing dataset manifests: {sorted(missing)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    split_path, window_path = write_outputs(_load_manifests(args.manifest), args.output_dir)
    print(json.dumps({"sequence_splits": str(split_path), "evaluation_windows": str(window_path)}, indent=2))


if __name__ == "__main__":
    main()
