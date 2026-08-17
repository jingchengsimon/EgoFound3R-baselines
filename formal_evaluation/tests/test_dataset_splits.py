from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from formal_evaluation.datasets.build_dataset_splits import (
    CAPABILITIES,
    DATASET_ORDER,
    EXPECTED_COUNTS,
    TACO_TEST_COUNTS,
    build_sequence_splits,
    write_outputs,
)


def _manifest(dataset: str) -> dict[str, object]:
    available, training_count, test_count = EXPECTED_COUNTS[dataset]
    if dataset == "h2o":
        sequence_ids = (
            [f"subject1_ego/action/train_{index:04d}" for index in range(114)]
            + [f"subject3_ego/action/val_{index:04d}" for index in range(24)]
            + [f"subject4_ego/action/test_{index:04d}" for index in range(46)]
        )
    else:
        sequence_ids = [f"{dataset}/sequence_{index:04d}" for index in range(available)]
    manifest: dict[str, object] = {
        "sequences": [
            {"sequence_id": sequence_id, "frame_ids": [f"{frame:06d}" for frame in range(96)]}
            for sequence_id in sequence_ids
        ]
    }
    if dataset == "h2o":
        official_paths = [
            sequence_id.replace("_ego", "") + "/cam4/rgb/000000.png"
            for sequence_id in sequence_ids
        ]
        manifest["official_splits"] = {
            "train": official_paths[:114],
            "val": official_paths[114:training_count],
            "test": official_paths[training_count:],
        }
    elif dataset == "taco":
        offset = training_count
        official = {"train": sequence_ids[:offset]}
        for split, count in TACO_TEST_COUNTS.items():
            official[split] = sequence_ids[offset : offset + count]
            offset += count
        manifest["official_splits"] = official
    return manifest


class DatasetSplitTests(unittest.TestCase):
    def test_counts_windows_and_reproducibility(self) -> None:
        manifests = {dataset: _manifest(dataset) for dataset in DATASET_ORDER}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first_split, first_windows = write_outputs(manifests, first)
            second_split, second_windows = write_outputs(manifests, second)
            self.assertEqual(first_split.read_bytes(), second_split.read_bytes())
            self.assertEqual(first_windows.read_bytes(), second_windows.read_bytes())

            split_manifest = json.loads(first_split.read_text())
            rows = [json.loads(line) for line in first_windows.read_text().splitlines()]
            self.assertEqual(Counter(row["dataset"] for row in rows), Counter({dataset: 300 for dataset in DATASET_ORDER}))
            self.assertEqual(len({(row["dataset"], row["window_id"]) for row in rows}), 1800)
            self.assertEqual(split_manifest["evaluation"]["total_window_count"], 1800)
            self.assertEqual(
                split_manifest["evaluation"]["sha256"],
                hashlib.sha256(first_windows.read_bytes()).hexdigest(),
            )
            for dataset, (available, training, test) in EXPECTED_COUNTS.items():
                entry = split_manifest["datasets"][dataset]
                per_dataset = json.loads((first / "sequence_splits" / f"{dataset}.json").read_text())
                self.assertEqual(per_dataset["dataset"], dataset)
                self.assertEqual(per_dataset["aggregate_sha256"], hashlib.sha256(first_split.read_bytes()).hexdigest())
                training_partition = "trainval" if dataset == "h2o" else "train"
                training_ids = entry[f"{training_partition}_sequence_ids"]
                self.assertEqual(entry["partition_names"], [training_partition, "test"])
                self.assertEqual((entry["available_sequence_count"], entry[f"{training_partition}_sequence_count"], entry["test_sequence_count"]), (available, training, test))
                self.assertFalse(set(training_ids) & set(entry["test_sequence_ids"]))
                self.assertEqual(
                    set(training_ids) | set(entry["test_sequence_ids"]),
                    {item["sequence_id"] for item in manifests[dataset]["sequences"]},
                )
                self.assertEqual(per_dataset[f"{training_partition}_sequence_ids"], entry[f"{training_partition}_sequence_ids"])
                self.assertEqual(per_dataset["test_sequence_ids"], entry["test_sequence_ids"])
            for row in rows:
                self.assertEqual(len(row["frame_ids"]), 12)
                self.assertEqual(row["window_stride"], 12)
                self.assertEqual(row["window_overlap"], 0)
                self.assertEqual(row["window_source"], "sequence_start_strict_nonoverlap")
                self.assertIn(row["sequence_id"], split_manifest["datasets"][row["dataset"]]["test_sequence_ids"])
                self.assertEqual([int(item) for item in row["frame_ids"]], list(range(int(row["frame_ids"][0]), int(row["frame_ids"][0]) + 12)))
                self.assertEqual(int(row["frame_ids"][0]) % 12, 0)
                self.assertEqual(row["depth_3r_status"], CAPABILITIES[row["dataset"]]["depth_3r_status"])

    def test_rejects_taco_official_count_drift(self) -> None:
        manifests = {dataset: _manifest(dataset) for dataset in DATASET_ORDER}
        manifests["taco"]["official_splits"]["test_1"].pop()
        with self.assertRaisesRegex(ValueError, "missing from official splits"):
            build_sequence_splits(manifests)

    def test_rejects_h2o_sequence_crossing_official_frame_splits(self) -> None:
        manifests = {dataset: _manifest(dataset) for dataset in DATASET_ORDER}
        manifests["h2o"]["official_splits"]["val"].append(
            manifests["h2o"]["official_splits"]["train"][0]
        )
        with self.assertRaisesRegex(ValueError, "multiple official splits"):
            build_sequence_splits(manifests)

    def test_ignores_incomplete_sequence_tail(self) -> None:
        manifests = {dataset: _manifest(dataset) for dataset in DATASET_ORDER}
        for dataset in DATASET_ORDER:
            for sequence in manifests[dataset]["sequences"]:
                sequence["frame_ids"].append("000096")
        with TemporaryDirectory() as directory:
            _, windows_path = write_outputs(manifests, Path(directory))
            rows = [json.loads(line) for line in windows_path.read_text().splitlines()]
        self.assertFalse(any("000096" in row["frame_ids"] for row in rows))


if __name__ == "__main__":
    unittest.main()
