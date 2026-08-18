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
    build_sequence_splits,
    write_outputs,
)
from formal_evaluation.datasets.export_dataset_manifests import _taco_splits


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
            {"sequence_id": sequence_id, "frame_ids": [f"{frame:06d}" for frame in range(480)]}
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
        test_ids = sequence_ids[training_count:]
        manifest["official_splits"] = {
            "train": sequence_ids[:training_count],
            "test_1": test_ids[:215],
            "test_2": test_ids[215:435],
            "test_3": test_ids[435:763],
            "test_4": test_ids[763:],
        }
    return manifest


class DatasetSplitTests(unittest.TestCase):
    def test_taco_export_uses_frozen_sequence_manifest(self) -> None:
        payload = {
            "datasets": {
                "taco": {
                    "train_sequence_ids": ["taco/train"],
                    "test_sequence_ids": ["taco/test"],
                }
            }
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            official = Path(directory) / "official.txt"
            path.write_text(json.dumps(payload))
            official.write_text("train,train\ntest,test_1\n")
            self.assertEqual(
                _taco_splits(path, {"taco/train", "taco/test"}, official),
                {
                    "train": ["taco/train"],
                    "test_1": ["taco/test"],
                    "test_2": [],
                    "test_3": [],
                    "test_4": [],
                },
            )

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
            expected_windows = Counter({"h2o": 368, "taco": 5096, "hot3d": 56, "oakink_v2": 248, "arctic": 296, "hoi4d": 1008})
            self.assertEqual(Counter(row["dataset"] for row in rows), expected_windows)
            self.assertEqual(len({(row["dataset"], row["window_id"]) for row in rows}), sum(expected_windows.values()))
            self.assertEqual(split_manifest["evaluation"]["total_window_count"], sum(expected_windows.values()))
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
            taco_selection = split_manifest["datasets"]["taco"]["evaluation_selection"]
            self.assertEqual(
                taco_selection["selected_sequence_counts"],
                {"test_1": 107, "test_2": 110, "test_3": 164, "test_4": 256},
            )
            self.assertEqual(taco_selection["selected_sequence_count"], 637)
            for row in rows:
                self.assertEqual(len(row["frame_ids"]), 60)
                self.assertEqual(row["window_stride"], 60)
                self.assertEqual(row["window_overlap"], 0)
                self.assertEqual(row["window_source"], "sequence_start_strict_nonoverlap")
                self.assertIn(row["sequence_id"], split_manifest["datasets"][row["dataset"]]["test_sequence_ids"])
                self.assertEqual([int(item) for item in row["frame_ids"]], list(range(int(row["frame_ids"][0]), int(row["frame_ids"][0]) + 60)))
                self.assertEqual(int(row["frame_ids"][0]) % 60, 0)
                self.assertEqual(row["depth_3r_status"], CAPABILITIES[row["dataset"]]["depth_3r_status"])

    def test_rejects_taco_official_split_drift(self) -> None:
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
                sequence["frame_ids"].append("000480")
        with TemporaryDirectory() as directory:
            _, windows_path = write_outputs(manifests, Path(directory))
            rows = [json.loads(line) for line in windows_path.read_text().splitlines()]
        self.assertFalse(any("000480" in row["frame_ids"] for row in rows))


if __name__ == "__main__":
    unittest.main()
