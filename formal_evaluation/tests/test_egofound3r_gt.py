from __future__ import annotations

import unittest

from formal_evaluation.datasets.egofound3r_gt import (
    canonical_sequence_id,
    source_indices_for_window,
    validate_window_row,
)


class _Dataset:
    def __init__(self) -> None:
        self.sequence_to_indices = {"sequence": [4, 8, 15, 16]}
        self._record_index = [None] * 17
        for index, frame_id in zip(self.sequence_to_indices["sequence"], ("00001", "00002", "00003", "00004"), strict=True):
            self._record_index[index] = {"frame_id": frame_id}


class EgoFound3RGroundTruthTests(unittest.TestCase):
    def test_h2o_sequence_identity(self) -> None:
        self.assertEqual(canonical_sequence_id("h2o", "subject4_ego/h1/0"), "subject4_ego/h1/0/cam4")
        self.assertEqual(canonical_sequence_id("taco", "task/recording"), "task/recording")

    def test_window_requires_exact_contiguous_loader_frames(self) -> None:
        dataset = _Dataset()
        self.assertEqual(source_indices_for_window(dataset, "sequence", ["00002", "00003"]), [8, 15])
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            source_indices_for_window(dataset, "sequence", ["00001", "00003"])

    def test_row_contract(self) -> None:
        self.assertEqual(
            validate_window_row({"dataset": "taco", "sequence_id": "task/recording", "frame_ids": ["00001", "00002"], "window_size": 2}),
            ("taco", "task/recording", ["00001", "00002"]),
        )
        with self.assertRaisesRegex(ValueError, "window_size"):
            validate_window_row({"dataset": "taco", "sequence_id": "task/recording", "frame_ids": ["00001"], "window_size": 60})


if __name__ == "__main__":
    unittest.main()
