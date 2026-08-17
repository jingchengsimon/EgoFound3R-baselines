from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from formal_evaluation.datasets.six_dataset_gt_cache import cache_paths, window_cache_id, write_window_cache


class SixDatasetGTCacheTests(unittest.TestCase):
    def test_window_identity_and_round_trip(self) -> None:
        row = {
            "dataset": "h2o", "sequence_id": "subject4_ego/h1/0", "window_id": "example",
            "frame_ids": ["000000", "000001"], "window_size": 2,
        }
        self.assertEqual(window_cache_id(row), window_cache_id(dict(row)))
        batch = {
            "camera_pose": np.tile(np.eye(4), (1, 2, 1, 1)),
            "camera_pose_supervision_mask": np.ones((1, 2), dtype=bool),
            "intrinsics": np.tile(np.eye(3), (1, 2, 1, 1)),
            "intrinsics_supervision_mask": np.ones((1, 2), dtype=bool),
            "joints_3d_targets": np.zeros((1, 2, 2, 21, 3), dtype=np.float32),
            "hand_valid_mask": np.ones((1, 2, 2), dtype=bool),
            "raw_joint_supervision_mask": np.ones((1, 2, 2), dtype=bool),
            "contact_targets": np.zeros((1, 2, 2, 21), dtype=np.float32),
            "contact_supervision_mask": np.ones((1, 2, 2, 21), dtype=bool),
            "marker_contact_targets": np.zeros((1, 2, 2, 195), dtype=np.float32),
            "marker_contact_supervision_mask": np.ones((1, 2, 2, 195), dtype=bool),
            "depth": np.ones((1, 2, 3, 4), dtype=np.float32),
            "depth_valid_mask": np.ones((1, 2, 3, 4), dtype=bool),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = write_window_cache(root, row, batch)
            self.assertEqual(result["status"], "written")
            data_path, metadata_path = cache_paths(root, row)
            self.assertTrue(data_path.is_file() and metadata_path.is_file())
            self.assertEqual(write_window_cache(root, row, batch)["status"], "reused")


if __name__ == "__main__":
    unittest.main()
