import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from formal_evaluation.complete_contact_distance import (
    geometry_matches,
    interact_probabilities,
    restrict_distances_to_predicted_hands,
)


class CompleteContactDistanceTest(unittest.TestCase):
    def test_interact_native_vertex_projection_preserves_existing_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "native").mkdir()
            native = np.arange(6890, dtype=np.float32) / 6890
            for frame in range(2):
                np.savez(root / "native" / f"{frame:03d}.npz", pred_contact_3d_smplh=native)
            mapping = {"left": list(range(778)), "right": list(range(778, 1556))}
            marker_ids = np.arange(195)
            marker = np.stack([native[mapping["left"]][marker_ids], native[mapping["right"]][marker_ids]])
            canonical = {
                "hand_valid": np.array([[True, True], [True, False]]),
                "joint_contact_probability": np.full((2, 2, 21), 0.25, np.float32),
                "marker_contact_probability": np.stack([marker, marker]),
            }
            canonical["marker_contact_probability"][1, 1] = np.nan
            with patch("formal_evaluation.complete_contact_distance.marker_vertex_ids_195", return_value=marker_ids):
                result = interact_probabilities(root, canonical, mapping, 2)
            np.testing.assert_array_equal(result["joint_contact_probability"], canonical["joint_contact_probability"])
            np.testing.assert_allclose(result["vertex_contact_probability"][0, 0], native[:778])
            self.assertTrue(np.isnan(result["vertex_contact_probability"][1, 1]).all())

    def test_geometry_match_requires_both_joints_and_vertices(self) -> None:
        vertices = np.zeros((2, 2, 778, 3), np.float32)
        joints = np.zeros((2, 2, 21, 3), np.float32)
        query = {"vertex": vertices.copy(), "joint": joints.copy()}
        target = {"hand_vertices_camera": vertices, "hand_joints_camera": joints}
        valid = np.array([[True, False], [False, True]])
        self.assertTrue(geometry_matches(query, target, valid))
        query["vertex"][0, 0, 0, 0] = 0.01
        self.assertFalse(geometry_matches(query, target, valid))
        query["vertex"] = vertices.copy()
        query["joint"][1, 1, 0, 0] = 0.01
        self.assertFalse(geometry_matches(query, target, valid))

    def test_gt_distance_reuse_excludes_unpredicted_hand(self) -> None:
        distances = {}
        for prefix, points in (("joint", 21), ("marker", 195), ("vertex", 778)):
            distances[f"{prefix}_contact_distance"] = np.ones((1, 2, points), np.float32)
            distances[f"{prefix}_contact_distance_mask"] = np.ones((1, 2, points), bool)
        result = restrict_distances_to_predicted_hands(distances, np.array([[False, True]]))
        self.assertTrue(np.isnan(result["vertex_contact_distance"][0, 0]).all())
        self.assertFalse(result["joint_contact_distance_mask"][0, 0].any())
        self.assertTrue(result["marker_contact_distance_mask"][0, 1].all())


if __name__ == "__main__":
    unittest.main()
