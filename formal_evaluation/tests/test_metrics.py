from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from formal_evaluation.contact.metrics import compute_contact_metrics
from formal_evaluation.evaluate import _load_predictions
from formal_evaluation.hand.metrics import compute_hand_metrics
from formal_evaluation.scene.metrics import compute_scene_metrics


class FormalMetricTests(unittest.TestCase):
    def test_hand_identity(self) -> None:
        joints = np.random.default_rng(0).normal(size=(4, 2, 21, 3))
        valid = np.ones((4, 2), dtype=bool)
        values = compute_hand_metrics(joints, joints, valid, valid)
        self.assertEqual(values["hand_left_mpjpe"], 0.0)
        self.assertLess(values["hand_right_pa_mpjpe"], 1e-8)

    def test_depth_keeps_small_positive_gt(self) -> None:
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        prediction = np.array([[0.005, 1.0]])
        target = np.array([[0.010, 1.0]])
        values = compute_scene_metrics(poses, poses, [(prediction, target)], scale_type="metric")
        self.assertGreater(values["depth_abs_rel"], 0.0)

    def test_contact_empty_denominator_is_nan(self) -> None:
        values = compute_contact_metrics(np.zeros(3), np.zeros(3), np.ones(3, dtype=bool))
        self.assertTrue(np.isnan(values["precision"]))
        self.assertTrue(np.isnan(values["recall"]))
        self.assertTrue(np.isnan(values["f1"]))

    def test_scene_loader_skips_unused_native_arrays(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            np.savez(path, camera_c2w=np.eye(4)[None], depth=np.ones((1, 2, 2)), world_points=np.ones((1, 2, 2, 3)))
            values = _load_predictions(path, {"group": ["scene"]})
        self.assertEqual(set(values), {"camera_c2w", "depth"})


if __name__ == "__main__":
    unittest.main()
