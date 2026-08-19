from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from formal_evaluation.contact.metrics import compute_contact_metrics
from formal_evaluation.common.mano_sampling import downsample_mano_vertices, marker_vertex_ids_195, upsample_mano_markers
from formal_evaluation.common.marker_vertices import MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195
from formal_evaluation.evaluate import _load_predictions
from formal_evaluation.evaluate_six_dataset import evaluate_window
from formal_evaluation.hand.metrics import compute_hand_metrics
from formal_evaluation.scene.metrics import camera_pose_auc, compute_scene_metrics


class FormalMetricTests(unittest.TestCase):
    def test_hand_identity(self) -> None:
        joints = np.random.default_rng(0).normal(size=(4, 2, 21, 3))
        valid = np.ones((4, 2), dtype=bool)
        values = compute_hand_metrics(joints, joints, valid, valid)
        self.assertEqual(values["hand_left_mpjpe"], 0.0)
        self.assertLess(values["hand_right_pa_mpjpe"], 1e-8)

    def test_hand_world_and_temporal_metrics(self) -> None:
        rng = np.random.default_rng(1)
        joints = rng.normal(size=(4, 2, 21, 3))
        valid = np.ones((4, 2), dtype=bool)
        shifted = joints + np.array([0.001, 0.0, 0.0])
        values = compute_hand_metrics(
            shifted, joints, valid, valid,
            world_prediction=shifted, world_target=joints,
        )
        self.assertAlmostEqual(values["hand_left_w_mpjpe"], 1.0, places=6)
        self.assertLess(values["hand_left_wa_mpjpe"], 1e-8)
        self.assertAlmostEqual(values["hand_right_mpjve"], 0.0, places=8)
        self.assertAlmostEqual(values["hand_right_mpjae"], 0.0, places=8)

    def test_marker_sampling_and_derived_vertex_metrics(self) -> None:
        vertices = np.arange(2 * 2 * 778 * 3, dtype=np.float32).reshape(2, 2, 778, 3)
        markers = downsample_mano_vertices(vertices)
        self.assertEqual(markers.shape, (2, 2, 195, 3))
        self.assertEqual(tuple(marker_vertex_ids_195()), MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195)
        self.assertTrue(np.array_equal(markers, vertices[:, :, marker_vertex_ids_195(), :]))
        restored = upsample_mano_markers(markers)
        self.assertEqual(restored.shape, vertices.shape)
        valid = np.ones((2, 2), dtype=bool)
        values = compute_hand_metrics(markers, markers, valid, valid, granularity="marker")
        self.assertEqual(values["hand_left_marker_mpmpe"], 0.0)

    def test_camera_pose_auc_identity_relative_motion(self) -> None:
        poses = np.repeat(np.eye(4)[None], 4, axis=0)
        poses[:, 0, 3] = np.arange(4)
        values = camera_pose_auc(poses, poses, np.ones(4, dtype=bool))
        self.assertAlmostEqual(values["auc"], 1.0, places=8)
        self.assertEqual(values["valid_count"], 3)

    def test_evaluator_derives_egofound3r_vertices_and_uses_gt_pose_fallback(self) -> None:
        rng = np.random.default_rng(3)
        T = 3
        vertices = rng.normal(size=(T, 2, 778, 3)).astype(np.float32)
        markers = downsample_mano_vertices(vertices)
        joints = rng.normal(size=(T, 2, 21, 3)).astype(np.float32)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], T, axis=0)
        poses[:, 0, 3] = np.arange(T, dtype=np.float32)
        target = {
            "camera_c2w": poses,
            "hand_joints_camera": joints,
            "hand_vertices_camera": vertices,
            "hand_markers_camera": markers,
            "hand_valid": np.ones((T, 2), dtype=bool),
        }
        prediction = {
            "hand_joints_camera": joints.copy(),
            "hand_markers_camera": markers.copy(),
            "hand_valid": np.ones((T, 2), dtype=bool),
        }
        result = evaluate_window(
            method="egofound3r", config={"group": ["hand"]},
            metadata={"frame_ids": ["0", "1", "2"]}, predictions=prediction,
            gt_metadata={"dataset": "h2o", "sequence_id": "s", "window_id": "w", "frame_ids": ["0", "1", "2"]},
            targets=target,
        )
        self.assertEqual(result["hand_vertex_geometry_provenance"], "derived_from_195_markers")
        self.assertEqual(result["hand_vertex_world_pose_source"], "gt_camera_c2w")
        self.assertTrue(np.isfinite(result["hand_left_vertex_w_mpvpe"]))

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
