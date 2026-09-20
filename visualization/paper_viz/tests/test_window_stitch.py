import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from paper_viz.inputs import WindowSources, resolve_prediction_file
from paper_viz.sequences3d import build_segment_sequences, camera_bundle_report


def pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


class WindowStitchTest(unittest.TestCase):
    def test_prediction_index_rebases_stale_dsw_path_to_formal_root(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "hawor" / "formal" / "cache" / "predictions.npz"
            expected.parent.mkdir(parents=True)
            expected.touch()

            actual = resolve_prediction_file(
                "/mnt/workspace/old-node/hawor/formal/cache", root, "cache", "hawor",
                required=True)

            self.assertEqual(actual, expected)

    def test_required_prediction_reports_all_rebased_candidates(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "missing hawor prediction"):
                resolve_prediction_file(
                    "/mnt/workspace/old-node/hawor/formal/cache", temporary,
                    "cache", "hawor", required=True)

    def test_predicted_camera_keeps_drift_across_59_60_and_119_120(self):
        windows = []
        for index in range(3):
            gt = np.stack([pose((index * 60 + frame) * 0.01) for frame in range(60)])
            pred = np.stack([pose(frame * 0.012) for frame in range(60)])
            hand = np.zeros((60, 2, 1, 3))
            hand[..., 2] = 1.0
            valid = np.ones((60, 2), bool)
            windows.append(WindowSources(
                cache_id=str(index), window_id=str(index), frame_ids=list(range(60)),
                rgb_dir=None, geometry_dir=None, record={"image_size_hw": [10, 10]},
                methods={
                    "gt": {"camera_c2w": gt, "intrinsics": np.eye(3),
                           "hand_vertices_camera": hand, "hand_valid": valid},
                    "ego": {"camera_c2w": pred, "hand_vertices_camera": hand,
                            "hand_valid": valid},
                }))

        segment = SimpleNamespace(windows=windows)
        mano = SimpleNamespace(faces=np.zeros((1, 3), dtype=int))
        camera = build_segment_sequences(segment, mano)["ego"]["camera"]["c2w"]
        gt = np.concatenate([window.methods["gt"]["camera_c2w"] for window in windows])

        for boundary in (60, 120):
            expected = camera[boundary - 1] @ np.linalg.inv(gt[boundary - 1]) @ gt[boundary]
            np.testing.assert_allclose(camera[boundary], expected, atol=1e-12)
            self.assertGreater(abs(camera[boundary, 0, 3] - gt[boundary, 0, 3]), 0.1)

    def test_formal_ego_markers_are_upsampled_for_rendering(self):
        hand_markers = np.zeros((60, 2, 1, 3))
        hand_markers[..., 2] = 1.0
        window = WindowSources(
            cache_id="0", window_id="0", frame_ids=list(range(60)),
            rgb_dir=None, geometry_dir=None, record={"image_size_hw": [10, 10]},
            methods={
                "gt": {
                    "camera_c2w": np.stack([pose(frame * 0.01) for frame in range(60)]),
                    "intrinsics": np.eye(3),
                    "hand_vertices_camera": np.repeat(hand_markers, 2, axis=2),
                    "hand_valid": np.ones((60, 2), bool),
                },
                "ego": {
                    "camera_c2w": np.stack([pose(frame * 0.012) for frame in range(60)]),
                    "hand_markers_camera": hand_markers,
                    "hand_valid": np.ones((60, 2), bool),
                },
            },
        )
        mano = SimpleNamespace(
            faces=np.array([[0, 1, 1]], dtype=int),
            upsample=lambda value: np.repeat(value, 2, axis=2),
        )

        vertices = build_segment_sequences(SimpleNamespace(windows=[window]), mano)["ego"]["vertices"]

        self.assertEqual(vertices.shape, (60, 2, 2, 3))
        np.testing.assert_allclose(vertices[..., 2], 1.0)

    def test_level_tolerance_is_independent_from_stitch_tolerance(self):
        camera_to_display = np.repeat(np.eye(4)[None], 2, axis=0)
        camera_to_display[:, 1, 1] = -np.sqrt(1.0 - 0.0018 ** 2)
        camera_to_display[:, 2, 1] = 0.0018
        camera = SimpleNamespace(
            camera_to_display=camera_to_display,
            camera_valid=np.ones(2, bool),
        )

        report = camera_bundle_report({}, camera)
        strict = camera_bundle_report({}, camera, up_tol=1e-3)

        self.assertFalse(report["violations"])
        self.assertTrue(strict["violations"])

    def test_level_check_measures_direction_not_mean_length(self):
        camera_to_display = np.repeat(np.eye(4)[None], 2, axis=0)
        camera_to_display[:, 1, 1] = -np.sqrt(1.0 - 0.1 ** 2)
        camera_to_display[0, 2, 1] = 0.1
        camera_to_display[1, 2, 1] = -0.1
        camera = SimpleNamespace(
            camera_to_display=camera_to_display,
            camera_valid=np.ones(2, bool),
        )

        report = camera_bundle_report({}, camera, up_tol=1e-6)

        self.assertLess(report["up_error"], 1e-12)
        self.assertFalse(report["violations"])

    def test_frustum_gate_rejects_systemic_not_isolated_outliers(self):
        frames = 100
        rig = np.repeat(np.diag([1.0, -1.0, 1.0, 1.0])[None], frames, axis=0)
        intrinsics = np.repeat(np.diag([100.0, 100.0, 1.0])[None], frames, axis=0)
        valid = np.ones((frames, 2, 1), bool)
        camera = SimpleNamespace(
            camera_to_display=rig,
            camera_valid=np.ones(frames, bool),
        )

        def report(bad_frames):
            vertices = np.zeros((frames, 2, 1, 3))
            vertices[..., 2] = 1.0
            vertices[:bad_frames, ..., 2] = -1.0
            return camera_bundle_report({
                "egoforce": {
                    "vertices": vertices,
                    "valid": valid,
                    "camera": {
                        "c2w": rig,
                        "valid": np.ones(frames, bool),
                        "K": intrinsics,
                        "K_valid": np.ones(frames, bool),
                        "size_hw": (100, 100),
                        "source": "gt",
                    },
                },
            }, camera)

        isolated = report(1)
        systemic = report(20)

        self.assertFalse(isolated["violations"])
        self.assertTrue(isolated["warnings"])
        self.assertTrue(systemic["violations"])

    def test_calibrated_baseline_behind_camera_is_marked_invalid(self):
        gt_hand = np.zeros((60, 2, 1, 3))
        gt_hand[..., 2] = 1.0
        baseline_hand = gt_hand.copy()
        baseline_hand[..., 2] = -1.0
        rig = np.stack([pose(frame * 0.01) for frame in range(60)])
        window = WindowSources(
            cache_id="0", window_id="0", frame_ids=list(range(60)),
            rgb_dir=None, geometry_dir=None, record={"image_size_hw": [10, 10]},
            methods={
                "gt": {
                    "camera_c2w": rig,
                    "intrinsics": np.eye(3),
                    "hand_vertices_camera": gt_hand,
                    "hand_valid": np.ones((60, 2), bool),
                },
                "egoforce": {
                    "hand_vertices_camera": baseline_hand,
                    "hand_valid": np.ones((60, 2), bool),
                },
            },
        )
        mano = SimpleNamespace(faces=np.zeros((1, 3), dtype=int))

        store = build_segment_sequences(SimpleNamespace(windows=[window]), mano)

        self.assertFalse(store["egoforce"]["valid"].any())
        self.assertTrue(store["gt"]["valid"].all())

    def test_reviv4d_behind_its_predicted_camera_is_marked_invalid(self):
        gt_hand = np.zeros((60, 2, 1, 3))
        gt_hand[..., 2] = 1.0
        reviv_joints = np.repeat(gt_hand, 21, axis=2)
        reviv_joints[:30, ..., 2] = -1.0
        rig = np.stack([pose(frame * 0.01) for frame in range(60)])
        window = WindowSources(
            cache_id="0", window_id="0", frame_ids=list(range(60)),
            rgb_dir=None, geometry_dir=None, record={"image_size_hw": [10, 10]},
            methods={
                "gt": {
                    "camera_c2w": rig,
                    "intrinsics": np.eye(3),
                    "hand_vertices_camera": gt_hand,
                    "hand_valid": np.ones((60, 2), bool),
                },
                "reviv4d": {
                    "camera_c2w": rig,
                    "hand_joints_camera": reviv_joints,
                    "hand_valid": np.ones((60, 2), bool),
                },
            },
        )
        mano = SimpleNamespace(faces=np.zeros((1, 3), dtype=int))

        store = build_segment_sequences(SimpleNamespace(windows=[window]), mano)

        self.assertFalse(store["reviv4d"]["valid"][:30].any())
        self.assertTrue(store["reviv4d"]["valid"][30:].all())
        self.assertTrue(store["reviv4d"]["camera"]["valid"].all())

    def test_distance_gate_rejects_systemic_not_isolated_outliers(self):
        frames = 100
        rig = np.repeat(np.diag([1.0, -1.0, 1.0, 1.0])[None], frames, axis=0)
        valid = np.ones((frames, 2, 1), bool)
        camera = SimpleNamespace(camera_to_display=rig, camera_valid=np.ones(frames, bool))

        def report(near_frames):
            vertices = np.zeros((frames, 2, 1, 3))
            vertices[..., 2] = 1.0
            vertices[:near_frames, ..., 2] = 0.01
            return camera_bundle_report({
                "egoforce": {
                    "vertices": vertices,
                    "valid": valid,
                    "camera": {
                        "c2w": rig,
                        "valid": np.ones(frames, bool),
                        "K": np.repeat(np.eye(3)[None], frames, axis=0),
                        "K_valid": np.ones(frames, bool),
                        "size_hw": (100, 100),
                        "source": "gt",
                    },
                },
            }, camera)

        isolated = report(1)
        systemic = report(20)

        self.assertFalse(isolated["violations"])
        self.assertTrue(isolated["warnings"])
        self.assertTrue(systemic["violations"])


if __name__ == "__main__":
    unittest.main()
