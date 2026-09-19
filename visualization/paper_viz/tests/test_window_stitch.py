import unittest
from types import SimpleNamespace

import numpy as np

from paper_viz.inputs import WindowSources
from paper_viz.sequences3d import build_segment_sequences, camera_bundle_report


def pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


class WindowStitchTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
