import unittest
from types import SimpleNamespace

import numpy as np

from paper_viz.inputs import WindowSources
from paper_viz.sequences3d import build_segment_sequences


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


if __name__ == "__main__":
    unittest.main()
