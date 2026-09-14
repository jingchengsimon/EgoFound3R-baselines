"""Guard the PAD-Hand formal adapter against dropping the left-hand track."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from formal_evaluation.hand.adapters.run_pad_hand_baseline import _split_wilor_by_side


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pad_prediction_slots", ROOT / "PAD-Hand/prediction_slots.py")
slots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(slots)


def hand(side, value):
    return {key: np.full(shape, value, dtype=np.float32) if shape else float(value)
            for key, shape in slots.SHAPES.items()} | {"is_right": float(side)}


class TestPadHandBimanual(unittest.TestCase):
    def test_both_hands_survive_selection_and_serialization(self):
        left, right = hand(0, 1), hand(1, 2)
        frames = [slots.select_hands([right, left], both_hands=True),
                  slots.select_hands([left], both_hands=True)]
        arrays = slots.pack_frames(frames, both_hands=True)
        self.assertEqual(arrays["vertices"].shape, (2, 2, 778, 3))
        np.testing.assert_array_equal(arrays["is_right"][0], [0, 1])
        self.assertTrue(np.isfinite(arrays["vertices"][0]).all())
        self.assertTrue(np.isnan(arrays["vertices"][1, 1]).all())
        self.assertEqual(float(arrays["vertices"][0, 0, 0, 0]), 1)
        self.assertEqual(float(arrays["vertices"][0, 1, 0, 0]), 2)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wilor.npz"
            np.savez_compressed(source, **arrays, fps=np.array([30], dtype=np.float32))
            left_path, right_path = _split_wilor_by_side(source, Path(directory))
            with np.load(left_path) as saved_left, np.load(right_path) as saved_right:
                self.assertEqual(saved_left["vertices"].shape, (2, 778, 3))
                self.assertEqual(float(saved_left["vertices"][0, 0, 0]), 1)
                self.assertEqual(float(saved_right["vertices"][0, 0, 0]), 2)
                self.assertTrue(np.isnan(saved_right["vertices"][1]).all())

    def test_single_hand_demo_output_stays_compatible(self):
        left, right = hand(0, 1), hand(1, 2)
        arrays = slots.pack_frames([slots.select_hands([left, right])])
        self.assertEqual(arrays["vertices"].shape, (1, 778, 3))
        self.assertEqual(float(arrays["is_right"][0]), 1)


if __name__ == "__main__":
    unittest.main()
