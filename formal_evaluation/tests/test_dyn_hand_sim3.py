import unittest

import numpy as np

from formal_evaluation.recompute_dyn_hand_sim3 import alignment_metrics


class DynHandSim3Test(unittest.TestCase):
    def test_historical_first2_and_all_alignment(self):
        rng = np.random.default_rng(14)
        target = rng.normal(size=(60, 21, 3))
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        prediction = (target - np.array([3.0, 4.0, 5.0])) @ rotation / 2.0
        values = alignment_metrics(prediction, target, np.ones(60, dtype=bool))
        self.assertLess(values["w"], 1e-8)
        self.assertLess(values["wa"], 1e-8)
        prediction[2:] += np.array([0.01, 0.0, 0.0])
        values = alignment_metrics(prediction, target, np.ones(60, dtype=bool))
        self.assertGreater(values["w"], values["wa"])
        self.assertEqual(values["w_frame_count"], 60)
        self.assertEqual(values["wa_frame_count"], 60)


if __name__ == "__main__":
    unittest.main()
