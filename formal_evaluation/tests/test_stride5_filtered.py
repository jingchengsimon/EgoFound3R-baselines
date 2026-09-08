import unittest
from unittest.mock import patch
import numpy as np
from formal_evaluation import analyze_stride5_filtered as m


class FrozenFilterTests(unittest.TestCase):
    def test_metric_selection_nan_and_dataset_quantiles(self):
        scores = np.zeros((1, 8, 60))
        scores[0, 0] = np.arange(60)
        scores[0, 2, 10] = 100
        scores[0, :, 20] = np.nan
        t, reason, mask = m.masks_for_scheme(scores, .95, (0, 1))
        self.assertFalse(mask[0, 10]); self.assertFalse(mask[0, 20])
        self.assertTrue(mask[0, 59]); self.assertFalse(reason[:, 2].any())
        _, _, eight = m.masks_for_scheme(scores, .975, tuple(range(8)))
        self.assertTrue(eight[0, 10])
        t2, _, _ = m.masks_for_scheme(scores * 10, .95, (0, 1))
        self.assertAlmostEqual(t2[0], t[0] * 10)

    def test_frozen_fits_temporal_support_and_hand_reduction(self):
        rng = np.random.default_rng(1)
        target = {'hand_valid': np.ones((60, 2), bool), 'camera_valid': np.ones(60, bool),
                  'camera_c2w': np.tile(np.eye(4), (60, 1, 1))}
        prediction = {'hand_valid': target['hand_valid'].copy()}
        for field, n in [('joints', 21), ('markers', 32), ('vertices', 778)]:
            points = rng.normal(0, .1, (60, 2, n, 3))
            target['hand_'+field+'_camera'] = points
            target['hand_'+field+'_world'] = points
            prediction['hand_'+field+'_camera'] = points + rng.normal(0, .003, points.shape)
        scores, frame, pair, triplet = m.freeze_window(prediction, target)
        np.testing.assert_allclose(scores[:6], np.max(frame[0], axis=0))
        self.assertTrue(np.isnan(scores[7, [0, 59]]).all())
        saved = frame.copy(); mask = np.zeros((1, 60), bool); mask[0, 30] = True
        with patch.object(m, 'world_aligned_mpjpe', side_effect=AssertionError('refit')), patch.object(m, '_procrustes_per_frame', side_effect=AssertionError('refit')):
            tables, rows = m.aggregate_scheme(frame[None], pair[None], triplet[None], mask)
        np.testing.assert_equal(frame, saved)
        for _, _, _, prefix, pos, vel, acc in m.GRANULARITIES:
            key = 'hand_left_'+prefix
            self.assertEqual(rows[0][key+'valid_frame_count'], 59)
            self.assertEqual(rows[0][key+vel+'_pair_count'], 57)
            self.assertEqual(rows[0][key+acc+'_triplet_count'], 55)
            self.assertAlmostEqual(rows[0][key+'wa_'+pos], frame[list(m.GRANULARITIES).index(next(g for g in m.GRANULARITIES if g[3]==prefix)),0,5,~mask[0]].mean())


if __name__ == '__main__':
    unittest.main()
