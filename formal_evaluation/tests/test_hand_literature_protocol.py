"""Numerical protocol checks independent of cached benchmark results."""
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
from formal_evaluation.recompute_hand_literature import selected_metrics
from formal_evaluation.registered_artifact_paths import audit_reports


class LiteratureProtocolTest(unittest.TestCase):
    def test_detailed_report_requires_completion_and_matching_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / 'report.json'
            report.write_text(json.dumps({'gt_windows': 1, 'methods': {}}))
            (root / 'validation_report.json').write_text(json.dumps({'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest()}))
            catalog = {'dataset': 'h2o', 'report_roots': [directory], 'detailed_report_paths': [str(report)]}
            self.assertFalse(audit_reports(catalog)['report_details'])
            (root / 'COMPLETE').write_text('complete')
            # Single-method recomputation indexes omit the redundant method field.
            (root / 'predictions.jsonl').write_text(json.dumps({'prediction_dir': '/exact/prediction'}) + '\n')
            self.assertIn(str(report), audit_reports(catalog)['report_details'])
            report.write_text(json.dumps({'gt_windows': 2, 'methods': {}}))
            self.assertFalse(audit_reports(catalog)['report_details'])

    def test_first_two_similarity_and_later_drift(self):
        rng = np.random.default_rng(4)
        gt = rng.normal(size=(60, 21, 3))
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        pred = (gt - np.array([4., 5., 6.])) @ rotation / 2
        valid = np.ones(60, bool)
        self.assertLess(selected_metrics(pred, gt, valid, world_pred=pred, world_gt=gt)['w'], 1e-8)
        pred[2:] += np.array([0.01, 0., 0.])
        expected = 20 * 58 / 60
        self.assertAlmostEqual(selected_metrics(pred, gt, valid, world_pred=pred, world_gt=gt)['w'], expected)

    def test_temporal_matches_finite_difference_reference_and_masks(self):
        rng = np.random.default_rng(7)
        p, g = rng.normal(size=(2, 60, 21, 3))
        valid = np.ones(60, bool)
        v = selected_metrics(p, g, valid, temporal=True, fps=30)
        self.assertAlmostEqual(v['velocity'], np.linalg.norm(np.diff(p-g, axis=0), axis=-1).mean()*30000)
        self.assertAlmostEqual(v['acceleration'], np.linalg.norm(np.diff(p-g, n=2, axis=0), axis=-1).mean()*900000)
        valid[20] = False
        v = selected_metrics(p, g, valid, temporal=True)
        self.assertEqual(v['velocity_pair_count'], 57)
        self.assertEqual(v['acceleration_triplet_count'], 55)

    def test_wrist_relative_removes_time_varying_translation(self):
        rng = np.random.default_rng(8)
        gt = rng.normal(size=(60, 21, 3))
        pred = gt + np.arange(60)[:, None, None] ** 2
        result = selected_metrics(pred, gt, np.ones(60, bool), temporal=True,
                                  roots_pred=pred[:, 0], roots_gt=gt[:, 0])
        self.assertLess(result['velocity'], 1e-7)
        self.assertLess(result['acceleration'], 1e-6)


if __name__ == '__main__':
    unittest.main()
