import hashlib
import json

import unittest

from formal_evaluation.run_gt_k_dyn100 import read_manifest


def test_exact_dyn100_rejects_duplicate_and_short_windows(tmp_path):
    counts = dict(zip(('h2o', 'hot3d', 'arctic', 'oakink_v2', 'taco', 'hoi4d'), (12, 17, 18, 17, 17, 19)))
    rows = [{'dataset': ds, 'window_id': str(i), 'frame_ids': list(range(60)),
             'geometry_paths': ['frame'] * 60} for ds, n in counts.items() for i in range(n)]
    path = tmp_path / 'manifest.json'

    def spec():
        path.write_text(json.dumps(rows))
        return {'manifest': str(path), 'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'expected_windows': counts}

    assert len(read_manifest(spec())) == 100
    rows[1]['window_id'] = rows[0]['window_id']
    with unittest.TestCase().assertRaisesRegex(ValueError, 'coverage mismatch'):
        read_manifest(spec())
    rows[1]['window_id'] = '1'
    rows[0]['frame_ids'].pop()
    with unittest.TestCase().assertRaisesRegex(ValueError, 'frame coverage'):
        read_manifest(spec())
