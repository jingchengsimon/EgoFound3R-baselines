import os
import tempfile
import unittest
from pathlib import Path

from formal_evaluation.cpfs_eval_storage import audit, registered_roots


class StorageAuditTest(unittest.TestCase):
    def test_deduplicates_hardlinks_and_does_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'a', root / 'b'
            a.mkdir()
            b.mkdir()
            (a / 'data').write_bytes(b'123')
            os.link(a / 'data', b / 'hardlink')
            (b / 'link').symlink_to(a, target_is_directory=True)
            result = audit([str(a), str(b)], directory)
            self.assertEqual(result['totals']['logical_bytes'], 3)
            self.assertEqual(result['totals']['regular_files'], 1)
            self.assertEqual(sum(row['symlinks'] for row in result['rows']), 1)

    def test_only_registered_paths_and_no_duplicate_subtrees(self):
        roots = registered_roots({'runs': [{'output_root': '/mnt/workspace/sjc/DATA/eval_artifacts/job'},
                                         {'gt_index': '/mnt/workspace/sjc/DATA/h2o_contact_baseline/cache/index.jsonl'},
                                         {'output_root': '/mnt/oss/pre-train/results'}]})
        self.assertNotIn('/mnt/workspace/sjc/DATA/eval_artifacts/job', roots)
        self.assertNotIn('/mnt/oss/pre-train/results', roots)
        self.assertIn('/mnt/workspace/sjc/DATA/h2o_contact_baseline/cache', roots)


if __name__ == '__main__':
    unittest.main()
