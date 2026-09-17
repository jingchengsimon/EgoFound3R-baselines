import tempfile
import unittest
from pathlib import Path
from formal_evaluation.cpfs_release_audit import audit_mirrors


class ReleaseAuditTest(unittest.TestCase):
    def test_missing_mirror_is_not_a_valid_backup(self):
        with tempfile.TemporaryDirectory() as d:
            s, t = Path(d)/'source', Path(d)/'target'
            s.mkdir(); t.mkdir()
            (s/'data').write_bytes(b'123')
            r = audit_mirrors([{'source':str(s),'destination':str(t)}])['mirrors'][0]
            self.assertFalse(r['size_mirror_verified'])
            self.assertEqual(r['bytes'], 3)
            (t/'data').write_bytes(b'123')
            self.assertTrue(audit_mirrors([{'source':str(s),'destination':str(t)}])['mirrors'][0]['size_mirror_verified'])

    def test_oss_link_to_source_blocks_release(self):
        with tempfile.TemporaryDirectory() as d:
            s, t = Path(d)/'source', Path(d)/'target'
            s.mkdir(); t.mkdir()
            for root in (s,t):
                (root/'data').write_bytes(b'123')
                (root/'link').symlink_to(s/'data')
            r=audit_mirrors([{'source':str(s),'destination':str(t)}])['mirrors'][0]
            self.assertFalse(r['size_mirror_verified'])
            self.assertEqual(r['examples'][0]['error'],'OSS_LINK_DEPENDS_ON_CPFS_SOURCE')


if __name__ == '__main__':
    unittest.main()
