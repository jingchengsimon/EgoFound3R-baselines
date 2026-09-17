import tempfile
import unittest
from pathlib import Path

from formal_evaluation.migrate_p95_freeze_to_oss import copy_file, inventory, rewrite_link_target


class P95ArtifactMigrationTest(unittest.TestCase):
    def test_copy_hash_and_internal_absolute_link_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            payload = source / "nested" / "payload.bin"
            payload.parent.mkdir()
            payload.write_bytes(b"result-data")
            result = copy_file(
                payload,
                destination / "nested" / "payload.bin",
                {"relative_path": "nested/payload.bin", "bytes": 11, "mtime_ns": payload.stat().st_mtime_ns},
            )
            self.assertEqual(result["sha256"], "b776d2622109691a4cde840a6af8732be8f5358ef692beb0909bd0f5c45ef2e5")
            link = destination / "link"
            rewritten = rewrite_link_target(source / "link", str(payload), [(source, destination)], link)
            self.assertEqual(rewritten, "nested/payload.bin")
            files, links = inventory(destination)
            self.assertEqual([row["relative_path"] for row in files], ["nested/payload.bin"])
            self.assertEqual(links, [])


if __name__ == "__main__":
    unittest.main()
