from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from formal_evaluation import artifact_copy_to_cpfs


class ArtifactCopyToCpfsTest(unittest.TestCase):
    def test_exact_catalog_roots_copy_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "oss"
            destination_root = root / "cpfs" / "result"
            gt = source_root / "gt"
            prediction = source_root / "prediction"
            gt.mkdir(parents=True)
            prediction.mkdir()
            (gt / "index.jsonl").write_text("index\n", encoding="utf-8")
            (prediction / "predictions.npz").write_bytes(b"prediction")
            spec = {
                "catalog_run_ids": ["catalog-run"],
                "methods": ["wilor"],
                "sources": [
                    {"dataset": "h2o", "role": "gt_cache", "source": str(gt)},
                    {"dataset": "h2o", "role": "wilor", "source": str(prediction)},
                ],
            }
            with (mock.patch.object(artifact_copy_to_cpfs, "OSS_ROOT", source_root),
                  mock.patch.object(artifact_copy_to_cpfs, "CPFS_ROOT", root / "cpfs"),
                  mock.patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(10_000, 0, 10_000))):
                first = artifact_copy_to_cpfs.run(spec, destination_root, 0)
                second = artifact_copy_to_cpfs.run(spec, destination_root, 0)
            self.assertEqual(first["copied_files"], 2)
            self.assertEqual(second["resumed_files"], 2)
            self.assertEqual((destination_root / "COMPLETE").read_text(), "complete\n")


if __name__ == "__main__":
    unittest.main()
