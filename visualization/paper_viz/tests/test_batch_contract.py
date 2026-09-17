from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
MANIFEST = (REPO_ROOT / "visualization" /
            "overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914" /
            "selected_manifest.jsonl")
BATCH_PATH = PACKAGE_ROOT / "tools" / "batch_2d_render.py"

sys.path.insert(0, str(PACKAGE_ROOT))
from paper_viz.cli import frame_locations  # noqa: E402

spec = importlib.util.spec_from_file_location("batch_2d_render", BATCH_PATH)
batch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(batch)

stage_spec = importlib.util.spec_from_file_location(
    "stage_ego_windows", PACKAGE_ROOT / "tools" / "stage_ego_windows.py")
stage = importlib.util.module_from_spec(stage_spec)
assert stage_spec.loader is not None
stage_spec.loader.exec_module(stage)


class BatchContractTest(unittest.TestCase):
    def test_frozen_manifest_has_unique_sequence_aware_ids(self):
        entries = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
        segment_ids = [batch.segment_id_of(entry) for entry in entries]
        self.assertEqual(len(entries), 114)
        self.assertEqual(len(segment_ids), len(set(segment_ids)))

    def test_frame_locations_keep_exact_selected_indices(self):
        segment = SimpleNamespace(windows=[
            SimpleNamespace(selected_indices=[4, 5, 9]),
            SimpleNamespace(selected_indices=[0, 1]),
        ])
        self.assertEqual(frame_locations(segment), [(0, 4), (0, 5), (0, 9), (1, 0), (1, 1)])

    def test_frozen_manifest_frame_refs_are_exact_and_cache_contiguous(self):
        entries = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
        for entry in entries:
            caches, plan = stage.entry_frame_plan(entry)
            grouped = [(cache, local) for cache in caches for _, local, _ in plan[cache]]
            manifest = [(ref["cache_id"], int(ref["index"])) for ref in entry["frame_refs"]]
            self.assertEqual(grouped, manifest)
            self.assertEqual(len(entry["frame_ids"]), entry["actual_length"])

    def test_partial_inference_is_padded_back_to_window_coordinates(self):
        source = np.array([[10], [20], [30]], dtype=np.float32)
        assignments = [(0, 1, "a"), (1, 3, "b"), (2, 4, "c")]
        padded = stage.padded_window(source, assignments, window=6, fill_value=np.nan)
        self.assertTrue(np.isnan(padded[[0, 2, 5]]).all())
        np.testing.assert_array_equal(padded[[1, 3, 4], 0], [10, 20, 30])

    def test_dry_run_does_not_create_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "must-not-exist"
            command = [
                sys.executable, str(BATCH_PATH),
                "--manifest", str(MANIFEST),
                "--ego-infer-root", str(Path(tmp) / "infer"),
                "--src-dir", str(Path(tmp) / "src"),
                "--out-root", str(out),
                "--dry-run",
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("114 segments selected", result.stdout)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
