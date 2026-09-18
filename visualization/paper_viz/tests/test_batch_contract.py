from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
MANIFEST = (REPO_ROOT / "visualization" /
            "overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914" /
            "selected_manifest.jsonl")
BATCH_PATH = PACKAGE_ROOT / "tools" / "batch_2d_render.py"

sys.path.insert(0, str(PACKAGE_ROOT))
from paper_viz.cli import VIDEO_GRID, frame_locations  # noqa: E402
from paper_viz import render2d  # noqa: E402

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

    def test_video_grid_is_semantic_4_by_5_and_covers_every_column_once(self):
        self.assertEqual((len(VIDEO_GRID), len(VIDEO_GRID[0])), (5, 4))
        flattened = [column for row in VIDEO_GRID for column in row]
        self.assertEqual(len(flattened), 16)
        self.assertEqual(set(flattened), set(render2d.COLUMNS))
        self.assertEqual(VIDEO_GRID[0], (("rgb", "geometry"),
                                        ("ego", "geometry"),
                                        ("ego_gt_k", "geometry"),
                                        ("gt", "geometry")))
        self.assertEqual(VIDEO_GRID[-2], (("ego", "visibility"),
                                         ("ego", "contact"),
                                         ("ego", "distance")))
        self.assertEqual(VIDEO_GRID[-1], (("gt", "visibility"),
                                         ("gt", "contact"),
                                         ("gt", "distance")))

    def test_ego_gt_k_reuses_ego_geometry_with_calibrated_k(self):
        frame = SimpleNamespace(window=SimpleNamespace(), K=np.eye(3),
                                ego_K=np.eye(3) * 2, width=1, height=1)
        geometry = (np.zeros((2, 778, 3)), np.ones(2, dtype=bool), None)
        with patch.object(render2d, "method_geometry", return_value=geometry), \
                patch.object(render2d, "geometry_cell",
                             side_effect=lambda *args, **kwargs: kwargs["K"]):
            self.assertIs(render2d.column_cell(frame, "ego", "geometry", 0), frame.ego_K)
            self.assertIs(render2d.column_cell(frame, "ego_gt_k", "geometry", 0), frame.K)

    def test_complete_results_rejects_partial_or_missing_segments(self):
        self.assertTrue(batch.complete_results(
            [{"status": "rendered"}, {"status": "skipped_existing"}], 2))
        self.assertFalse(batch.complete_results([{"status": "rendered"}], 2))
        self.assertFalse(batch.complete_results(
            [{"status": "rendered"}, {"status": "missing_ego_infer"}], 2))

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
