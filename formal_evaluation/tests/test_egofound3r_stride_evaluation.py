import json
from unittest.mock import patch

import pytest

from formal_evaluation.run_egofound3r_stride_evaluation import prepare_inputs


def test_prepare_exact_gt_coverage_and_reject_duplicate(tmp_path):
    cache = tmp_path / "gt_cache"
    cache.mkdir()
    (cache / "metadata.json").write_text("{}")
    (cache / "arrays.npz").write_bytes(b"fixture")
    gt = cache / "index.jsonl"
    gt.write_text(json.dumps({"window_id": "w", "metadata_path": str(cache / "metadata.json"),
                              "array_path": str(cache / "arrays.npz")}) + "\n")
    rgb = tmp_path / "rgb.png"
    rgb.write_bytes(b"rgb")
    record = {"dataset": "h2o", "window_id": "w", "frame_ids": list(map(str, range(60))),
              "rgb_paths": [str(rgb)] * 60, "cache_id": "c"}
    index = tmp_path / "inputs.jsonl"
    index.write_text(json.dumps({"window_input": "window.json"}) + "\n")
    index.with_suffix(".status.json").write_text(json.dumps({"status": "complete", "window_count": 1}))
    spec = {"gt_index": str(gt), "expected_windows": 1, "input_indices": [str(index)]}
    with patch("formal_evaluation.run_egofound3r_stride_evaluation.load_window_input", return_value=record):
        _, _, records = prepare_inputs("h2o", spec, tmp_path / "good")
        assert len(records) == 1 and records[0]["window_id"] == "w"
        with pytest.raises(ValueError, match="duplicate input window"):
            prepare_inputs("h2o", {**spec, "input_indices": [str(index)] * 2}, tmp_path / "duplicate")
        with pytest.raises(ValueError, match="input coverage"):
            prepare_inputs("h2o", {**spec, "input_indices": []}, tmp_path / "missing")
