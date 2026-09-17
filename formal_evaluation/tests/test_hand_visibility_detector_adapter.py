from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from formal_evaluation.common.schema import SCHEMA_VERSION, validate_comparison_output
from formal_evaluation.hand.adapters.run_hand_visibility_detector import _select_sides


def test_same_side_uses_highest_detection_confidence_and_schema() -> None:
    left_low = SimpleNamespace(is_right=False, bbox_conf=0.2, visibility=np.full(21, 0.2, dtype=np.float32))
    left_high = SimpleNamespace(is_right=False, bbox_conf=0.8, visibility=np.full(21, 0.8, dtype=np.float32))
    right = SimpleNamespace(is_right=True, bbox_conf=0.4, visibility=np.full(21, 0.4, dtype=np.float32))
    values, valid, detail = _select_sides([left_low, right, left_high])
    assert valid.tolist() == [True, True]
    np.testing.assert_array_equal(values[0], left_high.visibility)
    assert detail[0] == {"slot": 0, "detections": 2, "selected_bbox_confidence": 0.8}
    arrays = {"hand_visibility": values[None], "hand_valid": valid[None]}
    validate_comparison_output({"schema_version": SCHEMA_VERSION, "frame_ids": ["0"], "capabilities": {key: True for key in arrays}}, arrays)

    missing_values, missing_valid, missing_detail = _select_sides([])
    assert not missing_valid.any() and np.isnan(missing_values).all()
    assert [item["detections"] for item in missing_detail] == [0, 0]
    validate_comparison_output({"schema_version": SCHEMA_VERSION, "frame_ids": ["0"], "capabilities": {key: True for key in arrays}}, {"hand_visibility": missing_values[None], "hand_valid": missing_valid[None]})
