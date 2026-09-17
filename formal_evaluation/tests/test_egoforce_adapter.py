from __future__ import annotations

import numpy as np
import pytest

from formal_evaluation.hand.adapters.run_egoforce_baseline import (
    INPUT_HW,
    MARKER_IDS_195,
    _canonical_arrays,
    _empty_arrays,
    _resize_rgb_and_intrinsics,
    _load_rgb_window_input,
    _verified_sha256,
)
from formal_evaluation.datasets.window_inputs import WINDOW_INPUT_VERSION
from formal_evaluation.common.schema import SCHEMA_VERSION, validate_comparison_output


def test_resize_256_scales_each_intrinsic_axis() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    K = np.array([[100.0, 0.0, 50.0], [0.0, 80.0, 25.0], [0.0, 0.0, 1.0]])
    resized, scaled = _resize_rgb_and_intrinsics(rgb, K)
    assert resized.shape == (*INPUT_HW, 3)
    np.testing.assert_allclose(scaled, [[128.0, 0.0, 64.0], [0.0, 204.8, 64.0], [0.0, 0.0, 1.0]])


def test_canonical_output_preserves_invalid_hands_as_nan() -> None:
    arrays = _empty_arrays(1)
    vertices = np.arange(2 * 778 * 3, dtype=np.float32).reshape(2, 778, 3)
    joints = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3)
    _canonical_arrays({"pred_vertices": vertices, "pred_j3d": joints, "visible_hand": [True, False]}, 1, 0, arrays)
    assert arrays["hand_valid"].tolist() == [[True, False]]
    np.testing.assert_array_equal(arrays["hand_markers_camera"][0, 0], vertices[0, MARKER_IDS_195])
    assert np.isnan(arrays["hand_vertices_camera"][0, 1]).all()
    validate_comparison_output({"schema_version": SCHEMA_VERSION, "frame_ids": ["f0"], "capabilities": {key: True for key in arrays}}, arrays)
    with pytest.raises(ValueError, match="output shapes"):
        _canonical_arrays({"pred_vertices": vertices[:, :1], "pred_j3d": joints, "visible_hand": [True, True]}, 1, 0, arrays)


def test_checkpoint_hash_is_streamed_and_checked(tmp_path) -> None:
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"EgoForce checkpoint fixture")
    import hashlib
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert _verified_sha256(checkpoint, digest) == digest
    with pytest.raises(ValueError, match="mismatch"):
        _verified_sha256(checkpoint, "0" * 64)


def test_rgb_only_window_input_does_not_require_geometry(tmp_path) -> None:
    from PIL import Image
    image = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image)
    record = {"window_input_version": WINDOW_INPUT_VERSION, "frame_ids": ["0"], "rgb_paths": [str(image)]}
    path = tmp_path / "window_input.json"
    import json
    path.write_text(json.dumps(record))
    assert _load_rgb_window_input(path) == record
