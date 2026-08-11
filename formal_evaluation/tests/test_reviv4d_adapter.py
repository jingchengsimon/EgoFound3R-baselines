from __future__ import annotations

import numpy as np
import pytest

from formal_evaluation.hand.adapters.import_hand_motion_baseline import _indices
from formal_evaluation.scene.adapters.run_reviv4d_baseline import decode_camera_9d


def test_decode_reviv_camera_identity() -> None:
    camera = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32)
    decoded = decode_camera_9d(camera)
    np.testing.assert_allclose(decoded, np.eye(4, dtype=np.float32)[None])


def test_external_hand_import_requires_explicit_frame_map() -> None:
    with pytest.raises(ValueError, match="--frame-indices"):
        _indices(np.zeros((60, 2, 21, 3)), frame_count=12, requested=None)
