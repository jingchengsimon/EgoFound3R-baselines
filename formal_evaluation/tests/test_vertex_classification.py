import json

import numpy as np

from formal_evaluation.aggregate_vertex_classification import canonical_key, vertex_metrics


def test_vertex_metrics_masks_missing_hands():
    probability = np.ones((2, 2, 778), dtype=np.float32)
    target = np.ones_like(probability)
    mask = np.ones_like(probability, dtype=bool)
    probability[0, 0] = 0
    prediction = {
        "vertex_contact_probability": probability,
        "hand_valid": np.array([[True, False], [True, True]]),
    }
    result = vertex_metrics(prediction, {
        "vertex_contact_target": target,
        "vertex_contact_mask": mask,
    })
    assert result["vertex_contact_precision"] == 1
    assert result["vertex_contact_recall"] == 2 / 3
    assert result["vertex_contact_f1"] == 0.8
    assert result["vertex_contact_valid_count"] == 3 * 778
    json.dumps(result)


def test_vertex_gt_index_needs_only_window_id():
    assert canonical_key({"window_id": "h2o/window-1"}, {}) == "h2o/window-1"
