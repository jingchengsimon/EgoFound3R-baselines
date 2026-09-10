import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PATH = Path(__file__).parents[1] / "recompute_same_mask_all_methods.py"
SPEC = importlib.util.spec_from_file_location("same_mask_recompute", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_remap_only_moves_registered_oss_tree(tmp_path):
    root = tmp_path / "mirror"
    assert MODULE.remap(
        "/mnt/oss/pre-train/ego/eval_artifacts/a/b", root
    ) == root / "payload/a/b"
    assert MODULE.remap("/mnt/workspace/sjc/a", root) == Path("/mnt/workspace/sjc/a")
    assert MODULE.remap(
        "/mnt/oss/pre-train/ego/eval_artifacts/a/b", root, direct_oss=True
    ) == Path("/mnt/oss/pre-train/ego/eval_artifacts/a/b")


def test_direct_oss_gt_rows_follow_registered_index_directory(tmp_path):
    index_parent = tmp_path / "current" / "gt_cache"
    row = {
        "array_path": "/old/location/gt_cache/h2o/cache.npz",
        "metadata_path": "/old/location/gt_cache/h2o/cache.json",
    }
    mapped = MODULE.remap_gt_row(
        row, tmp_path / "migration", index_parent, direct_oss=True
    )
    assert mapped["array_path"] == str(index_parent / "h2o/cache.npz")
    assert mapped["metadata_path"] == str(index_parent / "h2o/cache.json")


def test_freeze_skips_unsupported_hand_granularity():
    frames = 3
    hand = SimpleNamespace(
        FRAME_METRICS=("MPJPE",),
        GRANULARITIES=(("marker", "marker", "markers", "", "", "", ""),),
        _prediction_geometry_camera=lambda prediction, granularity: (
            (prediction["hand_joints_camera"], "native")
            if granularity == "joint" else (None, None)
        ),
    )
    prediction = {
        "hand_joints_camera": np.zeros((frames, 2, 21, 3)),
        "hand_valid": np.ones((frames, 2), dtype=bool),
    }
    target = {
        "hand_joints_camera": np.zeros((frames, 2, 21, 3)),
        "hand_valid": np.ones((frames, 2), dtype=bool),
        "camera_c2w": np.repeat(np.eye(4)[None], frames, axis=0),
        "camera_valid": np.ones(frames, dtype=bool),
    }
    _, frozen_frame, frozen_pair, frozen_triplet = MODULE.freeze_hand_window(
        hand, prediction, target
    )
    assert np.isnan(frozen_frame).all()
    assert np.isnan(frozen_pair).all()
    assert np.isnan(frozen_triplet).all()


def test_prediction_rows_normalizes_cache_id_to_window_id(tmp_path):
    prediction = tmp_path / "formal/cache-a"
    prediction.mkdir(parents=True)
    (prediction / "metadata.json").write_text(
        '{"dataset":"h2o","window_id":"cache-a"}'
    )
    (prediction / "predictions.npz").write_bytes(b"x")
    outside = tmp_path / "formal/outside"
    outside.mkdir()
    (outside / "metadata.json").write_text(
        '{"dataset":"h2o","window_id":"outside-window"}'
    )
    (outside / "predictions.npz").write_bytes(b"x")
    rows = MODULE.prediction_rows(
        "s2contact",
        {
            "dataset": "h2o",
            "expected_windows": 1,
            "predictions": {"s2contact": {"formal_roots": [str(prediction.parent)]}},
        },
        tmp_path / "migration",
        {"cache-a": "sequence:000000-000059"},
        direct_oss=True,
    )
    assert rows == {"sequence:000000-000059": prediction}


def test_method_block_writes_all_four_schemes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "process_window",
        lambda task: (
            task[2],
            {
                scheme: {"window_id": task[2], "excluded_frame_count": 0}
                for scheme in MODULE.SCHEMES
            },
        ),
    )
    monkeypatch.setattr(
        MODULE, "aggregate_windows", lambda rows, method: {"method": method, "windows": len(rows)}
    )
    result = MODULE.process_method_block((
        "h2o", "wilor", {"hand"}, {"window-a": tmp_path / "prediction"},
        {"window-a": {}},
        {scheme: {"window-a": np.zeros(60, dtype=bool)} for scheme in MODULE.SCHEMES},
        "hand.py", "scene.py", str(tmp_path / "output"),
    ))
    assert result[:3] == ("h2o", "wilor", 1)
    assert set(result[3]) == set(MODULE.SCHEMES)
    for scheme in MODULE.SCHEMES:
        assert (tmp_path / "output/h2o/wilor" / f"{scheme}_window_metrics.jsonl").is_file()


def test_unfiltered_mask_is_false(tmp_path):
    dataset = "h2o"
    ids = {"a", "b"}
    p975 = tmp_path / "p975" / dataset
    variants = tmp_path / "variants" / dataset
    p975.mkdir(parents=True)
    variants.mkdir(parents=True)
    payload = {"window_ids": ["a", "b"], "excluded": [[0] * 60, [1] + [0] * 59]}
    (p975 / "p97_5_mask.json").write_text(__import__("json").dumps(payload))
    for name in ("all8_p95", "temporal_p95_other_p97_5"):
        (variants / f"{name}_mask.json").write_text(__import__("json").dumps(payload))
    masks = MODULE.read_masks(
        {"mask_roots": {"p97_5": str(tmp_path / "p975"), "variants": str(tmp_path / "variants")}},
        dataset,
        ids,
    )
    assert not np.any(masks["unfiltered"]["a"])
    assert masks["p97_5"]["b"][0]


def test_visibility_uses_same_frame_mask_and_prediction_validity():
    prediction = {
        "hand_visibility": np.array([[[0.9]], [[0.9]], [[0.1]]]),
        "marker_visibility": np.array([[[0.1]], [[0.9]], [[0.9]]]),
        "hand_valid": np.array([[True], [False], [True]]),
    }
    target = {
        "joint_visibility_target": np.array([[[1]], [[1]], [[0]]]),
        "joint_visibility_mask": np.ones((3, 1, 1), dtype=bool),
        "marker_visibility_target": np.array([[[0]], [[1]], [[1]]]),
        "marker_visibility_mask": np.ones((3, 1, 1), dtype=bool),
    }
    metrics = MODULE.aggregate_visibility_window(
        prediction, target, np.array([True, False, True])
    )
    assert metrics["joint_visibility_valid_count"] == 2
    assert metrics["joint_visibility_f1"] == 1.0
    assert metrics["marker_visibility_valid_count"] == 2
    assert metrics["marker_visibility_f1"] == 1.0
