import ast
import json
import tempfile
from pathlib import Path

import numpy as np


path = Path(__file__).resolve().parents[1] / "hand/adapters/run_hawor_baseline.py"
tree = ast.parse(path.read_text())
function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "require_native_camera_output"
)
namespace = {"np": np}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
require_native_camera_output = namespace["require_native_camera_output"]


loader_function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "load_hawor_window_input"
)

retry_function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "run_slam_with_keyframe_retry"
)
retry_classifier = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "is_keyframe_retryable_slam_error"
)
retry_namespace = {}
exec(
    compile(ast.Module(body=[retry_function, retry_classifier], type_ignores=[]), str(path), "exec"),
    retry_namespace,
)
run_slam_with_keyframe_retry = retry_namespace["run_slam_with_keyframe_retry"]
is_keyframe_retryable_slam_error = retry_namespace["is_keyframe_retryable_slam_error"]

droid_path = Path(__file__).resolve().parents[2] / "HaWoR/lib/pipeline/masked_droid_slam.py"
droid_tree = ast.parse(droid_path.read_text())
droid_nodes = [
    node for node in droid_tree.body
    if (isinstance(node, ast.ClassDef) and node.name == "InsufficientKeyframesError")
    or (isinstance(node, ast.FunctionDef) and node.name == "require_minimum_keyframes")
]
droid_namespace = {}
exec(compile(ast.Module(body=droid_nodes, type_ignores=[]), str(droid_path), "exec"), droid_namespace)
InsufficientKeyframesError = droid_namespace["InsufficientKeyframesError"]
require_minimum_keyframes = droid_namespace["require_minimum_keyframes"]

runner_path = Path(__file__).resolve().parents[1] / "run_hawor_native_camera_dataset.py"
runner_tree = ast.parse(runner_path.read_text())
resolver_function = next(
    node for node in runner_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "materialize_resolved_gt_index"
)
resolver_namespace = {"Path": Path, "json": json}
exec(compile(ast.Module(body=[resolver_function], type_ignores=[]), str(runner_path), "exec"), resolver_namespace)
materialize_resolved_gt_index = resolver_namespace["materialize_resolved_gt_index"]


def test_load_hawor_rgb_only_window(tmp_path):
    rgb = tmp_path / "frame.png"
    rgb.write_bytes(b"rgb")
    record = {
        "input_kind": "rgb_intrinsics_only_no_geometry",
        "dataset": "h2o", "sequence_id": "seq", "window_id": "window", "cache_id": "cache",
        "frame_ids": ["0"], "rgb_paths": [str(rgb)],
        "intrinsics": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
    }
    path = tmp_path / "window_input.json"
    path.write_text(__import__("json").dumps(record))
    loader_namespace = {"Path": Path, "json": __import__("json"), "load_window_input": None}
    exec(compile(ast.Module(body=[loader_function], type_ignores=[]), str(path), "exec"), loader_namespace)
    assert loader_namespace["load_hawor_window_input"](path) == record


def test_require_native_camera_output_rejects_fallback_and_invalid_values():
    camera = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    require_native_camera_output(
        {"camera_c2w": camera, "camera_valid": np.ones(2, dtype=bool)},
        {"slam_failed_identity_fallback": False},
    )
    for arrays, detail in (
        ({"camera_c2w": camera, "camera_valid": np.zeros(2, dtype=bool)},
         {"slam_failed_identity_fallback": False}),
        ({"camera_c2w": camera, "camera_valid": np.ones(2, dtype=bool)},
         {"slam_failed_identity_fallback": True}),
    ):
        try:
            require_native_camera_output(arrays, detail)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid native camera output was accepted")


def test_materialize_resolved_gt_index_uses_portable_cache_root(tmp_path):
    cache = tmp_path / "gt_cache"
    payload = cache / "h2o" / "window.npz"
    metadata = cache / "h2o" / "window.json"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"npz")
    metadata.write_text("{}")
    source = cache / "index.jsonl"
    source.write_text(json.dumps({
        "window_id": "window",
        "array_path": "/old/node/gt_cache/h2o/window.npz",
        "metadata_path": "/old/node/gt_cache/h2o/window.json",
    }) + "\n")
    destination = tmp_path / "resolved.jsonl"
    materialize_resolved_gt_index(source, destination)
    row = json.loads(destination.read_text())
    assert row["array_path"] == str(payload)
    assert row["metadata_path"] == str(metadata)


def test_droid_keyframe_guard_rejects_single_keyframe():
    class Counter:
        value = 1

    class Video:
        counter = Counter()

    class Droid:
        video = Video()

    try:
        require_minimum_keyframes(Droid())
    except InsufficientKeyframesError as error:
        assert "only 1 keyframes" in str(error)
    else:
        raise AssertionError("single-keyframe DROID run was accepted")


def test_slam_retries_only_insufficient_keyframes_at_zero_threshold():
    calls = []

    def fake_run(*_args, **kwargs):
        calls.append(kwargs["filter_thresh"])
        if len(calls) == 1:
            raise InsufficientKeyframesError("only one")
        return "droid", "trajectory"

    droid, traj, detail = run_slam_with_keyframe_retry(
        fake_run,
        lambda error: is_keyframe_retryable_slam_error(error, InsufficientKeyframesError),
        [], None, None, None,
    )
    assert (droid, traj) == ("droid", "trajectory")
    assert calls == [2.4, 0.0]
    assert detail["filter_threshold"] == 0.0
    assert [row["status"] for row in detail["attempts"]] == [
        "insufficient_keyframes", "success"
    ]


def test_slam_retry_classifier_supports_legacy_empty_graph_error():
    error = ValueError("not enough values to unpack (expected 2, got 0)")
    assert is_keyframe_retryable_slam_error(error)
    assert not is_keyframe_retryable_slam_error(ValueError("different failure"))


if __name__ == "__main__":
    test_require_native_camera_output_rejects_fallback_and_invalid_values()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_load_hawor_rgb_only_window(root)
        test_materialize_resolved_gt_index_uses_portable_cache_root(root)
    test_droid_keyframe_guard_rejects_single_keyframe()
    test_slam_retries_only_insufficient_keyframes_at_zero_threshold()
    test_slam_retry_classifier_supports_legacy_empty_graph_error()
