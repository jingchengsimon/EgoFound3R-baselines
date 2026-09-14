import ast
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


if __name__ == "__main__":
    test_require_native_camera_output_rejects_fallback_and_invalid_values()
    with tempfile.TemporaryDirectory() as directory:
        test_load_hawor_rgb_only_window(Path(directory))
