"""Optimized w2c inversion and official kept-span/scoring identity checks."""
import ast
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1] / 'hand/adapters/run_dyn_hamr_window.py'
tree = ast.parse(path.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_canonical_cameras')
namespace = {'np': np}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)


def test_camera_export():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotations = np.tile(rotation, (2, 3, 1, 1))
    translations = np.tile([2., 3., 4.], (2, 3, 1))
    poses, valid = namespace['_canonical_cameras'](rotations, translations, 5, 1, [0, 1, 3, 4])
    assert valid.tolist() == [False, True, True, False]
    assert np.isnan(poses[0]).all()
    point = np.array([4., 5., 6.]); camera = rotation @ point + translations[0, 0]
    assert np.allclose(poses[1, :3, :3] @ camera + poses[1, :3, 3], point)
    translations[1, 0, 0] += 1
    try:
        namespace['_canonical_cameras'](rotations, translations, 5, 1, [1])
    except ValueError:
        pass
    else:
        raise AssertionError('inconsistent per-track camera accepted')


if __name__ == '__main__':
    test_camera_export(); print('Dyn-HaMR camera export checks passed')
