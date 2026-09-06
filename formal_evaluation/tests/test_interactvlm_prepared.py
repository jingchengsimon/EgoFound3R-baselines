"""CPU-only tests of native contact topology projection and its rejection gates."""
import ast
from pathlib import Path
import unittest
import numpy as np

source = Path(__file__).parents[1] / 'contact/run_interactvlm_prepared.py'
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'hand_probabilities')
namespace = {'np': np, 'marker_vertex_ids_195': lambda: np.arange(195)}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
project = namespace['hand_probabilities']


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.mapping = {'left': list(range(778)), 'right': list(range(778, 1556))}
        vertices = np.zeros((1, 2, 778, 3)); vertices[..., 0] = np.arange(778)
        self.targets = {'hand_valid': np.array([[True, False]]),
                        'hand_vertices_camera': vertices,
                        'hand_joints_camera': vertices[:, :, :21].copy()}
        self.scores = np.arange(6890, dtype=np.float32)[None] / 6890

    def test_vertex_identity_and_invalid_hand(self):
        out = project(self.scores, self.targets, self.mapping)
        np.testing.assert_allclose(out['marker_contact_probability'][0, 0], self.scores[0, :195])
        self.assertEqual(out['joint_contact_probability'][0, 0, 4], self.scores[0, 745])
        self.assertTrue(np.isnan(out['joint_contact_probability'][0, 1]).all())
        self.targets['hand_valid'][0, 1] = True
        out = project(self.scores, self.targets, self.mapping)
        np.testing.assert_allclose(out['marker_contact_probability'][0, 1], self.scores[0, 778:973])

    def test_reject_invalid_native_and_mapping(self):
        for bad in (self.scores[:, :6889], self.scores * np.nan, self.scores + 2):
            with self.assertRaises(ValueError): project(bad, self.targets, self.mapping)
        self.mapping['left'] = [0] * 778
        with self.assertRaises(ValueError): project(self.scores, self.targets, self.mapping)


if __name__ == '__main__':
    unittest.main()
