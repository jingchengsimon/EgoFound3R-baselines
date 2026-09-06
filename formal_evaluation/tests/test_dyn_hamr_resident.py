"""Resident frontend identity and independent window dispatch checks."""
import ast
from functools import lru_cache
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / 'hand/adapters/run_dyn_hamr_window.py'

class ResidentTests(unittest.TestCase):
    def test_frozen_networks_loaded_once_per_identity(self):
        tree = ast.parse(SOURCE.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_frontend')
        model = Mock(); model.to.return_value = model; model.eval.return_value = model
        hamer = SimpleNamespace(load_hamer=Mock(return_value=(model, 'config')))
        detector = Mock(); factory = Mock(return_value=detector)
        droid = SimpleNamespace(load_network=Mock(return_value='network'))
        namespace = dict(lru_cache=lru_cache, Path=Path, _module=Mock(return_value=hamer))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        with patch.dict(sys.modules, ultralytics=SimpleNamespace(YOLO=factory), droid=SimpleNamespace(Droid=droid)):
            load = namespace['_frontend']; key = ('source', 'checkpoint', 'detector', 'droid', 'cuda:0')
            self.assertIs(load(*key), load(*key))
            self.assertEqual(hamer.load_hamer.call_count, 1)
            self.assertEqual(droid.load_network.call_count, 1)
            load('source', 'other-checkpoint', *key[2:])
            self.assertEqual(hamer.load_hamer.call_count, 2)
        self.assertNotIn('run_opt', ast.unparse(function))

if __name__ == '__main__':
    unittest.main()
