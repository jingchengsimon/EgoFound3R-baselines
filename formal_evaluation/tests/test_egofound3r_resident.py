import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from formal_evaluation.run_egofound3r_relay import transfer


class ResidentTest(unittest.TestCase):
    def test_runtime_loads_once_and_rejects_wrong_checkpoint(self):
        path = Path(__file__).parents[1] / 'scene/adapters/run_egofound3r_baseline.py'
        fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_runtime')
        model = Mock(); model.to.return_value = model
        build = Mock(return_value=model); load = Mock()
        config = SimpleNamespace(marker_runtime=SimpleNamespace(backend='vggt_omega'))
        def verify(path, expected):
            if expected != 'verified':
                raise ValueError('checkpoint mismatch')
            return expected
        ns = dict(lru_cache=lru_cache, Path=Path, _verified_checkpoint_sha256=verify,
                  _load_training_config_compat=lambda p: config,
                  torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
                  build_runtime_marker_model=build, load_marker_model_weights=load,
                  active_marker_model_config=lambda p: p, marker_model_floating_dtype=lambda m: "bf16")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
        args = ('config', 'checkpoint', 'backbone', 'verified', 'cuda:0', 12, 34)
        self.assertIs(ns['_runtime'](*args), ns['_runtime'](*args))
        self.assertEqual(build.call_count, 1); self.assertEqual(load.call_count, 1)
        with self.assertRaises(ValueError):
            ns['_runtime'](*args[:3], 'wrong', *args[4:])
        self.assertEqual(build.call_count, 1)

    def test_ablation_bf16_uses_strict_native_loader_and_separate_cache_entry(self):
        path = Path(__file__).parents[1] / 'scene/adapters/run_egofound3r_baseline.py'
        fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_runtime')
        model = Mock(); model.to.return_value = model
        config = SimpleNamespace(marker_runtime=SimpleNamespace(backend='vggt_omega'))
        load = Mock()
        dtype = Mock(side_effect=ValueError('mixed MANO FP32 and BF16 is intentional'))
        build = Mock(return_value=model)
        ns = dict(lru_cache=lru_cache, Path=Path, _verified_checkpoint_sha256=lambda p, expected: expected,
                  _load_training_config_compat=lambda p: config,
                  torch=SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(is_available=lambda: True)),
                  build_runtime_marker_model=build, load_marker_model_weights=load,
                  active_marker_model_config=lambda p: p, marker_model_floating_dtype=dtype)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
        args = ('config', 'checkpoint', 'backbone', 'verified', 'cuda:0', 12, 34)
        result = ns['_runtime'](*args, 'ablation_bf16')
        self.assertEqual(result[-1], 'bf16')
        model.to.assert_any_call(dtype='bf16')
        load.assert_called_once_with(model, Path('checkpoint'), model_config=config, resume_mode='weights')
        dtype.assert_not_called()
        self.assertIs(result, ns['_runtime'](*args, 'ablation_bf16'))
        with self.assertRaisesRegex(ValueError, 'mixed MANO'):
            ns['_runtime'](*args)  # Switching mode cannot reuse the ablation cache.
        self.assertEqual(build.call_count, 2)
        load.side_effect = ValueError('checkpoint state dtype mismatch')
        with self.assertRaisesRegex(ValueError, 'checkpoint state dtype mismatch'):
            ns['_runtime'](*args[:3], 'another-checkpoint', *args[4:], 'ablation_bf16')
        with self.assertRaisesRegex(ValueError, 'unsupported runtime_mode'):
            ns['_runtime'](*args, 'unknown')

    def test_transfer_preserves_source_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'source'; source.mkdir()
            (source/'predictions.npz').write_bytes(b'prediction data')
            target = root/'target'; hashes = transfer(source, target)
            self.assertEqual((source/'predictions.npz').read_bytes(), (target/'predictions.npz').read_bytes())
            self.assertIn('predictions.npz', hashes)
            with self.assertRaises(FileExistsError):
                transfer(source, target)
            self.assertTrue(source.exists())

    def test_mixed_checkpoint_is_strictly_loaded_before_bf16_conversion(self):
        path = Path(__file__).parents[1] / 'scene/adapters/run_egofound3r_baseline.py'
        fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_runtime')
        model = SimpleNamespace(dtype='fp32')
        aligned = Mock()
        def to(*args, **kwargs):
            model.dtype = kwargs.get('dtype', model.dtype)
            return model
        model.to = to; model.eval = lambda: None
        def load(*args, **kwargs):
            self.assertEqual(model.dtype, 'fp32')
            aligned.assert_called_once_with(model, 'checkpoint')
            self.assertEqual(kwargs, dict(model_config=config, resume_mode='weights'))
        config = SimpleNamespace(marker_runtime=SimpleNamespace(backend='vggt_omega'))
        loader = Mock(side_effect=load)
        ns = dict(lru_cache=lru_cache, Path=Path, _verified_checkpoint_sha256=lambda p, s: s,
                  _load_training_config_compat=lambda p: config,
                  torch=SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(is_available=lambda: True)),
                  build_runtime_marker_model=lambda c: model, load_marker_model_weights=loader,
                  _align_checkpoint_dtypes=aligned,
                  active_marker_model_config=lambda c: c)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
        args = ('config', 'checkpoint', 'backbone', 'verified', 'cuda:0', 12, 34,
                'ablation_checkpoint_dtypes_bf16')
        self.assertEqual(ns['_runtime'](*args)[-1], 'bf16')
        self.assertEqual(model.dtype, 'bf16'); loader.assert_called_once()
        model.dtype = 'fp32'; loader.side_effect = ValueError('checkpoint contract mismatch')
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            ns['_runtime'](*args[:3], 'another', *args[4:])
        self.assertEqual(model.dtype, 'fp32')

    def test_dtype_alignment_uses_saved_parameters_and_buffers_only(self):
        path = Path(__file__).parents[1] / 'scene/adapters/run_egofound3r_baseline.py'
        fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_align_checkpoint_dtypes')
        def tensor(dtype):
            value = SimpleNamespace(dtype=dtype, is_floating_point=lambda: dtype != 'int64')
            value.data = SimpleNamespace(to=lambda *, dtype: SimpleNamespace(dtype=dtype))
            return value
        state = dict(parameter=tensor('fp32'), buffer=tensor('bf16'), index=tensor('int64'), backbone=tensor('fp32'))
        original_index, original_backbone = state['index'].data, state['backbone'].data
        saved = dict(parameter=tensor('bf16'), buffer=tensor('fp32'), index=tensor('int64'))
        load = Mock(return_value={'model': saved})
        ns = dict(torch=SimpleNamespace(load=load))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
        ns['_align_checkpoint_dtypes'](SimpleNamespace(state_dict=lambda **kw: state), 'checkpoint')
        self.assertEqual(state['parameter'].data.dtype, 'bf16')
        self.assertEqual(state['buffer'].data.dtype, 'fp32')
        self.assertIs(state['index'].data, original_index)
        self.assertIs(state['backbone'].data, original_backbone)
        load.assert_called_once_with('checkpoint', map_location='cpu', weights_only=False, mmap=True)

if __name__ == '__main__':
    unittest.main()
