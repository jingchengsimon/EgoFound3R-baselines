import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from formal_evaluation.hand.adapters.rgb_cache import atomic_publish, read_rgb


class RGBPublicationTest(unittest.TestCase):
    def test_transient_bad_bytes_retry_without_auto_install_hook(self):
        out = io.BytesIO()
        Image.new('RGB', (3, 2), (12, 34, 56)).save(out, format='PNG')
        with patch.object(Path, 'read_bytes', side_effect=[b'partial', out.getvalue()]), patch('PIL.Image.open', side_effect=AssertionError('patched reader must not run')), patch('time.sleep'):
            rgb = read_rgb('cache.png')
        np.testing.assert_array_equal(rgb, np.full((2, 3, 3), [12, 34, 56], dtype=np.uint8))

    def test_permanent_corruption_remains_an_error(self):
        with patch.object(Path, 'read_bytes', return_value=b'broken'), patch('time.sleep'):
            with self.assertRaisesRegex(OSError, 'after 4 reads'):
                read_rgb('broken.png')

    def test_reader_never_observes_partial_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'cache.png'
            target.write_bytes(b'old complete')
            def writer(stream):
                stream.write(b'new')
                stream.flush()
                self.assertEqual(target.read_bytes(), b'old complete')
                stream.write(b' complete')
            atomic_publish(target, writer)
            self.assertEqual(target.read_bytes(), b'new complete')
            def broken(stream):
                stream.write(b'partial')
                raise RuntimeError('writer failed')
            with self.assertRaises(RuntimeError):
                atomic_publish(target, broken)
            self.assertEqual(target.read_bytes(), b'new complete')
            self.assertEqual(list(Path(directory).iterdir()), [target])

if __name__ == '__main__':
    unittest.main()
