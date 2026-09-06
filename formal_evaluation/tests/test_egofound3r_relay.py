from pathlib import Path
from unittest.mock import patch

import pytest

from formal_evaluation.run_egofound3r_relay import transfer


def test_verified_transfer_preserves_source_and_rejects_overwrite(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'predictions.npz').write_bytes(b'prediction data')
    target = tmp_path / 'oss' / 'window'
    hashes = transfer(source, target)
    assert (source / 'predictions.npz').read_bytes() == (target / 'predictions.npz').read_bytes()
    assert len(hashes['predictions.npz']) == 64
    with pytest.raises(FileExistsError):
        transfer(source, target)
    with patch('formal_evaluation.run_egofound3r_relay.digest', side_effect=['source-hash', 'bad-copy']):
        with pytest.raises(ValueError, match='HASH_MISMATCH'):
            transfer(source, tmp_path / 'oss' / 'bad-window')
    assert source.is_dir()
    assert not (tmp_path / 'oss' / 'bad-window').exists()
