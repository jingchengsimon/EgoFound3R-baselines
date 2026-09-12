"""Runnable checks for masking, undefined metrics and exact-window validation."""
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
from formal_evaluation.run_result3_classification_retry import classify, keyed, masks


def test_classification():
    prediction = {'hand_valid': np.ones((60, 2), dtype=bool)}
    target = {}
    for prefix, count in [('joint', 21), ('marker', 195), ('vertex', 778)]:
        shape = (60, 2, count)
        prediction[prefix + '_contact_probability'] = np.ones(shape)
        prediction[prefix + '_contact_probability'][0] = 0
        target[prefix + '_contact_target'] = np.ones(shape)
        target[prefix + '_contact_mask'] = np.ones(shape, dtype=bool)
    keep = np.ones(60, dtype=bool); keep[0] = False
    result = classify(prediction, target, keep, 'contact')
    json.dumps(result)
    assert result['marker_contact_f1'] == 1
    assert result['joint_contact_valid_count'] == 59 * 2 * 21
    assert np.isnan(classify(prediction, target, np.zeros(60, bool), 'contact')['joint_contact_f1'])
    for prefix in ['joint', 'marker']:
        prediction['hand_visibility' if prefix == 'joint' else 'marker_visibility'] = prediction[prefix + '_contact_probability']
        for suffix in ['target', 'mask']:
            target[prefix + '_visibility_' + suffix] = target[prefix + '_contact_' + suffix]
    assert classify(prediction, target, keep, 'visibility')['joint_visibility_f1'] == 1


def test_identity():
    try:
        keyed([{'window_id': 'a'}, {'window_id': 'a'}], 'h2o')
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate identity accepted')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'mask.json'
        path.write_text(json.dumps({'window_ids': ['a'], 'excluded': [[False] * 60]}))
        source = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        assert set(masks(source, ['a'])) == {'a'}
        for bad in [{'path': str(path), 'sha256': 'bad'}]:
            try:
                masks(bad, ['a'])
            except ValueError:
                pass
            else:
                raise AssertionError('modified mask accepted')


if __name__ == '__main__':
    test_classification(); test_identity(); print('Result3 classification checks passed')
