"""Exact-window selection and finite-side aggregation for matched 100-window rows."""
import json
import math

from formal_evaluation.aggregate_ego_matched100 import combined_hand, keyed, selected_predictions, summarize


def test_exact_window_and_frame_identity():
    expected = {'w0': [1, 2], 'w1': [3, 4]}
    rows = [{'dataset': 'h2o', 'window_id': 'w0', 'frame_ids': [1, 2]},
            {'dataset': 'h2o', 'window_id': 'w1', 'frame_ids': [3, 4]}]
    assert set(keyed(rows, expected, dataset='h2o', frame_ids=True)) == set(expected)
    wrong = [*rows[:-1], {**rows[-1], 'frame_ids': [3, 5]}]
    try:
        keyed(wrong, expected, dataset='h2o', frame_ids=True)
    except ValueError as error:
        assert 'frame identity' in str(error)
    else:
        raise AssertionError('mismatched frames accepted')


def test_null_window_is_undefined_and_sides_are_count_weighted():
    summary = summarize([
        {'hand_left_mpjpe': 10.0, 'hand_right_mpjpe': 20.0,
         'hand_left_mpjae': 2000.0},
        {'hand_left_mpjpe': None, 'hand_right_mpjpe': 40.0,
         'hand_left_mpjae': None},
    ], method='egofound3r_stride5')
    assert summary['hand_left_mpjpe_undefined_window_count'] == 1
    assert math.isclose(combined_hand(summary, 'mpjpe'), (10 + 20 + 40) / 3)
    assert combined_hand(summary, 'mpjae') == 2.0


def test_prediction_index_selects_original_windows_from_full_gt(tmp_path):
    source = tmp_path / 'dyn'
    source.mkdir()
    (source / 'COMPLETE').write_text('complete\n')
    prediction = source / 'prediction'
    prediction.mkdir()
    (prediction / 'metadata.json').write_text(json.dumps({'frame_ids': [3, 4]}))
    index = source / 'predictions.jsonl'
    index.write_text(json.dumps({'dataset': 'h2o', 'window_id': 'cache-w1',
                                 'prediction_dir': str(prediction)}) + '\n')
    selected, hashes = selected_predictions([str(index)], {'w0': [1, 2], 'w1': [3, 4]},
                                             {'cache-w1': 'w1'}, 'h2o', 1)
    assert selected == {'w1'}
    assert str(index) in hashes
