"""Undefined camera alignment must not discard otherwise valid hand windows."""
import math

from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.run_dyn_hamr_full_hand_retry import require_metric_fields


def test_missing_world_alignment_counts_undefined_window():
    rows = [
        {'hand_left_mpjpe': 12.0, 'hand_left_wa_mpjpe': 9.0},
        {'hand_left_mpjpe': 15.0},
    ]
    for row in rows:
        require_metric_fields(row, True, ['mpjpe', 'wa_mpjpe'])
    summary = aggregate_windows(rows, method='dyn_hamr')
    assert summary['hand_left_mpjpe_count'] == 2
    assert summary['hand_left_wa_mpjpe_count'] == 1
    assert summary['hand_left_wa_mpjpe_undefined_window_count'] == 1
    assert math.isnan(rows[1]['hand_left_wa_mpjpe'])


def test_missing_geometry_still_fails():
    try:
        require_metric_fields({}, True, ['mpjpe'])
    except ValueError as error:
        assert str(error) == 'required metric not emitted: mpjpe'
    else:
        raise AssertionError('missing geometry metric accepted')
