"""Recompute explicitly selected hand metrics from verified taskctl catalogs.

W: HaWoR/WHAM first-two-frame Sim(3), one transform per 60-frame window.
Temporal: VideoPose3D first differences and VIBE/WHAM second differences.
The caller must specify the coordinate and time convention; WA is never emitted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from egocentric_metrics import temporal_point_errors, world_aligned_mpjpe
from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.common.schema import validate_comparison_output


def selected_metrics(pred, gt, valid, *, world_pred=None, world_gt=None,
                     roots_pred=None, roots_gt=None, temporal=False, fps=30.0):
    """One side/geometry; retain counts and NaN instead of imputing missing data."""
    result = {}
    if world_pred is not None:
        mask = np.broadcast_to(valid[:, None], world_pred.shape[:2])
        errors = world_aligned_mpjpe(world_pred, world_gt, joint_mask=mask,
                                    mode="first2", chunk_length=len(pred), unit_scale=1000)
        good = np.isfinite(errors)
        result.update(w=float(errors[good].mean()) if good.any() else float("nan"),
                      w_frame_count=int(good.sum()))
    if temporal:
        if roots_pred is not None:
            pred = pred - roots_pred[:, None]
            gt = gt - roots_gt[:, None]
        values = temporal_point_errors(pred, gt, valid.copy(), fps=fps, unit_scale=1000)
        result.update(velocity=values['velocity_error'], acceleration=values['acceleration_error'],
                      velocity_pair_count=values['velocity_pair_count'],
                      acceleration_triplet_count=values['acceleration_triplet_count'])
    return result


def read_hand_prediction(path):
    metadata = json.loads((path / 'metadata.json').read_text())
    with np.load(path / 'predictions.npz', allow_pickle=False) as z:
        names = [k for k in z.files if k.startswith('hand_') or k in ('camera_c2w', 'camera_valid')]
        arrays = {k: z[k] for k in names}
    # Validate the complete hand/camera subset without allocating unrelated depth maps.
    subset = dict(metadata, capabilities={k: v for k, v in metadata['capabilities'].items()
                                         if k.startswith('hand_') or k in ('camera_c2w', 'camera_valid')})
    validate_comparison_output(subset, arrays)
    return metadata, arrays


def main(spec):
    from formal_evaluation.evaluate_six_dataset import (
        _prediction_geometry, _prediction_geometry_camera, _target_geometry, _camera_to_world)

    root = Path(spec['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'protocol.json').write_text(json.dumps(spec['protocol'], indent=2))
    reports = {}
    for catalog in spec['catalogs']:
        dataset = catalog['dataset']
        index = Path(catalog['gt_index'])
        raw_index = index.read_bytes()
        assert hashlib.sha256(raw_index).hexdigest() == catalog['gt_index_sha256']
        rows = [json.loads(line) for line in raw_index.splitlines() if line]
        assert len(rows) == catalog['expected_windows']
        aliases = {str(row['cache_id']): str(row['window_id']) for row in rows}
        wanted = {str(row['window_id']) for row in rows}
        methods = {}
        for method in spec['methods']:
            found = {}
            for raw_root in catalog['predictions'][method]['formal_roots']:
                for path in Path(raw_root).iterdir():
                    if not path.is_dir() or not (path / 'metadata.json').is_file() or not (path / 'predictions.npz').is_file():
                        continue
                    metadata = json.loads((path / 'metadata.json').read_text())
                    wid = aliases.get(str(metadata['window_id']), str(metadata['window_id']))
                    if wid not in wanted:
                        continue
                    assert metadata['dataset'] == dataset and wid not in found, (method, wid)
                    found[wid] = path
            assert set(found) == wanted, (dataset, method, len(found), len(wanted))
            out = root / dataset / method
            out.mkdir(parents=True)
            results = []
            with (out / 'window_metrics.jsonl').open('x') as stream:
                for i, row in enumerate(rows):
                    path = found[str(row['window_id'])]
                    metadata, prediction = read_hand_prediction(path)
                    assert metadata['frame_ids'] == row['frame_ids']
                    target_path = index.parent / str(row['array_path']).split('/gt_cache/', 1)[1]
                    with np.load(target_path, allow_pickle=False) as z:
                        target = {k: z[k] for k in z.files if k.startswith('hand_') or k in ('camera_c2w', 'camera_valid')}
                    frame_count = len(row['frame_ids'])
                    assert frame_count == 60
                    result = {'window_id': row['window_id'], 'prediction_dir': str(path),
                              'frame_ids': row['frame_ids'], 'temporal_fps_metadata': metadata.get('temporal_fps'),
                              'schema_validation': 'hand_camera_subset'}
                    camera_joints, _ = _prediction_geometry_camera(prediction, 'joint')
                    for granularity, field, pos, vel, acc in (
                        ('joint', 'joints', 'mpjpe', 'mpjve', 'mpjae'),
                        ('marker', 'markers', 'mpmpe', 'mpmve', 'mpmae'),
                        ('vertex', 'vertices', 'mpvpe', 'mpvve', 'mpvae')):
                        points, coordinate, provenance = _prediction_geometry(prediction, granularity)
                        if points is None:
                            continue
                        valid = prediction['hand_valid'] & target['hand_valid']
                        world_pred = world_gt = None
                        if 'w' in spec['metrics'] and method != 'pad_hand':
                            pose = prediction.get('camera_c2w')
                            if coordinate == 'world':
                                world_pred = points
                                source = 'native_world'
                            else:
                                if pose is None:
                                    assert method == 'wilor', 'GT camera fallback is explicit for WiLoR only'
                                    pose = target['camera_c2w']
                                    source = 'gt_camera_c2w_oracle'
                                else:
                                    source = 'predicted_camera_c2w'
                                world_pred = _camera_to_world(points, pose)
                            world_gt = _target_geometry(target, field, 'world')
                            camera_valid = target['camera_valid'].copy()
                            if 'camera_valid' in prediction:
                                camera_valid &= prediction['camera_valid']
                            world_pred = np.where(camera_valid[:, None, None, None], world_pred, np.nan)
                            result['hand_' + granularity + '_world_source'] = source
                        temporal = 'temporal' in spec['metrics']
                        metric_pred, _ = _prediction_geometry_camera(prediction, granularity)
                        metric_gt = _target_geometry(target, field, 'camera')
                        if metric_pred is None:
                            metric_pred = points
                            temporal = False
                        result['hand_' + granularity + '_geometry_provenance'] = provenance
                        for side, name in enumerate(('left', 'right')):
                            prefix = 'hand_' + name + '_' + ('' if granularity == 'joint' else granularity + '_')
                            roots_pred = roots_gt = None
                            if temporal:
                                assert spec['protocol']['temporal_coordinate'] == 'camera_wrist_relative'
                                assert camera_joints is not None
                                roots_pred = camera_joints[:, side, 0]
                                roots_gt = target['hand_joints_camera'][:, side, 0]
                            values = selected_metrics(metric_pred[:, side], metric_gt[:, side], valid[:, side],
                                world_pred=None if world_pred is None else world_pred[:, side],
                                world_gt=None if world_gt is None else world_gt[:, side],
                                roots_pred=roots_pred, roots_gt=roots_gt, temporal=temporal,
                                fps=spec['protocol'].get('fps', 30.0))
                            names = {'w': 'w_' + pos, 'w_frame_count': 'w_' + pos + '_frame_count',
                                     'velocity': vel, 'acceleration': acc, 'velocity_pair_count': vel + '_pair_count',
                                     'acceleration_triplet_count': acc + '_triplet_count'}
                            result.update({prefix + names[k]: v for k, v in values.items()})
                    assert not any('_wa_' in key for key in result)
                    results.append(result)
                    stream.write(json.dumps(result) + '\n')
                    if (i + 1) % 50 == 0:
                        stream.flush()
                        print(json.dumps({'dataset': dataset, 'method': method, 'processed': i + 1, 'total': len(rows)}), flush=True)
            metrics = aggregate_windows(results, method=method)
            # Official evaluators concatenate valid frame errors, rather than
            # assigning equal weight to windows with different visibility.
            for key in list(metrics):
                if not key.endswith('_mean'):
                    continue
                stem = key[:-5]
                suffix = ('_frame_count' if '_w_' in stem and stem.endswith(('mpjpe', 'mpmpe', 'mpvpe')) else
                          '_pair_count' if stem.endswith(('mpjve', 'mpmve', 'mpvve')) else
                          '_triplet_count' if stem.endswith(('mpjae', 'mpmae', 'mpvae')) else None)
                if suffix is None:
                    continue
                samples = [(r[stem], r.get(stem + suffix, 0)) for r in results
                           if stem in r and np.isfinite(r[stem]) and r.get(stem + suffix, 0) > 0]
                count = sum(n for _, n in samples)
                metrics[stem + '_sample_count'] = count
                metrics[stem + '_window_macro_mean'] = metrics[key]
                metrics[key] = sum(v * n for v, n in samples) / count if count else float('nan')
            report = {'gt_windows': len(rows), 'protocol': spec['protocol'],
                      'methods': {method: {'missing_prediction_windows': 0, 'datasets': {dataset: metrics}}}}
            report_path = out / 'report.json'
            report_path.write_text(json.dumps(report, indent=2))
            validation = {'status': 'complete', 'windows': len(rows), 'source_catalog_run_id': catalog['run_id'],
                          'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                          'metrics': spec['metrics'], 'wa_recomputed': False}
            (out / 'validation_report.json').write_text(json.dumps(validation, indent=2))
            (out / 'COMPLETE').write_text(json.dumps(validation))
            methods[method] = str(report_path)
            print(json.dumps({'status': 'method_complete', 'dataset': dataset, 'method': method, **validation}), flush=True)
        reports[dataset] = methods
    (root / 'summary.json').write_text(json.dumps({'status': 'complete', 'reports': reports, 'protocol': spec['protocol']}, indent=2))
    (root / 'COMPLETE').write_text('complete\n')
    print(json.dumps({'status': 'complete', 'reports': reports}), flush=True)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
