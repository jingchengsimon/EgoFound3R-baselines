"""GT K seven-table metrics on exact Dyn windows, reusing frozen full-window fits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
from formal_evaluation import run_ego_distance_mae as distance
from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.run_result3_classification_retry import classify


def initialize(spec):
    global HAND, SCENE
    distance.initialize(spec)
    HAND = distance.load('dyn100_frozen_hand', spec['hand_script'])
    SCENE = distance.load('dyn100_frozen_scene', spec['scene_script'])


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def compute(row):
    directory = Path(row['prediction_dir'])
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['frame_ids'] == row['frame_ids'] and len(row['frame_ids']) == 60
    keep = np.ones(60, dtype=bool)
    hand = arrays(row['frozen_hand'])
    _, hand_metrics = HAND.aggregate_scheme(
        hand['frame'][None], hand['pair'][None], hand['triplet'][None], ~keep[None]
    )
    metrics = dict(hand_metrics[0])
    metrics.update(SCENE.aggregate_scene_window(arrays(row['frozen_scene']), keep))
    prediction = arrays(directory / 'predictions.npz')
    target = arrays(row['contact_array'])
    geometry = distance.GEOMETRY.query_geometry(prediction)
    collected = {}
    assert len(row['geometry_paths']) == 60
    with distance.BVH.install(distance.GEOMETRY.surface_kernel()) as accelerator:
        for frame, path in enumerate(row['geometry_paths']):
            accelerator.clear()
            values = distance.GEOMETRY.frame_geometry_metrics(
                {k: v[frame] for k, v in geometry.items()},
                prediction['hand_valid'][frame], arrays(path), distance.FACES, 'cpu'
            )
            for key, value in values.items():
                collected.setdefault(key, []).append(value)
    overlay = {k: np.stack(v) for k, v in collected.items()}
    prediction['vertex_contact_probability'] = overlay['vertex_contact_target']
    metrics.update(classify(prediction, target, keep, 'contact'))
    metrics.update(classify(prediction, arrays(row['visibility_array']), keep, 'visibility'))
    for prefix in ('joint', 'marker', 'vertex'):
        overlay[prefix + '_contact_probability'] = prediction[prefix + '_contact_probability']
    metrics.update(distance.ERRORS.window_distance_errors(overlay, target, keep))
    return {'dataset': row['dataset'], 'window_id': row['window_id'],
            'frame_ids': row['frame_ids'], **metrics}


def read_manifest(spec):
    raw = Path(spec['manifest']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['manifest_sha256']:
        raise ValueError('manifest digest mismatch')
    rows = json.loads(raw)
    keys = {(r['dataset'], r['window_id']) for r in rows}
    counts = {d: sum(r['dataset'] == d for r in rows) for d in spec['expected_windows']}
    if len(keys) != len(rows) or counts != spec['expected_windows'] or len(rows) != 100:
        raise ValueError('Dyn original100 window coverage mismatch')
    if any(len(r['frame_ids']) != 60 or len(r['geometry_paths']) != 60 for r in rows):
        raise ValueError('full-window frame coverage mismatch')
    return rows


def run(spec):
    rows = read_manifest(spec)
    root = Path(spec['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    grouped = {d: [] for d in spec['expected_windows']}
    with ProcessPoolExecutor(max_workers=spec['workers'],
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(spec,)) as pool:
        with (root / 'window_metrics.jsonl').open('x') as output:
            for i, row in enumerate(pool.map(compute, rows, chunksize=1), 1):
                grouped[row['dataset']].append(row)
                output.write(json.dumps(row) + '\n')
                output.flush()
                print(json.dumps({'completed': i, 'total': 100, 'dataset': row['dataset']}), flush=True)
    report = {
        'status': 'complete', 'method': 'stride5_gt_k_dyn100', 'windows': 100,
        'selection': 'Exact Dyn original100 windows; all 60 frames, unfiltered',
        'manifest_sha256': spec['manifest_sha256'],
        'source_window_manifest': spec['source_window_manifest'],
        'hand_scene_fit': 'Reuse existing frozen full-window GT K artifacts; no refit',
        'world_pose_source': 'predicted_camera_c2w',
        'distance_reference': 'GT object union GT opposite hand for both query sets',
        'vertex_contact': 'derived 14mm geometry; not a native contact head',
        'datasets': {d: aggregate_windows(v, method='stride5_gt_k_dyn100') for d, v in grouped.items()},
    }
    for name in ('report.json', 'summary.json'):
        (root / name).write_text(json.dumps(report, indent=2))
    (root / 'COMPLETE').write_text('complete\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()))
