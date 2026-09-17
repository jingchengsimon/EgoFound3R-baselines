"""Validate and derive registered contact GT using the formal CPU surface kernel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import time
import zipfile
import numpy as np


def kernel():
    path = Path(__file__).parent / 'contact/geometry_metrics.py'
    spec = importlib.util.spec_from_file_location('cpu_contact_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_check():
    k = kernel()
    triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32)
    faces = np.array([[0, 1, 2]], np.int64)
    points = np.array([[.2, .2, .013], [.2, .2, .015], [np.nan, 0., 0.]], np.float32)
    query = {'vertex': np.stack([points, points])}
    scene = {'object_vertices': triangle, 'object_faces': faces,
             'hand_vertices': np.stack([triangle, triangle]), 'hand_valid': np.array([True, False])}
    a = k.frame_geometry_metrics(query, [True, False], scene, faces, 'cpu')
    np.testing.assert_allclose(a['vertex_contact_distance'][0, :2], [.013, .015], atol=1e-6)
    assert a['vertex_contact_target'][0, :2].tolist() == [1., 0.]
    assert not a['vertex_contact_mask'][0, 2] and np.isnan(a['vertex_contact_target'][0, 2])
    assert not a['vertex_contact_mask'][1].any() and np.isnan(a['vertex_contact_distance'][1]).all()
    # Opposite-hand surface reduces 15mm to 5mm; the query hand is never a surface.
    scene['hand_valid'][1] = True
    scene['hand_vertices'][1, :, 2] = .010
    a = k.frame_geometry_metrics(query, [True, False], scene, faces, 'cpu')
    np.testing.assert_allclose(a['vertex_contact_distance'][0, :2], [.003, .005], atol=1e-6)
    assert a['vertex_contact_target'][0, :2].tolist() == [1., 1.]
    scene['object_vertices'] = np.empty((0, 3), np.float32)
    scene['object_faces'] = np.empty((0, 3), np.int64)
    a = k.frame_geometry_metrics(query, [True, False], scene, faces, 'cpu')
    assert not a['vertex_contact_mask'].any() and np.isnan(a['vertex_contact_distance']).all()
    return {'device': 'cpu', 'surface_distance': 'passed', 'threshold_14mm': 'passed',
            'opposite_hand_union': 'passed', 'unknown_and_invalid_nan': 'passed'}


def records(spec):
    result = []
    for dataset, source in spec['datasets'].items():
        raw = Path(source['gt_index']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['gt_index_sha256']
        gt_rows = [json.loads(line) for line in raw.splitlines()]
        gt = {r['window_id']: r for r in gt_rows}
        assert len(gt) == len(gt_rows) == source['expected_windows']
        seen = set()
        for line in Path(source['input_index']).read_text().splitlines():
            row = json.loads(line)
            record = json.loads(Path(row['window_input']).read_text())
            key = record['window_id']
            if key not in gt:
                continue
            assert key not in seen and record['dataset'] == dataset
            seen.add(key)
            assert record['frame_ids'] == gt[key]['frame_ids']
            assert len(record['frame_ids']) == len(record['geometry_paths']) == 60
            for raw_path in record['geometry_paths']:
                path = Path(raw_path)
                if not path.is_file() or not os.access(path, os.R_OK):
                    raise FileNotFoundError(raw_path)
            result.append({**record, 'cache_id': gt[key]['cache_id']})
        assert seen == gt.keys()
    assert len(result) == spec['expected_windows'] == 2378
    return result


def load_faces(path):
    with Path(path).open('rb') as f:
        faces = np.asarray(pickle.load(f, encoding='latin1')['f'], dtype=np.int64)
    assert faces.ndim == 2 and faces.shape[1] == 3 and faces.min() >= 0 and faces.max() < 778
    return faces


def frame(k, scene, faces):
    assert scene['hand_vertices'].shape == (2, 778, 3)
    q = k.query_geometry({'hand_vertices_camera': scene['hand_vertices'],
                          'hand_joints_camera': scene['hand_joints']})
    a = k.frame_geometry_metrics(q, scene['hand_valid'], scene, faces, 'cpu')
    for prefix in ('joint', 'marker', 'vertex'):
        mask = a[prefix + '_contact_mask']
        assert np.isnan(a[prefix + '_contact_target'][~mask]).all()
        assert np.isnan(a[prefix + '_contact_distance'][~a[prefix + '_contact_distance_mask']]).all()
    assert a['vertex_contact_target'].shape == (2, 778)
    return a


def validate(spec, root):
    root.mkdir(parents=True, exist_ok=False)
    tests = synthetic_check()
    print(json.dumps({'stage': 'cpu_kernel_passed', **tests}), flush=True)
    rows = records(spec)
    faces = load_faces(spec['mano_right'])
    k = kernel()
    samples = {}
    for dataset in spec['datasets']:
        for record in (r for r in rows if r['dataset'] == dataset):
            for path in record['geometry_paths']:
                with np.load(path, allow_pickle=False) as a:
                    scene = dict(a)
                if not len(scene['object_faces']) or not np.asarray(scene['hand_valid']).any():
                    continue
                start = time.monotonic()
                arrays = frame(k, scene, faces)
                samples[dataset] = {'geometry_path': path, 'seconds': time.monotonic() - start,
                                    'valid_vertices': int(arrays['vertex_contact_mask'].sum())}
                assert samples[dataset]['valid_vertices'] > 0
                break
            if dataset in samples:
                break
        if dataset not in samples:
            raise ValueError('No valid object/hand frame in registered inputs: ' + dataset)
        print(json.dumps({'stage': 'sample_passed', 'dataset': dataset, **samples[dataset]}), flush=True)
    report = {'status': 'complete', 'windows': len(rows), 'frame_positions': len(rows)*60,
              'counts': {d: sum(r['dataset'] == d for r in rows) for d in spec['datasets']},
              'spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
              'tests': tests, 'samples': samples}
    (root/'report.json').write_text(json.dumps(report, indent=2))
    (root/'COMPLETE').write_text('complete\n')


def derive_window(args):
    record, mano, output_root = args
    k = kernel()
    faces = load_faces(mano)
    collected = {}
    for path in record['geometry_paths']:
        with np.load(path, allow_pickle=False) as a:
            values = frame(k, dict(a), faces)
        for key, value in values.items():
            collected.setdefault(key, []).append(value)
    arrays = {key: np.stack(value) for key, value in collected.items()}
    folder = Path(output_root)/record['dataset']
    folder.mkdir(exist_ok=True)
    path = folder/(record['cache_id'] + '.npz')
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
    return {'dataset': record['dataset'], 'window_id': record['window_id'], 'frame_ids': record['frame_ids'],
            'array_path': str(path), 'is_prediction_overlay': False,
            'valid_counts': {k: int(v.sum()) for k, v in arrays.items() if k.endswith('_mask')}}


def checked_resume(rows, source, handle):
    """Audit exact expected files only; the old process group must remain paused."""
    from formal_evaluation.remote_task_control import checked_handle, proc_identity
    registered, current = checked_handle(handle)
    assert current is not None and current['state'] == 'T', 'source writer is not paused'
    frontier = [registered['pid']]
    while frontier:
        pid = frontier.pop()
        identity = proc_identity(pid)
        if identity is None:
            continue
        assert identity['pgid'] == registered['pgid'] and identity['state'] in {'T', 'Z'}
        frontier.extend(int(v) for v in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
    indexed = {}
    if (source/'index.jsonl').exists():
        for line in (source/'index.jsonl').read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A paused coordinator may have an unfinished final line.
            indexed[(row['dataset'], row['window_id'])] = row
    reused = []; remaining = []; rejected = []
    for record in rows:
        path = source/record['dataset']/(record['cache_id']+'.npz')
        try:
            with np.load(path, allow_pickle=False) as arrays:
                counts = {}
                for prefix, size in [('joint', 21), ('marker', 195), ('vertex', 778)]:
                    for suffix in ('target', 'mask', 'distance', 'distance_mask'):
                        assert arrays[prefix+'_contact_'+suffix].shape == (60, 2, size)
                    for suffix, mask_suffix in [('target', 'mask'), ('distance', 'distance_mask')]:
                        value = arrays[prefix+'_contact_'+suffix]
                        mask = arrays[prefix+'_contact_'+mask_suffix]
                        assert mask.dtype == bool and np.isnan(value[~mask]).all()
                        assert np.isfinite(value[mask]).all() and (value[mask] >= 0).all()
                        if suffix == 'target': assert np.isin(value[mask], [0, 1]).all()
                        counts[prefix+'_contact_'+mask_suffix] = int(mask.sum())
            prior = indexed.get((record['dataset'], record['window_id']))
            if prior:
                assert prior['frame_ids'] == record['frame_ids'] and prior['array_path'] == str(path)
            reused.append({'dataset': record['dataset'], 'window_id': record['window_id'],
                           'frame_ids': record['frame_ids'], 'array_path': str(path),
                           'is_prediction_overlay': False, 'valid_counts': counts,
                           'reused': True, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError, AssertionError) as error:
            remaining.append(record)
            rejected.append({'dataset': record['dataset'], 'window_id': record['window_id'],
                             'path': str(path), 'reason': type(error).__name__})
    assert len(reused)+len(remaining) == 2378
    return reused, remaining, {'reused_windows': len(reused), 'remaining_windows': len(remaining),
        'indexed_windows': len(indexed), 'source_handle': str(handle), 'rejected_or_missing': rejected}


def run(spec, root, validation, workers, resume_root=None, resume_handle=None):
    assert (validation/'COMPLETE').is_file()
    evidence = json.loads((validation/'report.json').read_text())
    assert evidence['status'] == 'complete' and evidence['windows'] == 2378
    assert evidence['spec_sha256'] == hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    rows = records(spec)
    root.mkdir(parents=True, exist_ok=False)
    reused = []
    if resume_root:
        reused, rows, audit = checked_resume(rows, resume_root, resume_handle)
        (root/'resume_audit.json').write_text(json.dumps(audit, indent=2))
        print(json.dumps({'stage': 'resume_audit', 'reused_windows': len(reused), 'remaining_windows': len(rows)}), flush=True)
    counts = {}; valid = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        with (root/'index.jsonl').open('x') as out:
            jobs = ((r, spec['mano_right'], str(root)) for r in rows)
            import itertools
            for i, row in enumerate(itertools.chain(reused, pool.map(derive_window, jobs, chunksize=1)), 1):
                out.write(json.dumps(row) + '\n'); out.flush()
                d = row['dataset']; counts[d] = counts.get(d, 0) + 1
                for key, n in row['valid_counts'].items():
                    valid[key] = valid.get(key, 0) + n
                print(json.dumps({'index': i, 'dataset': d, 'status': 'written'}), flush=True)
    assert counts == evidence['counts']
    report = {'status': 'complete', 'windows': sum(counts.values()), 'counts': counts, 'valid_counts': valid,
              'distance': 'unsigned point-to-triangle surface union, metres', 'vertex_threshold_mm': 14,
              'validation_report': str(validation/'report.json'), 'device': 'cpu', 'workers': workers,
              'reused_windows': len(reused), 'new_windows': len(rows)}
    (root/'report.json').write_text(json.dumps(report, indent=2))
    (root/'summary.json').write_text(json.dumps(report, indent=2))
    (root/'COMPLETE').write_text('complete\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--validation', type=Path)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--resume-root', type=Path)
    p.add_argument('--resume-handle', type=Path)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 1 <= a.workers <= 48
    assert bool(a.resume_root) == bool(a.resume_handle)
    spec = json.loads(a.spec.read_text())
    if a.validate_only:
        validate(spec, a.output_root)
    else:
        run(spec, a.output_root, a.validation, a.workers, a.resume_root, a.resume_handle)
