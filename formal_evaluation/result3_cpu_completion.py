"""Registered Pi3 depth recomputation or serial contact archive restoration."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def pi3_window(row):
    import numpy as np
    from egocentric_metrics import depth_metrics
    meta = json.loads((Path(row['prediction_dir']) / 'metadata.json').read_text())
    assert meta['frame_ids'] == row['frame_ids'] and meta['dataset'] == row['dataset']
    with np.load(Path(row['prediction_dir']) / 'predictions.npz', allow_pickle=False) as a:
        points = a['camera_points']
        valid = a['camera_points_valid'] & np.isfinite(points).all(-1) & (points[..., 2] > 0)
        depth = np.where(valid, points[..., 2], np.nan)
    with np.load(row['gt_array_path'], allow_pickle=False) as a:
        target = a['depth']; target_valid = a['depth_valid']
    assert len(depth) == len(target) == len(row['frame_ids'])
    frames = []
    for i, (pred, gt) in enumerate(zip(depth, target)):
        if pred.shape != gt.shape:
            import cv2
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        gt = np.where(target_valid[i], gt, np.nan)
        values = depth_metrics(pred, gt)
        frames.append({'frame_id': row['frame_ids'][i], **{k: float(v) for k, v in values.items()}})
    mean = {}
    for key in frames[0]:
        if key == 'frame_id': continue
        values = np.array([f[key] for f in frames], dtype=float)
        mean[key] = float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float('nan')
    return {'dataset': row['dataset'], 'window_id': row['window_id'], 'frames': frames,
            'metrics': mean, 'depth_window_scale': 1.0}


def pi3(spec, root):
    import numpy as np
    index = Path(spec['input_index'])
    assert digest(index) == spec['input_sha256']
    rows = [json.loads(l) for l in index.read_text().splitlines() if l]
    assert len(rows) == 1144
    results = []
    with (root / 'windows.jsonl').open('x') as out, concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for count, value in enumerate(pool.map(pi3_window, rows), 1):
            out.write(json.dumps(value) + '\n'); out.flush(); results.append(value)
            print(json.dumps({'status': 'written', 'index': count, 'total': len(rows)}), flush=True)
    datasets = {}
    for dataset in ('h2o', 'taco', 'hoi4d'):
        selected = [r for r in results if r['dataset'] == dataset]
        metrics = {}; undefined = {}
        for key in selected[0]['metrics']:
            values = np.array([r['metrics'][key] for r in selected]); finite = np.isfinite(values)
            metrics[key] = float(values[finite].mean()) if finite.any() else float('nan')
            undefined[key] = int((~finite).sum())
        datasets[dataset] = {'windows': len(selected), 'metrics': metrics, 'undefined_windows': undefined}
    return {'status': 'complete', 'windows': len(results), 'datasets': datasets,
            'depth_source': 'camera_points[...,2]', 'depth_scale_alignment': 'none',
            'input_sha256': spec['input_sha256'], 'filtering': 'none; per-frame results retained for later Result3 P95'}


def restore_file(row, destination):
    parts = row.get('parts') or [{'path': row['path'], 'bytes': row['bytes']}]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < row['bytes'] + 100 * 1024**3:
        raise RuntimeError('INSUFFICIENT_SPACE_WITH_100GIB_RESERVE')
    h = hashlib.sha256(); total = 0
    with destination.open('xb') as out:
        for part in parts:
            assert part.get('offset', total) == total
            path = Path(part['path']); assert path.stat().st_size == part['bytes']
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(16 * 1024**2), b''):
                    out.write(chunk); h.update(chunk); total += len(chunk)
        out.flush(); os.fsync(out.fileno())
    assert total == row['bytes']
    sha = h.hexdigest()
    if row.get('sha256'): assert sha == row['sha256'], 'HISTORICAL_SHA_MISMATCH'
    assert digest(destination) == sha, 'RESTORED_SHA_MISMATCH'
    return {'source': row['path'], 'destination': str(destination), 'bytes': total,
            'sha256': sha, 'historical_sha_available': bool(row.get('sha256'))}


def restore(spec, root):
    results = []
    with (root / 'restored_files.jsonl').open('x') as log:
        for source in spec['sources']:
            gt = [json.loads(l) for l in Path(source['gt_index']).read_text().splitlines() if l]
            wanted = {r['cache_id']: r for r in gt}
            manifest_path = Path(source['manifest'])
            assert digest(manifest_path) == source['manifest_sha256']
            manifest = json.loads(manifest_path.read_text())
            rows = [r for r in manifest['files'] if not source.get('prefix') or r['relative_destination'].startswith(source['prefix'])]
            rows += source.get('external_files', [])
            for number, row in enumerate(rows):
                destination = root / source['dataset'] / ('%02d_' % number + Path(row['path']).name)
                result = {'dataset': source['dataset'], **restore_file(row, destination)}
                if destination.name.endswith('_index.jsonl'):
                    samples = [json.loads(l) for l in destination.read_text().splitlines() if l]
                    for sample in samples:
                        target = wanted[sample['cache_id']]
                        assert sample['dataset'] == source['dataset']
                        assert str(sample['frame_id']) == str(target['frame_ids'][int(sample['time_index'])])
                    result['frame_identity_checked'] = len(samples)
                    result['covered_windows'] = len({s['cache_id'] for s in samples})
                log.write(json.dumps(result) + '\n'); log.flush(); results.append(result)
                print(json.dumps({'status': 'restored', 'index': len(results), **result}), flush=True)
            (root / source['dataset'] / 'COMPLETE').write_text('restored and sha256 checked\n')
    return {'status': 'complete', 'windows': len(spec['sources']), 'files': results, 'bytes': sum(r['bytes'] for r in results),
            'cache_deserialization': 'required before forward; not inferred from byte checks'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['pi3', 'restore'], required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(); root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'COMPLETE').exists()
    report = (pi3 if args.kind == 'pi3' else restore)(json.loads(args.spec.read_text()), root)
    with (root / 'report.json').open('x') as f: json.dump(report, f, indent=2)
    (root / 'summary.json').write_text(json.dumps({'status': 'complete', 'windows': report.get('windows'), 'report_sha256': digest(root / 'report.json')}))
    (root / 'COMPLETE').write_text('complete\n')


if __name__ == '__main__': main()
