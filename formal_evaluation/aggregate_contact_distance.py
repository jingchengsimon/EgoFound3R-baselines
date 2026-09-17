"""Aggregate immutable per-frame contact distances using the registered frozen mask."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from formal_evaluation.common.aggregation import aggregate_windows


def window_metrics(arrays, keep):
    result = {}
    for prefix, points in [('joint', 21), ('marker', 195), ('vertex', 778)]:
        distance = np.asarray(arrays[prefix + '_contact_distance'])
        valid = np.asarray(arrays[prefix + '_contact_distance_mask'], dtype=bool)
        probability = np.asarray(arrays[prefix + '_contact_probability'])
        expected = (len(keep), 2, points)
        if any(x.shape != expected for x in (distance, valid, probability)):
            raise ValueError('distance/probability shape mismatch: ' + prefix)
        valid = valid & np.isfinite(distance) & keep[:, None, None]
        for name, selected in [('all_valid', valid), ('predicted_contact', valid & (probability >= .5))]:
            stem = prefix + '_contact_distance_' + name
            result[stem + '_mm'] = float(distance[selected].mean() * 1000) if selected.any() else float('nan')
            result[stem + '_count'] = int(selected.sum())
    return result


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(spec, root):
    rows = read_rows(spec['input_index'])
    if len(rows) != spec['expected_windows']:
        raise ValueError('overlay count mismatch')
    if len({(r['dataset'], r['window_id']) for r in rows}) != len(rows):
        raise ValueError('duplicate overlay window')
    root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'complete', 'method': spec['method'], 'windows': len(rows),
              'schemes': {'all8_p95': {}, 'unfiltered': {}}, 'source_index': spec['input_index'],
              'source_index_sha256': hashlib.sha256(Path(spec['input_index']).read_bytes()).hexdigest(),
              'mask_sha256': {}}
    for dataset, source in spec['datasets'].items():
        selected = [r for r in rows if r['dataset'] == dataset]
        gt = {r['window_id']: r for r in read_rows(source['gt_index'])}
        mask_bytes = Path(source['mask']['path']).read_bytes()
        if hashlib.sha256(mask_bytes).hexdigest() != source['mask']['sha256']:
            raise ValueError('mask checksum mismatch')
        mask = json.loads(mask_bytes)
        if len(mask['window_ids']) != len(set(mask['window_ids'])):
            raise ValueError('duplicate mask window')
        masks = dict(zip(mask['window_ids'], mask['excluded'], strict=True))
        if {r['window_id'] for r in selected} != set(masks) or set(masks) != set(gt):
            raise ValueError('window identities mismatch: ' + dataset)
        report['mask_sha256'][dataset] = source['mask']['sha256']
        metrics = {scheme: [] for scheme in report['schemes']}
        with (root / (dataset + '_windows.jsonl')).open('x') as output:
            for row in selected:
                key = row['window_id']
                if row['frame_ids'] != gt[key]['frame_ids']:
                    raise ValueError('frame identity mismatch: ' + key)
                excluded = np.asarray(masks[key], dtype=bool)
                if excluded.shape != (60,):
                    raise ValueError('mask frame shape mismatch')
                with np.load(row['array_path'], allow_pickle=False) as arrays:
                    for scheme in metrics:
                        keep = ~excluded if scheme == 'all8_p95' else np.ones(60, bool)
                        values = {'window_id': key, **window_metrics(arrays, keep)}
                        metrics[scheme].append(values)
                output.write(json.dumps({'window_id': key, 'frame_ids': row['frame_ids'],
                                         'schemes': {s: metrics[s][-1] for s in metrics}}) + '\n')
        for scheme, values in metrics.items():
            report['schemes'][scheme][dataset] = aggregate_windows(values, method=spec['method'])
        print(json.dumps({'dataset': dataset, 'windows': len(selected)}), flush=True)
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    (root / 'summary.json').write_text(json.dumps({'status': 'complete', 'windows': len(rows),
        'report_sha256': hashlib.sha256((root / 'report.json').read_bytes()).hexdigest()}))
    (root / 'COMPLETE').write_text('complete\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()), args.output_root)
