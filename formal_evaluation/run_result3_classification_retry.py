"""Exact-window contact/visibility aggregation with immutable Result3 masks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.common.mano_sampling import marker_parent_ids_778
from formal_evaluation.contact.metrics import compute_contact_metrics


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def keyed(rows, dataset, aliases=None):
    result = {}
    for row in rows:
        if row.get('dataset', dataset) != dataset:
            continue
        key = (aliases or {}).get(row['window_id'], row['window_id'])
        if key in result:
            raise ValueError('duplicate window: ' + key)
        result[key] = row
    return result


def masks(source, expected):
    raw = Path(source['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('mask SHA256 mismatch')
    value = json.loads(raw)
    result = dict(zip(value['window_ids'], value['excluded'], strict=True))
    if len(result) != len(value['window_ids']) or set(result) != set(expected):
        raise ValueError('mask identity mismatch')
    if any(np.asarray(v).shape != (60,) for v in result.values()):
        raise ValueError('mask frame shape mismatch')
    return result


def load_arrays(row, index=None, names=None):
    path = Path(row['array_path'])
    # The registered GT index owns paths below its gt_cache directory.
    if index and '/gt_cache/' in str(path):
        path = Path(index).parent / str(path).split('/gt_cache/', 1)[1]
    with np.load(path, allow_pickle=False) as archive:
        selected = archive.files if names is None else tuple(names)
        missing = sorted(set(selected) - set(archive.files))
        if missing:
            raise ValueError('classification arrays missing: ' + ','.join(missing))
        return {k: archive[k] for k in selected}


def classify(prediction, target, keep, mode):
    result = {}
    prefixes = ('joint', 'marker', 'vertex')
    for prefix in prefixes:
        field = prefix + '_' + mode
        if mode == 'visibility' and prefix == 'vertex':
            parents = marker_parent_ids_778()
            probability = np.asarray(prediction['marker_visibility'])[..., parents]
            labels = np.asarray(target['marker_visibility_target'])[..., parents]
            mask = np.asarray(target['marker_visibility_mask'], dtype=bool)[..., parents].copy()
        else:
            pred_key = field + '_probability' if mode == 'contact' else ('hand_visibility' if prefix == 'joint' else 'marker_visibility')
            probability = np.asarray(prediction[pred_key])
            labels = np.asarray(target[field + '_target'])
            mask = np.asarray(target[field + '_mask'], dtype=bool).copy()
        mask &= keep[:, None, None]
        if 'hand_valid' in prediction:
            mask &= np.asarray(prediction['hand_valid'], dtype=bool)[:, :, None]
        result.update({field + '_' + k: np.asarray(v['ap'] if k == 'average_precision' and isinstance(v, dict) else v).item() for k, v in compute_contact_metrics(probability, labels, mask).items()})
    return result


def run(spec, root):
    # Runtime, logs and handles live in a sibling directory; existing results are never reused.
    root.mkdir(parents=True, exist_ok=False)
    total = 0
    report = {'status': 'complete', 'method': spec['method'], 'windows': 0,
              'camera': spec['camera'], 'mode': spec['mode'], 'schemes': {'all8_p95': {}, 'unfiltered': {}},
              'datasets': {}, 'fixed_selection_sha256': spec['fixed_selection_sha256']}
    with (root / 'index.jsonl').open('x') as output:
        for dataset, source in spec['datasets'].items():
            gt = keyed(read_rows(source['gt_index']), dataset)
            aliases = {r['cache_id']: k for k, r in gt.items() if 'cache_id' in r}
            pred_rows = read_rows(source['prediction_index'])
            if source.get('prediction_method'):
                pred_rows = [r for r in pred_rows if r.get('method') == source['prediction_method']]
            predictions = keyed(pred_rows, dataset, aliases)
            if len(gt) != source['expected_windows'] or set(predictions) != set(gt):
                raise ValueError('prediction/GT coverage mismatch: ' + dataset)
            excluded = masks(source['mask'], gt)
            values = {s: [] for s in report['schemes']}
            for key, target_row in gt.items():
                prediction_row = predictions[key]
                frame_ids = target_row['frame_ids']
                if len(frame_ids) != 60:
                    raise ValueError('GT frame count: ' + key)
                if source['prediction_format'] == 'canonical':
                    directory = Path(prediction_row['prediction_dir'])
                    metadata = json.loads((directory / 'metadata.json').read_text())
                    prediction_names = {'hand_valid'} | (
                        {'joint_contact_probability', 'marker_contact_probability', 'vertex_contact_probability'}
                        if spec['mode'] == 'contact' else {'hand_visibility', 'marker_visibility'}
                    )
                    prediction = load_arrays(
                        {'array_path': str(directory / 'predictions.npz')}, names=prediction_names
                    )
                    capabilities = metadata.get('capabilities', {})
                    missing = sorted(name for name in prediction_names if not capabilities.get(name))
                    if missing:
                        raise ValueError('canonical capability missing: ' + ','.join(missing))
                    pred_frames = metadata['frame_ids']
                else:
                    prediction = load_arrays(prediction_row)
                    pred_frames = prediction_row['frame_ids']
                if pred_frames != frame_ids:
                    raise ValueError('frame identity mismatch: ' + key)
                target_names = {
                    prefix + '_' + spec['mode'] + '_' + suffix
                    for prefix in (('joint', 'marker', 'vertex') if spec['mode'] == 'contact' else ('joint', 'marker'))
                    for suffix in ('target', 'mask')
                }
                target = load_arrays(target_row, source['gt_index'], target_names)
                per_window = {}
                for scheme in values:
                    keep = ~np.asarray(excluded[key], dtype=bool) if scheme == 'all8_p95' else np.ones(60, dtype=bool)
                    metrics = classify(prediction, target, keep, spec['mode'])
                    values[scheme].append({'window_id': key, **metrics})
                    per_window[scheme] = metrics
                output.write(json.dumps({'dataset': dataset, 'window_id': key, 'frame_ids': frame_ids, 'schemes': per_window}) + '\n')
                total += 1
                if total % 10 == 0 or total == 1:
                    output.flush()
                    print(json.dumps({'index': total, 'dataset': dataset, 'window_id': key, 'status': 'aggregated'}), flush=True)
            for scheme in values:
                report['schemes'][scheme][dataset] = aggregate_windows(values[scheme], method=spec['method'])
            report['datasets'][dataset] = {'windows': len(gt), 'mask_sha256': source['mask']['sha256']}
    if total != spec['expected_windows']:
        raise ValueError('total coverage mismatch')
    report['windows'] = total
    (root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=True))
    summary = {'status': 'complete', 'windows': total, 'datasets': report['datasets'],
               'report_sha256': hashlib.sha256((root / 'report.json').read_bytes()).hexdigest(),
               'index_sha256': hashlib.sha256((root / 'index.jsonl').read_bytes()).hexdigest()}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    (root / 'COMPLETE').write_text('complete\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()), args.output_root)
