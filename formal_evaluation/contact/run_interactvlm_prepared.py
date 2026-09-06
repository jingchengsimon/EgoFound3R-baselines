"""Run prepared InteractVLM smoke, resident inference, and canonical contact reports."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.common.io import write_comparison_output
from formal_evaluation.common.mano_sampling import marker_vertex_ids_195
from formal_evaluation.common.schema import SCHEMA_VERSION, validate_comparison_output
from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
from formal_evaluation.evaluate_six_dataset import evaluate_window
from formal_evaluation.run_egofound3r_relay import transfer

COUNTS = {'h2o': 12, 'hot3d': 17, 'arctic': 18, 'oakink_v2': 17, 'taco': 17, 'hoi4d': 19}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hand_probabilities(scores, targets, mapping):
    scores = np.asarray(scores)
    if scores.shape != (len(targets['hand_valid']), 6890):
        raise ValueError(f'wrong native contact shape: {scores.shape}')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('native contact must be finite probabilities in [0,1]')
    joints = np.full((len(scores), 2, 21), np.nan, dtype=np.float32)
    markers = np.full((len(scores), 2, 195), np.nan, dtype=np.float32)
    valid = np.asarray(targets['hand_valid'], dtype=bool).copy()
    for side, name in enumerate(('left', 'right')):
        ids = np.asarray(mapping[name], dtype=np.int64)
        if ids.shape != (778,) or len(set(ids)) != 778 or np.any((ids < 0) | (ids >= 6890)):
            raise ValueError(f'invalid {name} MANO mapping')
        vertex_scores = scores[:, ids]
        for t in np.flatnonzero(valid[:, side]):
            vertices = targets['hand_vertices_camera'][t, side]
            points = targets['hand_joints_camera'][t, side]
            if not np.isfinite(vertices).all() or not np.isfinite(points).all():
                raise ValueError('valid GT geometry contains nonfinite values')
            nearest = ((vertices[:, None] - points[None]) ** 2).sum(-1).argmin(0)
            # Same projection and fingertip overrides as the existing contact adapters.
            nearest[[4, 8, 12, 16, 20]] = [745, 317, 444, 556, 673]
            joints[t, side] = vertex_scores[t, nearest]
            markers[t, side] = vertex_scores[t, marker_vertex_ids_195()]
    return {'joint_contact_probability': joints, 'marker_contact_probability': markers,
            'hand_valid': valid}


def run_native(args, manifest, label):
    output = args.work_dir / (label + '_predictions.jsonl')
    adapter = Path(__file__).parent / 'adapters/run_interactvlm.py'
    command = [sys.executable, str(adapter), '--source-root', str(args.source_root),
               '--checkpoint', str(args.checkpoint), '--vision-tower', str(args.vision_tower),
               '--precision', 'bf16', '--input-manifest', str(manifest),
               '--work-dir', str(args.work_dir / label), '--output-manifest', str(output)]
    subprocess.run(command, check=True)
    rows = jsonl(output)
    expected = jsonl(manifest)
    if len(rows) != len(expected):
        raise ValueError('incomplete native prediction coverage')
    for row, inp in zip(rows, expected, strict=True):
        for field in ('dataset', 'sequence', 'frame_id', 'object_name'):
            if row[field] != inp[field]:
                raise ValueError(f'native identity drift: {field}')
        with np.load(row['prediction_path'], allow_pickle=False) as archive:
            score = archive['pred_contact_3d_smplh'].reshape(-1)
        if score.shape != (6890,) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError('invalid native SMPL-H prediction')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('names-root', 'assets-root', 'work-dir', 'output-root', 'source-root', 'checkpoint', 'vision-tower'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    for root in (args.names_root, args.assets_root):
        if not (root / 'COMPLETE').is_file():
            raise ValueError(f'preparation incomplete: {root}')
    if args.work_dir.exists() or args.output_root.exists():
        raise FileExistsError('unique work and output roots required')
    args.work_dir.mkdir(parents=True)
    names = json.loads((args.names_root / 'summary.json').read_text())
    if names['windows'] != COUNTS or names['unresolved'] != 0 or names['frames'] != 6000:
        raise ValueError('name/coverage gate failed')
    mapping = json.loads((args.assets_root / 'mano_to_smplh.json').read_text())
    native = {}
    if not args.validate_only:
        smoke = run_native(args, args.names_root / 'six_frames.jsonl', 'smoke')
        if {row['dataset'] for row in smoke} != set(COUNTS):
            raise ValueError('six-dataset smoke coverage mismatch')
        (args.work_dir / 'GPU_SMOKE_VALIDATED').write_text('6/6 native predictions validated\n')
        print(json.dumps({'stage': 'gpu_smoke', 'frames': 6, 'status': 'complete'}), flush=True)
        for row in run_native(args, args.names_root / 'frames6000.jsonl', 'formal'):
            key = (row['dataset'], row['sequence'], str(row['frame_id']))
            if key in native:
                raise ValueError(f'duplicate native frame: {key}')
            native[key] = row
    reports, index, hashes = {}, [], {}
    staging = args.work_dir / 'canonical'
    for dataset, count in COUNTS.items():
        gt_rows = jsonl(args.assets_root / (dataset + '_gt.jsonl'))
        if len(gt_rows) != count:
            raise ValueError('GT subset coverage mismatch')
        results = []
        for gt_row in gt_rows:
            meta, gt = load_window_cache(gt_row)
            sources = []
            if args.validate_only:
                scores = np.zeros((len(meta['frame_ids']), 6890), dtype=np.float32)
            else:
                sources = [native[(dataset, meta['sequence_id'], str(fid))] for fid in meta['frame_ids']]
                scores = np.stack([np.load(r['prediction_path'], allow_pickle=False)['pred_contact_3d_smplh'].reshape(-1) for r in sources])
            arrays = hand_probabilities(scores, gt, mapping)
            metadata = {'schema_version': SCHEMA_VERSION, 'dataset': dataset,
                        'sequence_id': meta['sequence_id'], 'window_id': meta['window_id'],
                        'frame_ids': meta['frame_ids'], 'capabilities': {k: True for k in arrays},
                        'method': 'interactvlm', 'geometry_for_projection': 'GT MANO; not a predicted hand pose',
                        'validity_source': 'GT hand availability; no learned presence prediction'}
            validate_comparison_output(metadata, arrays)
            results.append(evaluate_window(method='interactvlm', config={'group': ['contact']},
                                           metadata=metadata, predictions=arrays, gt_metadata=meta, targets=gt))
            if not args.validate_only:
                relative = Path(dataset) / str(gt_row['cache_id'])
                directory = staging / relative
                target = args.output_root / relative
                archived_sources = [{k: v for k, v in row.items() if k != 'prepared_input'} for row in sources]
                for i, row in enumerate(archived_sources):
                    row['prediction_path'] = str(target / 'native' / f'{i:03d}.npz')
                write_comparison_output(directory, metadata=metadata, arrays=arrays,
                                        run={'status': 'success', 'checkpoint': str(args.checkpoint), 'precision': 'bf16'},
                                        native_metadata={'frames': archived_sources})
                for i, row in enumerate(sources):
                    shutil.copy2(row['prediction_path'], directory / 'native' / f'{i:03d}.npz')
                hashes[str(relative)] = transfer(directory, target)
                index.append({'method': 'interactvlm', 'dataset': dataset, 'window_id': meta['window_id'],
                              'prediction_dir': str(target)})
        reports[dataset] = aggregate_windows(results, method='interactvlm')
        print(json.dumps({'stage': 'cpu_validation' if args.validate_only else 'report',
                          'dataset': dataset, 'windows': len(results)}), flush=True)
    if args.validate_only:
        (args.work_dir / 'CPU_VALIDATED.json').write_text(json.dumps(
            {'synthetic_only': True, 'windows': COUNTS, 'total_windows': 100,
             'checks': ['native shape/range', 'two-hand topology mapping', 'canonical schema', 'contact aggregation']}))
        return
    report = {'gt_windows': 100, 'methods': {'interactvlm': {'missing_prediction_windows': 0, 'datasets': reports}}}
    (args.output_root / 'report.json').write_text(json.dumps(report, indent=2))
    (args.output_root / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in index))
    (args.output_root / 'transfer_hashes.json').write_text(json.dumps(hashes))
    for file in ('mapping_audit.json', 'summary.json', 'frames6000.jsonl'):
        shutil.copy2(args.names_root / file, args.output_root / ('input_' + file))
    (args.output_root / 'COMPLETE').write_text('100 windows; 6000 native frames; canonical/report/OSS verification complete\n')
    # Preserve logs/control and prepared inputs; only this run's generated staging is removed.
    for label in ('smoke', 'formal', 'canonical'):
        shutil.rmtree(args.work_dir / label)
    print(json.dumps({'status': 'complete', 'windows': 100, 'output_root': str(args.output_root)}), flush=True)


if __name__ == '__main__':
    main()
