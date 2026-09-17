#!/usr/bin/env python3
"""Resident Result3 stride inference and fixed-mask CPU hand metric completion."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def selection_for(selection, dataset, window_id, frame_ids):
    row = selection[(dataset, window_id)]
    if row['frame_ids'] != frame_ids or len(row['keep']) != len(frame_ids):
        raise ValueError('fixed selection frame identity mismatch')
    return row['keep']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    sys.path.insert(0, spec['baseline_root'])
    if spec['mode'] == 'stride':
        sys.path.insert(0, spec['model_root'])
    import numpy as np
    from formal_evaluation.common.aggregation import aggregate_windows
    from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
    from formal_evaluation.evaluate_six_dataset import _prediction_arrays
    import importlib.util
    def load(name, path):
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module
    common = load('completion_common', spec['same_mask_script'])
    hand = load('completion_frozen_hand', spec['hand_script'])
    raw = Path(spec['selection']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['selection_sha256']:
        raise ValueError('fixed selection digest mismatch')
    selected = [json.loads(line) for line in raw.splitlines() if line]
    selection = {(r['dataset'], r['window_id']): r for r in selected}
    if len(selection) != 2378 or len(selected) != 2378:
        raise ValueError('selection coverage mismatch')
    root = Path(spec['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'method': spec['method'], 'datasets': {},
              'selection_sha256': spec['selection_sha256'], 'world_pose_source': 'predicted_camera_c2w',
              'fit_on_full_unfiltered_window': True, 'no_refit_after_filter': True,
              'vertex_geometry': 'derived from 195 markers when native 778 vertices absent',
              'batch_size': 1 if spec['mode'] == 'stride' else None}
    completed_total = 0
    def progress(**row):
        row['time'] = time.time()
        with (root/'progress.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    for job in spec['jobs']:
        ds = job['dataset']; target_root = root/ds; target_root.mkdir()
        index = Path(job['gt_index'])
        gt_rows = common.read_jsonl(index)
        gt = {r['window_id']: common.remap_gt_row(r, Path('/unused'), index.parent, direct_oss=True) for r in gt_rows}
        if len(gt) != job['expected_windows'] or len(gt_rows) != len(gt):
            raise ValueError('GT coverage mismatch')
        if spec['mode'] == 'stride':
            from formal_evaluation.run_egofound3r_stride_evaluation import prepare_inputs
            from formal_evaluation.scene.adapters import run_egofound3r_baseline as adapter
            _, _, records = prepare_inputs(ds, job, target_root/'inputs')
            pred_root = target_root/'predictions'; pred_root.mkdir()
            found = {}
            for i, record in enumerate(records, 1):
                start = time.monotonic()
                adapter.main(['--phase','formal','--window-input',record['window_input'],
                    '--methods-config',spec['resident']['methods_config'],'--config',spec['resident']['config'],
                    '--checkpoint',spec['resident']['checkpoint'],'--backbone-checkpoint',spec['resident']['backbone'],
                    '--global-stride',str(spec['stride']),'--input-resolution','512x512','--output-root',str(pred_root)])
                if adapter._runtime.cache_info().misses != 1:
                    raise RuntimeError('model residency violated')
                folder = pred_root/'egofound3r/formal'/record['cache_id']
                meta = json.loads((folder/'metadata.json').read_text())
                if meta['global_stride'] != spec['stride'] or meta['processed_resolution_hw'] != [512,512]:
                    raise ValueError('inference provenance mismatch')
                found[record['window_id']] = folder
                progress(stage='inference',dataset=ds,completed=i,total=len(records),stride=spec['stride'],model_load_count=1,wall_seconds=time.monotonic()-start)
            (target_root/'predictions.jsonl').write_text(''.join(json.dumps({'method':'egofound3r','dataset':ds,'window_id':k,'prediction_dir':str(v)})+'\n' for k,v in found.items()))
        elif 'prediction_index' in job:
            indexed = common.read_jsonl(Path(job['prediction_index']))
            found = {r['window_id']: Path(r['prediction_dir']) for r in indexed}
            if len(found) != len(indexed):
                raise ValueError('duplicate prediction window')
        else:
            aliases = {r['cache_id']: r['window_id'] for r in gt_rows}
            found = common.prediction_rows('egofound3r', {'dataset':ds,'expected_windows':len(gt),'predictions':{'egofound3r':{'formal_roots':job['formal_roots']}}},Path('/unused'),aliases,direct_oss=True)
        if set(found) != set(gt):
            raise ValueError('prediction/GT coverage mismatch')
        rows=[]; frozen_root=target_root/'frozen'; frozen_root.mkdir()
        with (target_root/'window_metrics.jsonl').open('x') as output:
            for i, wid in enumerate(sorted(gt),1):
                pm,pred = _prediction_arrays(found[wid]); gm,target = load_window_cache(gt[wid])
                if pm['frame_ids'] != gm['frame_ids'] or 'camera_c2w' not in pred:
                    raise ValueError('prediction frame/camera identity mismatch')
                keep = np.asarray(selection_for(selection,ds,wid,gm['frame_ids']),bool)
                _,frame,pair,triplet = common.freeze_hand_window(hand,pred,target)
                np.savez_compressed(frozen_root/(gt[wid]['cache_id']+'.npz'),frame=frame,pair=pair,triplet=triplet)
                _, metrics = hand.aggregate_scheme(frame[None],pair[None],triplet[None],~keep[None])
                row={'window_id':wid,**metrics[0]}; rows.append(row)
                output.write(json.dumps(common.json_safe(row),allow_nan=False)+'\n');output.flush()
                if i==1 or i%10==0 or i==len(gt):
                    progress(stage='metrics',dataset=ds,completed=i,total=len(gt),method=spec['method'])
        report['datasets'][ds] = aggregate_windows(rows,method=spec['method'])
        completed_total += len(rows)
        (target_root/'COMPLETE').write_text('complete\n')
        (root/'report.json').write_text(json.dumps(common.json_safe(report),indent=2,allow_nan=False))
    if completed_total != sum(j['expected_windows'] for j in spec['jobs']):
        raise ValueError('final coverage mismatch')
    report.update(status='complete',windows=completed_total)
    (root/'report.json').write_text(json.dumps(common.json_safe(report),indent=2,allow_nan=False))
    (root/'summary.json').write_text(json.dumps({'status':'complete','windows':completed_total,'method':spec['method']}))
    (root/'COMPLETE').write_text('complete\n')
    progress(stage='complete',windows=completed_total)


if __name__ == '__main__':
    main()
