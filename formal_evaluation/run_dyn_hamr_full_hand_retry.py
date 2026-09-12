"""Registered original100 Dyn-HaMR rerun with cameras and a complete 24-metric gate."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate_one(spec, prediction_dir, output):
    import numpy as np
    from formal_evaluation.evaluate_six_dataset import _prediction_arrays, evaluate_window
    from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
    from formal_evaluation.recompute_same_mask_all_methods import remap_gt_row
    raw = Path(spec['gt_index']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['gt_index_sha256']:
        raise ValueError('GT index SHA256 mismatch')
    gt = {r['window_id']:r for r in map(json.loads, raw.splitlines())}
    metadata, prediction = _prediction_arrays(prediction_dir)
    row = remap_gt_row(gt[metadata['window_id']], Path('/unused'), Path(spec['gt_index']).parent, direct_oss=True)
    gm, target = load_window_cache(row)
    if gm['frame_ids'] != metadata['frame_ids']:
        raise ValueError('GT/prediction frame mismatch')
    config = json.loads(Path(spec['metric_methods_config']).read_text())['methods']['dyn_hamr']
    result = evaluate_window(method='dyn_hamr', config=config, metadata=metadata, predictions=prediction, gt_metadata=gm, targets=target)
    if prediction['hand_valid'].any():
        for stem in spec['metric_stems']:
            if not any('hand_'+side+'_'+stem in result for side in ['left','right']):
                raise ValueError('required metric not emitted: '+stem)
    def native(value):
        if isinstance(value, np.ndarray): return native(value.item() if value.ndim == 0 else value.tolist())
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, dict): return {k:native(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return [native(v) for v in value]
        return value
    output.write_text(json.dumps(native(result)))


def aggregate(spec, root):
    import numpy as np
    from formal_evaluation.common.aggregation import aggregate_windows
    metrics = rows(root/'metrics.jsonl')
    assert len(metrics) == spec['expected_windows']
    summary = aggregate_windows(metrics, method='dyn_hamr')
    complete24 = {}
    for stem in spec['metric_stems']:
        weighted = count = 0
        for side in ['left','right']:
            prefix='hand_'+side+'_'+stem
            mean=summary.get(prefix+'_mean',float('nan'));n=summary.get(prefix+'_count',0)
            if np.isfinite(mean) and n: weighted+=mean*n;count+=n
        if not count: raise ValueError('no defined windows for required metric: '+stem)
        complete24[stem]=weighted/count/(1000 if stem.endswith('ae') else 1)
    report={'status':'complete','method':'dyn_hamr','windows':len(metrics),'datasets':{spec['dataset']:summary},'complete24':complete24,'filtering':'none','comparison':'original100 identities; new predicted-camera rerun; excluded from same-sample bolding', 'camera_source':'optimized_Dyn-HaMR_world_cam_R_cam_t'}
    (root/'report.json').write_text(json.dumps(report,indent=2))
    (root/'summary.json').write_text(json.dumps({'status':'complete','windows':len(metrics),'metric_count':len(complete24),'report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),'prediction_index_sha256':hashlib.sha256((root/'predictions.jsonl').read_bytes()).hexdigest()}))
    (root/'COMPLETE').write_text('complete\n')


def infer(spec, root, spec_path):
    import numpy as np
    assert os.environ['PYOPENGL_PLATFORM']=='glx'
    import pyrender  # Establish GLX before the upstream viewer imports OpenGL.
    module_spec=importlib.util.spec_from_file_location('dyn_full_hand_adapter',spec['adapter'])
    adapter=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(adapter)
    expected={}
    for index in spec['reference_indices']:
        for r in rows(index):
            meta=json.loads((Path(r['prediction_dir'])/'metadata.json').read_text())
            key=(meta['dataset'],meta['window_id'])
            if key in expected:raise ValueError('duplicate original window')
            expected[key]=meta['frame_ids']
    inputs=rows(spec['prepared_index'])
    assert len(inputs)==len(expected)==spec['expected_windows']
    seen=set()
    for row in inputs:
        record=json.loads(Path(row['window_input']).read_text());key=(record['dataset'],record['window_id'])
        assert key not in seen and record['frame_ids']==expected[key];seen.add(key)
    assert seen==set(expected)
    root.mkdir(parents=True,exist_ok=False)
    cpu_env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':spec['metric_worktree']}
    with (root/'predictions.jsonl').open('x') as index, (root/'metrics.jsonl').open('x') as metrics:
        for i,row in enumerate(inputs,1):
            start=time.monotonic();record=json.loads(Path(row['window_input']).read_text());source=Path(spec['source_root'])
            adapter.main(['--phase','formal','--window-input',row['window_input'],'--methods-config',spec['inference_methods_config'],'--source-root',str(source),'--hamer-checkpoint-root',str(source),'--detector-weight',str(source/'third-party/hamer/pretrained_models/detector.pt'),'--droid-weight',str(source/'_DATA/droid.pth'),'--scratch-dir','/dev/shm','--output-root',str(root),'--context-frames','80','--root-iters','50','--smooth-iters','300','--export-cameras'])
            target=root/'dyn_hamr/formal'/record['cache_id']
            meta=json.loads((target/'metadata.json').read_text());status=json.loads((target/'run.json').read_text())['status']
            assert meta['frame_ids']==record['frame_ids'] and meta['window_id']==record['window_id']
            if status not in ['success','blocked_no_track_over_min_track_len']:raise ValueError(status)
            metric_file=root/(record['cache_id']+'_metrics.json')
            subprocess.run([spec['metric_python'],__file__,'--spec',str(spec_path),'--output-root',str(root),'--evaluate-one',str(target),'--metric-output',str(metric_file)],env=cpu_env,cwd=spec['metric_worktree'],check=True)
            metric=json.loads(metric_file.read_text());metrics.write(json.dumps(metric)+'\n');metrics.flush()
            index.write(json.dumps({'method':'dyn_hamr','dataset':spec['dataset'],'window_id':record['window_id'],'prediction_dir':str(target),'status':status})+'\n');index.flush()
            print(json.dumps({'index':i,'dataset':spec['dataset'],'target_windows':spec['expected_windows'],'status':'predictions_and_24_metric_fields_verified','seconds':time.monotonic()-start}),flush=True)
            gc.collect()
    subprocess.run([spec['metric_python'],__file__,'--spec',str(spec_path),'--output-root',str(root),'--aggregate'],env=cpu_env,cwd=spec['metric_worktree'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--evaluate-one',type=Path);p.add_argument('--metric-output',type=Path);p.add_argument('--aggregate',action='store_true');a=p.parse_args();spec=json.loads(a.spec.read_text())
    if a.evaluate_one:evaluate_one(spec,a.evaluate_one,a.metric_output)
    elif a.aggregate:aggregate(spec,a.output_root)
    else:infer(spec,a.output_root,a.spec)
