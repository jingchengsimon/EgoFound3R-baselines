"""CPU-only Result3 oracle and external100 completion on the frozen runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.evaluate_six_dataset import evaluate_window, _prediction_arrays
from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
from formal_evaluation.recompute_same_mask_all_methods import read_jsonl, remap_gt_row, prediction_rows, load_module, freeze_hand_window

def checked_gt_index(source):
    path=Path(source['gt_index'])
    if source.get('gt_index_sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=source['gt_index_sha256']:
        raise ValueError('GT index SHA-256 mismatch: '+str(path))
    return path

def fixed_mask(source, window_ids):
    raw=Path(source['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('fixed mask SHA-256 mismatch')
    value=json.loads(raw)
    if len(value['window_ids'])!=len(set(value['window_ids'])):raise ValueError('duplicate fixed mask window')
    if set(value['window_ids'])!=set(window_ids):raise ValueError('fixed mask window identity mismatch')
    mapping=dict(zip(value['window_ids'],value['excluded'],strict=True))
    result=np.asarray([mapping[k] for k in window_ids],dtype=bool)
    if result.shape!=(len(window_ids),60):raise ValueError('fixed mask frame shape mismatch')
    return result

def prediction_map(source, dataset, gt):
    aliases = {r['cache_id']:r['window_id'] for r in gt.values()}
    if 'catalog' in source:
        return prediction_rows(source['config_method'], source['catalog'], Path('/unused'), aliases, direct_oss=True)
    result = {}
    for index in source['indices']:
        for row in read_jsonl(Path(index)):
            if row['dataset'] != dataset: continue
            if row.get('method') != source.get('index_method', source['config_method']): continue
            key = aliases.get(row['window_id'], row['window_id'])
            if key in result: raise ValueError('duplicate prediction: '+key)
            if key not in gt: raise ValueError('prediction outside registered GT: '+key)
            result[key] = Path(row['prediction_dir'])
    if len(result) != source['expected_windows']: raise ValueError('prediction coverage mismatch')
    return result

def oracle_prediction(prediction, target):
    """Separate evaluation view, preserving the original prediction dictionary."""
    return {**prediction, 'camera_c2w': target['camera_c2w'], 'camera_valid': target['camera_valid']}


def run_oracle(spec, root):
    for raw in spec['complete_dependencies']:
        if not Path(raw).is_file():raise FileNotFoundError(raw)
    root.mkdir(parents=True,exist_ok=False)
    mode=spec['mode'];count=0
    if mode != 'oracle': raise ValueError(mode)
    hand=load_module('result3_aux_hand',Path(spec['hand_script'])) if mode=='oracle' else None
    with (root/'index.jsonl').open('x') as output:
        for dataset,source in spec['datasets'].items():
            index=checked_gt_index(source)
            gt={r['window_id']:remap_gt_row(r,Path('/unused'),index.parent,direct_oss=True) for r in read_jsonl(index)}
            if len(gt)!=source['expected_windows']:raise ValueError('GT coverage mismatch')
            preds=prediction_map(source['prediction'],dataset,gt) if mode=='oracle' else {}
            for key,row in gt.items():
                gm,target=load_window_cache(row)
                if mode=='oracle':
                    pm,pred=_prediction_arrays(preds[key])
                    if pm['frame_ids']!=gm['frame_ids']:raise ValueError('oracle frame mismatch')
                    _,frame,pair,triplet=freeze_hand_window(hand,oracle_prediction(pred,target),target)
                    arrays={'frame':frame,'pair':pair,'triplet':triplet}
                elif mode=='visibility':
                    arrays={k:v for k,v in target.items() if 'visibility' in k}
                    if not arrays:raise ValueError('visibility fields absent: '+key)
                else:raise ValueError(mode)
                folder=root/dataset;folder.mkdir(exist_ok=True);path=folder/(row['cache_id']+'.npz')
                np.savez_compressed(path,**arrays)
                output.write(json.dumps({'dataset':dataset,'window_id':key,'frame_ids':gm['frame_ids'],'array_path':str(path)})+'\n')
                count+=1
                if count%20==0: print(json.dumps({'stage':'freeze','windows':count}),flush=True)
    if count!=2378:raise ValueError('expected2378')

    # All full-window fits have finished before selecting any frame.
    report={'status':'complete','stage':'oracle','windows':count,'selection_sha256':spec['fixed_selection_sha256'],
            'comparison':'oracle diagnostic; excluded from thresholds, bolding and wins','schemes':{'unfiltered':{},'all8_p95':{}}}
    frozen=read_jsonl(root/'index.jsonl')
    for dataset in spec['datasets']:
        records=sorted((r for r in frozen if r['dataset']==dataset),key=lambda r:r['window_id'])
        excluded=fixed_mask(spec['fixed_masks'][dataset],[r['window_id'] for r in records])
        for scheme in report['schemes']:
            values=[]
            for i,row in enumerate(records):
                with np.load(row['array_path'],allow_pickle=False) as a:
                    mask=excluded[i] if scheme=='all8_p95' else np.zeros(60,bool)
                    _,metrics=hand.aggregate_scheme(a['frame'][None],a['pair'][None],a['triplet'][None],mask[None])
                values.append({'window_id':row['window_id'],**metrics[0]})
            report['schemes'][scheme][dataset]=aggregate_windows(values,method='egofound3r_stride5_gt_camera')
    (root/'report.json').write_text(json.dumps(report,indent=2))
    (root/'summary.json').write_text(json.dumps({'status':'complete','windows':count,'report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest()}))
    (root/'COMPLETE').write_text('complete\n')

def json_values(value):
    """Convert NumPy scalars before aggregation, preserving arrays and NaN."""
    if isinstance(value, np.ndarray):
        return json_values(value.item() if value.ndim == 0 else value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    return value


def run_external(spec, root):
    method = spec['method']
    if method not in {'dyn_hamr','interactvlm'}:raise ValueError(method)
    root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(spec['methods_config']).read_text())['methods'][method]
    datasets={};total=0
    for source in spec['catalogs']:
        dataset=source['dataset']; index=checked_gt_index(source)
        gt={r['window_id']:remap_gt_row(r,Path('/unused'),index.parent,direct_oss=True) for r in read_jsonl(index)}
        aliases={r['cache_id']:r['window_id'] for r in gt.values()}
        if source.get('prediction_index'):
            expected=source.get('expected_windows')
            if expected is None:
                original=json.loads(Path(source['base_reports'][0]).read_text())
                expected=original['methods'][method]['datasets'][dataset]['n_windows']
            preds=prediction_map({'indices':[source['prediction_index']],'config_method':method,'expected_windows':expected},dataset,gt)
        else:
            preds=prediction_rows(method,source,Path('/unused'),aliases,direct_oss=True)
        metrics=[]
        for key,path in preds.items():
            gm,target=load_window_cache(gt[key]);pm,prediction=_prediction_arrays(path)
            metrics.append(json_values(evaluate_window(method=method,config=config,metadata=pm,predictions=prediction,gt_metadata=gm,targets=target)))
        aggregate=aggregate_windows(metrics,method=method)
        # Preserve original report values; add only undefined/missing supported fields.
        for raw in source.get('base_reports',[]):
            report=json.loads(Path(raw).read_text())
            old=report['methods'][method]['datasets'][dataset]
            filled = [k[:-5] for k,v in old.items() if k.endswith('_mean') and isinstance(v,(int,float)) and not np.isfinite(v) and np.isfinite(aggregate.get(k,float('nan')))]
            for key,value in old.items():
                if any(key.startswith(stem+'_') for stem in filled): continue
                if isinstance(value,(int,float)) and np.isfinite(value):aggregate[key]=value
        datasets[dataset]=aggregate;total+=len(metrics)
        (root/(dataset+'_windows.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in metrics))
        print(json.dumps({'dataset':dataset,'windows':len(metrics),'status':'written'}),flush=True)
    if total!=100:raise ValueError('external method must retain exactly100 original windows')
    report={'status':'complete','method':method,'windows':total,'datasets':datasets,'filtering':'none','comparison':'non-same-sample; excluded from bolding and wins'}
    (root/'report.json').write_text(json.dumps(report,indent=2))
    (root/'summary.json').write_text(json.dumps({'status':'complete','windows':100,'report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest()}))
    (root/'COMPLETE').write_text('complete\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('CPU-only worker requires CUDA_VISIBLE_DEVICES empty')
    spec=json.loads(a.spec.read_text())
    (run_oracle if spec.get('mode')=='oracle' else run_external)(spec,a.output_root)
