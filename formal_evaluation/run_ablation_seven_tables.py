"""Complete scene/contact tables from canonical predictions and frozen hand reports."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(spec, pilot=False):
    sys.path.insert(0, spec['baseline_root'])
    import numpy as np
    from formal_evaluation.common.aggregation import aggregate_windows
    from formal_evaluation.common.mano_sampling import _level1_matrices
    from formal_evaluation.contact.metrics import compute_contact_metrics
    from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
    from formal_evaluation.evaluate_six_dataset import _prediction_arrays
    common = load('seven_common', spec['same_mask_script'])
    scene = load('seven_scene', spec['scene_script'])
    geometry = load('seven_geometry', spec['geometry_script'])
    raw = Path(spec['selection']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == spec['selection_sha256']
    selection = {(r['dataset'], r['window_id']): r for r in map(json.loads, raw.splitlines())}
    assert len(selection) == 2378
    upstream = Path(spec['upstream_root'])
    if not (upstream/'COMPLETE').is_file():
        raise RuntimeError('upstream hand evaluation incomplete: '+str(upstream))
    previous = json.loads((upstream/'report.json').read_text())
    assert previous['windows'] == 2378 and previous['selection_sha256'] == spec['selection_sha256']
    root = Path(spec['output_root']); root.mkdir(parents=True, exist_ok=False)
    def progress(**row):
        row['time'] = time.time()
        with (root/'progress.jsonl').open('a') as out: out.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    def arrays(row, index=None):
        path = Path(row['array_path'])
        if index and '/gt_cache/' in str(path): path = Path(index).parent / str(path).split('/gt_cache/',1)[1]
        with np.load(path, allow_pickle=False) as archive: return dict(archive)
    def classify(pred, target, keep, mode):
        result = {}
        for prefix in ('joint','marker','vertex'):
            field = prefix+'_'+mode
            if mode == 'visibility' and prefix == 'vertex':
                parent = np.abs(_level1_matrices()[1]).argmax(axis=1)
                prob = pred['marker_visibility'][...,parent]
                labels = target['marker_visibility_target'][...,parent]
                valid = target['marker_visibility_mask'][...,parent].astype(bool)
            else:
                key = field+'_probability' if mode == 'contact' else ('hand_visibility' if prefix=='joint' else 'marker_visibility')
                prob = pred[key]; labels=target[field+'_target']; valid=target[field+'_mask'].astype(bool).copy()
            valid &= keep[:,None,None] & pred['hand_valid'][...,None]
            result.update({field+'_'+k: np.asarray(v['ap'] if k=='average_precision' and isinstance(v,dict) else v).item() for k,v in compute_contact_metrics(prob,labels,valid).items()})
        return result
    report = {'status':'running','method':spec['method'],'windows':0,'datasets':{},'selection_sha256':spec['selection_sha256'],
              'world_pose_source':'predicted_camera_c2w','fit_on_full_unfiltered_window':True,'no_refit_after_filter':True,
              'vertex_contact':'derived 14mm geometry rule, matching formal stride5; not a native contact head',
              'vertex_visibility':'dominant fixed marker parent, matching formal stride5','hand_source':str(upstream/'report.json')}
    faces = None
    import pickle
    with open(spec['mano_asset'],'rb') as f: faces=np.asarray(pickle.load(f,encoding='latin1')['f'])
    contact_rows = common.read_jsonl(Path(spec['contact_gt_index']))
    for job in spec['jobs']:
        ds=job['dataset']; index=Path(job['gt_index']); raw_gt=common.read_jsonl(index)
        gt={r['window_id']:common.remap_gt_row(r,Path('/unused'),index.parent,direct_oss=True) for r in raw_gt}
        aliases={r['cache_id']:r['window_id'] for r in raw_gt}
        if 'prediction_index' in job:
            found={aliases.get(r['window_id'],r['window_id']):Path(r['prediction_dir']) for r in common.read_jsonl(Path(job['prediction_index']))}
        else: found=common.prediction_rows('egofound3r',{'dataset':ds,'expected_windows':len(gt),'predictions':{'egofound3r':{'formal_roots':job['formal_roots']}}},Path('/unused'),aliases,direct_oss=True)
        inputs={}
        for r in common.read_jsonl(Path(job['input_index'])):
            record=json.loads(Path(r['window_input']).read_text()); inputs[record['window_id']]=record
        cg={r['window_id']:r for r in contact_rows if r['dataset']==ds}
        vg={r['window_id']:r for r in common.read_jsonl(Path(job['visibility_gt_index']))}
        assert set(found)==set(gt)==set(inputs)==set(cg)==set(vg) and len(gt)==job['expected_windows'], ds
        folder=root/ds;folder.mkdir(); values=[]
        with (folder/'window_metrics.jsonl').open('x') as out:
            for i,wid in enumerate(sorted(gt),1):
                pm,pred=_prediction_arrays(found[wid]);gm,target=load_window_cache(gt[wid]);record=inputs[wid];sel=selection[(ds,wid)]
                assert pm['frame_ids']==gm['frame_ids']==record['frame_ids']==sel['frame_ids']==cg[wid]['frame_ids']==vg[wid]['frame_ids']
                keep=np.asarray(sel['keep'],bool)
                frozen=scene.freeze_scene(pred,target)
                metrics=scene.aggregate_scene_window(frozen,keep)
                query=geometry.query_geometry(pred); collected={}
                assert len(record['geometry_paths'])==len(keep)
                for f,path in enumerate(record['geometry_paths']):
                    with np.load(path,allow_pickle=False) as a: surfaces=dict(a)
                    for key,v in geometry.frame_geometry_metrics({k:v[f] for k,v in query.items()},pred['hand_valid'][f],surfaces,faces,'cpu').items(): collected.setdefault(key,[]).append(v)
                overlay={k:np.stack(v) for k,v in collected.items()}
                pred['vertex_contact_probability']=overlay['vertex_contact_target']
                metrics.update(classify(pred,arrays(cg[wid]),keep,'contact'))
                metrics.update(classify(pred,arrays(vg[wid],job['visibility_gt_index']),keep,'visibility'))
                metrics.update({k:v for k,v in geometry.aggregate_geometry_contact({**pred,**{k:v for k,v in overlay.items() if '_distance' in k}}, {},keep).items() if '_distance_' in k})
                row={'window_id':wid,'frame_ids':gm['frame_ids'],**metrics};values.append(row)
                out.write(json.dumps(common.json_safe(row),allow_nan=False)+'\n');out.flush()
                np.savez_compressed(folder/(gt[wid]['cache_id']+'_scene.npz'),**frozen)
                progress(stage='scene_contact',dataset=ds,completed=i,total=len(gt))
                if pilot:
                    (root/'pilot.json').write_text(json.dumps(common.json_safe(row),allow_nan=False));return
        merged=dict(previous['datasets'][ds]); merged.update(aggregate_windows(values,method=spec['method']))
        report['datasets'][ds]=merged;report['windows']+=len(values)
        (folder/'COMPLETE').write_text('complete\n')
        (root/'report.json').write_text(json.dumps(common.json_safe(report),allow_nan=False))
    assert report['windows']==2378
    report['status']='complete'
    (root/'report.json').write_text(json.dumps(common.json_safe(report),allow_nan=False))
    (root/'summary.json').write_text(json.dumps({'status':'complete','windows':2378,'tables':7,'method':spec['method']}))
    (root/'COMPLETE').write_text('complete\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();run(json.loads(args.spec.read_text()),args.pilot)
