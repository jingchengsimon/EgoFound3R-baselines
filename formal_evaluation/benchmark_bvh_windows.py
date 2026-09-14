"""Isolated full-window hand/scene/contact benchmark; no production resumption."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from formal_evaluation.contact.bvh_compat import install


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def init():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def window(task):
    s,job,output,backend=task
    started=time.perf_counter();cpu=time.process_time();stages={}
    from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
    from formal_evaluation.evaluate_six_dataset import _prediction_arrays
    from formal_evaluation.common.mano_sampling import _level1_matrices
    from formal_evaluation.contact.metrics import compute_contact_metrics
    common=load('window_common',s['same_mask_script']);hand=load('window_hand',s['hand_script'])
    scene=load('window_scene',s['scene_script']);geometry=load('window_geometry',s['geometry_script'])
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    ds=job['dataset'];index=Path(job['gt_index'])
    raw=common.read_jsonl(index);gt0=sorted(raw,key=lambda x:x['window_id'])[0]
    gt=common.remap_gt_row(gt0,Path('/unused'),index.parent,direct_oss=True);wid=gt['window_id']
    if 'prediction_index' in job:
        candidates=[Path(r['prediction_dir']) for r in common.read_jsonl(Path(job['prediction_index'])) if r['window_id']==wid]
    else:
        candidates=[Path(p)/gt['cache_id'] for p in job['formal_roots'] if (Path(p)/gt['cache_id']/'predictions.npz').is_file()]
    assert len(candidates)==1, (s['method'],ds,wid,candidates)
    pm,pred=_prediction_arrays(candidates[0]);gm,target=load_window_cache(gt)
    selection_raw=Path(s['selection']).read_bytes();assert hashlib.sha256(selection_raw).hexdigest()==s['selection_sha256']
    sel=next(r for r in map(json.loads,selection_raw.splitlines()) if r['dataset']==ds and r['window_id']==wid)
    record=None
    for r in common.read_jsonl(Path(job['input_index'])):
        item=json.loads(Path(r['window_input']).read_text())
        if item['window_id']==wid:record=item;break
    assert record is not None and pm['frame_ids']==gm['frame_ids']==record['frame_ids']==sel['frame_ids']
    assert len(pm['frame_ids'])==60 and 'camera_c2w' in pred
    keep=np.asarray(sel['keep'],bool)
    def arrays(row,idx=None):
        path=Path(row['array_path'])
        if idx and '/gt_cache/' in str(path):path=Path(idx).parent/str(path).split('/gt_cache/',1)[1]
        with np.load(path,allow_pickle=False) as a:return dict(a)
    cg=next(r for r in common.read_jsonl(Path(s['contact_gt_index'])) if r['dataset']==ds and r['window_id']==wid)
    vg=next(r for r in common.read_jsonl(Path(job['visibility_gt_index'])) if r['window_id']==wid)
    assert cg['frame_ids']==vg['frame_ids']==gm['frame_ids']
    contact=arrays(cg);visibility=arrays(vg,job['visibility_gt_index'])
    faces=np.asarray(pickle.load(open(s['mano_asset'],'rb'),encoding='latin1')['f'])
    stages['load_s']=time.perf_counter()-started
    t=time.perf_counter();_,frame,pair,triplet=common.freeze_hand_window(hand,pred,target)
    _,hr=hand.aggregate_scheme(frame[None],pair[None],triplet[None],~keep[None])
    hand_metrics=hr[0];stages['hand_s']=time.perf_counter()-t
    t=time.perf_counter();frozen=scene.freeze_scene(pred,target);scene_metrics=scene.aggregate_scene_window(frozen,keep)
    stages['scene_s']=time.perf_counter()-t
    t=time.perf_counter();query=geometry.query_geometry(pred);collected={};io=0.;stats={}
    kernel=geometry.surface_kernel()
    with (install(kernel) if backend=='bvh' else nullcontext()) as fast:
        for f,path in enumerate(record['geometry_paths']):
            ti=time.perf_counter()
            with np.load(path,allow_pickle=False) as a:surfaces=dict(a)
            io+=time.perf_counter()-ti
            if fast is not None:fast.clear()
            values=geometry.frame_geometry_metrics({k:v[f] for k,v in query.items()},pred['hand_valid'][f],surfaces,faces,'cpu')
            for key,value in values.items():collected.setdefault(key,[]).append(value)
        if fast is not None:stats=dict(fast.stats)
    overlay={k:np.stack(v) for k,v in collected.items()}
    stages['geometry_io_s']=io;stages['geometry_s']=time.perf_counter()-t-io
    t=time.perf_counter();pred['vertex_contact_probability']=overlay['vertex_contact_target'];classification={}
    for mode,target_arrays in [('contact',contact),('visibility',visibility)]:
        for prefix in ('joint','marker','vertex'):
            field=prefix+'_'+mode
            if mode=='visibility' and prefix=='vertex':
                parent=np.abs(_level1_matrices()[1]).argmax(axis=1)
                prob=pred['marker_visibility'][...,parent];labels=target_arrays['marker_visibility_target'][...,parent]
                mask=target_arrays['marker_visibility_mask'][...,parent].astype(bool)
            else:
                key=field+'_probability' if mode=='contact' else ('hand_visibility' if prefix=='joint' else 'marker_visibility')
                prob=pred[key];labels=target_arrays[field+'_target'];mask=target_arrays[field+'_mask'].astype(bool).copy()
            mask &= keep[:,None,None] & pred['hand_valid'][...,None]
            for key,value in compute_contact_metrics(prob,labels,mask).items():
                classification[field+'_'+key]=np.asarray(value['ap'] if key=='average_precision' and isinstance(value,dict) else value).item()
    distances={k:v for k,v in geometry.aggregate_geometry_contact({**pred,**{k:v for k,v in overlay.items() if '_distance' in k}}, {},keep).items() if '_distance_' in k}
    stages['classification_s']=time.perf_counter()-t
    t=time.perf_counter()
    np.savez_compressed(root/'hand_frozen.npz',frame=frame,pair=pair,triplet=triplet)
    np.savez_compressed(root/'scene_frozen.npz',**frozen)
    np.savez_compressed(root/'geometry.npz',**overlay)
    result=dict(method=s['method'],dataset=ds,window_id=wid,frame_ids=gm['frame_ids'],backend=backend,pid=os.getpid(),
        hand=hand_metrics,scene=scene_metrics,classification=classification,distances=distances,stages=stages,bvh_stats=stats,
        output_root=str(root),selection_sha256=s['selection_sha256'],fit_before_filter=True,predicted_camera=True)
    (root/'metrics.json').write_text(json.dumps(common.json_safe(result),allow_nan=False))
    stages['write_s']=time.perf_counter()-t;result['wall_s']=time.perf_counter()-started;result['cpu_s']=time.process_time()-cpu
    (root/'timing.json').write_text(json.dumps(common.json_safe(result),allow_nan=False));(root/'COMPLETE').write_text('complete\n')
    print(json.dumps(dict(stage='window_done',method=s['method'],dataset=ds,backend=backend,wall_s=result['wall_s'])),flush=True)
    return common.json_safe(result)


def parity(a,b):
    assert a['window_id']==b['window_id'] and a['frame_ids']==b['frame_ids']
    report={}
    for group in ['hand','scene','classification','distances']:
        assert a[group].keys()==b[group].keys()
        diffs=[]
        for key,x in a[group].items():
            y=b[group][key]
            if x is None or y is None:assert x is None and y is None,(group,key);continue
            error=abs(float(x)-float(y));diffs.append(error)
            assert error <= (1e-3 if group=='distances' and key.endswith('_mm') else 1e-10),(group,key,x,y)
        report[group]=dict(max_abs=max(diffs,default=0.),fields=len(a[group]))
    with np.load(Path(a['output_root'])/'geometry.npz') as old, np.load(Path(b['output_root'])/'geometry.npz') as new:
        for key in old.files:
            x=old[key];y=new[key];assert np.array_equal(np.isfinite(x),np.isfinite(y))
            good=np.isfinite(x);error=float(np.abs(x[good].astype(float)-y[good].astype(float)).max(initial=0))
            assert error <= (1e-6 if key.endswith('_distance') else 0),(key,error)
            report[key]=dict(max_abs=error,count=int(good.sum()))
    return report


def main(spec):
    init();root=Path(spec['output_root']);root.mkdir(parents=True,exist_ok=False)
    s=spec['tasks'][0]['spec'];geometry=load('gate_geometry',s['geometry_script'])
    tests=load('gate_tests',Path(__file__).parent/'tests/test_bvh_compat.py')
    regression=tests.check(geometry.surface_kernel());(root/'regression.json').write_text(json.dumps(regression,indent=2))
    first=spec['tasks'][0]
    old=window((first['spec'],first['job'],str(root/'reference'),'legacy'))
    fast=window((first['spec'],first['job'],str(root/'serial_bvh'),'bvh'))
    comparison=parity(old,fast);(root/'parity.json').write_text(json.dumps(comparison,indent=2))
    report=dict(status='running',workers=spec['workers'],regression=regression,serial_reference=old,serial_bvh=fast,parity=comparison,parallel=[])
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=spec['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        futures={pool.submit(window,(t['spec'],t['job'],str(root/'parallel'/str(i)),'bvh')):i for i,t in enumerate(spec['tasks'])}
        for future in as_completed(futures):
            row=future.result();row['task_index']=futures[future];report['parallel'].append(row)
            (root/'report.json').write_text(json.dumps(report,indent=2))
    report.update(status='complete',parallel_wall_s=time.perf_counter()-started)
    # The repeated reference window also verifies equivalence under concurrency.
    report['parallel_parity']=parity(old,next(r for r in report['parallel'] if r['task_index']==0))
    (root/'report.json').write_text(json.dumps(report,indent=2));(root/'COMPLETE').write_text('complete\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);args=parser.parse_args()
    main(json.loads(args.spec.read_text()))
