"""Resident B=1 GT-K ablation of the frozen Result3 inference adapter."""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


def run(spec):
    for root, commit in ((spec['model_root'],spec['model_commit']), (spec['adapter_root'],spec['adapter_commit'])):
        actual=subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip()
        if actual != commit:raise ValueError('source commit mismatch: '+root)
    sys.path[:0]=[spec['adapter_root'],spec['model_root']]
    import torch
    import numpy as np
    path=Path(spec['adapter_root'])/'formal_evaluation/scene/adapters/run_egofound3r_baseline.py'
    module_spec=importlib.util.spec_from_file_location('frozen_result3_adapter',path)
    adapter=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(adapter)
    original_frames=adapter._load_frames
    original_contract=adapter._build_prediction_marker_forward_contract
    original_write=adapter.write_comparison_output
    current={}

    def load_frames(paths, *, target_hw, crop_config):
        record=current['record'];raw=np.asarray(record['intrinsics'],dtype=np.float32)
        valid=np.asarray(record['intrinsics_valid'],dtype=bool)
        if raw.shape!=(len(paths),3,3) or valid.shape!=(len(paths),) or not valid.all() or not np.isfinite(raw).all():
            raise ValueError('missing or invalid GT K: '+record['window_id'])
        samples=[{'rgb':adapter.load_media_ref(adapter.MediaRef(kind='path',path=str(p)),is_rgb=True),'hand_annos':[], 'intrinsics':torch.from_numpy(k.copy())} for p,k in zip(paths,raw)]
        crop=replace(crop_config,enabled=True,crop_probability=1.,target_shapes=[list(target_hw)])
        collator=adapter.MarkerRuntimeCollator(adapter._first_chunk,target_height=target_hw[0],target_width=target_hw[1],random_crop_resize=crop)
        chunk=collator([{'samples':samples,'target_image_size_hw':target_hw}])
        images=chunk['rgb'].unsqueeze(0)
        current['K']=torch.stack([s['intrinsics'] for s in chunk['samples']]).unsqueeze(0).float()
        if spec.get('pilot'):
            reference=original_frames(paths,target_hw=target_hw,crop_config=crop_config)
            if not torch.equal(images,reference):raise ValueError('GT K changed RGB preprocessing')
            current['rgb_bitwise_equal']=True
        return images

    def contract(**kwargs):
        kwargs['batch']={'intrinsics':current['K'],'intrinsics_supervision_mask':torch.ones(current['K'].shape[:2],dtype=torch.bool)}
        kwargs['use_gt_root_intrinsics']=True
        result,metrics=original_contract(**kwargs)
        if not torch.equal(result['root_solver_intrinsics'].cpu(),current['K']):raise ValueError('GT K forward contract mismatch')
        if not result['root_solver_intrinsics_valid'].all():raise ValueError('GT K silently invalidated')
        return result,metrics

    def write(output_dir, **kwargs):
        kwargs['metadata'].update(root_intrinsics_source='gt',root_intrinsics_preprocessing='same training center crop and resize as RGB',world_pose_source='predicted_camera_c2w',ablation='stride5_gt_k',base_adapter_commit=spec['adapter_commit'])
        kwargs['native_arrays']['root_solver_gt_intrinsics']=current['K'][0].numpy()
        kwargs['native_metadata']['gt_k_audit']={'all_frames_valid':True,'rgb_bitwise_equal_to_base':current.get('rgb_bitwise_equal')}
        return original_write(output_dir,**kwargs)
    adapter._load_frames=load_frames
    adapter._build_prediction_marker_forward_contract=contract
    adapter.write_comparison_output=write
    records=[]
    for raw in Path(spec['input_index']).read_text().splitlines():
        row=json.loads(raw);p=Path(row['window_input']);record=json.loads(p.read_text())
        if spec.get('window_ids') is not None and record['window_id'] not in spec['window_ids']:continue
        if record['dataset']!=spec['dataset'] or len(record['frame_ids'])!=60:raise ValueError('input identity mismatch')
        k=np.asarray(record['intrinsics']);v=np.asarray(record['intrinsics_valid'])
        if k.shape!=(60,3,3) or v.shape!=(60,) or not v.all() or not np.isfinite(k).all():raise ValueError('invalid GT K: '+record['window_id'])
        for path in record['rgb_paths']:
            if not Path(path).is_file():raise FileNotFoundError(path)
        records.append((p,record))
    if len(records)!=spec['expected_windows'] or len({r['window_id'] for _,r in records})!=len(records):raise ValueError('window count mismatch')
    root=Path(spec['output_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'input_audit.json').write_text(json.dumps({'windows':len(records),'gt_k_valid_frames':len(records)*60,'dataset':spec['dataset']}))
    with (root/'predictions.jsonl').open('x') as index, (root/'progress.jsonl').open('x') as progress:
        for i,(path,record) in enumerate(records,1):
            current.clear();current['record']=record;start=time.time()
            adapter.main(['--phase','formal','--window-input',str(path),'--methods-config',spec['methods_config'],'--config',spec['config'],'--checkpoint',spec['checkpoint'],'--backbone-checkpoint',spec['backbone'],'--global-stride','5','--input-resolution','512x512','--output-root',str(root)])
            if adapter._runtime.cache_info().misses!=1:raise ValueError('model not resident')
            index.write(json.dumps({'method':'egofound3r','variant':'stride5_gt_k','dataset':spec['dataset'],'window_id':record['window_id'],'prediction_dir':str(root/'egofound3r/formal'/record['cache_id'])})+'\n');index.flush()
            row={'completed':i,'total':len(records),'window_id':record['window_id'],'seconds':time.time()-start,'model_loads':1,'root_intrinsics_source':'gt','rgb_parity':current.get('rgb_bitwise_equal')}
            progress.write(json.dumps(row)+'\n');progress.flush();print(json.dumps(row),flush=True)
    summary={'status':'complete','phase':'inference_only','windows':len(records),'dataset':spec['dataset'],'root_intrinsics_source':'gt','world_pose_source':'predicted_camera_c2w','model_commit':spec['model_commit'],'adapter_commit':spec['adapter_commit'],'checkpoint_sha256':spec['checkpoint_sha256'],'model_resident':True,'batch_size':1,'stride':5,'input_resolution':'512x512','pilot':spec.get('pilot',False)}
    (root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'COMPLETE').write_text('complete\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();run(json.loads(a.spec.read_text()))
