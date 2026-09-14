"""Isolated CPU BVH parity/timing probe; never modifies formal predictions."""
import argparse
import importlib.util
import json
import pickle
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def compare(a,b):
    report={}
    for key,x in a.items():
        y=b[key];valid=np.isfinite(x)&np.isfinite(y);diff=np.abs(x[valid].astype(float)-y[valid].astype(float))
        report[key]={'finite_equal':bool(np.array_equal(np.isfinite(x),np.isfinite(y))),'max_abs':float(diff.max()) if diff.size else 0.,'changed':int(np.count_nonzero(x[valid]!=y[valid])),'n':int(valid.sum())}
    return report


def main(s):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,s['baseline_root'])
    from formal_evaluation.evaluate_six_dataset import _prediction_arrays
    geometry=load('probe_geometry',s['geometry_script']);kernel=geometry.surface_kernel();original=kernel._point_to_mesh_distances
    accel=load('probe_accel',s['accelerator_script'])
    scene_class=accel.o3d.t.geometry.RaycastingScene
    class SingleThreadScene:
        def __init__(self):self.scene=scene_class(nthreads=1)
        def add_triangles(self,*a,**kw):return self.scene.add_triangles(*a,**kw)
        def compute_distance(self,*a,**kw):return self.scene.compute_distance(*a,nthreads=1,**kw)
    accel.o3d.t.geometry.RaycastingScene=SingleThreadScene
    root=Path(s['output_root']);root.mkdir(parents=True,exist_ok=False)
    report={'threads':1,'real':[],'synthetic':[],'timing_includes_bvh_build':True}
    # Exercise small faces at the PyTorch3D degeneracy boundary and contact thresholds.
    for size in [1.,0.00008]:
        verts=torch.tensor([[0.,0.,0.],[size,0.,0.],[0.,size,0.]])
        mesh=kernel._valid_mesh((verts,torch.tensor([[0,1,2]])),device=torch.device('cpu'))
        points=torch.tensor([[size/4,size/4,z] for z in [0.,0.000001,0.0139999,0.014,0.0140001,0.0179999,0.018,0.0180001]])
        d,valid=original(points,torch.ones(len(points),dtype=torch.bool),mesh)
        fast=accel.ObjectMeshAccel.from_local_mesh(*mesh).distances_camera(points,torch.eye(4))
        report['synthetic'].append({'triangle_size_m':size,'old':d.tolist(),'bvh':fast.tolist(),'max_abs_m':float((d-fast).abs().max()),'flips_14mm':int(((d<=.014)!=(fast<=.014)).sum()),'flips_18mm':int(((d<=.018)!=(fast<=.018)).sum())})
    faces=np.asarray(pickle.load(open(s['mano_asset'],'rb'),encoding='latin1')['f'])
    selection={(r['dataset'],r['window_id']):r for r in map(json.loads,Path(s['selection']).read_text().splitlines())}
    for job in s['jobs']:
        ds=job['dataset'];gt=json.loads(Path(job['gt_index']).read_text().splitlines()[0]);wid=gt['window_id'];pred_dir=next(Path(p)/gt['cache_id'] for p in job['formal_roots'] if (Path(p)/gt['cache_id']/'predictions.npz').exists())
        meta,pred=_prediction_arrays(pred_dir);record=None
        for line in Path(job['input_index']).read_text().splitlines():
            item=json.loads(Path(json.loads(line)['window_input']).read_text())
            if item['window_id']==wid:record=item;break
        assert record is not None and record['frame_ids']==meta['frame_ids']==selection[(ds,wid)]['frame_ids']
        query=geometry.query_geometry(pred);candidates=np.flatnonzero(np.asarray(selection[(ds,wid)]['keep']) & np.asarray(pred['hand_valid']).any(axis=1));chosen=[int(candidates[0]),int(candidates[len(candidates)//2])];assert len(set(chosen))==2
        for frame in chosen:
            with np.load(record['geometry_paths'][frame],allow_pickle=False) as a:surface=dict(a)
            points={k:v[frame] for k,v in query.items()};valid=pred['hand_valid'][frame]
            t=time.perf_counter();c=time.process_time();reference=geometry.frame_geometry_metrics(points,valid,surface,faces,'cpu');ref_cpu=time.process_time()-c;ref_wall=time.perf_counter()-t
            times=[];cpu_times=[];build_counts=[]
            for repeat in range(2):
                cache={}
                def accelerated(points,mask,mesh,*,computation_device=None):
                    distances=torch.zeros(mask.shape,dtype=torch.float32);good=torch.zeros_like(mask)
                    if points is None or mesh is None or not bool(mask.any()):return distances,good
                    key=(mesh[0].data_ptr(),tuple(mesh[0].shape),tuple(mesh[1].shape))
                    if key not in cache:cache[key]=accel.ObjectMeshAccel.from_local_mesh(*mesh)
                    else:assert torch.equal(cache[key].local_faces,mesh[1])
                    indices=torch.nonzero(mask).flatten();d=cache[key].distances_camera(points[indices],torch.eye(4));distances[indices]=d;good[indices]=torch.isfinite(d)&(d>=0);return distances,good
                kernel._point_to_mesh_distances=accelerated
                try:
                    t=time.perf_counter();c=time.process_time();fast=geometry.frame_geometry_metrics(points,valid,surface,faces,'cpu');cpu_times.append(time.process_time()-c);times.append(time.perf_counter()-t);build_counts.append(len(cache))
                finally:kernel._point_to_mesh_distances=original
            diff=compare(reference,fast);row=dict(dataset=ds,window_id=wid,frame=frame,object_faces=len(surface['object_faces']),ref_wall_s=ref_wall,ref_cpu_s=ref_cpu,bvh_wall_s=statistics.median(times),bvh_cpu_s=statistics.median(cpu_times),build_counts=build_counts,speedup=ref_wall/statistics.median(times),differences=diff)
            report['real'].append(row);np.savez_compressed(root/(ds+'_'+str(frame)+'.npz'),**{'old_'+k:v for k,v in reference.items()},**{'bvh_'+k:v for k,v in fast.items()});(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    report['status']='complete';(root/'report.json').write_text(json.dumps(report,indent=2));(root/'COMPLETE').write_text('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();main(json.loads(a.spec.read_text()))
