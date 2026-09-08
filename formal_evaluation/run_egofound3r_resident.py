"""One resident model, real clip batching, and exact non-overwriting continuation."""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formal_evaluation.scene.adapters import run_egofound3r_baseline as adapter
from formal_evaluation.run_egofound3r_stride_evaluation import prepare_inputs


def batch_slices(outputs, frame_map, index, batch_size):
    keys = ('camera_pose_refined_high', 'camera_refined_valid_high',
            'interpolation_scene_metric_scale_valid', 'interpolation_scene_metric_scale_factor',
            'intrinsics_global', 'depth_global', 'depth_conf_global', 'dense_joint_xyz',
            'dense_vertex_xyz', 'in_view_probability', 'root_translation_valid',
            'dense_joint_visibility_logits', 'dense_vertex_visibility_logits',
            'dense_joint_contact_logits', 'dense_vertex_contact_logits', 'camera_pose_encoding_global')
    result = {}
    for key in keys:
        value = outputs[key]
        assert isinstance(value, torch.Tensor) and value.shape[0] == batch_size, (key, value.shape)
        result[key] = value[index:index + 1]
    values = {}
    for field in dataclasses.fields(frame_map):
        value = getattr(frame_map, field.name)
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == batch_size, field.name
            value = value[index:index + 1]
        values[field.name] = value
    return result, {'multirate_frame_map': dataclasses.replace(frame_map, **values)}


class Resident:
    def __init__(self, spec):
        self.spec = spec
        self.method = json.loads(Path(spec['methods_config']).read_text())['methods']['egofound3r']
        torch.cuda.set_device(0)
        checkpoint = Path(spec['checkpoint'])
        self.sha, self.config, self.model, self.dtype = adapter._runtime(
            spec['config'], spec['checkpoint'], spec['backbone'], spec['checkpoint_sha256'],
            'cuda:0', checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns,
            self.method.get('runtime_mode', 'checkpoint_native'))
        self.model_id = id(self.model)

    def run(self, records, output_root, batch_size):
        assert records and len(records) <= batch_size
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        # Load directly into the target dtype; keep the 60-frame axis separate from B.
        frames = torch.cat([adapter._load_frames(
            [Path(x) for x in r['rgb_paths']], height=self.config.marker_runtime.image_height,
            width=self.config.marker_runtime.image_width).to(dtype=self.dtype) for r in records])
        frames = frames.to('cuda:0')
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with torch.inference_mode():
            contract, _ = adapter._build_prediction_marker_forward_contract(
                project_config=self.config, batch={}, images=frames, global_stride=5, global_anchor_phase=2)
            outputs = adapter._public_marker_outputs(adapter._call_marker_model(self.model, frames, **contract))
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_start
        peak = torch.cuda.max_memory_allocated()
        for i, record in enumerate(records):
            sliced, sliced_contract = batch_slices(outputs, contract['multirate_frame_map'], i, len(records))
            args = SimpleNamespace(global_stride=5, phase='formal', device='cuda:0',
                                   output_root=Path(output_root), checkpoint=Path(self.spec['checkpoint']))
            window = (record['sequence_id'], record['window_id'], record['frame_ids'],
                      [Path(x) for x in record['rgb_paths']], record['dataset'], record['cache_id'])
            adapter.export_window(args, window, self.config, self.model, self.dtype, self.sha,
                                  self.method, sliced, sliced_contract, forward_seconds / len(records), peak)
            del sliced, sliced_contract
        del outputs, contract, frames
        torch.cuda.synchronize()
        assert id(self.model) == self.model_id
        return dict(batch_size=len(records), requested_batch_size=batch_size, model_resident=True,
                    forward_seconds=forward_seconds, wall_seconds=time.perf_counter()-started,
                    peak_allocated_bytes=peak, allocated_after_bytes=torch.cuda.memory_allocated(),
                    reserved_after_bytes=torch.cuda.memory_reserved())


def existing(record, roots, spec):
    found = []
    for root in roots:
        path = Path(root)/record['cache_id']
        if not (path/'run.json').is_file():
            continue
        try:
            meta=json.loads((path/'metadata.json').read_text());run=json.loads((path/'run.json').read_text())
            assert run['status']=='success' and (path/'predictions.npz').stat().st_size>0
            assert meta['window_id']==record['window_id'] and meta['frame_ids']==record['frame_ids']
            assert all(meta[k]==spec[k] for k in ('source_commit','inference_commit','checkpoint_sha256'))
            assert meta['global_stride']==5 and meta['global_anchor_phase']==2
        except (OSError,ValueError,KeyError,AssertionError):
            continue
        found.append(path)
    if len(found)>1:raise ValueError(('DUPLICATE_PRESERVED_WINDOW',record['cache_id']))
    return found[0] if found else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,choices=(1,16),required=True)
    args=parser.parse_args();spec=json.loads(args.spec.read_text());root=args.output_root
    root.mkdir(parents=True,exist_ok=False)
    runtime=Resident(spec)
    for dataset,data in spec['datasets'].items():
        _,gt,records=prepare_inputs(dataset,data,root/'inputs'/dataset)
        reused={};pending=[]
        for record in records:
            path=existing(record,spec.get('prior_roots',{}).get(dataset,[]),spec)
            if path is None:pending.append(record)
            else:reused[record['cache_id']]=str(path)
        out=root/dataset;out.mkdir()
        print(json.dumps(dict(dataset=dataset,preserved=len(reused),pending=len(pending),mode='resident',batch_size=args.batch_size)),flush=True)
        with (out/'batch_stats.jsonl').open('x') as log:
            for offset in range(0,len(pending),args.batch_size):
                batch=pending[offset:offset+args.batch_size]
                stats=runtime.run(batch,out,args.batch_size)
                stats.update(dataset=dataset,completed=offset+len(batch),pending_total=len(pending))
                log.write(json.dumps(stats)+'\n');log.flush();print(json.dumps(stats),flush=True)
        index=out/'predictions.jsonl'
        index.write_text(''.join(json.dumps(dict(method='egofound3r',dataset=dataset,window_id=r['window_id'],
            prediction_dir=reused.get(r['cache_id'],str(out/'egofound3r/formal'/r['cache_id']))))+'\n' for r in records))
        import subprocess
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_six_dataset.py')),
                        '--gt-index',str(gt),'--prediction-index',str(index),
                        '--methods-config',spec['methods_config'],'--report-path',str(out/'report.json')],check=True)
        (out/'COMPLETE').write_text('complete\n')
    (root/'summary.json').write_text(json.dumps(dict(status='complete',datasets=list(spec['datasets']),batch_size=args.batch_size,model_resident=True)))
    (root/'COMPLETE').write_text('complete\n')

if __name__=='__main__':main()
