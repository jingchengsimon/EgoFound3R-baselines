"""Eight CPU workers, one scene report referenced by both no-VGGT and one-way."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def window(args):
    spec, job, gt, selected, source, destination = args
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), spec['model_root'], spec['baseline_root']]
    import numpy as np
    from formal_evaluation.raw_omega_scene import recover_raw_scene
    from formal_evaluation.run_ablation_seven_tables import load
    from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
    scene = load('raw_scene_metrics', spec['scene_script'])
    source = Path(source)
    meta = json.loads((source/'metadata.json').read_text())
    assert meta['frame_ids'] == gt['frame_ids'] == selected['frame_ids']
    assert meta['inference_commit'] == spec['model_commit']
    assert meta['processed_resolution_hw'] == [256, 256]
    assert meta['global_stride'] == 5 and meta['global_anchor_phase'] == 2
    with np.load(source/'predictions.npz', allow_pickle=False) as saved:
        prediction = {k:saved[k] for k in ('depth','depth_confidence')}
    with np.load(source/'native/predictions.npz', allow_pickle=False) as saved:
        native = {k:saved[k] for k in ('global_anchor_indices','camera_pose_encoding_global','metric_scale_factor')}
    raw = recover_raw_scene(prediction, native)
    _, target = load_window_cache(gt)
    frozen = scene.freeze_scene(raw, target)
    row = dict(window_id=gt['window_id'], frame_ids=gt['frame_ids'], **scene.aggregate_scene_window(frozen, np.asarray(selected['keep'], bool)))
    np.savez_compressed(Path(destination)/(gt['cache_id']+'_scene.npz'), **frozen)
    return row


def run(spec, pilot=False):
    assert spec['workers'] == 8
    assert subprocess.check_output(['git','-C',spec['model_root'],'rev-parse','HEAD'],text=True).strip() == spec['model_commit']
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), spec['baseline_root']]
    from formal_evaluation.run_ablation_seven_tables import load
    from formal_evaluation.common.aggregation import aggregate_windows
    common = load('raw_scene_common',spec['same_mask_script'])
    raw = Path(spec['selection']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == spec['selection_sha256']
    selection = {(r['dataset'],r['window_id']):r for r in map(json.loads,raw.splitlines())}
    root=Path(spec['output_root']);root.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',method='shared_raw_omega',consumers=['no_vggt','one_way'],windows=0,datasets={},selection_sha256=spec['selection_sha256'],protocol='Omega raw native pose encoding and recovered unscaled depth; camera-only deterministic cubic/SQUAD; full-window fit then frozen P95; no Hand refinement or scale',source_jobs=spec['jobs'])
    with ProcessPoolExecutor(max_workers=8) as pool:
        for job in spec['jobs']:
            ds=job['dataset'];index=Path(job['gt_index']);rows=common.read_jsonl(index)
            gt={r['window_id']:common.remap_gt_row(r,Path('/unused'),index.parent,direct_oss=True) for r in rows}
            aliases={r['cache_id']:r['window_id'] for r in rows}
            found=common.prediction_rows('egofound3r',{'dataset':ds,'expected_windows':len(gt),'predictions':{'egofound3r':{'formal_roots':job['formal_roots']}}},Path('/unused'),aliases,direct_oss=True)
            assert set(found)==set(gt) and len(gt)==job['expected_windows']
            folder=root/ds;folder.mkdir();ids=sorted(gt)[:1] if pilot else sorted(gt);values=[]
            tasks=[(spec,job,gt[wid],selection[(ds,wid)],str(found[wid]),str(folder)) for wid in ids]
            with (folder/'window_metrics.jsonl').open('x') as output:
                for row in pool.map(window,tasks,chunksize=1):
                    values.append(row);output.write(json.dumps(common.json_safe(row),allow_nan=False)+'\n');output.flush()
                    print(json.dumps(dict(dataset=ds,completed=len(values),total=len(ids))),flush=True)
            report['datasets'][ds]=aggregate_windows(values,method='shared_raw_omega');report['windows']+=len(values)
            (folder/'COMPLETE').write_text('complete')
            (root/'report.json').write_text(json.dumps(common.json_safe(report),allow_nan=False))
            if pilot:break
    assert report['windows']==(1 if pilot else 2378)
    report['status']='complete';(root/'report.json').write_text(json.dumps(common.json_safe(report),allow_nan=False));(root/'COMPLETE').write_text('complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--pilot',action='store_true');args=parser.parse_args();run(json.loads(args.spec.read_text()),args.pilot)
