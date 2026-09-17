"""Recompute paired geometric MAE on registered Ego windows with frozen selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def remap(path, mappings):
    for old, new in mappings:
        if path == old or path.startswith(old + '/'):
            return new + path[len(old):]
    return path


def initialize(spec):
    global CONFIG, GEOMETRY, BVH, ERRORS, FACES
    CONFIG = spec
    sys.path.insert(0, spec['baseline_root'])
    GEOMETRY = load('distance_geometry', spec['geometry_script'])
    BVH = load('distance_bvh', spec['bvh_script'])
    ERRORS = load('distance_errors', Path(__file__).parent/'contact/distance_error.py')
    import numpy as np
    with open(spec['mano_asset'], 'rb') as stream:
        FACES = np.asarray(pickle.load(stream, encoding='latin1')['f'])


def compute(job):
    import numpy as np
    ds, wid, pred_dir, record, target_row, selection = job
    pred_dir = Path(pred_dir)
    metadata = json.loads((pred_dir/'metadata.json').read_text())
    assert metadata['frame_ids'] == record['frame_ids'] == target_row['frame_ids'] == selection['frame_ids']
    with np.load(pred_dir/'predictions.npz', allow_pickle=False) as a:
        pred = dict(a)
    with np.load(remap(target_row['array_path'], CONFIG['path_mappings']), allow_pickle=False) as a:
        target = dict(a)
    geometry = GEOMETRY.query_geometry(pred)
    collected = {}
    with BVH.install(GEOMETRY.surface_kernel()) as accelerator:
        for frame, path in enumerate(record['geometry_paths']):
            with np.load(remap(path, CONFIG['path_mappings']), allow_pickle=False) as a:
                scene = dict(a)
            accelerator.clear()
            for key, value in GEOMETRY.frame_geometry_metrics(
                {k:v[frame] for k,v in geometry.items()}, pred['hand_valid'][frame], scene, FACES, 'cpu'
            ).items(): collected.setdefault(key, []).append(value)
    overlay = {k:np.stack(v) for k,v in collected.items()}
    overlay['joint_contact_probability'] = pred['joint_contact_probability']
    overlay['marker_contact_probability'] = pred['marker_contact_probability']
    overlay['vertex_contact_probability'] = overlay['vertex_contact_target']
    metrics = ERRORS.window_distance_errors(overlay, target, selection['keep'])
    return {'dataset':ds, 'window_id':wid, 'frame_ids':record['frame_ids'], **metrics}


def run(spec):
    import numpy as np
    sys.path.insert(0, spec['baseline_root'])
    from formal_evaluation.common.aggregation import aggregate_windows
    raw = Path(spec['selection']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == spec['selection_sha256']
    selection = {(r['dataset'],r['window_id']):r for r in map(json.loads,raw.splitlines())}
    target = {(r['dataset'],r['window_id']):r for r in rows(spec['contact_gt_index'])}
    jobs = []
    for job in spec['jobs']:
        ds=job['dataset']
        predictions=rows(job['prediction_index'])
        records=[json.loads(Path(remap(r['window_input'],spec['path_mappings'])).read_text()) for r in rows(job['input_index'])]
        records={r['window_id']:r for r in records}
        wanted={w for d,w in selection if d==ds}
        assert len(predictions)==len(wanted)==job['expected_windows']
        assert {r['window_id'] for r in predictions}==set(records)==wanted
        for p in predictions:
            wid=p['window_id'];record=records[wid];cg=target[(ds,wid)];sel=selection[(ds,wid)]
            assert record['frame_ids']==cg['frame_ids']==sel['frame_ids'] and len(record['geometry_paths'])==60
            pd=remap(p['prediction_dir'],spec['path_mappings'])
            for path in [pd+'/metadata.json',pd+'/predictions.npz',remap(cg['array_path'],spec['path_mappings']),*[remap(x,spec['path_mappings']) for x in record['geometry_paths']]]:
                if not Path(path).is_file():raise FileNotFoundError(path)
            jobs.append((ds,wid,pd,record,cg,sel))
    assert len(jobs)==sum(j['expected_windows'] for j in spec['jobs'])
    root=Path(spec['output_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'input_audit.json').write_text(json.dumps({'windows':len(jobs),'status':'complete','selection_sha256':spec['selection_sha256']}))
    grouped={j['dataset']:[] for j in spec['jobs']}
    with ProcessPoolExecutor(max_workers=spec['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
        with (root/'window_metrics.jsonl').open('x') as out:
            for i,row in enumerate(pool.map(compute,jobs,chunksize=1),1):
                grouped[row['dataset']].append(row)
                out.write(json.dumps(row)+'\n');out.flush()
                print(json.dumps({'completed':i,'total':len(jobs),'dataset':row['dataset'],'time':time.time()}),flush=True)
    report={'status':'complete','method':spec['method'],'windows':len(jobs),'selection_sha256':spec['selection_sha256'],'distance_reference':'GT object union GT opposite hand for both query sets','vertex_pred_contact':'derived geometry; no native vertex contact head','datasets':{d:aggregate_windows(v,method=spec['method']) for d,v in grouped.items()}}
    for name in ('report.json','summary.json'):(root/name).write_text(json.dumps(report,indent=2))
    (root/'COMPLETE').write_text('complete\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();run(json.loads(args.spec.read_text()))
