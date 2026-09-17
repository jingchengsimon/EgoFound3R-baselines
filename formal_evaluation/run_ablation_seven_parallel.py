"""CPU continuation; preserve paused outputs and merge exact windows."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def materialize_resume(spec, destination):
    from formal_evaluation.run_ablation_seven_tables import reusable_rows
    raw=Path(spec['selection']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==spec['selection_sha256']
    selection={(r['dataset'],r['window_id']):r for r in map(json.loads,raw.splitlines())}
    destination=Path(destination);destination.mkdir()
    manifest={}
    for job in spec['jobs']:
        ds=job['dataset'];gt={r['window_id']:r for r in map(json.loads,Path(job['gt_index']).read_text().splitlines())}
        assert len(gt)==job['expected_windows']
        folder=destination/ds;folder.mkdir();merged={};sources={}
        for origin in spec['resume_roots']:
            for wid,row in reusable_rows(origin,ds,gt).items():
                if wid in merged:raise ValueError('duplicate resume window: '+ds+' '+wid)
                assert row['frame_ids']==selection[(ds,wid)]['frame_ids']
                merged[wid]=row;sources[wid]=origin
                name=gt[wid]['cache_id']+'_scene.npz'
                (folder/name).symlink_to((Path(origin)/ds/name).resolve())
        (folder/'window_metrics.jsonl').write_text(''.join(json.dumps(merged[w])+'\n' for w in sorted(merged)))
        shards=[[w for i,w in enumerate(sorted(gt)) if i%spec['workers']==j and w not in merged] for j in range(spec['workers'])]
        flat=sum(shards,[])
        assert len(flat)==len(set(flat)) and set(flat).isdisjoint(merged) and set(flat)|set(merged)==set(gt)
        manifest[ds]={'reused':len(merged),'remaining':len(flat),'sources':sources,'shards':shards}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def run(spec):
    workers=spec.get('workers')
    if workers not in (8,16):
        raise ValueError('this continuation requires eight or sixteen CPU workers')
    root=Path(spec['output_root']);root.mkdir(parents=True,exist_ok=False)
    if spec.get('resume_roots'):
        spec=dict(spec,resume_root=str(root/'resume_snapshot'))
        manifest=materialize_resume(spec,spec['resume_root'])
        print(json.dumps({'stage':'resume_snapshot','datasets':{k:{n:v[n] for n in ('reused','remaining')} for k,v in manifest.items()}}),flush=True)
    old=Path(spec['resume_root'])
    if old.exists() and (old/'report.json').exists():
        previous=json.loads((old/'report.json').read_text())
        assert previous['method']==spec['method'] and previous['selection_sha256']==spec['selection_sha256']
    children=[]
    for i in range(workers):
        output=root/'shards'/str(i);child_spec=dict(spec,shard_id=i,shard_count=workers,output_root=str(output))
        path=root/('shard_'+str(i)+'.json');path.write_text(json.dumps(child_spec))
        log=(root/('shard_'+str(i)+'.log')).open('x')
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('run_ablation_seven_tables.py')),'--spec',str(path)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        log.close();children.append(child)
    (root/'workers.json').write_text(json.dumps([{'shard':i,'pid':c.pid} for i,c in enumerate(children)]))
    print(json.dumps({'stage':'launched','workers':workers,'pids':[c.pid for c in children]}),flush=True)
    codes=[child.wait() for child in children]
    (root/'worker_exits.json').write_text(json.dumps(codes))
    if any(codes):raise RuntimeError('CPU shard failure; outputs preserved: '+str(codes))
    sys.path.insert(0,str(Path(__file__).parent))
    from run_ablation_seven_tables import load,reusable_rows
    sys.path.insert(0,spec['baseline_root'])
    from formal_evaluation.common.aggregation import aggregate_windows
    common=load('parallel_common',spec['same_mask_script'])
    raw=Path(spec['selection']).read_bytes();assert hashlib.sha256(raw).hexdigest()==spec['selection_sha256']
    selection={(r['dataset'],r['window_id']):r for r in map(json.loads,raw.splitlines())}
    upstream=Path(spec['upstream_root']);assert (upstream/'COMPLETE').exists()
    report=json.loads((upstream/'report.json').read_text());assert report['windows']==sum(j['expected_windows'] for j in spec['jobs']) and report['selection_sha256']==spec['selection_sha256']
    report.update(status='running',tables=7,workers=workers,resume_root=str(old),windows=0,
                  geometry_backend_for_new_windows=spec.get('geometry_backend','legacy'),
                  completed_metrics_preserved=True)
    provenance=[]
    for job in spec['jobs']:
        ds=job['dataset'];gt={r['window_id']:r for r in common.read_jsonl(Path(job['gt_index']))}
        assert len(gt)==job['expected_windows']
        merged={};folder=root/ds;folder.mkdir()
        for origin in [old]+[root/'shards'/str(i) for i in range(workers)]:
            rows=reusable_rows(origin,ds,gt)
            for wid,row in rows.items():
                assert wid not in merged and row['frame_ids']==selection[(ds,wid)]['frame_ids']
                merged[wid]=row
                provenance.append({'dataset':ds,'window_id':wid,'source_root':str(origin)})
        assert set(merged)==set(gt), (ds,len(merged),len(gt))
        with (folder/'window_metrics.jsonl').open('x') as out:
            for wid in sorted(merged):out.write(json.dumps(merged[wid],allow_nan=False)+'\n')
        report['datasets'][ds].update(aggregate_windows(list(merged.values()),method=spec['method']))
        report['windows']+=len(merged);(folder/'COMPLETE').write_text('complete\n')
    assert report['windows']==sum(j['expected_windows'] for j in spec['jobs'])
    report['status']='complete'
    (root/'report.json').write_text(json.dumps(common.json_safe(report),allow_nan=False))
    (root/'source_windows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in provenance))
    (root/'summary.json').write_text(json.dumps({'status':'complete','windows':report['windows'],'tables':7,'workers':workers,'method':spec['method']}))
    (root/'COMPLETE').write_text('complete\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();run(json.loads(a.spec.read_text()))
