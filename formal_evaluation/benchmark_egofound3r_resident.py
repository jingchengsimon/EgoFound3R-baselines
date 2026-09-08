"""Paired resident B=1/B=16 speed, output consistency and memory stability gate."""
import argparse,json,sys
from pathlib import Path
import numpy as np
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from formal_evaluation.run_egofound3r_resident import Resident,prepare_inputs


def compare(a,b):
    issues=[]
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        for key in x.files:
            u,v=x[key],y[key]
            if u.dtype.kind=='b':
                if not np.array_equal(u,v):issues.append(dict(key=key,mask_mismatches=int(np.count_nonzero(u!=v))))
                continue
            atol=0.001 if ('joints' in key or 'markers' in key or key=='camera_c2w') else 0.01
            if not np.allclose(u,v,rtol=0.01,atol=atol,equal_nan=True):
                good=np.isfinite(u)&np.isfinite(v)
                issues.append(dict(key=key,max_abs=float(np.max(np.abs(u[good]-v[good]))) if good.any() else None,atol=atol))
    return dict(passed=not issues,issues=issues)


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.spec.read_text());root=a.output_root;root.mkdir(parents=True,exist_ok=False);records=[]
    for ds,data in spec['datasets'].items():
        _,_,items=prepare_inputs(ds,data,root/'inputs'/ds);records.extend(items[:3 if len(records)<12 else 2])
    records=records[:16];assert len(records)==16 and len({r['dataset'] for r in records})==6
    runtime=Resident(spec);stats=[];batches=[];comparisons=[]
    runtime.run(records[:1],root/'warmup',1)
    for r in records:stats.append(runtime.run([r],root/'b1',1))
    (root/'b1_summary.json').write_text(json.dumps(stats));print(json.dumps({'stage':'resident_b1_complete','windows':16,'wall_seconds':sum(x['wall_seconds'] for x in stats)}),flush=True)
    try:
        for repeat in range(3):
            order=records if repeat!=1 else list(reversed(records));out=root/('b16_'+str(repeat))
            s=runtime.run(order,out,16);batches.append(s)
            for r in records:
                rel=Path('egofound3r/formal')/r['cache_id']/'predictions.npz'
                comparisons.append(dict(repeat=repeat,window_id=r['window_id'],dataset=r['dataset'],**compare(root/'b1'/rel,out/rel)))
            print(json.dumps({'stage':'b16_batch_complete','repeat':repeat,**s}),flush=True)
    except Exception as error:
        report={'status':'failed','error':repr(error),'b1':stats,'b16':batches,'comparisons':comparisons}
        (root/'report.json').write_text(json.dumps(report));print(json.dumps({'status':'failed','error':repr(error)}),flush=True);raise
    speedup=float(sum(x['wall_seconds'] for x in stats)/np.median([x['wall_seconds'] for x in batches]))
    growth=max(x['allocated_after_bytes'] for x in batches)-min(x['allocated_after_bytes'] for x in batches)
    passed=all(x['passed'] for x in comparisons) and growth<512*1024**2 and speedup>1
    report=dict(status='passed' if passed else 'failed',b1=stats,b16=batches,comparisons=comparisons,speedup_wall=speedup,memory_growth_bytes=growth,
                gate={'boolean_masks':'exact','float_rtol':0.01,'geometry_atol_m':0.001,'other_atol':0.01,'max_live_memory_growth_bytes':512*1024**2,'speedup_must_exceed':1})
    (root/'report.json').write_text(json.dumps(report));(root/'COMPLETE').write_text(report['status'])
    print(json.dumps({k:v for k,v in report.items() if k not in ['b1','b16','comparisons']}),flush=True)
if __name__=='__main__':main()
