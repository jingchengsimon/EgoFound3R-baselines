"""Aggregate contact classification metrics from registered per-window outputs."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.contact.metrics import compute_contact_metrics

def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def run(spec, root):
    predictions=rows(spec['prediction_index']); gt_index=rows(spec['gt_index']); distance={r['window_id']:r for r in rows(spec['distance_index'])}
    aliases={r['cache_id']:r['window_id'] for r in gt_index}; gt={r['window_id']:r for r in gt_index}
    masks=json.loads(Path(spec['mask']['path']).read_text()); masks=dict(zip(masks['window_ids'],masks['excluded'],strict=True))
    if len(predictions)!=spec['expected_windows'] or set(masks)!=set(gt): raise ValueError('coverage mismatch')
    root.mkdir(parents=True,exist_ok=False); report={'status':'complete','method':spec['method'],'windows':len(predictions),'schemes':{'all8_p95':{},'unfiltered':{}}}
    for dataset in spec['datasets']:
        vals={k:[] for k in report['schemes']}
        for row in [x for x in predictions if x['dataset']==dataset]:
            key=aliases.get(row['window_id'],row['window_id'])
            with np.load(row['array_path'],allow_pickle=False) as pred, np.load(distance[key]['array_path'],allow_pickle=False) as target:
                for scheme in vals:
                    keep=~np.asarray(masks[key],bool) if scheme=='all8_p95' else np.ones(60,bool)
                    out={'window_id':key}
                    for prefix in ('joint','marker','vertex'):
                        p=pred[prefix+'_contact_probability']; t=target[prefix+'_contact_target']; m=target[prefix+'_contact_mask'] & keep[:,None,None]
                        out.update({prefix+'_contact_'+k:v for k,v in compute_contact_metrics(p,t,m).items()})
                    vals[scheme].append(out)
        for scheme,v in vals.items(): report['schemes'][scheme][dataset]=aggregate_windows(v,method=spec['method'])
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=True));(root/'summary.json').write_text(json.dumps({'status':'complete','windows':len(predictions),'report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest()}));(root/'COMPLETE').write_text('complete\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();run(json.loads(a.spec.read_text()),a.output_root)
