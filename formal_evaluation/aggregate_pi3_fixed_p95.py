"""Aggregate completed Pi3 depth residuals using an immutable shared selection."""
import argparse,hashlib,json,math
from pathlib import Path

def mean(values):
    v=[float(x) for x in values if math.isfinite(float(x))]
    return sum(v)/len(v) if v else float('nan')

def run(source,selection,expected_sha,root):
    assert (source.parent/'COMPLETE').is_file()
    assert hashlib.sha256(selection.read_bytes()).hexdigest()==expected_sha
    try:manifest=json.loads(selection.read_text())
    except json.JSONDecodeError:manifest=None
    if manifest and 'datasets' in manifest:
        selection_rows=[]
        for dataset,source_spec in manifest['datasets'].items():
            raw=Path(source_spec['source_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==source_spec['sha256']
            gt_raw=Path(source_spec['gt_index']).read_bytes();assert hashlib.sha256(gt_raw).hexdigest()==source_spec['gt_index_sha256']
            gt={r['window_id']:r for r in map(json.loads,gt_raw.splitlines())};mask=json.loads(raw)
            assert set(mask['window_ids'])==set(gt)
            for key,excluded in zip(mask['window_ids'],mask['excluded'],strict=True):
                frames=gt[key]['frame_ids'];assert len(frames)==len(excluded)==60
                keep=[not bool(v) for v in excluded]
                selection_rows.append({'dataset':dataset,'window_id':key,'frame_ids':frames,'keep':keep,'keep_pair':[a and b for a,b in zip(keep,keep[1:])],'keep_triplet':[a and b and c for a,b,c in zip(keep,keep[1:],keep[2:])]})
        rebuilt=''.join(json.dumps(r)+'\n' for r in selection_rows)
        assert hashlib.sha256(rebuilt.encode()).hexdigest()==manifest['selection_sha256']
        expected_sha=manifest['selection_sha256']
    else:selection_rows=[json.loads(line) for line in selection.read_text().splitlines()]
    masks={}
    for row in selection_rows:
        key=(row['dataset'],row['window_id'])
        if key in masks:raise ValueError('duplicate mask identity')
        masks[key]=row
    assert len(masks)==2378
    root.mkdir(parents=True,exist_ok=False);datasets={};seen=set()
    with (root/'windows.jsonl').open('x') as out:
        for line in source.read_text().splitlines():
            row=json.loads(line);key=(row['dataset'],row['window_id']);mask=masks[key]
            assert key not in seen;seen.add(key)
            assert [f['frame_id'] for f in row['frames']]==mask['frame_ids']
            assert row['depth_window_scale']==1.0
            selected=[f for f,keep in zip(row['frames'],mask['keep']) if keep]
            metrics={k:mean([f[k] for f in selected]) for k in row['metrics']}
            record={'dataset':key[0],'window_id':key[1],'retained_frames':len(selected),'metrics':metrics}
            out.write(json.dumps(record)+'\n');datasets.setdefault(key[0],[]).append(record)
    counts={'h2o':283,'taco':400,'hoi4d':461};assert {d:len(v) for d,v in datasets.items()}==counts
    report={'status':'complete','windows':1144,'selection_sha256':expected_sha,'source_windows':str(source),'depth_scale_alignment':'none','datasets':{}}
    for d,rows in datasets.items():
        keys=rows[0]['metrics'];report['datasets'][d]={'windows':len(rows),'retained_frames':sum(r['retained_frames'] for r in rows),'metrics':{k:mean([r['metrics'][k] for r in rows]) for k in keys},'undefined_windows':{k:sum(not math.isfinite(r['metrics'][k]) for r in rows) for k in keys}}
    (root/'report.json').write_text(json.dumps(report,indent=2));(root/'summary.json').write_text(json.dumps(report));(root/'COMPLETE').write_text('complete\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();run(a.source,a.selection,a.sha256,a.output_root)
