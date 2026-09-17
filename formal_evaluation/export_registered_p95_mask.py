"""Export a registered v7 P95 mask and its exact frame identities, read-only remotely."""
import hashlib,json,shlex
from pathlib import Path

def export(run,registry,destination,remote):
    identity=run['identity']
    if identity.get('protocol')!='same-mask-frozen-v1':raise ValueError('not a registered same-mask source')
    sources=[]
    for key in identity['catalog_run_ids'].split(','):
        cat=registry['runs'][key]['artifact_catalog'];d=cat['dataset']
        sources.append({'dataset':d,'expected':cat['expected_windows'],'gt_index':cat['gt_index'],'gt_sha256':cat['gt_index_sha256'],'mask':identity['variants_root']+'/'+d+'/all8_p95_mask.json'})
    script='''import json,hashlib,sys
from pathlib import Path
out=[]
for s in json.loads(sys.argv[1]):
 raw=Path(s['mask']).read_bytes(); gt=Path(s['gt_index']).read_bytes()
 assert len(raw)<2000000 and hashlib.sha256(gt).hexdigest()==s['gt_sha256']
 out.append(dict(s,raw=raw.decode(),mask_sha256=hashlib.sha256(raw).hexdigest(),gt=[json.loads(l) for l in gt.splitlines() if l]))
print(json.dumps(out))'''
    reply=remote(run,5000,'python3 -c '+shlex.quote(script)+' '+shlex.quote(json.dumps(sources)),timeout=120)
    if reply.returncode:raise RuntimeError(reply.stderr)
    items=json.loads(reply.stdout);root=Path(destination);root.mkdir(parents=True,exist_ok=False)
    rows=[];datasets={}
    for item in items:
        mask=json.loads(item['raw']);gt={r['window_id']:r for r in item['gt']}
        assert len(gt)==item['expected'] and set(mask['window_ids'])==set(gt)
        assert len(mask['excluded'])==len(gt)
        path=root/(item['dataset']+'_all8_p95_mask.json');path.write_text(item['raw'])
        kept=0
        for key,excluded in zip(mask['window_ids'],mask['excluded']):
            frames=gt[key].get('frame_ids')
            assert frames is not None and len(frames)==len(excluded)==60
            keep=[not bool(v) for v in excluded];kept+=sum(keep)
            rows.append({'dataset':item['dataset'],'window_id':key,'frame_ids':frames,'keep':keep,'keep_pair':[a and b for a,b in zip(keep,keep[1:])],'keep_triplet':[a and b and c for a,b,c in zip(keep,keep[1:],keep[2:])]})
        datasets[item['dataset']]={'windows':len(gt),'retained_frames':kept,'total_frames':60*len(gt),'source_path':item['mask'],'sha256':item['mask_sha256'],'local_path':str(path.resolve()),'gt_index':item['gt_index'],'gt_index_sha256':item['gt_sha256']}
    assert len(rows)==2378
    selection=root/'selection.jsonl';selection.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    receipt={'source_run_id':run['run_id'],'scheme':'all8_p95','excluded_true_means_drop':True,'windows':2378,'datasets':datasets,'selection_sha256':hashlib.sha256(selection.read_bytes()).hexdigest(),'selection_path':str(selection.resolve()),'no_mask_regeneration':True}
    (root/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
