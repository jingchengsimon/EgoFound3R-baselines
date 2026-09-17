"""Read-only projection preflight using one exact registered P95 source."""
import json
import shlex
from pathlib import Path

REMOTE = r'''
import json,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
spec=json.loads(sys.argv[1]); results=[]
catalog={x['dataset']:x for x in json.loads(Path(spec['result3_catalog']).read_text())['catalogs']}
for item in spec['catalogs']:
    ds=item['dataset']; r={'dataset':ds}
    try:
        idx=Path(item['gt_index']);raw=idx.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==item['gt_index_sha256'],'GT_INDEX_HASH'
        gt=[json.loads(x) for x in raw.splitlines() if x.strip()]
        pi=Path(catalog[ds]['prediction_index']); pred=[json.loads(x) for x in pi.read_text().splitlines() if x.strip()]
        r.update(gt_windows=len(gt),prediction_windows=len(pred),prediction_index=str(pi))
        g=gt[0];pr=next(p for p in pred if p['window_id']==g['window_id'])
        gp=idx.parent/g['array_path'].split('/gt_cache/',1)[1]
        pp=Path(pr['prediction_dir'])/'predictions.npz'
        r.update(gt_path=str(gp),prediction_path=str(pp))
        with np.load(gp,allow_pickle=False) as z:
            r['gt_fields']={k:list(z[k].shape) for k in z.files}
        with np.load(pp,allow_pickle=False) as z:
            keys=[k for k in z.files if any(s in k for s in ['hand_','marker_','intrinsics','camera','distance','contact','visibility'])]
            r['ego_fields']={k:list(z[k].shape) for k in keys}
            r['intrinsics_valid_count']=int(z['intrinsics_valid'].sum())
        md=json.loads((pp.parent/'metadata.json').read_text());r['metadata_keys']=list(md)
        inputs=[]
        for path in spec['input_indices'][ds]:
            inputs.extend(json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())
        match=[x for x in inputs if x.get('window_id')==g['window_id'] or x.get('cache_id')==g['cache_id'] or Path(x.get('window_input','')).parent.name==g['cache_id']]
        assert len(match)==1,('INPUT_MATCH',len(match))
        recpath=match[0]['window_input'];rec=json.loads(Path(recpath).read_text())
        assert rec['frame_ids']==g['frame_ids'],'FRAME_MISMATCH'
        rgb=Path(rec['rgb_paths'][0])
        with Image.open(rgb) as im:r['rgb_size']=list(im.size)
        r.update(input_record=recpath,rgb_path=str(rgb),record_keys=list(rec),ok=True)
    except Exception as e:r.update(ok=False,error=type(e).__name__+':'+str(e))
    results.append(r)
print(json.dumps({'ok':all(x['ok'] for x in results),'datasets':results,'numpy':np.__version__}))
'''

def audit(run, registry, destination, remote):
    identity=run['identity']
    if identity.get('protocol')!='same-mask-frozen-v1':
        raise ValueError('REGISTERED_P95_SOURCE_REQUIRED')
    root=Path(destination)
    if root.exists():raise FileExistsError(root)
    project=Path(__file__).resolve().parents[1]
    inputs=json.loads((project/'visualization/six_p95_rgb_overlay_20260910/fetch_spec.json').read_text())
    spec={'result3_catalog':identity['result3_catalog'],
          'catalogs':[{field:registry['runs'][key]['artifact_catalog'][field]
                       for field in ('dataset','gt_index','gt_index_sha256','expected_windows')}
                      for key in identity['catalog_run_ids'].split(',')],
          'input_indices':{ds:value['input_indices'] for ds,value in inputs.items()}}
    reply=remote(run,run.get('_current_instance',{}).get('port',5000),
                 shlex.join([identity['python'],'-c',REMOTE,json.dumps(spec)]),timeout=180)
    if reply.returncode:raise RuntimeError('OVERLAP_PREFLIGHT_FAILED:'+reply.stderr[-1500:])
    result=json.loads(reply.stdout)
    root.mkdir(parents=True)
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'source_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    return {'ok':result['ok'],'run_id':run['run_id'],'audit_path':str(root/'audit.json'),
            'datasets':[{k:v for k,v in x.items() if k in ['dataset','ok','error','gt_windows','prediction_windows','intrinsics_valid_count']} for x in result['datasets']]}


def select(run, registry, remote):
    import base64
    import collections
    import statistics
    import time
    identity=run['identity']
    if run.get('task_type')!='visualization' or identity.get('selection_protocol')!='independent-2d-overlap-v1':
        raise ValueError('REGISTERED_2D_SELECTION_REQUIRED')
    root=Path(run['output_root'])
    if (root/'summary.json').exists():raise FileExistsError('SELECTION_ALREADY_EXISTS')
    project=Path(__file__).resolve().parents[1]
    source=registry['runs'][identity['source_run_id']]
    spec=json.loads(Path(identity['preflight_spec']).read_text())
    spec['variants_root']=source['identity']['variants_root']
    asset=project/'visualization/six_same_mask_v7_p95_bestego_20260909/mano_195_to_778.npz'
    spec['asset']=base64.b64encode(asset.read_bytes()).decode()
    worker=project/'formal_evaluation/select_2d_overlap_worker.py'
    root.mkdir(parents=True,exist_ok=True)
    state_path=project/run['state_file']
    def state(status, complete=0, error=None):
        data={'jobs':{'six::selection':{'status':status,'count':complete,'expected_count':2378,'node':5000,'error':error}},'updated_at_epoch':time.time()}
        state_path.write_text(json.dumps(data,indent=2)+'\n')
    state('running')
    protocol={'protocol':identity['selection_protocol'],'p95_is_eligibility_filter':True,
        'uses_3d_selection':False,'uses_w_mpjpe':False,'vertex_count':778,
        'ego_camera':'native camera geometry; predicted K, inverse deterministic 512 center crop',
        'gt_camera':'native camera geometry; calibrated RGB K',
        'extrinsics':'no second transform for native camera geometry; not a world-pose evaluation',
        'no_fitted_alignment':True,'raster_long_edge':512,'boundary_tolerance_diagonal_fraction':.005,
        'near_plane_crossings':'exclude explicitly','no_object_occlusion_mask':True,
        'weights':{'iou':.60,'boundary_f':.25,'center_error':-.10,'area_error':-.05},
        'error_normalization':'per-dataset min-max on scored unique frames; constant column -> zero',
        'hand_aggregation':'equal mean over common native-valid hands; missing GT hands recorded separately',
        'minimum_mean_iou':.30,'duplicate_policy':'sequence + source frame; nearest window midpoint, then window ID',
        'output_classes':['geometry','contact_distance','contact','visibility'],'overlay_images_rendered':False}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    try:
        reply=remote(run,5000,shlex.join([source['identity']['python'],'-c',worker.read_text(),json.dumps(spec)]),timeout=1800)
        if reply.returncode:raise RuntimeError('OVERLAP_SELECTION_FAILED:'+reply.stderr[-1500:])
        result=json.loads(reply.stdout)
    except Exception as e:
        state('blocked',error=str(e));raise
    (root/'raw_measurements.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    summary={'ok':result['ok'],'status':'complete' if result['ok'] else 'blocked',
             'run_id':run['run_id'],'datasets':{},'errors':result['errors'],'protocol':protocol}
    selected=[];ranked=[];sequence_best=[]
    for ds,count in result['counts'].items():
        rows=sorted((x for x in result['rows'] if x['dataset']==ds),key=lambda x:(abs(x['frame_index']-29.5),x['window_id']))
        unique={}
        for row in rows:unique.setdefault((row['sequence_id'],row['frame_id']),row)
        rows=list(unique.values());normalization={}
        for k in ['center_error','area_error']:
            values=[r[k] for r in rows];lo=min(values,default=0.);hi=max(values,default=0.)
            normalization[k]=[lo,hi]
            for r in rows:r[k+'_normalized']=(r[k]-lo)/(hi-lo) if hi>lo else 0.
        for r in rows:
            r['score']=.60*r['iou']+.25*r['boundary_f']-.10*r['center_error_normalized']-.05*r['area_error_normalized']
            r['eligible']=r['iou']>=.30
        rows.sort(key=lambda r:(-r['score'],-r['valid_hand_count'],-r['boundary_f'],abs(r['frame_index']-29.5),r['sequence_id'],r['frame_id']))
        eligible=[r for r in rows if r['eligible']]
        for i,r in enumerate(rows):r['rank']=i+1
        seq={}
        for r in eligible:seq.setdefault(r['sequence_id'],r)
        ranked.extend(rows);sequence_best.extend(seq.values())
        if eligible:selected.append(eligible[0])
        summary['datasets'][ds]={'windows':result['window_counts'][ds],**count,
            'scored_frame_records':len([r for r in result['rows'] if r['dataset']==ds]),
            'unique_scored_frames':len(rows),'eligible_frames':len(eligible),'eligible_sequences':len(seq),
            'median_iou':statistics.median([r['iou'] for r in eligible]) if eligible else None,
            'median_score':statistics.median([r['score'] for r in eligible]) if eligible else None,
            'recommended_initial_images':int(bool(eligible)),
            'single_common_hand_frames':sum(r['valid_hand_count']==1 for r in eligible),
            'missing_gt_hand_frames':sum(r['missing_gt_hand_count']>0 for r in eligible),
            'contact_distance_available_frames':sum(bool(r['distance_fields']) for r in eligible),
            'normalization':normalization,'best':eligible[0] if eligible else None}
    for name,rows in [('candidates.jsonl',ranked),('selected_manifest.jsonl',selected),('sequence_best.jsonl',sequence_best)]:
        (root/name).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Six-dataset 2D overlap candidate selection','',
        'Cached Ego stride5 + registered Joint8 P95 eligibility; no inference or overlay rendering.','',
        '|Dataset|Windows|P95 frames|Scored unique|IoU >= .30|Sequences|Median IoU|Initial images|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds,s in summary['datasets'].items():
        med=f"{s['median_iou']:.4f}" if s['median_iou'] is not None else 'N/A'
        lines.append(f"|{ds}|{s['windows']}|{s.get('p95_retained',0)}|{s['unique_scored_frames']}|{s['eligible_frames']}|{s['eligible_sequences']}|{med}|{s['recommended_initial_images']}|")
    lines+=['','[Protocol](protocol.json) · [Summary](summary.json) · [Candidates](candidates.jsonl) · [Selected frames](selected_manifest.jsonl) · [Best per sequence](sequence_best.jsonl)','',
        'Contact distance is not present in these sampled canonical prediction fields. Geometry/contact/visibility eligibility does not certify all four final panels.',
        'Scores measure full projected hand silhouettes, not observed RGB segmentation or physical object visibility. Missing native-valid predictions are recorded, not used to hide GT.',
        'GT visibility is absent in the legacy cache. Ego 778 geometry is reconstructed from official 195-marker mapping where native 778 is absent.']
    (root/'index.md').write_text('\n'.join(lines)+'\n')
    state('done' if result['ok'] else 'blocked',sum(result['window_counts'].values())-len(result['errors']))
    return {'ok':result['ok'],'run_id':run['run_id'],'summary_path':str(root/'summary.json'),
            'datasets':{d:{k:v for k,v in x.items() if k in ['windows','eligible_frames','eligible_sequences','median_iou','recommended_initial_images']} for d,x in summary['datasets'].items()},'errors':result['errors']}
