import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict

def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--catalog',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 centers=read(a.manifest); axes={}
 spec=json.loads(a.spec.read_text())
 cat_doc=json.loads(a.catalog.read_text())
 if 'catalogs' not in cat_doc:
  # Older launch records pass source_spec as --catalog; resolve its
  # registered result3 catalog instead of treating the spec as a catalog.
  rp=spec.get('result3_catalog')
  if isinstance(rp, str) and Path(rp).is_file(): cat_doc=json.loads(Path(rp).read_text())
  elif isinstance(rp, dict): cat_doc={'catalogs':list(rp.values())}
 if isinstance(spec.get('result3_catalog'), str) and Path(spec['result3_catalog']).is_file():
  full=json.loads(Path(spec['result3_catalog']).read_text())
  full_by={x.get('dataset'):x for x in full.get('catalogs',full if isinstance(full,list) else [])}
  for x in cat_doc.get('catalogs',[]):
   y=full_by.get(x.get('dataset'),{})
   for k in ('prediction_index','predictions_index'):
    if k in y and k not in x: x[k]=y[k]
 cat={x['dataset']:x for x in cat_doc.get('catalogs',[])}
 inputs=defaultdict(list)
 for ds,paths in spec['input_indices'].items():
  for p in paths:
   for e in read(p):
    rec=json.loads(Path(e['window_input']).read_text()); seq=rec.get('sequence_id') or rec['window_id'].rsplit(':',1)[0];inputs[(ds,seq)].append(rec)
 indexes={}
 for ds,c in cat.items():
  gt={x['window_id']:x for x in read(c['gt_index'])}; pi={x['window_id']:x for x in read(c.get('prediction_index', c.get('predictions_index')))};indexes[ds]=(gt,pi)
 results=[]
 for r in centers:
  key=(r['dataset'],r['sequence_id']); seqframes=sorted({str(fid) for rec in inputs[key] for fid in rec['frame_ids']}, key=lambda x:int(x)); ax={'frame_ids':[{'fid':x} for x in seqframes]} if seqframes else None; status='ok'; reasons=[]
  ordered=[str(x['fid']) for x in ax['frame_ids']] if ax else []; center=str(r['frame_id'])
  if not ax or center not in ordered: status='unresolved';reasons.append('center_missing_from_axis');required=[]
  else:
   c=ordered.index(center);required=ordered[max(0,c-150):c+150]
   if len(required)<300: status='boundary_truncated';reasons.append(f'axis_length_{len(required)}')
   recs={str(fid):v for rec in inputs[key] for fid in v['frame_ids']}
   gt,pi=indexes[r['dataset']]
   if not pi:
    status='unresolved'; reasons.append('prediction_index_unregistered'); results.append({**r,'video_start_frame':required[0] if required else None,'video_end_frame':required[-1] if required else None,'video_frame_count':len(required),'coverage_status':status,'coverage_reasons':reasons}); continue
   for fid in required:
    rec=recs.get(fid)
    if not rec or not Path(rec['rgb_paths'][rec['frame_ids'].index(fid)]).is_file(): status='unresolved';reasons.append('rgb_missing');break
    windows=[w for w in gt.values() if fid in [str(x) for x in w.get('frame_ids',[])]]
    preds=[w for w in pi.values() if fid in [str(x) for x in w.get('frame_ids',[])]]
    if not windows: status='unresolved';reasons.append('gt_index_missing');break
    if not preds: status='unresolved';reasons.append('prediction_index_missing');break
    if not Path(windows[0].get('path',windows[0].get('gt_path',''))).is_file(): status='unresolved';reasons.append('gt_file_missing');break
    if not Path(preds[0].get('prediction_dir','')) .joinpath('predictions.npz').is_file(): status='unresolved';reasons.append('prediction_file_missing');break
  results.append({**r,'video_start_frame':required[0] if required else None,'video_end_frame':required[-1] if required else None,'video_frame_count':len(required),'coverage_status':status,'coverage_reasons':sorted(set(reasons))})
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'coverage_results.jsonl').write_text('\n'.join(json.dumps(x) for x in results)+'\n')
 summary={'status':'complete','total':len(results),'renderable':sum(x['coverage_status']=='ok' for x in results),'boundary_truncated':sum(x['coverage_status']=='boundary_truncated' for x in results),'unresolved':sum(x['coverage_status']=='unresolved' for x in results),'by_dataset':{}}
 for x in results: summary['by_dataset'].setdefault(x['dataset'],{'total':0,'renderable':0,'boundary_truncated':0,'unresolved':0}); summary['by_dataset'][x['dataset']]['total']+=1;summary['by_dataset'][x['dataset']][{'ok':'renderable','boundary_truncated':'boundary_truncated','unresolved':'unresolved'}[x['coverage_status']]]+=1
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
