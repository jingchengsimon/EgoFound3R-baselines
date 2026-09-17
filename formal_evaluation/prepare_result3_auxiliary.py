import argparse, hashlib, json
from pathlib import Path
import numpy as np
from formal_evaluation.datasets.six_dataset_gt_cache import load_window_cache
from formal_evaluation.recompute_same_mask_all_methods import read_jsonl, remap_gt_row

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    s=json.loads(a.spec.read_text()); a.output_root.mkdir(parents=True,exist_ok=False); index=[]; count=0
    for dataset,src in s['datasets'].items():
        rows=read_jsonl(Path(src['gt_index'])); assert len(rows)==src['expected_windows']
        for raw in rows:
            row=remap_gt_row(raw,Path('/unused'),Path(src['gt_index']).parent,direct_oss=True); _,target=load_window_cache(row)
            arrays={k:v for k,v in target.items() if 'visibility' in k}; assert arrays
            out=a.output_root/dataset;out.mkdir(exist_ok=True); path=out/(raw['cache_id']+'.npz');np.savez_compressed(path,**arrays)
            index.append({'dataset':dataset,'window_id':raw['window_id'],'frame_ids':raw['frame_ids'],'array_path':str(path)});count+=1
    (a.output_root/'index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index)); report={'status':'complete','windows':count,'datasets':list(s['datasets'])};(a.output_root/'report.json').write_text(json.dumps(report,indent=2));(a.output_root/'summary.json').write_text(json.dumps(report));(a.output_root/'COMPLETE').write_text('complete\n')
if __name__=='__main__': main()
