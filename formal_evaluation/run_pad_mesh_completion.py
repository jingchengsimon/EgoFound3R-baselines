"""PAD full-mesh completion over exact prepared windows, one serial lane per GPU."""
import argparse,concurrent.futures,json,os,subprocess,sys,shutil
from pathlib import Path

def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l]
def run(spec,root):
    from formal_evaluation.evaluate_six_dataset import _prediction_arrays
    gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(gpus)==len(set(gpus))==2
    assert shutil.disk_usage(root.parent).free>=100*1024**3
    runtime=spec['runtime']; records=[];seen=set()
    subprocess.run([sys.executable,'formal_evaluation/validate_runtime_registry.py','--registry',spec['runtime_registry'],'--method','pad_hand','--strict'],check=True)
    for dataset,source in spec['datasets'].items():
        gt={r['window_id']:r for r in rows(source['gt_index'])}
        assert len(gt)==source['expected_windows'];selected={}
        for row in rows(source['input_index']):
            p=Path(row['window_input']);record=json.loads(p.read_text());key=record['window_id']
            assert record['dataset']==dataset and key in gt and key not in selected
            assert record['frame_ids']==gt[key]['frame_ids'] and len(record['frame_ids'])==60
            mapping=json.loads((p.parent/'mapping.json').read_text())
            assert mapping['window_id']==key and mapping['frame_ids']==record['frame_ids']
            assert (p.parent/'input.mp4').stat().st_size>0
            selected[key]={**record,'window_input':str(p)}
        assert selected.keys()==gt.keys();records.extend(selected[k] for k in sorted(selected))
    assert len(records)==2095
    root.mkdir(parents=True,exist_ok=False)
    def lane(index):
        result=[];env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus[index],'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        for record in records[index::2]:
            destination=root/record['dataset']
            subprocess.run([runtime['python'],'-m','formal_evaluation.hand.adapters.run_pad_hand_baseline','--phase','formal','--window-input',record['window_input'],'--methods-config',spec['methods_config'],'--source-root',runtime['source_root'],'--wilor-python',runtime['wilor_python'],'--checkpoint',runtime['checkpoint'],'--output-root',str(destination)],env=env,check=True)
            path=destination/'pad_hand/formal'/record['cache_id'];meta,arrays=_prediction_arrays(path)
            assert meta['frame_ids']==record['frame_ids']
            assert arrays['hand_vertices_camera'].shape==(60,2,778,3)
            result.append({'dataset':record['dataset'],'window_id':record['window_id'],'method':'pad_hand','prediction_dir':str(path)})
            print(json.dumps({'lane':index,'gpu':gpus[index],'completed':len(result),'status':'mesh_written'}),flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outputs=list(pool.map(lane,range(2)))
    values=[r for output in outputs for r in output];assert len(values)==2095
    (root/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    report={'status':'complete','windows':2095,'granularities':['native_vertex778','marker195_derived_from_vertices'],'gpus':gpus,'filtering':'none; full windows preserved'}
    (root/'report.json').write_text(json.dumps(report));(root/'summary.json').write_text(json.dumps(report));(root/'COMPLETE').write_text('complete\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();run(json.loads(a.spec.read_text()),a.output_root)
