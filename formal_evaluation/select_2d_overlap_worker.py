"""CPU-only cached silhouette ranking. No model, RGB modification, or remote writes."""
import base64
import collections
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, binary_dilation


def mask(vertices, K, faces, size):
    if not np.isfinite(vertices).all() or not np.isfinite(K).all():
        raise ValueError('NONFINITE_GEOMETRY_OR_CAMERA')
    # A near-plane crossing is excluded explicitly, never silently drop triangles.
    if np.any(vertices[:, 2] <= 1e-6):
        raise ValueError('NEAR_PLANE_GEOMETRY')
    homogeneous = vertices @ K.T
    uv = homogeneous[:, :2] / homogeneous[:, 2:]
    image = Image.new('1', size)
    draw = ImageDraw.Draw(image)
    for f in faces:
        draw.polygon([tuple(p) for p in uv[f]], fill=1)
    return np.asarray(image, dtype=bool)


def measures(a, b):
    union = (a | b).sum()
    if not a.any() or not b.any():
        return None
    ea, eb = a ^ binary_erosion(a), b ^ binary_erosion(b)
    radius = max(1, int(round(.005 * math.hypot(*a.shape))))
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = xx*xx + yy*yy <= radius*radius
    precision = (ea & binary_dilation(eb, structure=disk)).sum() / ea.sum()
    recall = (eb & binary_dilation(ea, structure=disk)).sum() / eb.sum()
    ca = np.array(np.nonzero(a)).mean(axis=1)
    cb = np.array(np.nonzero(b)).mean(axis=1)
    return {'iou':float((a & b).sum()/union),
            'boundary_f':float(2*precision*recall/(precision+recall)) if precision+recall else 0.,
            'center_error':float(np.linalg.norm(ca-cb)/math.sqrt(a.size)),
            'area_error':float(abs(math.log(a.sum()/b.sum())))}


def process(task):
    ds,g,pr,record_path,excluded,gp = task
    counters=collections.Counter(); rows=[]
    try:
        pp=Path(pr['prediction_dir'])/'predictions.npz'
        md=json.loads((pp.parent/'metadata.json').read_text())
        assert md['dataset']==ds and md['frame_ids']==g['frame_ids'],'PREDICTION_IDENTITY'
        assert md['global_stride']==5,'STRIDE_IDENTITY'
        record=json.loads(Path(record_path).read_text())
        assert record['frame_ids']==g['frame_ids'],'RGB_FRAME_IDENTITY'
        with np.load(pp,allow_pickle=False) as z:
            keys=['intrinsics','intrinsics_valid','camera_valid','hand_valid','hand_markers_camera']
            ego={k:z[k] for k in keys}
            if 'hand_vertices_camera' in z.files:ego['hand_vertices_camera']=z['hand_vertices_camera']
            fields=set(z.files)
        with np.load(gp,allow_pickle=False) as z:
            gt={k:z[k] for k in ['intrinsics','intrinsics_valid','hand_valid','hand_vertices_camera']}
        assert gt['hand_vertices_camera'].shape==(60,2,778,3),'GT_VERTEX_SHAPE'
        for t,fid in enumerate(g['frame_ids']):
            counters['all_frames']+=1
            if excluded[t]:counters['p95_excluded']+=1;continue
            counters['p95_retained']+=1
            if not ego['intrinsics_valid'][t] or not gt['intrinsics_valid'][t]:
                counters['invalid_intrinsics']+=1;continue
            counters['valid_intrinsics_frames']+=1
            hands=ego['hand_valid'][t] & gt['hand_valid'][t]
            if not hands.any():counters['no_common_valid_hand']+=1;continue
            rgb=Path(record['rgb_paths'][t])
            with Image.open(rgb) as im:w,h=im.size
            sc=512/max(w,h);size=(round(w*sc),round(h*sc))
            sx,sy=size[0]/w,size[1]/h
            display=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]])
            side=min(w,h);s=side/512
            crop_inverse=np.array([[s,0,(w-side)//2+(s-1)/2],[0,s,(h-side)//2+(s-1)/2],[0,0,1]])
            kp=display@crop_inverse@ego['intrinsics'][t];kg=display@gt['intrinsics'][t]
            assert np.allclose(gt['intrinsics'][t],record['intrinsics'][t]),'GT_RGB_CALIBRATION'
            if 'hand_vertices_camera' in ego:
                ev=ego['hand_vertices_camera'][t];geometry_source='cached_778'
            else:
                m=ego['hand_markers_camera'][t];c=m.mean(-2,keepdims=True)
                ev=c+((m-c)[:,NEI,:]*WEIGHTS[...,None]).sum(-2);ev[:,IDS]=m
                geometry_source='official_195_to_778'
            values=[];per_hand={};error=None
            for hand in np.flatnonzero(hands):
                try:
                    a=mask(ev[hand],kp,FACES,size);b=mask(gt['hand_vertices_camera'][t,hand],kg,FACES,size)
                    v=measures(a,b)
                    if v is None:raise ValueError('EMPTY_PROJECTED_HAND')
                    values.append(v);per_hand[['left','right'][hand]]=v
                except ValueError as e:error=str(e);break
            if error:counters[error]+=1;continue
            means={k:float(np.mean([v[k] for v in values])) for k in values[0]}
            rows.append(dict(dataset=ds,sequence_id=g['sequence_id'],window_id=g['window_id'],
                frame_id=fid,frame_index=t,**means,per_hand=per_hand,valid_hand_count=len(values),
                ego_hand_valid=ego['hand_valid'][t].tolist(),gt_hand_valid=gt['hand_valid'][t].tolist(),
                missing_gt_hand_count=int((gt['hand_valid'][t]&~ego['hand_valid'][t]).sum()),
                rgb_path=str(rgb),rgb_size=[w,h],prediction_path=str(pp),gt_path=str(gp),
                input_record=record_path,geometry_source=geometry_source,
                contact_available='marker_contact_probability' in fields,
                visibility_available='marker_visibility' in fields,
                distance_fields=sorted(k for k in fields if 'distance' in k)))
        return {'dataset':ds,'window_id':g['window_id'],'counts':dict(counters),'rows':rows,'ok':True}
    except Exception as e:
        return {'dataset':ds,'window_id':g['window_id'],'ok':False,'error':type(e).__name__+':'+str(e),'counts':dict(counters),'rows':[]}


def run(spec):
    global NEI,WEIGHTS,IDS,FACES
    with np.load(io.BytesIO(base64.b64decode(spec['asset'])),allow_pickle=False) as z:
        NEI=z['neighbor_indices'];WEIGHTS=z['geometry_weights'];IDS=z['source_vertex_ids'];FACES=z['faces']
    catalogs={x['dataset']:x for x in json.loads(Path(spec['result3_catalog']).read_text())['catalogs']}
    all_rows=[];errors=[];counts={};window_counts={};sources=[]
    with multiprocessing.get_context('fork').Pool(8) as pool:
        for c in spec['catalogs']:
            ds=c['dataset'];idx=Path(c['gt_index']);raw=idx.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==c['gt_index_sha256'],'GT_INDEX_HASH'
            gt=[json.loads(x) for x in raw.splitlines() if x.strip()]
            pi=Path(catalogs[ds]['prediction_index']);pr={x['window_id']:x for x in (json.loads(s) for s in pi.read_text().splitlines() if s.strip())}
            pm=Path(spec['variants_root'])/ds/'all8_p95_mask.json';mask_data=json.loads(pm.read_text())
            ex=dict(zip(mask_data['window_ids'],mask_data['excluded']))
            assert len(gt)==c['expected_windows'] and set(pr)==set(ex)=={g['window_id'] for g in gt},'WINDOW_COVERAGE'
            records={}
            for path in spec['input_indices'][ds]:
                for row in (json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()):
                    record=row['window_input'];key=Path(record).parent.name
                    assert key not in records or records[key]==record,'DUPLICATE_INPUT'
                    records[key]=record
            tasks=[(ds,g,pr[g['window_id']],records[g['cache_id']],ex[g['window_id']],str(idx.parent/g['array_path'].split('/gt_cache/',1)[1])) for g in gt]
            count=collections.Counter();window_counts[ds]=len(tasks)
            for out in pool.imap_unordered(process,tasks,chunksize=1):
                count.update(out['counts']);all_rows.extend(out['rows'])
                if not out['ok']:errors.append({k:v for k,v in out.items() if k not in ['rows','counts']})
            counts[ds]=dict(count)
            sources.append({'dataset':ds,'prediction_index':str(pi),'prediction_index_sha256':hashlib.sha256(pi.read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256(pm.read_bytes()).hexdigest(),'gt_index_sha256':c['gt_index_sha256']})
    return {'ok':not errors,'rows':all_rows,'errors':errors,'counts':counts,'window_counts':window_counts,'sources':sources,'remote_pid':os.getpid()}


if __name__=='__main__':
    print(json.dumps(run(json.loads(sys.argv[1])),allow_nan=False))
