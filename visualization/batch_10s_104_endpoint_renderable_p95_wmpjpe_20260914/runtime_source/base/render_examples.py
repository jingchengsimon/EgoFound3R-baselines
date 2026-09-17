"""Four-row extension of the frozen shared-mask visualization base."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'h2o_result3_stride5_minmpjpe_5frames_20260909'
module_spec=importlib.util.spec_from_file_location('shared_hand_render',BASE/'render.py')
base=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(base)
CONFIG=json.loads((ROOT/'render_config.json').read_text())
VIEWS=[base.VIEW_PRESETS[x] for x in CONFIG['views']]
METHODS=['egofound3r_stride5','wilor','hawor','gt']
LABELS=['Ego stride5','WiLoR','HaWoR','GT']
NAMES={'h2o':'H2O','hot3d':'HOT3D','arctic':'ARCTIC','oakink_v2':'OakInk-v2','taco':'TACO','hoi4d':'HOI4D'}

def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def camera_points(a,field):
    key='hand_'+field+'_camera'
    if key in a:return a[key].astype(float)
    key='hand_'+field+'_world'
    assert key in a and 'camera_c2w' in a,(field,list(a))
    pose=a['camera_c2w'].astype(float)
    return np.einsum('tji,tsvj->tsvi',pose[:,:3,:3],a[key]-pose[:,None,None,:3,3])

def pooled_score(row):
    v=[(row['hand_'+s+'_mpjpe'],row['hand_'+s+'_valid_frame_count']) for s in ['left','right']]
    v=[(x,n) for x,n in v if x is not None and np.isfinite(x) and n]
    return sum(x*n for x,n in v)/sum(n for x,n in v)

def prepare(dataset):
    p=ROOT/dataset/'inputs'
    s=json.loads((p/'selection.json').read_text())
    arrays={m:load_arrays(p/(m+'.npz')) for m in METHODS};gt=arrays['gt']
    metadata={m:json.loads((p/(m+'_metadata.json')).read_text()) for m in METHODS[:-1]}
    for md in metadata.values():assert md['frame_ids']==s['frame_ids']
    ego_meta=metadata['egofound3r_stride5']
    assert ego_meta['global_stride']==5 and ego_meta['global_anchor_phase']==2
    excluded=np.array(s['excluded'],bool)
    gt_valid=gt['hand_valid'].astype(bool)&np.isfinite(gt['hand_joints_camera']).all(axis=(2,3))
    keep=np.flatnonzero(~excluded & gt_valid.any(1))
    assert len(keep)>=CONFIG['num_frames']
    selected=keep[np.rint(np.linspace(0,len(keep)-1,CONFIG['num_frames'])).astype(int)]
    assert len(set(selected))==CONFIG['num_frames'] and not excluded[selected].any()
    asset=ROOT/'mano_195_to_778.npz'
    with np.load(asset) as z:
        neighbors=z['neighbor_indices'];weights=z['geometry_weights'];ids=z['source_vertex_ids'];faces=z['faces']
    assert np.allclose(gt['hand_vertices_camera'][...,ids,:],gt['hand_markers_camera'],equal_nan=True,atol=1e-6)
    scores={};mesh_valid={};meshes={};counts={}
    for m,a in arrays.items():
        joints=camera_points(a,'joints')
        valid=a['hand_valid'].astype(bool)&gt_valid&~excluded[:,None]&np.isfinite(joints).all(axis=(2,3))
        if m!='gt':
            residual=np.linalg.norm(joints-gt['hand_joints_camera'],axis=-1).mean(-1)*1000
            scores[m]=float(residual[valid].mean())
            expected=pooled_score(s['method_window_metrics'][m])
            assert np.isclose(scores[m],expected,atol=2e-4), (dataset,m,scores[m],expected)
        counts[m]=valid.sum(0).tolist()
        if m=='egofound3r_stride5':
            markers=camera_points(a,'markers');center=markers.mean(-2,keepdims=True)
            vertices=center+((markers-center)[...,neighbors,:]*weights[...,None]).sum(-2)
            vertices[...,ids,:]=markers
            assert np.allclose(vertices[...,ids,:],markers,equal_nan=True,atol=0)
        else:vertices=camera_points(a,'vertices')
        assert vertices.shape==(60,2,778,3)
        meshes[m]=vertices
        # Method-native missing predictions stay missing, never remove another row's GT.
        mesh_valid[m]=valid & np.isfinite(vertices).all(axis=(2,3))
    pose=gt['camera_c2w'].astype(float)
    assert gt['camera_valid'][selected].all()
    display=np.array([[1,0,0],[0,0,-1],[0,-1,0]])@pose[0,:3,:3].T
    origin=pose[0,:3,3]
    transform=lambda x:(x-origin)@display.T
    xyz={m:transform(np.einsum('tij,tsvj->tsvi',pose[:,:3,:3],v)+pose[:,None,None,:3,3]) for m,v in meshes.items()}
    # Same five GT frustums and trajectory for every method, under the base scheme.
    h,w=ego_meta['source_resolution_hw'];camera_parts=[]
    edges=[(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
    for t in selected:
        assert gt['intrinsics_valid'][t]
        rays=np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]])@np.linalg.inv(gt['intrinsics'][t]).T
        corners=np.vstack([np.zeros(3),rays*.032])
        corners=transform(corners@pose[t,:3,:3].T+pose[t,:3,3])
        camera_parts.extend([[corners[a],corners[b]] for a,b in edges])
    centers=transform(pose[selected,:3,3]);camera_parts.extend([[a,b] for a,b in zip(centers[:-1],centers[1:])])
    points=np.concatenate([v[selected][mesh_valid[m][selected]].reshape(-1,3) for m,v in xyz.items()]+[np.asarray(camera_parts).reshape(-1,3)])
    low,high=points.min(0),points.max(0);center=(low+high)/2
    fw=max(high[0]-low[0],.3)*1.12;fd=max(high[1]-low[1],.3)*1.12
    floor=np.array([[center[0]+a*fw/2,center[1]+b*fd/2,low[2]-.03] for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    combined=np.concatenate([points,floor]);bounds=(combined.min(0),combined.max(0))
    provenance={'dataset':dataset,'window_id':s['window_id'],'source_run':s['source_run'],'render_config':CONFIG,
      'ranking_count':s['ranking_count'],'mpjpe_mm':scores,'valid_joint_frame_counts_left_right':counts,
      'excluded_indices':np.flatnonzero(excluded).tolist(),'retained_indices':keep.tolist(),'display_indices':selected.tolist(),
      'display_source_frame_ids':[s['frame_ids'][i] for i in selected],'shared_exclusion_mask':excluded.tolist(),
      'display_hand_valid':{m:v[selected].tolist() for m,v in mesh_valid.items()},'display_rotation':display.tolist(),'display_origin':origin.tolist(),
      'world_display':'camera-space geometry -> common GT camera_c2w -> shared display coordinates; no fitted transform',
      'hawor_coordinate_conversion':'own predicted camera_c2w inverse, then common GT camera_c2w',
      'ranking_metric':'v7 all8_p95 Joint MPJPE pooled by valid side/frame support; no baseline-dependent window selection',
      'sampling':'equally spaced retained-frame positions with rounded indices; identical selected source frames for every row',
      'geometry':'Ego reference centroid-local 195-to-778 interpolation with exact source anchors; native baseline/GT 778 vertices',
      'views':[],'layout':{'rows':4,'columns':6},'status':'complete'}
    return xyz,faces,selected,mesh_valid,camera_parts,floor,bounds,provenance

def render(dataset):
    xyz,faces,selected,valid,cameras,floor,bounds,info=prepare(dataset)
    p=ROOT/dataset; panels=p/'rendered';panels.mkdir(exist_ok=True)
    images={}
    for col,(label,elev,azim) in enumerate(VIEWS):
        group=[base.render_cell(xyz[m],faces,selected,valid[m],cameras,floor,bounds,elev,azim) for m in METHODS]
        boxes=[]
        for im in group:
            yy,xx=np.where((np.asarray(im)<247).any(-1));boxes.append([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
        b=np.array(boxes);crop=(max(0,int(b[:,0].min())-35),max(0,int(b[:,1].min())-35),min(1200,int(b[:,2].max())+35),min(1200,int(b[:,3].max())+35))
        info['views'].append({'name':label,'elevation':elev,'azimuth':azim,'shared_crop_all_four_rows':crop})
        for row,(m,im) in enumerate(zip(METHODS,group)):
            im=im.crop(crop);im.thumbnail((940,850),Image.Resampling.LANCZOS)
            cell=Image.new('RGB',(960,870),'white');cell.paste(im,((960-im.width)//2,(870-im.height)//2))
            images[row,col]=cell;cell.save(panels/(m+'_'+CONFIG['views'][col]+'.png'))
        print(json.dumps({'dataset':dataset,'view':label}),flush=True)
    width=5900;height=4090;canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    draw.text((40,24),NAMES[dataset]+'  |  best Ego stride5 MPJPE window  |  Joint8 P95',font=base.font(46,True),fill='#202a37')
    draw.text((40,86),info['window_id']+f"   |   Ego MPJPE {info['mpjpe_mm']['egofound3r_stride5']:.2f} mm   |   rank 1 / {info['ranking_count']}   |   retained {len(info['retained_indices'])}/60",font=base.font(29),fill='#566272')
    for col,(name,_,_) in enumerate(VIEWS):draw.text((40+col*980,150),name,font=base.font(37),fill='#26313f')
    for row,(m,label) in enumerate(zip(METHODS,LABELS)):
        y=208+row*940
        suffix='' if m=='gt' else f"   |   MPJPE {info['mpjpe_mm'][m]:.2f} mm"
        support=info['valid_joint_frame_counts_left_right'][m]
        draw.text((40,y),label+suffix+f'   |   valid L/R: {support[0]}/{support[1]}',font=base.font(34,True),fill='#26313f')
        draw.line((40,y+51,width-40,y+51),fill='#dce0e5',width=2)
        for col in range(6):canvas.paste(images[row,col],(20+col*980,y+65))
    y=3990
    for side,label in enumerate(['Left','Right']):
        x=40+side*460;draw.text((x,y),label,font=base.font(28),fill='#475261')
        for i in range(280):
            rgb=base.LIGHT[side]*(1-i/279)+base.DARK[side]*(i/279)
            draw.line((x+95+i,y+3,x+95+i,y+24),fill=tuple((rgb*255).astype(int)))
    draw.text((1020,y),'Earlier → Later   |   common GT camera display   |   same P95 mask and 5 source frames in all rows',font=base.font(27),fill='#606b78')
    ids=', '.join(info['display_source_frame_ids'])
    draw.text((40,y+47),'5 shared frames: '+ids+f"   |   excluded {len(info['excluded_indices'])}/60   |   same-mask v7; selected by Ego only",font=base.font(26),fill='#606b78')
    canvas.save(p/'summary.png',dpi=(300,300))
    preview=canvas.copy();preview.thumbnail((1770,1227),Image.Resampling.LANCZOS);preview.save(p/'preview.jpg',quality=93)
    (p/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    files=[p/'summary.png',p/'summary.json',*sorted(panels.glob('*.png'))]
    (p/'verification.json').write_text(json.dumps({'status':'complete','numeric_mpjpe_all_three_methods':'matches v7 report within 0.0002 mm','shared_frames':True,'panel_count':24,
        'sha256':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
    (p/'COMPLETE').write_text('complete\n')
    print(json.dumps({'dataset':dataset,'complete':True,'summary':str(p/'summary.png')}),flush=True)

if __name__=='__main__':
    for dataset in sys.argv[1:] or list(NAMES):render(dataset)
