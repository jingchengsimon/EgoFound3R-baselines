"""Offline GT-camera concatenation of existing clips; no inference or refitting."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'six_same_mask_v7_p95_bestego_20260909'
spec=importlib.util.spec_from_file_location('old_render',OLD/'render_examples.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
base=old.base
METHODS=['ego','wilor','hawor','gt']
LABELS={'ego':'Ego stride5','wilor':'WiLoR','hawor':'HaWoR','gt':'Ground truth'}
VIEWS=[base.VIEW_PRESETS[k] for k in ['front','left','top','right','side','bottom']]
def prepare(ds,p,emit_stitched=True):
    s=json.loads((p/'selection.json').read_text());arrays={}
    for method in METHODS:
        aa=[old.load_arrays(p/f'{i}_{method}.npz') for i in range(5)]
        arrays[method]={k:np.concatenate([a[k] for a in aa]) for k in aa[0]}
    for method in ['wilor','hawor']:
        for i,win in enumerate(s['windows']):
            md=json.loads((p/f'{i}_{method}_metadata.json').read_text());assert md['frame_ids']==win['gt']['frame_ids']
    e,g=arrays['ego'],arrays['gt'];mask=np.array([x for w in s['windows'] for x in w['excluded']],bool)
    ids=[x for w in s['windows'] for x in w['gt']['frame_ids']]
    assert len(ids)==len(set(ids))==300 and all(int(a)<int(b) for a,b in zip(ids[:-1],ids[1:]))
    assert g['camera_valid'].all() and np.isfinite(g['camera_c2w']).all()
    valid_gt=g['hand_valid']&np.isfinite(g['hand_joints_camera']).all(axis=(2,3))&~mask[:,None]
    valid_ego=e['hand_valid']&valid_gt&np.isfinite(e['hand_joints_camera']).all(axis=(2,3))
    keep=np.flatnonzero(valid_gt.any(1));assert len(keep)>=5
    targets=np.linspace(0,299,5);selected=np.array([keep[np.argmin(abs(keep-t))] for t in targets]);assert len(set(selected))==5
    with np.load(OLD/'mano_195_to_778.npz') as a:nei=a['neighbor_indices'];weights=a['geometry_weights'];anchors=a['source_vertex_ids'];faces=a['faces']
    markers=e['hand_markers_camera'];center=markers.mean(-2,keepdims=True)
    ev=center+((markers-center)[...,nei,:]*weights[...,None]).sum(-2);ev[...,anchors,:]=markers
    assert np.allclose(ev[...,anchors,:],markers,atol=0)
    c2w=g['camera_c2w'];to_world=lambda x:np.einsum('tij,tsvj->tsvi',c2w[:,:3,:3],x)+c2w[:,None,None,:3,3]
    world={'ego':to_world(ev),'gt':to_world(g['hand_vertices_camera'])}
    predj=to_world(e['hand_joints_camera']);gtj=to_world(g['hand_joints_camera'])
    errc=np.linalg.norm(e['hand_joints_camera']-g['hand_joints_camera'],axis=-1)
    errw=np.linalg.norm(predj-gtj,axis=-1)
    assert np.allclose(errc[valid_ego],errw[valid_ego],atol=2e-5)
    # One display transform for all five clips; never recenter each window.
    rotation=np.array([[1,0,0],[0,0,-1],[0,-1,0]])@c2w[0,:3,:3].T;origin=c2w[0,:3,3]
    display=lambda x:(x-origin)@rotation.T
    xyz={m:display(x) for m,x in world.items()};valid={'ego':valid_ego,'gt':valid_gt}
    for m in valid:valid[m]&=np.isfinite(xyz[m]).all(axis=(2,3))
    for method in ['wilor','hawor']:
        a=arrays[method];cam=old.camera_points(a,'vertices');j=old.camera_points(a,'joints')
        if 'hand_vertices_camera' not in a:
            rt=np.einsum('tij,tsvj->tsvi',a['camera_c2w'][:,:3,:3],cam)+a['camera_c2w'][:,None,None,:3,3]
            assert np.allclose(rt,a['hand_vertices_world'],atol=2e-5,equal_nan=True)
        world[method]=to_world(cam);xyz[method]=display(world[method])
        valid[method]=a['hand_valid']&valid_gt&np.isfinite(j).all(axis=(2,3))&np.isfinite(cam).all(axis=(2,3))
        if 'camera_valid' in a and 'hand_vertices_camera' not in a:valid[method]&=a['camera_valid'][:,None]
    cameras=[];edges=[(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
    md=json.loads((p/'0_ego_metadata.json').read_text());h,w=md['source_resolution_hw']
    for t in selected:
        assert g['intrinsics_valid'][t]
        rays=np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]])@np.linalg.inv(g['intrinsics'][t]).T
        corners=display(np.vstack([np.zeros(3),rays*.032])@c2w[t,:3,:3].T+c2w[t,:3,3])
        cameras.extend([[corners[a],corners[b]] for a,b in edges])
    centers=display(c2w[:,:3,3]);cameras.extend([[a,b] for a,b in zip(centers[:-1],centers[1:])])
    points=np.concatenate([xyz[m][selected][valid[m][selected]].reshape(-1,3) for m in xyz]+[np.array(cameras).reshape(-1,3)])
    low,high=points.min(0),points.max(0);center=(low+high)/2;fw=max(high[0]-low[0],.3)*1.12;fd=max(high[1]-low[1],.3)*1.12
    floor=np.array([[center[0]+a*fw/2,center[1]+b*fd/2,low[2]-.03] for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    points=np.concatenate([points,floor]);bounds=(points.min(0),points.max(0))
    out=p.parent
    if emit_stitched:
        np.savez_compressed(out/'stitched.npz',ego_joints_world_gt_camera=predj,ego_markers_world_gt_camera=to_world(markers),gt_joints_world=gtj,gt_vertices_world=world['gt'],gt_camera_c2w=c2w,ego_valid=valid_ego,gt_valid=valid_gt,shared_excluded=mask,display_indices=selected,display_times_seconds=np.arange(300)/30,frame_ids=np.array(ids),source_window_index=np.repeat(np.arange(5),60),wilor_vertices_world_gt_camera=world['wilor'],hawor_vertices_world_gt_camera=world['hawor'],wilor_valid=valid['wilor'],hawor_valid=valid['hawor'])
    info={'dataset':ds,'sequence_id':s['sequence_id'],'window_ids':[w['window_id'] for w in s['windows']],'selection_score_existing_window_w_mpjpe_mm':s['weighted_existing_w_mpjpe_mm'],'selection_candidate_count':s['candidate_count'],'selection_rank':s['rank'],'selection_band':s['selection_band'],'segment_id':s['segment_id'],'source_frame_start':ids[0],'source_frame_end':ids[-1],'source_subwindows':[[w['gt']['frame_ids'][0],w['gt']['frame_ids'][-1]] for w in s['windows']],'metric_note':'existing per-window P95 W-MPJPE pooled by valid side-frame support; not refitted 10s W-MPJPE','time_basis':s['time_basis'],'context_frames':300,'display_duration_seconds':10,'actual_source_duration_verified':False,'source_run':s['source_run'],'display_indices':selected.tolist(),'display_source_frame_ids':[ids[t] for t in selected],'display_relative_times_s':(selected/30).tolist(),'shared_excluded_count':int(mask.sum()),'valid_left_right':{m:v.sum(0).tolist() for m,v in valid.items()},'camera':'GT c2w per frame; common display transform across windows; no Sim3 or boundary fit','inference_run':False,'rigid_transform_preserves_camera_error':True,'views':[]}
    return xyz,faces,selected,valid,cameras,floor,bounds,info
def render(s):
    ds=s['dataset'];p=ROOT/ds/'segments'/s['segment_id'];gallery=ROOT/'png_gallery';gallery.mkdir(exist_ok=True)
    target=gallery/s['png_filename']
    if (p/'COMPLETE').exists() and target.exists():return s['segment_id']+' cached'
    xyz,faces,selected,valid,cameras,floor,bounds,info=prepare(ds,p/'inputs');cells={}
    for col,(label,elev,azim) in enumerate(VIEWS):
        pair=[base.render_cell(xyz[m],faces,selected,valid[m],cameras,floor,bounds,elev,azim) for m in METHODS]
        boxes=[]
        for im in pair:
            yy,xx=np.where((np.asarray(im)<247).any(-1));boxes.append([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
        b=np.array(boxes);crop=(max(0,int(b[:,0].min())-35),max(0,int(b[:,1].min())-35),min(1200,int(b[:,2].max())+35),min(1200,int(b[:,3].max())+35))
        for row,im in enumerate(pair):
            im=im.crop(crop);im.thumbnail((940,850),Image.Resampling.LANCZOS);cell=Image.new('RGB',(960,870),'white');cell.paste(im,((960-im.width)//2,(870-im.height)//2));cells[row,col]=cell
        info['views'].append({'name':label,'elevation':elev,'azimuth':azim,'shared_crop':crop})
    canvas=Image.new('RGB',(5900,4340),'white');d=ImageDraw.Draw(canvas)
    def heading(y,text,size=30,bold=False):
        while size>16 and d.textlength(text,font=base.font(size,bold))>5800:size-=1
        assert d.textlength(text,font=base.font(size,bold))<=5800
        d.text((40,y),text,font=base.font(size,bold),fill='#26313f')
    heading(20,f"{old.NAMES[ds]} | {s['selection_band']} | Ego stride5 / WiLoR / HaWoR / GT | 10s display (300 samples / 30 FPS)",43,True)
    heading(83,'Sequence: '+info['sequence_id'],34,True)
    heading(133,f"Source frame/time IDs: {info['source_frame_start']} -> {info['source_frame_end']} | Segment: {s['segment_id']}",30,True)
    heading(178,'Five source windows: '+' | '.join(a+'..'+b for a,b in info['source_subwindows']),28)
    heading(218,'Shown source IDs (earlier -> later): '+', '.join(info['display_source_frame_ids']),28)
    heading(260,f"Ego pooled P95 W-MPJPE: {info['selection_score_existing_window_w_mpjpe_mm']:.2f} mm | rank {s['rank']}/{s['candidate_count']} | GT-camera stitching; source elapsed time not certified",28)
    for col,(label,_,_) in enumerate(VIEWS):d.text((40+980*col,310),label,font=base.font(36),fill='#26313f')
    for row,m in enumerate(METHODS):
        y=363+row*945;support=info['valid_left_right'][m]
        d.text((40,y),LABELS[m]+f' | valid L/R: {support[0]}/{support[1]}',font=base.font(34,True),fill='#26313f')
        for col in range(6):canvas.paste(cells[row,col],(20+980*col,y+55))
    heading(4160,'Left: pink | Right: blue | Earlier -> Later | Same P95 exclusions and display frames | No inference or alignment refit',29)
    heading(4210,'Display times: '+', '.join(f'{t:.2f}s' for t in info['display_relative_times_s'])+f" | Excluded {info['shared_excluded_count']}/300 | Raw source IDs above locate the original data",28)
    canvas.save(target,dpi=(300,300));preview=canvas.copy();preview.thumbnail((1770,1350));preview.save(p/'preview.jpg',quality=91)
    with Image.open(target) as im:assert im.size==(5900,4340);im.verify()
    info.update(status='complete',png=str(target),png_filename=s['png_filename'],methods=METHODS,source_hashes=json.loads((p/'inputs/sources.json').read_text()))
    (p/'summary.json').write_text(json.dumps(info,indent=2));(p/'COMPLETE').write_text('41-batch member: identity, geometry, PNG verified\n')
    return s['segment_id']+' rendered'
if __name__=='__main__':
    import concurrent.futures,time
    rows=[json.loads(x) for x in (ROOT/'selected_manifest.jsonl').read_text().splitlines()]
    if len(sys.argv)>1:rows=[s for s in rows if s['segment_id'] in sys.argv[1:]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:
        tasks=[ex.submit(render,s) for s in rows]
        for t in concurrent.futures.as_completed(tasks):
            print(t.result(),flush=True)
            total=sum((ROOT/r['dataset']/'segments'/r['segment_id']/'COMPLETE').exists() for r in [json.loads(x) for x in (ROOT/'selected_manifest.jsonl').read_text().splitlines()])
            (ROOT/'state.json').write_text(json.dumps({'jobs':{'five::visualization':{'status':'running','completed_windows':total,'target_windows':41,'output_root':str(ROOT),'summary_path':str(ROOT/'summary.json'),'updated_at_epoch':time.time()}}},indent=2))
