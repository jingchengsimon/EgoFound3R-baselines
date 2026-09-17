import hashlib,json,math,os,subprocess,sys,time,uuid
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np

W,H=1920,1200
HEADER,FOOTER=80,40
COL_W=480
ROW_H=540
LABEL_H=36
STATUS_H=28
IMG_H=ROW_H-LABEL_H-STATUS_H
METHODS=('GT','Ego stride5 + display fill')
PROPS=('geometry','contact distance','contact','visibility')
LEFT_COLOR=(126,62,204)
RIGHT_COLOR=(206,132,45)

def load_npz(path,fields):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in fields}

def write_text(im,value,x,y,scale=.53,color=(235,238,240),thickness=1):
    cv2.putText(im,str(value),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)

def interpolate(vals,nei,sw,anchors):
    out=(vals[:,nei]*sw).sum(-1)
    out[:,anchors]=vals
    return out

def make_mesh(markers,nei,weights,anchors):
    center=markers.mean(axis=1,keepdims=True)
    mesh=center+((markers[:,nei,:]-center[:,:,None,:])*weights[None,:,:,None]).sum(axis=2)
    mesh[:,anchors,:]=markers
    return mesh

def project(xyz,K,scale,x0,y0):
    q=xyz@K.T
    z=xyz[...,2]
    uv=q[...,:2]/np.where(z[...,None]>1e-6,q[...,2:3],np.nan)
    uv=uv*scale+np.array([x0,y0])
    valid=(z>1e-6)&np.isfinite(uv).all(axis=-1)&(np.abs(uv)<1e6).all(axis=-1)
    return uv,valid

def draw_mesh(image,uv,valid_points,hand_valid,faces):
    for side,color in enumerate((LEFT_COLOR,RIGHT_COLOR)):
        if not hand_valid[side]:continue
        good=valid_points[side,faces].all(axis=1)
        if not good.any():continue
        tri=np.round(uv[side,faces[good]]).astype(np.int32)
        mask=np.zeros(image.shape[:2],np.uint8)
        cv2.fillPoly(mask,list(tri),255)
        if mask.any():
            tinted=np.empty_like(image);tinted[:]=color
            image[:]=np.where(mask[...,None]>0,cv2.addWeighted(image,.48,tinted,.52,0),image)
        # Sparse source vertices make the reconstructed/native geometry readable.
        pts=np.round(uv[side,valid_points[side]]).astype(np.int32)
        inbounds=(pts[:,0]>=0)&(pts[:,0]<image.shape[1])&(pts[:,1]>=0)&(pts[:,1]<image.shape[0])
        pts=pts[inbounds]
        image[pts[:,1],pts[:,0]]=color

def draw_values(image,uv,point_valid,hand_valid,values,value_mask,kind):
    ih,iw=image.shape[:2]
    count=0
    for side in range(2):
        if not hand_valid[side]:continue
        mask=point_valid[side]&value_mask[side]&np.isfinite(values[side])
        if not mask.any():continue
        pts=np.rint(uv[side,mask]).astype(np.int32)
        vals=values[side,mask]
        inside=(pts[:,0]>=1)&(pts[:,0]<iw-1)&(pts[:,1]>=1)&(pts[:,1]<ih-1)
        pts=pts[inside];vals=vals[inside]
        if not len(pts):continue
        if kind=='distance':
            colors=cv2.applyColorMap(np.uint8(np.clip(vals*1000/50*255,0,255)).reshape(-1,1),cv2.COLORMAP_TURBO).reshape(-1,3)
        elif kind=='contact':
            colors=np.where((vals>=.5)[:,None],np.array([60,60,245],np.uint8),np.array([220,200,55],np.uint8))
        else:
            colors=np.where((vals>=.5)[:,None],np.array([65,220,65],np.uint8),np.array([65,65,225],np.uint8))
        xx=pts[:,0];yy=pts[:,1]
        for dy in (-1,0,1):
            for dx in (-1,0,1):image[yy+dy,xx+dx]=colors
        count+=len(pts)
    return count

def get_window(ref,clip,payload,cache):
    key=(clip['dataset'],ref['cache_id'])
    if key in cache:
        cache.move_to_end(key);return cache[key]
    ds,cid=key
    roots=payload['roots'][ds]
    root_record=Path(roots['record'])
    root_ego=Path(roots['ego'])
    root_gt=Path(roots['gt'])
    rec=json.loads((root_record/cid/'window_input.json').read_text())
    md=json.loads((root_ego/cid/'metadata.json').read_text())
    if (rec['dataset'],rec['sequence_id'],rec['window_id'],rec['cache_id'])!=(ds,clip['sequence_id'],ref['window_id'],cid):raise ValueError('record identity mismatch')
    if (md['dataset'],md['window_id'],md['frame_ids'],md.get('global_stride'))!=(ds,ref['window_id'],rec['frame_ids'],5):raise ValueError('Ego metadata identity mismatch')
    ego=load_npz(root_ego/cid/'predictions.npz',('intrinsics','intrinsics_valid','hand_valid','hand_joints_camera','hand_markers_camera','marker_visibility','marker_contact_probability'))
    k_valid=np.asarray(ego['intrinsics_valid'],dtype=bool)&np.isfinite(ego['intrinsics']).all(axis=(1,2))
    anchors=np.flatnonzero(k_valid)
    if not len(anchors):raise ValueError('no valid Ego K anchor '+cid)
    nearest=anchors[np.abs(np.arange(len(k_valid))[:,None]-anchors[None,:]).argmin(axis=1)]
    gt=load_npz(root_gt/(cid+'.npz'),('intrinsics','intrinsics_valid','hand_valid','hand_vertices_camera'))
    idx=payload['distance_paths'][ds+'|'+cid]
    ev=load_npz(idx['ego'],('hand_valid','vertex_contact_distance','vertex_contact_distance_mask'))
    gv=load_npz(idx['gt'],('vertex_contact_target','vertex_contact_mask','vertex_contact_distance','vertex_contact_distance_mask'))
    item={'record':rec,'ego':ego,'gt':gt,'ev':ev,'gv':gv,'k_source':nearest}
    cache[key]=item
    if len(cache)>12:cache.popitem(last=False)
    return item

def missing_runs(valid):
    result=[]
    for side in range(2):
        start=None
        for index,value in enumerate(list(valid[:,side])+[True]):
            if not value and start is None:start=index
            elif value and start is not None:
                result.append({'side':['left','right'][side],'start_display_index':start,'end_display_index':index-1,'length':index-start})
                start=None
    return result

def smoothstep_fill_geometry(joint,marker,trusted):
    joint=joint.copy();marker=marker.copy();fill=np.zeros_like(trusted);mode=np.zeros_like(trusted,dtype=np.int8)
    for side in range(2):
        valid=trusted[:,side]&np.isfinite(joint[:,side]).all(axis=(1,2))&np.isfinite(marker[:,side]).all(axis=(1,2))
        indices=np.flatnonzero(valid)
        if not len(indices):raise ValueError('no trusted anchor for '+['left','right'][side])
        for frame in np.flatnonzero(~valid):
            previous=indices[indices<frame];following=indices[indices>frame]
            if not len(previous) or not len(following):raise ValueError('unbracketed missing Hand after visibility60 selection')
            left=int(previous[-1]);right=int(following[0]);alpha=(frame-left)/float(right-left);alpha=alpha*alpha*(3.0-2.0*alpha)
            left_root=joint[left,side,0];right_root=joint[right,side,0]
            root=(1-alpha)*left_root+alpha*right_root
            joint_local=(1-alpha)*(joint[left,side]-left_root)+alpha*(joint[right,side]-right_root)
            marker_local=(1-alpha)*(marker[left,side]-left_root)+alpha*(marker[right,side]-right_root)
            joint[frame,side]=root+joint_local;marker[frame,side]=root+marker_local
            fill[frame,side]=True;mode[frame,side]=1
    return joint,marker,trusted|fill,fill,mode

def smoothstep_fill_values(values,trusted,value_mask=None):
    values=values.copy();mask=np.isfinite(values) if value_mask is None else (value_mask.copy()&np.isfinite(values))
    for side in range(2):
        indices=np.flatnonzero(trusted[:,side])
        for frame in np.flatnonzero(~trusted[:,side]):
            previous=indices[indices<frame];following=indices[indices>frame]
            if not len(previous) or not len(following):raise ValueError('unbracketed scalar fill')
            left=int(previous[-1]);right=int(following[0]);alpha=(frame-left)/float(right-left);alpha=alpha*alpha*(3.0-2.0*alpha)
            common=mask[left,side]&mask[right,side]
            values[frame,side]=np.where(common,(1-alpha)*values[left,side]+alpha*values[right,side],values[frame,side])
            mask[frame,side]=common
    return values,mask

def smooth_local_shape(joint,marker,valid,exact_anchor):
    out_joint=joint.copy();out_marker=marker.copy();applied=np.zeros_like(valid)
    weights=np.array([.25,.5,.25],dtype=joint.dtype).reshape(3,1,1)
    for frame in range(1,len(joint)-1):
        support=valid[frame-1]&valid[frame]&valid[frame+1]&~exact_anchor[frame]
        for side in np.flatnonzero(support):
            root=joint[frame,side,0]
            local_joint=np.stack((joint[frame-1,side]-joint[frame-1,side,:1],joint[frame,side]-root,joint[frame+1,side]-joint[frame+1,side,:1]))
            local_marker=np.stack((marker[frame-1,side]-joint[frame-1,side,:1],marker[frame,side]-root,marker[frame+1,side]-joint[frame+1,side,:1]))
            out_joint[frame,side]=root+(local_joint*weights).sum(0)
            out_marker[frame,side]=root+(local_marker*weights).sum(0)
            applied[frame,side]=True
    return out_joint,out_marker,applied

def prepare_clip(clip,payload,cache):
    joint=[];marker=[];contact=[];visibility=[];distance=[];distance_mask=[];trusted=[];exact=[]
    for frame_id,ref in zip(clip['frame_ids'],clip['frame_refs']):
        arrays=get_window(ref,clip,payload,cache);t=ref['index'];e=arrays['ego'];g=arrays['gt'];ev=arrays['ev']
        if arrays['record']['frame_ids'][t]!=frame_id:raise ValueError('prepare frame identity mismatch')
        evalid=e['hand_valid'][t].astype(bool)&np.isfinite(e['hand_joints_camera'][t]).all(axis=(1,2))&np.isfinite(e['hand_markers_camera'][t]).all(axis=(1,2))
        gvalid=g['hand_valid'][t].astype(bool)&np.isfinite(g['hand_vertices_camera'][t]).all(axis=(1,2))
        strict=evalid&gvalid
        joint.append(e['hand_joints_camera'][t]);marker.append(e['hand_markers_camera'][t]);contact.append(e['marker_contact_probability'][t]);visibility.append(e['marker_visibility'][t]);distance.append(ev['vertex_contact_distance'][t]);distance_mask.append(ev['vertex_contact_distance_mask'][t]);trusted.append(strict);exact.append(strict&bool(e['intrinsics_valid'][t]))
    joint=np.asarray(joint);marker=np.asarray(marker);contact=np.asarray(contact);visibility=np.asarray(visibility);distance=np.asarray(distance);distance_mask=np.asarray(distance_mask,dtype=bool);trusted=np.asarray(trusted,dtype=bool);exact=np.asarray(exact,dtype=bool)
    if trusted.all(axis=1).tolist()!=clip['per_frame_both_valid']:raise ValueError('frozen visibility mismatch')
    runs=missing_runs(trusted)
    if any(x['length']>60 for x in runs):raise ValueError('internal side gap exceeds 60')
    if runs:
        joint,marker,display,fill,mode=smoothstep_fill_geometry(joint,marker,trusted)
        contact,_=smoothstep_fill_values(contact,trusted)
        visibility,_=smoothstep_fill_values(visibility,trusted)
        distance,distance_mask=smoothstep_fill_values(distance,trusted,distance_mask)
        joint,marker,smoothed=smooth_local_shape(joint,marker,display,exact)
    else:
        display=trusted.copy();fill=np.zeros_like(trusted);mode=np.zeros_like(trusted,dtype=np.int8);smoothed=np.zeros_like(trusted)
    if not display.all():raise ValueError('selected clip still has unfilled Hand')
    return {'joint':joint,'marker':marker,'contact':contact,'visibility':visibility,'distance':distance,'distance_mask':distance_mask,'strict_valid':trusted,'display_valid':display,'fill':fill,'fill_mode':mode,'local_smoothing':smoothed,'missing_runs':runs}

def frame_canvas(bgr,arrays,t,payload,clip,frame_id,frame_offset,prepared):
    src_h,src_w=bgr.shape[:2]
    scale=min(COL_W/src_w,IMG_H/src_h)
    im_w=max(1,round(src_w*scale));im_h=max(1,round(src_h*scale))
    resized=cv2.resize(bgr,(im_w,im_h),interpolation=cv2.INTER_AREA)
    image_x=(COL_W-im_w)//2;image_y=(IMG_H-im_h)//2
    base=np.full((IMG_H,COL_W,3),(34,39,44),np.uint8)
    base[image_y:image_y+im_h,image_x:image_x+im_w]=resized
    canvas=np.full((H,W,3),(22,27,32),np.uint8)
    seq=clip['sequence_id']
    title=f"{clip['dataset']} | rank {clip['dataset_rank']} | center {clip['frame_id']} | {seq}"
    write_text(canvas,title[:145],16,29,.62,thickness=2)
    write_text(canvas,f"source {clip['first_frame_id']} .. {clip['last_frame_id']} | {clip['actual_length']} frames @ 30 FPS | frame {frame_offset+1}/{clip['actual_length']} = {frame_id}",16,58,.55)
    e=arrays['ego'];g=arrays['gt'];ev=arrays['ev'];gv=arrays['gv'];rec=arrays['record']
    if rec['frame_ids'][t]!=frame_id:raise ValueError('frame identity mismatch')
    if not np.allclose(np.asarray(rec['intrinsics'][t]),g['intrinsics'][t],atol=1e-5):raise ValueError('GT RGB K mismatch')
    strict=prepared['strict_valid'][frame_offset]
    eh=prepared['display_valid'][frame_offset]
    fill=prepared['fill'][frame_offset]
    gh=g['hand_valid'][t].astype(bool)
    eKvalid=bool(e['intrinsics_valid'][t]) and bool(np.isfinite(e['intrinsics'][t]).all())
    k_source=int(arrays['k_source'][t])
    gKvalid=bool(g['intrinsics_valid'][t]) and bool(np.isfinite(g['intrinsics'][t]).all())
    if not gKvalid:raise ValueError('GT K invalid')
    side=min(src_w,src_h);x_crop=(src_w-side)//2;y_crop=(src_h-side)//2;s=side/512
    crop_to_rgb=np.array([[s,0,x_crop+(s-1)/2],[0,s,y_crop+(s-1)/2],[0,0,1]],dtype=np.float64)
    egoK=crop_to_rgb@e['intrinsics'][k_source]
    gtK=g['intrinsics'][t]
    mapping=payload['mapping'];nei=np.asarray(mapping['neighbor_indices']);weights=np.asarray(mapping['geometry_weights']);anchors=np.asarray(mapping['source_vertex_ids']);faces=np.asarray(mapping['faces']);sw=np.asarray(mapping['scalar_weights'])
    gt_xyz=g['hand_vertices_camera'][t]
    ego_xyz=make_mesh(prepared['marker'][frame_offset],nei,weights,anchors) if eh.any() else None
    gt_uv,gt_ok=project(gt_xyz,gtK,scale,image_x,image_y)
    if ego_xyz is not None:ego_uv,ego_ok=project(ego_xyz,egoK,scale,image_x,image_y)
    else:ego_uv=ego_ok=None
    statuses=[]
    for row,method in enumerate(METHODS):
        for col,prop in enumerate(PROPS):
            tile=base.copy();reason='ok'
            if row==0:
                xyz,uv,ok,hv=gt_xyz,gt_uv,gt_ok,gh
                if prop=='visibility':reason='GT visibility N/A'
                elif not hv.any():reason='GT hand invalid'
                elif prop=='geometry':draw_mesh(tile,uv,ok,hv,faces)
                elif prop=='contact':
                    n=draw_values(tile,uv,ok,hv,gv['vertex_contact_target'][t],gv['vertex_contact_mask'][t],'contact')
                    if n==0:reason='contact mask empty'
                else:
                    n=draw_values(tile,uv,ok,hv,gv['vertex_contact_distance'][t],gv['vertex_contact_distance_mask'][t],'distance')
                    if n==0:reason='distance mask empty'
            else:
                if not eh.any():reason='Ego/GT hand invalid'
                elif prop=='geometry':draw_mesh(tile,ego_uv,ego_ok,eh,faces)
                elif prop=='contact distance':
                    n=draw_values(tile,ego_uv,ego_ok,eh,prepared['distance'][frame_offset],prepared['distance_mask'][frame_offset],'distance')
                    if n==0:reason='distance mask empty'
                elif prop=='contact':
                    values=interpolate(prepared['contact'][frame_offset],nei,sw,anchors)
                    n=draw_values(tile,ego_uv,ego_ok,eh,values,np.isfinite(values),'contact')
                    if n==0:reason='contact unavailable'
                else:
                    values=interpolate(prepared['visibility'][frame_offset],nei,sw,anchors)
                    n=draw_values(tile,ego_uv,ego_ok,eh,values,np.isfinite(values),'visibility')
                    if n==0:reason='visibility unavailable'
                if k_source!=t:reason+=f' | K nearest {rec["frame_ids"][k_source]}'
                if fill.any():reason+=f" | fill 8fc061a: {','.join(np.array(['L','R'])[fill])}"
            x=col*COL_W;y=HEADER+row*ROW_H
            canvas[y+LABEL_H:y+LABEL_H+IMG_H,x:x+COL_W]=tile
            cv2.rectangle(canvas,(x,y),(x+COL_W-1,y+ROW_H-1),(75,82,89),1)
            label=f'{method} | {prop}'
            if row==1 and prop=='contact':label+=' [derived 195->778]'
            if row==1 and prop=='visibility':label=f'{method} | predicted visibility [195->778]'
            if prop=='contact distance':label+=' [mm: 0-50]'
            write_text(canvas,label,x+8,y+24,.50,thickness=1)
            write_text(canvas,reason,x+8,y+ROW_H-8,.46,color=(195,204,214))
            statuses.append(reason)
    write_text(canvas,'Pink/blue = left/right | red/cyan = contact/no contact | green/red = predicted visibility (p>=0.5) | masked = undefined',14,H-13,.48)
    rgb_roi=(3*COL_W+image_x,HEADER+LABEL_H+image_y,im_w,im_h)
    return canvas,statuses,rgb_roi,resized,eKvalid,k_source,strict.tolist(),eh.tolist(),fill.tolist(),prepared['local_smoothing'][frame_offset].tolist()

def verify_video(path,expected,first_rgb,last_rgb,first_roi,last_roi):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError('cannot decode MP4')
    count=0;first=None;last=None
    while True:
        ok,frame=cap.read()
        if not ok:break
        if first is None:first=frame
        last=frame;count+=1
    cap.release()
    if count!=expected:raise ValueError(f'MP4 frames {count} != {expected}')
    errs=[]
    for frame,source,roi in ((first,first_rgb,first_roi),(last,last_rgb,last_roi)):
        x,y,w,h=roi;crop=frame[y:y+h,x:x+w]
        if crop.shape!=source.shape:raise ValueError('RGB ROI shape mismatch')
        mae=float(np.abs(crop.astype(np.int16)-source.astype(np.int16)).mean())
        if mae>12:raise ValueError(f'RGB tile source identity MAE {mae:.3f} > 12')
        errs.append(mae)
    return count,errs

def render_clip(clip,payload,out):
    ds=clip['dataset'];rank=clip['dataset_rank']
    h=hashlib.sha1((ds+'|'+clip['sequence_id']+'|'+clip['frame_id']).encode()).hexdigest()[:10]
    stem=f'{ds}_r{rank:03d}_{h}'
    png=out/'png_gallery'/(stem+'.png');video=out/'video_gallery'/(stem+'.mp4')
    frames_file=out/'frame_manifests'/(stem+'.jsonl');report=out/'reports'/(stem+'.json')
    if any(p.exists() for p in (png,video,frames_file,report)):raise ValueError(f'output exists for {stem}')
    tmpvideo=video.with_suffix('.tmp-'+uuid.uuid4().hex+'.mp4')
    tmppng=png.with_suffix('.tmp-'+uuid.uuid4().hex+'.png')
    tmpframes=frames_file.with_suffix('.tmp-'+uuid.uuid4().hex+'.jsonl')
    cache=OrderedDict();prepared=prepare_clip(clip,payload,cache);first_rgb=last_rgb=first_roi=last_roi=None;center_written=False
    ff=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(tmpvideo)]
    proc=subprocess.Popen(ff,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    counts={};started=time.monotonic()
    try:
        with tmpframes.open('w') as sidecar:
            for offset,(fid,ref) in enumerate(zip(clip['frame_ids'],clip['frame_refs'])):
                arrays=get_window(ref,clip,payload,cache);t=ref['index']
                rgb_path=arrays['record']['rgb_paths'][t]
                bgr=cv2.imread(rgb_path,cv2.IMREAD_COLOR)
                if bgr is None:raise ValueError('RGB unreadable '+rgb_path)
                canvas,statuses,roi,resized,egoKvalid,k_source,ego_metric_valid,ego_display_valid,fill,local_smoothing=frame_canvas(bgr,arrays,t,payload,clip,fid,offset,prepared)
                if offset==0:first_rgb=resized;first_roi=roi
                last_rgb=resized;last_roi=roi
                if offset==clip['center_index_in_clip']:
                    if not cv2.imwrite(str(tmppng),canvas):raise ValueError('PNG write failed')
                    center_written=True
                proc.stdin.write(canvas.tobytes())
                sidecar.write(json.dumps({'display_index':offset,'source_frame_id':fid,'window_id':ref['window_id'],'cache_id':ref['cache_id'],'window_index':t,'rgb_path':rgb_path,'ego_intrinsics_valid':egoKvalid,'ego_intrinsics_source_frame_id':arrays['record']['frame_ids'][k_source],'ego_intrinsics_mode':'native' if k_source==t else 'nearest_anchor','ego_metric_hand_valid':ego_metric_valid,'ego_display_hand_valid':ego_display_valid,'ego_hand_missing_fill':fill,'ego_fill_mode':['bracketed' if x else 'observed' for x in fill],'ego_local_smoothing_applied':local_smoothing,'interpolation_commit':payload['interpolation_commit'],'panel_statuses':statuses},separators=(',',':'))+'\n')
        proc.stdin.close();err=proc.stderr.read().decode(errors='replace');rc=proc.wait()
        if rc:raise ValueError('ffmpeg failed '+err[-500:])
        if not center_written:raise ValueError('center PNG missing')
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames','-of','json',str(tmpvideo)],capture_output=True,text=True,check=True)
        stream=json.loads(probe.stdout)['streams'][0]
        if (stream['width'],stream['height'],stream['r_frame_rate'],int(stream['nb_frames']))!=(W,H,'30/1',clip['actual_length']):raise ValueError('ffprobe contract mismatch '+str(stream))
        decoded,maes=verify_video(tmpvideo,clip['actual_length'],first_rgb,last_rgb,first_roi,last_roi)
        os.replace(tmppng,png);os.replace(tmpframes,frames_file);os.replace(tmpvideo,video)
        result={'dataset':ds,'sequence_id':clip['sequence_id'],'dataset_rank':rank,'center_frame_id':clip['frame_id'],'center_index':clip['center_index_in_clip'],'first_frame_id':clip['first_frame_id'],'last_frame_id':clip['last_frame_id'],'frame_count':clip['actual_length'],'fps':30,'png':str(png),'video':str(video),'frame_manifest':str(frames_file),'strict_valid_side_frames':int(prepared['strict_valid'].sum()),'filled_side_frames':int(prepared['fill'].sum()),'missing_runs':prepared['missing_runs'],'local_smoothing_side_frames':int(prepared['local_smoothing'].sum()),'center_fill_sides':prepared['fill'][clip['center_index_in_clip']].tolist(),'interpolation_commit':payload['interpolation_commit'],'interpolation_scope':'visualization-only camera-space smoothstep fill plus final root-relative [1,2,1]/4 smoothing; strict validity unchanged','first_last_rgb_mae':maes,'decoded_frames':decoded,'elapsed_seconds':round(time.monotonic()-started,3),'status':'verified'}
        report.write_text(json.dumps(result,indent=2)+'\n')
        return result
    except Exception:
        try:
            if proc.poll() is None:proc.kill();proc.wait(timeout=5)
        except Exception:pass
        raise

def main():
    payload=json.load(sys.stdin)
    out=Path(payload['output_root'])
    expected=payload['selection_sha256']
    if len(expected)!=64:raise ValueError('missing freeze hash')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('png_gallery','video_gallery','frame_manifests','reports','errors'):(out/name).mkdir(exist_ok=True)
    contract=out/'run_contract.json'
    if contract.exists():
        old=json.loads(contract.read_text())
        if old['selection_sha256']!=expected:raise ValueError('output selection mismatch')
    else:
        contract.write_text(json.dumps({'selection_sha256':expected,'expected_centers':payload['expected_centers'],'expected_png':payload['expected_centers'],'expected_mp4':payload['expected_centers'],'expected_display_frames':payload['expected_display_frames'],'expected_filled_side_frames':payload['expected_filled_side_frames'],'fps':30,'layout':'2x4 GT/Ego x geometry/distance/contact/visibility','ego_intrinsics':'nearest valid stride5 anchor when native K invalid','interpolation_commit':payload['interpolation_commit'],'interpolation_functions':'numpy-equivalent of _camera_space_smooth_fill and smooth_display_hand_local_shape from commit','interpolation_scope':'visualization only; source prediction and strict metric validity unchanged; no model rerun or root-UV re-solve','distance_panel':'GT and Ego vertex_contact_distance in meters, Turbo 0-50 mm','visibility_panel':'Ego marker_visibility probability projected via fixed 195-to-778 mapping, threshold 0.5; prediction, not RGB-observed hand presence','status':'in_progress'},indent=2)+'\n')
    for clip in payload['clips']:
        try:
            result=render_clip(clip,payload,out)
            print(json.dumps({'event':'clip_verified','dataset':result['dataset'],'rank':result['dataset_rank'],'frames':result['frame_count'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
        except Exception as e:
            (out/'errors'/(f"{clip['dataset']}_r{clip['dataset_rank']:03d}.txt")).write_text(repr(e)+'\n')
            print(json.dumps({'event':'clip_failed','dataset':clip['dataset'],'rank':clip['dataset_rank'],'error':str(e)[:500]}),flush=True)
            raise
if __name__=='__main__':main()
