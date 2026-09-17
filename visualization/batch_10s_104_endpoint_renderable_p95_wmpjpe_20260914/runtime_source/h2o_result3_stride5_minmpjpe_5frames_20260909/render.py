"""Render registered result-3 best H2O window, shared mask and geometry gauge."""
import hashlib
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/h2o_result3_mpl' if os.path.isdir('/private/tmp') else '/tmp/h2o_result3_mpl')
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEW_PRESETS = {
    'front': ('Front', 15., 90.), 'left': ('Oblique left', 30., 150.),
    'top': ('Top', 65., 90.), 'right': ('Oblique right', 30., 30.),
    'side': ('Side', 15., 0.), 'bottom': ('Bottom', -65., 90.),
}
VIEWS = []
CONFIG = {}
LIGHT = np.array([[.91,.72,.77],[.70,.82,.92]])
DARK = np.array([[.70,.32,.43],[.25,.46,.68]])


def arrays(name):
    with np.load(ROOT/'inputs'/name, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def geometry():
    selection=json.loads((ROOT/'inputs/selection.json').read_text())
    metadata=json.loads((ROOT/'inputs/metadata.json').read_text())
    pred,gt=arrays('prediction.npz'),arrays('gt.npz')
    assert metadata['window_id']==selection['window_id']
    assert metadata['global_stride']==CONFIG['global_stride'] and metadata['global_anchor_phase']==CONFIG['global_anchor_phase']
    assert metadata['frame_ids']==selection['indices']['gt']['frame_ids']
    excluded=np.array(selection['excluded'],bool)
    joint_valid=pred['hand_valid'] & gt['hand_valid'] & ~excluded[:,None]
    distance=np.linalg.norm(pred['hand_joints_camera'].astype(float)-gt['hand_joints_camera'].astype(float),axis=-1).mean(-1)*1000
    actual=float(distance[joint_valid].mean())
    assert abs(actual-selection['mpjpe_mm'])<1e-4, (actual,selection['mpjpe_mm'])
    keep=np.flatnonzero(~excluded & joint_valid.any(1))
    count=CONFIG['num_frames']
    assert count>=2 and len(keep)>=count
    selected=keep[np.rint(np.linspace(0,len(keep)-1,count)).astype(int)]
    assert len(np.unique(selected))==count and not excluded[selected].any()
    asset=ROOT/'inputs/mano_195_to_778.npz'
    with np.load(asset,allow_pickle=False) as z:
        neighbors=z['neighbor_indices']; weights=z['geometry_weights']
        source_ids=z['source_vertex_ids']; faces=z['faces']
    assert neighbors.shape==(778,3) and faces.shape==(1538,3)
    assert np.allclose(gt['hand_vertices_camera'][...,source_ids,:],gt['hand_markers_camera'],atol=1e-6)
    def interpolate(markers):
        # Exact reference renderer contract from hand_upsampling.build_mano778_hand:
        # centroid-local signed weights, then restore all observed marker anchors.
        source=markers.astype(float)
        center=source.mean(-2,keepdims=True)
        result=center+((source-center)[...,neighbors,:]*weights[...,None]).sum(-2)
        result[...,source_ids,:]=source
        return result
    vertices=interpolate(pred['hand_markers_camera'])
    assert np.array_equal(vertices[...,source_ids,:],pred['hand_markers_camera'])
    shift=np.array([.17,-.32,.65])
    assert np.allclose(interpolate(pred['hand_markers_camera'].astype(float)+shift),vertices+shift,atol=1e-7)
    meshes=[vertices,gt['hand_vertices_camera'].astype(float)]
    c2w=gt['camera_c2w'].astype(float)
    assert gt['camera_valid'][selected].all()
    # Same rigid display coordinate transform for both rows, no alignment fit.
    display=np.array([[1,0,0],[0,0,-1],[0,-1,0]])@c2w[0,:3,:3].T
    origin=c2w[0,:3,3]
    def to_display(x): return (x-origin)@display.T
    world=[np.einsum('tij,tsvj->tsvi',c2w[:,:3,:3],v)+c2w[:,None,None,:3,3] for v in meshes]
    xyz=[to_display(v) for v in world]
    # Frustums are glyphs; no geometry scaling. Use GT intrinsics for both rows.
    camera_parts=[]
    K=gt['intrinsics']
    height,width=metadata['source_resolution_hw']
    edges=[(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
    for t in selected:
        assert gt['intrinsics_valid'][t]
        pixels=np.array([[0,0,1],[width-1,0,1],[width-1,height-1,1],[0,height-1,1]],float)
        rays=pixels@np.linalg.inv(K[t]).T
        corners=np.vstack([np.zeros(3),rays*.032])
        corners=to_display(corners@c2w[t,:3,:3].T+c2w[t,:3,3])
        camera_parts.extend([[corners[a],corners[b]] for a,b in edges])
    centers=to_display(c2w[selected,:3,3])
    camera_parts.extend([[a,b] for a,b in zip(centers[:-1],centers[1:])])
    points=np.concatenate([x[selected][joint_valid[selected]].reshape(-1,3) for x in xyz]+[np.asarray(camera_parts).reshape(-1,3)])
    low,high=points.min(0),points.max(0)
    floor_z=low[2]-.03
    center=(low+high)/2
    floor_width=max(high[0]-low[0],.3)*1.12
    floor_depth=max(high[1]-low[1],.3)*1.12
    floor=np.array([[center[0]+a*floor_width/2,center[1]+b*floor_depth/2,floor_z] for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    bounds=np.concatenate([points,floor])
    low,high=bounds.min(0),bounds.max(0)
    provenance={
        'status':'complete','window_id':selection['window_id'],'ranking_count':selection['ranking_count'],
        'mpjpe_mm':actual,'ranking_metric':'retained-frame joint MPJPE; pooled valid left/right frame means',
        'excluded_frame_count':int(excluded.sum()),'shared_retained_indices':keep.tolist(),
        'display_frame_indices':selected.tolist(),'display_source_frame_ids':[metadata['frame_ids'][i] for i in selected],
        'display_relative_times_s':(selected/CONFIG['fps']).tolist(),
        'sampling':f'{count} nearest discrete frames to linspace over retained-frame order, including endpoints',
        'shared_hand_valid':joint_valid[selected].tolist(),'camera_source':'GT camera_c2w for both rows (result-3 oracle)',
        'display_rotation':display.tolist(),'display_origin':origin.tolist(),'fitted_alignment':False,
        'pred_geometry':'reference visualization centroid-local signed U interpolation with exact 195 source anchors; 778 derived vertices',
        'geometry_contract_source':'/private/tmp/egofound3r-model-2b9c180-20260908/egohandmetric_prompt/hand_upsampling.py:build_mano778_hand',
        'gt_geometry':'native 778 MANO vertices','mano_asset':str(asset),'mano_asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
        'views':[{'name':n,'elevation':e,'azimuth':a} for n,e,a in VIEWS],
        'renderer':'Matplotlib Agg, shared orthographic axes; PIL composition',
        'camera_frustum_depth_m':.032,'source_prediction_metadata':metadata,
        'run_id':CONFIG['run_id'], 'render_config':CONFIG,
        'layout':{'rows':2,'columns':len(VIEWS)},
        'bottom_view_ground':False,
    }
    return xyz,faces,selected,joint_valid,camera_parts,floor,(low,high),provenance


def render_cell(xyz,faces,selected,valid,camera_parts,floor,bounds,elev,azim,pixel_size=1200):
    fig=Figure(figsize=(pixel_size/150,pixel_size/150),dpi=150,facecolor='white')
    canvas=FigureCanvasAgg(fig)
    ax=fig.add_axes([0,0,1,1],projection='3d',proj_type='ortho',computed_zorder=False)
    ax.set_axis_off()
    ax.view_init(elev=elev,azim=azim)
    low,high=bounds
    center=(low+high)/2
    span=max(high-low)*.54
    ax.set_xlim(center[0]-span,center[0]+span)
    ax.set_ylim(center[1]-span,center[1]+span)
    ax.set_zlim(center[2]-span,center[2]+span)
    ax.set_box_aspect((1,1,1),zoom=1.15)
    polygons=[]; colors=[]
    # Solid, softly shaded surfaces; sorting occurs across every mesh triangle.
    for t in selected:
        phase=(t-selected[0])/max(int(selected[-1]-selected[0]),1)
        for side in range(2):
            if not valid[t,side]: continue
            tri=xyz[t,side][faces]
            normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
            light=np.array([-.4,.25,.88]); light/=np.linalg.norm(light)
            intensity=.45+.50*np.abs(normal@light)
            rgb=LIGHT[side]*(1-phase)+DARK[side]*phase
            polygons.extend(tri)
            colors.extend(np.clip(intensity[:,None]*rgb,0,1))
    # Two-level neutral stage, with no claim to reconstructed scene geometry.
    for scale,z,color in ([(1.10,-.012,[.87,.88,.90]),(1,0,[.94,.945,.95])] if elev>=0 else []):
        stage=floor.copy(); mid=stage.mean(0)
        stage[:,:2]=mid[:2]+(stage[:,:2]-mid[:2])*scale; stage[:,2]+=z
        ax.add_collection3d(Poly3DCollection([stage],facecolors=[color],edgecolors='none',zorder=0 if z else 1))
    collection=Poly3DCollection(polygons,facecolors=colors,edgecolors='none',linewidths=0,antialiased=False,zsort='average',zorder=3)
    ax.add_collection3d(collection)
    ax.add_collection3d(Line3DCollection(camera_parts,colors=[(.72,.48,.14,.60)],linewidths=.55,zorder=2))
    canvas.draw()
    result=Image.fromarray(np.asarray(canvas.buffer_rgba())[:,:,:3].copy())
    fig.clear()
    return result


def font(size,bold=False):
    base='/System/Library/Fonts/Supplemental/Arial'
    path=base+(' Bold.ttf' if bold else '.ttf')
    return ImageFont.truetype(path if os.path.isfile(path) else ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'),size)


def main():
    xyz,faces,selected,valid,cameras,floor,bounds,provenance=geometry()
    output=ROOT/'rendered'
    output.mkdir(exist_ok=True)
    images={}
    for col,(label,elev,azim) in enumerate(VIEWS):
        pair=[render_cell(x,faces,selected,valid,cameras,floor,bounds,elev,azim) for x in xyz]
        # The SAME crop is applied to both rows for a given view.
        boxes=[]
        for im in pair:
            mask=(np.asarray(im)<247).any(-1)
            yy,xx=np.where(mask)
            boxes.append([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
        box=np.array(boxes)
        crop=(max(0,int(box[:,0].min())-35),max(0,int(box[:,1].min())-35),min(1200,int(box[:,2].max())+35),min(1200,int(box[:,3].max())+35))
        provenance['views'][col]['shared_pixel_crop']=list(crop)
        for row,im in enumerate(pair):
            im=im.crop(crop)
            im.thumbnail((940,850),Image.Resampling.LANCZOS)
            cell=Image.new('RGB',(960,870),'white')
            cell.paste(im,((960-im.width)//2,(870-im.height)//2))
            images[row,col]=cell
            cell.save(output/f'{"ego_stride5" if row==0 else "gt"}_{label.lower().replace(" ","_")}.png')
        print(json.dumps({'rendered':label}),flush=True)
    width=20+980*len(VIEWS); height=2210
    canvas=Image.new('RGB',(width,height),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((40,24),CONFIG['title'],font=font(46,True),fill='#202a37')
    draw.text((40,86),f"{provenance['window_id']}   |   MPJPE {provenance['mpjpe_mm']:.2f} mm   |   rank 1 / {provenance['ranking_count']}",font=font(29),fill='#566272')
    for col,(label,_,_) in enumerate(VIEWS):
        draw.text((40+col*980,150),label,font=font(37),fill='#26313f')
    for row,label in enumerate([CONFIG['prediction_label'],'GT']):
        y=208+row*940
        draw.text((40,y),label,font=font(36,True),fill='#26313f')
        draw.line((40,y+51,width-40,y+51),fill='#dce0e5',width=2)
        for col in range(len(VIEWS)): canvas.paste(images[row,col],(20+col*980,y+65))
    y=2110
    for side,label in enumerate(['Left','Right']):
        x=40+side*460
        draw.text((x,y),label,font=font(28),fill='#475261')
        for i in range(280):
            color=LIGHT[side]*(1-i/279)+DARK[side]*(i/279)
            draw.line((x+95+i,y+3,x+95+i,y+24),fill=tuple((color*255).astype(int)),width=1)
    draw.text((1020,y),'Earlier → Later   |   GT camera for both rows',font=font(27),fill='#606b78')
    ids=', '.join(provenance['display_source_frame_ids'])
    draw.text((40,y+47),f"{len(selected)} shared frames: {ids}   |   {CONFIG['mask_label']} mask: {provenance['excluded_frame_count']} / {len(valid)} excluded   |   fixed 195→778 interpolation for Ego",font=font(27),fill='#606b78')
    canvas.save(ROOT/'summary.png',dpi=(300,300))
    canvas.save(ROOT/'summary.pdf',resolution=300)
    (ROOT/'summary.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'summary':str(ROOT/'summary.png'),'mpjpe_mm':provenance['mpjpe_mm'],'frames':provenance['display_source_frame_ids']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT,help='Artifact folder containing inputs/ and render_config.json')
    parser.add_argument('--views',help='Comma-separated preset names; defaults to render_config.json')
    parser.add_argument('--num-frames',type=int,help='Override number of shared temporal samples')
    args=parser.parse_args()
    ROOT=args.root.resolve()
    CONFIG=json.loads((ROOT/'render_config.json').read_text())
    if args.views: CONFIG['views']=args.views.split(',')
    if args.num_frames is not None: CONFIG['num_frames']=args.num_frames
    assert len(CONFIG['views'])==len(set(CONFIG['views'])) and CONFIG['views']
    VIEWS=[VIEW_PRESETS[key] for key in CONFIG['views']]
    main()
