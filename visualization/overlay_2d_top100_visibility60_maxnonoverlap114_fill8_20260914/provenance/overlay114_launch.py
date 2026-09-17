import argparse, hashlib, json, shlex, subprocess
from pathlib import Path
import numpy as np
BASE=Path('/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/EgoFound3R-baselines')
SELROOT=Path('/private/tmp/overlay2d_visibility60_selected114')
OUT='/mnt/workspace/sjc/eval_artifacts/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914'
COMMIT='8fc061a615895bd3b5a556f7387bae306e32d9db'
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('smoke','batch'),required=True);parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1);args=parser.parse_args()
raw=(SELROOT/'selected_manifest.jsonl').read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha=='6a95a1c01288fbe91589345e013bd7c96bcb038ebef2414bea140240824f4d2f'
clips=[json.loads(s) for s in raw.splitlines()]
smoke=next(x for x in clips if x['dataset']=='arctic' and x['dataset_rank']==28)
if args.phase=='smoke': chosen=[smoke]
else:
 remaining=[x for x in clips if x is not smoke];assert 0<=args.shard_index<args.shard_count
 chosen=[x for i,x in enumerate(remaining) if i%args.shard_count==args.shard_index]
roots={
'arctic':{'record':'/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_arctic_20260820T000131Z/window_inputs/arctic','ego':'/mnt/cpfs/sjc/eval_artifacts/base3e_traincrop_512_stride5_1078f15_20260908/arctic/arctic/egofound3r/formal','gt':'/mnt/oss/pre-train/ego/eval_artifacts/formal_standard_reports_10methods_20260824/arctic/gt_cache/arctic'},
'h2o':{'record':'/mnt/workspace/sjc/eval_artifacts/prep_h2o_60f_strict_20260822T030835Z_5001_g1/window_inputs/h2o','ego':'/mnt/cpfs/sjc/eval_artifacts/base3e_traincrop_512_stride5_1078f15_20260908/h2o/h2o/egofound3r/formal','gt':'/mnt/oss/pre-train/ego/eval_artifacts/formal_standard_reports_10methods_20260824/h2o/gt_cache/h2o'},
'hot3d':{'record':'/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_20260818T0410Z_8b1a806/window_inputs/hot3d','ego':'/mnt/cpfs/sjc/eval_artifacts/base3e_traincrop_512_stride5_1078f15_20260908/hot3d/hot3d/egofound3r/formal','gt':'/mnt/oss/pre-train/ego/eval_artifacts/formal_standard_reports_10methods_20260824/hot3d/gt_cache/hot3d'},
'oakink_v2':{'record':'/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_oakink_v2_20260819T144050Z/window_inputs/oakink_v2','ego':'/mnt/cpfs/sjc/eval_artifacts/base3e_traincrop_512_stride5_1078f15_20260908/oakink_v2/oakink_v2/egofound3r/formal','gt':'/mnt/oss/pre-train/ego/eval_artifacts/formal_standard_reports_10methods_20260824/oakink_v2/gt_cache/oakink_v2'},
'taco':{'record':'/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_taco_20260822T101500Z_5001_g4_bin0/window_inputs/taco','ego':'/mnt/cpfs/sjc/eval_artifacts/base3e_traincrop_512_stride5_1078f15_20260908/taco/taco/egofound3r/formal','gt':'/mnt/workspace/sjc/DATA/eval_artifacts/formal_contact_taco_metrics_20260831T234159Z/gt_cache/taco'}}
idx=json.load(open('/private/tmp/top100_truncated_distance_index_audit_20260914.json'))
paths={x['dataset']+'|'+x['cache_id']:{'ego':x['ego_array_path'],'gt':x['gt_array_path']} for x in idx['rows']}
asset=Path('/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/EgoFound3R_dev/.worktrees/ablation-one-way-vggt-3e533881-20260907/egohandmetric_prompt/data/mano_upsampling/mano_195_to_778.npz')
with np.load(asset,allow_pickle=False) as z:
 w=np.maximum(z['geometry_weights'],0);w=w/w.sum(-1,keepdims=True)
 mapping={'neighbor_indices':z['neighbor_indices'].tolist(),'geometry_weights':z['geometry_weights'].tolist(),'source_vertex_ids':z['source_vertex_ids'].tolist(),'faces':z['faces'].tolist(),'scalar_weights':w.tolist()}
inst=json.loads((BASE/'.auto_scheduler/current_dsw_instance.json').read_text());code=Path('/private/tmp/overlay114_remote.py').read_text()
payload={'selection_sha256':sha,'output_root':OUT,'expected_centers':114,'expected_display_frames':12559,'expected_filled_side_frames':310,'interpolation_commit':COMMIT,'clips':chosen,'roots':roots,'distance_paths':paths,'mapping':mapping}
cmd=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','-o','ConnectTimeout=15','-i',str(Path(inst['key']).expanduser()),'-p','5001','root@'+inst['host'],'python3 -c '+shlex.quote(code)]
print(json.dumps({'phase':args.phase,'shard_index':args.shard_index,'shard_count':args.shard_count,'clips':len(chosen),'output_root':OUT,'selection_sha256':sha}),flush=True)
p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
p.stdin.write(json.dumps(payload,separators=(',',':')));p.stdin.close()
for line in p.stdout:print(line.rstrip(),flush=True)
err=p.stderr.read();rc=p.wait()
if rc:raise SystemExit('SSH_RENDER_EXIT_'+str(rc)+' '+err[-3000:])
print('PHASE_DONE',args.phase,len(chosen),flush=True)
