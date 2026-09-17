"""Read-only export of exact registered stride5 Ego windows to local inputs."""

import base64
import hashlib
import io
import json
import os
import shlex
import tempfile
import zipfile
from pathlib import Path


REMOTE = r'''
import base64,hashlib,io,json,sys,zipfile
from pathlib import Path
p=json.loads(sys.argv[1]);evidence=[];out=io.BytesIO()
with zipfile.ZipFile(out,'w',zipfile.ZIP_STORED) as archive:
    for i,win in enumerate(p['windows']):
        directory=Path(win['prediction_dir']);npz=directory/'predictions.npz';meta=directory/'metadata.json'
        assert npz.is_file() and meta.is_file()
        md=json.loads(meta.read_text())
        assert md['dataset']==p['dataset'] and md['method']=='egofound3r' and md['global_stride']==5
        assert hashlib.sha256(json.dumps(md['frame_ids'],separators=(',',':')).encode()).hexdigest()==win['frame_ids_sha256']
        slim=io.BytesIO()
        with zipfile.ZipFile(npz) as source, zipfile.ZipFile(slim,'w',zipfile.ZIP_DEFLATED) as output:
            for key in ('hand_valid','hand_joints_camera','hand_markers_camera','camera_c2w','camera_valid'):
                name=key+'.npy';output.writestr(name,source.read(name))
        for name,data,source_path in ((f'{i}_ego.npz',slim.getvalue(),npz),(f'{i}_ego_metadata.json',meta.read_bytes(),meta)):
            archive.writestr(name,data)
            evidence.append({'name':name,'source_path':str(source_path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    archive.writestr('sources_ego.json',json.dumps(evidence,indent=2))
sys.stdout.write(base64.b64encode(out.getvalue()).decode())
'''


REMOTE_AUDIT = r'''
import json,sys
from pathlib import Path
p=json.loads(sys.argv[1]);root=Path(p['output_root']);entry=Path('/mnt/cpfs/sjc/eval_artifacts')
rows=[]
for win in p['windows']:
    directory=Path(win['prediction_dir'])
    rows.append({'prediction_dir':str(directory),'directory_exists':directory.is_dir(),
                 'predictions_exists':(directory/'predictions.npz').is_file(),
                 'metadata_exists':(directory/'metadata.json').is_file()})
print(json.dumps({'cpfs_is_mount':Path('/mnt/cpfs').is_mount(),
                  'reader_entrance':p['reader_entrance'],
                  'reader_entrance_is_mount':Path(p['reader_entrance']).is_mount(),
                  'source_entry_is_symlink':entry.is_symlink(),
                  'source_entry_target':str(entry.readlink()) if entry.is_symlink() else None,
                  'source_entry_target_exists':entry.exists(),
                  'run_root_exists':root.is_dir(),'windows':rows}))
'''

REMOTE_AUDIT_ALL = r'''
import hashlib,json,sys
from pathlib import Path
p=json.loads(sys.argv[1]);out={'segments':len(p['segment_ids']),'windows':0,'source_bytes':0,'errors':[]}
out['reader_entrance_is_mount']=Path(p['reader_entrance']).is_mount()
out['run_root_exists']=Path(p['output_root']).is_dir()
for win in p['windows']:
    directory=Path(win['prediction_dir']);npz=directory/'predictions.npz';meta=directory/'metadata.json'
    try:
        if not npz.is_file() or not meta.is_file():raise ValueError('PREDICTION_PAIR_MISSING')
        md=json.loads(meta.read_text())
        if md.get('dataset')!=p['dataset'] or md.get('method')!='egofound3r' or md.get('global_stride')!=5:
            raise ValueError('METHOD_OR_DATASET_MISMATCH')
        digest=hashlib.sha256(json.dumps(md['frame_ids'],separators=(',',':')).encode()).hexdigest()
        if digest!=win['frame_ids_sha256']:raise ValueError('FRAME_IDS_MISMATCH')
        size=npz.stat().st_size
        if size<=0:raise ValueError('EMPTY_PREDICTION')
        out['windows']+=1;out['source_bytes']+=size
    except Exception as error:
        out['errors'].append({'segment_id':win['segment_id'],'cache_id':directory.name,'error':str(error)})
print(json.dumps(out))
'''


def _segment_payload(run, manifest_path, segment_id, reader_node, selected=None):
    if reader_node is None or not str(run.get("output_root", "")).startswith("/mnt/cpfs/"):
        raise ValueError("exact registered CPFS run and reader required")
    if selected is None:
        selected = [json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip()]
    match = [row for row in selected if row["segment_id"] == segment_id]
    if len(match) != 1:
        raise ValueError("segment is not unique in the frozen manifest")
    row = match[0]
    if row["rgb_source_task"] != run["logical_task_id"] or len(row["windows"]) != 5:
        raise ValueError("segment does not belong to this registered evaluation run")
    windows = []
    for item in row["windows"]:
        directory = item["pred"]["prediction_dir"]
        cache_id = item["gt"]["cache_id"]
        if not directory.startswith(run["output_root"].rstrip("/") + "/") or Path(directory).name != cache_id:
            raise ValueError("Ego directory is outside the registered run")
        windows.append({"prediction_dir": directory,
                        "frame_ids_sha256": hashlib.sha256(json.dumps(item["gt"]["frame_ids"], separators=(",", ":")).encode()).hexdigest()})
    payload = {"dataset": row["dataset"], "windows": windows, "output_root": run["output_root"],
               "reader_entrance": run.get("_current_instance", {}).get("entrance", "/mnt/cpfs")}
    return row, payload


def audit(run, manifest_path, segment_id, reader_node, ssh):
    row, payload = _segment_payload(run, manifest_path, segment_id, reader_node)
    remote = "python3 -c " + shlex.quote(REMOTE_AUDIT) + " " + shlex.quote(json.dumps(payload))
    completed = ssh(run, reader_node, remote, timeout=120)
    if completed.returncode:
        raise RuntimeError("REGISTERED_EGO_AUDIT_FAILED:" + completed.stderr[-1000:])
    result = json.loads(completed.stdout)
    result.update(segment_id=segment_id, dataset=row["dataset"], run_id=run["run_id"], reader_node=reader_node)
    result["ok"] = result["reader_entrance_is_mount"] and result["run_root_exists"] and all(
        item["predictions_exists"] and item["metadata_exists"] for item in result["windows"])
    return result


def audit_all(run, manifest_path, reader_node, ssh):
    selected = [json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip()]
    matches = [row for row in selected if row["rgb_source_task"] == run["logical_task_id"]]
    if not matches:
        raise ValueError("no selected segments for registered evaluation run")
    segment_ids = [row["segment_id"] for row in matches]
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError("duplicate segment in frozen manifest")
    payloads = [_segment_payload(run, manifest_path, segment_id, reader_node, selected)[1]
                for segment_id in segment_ids]
    payload = {"dataset": matches[0]["dataset"], "output_root": run["output_root"],
               "reader_entrance": payloads[0]["reader_entrance"], "segment_ids": segment_ids,
               "windows": [{**win, "segment_id": segment_id}
                           for segment_id, item in zip(segment_ids, payloads) for win in item["windows"]]}
    remote = "python3 -c " + shlex.quote(REMOTE_AUDIT_ALL) + " " + shlex.quote(json.dumps(payload))
    completed = ssh(run, reader_node, remote, timeout=600)
    if completed.returncode:
        raise RuntimeError("REGISTERED_EGO_BATCH_AUDIT_FAILED:" + completed.stderr[-1000:])
    result = json.loads(completed.stdout)
    result.update(dataset=payload["dataset"], run_id=run["run_id"], reader_node=reader_node,
                  expected_windows=len(payload["windows"]))
    result["ok"] = (result["reader_entrance_is_mount"] and result["run_root_exists"] and
                    result["windows"] == result["expected_windows"] and not result["errors"])
    return result


def export(run, manifest_path, segment_id, output_dir, reader_node, ssh):
    row, payload = _segment_payload(run, manifest_path, segment_id, reader_node)
    remote = "python3 -c " + shlex.quote(REMOTE) + " " + shlex.quote(json.dumps(payload))
    completed = ssh(run, reader_node, remote, timeout=1200)
    if completed.returncode:
        raise RuntimeError("REGISTERED_EGO_EXPORT_FAILED:" + completed.stderr[-1000:])
    data = base64.b64decode(completed.stdout, validate=True)
    root = Path(output_dir)
    if not root.is_dir() or any((root / f"{i}_ego.npz").exists() for i in range(5)):
        raise ValueError("new local common-input directory without Ego files required")
    with tempfile.TemporaryDirectory(prefix="ego-incoming-", dir=root.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.testzip() is not None:
                raise ValueError("Ego ZIP CRC failure")
            for name in archive.namelist():
                if Path(name).name != name:
                    raise ValueError("unsafe Ego ZIP member")
                (staging / name).write_bytes(archive.read(name))
        evidence = json.loads((staging / "sources_ego.json").read_text())
        if len(evidence) != 10 or any(hashlib.sha256((staging / item["name"]).read_bytes()).hexdigest() != item["sha256"] for item in evidence):
            raise ValueError("Ego evidence mismatch")
        for item in evidence:
            os.replace(staging / item["name"], root / item["name"])
        os.replace(staging / "sources_ego.json", root / "sources_ego.json")
    return {"ok": True, "segment_id": segment_id, "dataset": row["dataset"],
            "files": len(evidence), "output_dir": str(root), "archive_bytes": len(data),
            "run_id": run["run_id"], "reader_node": reader_node}
