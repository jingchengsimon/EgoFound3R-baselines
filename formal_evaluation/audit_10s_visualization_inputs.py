"""Read-only registered OSS predictions and workspace RGB for fixed 10s segments."""

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
import hashlib,json,sys,zipfile
from pathlib import Path
p=json.loads(sys.argv[1]); ds=p['dataset']; errors=[]; methods={name:0 for name in p['roots']}; rgb_count=0; windows_count=0; gt_count=0
rows=[]
for index in p['rgb_indices']:
    rows.extend(json.loads(line) for line in Path(index).read_text().splitlines() if line.strip())
by_id={}
for row in rows:
    for key in (row.get('window_id'),row.get('cache_id'),Path(row.get('window_input','')).parent.name):
        if key:by_id.setdefault(key,[]).append(row)
seen=set()
for segment in p['segments']:
    for i,win in enumerate(segment['windows']):
        key=(win['cache_id'],win['window_id'])
        if key in seen:
            errors.append({'segment':segment['segment_id'],'window':win['window_id'],'error':'duplicate selected window'});continue
        seen.add(key);windows_count+=1
        try:
            matches=by_id.get(win['window_id']) or by_id.get(win['cache_id'])
            if not matches or len(matches)!=1:raise ValueError('RGB_INDEX_MATCH_COUNT:'+str(len(matches or [])))
            row=matches[0]
            rec_path=row.get('window_input',row.get('window_input_path',row.get('input_path',row.get('record_path'))))
            if not rec_path:raise ValueError('RGB_RECORD_PATH_MISSING')
            record=json.loads(Path(rec_path).read_text())
            ids=record['frame_ids'];digest=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
            if record['dataset']!=ds or digest!=win['frame_ids_sha256'] or len(record['rgb_paths'])!=60:raise ValueError('RGB_FRAME_IDENTITY_MISMATCH')
            for target in segment['rgb_samples']:
                if target['window_index']!=i:continue
                t=target['frame_index'];path=Path(record['rgb_paths'][t])
                if ids[t]!=target['frame_id'] or not path.is_file() or path.stat().st_size==0:raise ValueError('RGB_SAMPLE_MISSING_OR_MISMATCH:'+str(path))
                rgb_count+=1
            for method,roots in p['roots'].items():
                candidates=[Path(root)/win['cache_id'] for root in roots]
                found=[path for path in candidates if (path/'predictions.npz').is_file() and (path/'metadata.json').is_file()]
                if len(found)!=1:raise ValueError(method+'_PREDICTION_MATCH_COUNT:'+str(len(found)))
                md=json.loads((found[0]/'metadata.json').read_text())
                if md['dataset']!=ds or md['frame_ids']!=ids or md['method']!=method:raise ValueError(method+'_METADATA_IDENTITY_MISMATCH')
                npz=found[0]/'predictions.npz'
                if npz.stat().st_size==0 or not zipfile.is_zipfile(npz):raise ValueError(method+'_PREDICTION_INVALID')
                methods[method]+=1
            if p.get('check_common_all'):
                gt=Path(win['gt_path'])
                if not gt.is_file() or gt.stat().st_size==0 or not zipfile.is_zipfile(gt):raise ValueError('GT_ARRAY_MISSING_OR_INVALID')
                gt_count+=1
        except Exception as error:
            errors.append({'segment':segment['segment_id'],'window':win['window_id'],'error':str(error)[:240]})
print(json.dumps({'dataset':ds,'segments':len(p['segments']),'windows_checked':windows_count,'method_windows':methods,'gt_windows':gt_count,'rgb_sample_frames':rgb_count,'error_count':len(errors),'errors':errors[:20],'ok':not errors and windows_count==len(p['segments'])*5 and all(v==windows_count for v in methods.values()) and (not p.get('check_common_all') or gt_count==windows_count) and rgb_count==len(p['segments'])*3}))
'''

REMOTE_EXPORT = r'''
import base64,hashlib,io,json,sys,zipfile
from pathlib import Path
p=json.loads(sys.argv[1]);s=p['segments'][0];ds=p['dataset'];rows=[];evidence=[]
for index in p['rgb_indices']:
    rows.extend(json.loads(line) for line in Path(index).read_text().splitlines() if line.strip())
out=io.BytesIO()
with zipfile.ZipFile(out,'w',zipfile.ZIP_STORED) as z:
    for i,win in enumerate(s['windows']):
        matches=[r for r in rows if r.get('window_id')==win['window_id'] or r.get('cache_id')==win['cache_id'] or Path(r.get('window_input','')).parent.name==win['cache_id']]
        assert len(matches)==1,(win['window_id'],len(matches))
        row=matches[0];rec_path=row.get('window_input',row.get('window_input_path',row.get('input_path',row.get('record_path'))))
        record=json.loads(Path(rec_path).read_text());ids=record['frame_ids']
        assert record['dataset']==ds and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==win['frame_ids_sha256']
        for method,roots in p['roots'].items():
            found=[Path(root)/win['cache_id'] for root in roots if (Path(root)/win['cache_id']/'predictions.npz').is_file() and (Path(root)/win['cache_id']/'metadata.json').is_file()]
            assert len(found)==1,(method,win['window_id'],len(found))
            metadata=json.loads((found[0]/'metadata.json').read_text())
            assert metadata['dataset']==ds and metadata['method']==method and metadata['frame_ids']==ids
            for suffix,source in (('.npz',found[0]/'predictions.npz'),('_metadata.json',found[0]/'metadata.json')):
                data=source.read_bytes();name=f'{i}_{method}'+suffix;z.writestr(name,data)
                evidence.append({'name':name,'source_path':str(source),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        if p.get('include_common'):
            source=Path(win['gt_path']);assert source.is_file() and source.stat().st_size>0 and zipfile.is_zipfile(source)
            data=source.read_bytes();name=f'{i}_gt.npz';z.writestr(name,data)
            evidence.append({'name':name,'source_path':str(source),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        for target in s['rgb_samples']:
            if target['window_index']!=i:continue
            frame_index=target['frame_index'];source=Path(record['rgb_paths'][frame_index])
            assert ids[frame_index]==target['frame_id'] and source.is_file()
            data=source.read_bytes();assert data
            name=f"rgb_{target['global_index']:03d}"+source.suffix.lower();z.writestr(name,data)
            evidence.append({'name':name,'source_path':str(source),'frame_id':ids[frame_index],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    z.writestr('sources.json',json.dumps(evidence,indent=2))
sys.stdout.write(base64.b64encode(out.getvalue()).decode())
'''


def audit(catalog_run, source_run, manifest_path, ssh, *, export_segment_id=None, output_dir=None, include_common=False, check_common_all=False):
    catalog = catalog_run.get("artifact_catalog")
    if not catalog or catalog["node"] != 5000:
        raise ValueError("expected a registered 5000 baseline artifact catalog")
    ds = catalog["dataset"]
    if source_run.get("task_type") != "evaluation" or str(source_run.get("identity", {}).get("node")) not in ("5000", "5001"):
        raise ValueError("RGB source is not the registered evaluation task")
    selected = [json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip()]
    rows = [row for row in selected if row["dataset"] == ds]
    if not rows or any(len(row["windows"]) != 5 for row in rows):
        raise ValueError("no five-window segments for catalog dataset")
    indices = rows[0]["rgb_input_indices"]
    if any(row["rgb_input_indices"] != indices or row["rgb_source_task"] != source_run["logical_task_id"] for row in rows):
        raise ValueError("RGB source differs across selected segments")
    launch = json.dumps(source_run.get("launch", {}))
    if not all(path.startswith("/mnt/workspace/") and path in launch for path in indices):
        raise ValueError("RGB input index is not in the registered evaluation launch")
    roots = {method: catalog["predictions"][method]["formal_roots"] for method in ("pad_hand", "reviv4d")}
    if include_common or check_common_all:
        if include_common and not export_segment_id:
            raise ValueError("common geometry export requires an exact segment")
        roots.update({method: catalog["predictions"][method]["formal_roots"] for method in ("wilor", "hawor")})
    if any(not root.startswith("/mnt/oss/") for values in roots.values() for root in values):
        raise ValueError("baseline source is outside the registered OSS mount")
    segments = []
    for row in rows:
        windows = []
        for win in row["windows"]:
            ids = win["gt"]["frame_ids"]
            gt_path = win["gt_path"]
            gt_root = str(Path(catalog["gt_index"]).parent) + "/"
            if (include_common or check_common_all) and (not gt_path.startswith(gt_root) or Path(gt_path).stem != win["gt"]["cache_id"]):
                raise ValueError("GT path is outside the exact registered cache")
            windows.append({"cache_id": win["gt"]["cache_id"], "window_id": win["window_id"], "gt_path": gt_path,
                            "frame_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()})
        samples = [{"global_index": t, "window_index": t // 60, "frame_index": t % 60,
                    "frame_id": row["frame_ids"][t]} for t in (0, 149, 299)]
        segments.append({"segment_id": row["segment_id"], "windows": windows, "rgb_samples": samples})
    if export_segment_id:
        segments = [segment for segment in segments if segment["segment_id"] == export_segment_id]
        if len(segments) != 1 or output_dir is None:
            raise ValueError("export requires one exact selected segment and a local output directory")
    payload = {"dataset": ds, "rgb_indices": indices, "roots": roots,
               "include_common": include_common, "check_common_all": check_common_all}
    def run_remote(batch):
        remote_payload = {**payload, "segments": batch}
        remote = "python3 -c " + shlex.quote(REMOTE_EXPORT if export_segment_id else REMOTE) + " " + shlex.quote(json.dumps(remote_payload))
        completed = ssh(catalog_run, 5000, remote, timeout=1200)
        if completed.returncode:
            raise RuntimeError("VISUALIZATION_INPUT_AUDIT_FAILED:" + completed.stderr[-1000:])
        return completed.stdout
    if export_segment_id:
        output = run_remote(segments)
        root = Path(output_dir)
        if root.exists():
            raise FileExistsError(str(root))
        root.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(output, validate=True)
        with tempfile.TemporaryDirectory(prefix="visualization-incoming-", dir=root.parent) as temporary:
            staging = Path(temporary) / "inputs"
            staging.mkdir()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                assert z.testzip() is None
                for name in z.namelist():
                    if Path(name).name != name:
                        raise ValueError("unsafe archive member")
                    (staging / name).write_bytes(z.read(name))
            evidence = json.loads((staging / "sources.json").read_text())
            expected_files = 48 if include_common else 23
            if len(evidence) != expected_files or any(hashlib.sha256((staging / item["name"]).read_bytes()).hexdigest() != item["sha256"] for item in evidence):
                raise ValueError("export evidence mismatch")
            os.replace(staging, root)
        return {"ok": True, "dataset": ds, "segment_id": export_segment_id, "output_dir": str(root),
                "files": len(evidence), "archive_bytes": len(data), "catalog_run_id": catalog_run["run_id"]}
    parts = [json.loads(run_remote(segments[start:start + 10])) for start in range(0, len(segments), 10)]
    result = {"dataset": ds, "segments": sum(part["segments"] for part in parts),
              "windows_checked": sum(part["windows_checked"] for part in parts),
              "gt_windows": sum(part["gt_windows"] for part in parts),
              "rgb_sample_frames": sum(part["rgb_sample_frames"] for part in parts),
              "method_windows": {method: sum(part["method_windows"][method] for part in parts) for method in roots},
              "error_count": sum(part["error_count"] for part in parts),
              "errors": [error for part in parts for error in part["errors"]][:20],
              "ok": all(part["ok"] for part in parts)}
    result.update(catalog_run_id=catalog_run["run_id"], rgb_source_run_id=source_run["run_id"])
    return result
