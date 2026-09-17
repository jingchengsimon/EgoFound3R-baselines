"""Copy one completed registered 10s gallery to a self-contained local directory."""

import gzip
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


RELATIVE = Path("visualization/batch_10s_177_p95_wmpjpe_20260912")


REMOTE_MANIFEST = r'''import gzip,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);expected_digest=sys.argv[2]
manifest=gzip.decompress((root/'runtime/visualization/batch_10s_177_p95_wmpjpe_20260912/selected_manifest.jsonl.gz').read_bytes())
if hashlib.sha256(manifest).hexdigest()!=expected_digest:raise ValueError('MANIFEST_DIGEST_MISMATCH')
rows=[json.loads(line) for line in manifest.splitlines() if line]
if len(rows)!=177 or not (root/'COMPLETE').is_file():raise ValueError('GALLERY_NOT_COMPLETE')
files={}
for gallery,key,suffix in (('png_gallery','png_filename','.png'),('video_gallery','video_filename','.mp4')):
    names=[row[key] for row in rows]
    if len(set(names))!=177 or set(names)!={p.name for p in (root/gallery).glob('*'+suffix)}:
        raise ValueError('GALLERY_FILE_SET_MISMATCH:'+gallery)
    for name in names:
        path=root/gallery/name
        if not path.is_file():raise ValueError('GALLERY_SOURCE_MISSING:'+name)
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda:source.read(4*1024*1024),b''):digest.update(block)
        files[gallery+'/'+name]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
for name in ('COMPLETE','summary.json','progress.json'):
    path=root/name
    if not path.is_file():raise ValueError('GALLERY_METADATA_MISSING:'+name)
    files[name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({'manifest_sha256':expected_digest,'files':files,'total_bytes':sum(x['bytes'] for x in files.values())}))'''


REMOTE_SAMPLE = r'''import gzip,hashlib,json,subprocess,sys
from pathlib import Path
from PIL import Image
root=Path(sys.argv[1]);relative=sys.argv[2];manifest_name=sys.argv[3];expected_digest=sys.argv[4]
raw=gzip.decompress((root/'runtime'/relative/manifest_name).read_bytes())
if hashlib.sha256(raw).hexdigest()!=expected_digest:raise ValueError('MANIFEST_DIGEST_MISMATCH')
rows=[json.loads(line) for line in raw.splitlines() if line]
png={p.stem:p for p in (root/'png_gallery').glob('*.png')}
mp4={p.stem:p for p in (root/'video_gallery').glob('*.mp4') if '.partial.' not in p.name}
row=next((item for item in rows if item['gallery_stem'] in png and item['gallery_stem'] in mp4),None)
if row is None:raise ValueError('NO_COMPLETED_MEDIA_PAIR')
stem=row['gallery_stem'];report=root/'reports'/(stem+'.json')
if not report.is_file():raise ValueError('REPORT_MISSING:'+stem)
report_data=json.loads(report.read_text())
with Image.open(png[stem]) as image:
    expected_png=tuple(report_data.get('size',(4450,4240)))
    if image.size!=expected_png:raise ValueError('PNG_SIZE:'+str(image.size)+':EXPECTED:'+str(expected_png))
    image.verify()
probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(mp4[stem])],capture_output=True,text=True,check=True)
stream=json.loads(probe.stdout)['streams'][0]
expected_video=tuple(report_data.get('composite_resolution',(2000,1780)))
if stream['codec_name']!='h264' or (stream['width'],stream['height'])!=expected_video or stream['r_frame_rate']!='30/1' or int(stream['nb_frames'])!=300 or abs(float(stream['duration'])-10)>0.02:raise ValueError('MP4_CONTRACT:'+str(stream)+':EXPECTED:'+str(expected_video))
if report_data.get('hawor_label')!='HaWoR native camera (unmasked SLAM)':raise ValueError('HAWOR_LABEL_MISMATCH')
files={}
paths=[png[stem],mp4[stem],report]
panel_count=int(report_data.get('panel_count',0))
if panel_count:
    panels=sorted((root/'panel_gallery'/stem).glob('*.png'))
    if len(panels)!=panel_count:raise ValueError('PANEL_COUNT:'+str(len(panels))+':EXPECTED:'+str(panel_count))
    expected_panel=tuple(report_data.get('panel_size',(2048,2048)))
    for panel in panels:
        with Image.open(panel) as image:
            if image.size!=expected_panel:raise ValueError('PANEL_SIZE:'+str(panel)+':'+str(image.size))
            image.verify()
    paths.extend(panels)
for path in paths:
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):digest.update(block)
    relative_path=str(path.relative_to(root));files[relative_path]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
print(json.dumps({'segment_id':row['segment_id'],'gallery_stem':stem,'dataset':row['dataset'],'sequence_id':row['sequence_id'],'source_frame_start':row['frame_ids'][0],'source_frame_end':row['frame_ids'][-1],'files':files,'video':stream,'hawor_label':report_data['hawor_label'],'hawor_camera':report_data.get('hawor_camera'),'methods':report_data.get('methods'),'external_pose_modes':report_data.get('external_pose_modes'),'dyn_hamr_available_windows':report_data.get('dyn_hamr_available_windows'),'video_view_policy':report_data.get('video_view_policy'),'video_tracking_half_span_m':report_data.get('video_tracking_half_span_m'),'panel_count':panel_count,'panel_size':report_data.get('panel_size'),'total_bytes':sum(item['bytes'] for item in files.values())}))'''


REMOTE_ENDPOINT_MANIFEST = r'''import gzip,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);relative=sys.argv[2];manifest_name=sys.argv[3];expected_digest=sys.argv[4]
raw=gzip.decompress((root/'runtime'/relative/manifest_name).read_bytes())
if hashlib.sha256(raw).hexdigest()!=expected_digest:raise ValueError('MANIFEST_DIGEST_MISMATCH')
rows=[json.loads(line) for line in raw.splitlines() if line]
if len(rows)!=104 or not (root/'COMPLETE').is_file():raise ValueError('ENDPOINT104_GALLERY_NOT_COMPLETE')
files={}
for gallery,suffix in (('png_gallery','.png'),('video_gallery','.mp4')):
    names=[row['gallery_stem']+suffix for row in rows]
    observed={p.name for p in (root/gallery).glob('*'+suffix) if '.partial.' not in p.name}
    if len(set(names))!=104 or set(names)!=observed:raise ValueError('GALLERY_FILE_SET_MISMATCH:'+gallery)
    for name in names:
        path=root/gallery/name;digest=hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda:source.read(4*1024*1024),b''):digest.update(block)
        files[gallery+'/'+name]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
for name in ('COMPLETE','summary.json','preflight.json'):
    path=root/name
    if not path.is_file():raise ValueError('GALLERY_METADATA_MISSING:'+name)
    files[name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({'manifest_sha256':expected_digest,'files':files,'total_bytes':sum(x['bytes'] for x in files.values())}))'''


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(run, project_root: Path, target: Path, ssh_probe):
    identity = run.get("identity", {})
    launch = run.get("launch", {})
    root = str(run.get("output_root", ""))
    endpoint104 = (
        run.get("task_type") == "visualization"
        and identity.get("reader_node") in (5000, 5001)
        and run.get("target_windows_per_method") == 104
        and root.startswith("/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_endpoint104_")
    )
    if endpoint104:
        return _export_endpoint104(run, project_root, target, ssh_probe)
    if (run.get("task_type") != "visualization" or identity.get("reader_node") != 5000
            or run.get("target_windows_per_method") != 177
            or not root.startswith("/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_")
            or not root.endswith("_gallery")):
        raise ValueError("REGISTERED_COMPLETE_10S_GALLERY_REQUIRED")
    parent = project_root / "visualization"
    if target.parent.resolve() != parent.resolve() or target.name != Path(root).name:
        raise ValueError("LOCAL_GALLERY_TARGET_MUST_MATCH_REGISTERED_NAME")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"LOCAL_GALLERY_TARGET_EXISTS:{target}")
    manifest = (project_root / RELATIVE / "selected_manifest.jsonl").read_bytes()
    digest = hashlib.sha256(manifest).hexdigest()
    if digest != identity.get("manifest_sha256"):
        raise ValueError("LOCAL_MANIFEST_IDENTITY_MISMATCH")
    remote = "python3 -c " + shlex.quote(REMOTE_MANIFEST) + " " + shlex.join([root, digest])
    observed = ssh_probe(run, 5000, remote, timeout=1200)
    if observed.returncode:
        raise RuntimeError("REGISTERED_GALLERY_REMOTE_MANIFEST_FAILED:" + observed.stderr[-400:])
    expected = json.loads(observed.stdout)
    files = expected["files"]
    if len(files) != 357 or expected["manifest_sha256"] != digest:
        raise ValueError("REMOTE_GALLERY_MANIFEST_INCOMPLETE")
    free = shutil.disk_usage(parent).free
    if free < expected["total_bytes"] + 1024**3:
        raise OSError(f"LOCAL_GALLERY_CAPACITY_INSUFFICIENT:{free}:{expected['total_bytes']}")
    staging = Path(tempfile.mkdtemp(prefix=".incoming_" + target.name + "_", dir=parent))
    ssh = run["ssh"]
    ssh_command = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                   "-i", os.path.expanduser(str(ssh["key"])), "-p", "5000",
                   f"root@{ssh['host']}",
                   "tar -chf - --exclude='./runtime' --exclude='./control' --exclude='./logs' -C "
                   + shlex.quote(root) + " ."]
    with subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as source:
        with subprocess.Popen(["tar", "-xf", "-", "-C", str(staging)], stdin=source.stdout,
                              stderr=subprocess.PIPE) as extract:
            source.stdout.close()
            extract_error = extract.communicate()[1]
        source_error = source.stderr.read()
        source.wait()
        if source.returncode or extract.returncode:
            raise RuntimeError("GALLERY_TRANSFER_FAILED:" +
                               (source_error + extract_error).decode(errors="replace")[-500:] +
                               f":STAGING:{staging}")
    observed_paths = {str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()}
    if observed_paths != set(files) or any(p.is_symlink() for p in staging.rglob("*")):
        raise ValueError(f"LOCAL_GALLERY_FILE_SET_MISMATCH:STAGING:{staging}")
    for relative, record in files.items():
        path = staging / relative
        if path.stat().st_size != record["bytes"] or _hash(path) != record["sha256"]:
            raise ValueError(f"LOCAL_GALLERY_HASH_MISMATCH:{relative}:STAGING:{staging}")
    (staging / "transfer_manifest.json").write_text(json.dumps({
        "source_run_id": run["run_id"], "source_root": root,
        "manifest_sha256": digest, "files": files,
    }, indent=2) + "\n")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"LOCAL_GALLERY_TARGET_APPEARED:{target}:STAGING:{staging}")
    os.rename(staging, target)
    return {"destination": str(target), "source_run_id": run["run_id"],
            "media_pairs": 177, "copied_files": len(files), "copied_bytes": expected["total_bytes"],
            "manifest_sha256": digest, "verified_sha256": True,
            "free_bytes_before_copy": free}


def _export_endpoint104(run, project_root: Path, target: Path, ssh_probe):
    identity = run["identity"]
    launch = run["launch"]
    root = str(run["output_root"])
    parent = project_root / "visualization"
    if target.parent.resolve() != parent.resolve() or target.name != Path(root).name:
        raise ValueError("LOCAL_GALLERY_TARGET_MUST_MATCH_REGISTERED_NAME")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"LOCAL_GALLERY_TARGET_EXISTS:{target}")
    relative = str(Path(launch["manifest_relative"]).parent)
    manifest_name = Path(launch["manifest_relative"]).name
    local_manifest = project_root / launch["manifest_relative"]
    raw = gzip.decompress(local_manifest.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != identity.get("manifest_sha256"):
        raise ValueError("LOCAL_MANIFEST_IDENTITY_MISMATCH")
    command = "python3 -c " + shlex.quote(REMOTE_ENDPOINT_MANIFEST) + " " + shlex.join(
        [root, relative, manifest_name, digest])
    reader_node = int(identity["reader_node"])
    completed = ssh_probe(run, reader_node, command, timeout=1200)
    if completed.returncode:
        raise RuntimeError("REGISTERED_ENDPOINT104_REMOTE_MANIFEST_FAILED:" + completed.stderr[-500:])
    expected = json.loads(completed.stdout)
    files = expected["files"]
    if len(files) != 211 or expected["manifest_sha256"] != digest:
        raise ValueError("REMOTE_ENDPOINT104_GALLERY_MANIFEST_INCOMPLETE")
    free = shutil.disk_usage(parent).free
    if free < expected["total_bytes"] + 1024**3:
        raise OSError(f"LOCAL_GALLERY_CAPACITY_INSUFFICIENT:{free}:{expected['total_bytes']}")
    staging = Path(tempfile.mkdtemp(prefix=".incoming_" + target.name + "_", dir=parent))
    ssh = run["ssh"]
    remote_files = sorted(files)
    ssh_command = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                   "-i", os.path.expanduser(str(ssh["key"])), "-p", str(reader_node),
                   f"root@{ssh['host']}", "tar -chf - -C " + shlex.quote(root) + " "
                   + " ".join(shlex.quote(path) for path in remote_files)]
    with subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as source:
        with subprocess.Popen(["tar", "-xf", "-", "-C", str(staging)], stdin=source.stdout,
                              stderr=subprocess.PIPE) as extract:
            source.stdout.close()
            extract_error = extract.communicate()[1]
        source_error = source.stderr.read()
        source.wait()
        if source.returncode or extract.returncode:
            raise RuntimeError("GALLERY_TRANSFER_FAILED:" +
                               (source_error + extract_error).decode(errors="replace")[-500:] +
                               f":STAGING:{staging}")
    observed = {str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()}
    if observed != set(files) or any(path.is_symlink() for path in staging.rglob("*")):
        raise ValueError(f"LOCAL_GALLERY_FILE_SET_MISMATCH:STAGING:{staging}")
    for relative_path, record in files.items():
        path = staging / relative_path
        if path.stat().st_size != record["bytes"] or _hash(path) != record["sha256"]:
            raise ValueError(f"LOCAL_GALLERY_HASH_MISMATCH:{relative_path}:STAGING:{staging}")
    (staging / "transfer_manifest.json").write_text(json.dumps({
        "source_run_id": run["run_id"], "source_root": root,
        "manifest_sha256": digest, "files": files, "verified_sha256": True,
    }, indent=2) + "\n")
    os.rename(staging, target)
    return {"destination": str(target), "source_run_id": run["run_id"],
            "media_pairs": 104, "copied_files": len(files),
            "copied_bytes": expected["total_bytes"], "manifest_sha256": digest,
            "verified_sha256": True, "free_bytes_before_copy": free}


def export_sample(run, project_root: Path, target: Path, ssh_probe):
    identity = run.get("identity", {})
    launch = run.get("launch", {})
    root = str(run.get("output_root", ""))
    is_zoom_pilot = (
        run.get("target_windows_per_method") == 1
        and identity.get("visualization_layout") in {
            "hand-focused fixed per-method/view crop",
            "auxiliary methods fixed near-hand crop with individual 2048 panels",
        }
        and bool(identity.get("pilot_segment"))
    )
    if (run.get("task_type") != "visualization" or identity.get("reader_node") not in (5000, 5001)
            or (run.get("target_windows_per_method") != 104 and not is_zoom_pilot)
            or not root.startswith("/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_endpoint104_")):
        raise ValueError("REGISTERED_ENDPOINT104_GALLERY_REQUIRED")
    parent = project_root / "visualization"
    if target.parent.resolve() != parent.resolve() or not target.name.endswith("_sample"):
        raise ValueError("LOCAL_SAMPLE_TARGET_MUST_BE_NEW_VISUALIZATION_SAMPLE_DIR")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"LOCAL_SAMPLE_TARGET_EXISTS:{target}")
    relative = str(Path(launch["manifest_relative"]).parent)
    manifest_name = Path(launch["manifest_relative"]).name
    digest = str(identity["manifest_sha256"])
    command = "python3 -c " + shlex.quote(REMOTE_SAMPLE) + " " + shlex.join(
        [root, relative, manifest_name, digest])
    completed = ssh_probe(run, int(identity["reader_node"]), command, timeout=1200)
    if completed.returncode:
        raise RuntimeError("REGISTERED_GALLERY_SAMPLE_AUDIT_FAILED:" + completed.stderr[-500:])
    expected = json.loads(completed.stdout)
    files = expected["files"]
    free = shutil.disk_usage(parent).free
    if free < expected["total_bytes"] + 256 * 1024**2:
        raise OSError(f"LOCAL_SAMPLE_CAPACITY_INSUFFICIENT:{free}:{expected['total_bytes']}")
    staging = Path(tempfile.mkdtemp(prefix=".incoming_" + target.name + "_", dir=parent))
    ssh = run["ssh"]
    remote_files = sorted(files)
    ssh_command = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                   "-i", os.path.expanduser(str(ssh["key"])), "-p", str(identity["reader_node"]),
                   f"root@{ssh['host']}", "tar -chf - -C " + shlex.quote(root) + " "
                   + " ".join(shlex.quote(path) for path in remote_files)]
    with subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as source:
        with subprocess.Popen(["tar", "-xf", "-", "-C", str(staging)], stdin=source.stdout,
                              stderr=subprocess.PIPE) as extract:
            source.stdout.close()
            extract_error = extract.communicate()[1]
        source_error = source.stderr.read()
        source.wait()
        if source.returncode or extract.returncode:
            raise RuntimeError("GALLERY_SAMPLE_TRANSFER_FAILED:" +
                               (source_error + extract_error).decode(errors="replace")[-500:])
    observed = {str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()}
    if observed != set(files) or any(path.is_symlink() for path in staging.rglob("*")):
        raise ValueError(f"LOCAL_SAMPLE_FILE_SET_MISMATCH:STAGING:{staging}")
    for relative_path, record in files.items():
        path = staging / relative_path
        if path.stat().st_size != record["bytes"] or _hash(path) != record["sha256"]:
            raise ValueError(f"LOCAL_SAMPLE_HASH_MISMATCH:{relative_path}:STAGING:{staging}")
    (staging / "transfer_manifest.json").write_text(json.dumps({
        "source_run_id": run["run_id"], "source_root": root,
        **expected, "verified_sha256": True,
    }, indent=2) + "\n")
    os.rename(staging, target)
    return {"destination": str(target), "source_run_id": run["run_id"],
            "copied_files": len(files), "copied_bytes": expected["total_bytes"],
            "verified_sha256": True, **{key: expected[key] for key in
            ("segment_id", "gallery_stem", "dataset", "sequence_id", "source_frame_start",
             "source_frame_end", "hawor_label", "hawor_camera", "video", "methods",
             "external_pose_modes", "dyn_hamr_available_windows", "video_view_policy",
             "video_tracking_half_span_m", "panel_count", "panel_size")}}
