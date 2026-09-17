import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/mnt/workspace/sjc/eval_artifacts/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914')
SEL = ROOT / 'selected_manifest.jsonl'
EXPECTED_SHA = '6a95a1c01288fbe91589345e013bd7c96bcb038ebef2414bea140240824f4d2f'
COMMIT = '8fc061a615895bd3b5a556f7387bae306e32d9db'
EXPECTED_DATASETS = {'arctic': 9, 'h2o': 26, 'hoi4d': 0, 'hot3d': 1, 'oakink_v2': 18, 'taco': 60}

errors = []
raw = SEL.read_bytes()
if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA:
    errors.append('selected_manifest sha256 mismatch')
selected = [json.loads(line) for line in raw.splitlines() if line.strip()]
if len(selected) != 114:
    errors.append(f'selected count {len(selected)} != 114')

reports = [json.loads(p.read_text()) for p in sorted((ROOT / 'reports').glob('*.json'))]
report_by_key = {(r['dataset'], r['dataset_rank']): r for r in reports}
if len(report_by_key) != len(reports):
    errors.append('duplicate dataset/rank reports')

counts = {
    'png': len(list((ROOT / 'png_gallery').glob('*.png'))),
    'mp4': len(list((ROOT / 'video_gallery').glob('*.mp4'))),
    'frame_manifest': len(list((ROOT / 'frame_manifests').glob('*.jsonl'))),
    'report': len(reports),
}
for kind, value in counts.items():
    if value != 114:
        errors.append(f'{kind} count {value} != 114')

dataset_counts = Counter()
dataset_frames = Counter()
render_rows = []
filled_clips = []
center_filled = []
total_frames = 0
total_decoded = 0
total_filled = 0
intervals = defaultdict(list)

for clip in selected:
    key = (clip['dataset'], clip['dataset_rank'])
    report = report_by_key.get(key)
    if report is None:
        errors.append(f'missing report {key}')
        continue
    dataset_counts[clip['dataset']] += 1
    dataset_frames[clip['dataset']] += len(clip['frame_ids'])
    intervals[(clip['dataset'], clip['sequence_id'])].append(
        (clip['actual_start_position'], clip['actual_end_position_exclusive'], clip['dataset_rank']))
    total_frames += len(clip['frame_ids'])
    total_decoded += report['decoded_frames']
    total_filled += report['filled_side_frames']
    if report['filled_side_frames']:
        filled_clips.append({'dataset': clip['dataset'], 'dataset_rank': clip['dataset_rank'], 'filled_side_frames': report['filled_side_frames'], 'missing_runs': report['missing_runs']})
    if any(report['center_fill_sides']):
        center_filled.append({'dataset': clip['dataset'], 'dataset_rank': clip['dataset_rank'], 'center_frame_id': clip['frame_id'], 'sides': report['center_fill_sides']})

    manifest_path = Path(report['frame_manifest'])
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    source_ids = [row['source_frame_id'] for row in rows]
    if source_ids != clip['frame_ids']:
        errors.append(f'frame identity mismatch {key}')
    if [row['display_index'] for row in rows] != list(range(len(rows))):
        errors.append(f'display index mismatch {key}')
    if len(rows) != clip['actual_length'] or report['frame_count'] != len(rows):
        errors.append(f'frame length mismatch {key}')
    if report['first_frame_id'] != clip['frame_ids'][0] or report['last_frame_id'] != clip['frame_ids'][-1]:
        errors.append(f'first/last mismatch {key}')
    if clip['frame_ids'][clip['center_index_in_clip']] != clip['frame_id']:
        errors.append(f'center identity mismatch {key}')
    if report['center_index'] != clip['center_index_in_clip']:
        errors.append(f'center index mismatch {key}')
    fill_count = sum(sum(bool(v) for v in row['ego_hand_missing_fill']) for row in rows)
    if fill_count != report['filled_side_frames']:
        errors.append(f'fill count mismatch {key}')
    if rows and (any(rows[0]['ego_hand_missing_fill']) or any(rows[-1]['ego_hand_missing_fill'])):
        errors.append(f'endpoint filled {key}')
    if any(row['interpolation_commit'] != COMMIT for row in rows):
        errors.append(f'interpolation commit mismatch {key}')

    video = Path(report['video'])
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames',
        '-of', 'json', str(video)
    ]))['streams'][0]
    decoded = int(probe['nb_read_frames'])
    if decoded != len(rows) or probe['avg_frame_rate'] != '30/1':
        errors.append(f'ffprobe mismatch {key}: frames={decoded}, fps={probe["avg_frame_rate"]}')
    png = Path(report['png'])
    if not png.is_file() or png.stat().st_size == 0 or not video.is_file() or video.stat().st_size == 0:
        errors.append(f'empty media output {key}')

    render_rows.append({
        'dataset': clip['dataset'], 'dataset_rank': clip['dataset_rank'],
        'sequence_id': clip['sequence_id'], 'center_frame_id': clip['frame_id'],
        'center_index': clip['center_index_in_clip'], 'first_frame_id': clip['frame_ids'][0],
        'last_frame_id': clip['frame_ids'][-1], 'frame_count': len(rows), 'fps': 30,
        'png': str(png.relative_to(ROOT)), 'mp4': str(video.relative_to(ROOT)),
        'frame_manifest': str(manifest_path.relative_to(ROOT)),
        'report': str((ROOT / 'reports' / (video.stem + '.json')).relative_to(ROOT)),
        'filled_side_frames': report['filled_side_frames'], 'missing_runs': report['missing_runs'],
        'center_fill_sides': report['center_fill_sides'],
    })

for key, group in intervals.items():
    group.sort()
    for left, right in zip(group, group[1:]):
        if left[1] > right[0]:
            errors.append(f'overlap {key}: ranks {left[2]} and {right[2]}')

if dict(dataset_counts) != {k: v for k, v in EXPECTED_DATASETS.items() if v}:
    errors.append(f'dataset counts mismatch {dict(dataset_counts)}')
if total_frames != 12559 or total_decoded != 12559:
    errors.append(f'total frames mismatch listed={total_frames} decoded={total_decoded}')
if total_filled != 310:
    errors.append(f'total filled side frames {total_filled} != 310')
if len(filled_clips) != 8:
    errors.append(f'filled clip count {len(filled_clips)} != 8')
if len(center_filled) != 2:
    errors.append(f'center-filled clip count {len(center_filled)} != 2')

audit = {
    'schema': 'overlay-2d-visibility60-max-nonoverlap114-render-audit-v1',
    'status': 'verified' if not errors else 'failed',
    'selection_sha256': EXPECTED_SHA,
    'selection_count': len(selected),
    'output_counts': counts,
    'per_dataset_clips': {k: dataset_counts.get(k, 0) for k in EXPECTED_DATASETS},
    'per_dataset_display_frames': {k: dataset_frames.get(k, 0) for k in EXPECTED_DATASETS},
    'display_frames_manifest': total_frames,
    'display_frames_decoded': total_decoded,
    'fps': 30,
    'nonoverlap_verified': not any(e.startswith('overlap ') for e in errors),
    'filled_clips': filled_clips,
    'filled_clip_count': len(filled_clips),
    'filled_hand_side_frames': total_filled,
    'center_filled': center_filled,
    'interpolation_commit': COMMIT,
    'interpolation_scope': 'visualization-only exact-equivalent camera-space fill and local-shape smoothing; source predictions and strict metric validity unchanged; no inference rerun or root-UV re-solve',
    'errors': errors,
}
(ROOT / 'render_manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in render_rows))
(ROOT / 'completeness_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')

contract_path = ROOT / 'run_contract.json'
contract = json.loads(contract_path.read_text())
contract['status'] = audit['status']
contract['verified_counts'] = counts
contract['verified_display_frames'] = total_decoded
contract['verified_filled_hand_side_frames'] = total_filled
contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + '\n')

readme = f'''# 2D overlay: visibility-60 maximum non-overlap 114

Status: **{audit['status']}**

- Frozen selection: 114 center frames; selection SHA-256 `{EXPECTED_SHA}`.
- Criterion: both hands valid at the first and last display frame; each internal combined missing run is at most 60 frames; then maximum non-overlapping interval selection per dataset/sequence.
- Outputs: 114 center-frame PNGs and 114 actual-length MP4s at 30 FPS; {total_decoded} decoded display frames in total.
- Dataset clips: ARCTIC 9, H2O 26, HOI4D 0, HOT3D 1, OakInk-v2 18, TACO 60.
- Interpolation: {len(filled_clips)} clips, {total_filled} hand-side frames. Commit `{COMMIT}` visualization postprocess (`_camera_space_smooth_fill` plus root-relative `[1,2,1]/4` local-shape smoothing). Source predictions and strict metric validity remain unchanged; no inference rerun or root-UV re-solve.
- Center PNG interpolation: {len(center_filled)} centers ({', '.join(x['dataset'] + ' rank ' + str(x['dataset_rank']) for x in center_filled)}).
- Layout: 2x4, top GT and bottom Ego stride5 + display fill; columns geometry, contact distance, contact, visibility.
- Contact distance: `vertex_contact_distance`, Turbo scale 0-50 mm. Visibility: predicted `marker_visibility` spatially mapped 195 to 778, threshold 0.5; it is not RGB-observed visibility.
- All videos were re-probed for decoded frame count and 30 FPS; frame sidecars match the frozen source-frame order, center, and first/last identities; non-overlap is verified.

See `selected_manifest.jsonl`, `selection_summary.json`, `render_manifest.jsonl`, and `completeness_audit.json` for machine-readable provenance.
'''
(ROOT / 'README.md').write_text(readme)
print(json.dumps(audit, ensure_ascii=False, indent=2))
raise SystemExit(1 if errors else 0)
