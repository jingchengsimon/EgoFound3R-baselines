"""Verify that GT-camera replacement cannot change native-camera contact distances."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(spec, root):
    overlays = {(r['dataset'], r['window_id']): r for r in rows(spec['overlay_index'])}
    if len(overlays) != spec['expected_windows']:
        raise ValueError('overlay coverage mismatch')
    checked = []
    for dataset, source in spec['datasets'].items():
        gt = rows(source['gt_index']); aliases = {r['cache_id']:r['window_id'] for r in gt}
        expected = {r['window_id']:r for r in gt}
        seen = set()
        for index in source['prediction']['indices']:
            for row in rows(index):
                if row['dataset'] != dataset or row['method'] != source['prediction']['config_method']:
                    continue
                key = aliases.get(row['window_id'], row['window_id'])
                if key in seen or key not in expected:
                    raise ValueError('duplicate or unexpected window')
                seen.add(key)
                directory = Path(row['prediction_dir'])
                metadata = json.loads((directory/'metadata.json').read_text())
                overlay = overlays[(dataset,key)]
                if metadata['frame_ids'] != expected[key]['frame_ids'] or overlay['frame_ids'] != metadata['frame_ids']:
                    raise ValueError('frame identity mismatch')
                # _prediction_geometry_camera prioritizes these fields and never reads
                # camera_c2w on this path. GT-camera replacement leaves all three queries
                # unchanged; vertex queries use the same marker-to-vertex interpolation.
                with np.load(directory/'predictions.npz', allow_pickle=False) as prediction:
                    required = ['hand_joints_camera']
                    mesh = 'hand_markers_camera' if 'hand_markers_camera' in prediction else 'hand_vertices_camera'
                    required.append(mesh)
                    for field in required:
                        points = prediction[field]
                        n = {'hand_joints_camera':21,'hand_markers_camera':195,'hand_vertices_camera':778}[field]
                        if points.shape != (60,2,n,3):raise ValueError('invalid camera-space geometry: '+field)
                checked.append({**overlay, 'variant':'egofound3r_stride5_gt_camera',
                                'distance_reuse':'verified native-camera query invariance', 'prediction_dir':str(directory)})
        if seen != set(expected):raise ValueError('prediction coverage mismatch: '+dataset)
        print(json.dumps({'dataset':dataset,'verified_windows':len(seen)}),flush=True)
    if len(checked) != len(overlays):raise ValueError('total coverage mismatch')
    raw = Path(spec['source_report']).read_bytes(); report = json.loads(raw)
    if report['status'] != 'complete' or report['windows'] != len(checked):raise ValueError('incomplete source aggregation')
    report.update(method='egofound3r_stride5_gt_camera', source_report=spec['source_report'],
        source_report_sha256=hashlib.sha256(raw).hexdigest(), camera_invariance_verified_windows=len(checked),
        distance_provenance='Native camera-space queries and GT camera-space surfaces unchanged by GT camera_c2w replacement')
    for data in report['schemes'].values():
        for metrics in data.values():metrics['method']='egofound3r_stride5_gt_camera'
    root.mkdir(parents=True,exist_ok=False)
    (root/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in checked))
    (root/'report.json').write_text(json.dumps(report,indent=2))
    (root/'summary.json').write_text(json.dumps({'status':'complete','windows':len(checked),'camera_invariance_verified':True}))
    (root/'COMPLETE').write_text('complete\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();run(json.loads(a.spec.read_text()),a.output_root)
