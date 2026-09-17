"""Reaggregate EgoFound3R stride5 on Dyn-HaMR and InteractVLM original100 windows.

All source metrics were computed before selection. This worker only selects exact
window identities and applies the existing finite-window aggregation protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from formal_evaluation.common.aggregation import aggregate_windows


HAND_STEMS = {
    'joint': ('mpjpe', 'rr_mpjpe', 'pa_mpjpe', 'global_sim3_mpjpe',
              'w_mpjpe', 'wa_mpjpe', 'mpjve', 'mpjae'),
    'marker': ('marker_mpmpe', 'marker_rr_mpmpe', 'marker_pa_mpmpe',
               'marker_global_sim3_mpmpe', 'marker_w_mpmpe',
               'marker_wa_mpmpe', 'marker_mpmve', 'marker_mpmae'),
    'vertex': ('vertex_mpvpe', 'vertex_rr_mpvpe', 'vertex_pa_mpvpe',
               'vertex_global_sim3_mpvpe', 'vertex_w_mpvpe',
               'vertex_wa_mpvpe', 'vertex_mpvve', 'vertex_mpvae'),
}
CONTACT_STEMS = ('contact_precision', 'contact_recall', 'contact_f1',
                 'visibility_precision', 'visibility_recall', 'visibility_f1',
                 'contact_distance_all_valid_mm',
                 'contact_distance_predicted_contact_mm')


def file_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checked_rows(path: Path, sha256: str) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f'SHA256 mismatch: {path}')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def selected_predictions(paths: list[str], gt: dict[str, list], aliases: dict[str, str],
                         dataset: str, expected: int) -> tuple[set[str], dict[str, str]]:
    selected = set()
    hashes = {}
    for source in paths:
        index = Path(source)
        parent = index.parent
        if not (parent / 'COMPLETE').is_file():
            raise ValueError(f'prediction source incomplete: {parent}')
        rows = file_rows(index)
        hashes[source] = hashlib.sha256(index.read_bytes()).hexdigest()
        for row in rows:
            if row.get('dataset', dataset) != dataset:
                raise ValueError(f'prediction dataset mismatch: {source}')
            key = aliases.get(str(row['window_id']), str(row['window_id']))
            if key not in gt or key in selected:
                raise ValueError(f'prediction duplicate/unregistered window: {dataset}:{key}')
            directory = Path(row['prediction_dir'])
            metadata = json.loads((directory / 'metadata.json').read_text())
            if metadata['frame_ids'] != gt[key]:
                raise ValueError(f'prediction frame identity mismatch: {dataset}:{key}')
            selected.add(key)
    if len(selected) != expected:
        raise ValueError(f'prediction selection coverage: {dataset}: {len(selected)}/{expected}')
    return selected, hashes


def keyed(rows: list[dict], expected: dict[str, list], *, dataset: str,
          frame_ids: bool) -> dict[str, dict]:
    found = {}
    for row in rows:
        if row.get('dataset', dataset) != dataset:
            raise ValueError(f'dataset mismatch: {dataset}')
        key = str(row['window_id'])
        if key in found or key not in expected:
            raise ValueError(f'duplicate or unexpected window: {dataset}:{key}')
        if frame_ids and row['frame_ids'] != expected[key]:
            raise ValueError(f'frame identity mismatch: {dataset}:{key}')
        found[key] = row
    if set(found) != set(expected):
        raise ValueError(f'full source coverage mismatch: {dataset}: {len(found)}/{len(expected)}')
    return found


def summarize(rows: list[dict], *, method: str) -> dict:
    numeric = {key for row in rows for key, value in row.items()
               if isinstance(value, (int, float)) or value is None}
    normalized = [{key: (float('nan') if row.get(key) is None else row[key])
                   for key in numeric if key in row} for row in rows]
    return aggregate_windows(normalized, method=method)


def combined_hand(summary: dict, stem: str) -> float:
    weighted = count = 0
    for side in ('left', 'right'):
        prefix = f'hand_{side}_{stem}'
        mean = summary.get(prefix + '_mean', float('nan'))
        n = summary.get(prefix + '_count', 0)
        if n and math.isfinite(mean):
            weighted += mean * n
            count += n
    return weighted / count / (1000 if stem.endswith('ae') else 1) if count else float('nan')


def complete_source(root: Path, *, status: bool = True, windows: int | None = None) -> str:
    if not (root / 'COMPLETE').is_file():
        raise ValueError(f'source COMPLETE absent: {root}')
    summary = json.loads((root / 'summary.json').read_text())
    if status and summary.get('status') != 'complete':
        raise ValueError(f'source summary incomplete: {root}')
    if windows is not None and summary.get('windows') != windows:
        raise ValueError(f'source window count mismatch: {root}')
    report_path = root / 'report.json'
    if report_path.is_file():
        raw = report_path.read_bytes()
        report = json.loads(raw)
        if report.get('status') != 'complete' or (windows is not None and report.get('windows') != windows):
            raise ValueError(f'source report incomplete: {root}')
        digest = hashlib.sha256(raw).hexdigest()
        if summary.get('report_sha256') and summary['report_sha256'] != digest:
            raise ValueError(f'source report SHA256 mismatch: {root}')
        return digest
    return hashlib.sha256((root / 'summary.json').read_bytes()).hexdigest()


def run(spec: dict, root: Path) -> dict:
    total = sum(d['original100_windows'] for d in spec['datasets'].values())
    if total != 100:
        raise ValueError('matched selection must contain 100 windows')
    roots = {name: Path(path) for name, path in spec['source_roots'].items()}
    source_hashes = {
        name: complete_source(path, status=name != 'distance',
                              windows=2378 if name in ('classification', 'visibility', 'distance') else None)
        for name, path in roots.items()
    }
    hand_summary = json.loads((roots['hand'] / 'summary.json').read_text())
    if not (hand_summary['protocol']['fit_on_full_unfiltered_window']
            and hand_summary['protocol']['same_frame_mask_for_all_aligned_methods']):
        raise ValueError('hand source is not the frozen full-window protocol')
    interact_index = Path(spec['interact_prediction_index'])
    interact_root = interact_index.parent
    if not (interact_root / 'COMPLETE').is_file():
        raise ValueError('InteractVLM source COMPLETE absent')
    interact_report_bytes = (interact_root / 'report.json').read_bytes()
    interact_report = json.loads(interact_report_bytes)
    if (interact_report.get('gt_windows') != 100 or
            interact_report.get('methods', {}).get('interactvlm', {}).get('missing_prediction_windows') != 0):
        raise ValueError('InteractVLM source report incomplete')
    source_hashes['interactvlm'] = hashlib.sha256(interact_report_bytes).hexdigest()
    interact_rows = file_rows(interact_index)
    raw_interact_keys = {(row['dataset'], str(row['window_id'])) for row in interact_rows}
    if len(interact_rows) != 100 or len(raw_interact_keys) != 100:
        raise ValueError('InteractVLM index is not 100 unique windows')

    report = {'status': 'complete', 'method': 'egofound3r_stride5',
              'camera_source': 'predicted_camera_c2w', 'selection': 'Dyn-HaMR hand and InteractVLM contact original100; unfiltered',
              'windows': total, 'source_report_sha256': source_hashes, 'datasets': {}}
    window_index = []
    for dataset, cfg in spec['datasets'].items():
        full_rows = checked_rows(Path(cfg['gt_index']), cfg['gt_sha256'])
        full = {str(row['window_id']): row['frame_ids'] for row in full_rows}
        if len(full) != cfg['ego_windows']:
            raise ValueError(f'full GT index coverage mismatch: {dataset}')
        aliases = {str(row['cache_id']): str(row['window_id']) for row in full_rows if 'cache_id' in row}
        hand_keys, dyn_index_hashes = selected_predictions(cfg['dyn_prediction_indices'], full,
                                                             aliases, dataset, cfg['original100_windows'])
        actual_interact = {}
        for row in interact_rows:
            if row['dataset'] != dataset:
                continue
            key = aliases.get(str(row['window_id']), str(row['window_id']))
            if key in actual_interact or key not in full:
                raise ValueError(f'InteractVLM duplicate/unexpected window: {dataset}:{key}')
            metadata = json.loads((Path(row['prediction_dir']) / 'metadata.json').read_text())
            if (metadata['dataset'] != dataset or
                    aliases.get(str(metadata['window_id']), str(metadata['window_id'])) != key or
                    metadata['frame_ids'] != full[key]):
                raise ValueError(f'InteractVLM frame identity mismatch: {dataset}:{key}')
            actual_interact[key] = row
        if len(actual_interact) != cfg['original100_windows']:
            raise ValueError(f'InteractVLM prediction coverage mismatch: {dataset}')
        contact_keys = set(actual_interact)
        sources = {
            'hand': (file_rows(roots['hand'] / dataset / 'egofound3r_stride5' / 'unfiltered_window_metrics.jsonl'), False),
            'classification': ([r for r in file_rows(roots['classification'] / 'index.jsonl') if r['dataset'] == dataset], True),
            'visibility': ([r for r in file_rows(roots['visibility'] / 'index.jsonl') if r['dataset'] == dataset], True),
            'distance': (file_rows(roots['distance'] / f'{dataset}_windows.jsonl'), True),
        }
        source_rows = {name: keyed(rows, full, dataset=dataset, frame_ids=frames)
                       for name, (rows, frames) in sources.items()}
        hand = summarize([source_rows['hand'][key] for key in sorted(hand_keys)], method='egofound3r_stride5')
        contact_parts = {
            name: summarize([source_rows[name][key]['schemes']['unfiltered'] for key in sorted(contact_keys)],
                            method='egofound3r_stride5')
            for name in ('classification', 'visibility', 'distance')
        }
        hand_values = {grain: {stem: combined_hand(hand, stem) for stem in stems}
                       for grain, stems in HAND_STEMS.items()}
        contact_values = {}
        for grain in HAND_STEMS:
            contact_values[grain] = {}
            for stem in CONTACT_STEMS:
                source = ('visibility' if stem.startswith('visibility_') else
                          'distance' if stem.startswith('contact_distance_') else 'classification')
                key = f'{grain}_{stem}_mean'
                contact_values[grain][stem] = contact_parts[source].get(key, float('nan'))
        report['datasets'][dataset] = {
            'hand_windows': len(hand_keys), 'contact_windows': len(contact_keys),
            'overlap_windows': len(hand_keys & contact_keys), 'gt_index_sha256': cfg['gt_sha256'],
            'dyn_prediction_index_sha256': dyn_index_hashes,
            'hand': hand_values, 'contact': contact_values,
            'undefined_window_counts': {name: {k:v for k,v in summary.items() if k.endswith('_undefined_window_count')}
                                        for name,summary in {'hand':hand,**contact_parts}.items()},
        }
        window_index.extend({'dataset': dataset, 'window_id': key, 'frame_ids': full[key],
                             'hand_selected': key in hand_keys, 'contact_selected': key in contact_keys}
                            for key in sorted(hand_keys | contact_keys))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'windows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in window_index))
    (root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=True))
    summary = {'status': 'complete', 'windows': total,
               'report_sha256': hashlib.sha256((root / 'report.json').read_bytes()).hexdigest(),
               'window_index_sha256': hashlib.sha256((root / 'windows.jsonl').read_bytes()).hexdigest()}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    (root / 'COMPLETE').write_text('complete\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.spec.read_text()), args.output_root)), flush=True)
