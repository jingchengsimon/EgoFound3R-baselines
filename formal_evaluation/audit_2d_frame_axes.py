"""Read exact registered input indices; preserve frame identities, never infer boundaries."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        raise FileExistsError('Existing audit output must not be overwritten')
    spec = json.loads(args.spec.read_text())
    centers = rows(args.manifest)
    wanted = {(r['dataset'], r['sequence_id']) for r in centers}
    axes = defaultdict(dict)
    evidence, errors = [], []
    for dataset, paths in spec['input_indices'].items():
        seen = set()
        for index in paths:
            try:
                raw = Path(index).read_bytes()
                evidence.append({'path': index, 'sha256': hashlib.sha256(raw).hexdigest()})
                for entry in rows(index):
                    record_path = entry['window_input']
                    if record_path in seen:
                        continue
                    seen.add(record_path)
                    record = json.loads(Path(record_path).read_text())
                    sequence = record.get('sequence_id')
                    if sequence is None:
                        sequence = record['window_id'].rsplit(':', 1)[0]
                    if (dataset, sequence) not in wanted:
                        continue
                    ids, rgb = record['frame_ids'], record['rgb_paths']
                    if len(ids) != len(rgb) or len(set(map(str, ids))) != len(ids):
                        raise ValueError('FRAME_IDENTITY_MISMATCH:' + record_path)
                    for offset, (fid, image) in enumerate(zip(ids, rgb)):
                        axes[(dataset, sequence)].setdefault(str(fid), []).append({
                            'input_record': record_path, 'offset': offset,
                            'rgb_path': image, 'rgb_exists': Path(image).is_file(),
                            'window_id': record.get('window_id'),
                        })
            except (OSError, ValueError, KeyError) as error:
                errors.append({'dataset': dataset, 'index': index, 'error': str(error)})
        print(json.dumps({'dataset': dataset, 'input_records_read': len(seen), 'errors': len(errors)}), flush=True)
    with (args.output / 'frame_axes.jsonl').open('x') as stream:
        for (dataset, sequence), frames in sorted(axes.items()):
            ordered = sorted(frames, key=lambda fid: int(fid))
            stream.write(json.dumps({'dataset': dataset, 'sequence_id': sequence,
                                     'frame_ids': ordered, 'sources': frames,
                                     'sequence_boundaries_verified': False}) + '\n')
    summary = {'stage': 'input_frame_axis_inventory', 'status': 'complete' if not errors else 'incomplete',
               'selected_centers': len(centers), 'indexed_sequences': len(axes),
               'required_sequences': len(wanted), 'errors': errors, 'source_indices': evidence,
               'video_coverage_verified': False, 'renderable_count': None}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
