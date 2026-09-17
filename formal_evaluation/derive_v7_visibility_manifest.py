"""Join the original window manifest to six exact registered v7 GT indexes."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity(row):
    return row['dataset'], row['window_id'], tuple(row['frame_ids'])


def join(source, catalogs):
    by_identity = {}
    for row in source:
        key = identity(row)
        if key in by_identity:
            raise ValueError(f'duplicate manifest identity: {key[:2]}')
        by_identity[key] = row
    selected = []
    seen = set()
    for dataset, expected, index_rows in catalogs:
        if len(index_rows) != expected:
            raise ValueError(f'{dataset}: expected {expected}, got {len(index_rows)}')
        for row in index_rows:
            key = identity(row)
            if key[0] != dataset or key[:2] in seen:
                raise ValueError(f'invalid or duplicate GT identity: {key[:2]}')
            seen.add(key[:2])
            if key not in by_identity:
                raise ValueError(f'GT identity absent from original manifest: {key[:2]}')
            original = by_identity[key]
            if len(key[2]) != 60 or original.get('window_stride') != 60 or original.get('window_overlap') != 0:
                raise ValueError(f'invalid strict-window contract: {key[:2]}')
            selected.append(original)
    return selected


def run(spec, root):
    if (root / 'summary.json').exists() or (root / 'windows.jsonl').exists():
        raise FileExistsError(root)
    source = Path(spec['source_manifest'])
    if digest(source) != spec['source_sha256']:
        raise ValueError('source manifest checksum mismatch')
    catalogs, evidence = [], []
    for item in spec['catalogs']:
        index = Path(item['gt_index'])
        index_rows = rows(index)
        catalogs.append((item['dataset'], item['expected_windows'], index_rows))
        evidence.append({**item, 'sha256': digest(index), 'rows': len(index_rows)})
    selected = join(rows(source), catalogs)
    if len(selected) != 2378:
        raise ValueError(f'expected 2378 joined windows, got {len(selected)}')
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'windows.jsonl'
    with manifest.open('x') as output:
        for row in selected:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    summary = {'status': 'complete', 'manifest': str(manifest), 'sha256': digest(manifest),
               'source_manifest': str(source), 'source_sha256': digest(source),
               'source_indexes': evidence, 'counts': dict(Counter(r['dataset'] for r in selected)),
               'windows': len(selected), 'join_keys': ['dataset', 'window_id', 'frame_ids'],
               'identity_exact': True}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (root / 'COMPLETE').write_text('complete\n')
    print(json.dumps(summary), flush=True)


def self_check():
    row = dict(dataset='h2o', window_id='a', frame_ids=list(range(60)), window_stride=60, window_overlap=0)
    assert join([row], [('h2o', 1, [row])]) == [row]
    for bad in ([{**row, 'frame_ids': list(range(1, 61))}], [row, row]):
        try:
            join([row], [('h2o', len(bad), bad)])
        except ValueError:
            pass
        else:
            raise AssertionError('identity mismatch/duplicate accepted')
    print('self-check passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec-b64')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        run(json.loads(base64.urlsafe_b64decode(args.spec_b64)), args.output_root)
