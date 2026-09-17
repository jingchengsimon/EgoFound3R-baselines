"""Read-only comparison of exact artifact trees, writing a separate audit report."""
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path


def inventory(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError('INVALID_ROOT:' + str(root))
    rows = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        for name in names + dirs:
            p = Path(current) / name
            s = p.lstat()
            kind = 'link' if stat.S_ISLNK(s.st_mode) else 'file' if stat.S_ISREG(s.st_mode) else 'dir' if stat.S_ISDIR(s.st_mode) else 'unsupported'
            rows[str(p.relative_to(root))] = {'kind': kind, 'bytes': s.st_size if kind == 'file' else 0,
                'mtime_ns': s.st_mtime_ns, 'link': os.readlink(p) if kind == 'link' else None}
    return rows


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare(source, target, records, progress=lambda _: None):
    before, destination_before = inventory(source), inventory(target)
    failures, count, size = [], 0, 0
    for relative, row in sorted(before.items()):
        other = destination_before.get(relative)
        result = {'relative_path': relative, 'source': row, 'destination': other}
        error = None
        if other is None or row['kind'] != other['kind']:
            error = 'MISSING_OR_TYPE_MISMATCH'
        elif row['kind'] == 'file':
            if row['bytes'] != other['bytes']:
                error = 'SIZE_MISMATCH'
            else:
                result['source_sha256'] = digest(source / relative)
                result['destination_sha256'] = digest(target / relative)
                if result['source_sha256'] != result['destination_sha256']:
                    error = 'CONTENT_MISMATCH'
            count += 1
            size += row['bytes']
        elif row['kind'] == 'link':
            resolved = (target / relative).resolve()
            if row['link'] != other['link']:
                error = 'LINK_MISMATCH'
            elif not (target / relative).exists():
                error = 'BROKEN_OSS_LINK'
            elif str(resolved).startswith(('/mnt/workspace/', '/mnt/cpfs/')):
                error = 'OSS_LINK_DEPENDS_ON_CPFS'
            result['destination_resolved'] = str(resolved)
        elif row['kind'] == 'unsupported':
            error = 'UNSUPPORTED_FILE_TYPE'
        if error:
            failures.append({'relative_path': relative, 'error': error})
        result['error'] = error
        records.write(json.dumps(result) + '\n')
        if count and count % 100 == 0:
            records.flush()
            progress({'checked_files': count, 'checked_bytes': size, 'failure_count': len(failures)})
    stable = before == inventory(source) and destination_before == inventory(target)
    if not stable:
        failures.append({'error': 'TREE_CHANGED_DURING_AUDIT'})
    return {'source': str(source), 'destination': str(target), 'checked_files': count,
        'checked_bytes': size, 'source_fully_backed_up': not failures, 'failures': failures,
        'destination_only': sorted(set(destination_before) - set(before)), 'trees_stable': stable}


def main():
    spec = json.loads(sys.argv[1])
    output = Path(spec['output_root'])
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for pair in spec['pairs']:
        name = pair['name']
        with (output / (name + '_files.jsonl')).open('w') as records:
            result = compare(Path(pair['source']), Path(pair['destination']), records,
                lambda row: print(json.dumps({'dataset': name, **row}), flush=True))
        result['records_sha256'] = digest(output / (name + '_files.jsonl'))
        (output / (name + '_report.json')).write_text(json.dumps(result, indent=2))
        results.append(result)
        print(json.dumps({'dataset': name, **result}), flush=True)
    report = {'completed_at_epoch': int(time.time()), 'results': results,
              'all_sources_backed_up': all(r['source_fully_backed_up'] for r in results), 'source_deleted': False}
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    (output / 'COMPLETE').write_text('complete\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
