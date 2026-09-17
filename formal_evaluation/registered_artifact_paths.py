"""Read-only validation of explicitly registered GT and prediction roots; no search."""
import ast
import struct
import zipfile
from collections import Counter
import hashlib
import json
import time
from pathlib import Path


def npz_headers(path):
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue
            with archive.open(name) as handle:
                if handle.read(6) != b'\x93NUMPY':
                    raise ValueError('INVALID_NPY_MAGIC:' + name)
                version = tuple(handle.read(2))
                size = struct.unpack('<H' if version == (1, 0) else '<I', handle.read(2 if version == (1, 0) else 4))[0]
                if size > 65536:
                    raise ValueError('NPY_HEADER_TOO_LARGE:' + name)
                header = ast.literal_eval(handle.read(size).decode('utf-8' if version == (3, 0) else 'latin1'))
                shapes[name[:-4]] = list(header['shape'])
    return shapes


def count_headers(counter, path):
    for key, shape in npz_headers(path).items():
        counter[key + ':' + str(shape)] += 1


def audit_catalog(catalog):
    index = Path(catalog['gt_index'])
    result = {'checked_at_epoch': int(time.time()), 'dataset': catalog['dataset'], 'gt_cache': {}, 'predictions': {}}
    try:
        raw = index.read_bytes()
        if catalog.get('gt_index_sha256') and hashlib.sha256(raw).hexdigest() != catalog['gt_index_sha256']:
            raise ValueError('GT_INDEX_CHANGED_SINCE_REGISTRATION')
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        wanted = {str(row['window_id']) for row in rows}
        cache_ids = {str(row['cache_id']): str(row['window_id']) for row in rows}
        if len(rows) != catalog['expected_windows'] or len(wanted) != len(rows):
            raise ValueError('GT_COUNT_OR_IDENTITY_MISMATCH')
        gt_fields = Counter()
        remapped = 0
        for row in rows:
            for key in ('array_path', 'metadata_path'):
                original = str(row[key])
                path = index.parent / original.split('/gt_cache/', 1)[1] if '/gt_cache/' in original else Path(original)
                remapped += str(path) != original
                if catalog.get('audit_fields') and key == 'array_path':
                    count_headers(gt_fields, path)
                with path.open('rb') as handle:
                    if not handle.read(1):
                        raise ValueError('EMPTY_GT_ARTIFACT:' + str(path))
        result['gt_cache'] = {'status': 'verified', 'index': str(index), 'windows': len(rows), 'sha256': hashlib.sha256(raw).hexdigest(), 'remapped_references': remapped}
    except (OSError, ValueError, KeyError) as error:
        result['gt_cache'] = {'status': 'blocked', 'error': str(error), 'registered_index': str(index)}
        return result
    if catalog.get('audit_fields'):
        result['gt_cache']['array_header_counts'] = dict(gt_fields)
    for method, entry in catalog['predictions'].items():
        roots = entry.get('formal_roots', [])
        if not roots:
            result['predictions'][method] = {'status': 'blocked', 'error': entry.get('error', 'NO_REGISTERED_PREDICTIONS')}
            continue
        seen, errors, root_counts, outside, incomplete = {}, [], {}, 0, 0
        canonical_fields, native_fields = Counter(), Counter()
        native_windows = 0
        for raw_root in roots:
            root = Path(raw_root)
            count = 0
            try:
                children = list(root.iterdir())
            except OSError as error:
                errors.append({'root': str(root), 'error': type(error).__name__})
                root_counts[str(root)] = 0
                continue
            for child in children:
                if not child.is_dir():
                    continue
                if not (child / 'metadata.json').is_file() or not (child / 'predictions.npz').is_file():
                    incomplete += 1
                    continue
                try:
                    metadata = json.loads((child / 'metadata.json').read_text())
                    with (child / 'predictions.npz').open('rb') as handle:
                        if not handle.read(1):
                            raise ValueError('EMPTY_PREDICTION')
                    if metadata.get('dataset') != catalog['dataset']:
                        raise ValueError('DATASET_MISMATCH')
                    key = str(metadata['window_id'])
                    key = cache_ids.get(key, key)
                    if key not in wanted:
                        outside += 1
                        continue
                    if key in seen:
                        raise ValueError('DUPLICATE_WINDOW:' + key)
                    if catalog.get('audit_fields'):
                        count_headers(canonical_fields, child / 'predictions.npz')
                        native_path = child / 'native' / 'predictions.npz'
                        if native_path.is_file():
                            count_headers(native_fields, native_path)
                            native_windows += 1
                    seen[key] = str(child)
                    count += 1
                except (OSError, ValueError, KeyError) as error:
                    if len(errors) < 12:
                        errors.append({'path': str(child), 'error': str(error)})
            root_counts[str(root)] = count
        # Empty candidate layouts may be removed by the one-time registry repair.
        status = 'verified' if len(seen) == len(wanted) and not errors else 'blocked'
        result['predictions'][method] = {'status': status, 'windows': len(seen), 'expected_windows': len(wanted), 'missing_windows': len(wanted - set(seen)), 'outside_gt_windows': outside, 'incomplete_directories_excluded': incomplete, 'formal_roots': roots if status == 'verified' else [], 'registered_root_counts': root_counts, 'errors': errors}
        if catalog.get('audit_fields'):
            result['predictions'][method]['array_header_counts'] = dict(canonical_fields)
            result['predictions'][method]['native_array_header_counts'] = dict(native_fields)
            result['predictions'][method]['native_windows'] = native_windows
    return result


def audit_reports(catalog):
    reports, directories, indices = {}, {}, {}
    details = {}
    detail_paths = set(catalog.get('detailed_report_paths', []))
    for raw in catalog.get('report_roots', []):
        root = Path(raw)
        try:
            children = list(root.iterdir())
            files = [x for x in children if x.is_file() and x.suffix == '.json']
            directories[raw] = {'resolved_root': str(root.resolve(strict=True)), 'files': [x.name for x in children if x.is_file()], 'json_files': [x.name for x in files], 'directories': [x.name for x in children if x.is_dir()]}
            for path in [x for x in children if x.is_file() and x.suffix == '.jsonl'][:20]:
                if path.stat().st_size > 32 * 1024 * 1024:
                    continue
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if rows and all('prediction_dir' in row and 'method' in row for row in rows):
                    counts, roots, selected = {}, {}, []
                    for row in rows:
                        method = row['method']
                        counts[method] = counts.get(method, 0) + 1
                        root_key = str(Path(row['prediction_dir']).parent)
                        roots.setdefault(method, {})[root_key] = roots.setdefault(method, {}).get(root_key, 0) + 1
                        if '382ecb2dc0d5f351dfb0f79a' in row['prediction_dir']:
                            selected.append(row)
                    indices[str(path)] = {'counts': counts, 'formal_roots': roots, 'selected_window': selected}
            for path in files[:100]:
                if path.stat().st_size > 16 * 1024 * 1024:
                    continue
                payload = json.loads(path.read_text())
                if not isinstance(payload, dict):
                    continue
                methods = payload.get('methods', {})
                if isinstance(methods, dict) and 'gt_windows' in payload:
                    reports[str(path)] = {'gt_windows': payload['gt_windows'], 'methods': {
                        method: {'missing_prediction_windows': value.get('missing_prediction_windows'),
                                 'datasets': {d: {'n_windows': v.get('n_windows')} for d, v in value.get('datasets', {}).items()}}
                        for method, value in methods.items() if isinstance(value, dict)}}
                    if str(path) in detail_paths:
                        validation_path = path.parent / 'validation_report.json'
                        validation = json.loads(validation_path.read_text())
                        digest = hashlib.sha256(path.read_bytes()).hexdigest()
                        if not (path.parent / 'COMPLETE').is_file() or validation.get('report_sha256') != digest:
                            raise ValueError('REPORT_COMPLETION_OR_DIGEST_MISMATCH:' + str(path))
                        details[str(path)] = {'report': payload, 'sha256': digest, 'validation': validation}
                else:
                    reports[str(path)] = payload
        except (OSError, ValueError) as error:
            directories[raw] = {'error': str(error)}
    return {'dataset': catalog['dataset'], 'checked_at_epoch': int(time.time()), 'reports': reports, 'report_details': details, 'reference_indices': indices, 'registered_directories': directories, 'all_verified': bool(reports)}


if __name__ == '__main__':
    import sys
    catalog = json.loads(sys.argv[1])
    print(json.dumps((audit_reports if catalog.get('probe_kind') == 'reports' else audit_catalog)(catalog), sort_keys=True))
