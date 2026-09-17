"""Read-only open-file and command dependency audit of exact registered sources."""
import json
import hashlib
import os
import stat
import sys
from pathlib import Path


def audit(roots, include_inventory=True):
    requested_roots = list(roots)
    roots = [str(Path(root).resolve()) for root in roots]
    ancestors = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
    matches, errors = [], []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestors:
            continue
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            hit = {root for root in roots if root in command}
            for entry in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(entry)
                except FileNotFoundError:
                    continue
                hit.update(root for root in roots if target == root or target.startswith(root + '/'))
            maps = (proc / 'maps').read_text()
            hit.update(root for root in roots if root in maps)
            if hit:
                matches.append({'pid': int(proc.name), 'roots': sorted(hit), 'executable': command.split(' ', 1)[0]})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            errors.append(str(exc))
    sources = {}
    for root in roots:
        p = Path(root)
        rows = []
        if p.is_dir():
            for child in p.iterdir():
                info = child.lstat()
                rows.append({'name': child.name, 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                             'file': child.is_file(), 'symlink': child.is_symlink()})
        sources[root] = rows
    identities = {}
    for raw in requested_roots if include_inventory else []:
        p = Path(raw)
        alias = Path(raw.replace('/mnt/cpfs/', '/mnt/workspace/'))
        rows = []
        if p.is_dir():
            for current, _, names in os.walk(p, followlinks=False):
                for name in names:
                    f = Path(current) / name
                    st = f.stat()
                    rows.append({'path': str(f.relative_to(p)), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns})
        identities[raw] = {'resolved': str(p.resolve()), 'exists': p.is_dir(),
                           'workspace_alias_samefile': p.exists() and alias.exists() and p.samefile(alias),
                           'files': sorted(rows, key=lambda x: x['path'])}
    return {'matches': matches, 'errors': errors, 'sources': sources, 'identities': identities,
            'hostname': os.uname().nodename, 'safe_from_observed_processes': not matches and not errors}


def audit_mirrors(pairs):
    results = []
    for pair in pairs:
        source, destination = map(Path, (pair['source'], pair['destination']))
        canonical_source = str(source.resolve()).replace('/mnt/workspace/sjc/eval_artifacts/', '/mnt/workspace/sjc/DATA/eval_artifacts/')
        count = total = failures = 0
        examples = []
        for current, dirs, names in os.walk(source, followlinks=False):
            for name in names + [d for d in dirs if (Path(current) / d).is_symlink()]:
                path = Path(current) / name
                target = destination / path.relative_to(source)
                try:
                    src = path.lstat()
                    if stat.S_ISREG(src.st_mode):
                        count += 1
                        total += src.st_size
                    dst = target.lstat()
                    if stat.S_ISREG(src.st_mode):
                        if not stat.S_ISREG(dst.st_mode) or src.st_size != dst.st_size:
                            raise ValueError('TYPE_OR_SIZE_MISMATCH')
                    elif stat.S_ISLNK(src.st_mode):
                        if not target.is_symlink() or os.readlink(path) != os.readlink(target):
                            raise ValueError('SYMLINK_MISMATCH')
                        resolved = str(target.resolve()).replace('/mnt/workspace/sjc/eval_artifacts/', '/mnt/workspace/sjc/DATA/eval_artifacts/')
                        if resolved == canonical_source or resolved.startswith(canonical_source + '/'):
                            raise ValueError('OSS_LINK_DEPENDS_ON_CPFS_SOURCE')
                except (OSError, ValueError) as exc:
                    failures += 1
                    if len(examples) < 10:
                        examples.append({'source': str(path), 'error': str(exc)})
        results.append({**pair, 'regular_files': count, 'bytes': total, 'failure_count': failures,
                        'examples': examples, 'size_mirror_verified': source.is_dir() and destination.is_dir() and not failures,
                        'content_hash_verified': False})
    return {'mirrors': results}


def audit_archive(raw):
    path = Path(raw)
    catalog = json.loads(path.read_text()) if path.is_file() else {}
    results = {}
    for dataset, entry in catalog.items():
        manifest_path = Path(entry['manifest'])
        assert path.parent in manifest_path.parents
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert digest == entry['manifest_sha256'] == (manifest_path.parent / 'VERIFIED').read_text().strip()
        manifest = json.loads(manifest_path.read_text())
        assert manifest['content_verified'] is True
        for row in manifest['files']:
            for stored in row.get('parts', [{'path': row['path'], 'bytes': row['bytes']}]):
                target = Path(stored['path'])
                assert path.parent in target.parents and target.stat().st_size == stored['bytes']
        results[dataset] = {**entry, 'verified_files': len(manifest['files'])}
    return {'archive_catalog': raw, 'artifacts': results,
            'all_verified': (path.parent / 'COMPLETE').is_file() and len(results) == 3}


def audit_drift(spec):
    # Exact previous audit records, no content reads or writes to artifact trees.
    rows = []
    for pair in spec['pairs']:
        expected = {}
        for line in Path(pair['records']).read_text().splitlines():
            row = json.loads(line)
            expected[row['relative_path']] = row['destination']
        differences = []
        root = Path(pair['destination'])
        for rel, old in expected.items():
            p = root / rel
            try:
                st = p.lstat()
                kind = 'link' if stat.S_ISLNK(st.st_mode) else 'file' if stat.S_ISREG(st.st_mode) else 'dir' if stat.S_ISDIR(st.st_mode) else 'unsupported'
                new = {'kind': kind, 'bytes': st.st_size if kind == 'file' else 0, 'mtime_ns': st.st_mtime_ns, 'link': os.readlink(p) if kind == 'link' else None}
                if old != new:
                    differences.append({'path': str(p), 'old': old, 'new': new})
            except OSError as exc:
                differences.append({'path': str(p), 'error': str(exc)})
        rows.append({'destination': str(root), 'difference_count': len(differences), 'differences': differences[:30]})
    return {'drift': rows}


if __name__ == '__main__':
    spec = json.loads(sys.argv[1])
    result = (audit_drift(spec['drift']) if isinstance(spec, dict) and 'drift' in spec else
              audit_archive(spec['archive_catalog']) if isinstance(spec, dict) and 'archive_catalog' in spec else
              audit_mirrors(spec['mirrors']) if isinstance(spec, dict) and 'mirrors' in spec else
              audit(spec['roots'], False) if isinstance(spec, dict) else audit(spec))
    print(json.dumps(result, sort_keys=True))
