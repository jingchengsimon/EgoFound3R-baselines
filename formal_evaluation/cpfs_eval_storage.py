"""Read-only allocated/logical size of exact CPFS evaluation roots."""
import json
import os
import stat
import sys
import time
from pathlib import Path


def registered_roots(registry):
    roots = {'/mnt/workspace/sjc/DATA/eval_artifacts', '/mnt/workspace/sjc/eval_artifacts'}
    fields = {'output_root', 'gt_index', 'report_path'}
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in fields and isinstance(item, str) and item.startswith(('/mnt/workspace/', '/mnt/cpfs/')):
                    path = Path(item)
                    if path.suffix in {'.json', '.jsonl', '.pkl', '.npz'}:
                        path = path.parent
                    roots.add(str(path))
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(registry)
    return [root for root in sorted(roots) if not any(root.startswith(other + '/') for other in roots if root != other)]


def audit(roots, mount='/mnt/cpfs'):
    started = time.time()
    device = os.stat(mount).st_dev
    seen = set()
    rows, errors = [], []
    error_count = 0
    def error(exc):
        nonlocal error_count
        error_count += 1
        if len(errors) < 10:
            errors.append(str(exc))
    for raw in roots:
        root = Path(raw)
        row = dict(path=raw, regular_files=0, logical_bytes=0, allocated_bytes=0, symlinks=0,
                   duplicate_files=0, status='ok')
        rows.append(row)
        if root.is_symlink():
            row.update(status='symlink_not_followed', target=os.readlink(root))
            continue
        if not root.exists():
            row['status'] = 'missing_registered_path'
            continue
        if root.stat().st_dev != device:
            row['status'] = 'non_cpfs_skipped'
            continue
        for current, directories, files in os.walk(root, followlinks=False, onerror=error):
            for name in list(directories):
                path = Path(current) / name
                try:
                    info = path.lstat()
                    if stat.S_ISLNK(info.st_mode) or info.st_dev != device:
                        directories.remove(name)
                        row['symlinks'] += int(stat.S_ISLNK(info.st_mode))
                except OSError as exc:
                    directories.remove(name)
                    error(exc)
            for name in files:
                try:
                    info = os.lstat(os.path.join(current, name))
                except OSError as exc:
                    error(exc)
                    continue
                if stat.S_ISLNK(info.st_mode):
                    row['symlinks'] += 1
                elif stat.S_ISREG(info.st_mode) and info.st_dev == device:
                    identity = (info.st_dev, info.st_ino)
                    if identity in seen:
                        row['duplicate_files'] += 1
                        continue
                    seen.add(identity)
                    row['regular_files'] += 1
                    row['logical_bytes'] += info.st_size
                    row['allocated_bytes'] += info.st_blocks * 512
    totals = {key: sum(row[key] for row in rows) for key in ('regular_files', 'logical_bytes', 'allocated_bytes')}
    space = os.statvfs(mount)
    return dict(status='partial' if error_count else 'ok', rows=rows, totals=totals,
                cpfs=dict(mount=mount, total_bytes=space.f_blocks*space.f_frsize,
                          available_bytes=space.f_bavail*space.f_frsize),
                errors=errors, error_count=error_count, checked_at_epoch=int(time.time()),
                elapsed_seconds=round(time.time()-started, 2),
                scope='fixed eval artifact roots plus registered CPFS output/GT/report roots; no symlink traversal; regular files deduplicated by device/inode; live scan, not atomic')


if __name__ == '__main__':
    print(json.dumps(audit(json.loads(sys.argv[1])), sort_keys=True))
