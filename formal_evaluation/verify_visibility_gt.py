"""Validate exact derived-manifest coverage before publishing a GT COMPLETE sentinel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row):
    return row['dataset'], row['window_id'], tuple(row['frame_ids'])


def run(manifest, sha256, dataset, root):
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == sha256, 'manifest checksum mismatch'
    expected = [r for r in rows(manifest) if r['dataset'] == dataset]
    actual = rows(root / 'index_shard_000_of_001.jsonl')
    assert len(actual) == len(expected) and len({key(r) for r in actual}) == len(actual)
    assert {key(r) for r in actual} == {key(r) for r in expected}, 'GT identity mismatch'
    counts = dict(joint_valid_labels=0, marker_valid_labels=0)
    for row in actual:
        metadata = json.loads(Path(row['metadata_path']).read_text())
        assert key(metadata) == key(row) and metadata['cache_version'] == 'six_dataset_window_gt_v3'
        with np.load(row['array_path'], allow_pickle=False) as arrays:
            for name, n in [('joint', 21), ('marker', 195)]:
                target, mask = arrays[name+'_visibility_target'], arrays[name+'_visibility_mask']
                assert target.shape == mask.shape == (60, 2, n), (name, target.shape, mask.shape)
                assert np.isfinite(target[mask.astype(bool)]).all()
                counts[name+'_valid_labels'] += int(mask.sum())
    summary = dict(status='complete', dataset=dataset, windows=len(actual), identity_exact=True,
                   manifest=str(manifest), manifest_sha256=sha256, **counts)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root / 'COMPLETE').write_text('complete\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    run(a.manifest, a.sha256, a.dataset, a.output_root)
