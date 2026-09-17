import json
from pathlib import Path

from formal_evaluation.registered_artifact_paths import audit_catalog


def fixture(tmp_path):
    gt = tmp_path / 'gt_cache'
    (gt / 'h2o').mkdir(parents=True)
    (gt / 'h2o/a.npz').write_bytes(b'x')
    (gt / 'h2o/a.json').write_text('{}')
    index = gt / 'index.jsonl'
    index.write_text(json.dumps({'cache_id': 'cache-a', 'window_id': 'window-a', 'array_path': '/old/gt_cache/h2o/a.npz', 'metadata_path': '/old/gt_cache/h2o/a.json'}) + '\n')
    formal = tmp_path / 'predictions/formal'
    (formal / 'cache-a').mkdir(parents=True)
    (formal / 'cache-a/metadata.json').write_text(json.dumps({'dataset': 'h2o', 'window_id': 'cache-a'}))
    (formal / 'cache-a/predictions.npz').write_bytes(b'x')
    return {'dataset': 'h2o', 'expected_windows': 1, 'gt_index': str(index), 'predictions': {'wilor': {'formal_roots': [str(formal)]}}}


def test_migrated_gt_and_canonical_prediction_identity(tmp_path):
    result = audit_catalog(fixture(tmp_path))
    assert result['gt_cache']['remapped_references'] == 2
    assert result['predictions']['wilor']['status'] == 'verified'


def test_missing_exact_path_never_falls_back_to_existing_neighbor(tmp_path):
    catalog = fixture(tmp_path)
    catalog['predictions']['wilor']['formal_roots'] = [str(tmp_path / 'old/formal')]
    result = audit_catalog(catalog)['predictions']['wilor']
    assert result['status'] == 'blocked'
    assert result['formal_roots'] == []
    assert result['missing_windows'] == 1


def test_empty_npz_and_duplicate_windows_are_blockers(tmp_path):
    catalog = fixture(tmp_path)
    root = Path(catalog['predictions']['wilor']['formal_roots'][0])
    (root / 'cache-a/predictions.npz').write_bytes(b'')
    assert audit_catalog(catalog)['predictions']['wilor']['status'] == 'blocked'
    (root / 'cache-a/predictions.npz').write_bytes(b'x')
    catalog['predictions']['wilor']['formal_roots'] *= 2
    assert audit_catalog(catalog)['predictions']['wilor']['status'] == 'blocked'


def test_taskctl_catalog_selector_uses_registered_node_only(tmp_path, monkeypatch):
    import subprocess
    import shlex
    from formal_evaluation import taskctl
    catalog = fixture(tmp_path)
    catalog['node'] = 5001
    catalog['predictions']['dyn_hamr'] = {'formal_roots': [], 'error': 'NO_REGISTERED_PREDICTIONS'}
    calls = []
    def ssh(run, node, command, timeout):
        calls.append(node)
        selected = json.loads(shlex.split(command)[-1])
        return subprocess.CompletedProcess([], 0, json.dumps(audit_catalog(selected)), '')
    monkeypatch.setattr(taskctl, '_ssh', ssh)
    run = {'run_id': 'exact-catalog', 'artifact_catalog': catalog}
    assert taskctl.catalog_paths(run, 'predictions', 'wilor')['all_verified']
    assert not taskctl.catalog_paths(run, 'predictions', 'dyn_hamr')['all_verified']
    assert taskctl.catalog_paths(run, 'gt-cache')['predictions'] == {}
    assert calls == [5001, 5001, 5001]


def test_changed_gt_index_is_rejected(tmp_path):
    catalog = fixture(tmp_path)
    catalog['gt_index_sha256'] = 'wrong'
    result = audit_catalog(catalog)
    assert result['gt_cache']['error'] == 'GT_INDEX_CHANGED_SINCE_REGISTRATION'
    assert result['predictions'] == {}


def test_old_failed_window_directory_does_not_replace_valid_backfill(tmp_path):
    catalog = fixture(tmp_path)
    empty = tmp_path / 'failed/formal/cache-a'
    empty.mkdir(parents=True)
    catalog['predictions']['wilor']['formal_roots'].append(str(empty.parent))
    result = audit_catalog(catalog)['predictions']['wilor']
    assert result['status'] == 'verified'
    assert result['windows'] == 1
    assert result['incomplete_directories_excluded'] == 1


def test_report_audit_keeps_complete_report_and_backfill_index(tmp_path):
    from formal_evaluation.registered_artifact_paths import audit_reports
    root = tmp_path / 'reports'
    root.mkdir()
    (root / 'report.json').write_text(json.dumps({'gt_windows': 400, 'methods': {'pad_hand': {'missing_prediction_windows': 0, 'datasets': {'hot3d': {'n_windows': 400}}}}}))
    (root / 'predictions.jsonl').write_text('\n'.join(json.dumps({'method': 'pad_hand', 'prediction_dir': str(tmp_path / ('main' if i < 399 else 'backfill') / str(i))}) for i in range(400)))
    result = audit_reports({'dataset': 'hot3d', 'report_roots': [str(root)]})
    index = result['reference_indices'][str(root / 'predictions.jsonl')]
    assert index['counts']['pad_hand'] == 400
    assert sorted(index['formal_roots']['pad_hand'].values()) == [1, 399]
    assert result['reports'][str(root / 'report.json')]['methods']['pad_hand']['missing_prediction_windows'] == 0
