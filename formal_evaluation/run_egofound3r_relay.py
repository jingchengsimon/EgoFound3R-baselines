"""Bounded CPFS relay with optional resident inference and verified prior receipts."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from formal_evaluation.run_egofound3r_stride_evaluation import prepare_inputs
from formal_evaluation.run_formal_window_queue import _validated_output


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def wait_for(path, failure, timeout=1800):
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if failure.is_file():
            raise RuntimeError(f'peer failed: {failure.read_text()}')
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(2)


def transfer(source, target):
    """Never remove a source until every destination file is hash verified."""
    if target.exists():
        raise FileExistsError(target)
    files = sorted(p for p in source.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('symlinks are not relay artifacts')
    hashes = {str(p.relative_to(source)): digest(p) for p in files}
    incoming = target.with_name(target.name + '.incoming')
    if incoming.exists():
        raise FileExistsError(incoming)
    incoming.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, incoming)
    if any(digest(incoming / name) != value for name, value in hashes.items()):
        raise ValueError('OSS_COPY_HASH_MISMATCH')
    incoming.rename(target)
    if any(digest(target / name) != value for name, value in hashes.items()):
        raise ValueError('OSS_FINAL_HASH_MISMATCH')
    return hashes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--role', choices=('prepare', 'producer', 'consumer'), required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--plan-root', type=Path, required=True)
    parser.add_argument('--mailbox', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--stride', type=int, choices=(1, 2, 3, 4, 5), required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-count', type=int, default=1)
    parser.add_argument('--persistent', action='store_true')
    parser.add_argument('--max-inflight', type=int, default=1)
    parser.add_argument('--previous-mailbox', type=Path, action='append', default=[])

    args = parser.parse_args()
    if not 1 <= args.max_inflight <= 8 or args.smoke_count < 1:
        raise ValueError('invalid relay bounds')
    spec = json.loads(args.spec.read_text())
    repo = Path(__file__).resolve().parents[1]
    args.mailbox.mkdir(parents=True, exist_ok=True)
    failure = args.mailbox / 'FAILED.json'
    try:
        if args.role == 'prepare':
            if (args.plan_root / 'COMPLETE').exists():
                raise FileExistsError(args.plan_root)
            plans = {}
            for dataset, item in spec['datasets'].items():
                _, gt, records = prepare_inputs(dataset, item, args.plan_root / dataset)
                plans[dataset] = {'gt': str(gt), 'records': records}
                print(json.dumps({'dataset': dataset, 'verified_windows': len(records)}), flush=True)
            write_json(args.plan_root / 'plan.json', plans)
            (args.plan_root / 'COMPLETE').write_text('complete\n')
            (args.mailbox / 'COMPLETE').write_text('complete\n')
            return
        if not (args.plan_root / 'COMPLETE').is_file():
            raise RuntimeError('INPUT_PREFLIGHT_NOT_COMPLETE')
        plans = json.loads((args.plan_root / 'plan.json').read_text())
        if args.smoke:
            plans = {'h2o': {**plans['h2o'], 'records': plans['h2o']['records'][:args.smoke_count]}}
        elif not all(Path(path).is_file() for path in spec['relay_smoke_receipts']):
            raise RuntimeError('RELAY_SMOKE_NOT_COMPLETE')
        if args.role == 'consumer':
            args.output_root.mkdir(parents=True, exist_ok=False)
            write_json(args.mailbox / 'CONSUMER_READY.json', {'output': str(args.output_root)})
        else:
            wait_for(args.mailbox / 'CONSUMER_READY.json', failure)
        pending = deque()
        if args.role == 'producer' and args.persistent:
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['TASKCTL_GPU']
            from formal_evaluation.scene.adapters.run_egofound3r_baseline import main as infer_window
        completed = 0
        reports = {}
        for dataset, plan in plans.items():
            predictions = []
            for record in plan['records']:
                cache_id = str(record['cache_id'])
                if Path(cache_id).name != cache_id:
                    raise ValueError('unsafe cache id')
                token = dataset + '_' + cache_id
                ready = args.mailbox / 'ready' / (token + '.json')
                receipt = args.mailbox / 'receipts' / (token + '.json')
                source = args.mailbox / 'pending' / 'egofound3r' / 'formal' / cache_id
                target = args.output_root / dataset / 'egofound3r' / 'formal' / cache_id
                previous = next((p / 'receipts' / (token + '.json') for p in args.previous_mailbox
                                 if (p / 'receipts' / (token + '.json')).is_file()), None)
                if previous is not None:
                    prior = json.loads(previous.read_text())
                    if args.role == 'consumer':
                        target = Path(prior['target'])
                        metadata = json.loads((target / 'metadata.json').read_text())
                        expected = {'global_stride': args.stride, 'global_anchor_phase': args.stride // 2,
                                    **{k: spec[k] for k in ('checkpoint_sha256', 'source_commit', 'inference_commit')}}
                        if any(metadata.get(k) != v for k, v in expected.items()):
                            raise ValueError('REUSED_PROVENANCE_MISMATCH')
                        if metadata['dataset'] != dataset or metadata['frame_ids'] != record['frame_ids'] or metadata['window_id'] != record['window_id']:
                            raise ValueError('REUSED_WINDOW_IDENTITY_MISMATCH')
                        for name in ('metadata.json', 'run.json'):
                            if digest(target / name) != prior['sha256'][name]:
                                raise ValueError('REUSED_RECEIPT_HASH_MISMATCH')
                        if (target / 'predictions.npz').stat().st_size <= 0:
                            raise ValueError('REUSED_PREDICTION_EMPTY')
                        write_json(receipt, {**prior, 'reused_receipt': str(previous)})
                        predictions.append({'method': 'egofound3r', 'dataset': dataset,
                                            'window_id': record['window_id'], 'prediction_dir': str(target)})
                elif args.role == 'producer':
                    while len(pending) >= args.max_inflight:
                        wait_for(pending.popleft(), failure)
                    if shutil.disk_usage(args.mailbox).free < 20 * 1024**3:
                        raise RuntimeError('CPFS_FREE_BELOW_20_GIB')
                    command = [sys.executable, str(repo / 'formal_evaluation/scene/adapters/run_egofound3r_baseline.py'),
                               '--phase', 'formal', '--window-input', record['window_input'],
                               '--methods-config', str(repo / 'formal_evaluation/config/methods_v1.json'),
                               '--config', spec['config'], '--checkpoint', spec['checkpoint'],
                               '--backbone-checkpoint', spec['backbone'], '--global-stride', str(args.stride),
                               '--output-root', str(args.mailbox / 'pending')]
                    window_started = time.monotonic()
                    if args.persistent:
                        infer_window(command[2:])
                    else:
                        subprocess.run(command, check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ['TASKCTL_GPU']})
                    print(json.dumps({'inference_wall_seconds': time.monotonic() - window_started,
                                      'cache_id': cache_id, 'persistent': args.persistent}), flush=True)
                    _validated_output(source, record, 'egofound3r')
                    size = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
                    if size > 2 * 1024**3:
                        raise RuntimeError('WINDOW_EXCEEDS_2_GIB_RELAY_LIMIT')
                    write_json(ready, {'cache_id': cache_id, 'bytes': size})
                    pending.append(receipt)
                else:
                    wait_for(ready, failure)
                    _validated_output(source, record, 'egofound3r')
                    metadata = json.loads((source / 'metadata.json').read_text())
                    expected = {'global_stride': args.stride, 'global_anchor_phase': args.stride // 2,
                                **{k: spec[k] for k in ('checkpoint_sha256', 'source_commit', 'inference_commit')}}
                    if any(metadata.get(k) != v for k, v in expected.items()):
                        raise ValueError('PREDICTION_PROVENANCE_MISMATCH')
                    hashes = transfer(source, target)
                    _validated_output(target, record, 'egofound3r')
                    # Only this unique mailbox window is released, after verified upload.
                    shutil.rmtree(source)
                    write_json(receipt, {'target': str(target), 'sha256': hashes})
                    predictions.append({'method': 'egofound3r', 'dataset': dataset,
                                        'window_id': record['window_id'], 'prediction_dir': str(target)})
                completed += 1
                print(json.dumps({'role': args.role, 'uploaded_windows' if args.role == 'consumer' else 'produced_windows': completed, 'dataset': dataset, 'reused': previous is not None}), flush=True)
            if args.role == 'consumer':
                index = args.output_root / dataset / 'predictions.jsonl'
                index.parent.mkdir(parents=True, exist_ok=True)
                index.write_text(''.join(json.dumps(row) + '\n' for row in predictions))
                gt = Path(plan['gt'])
                if args.smoke:
                    rows = [json.loads(line) for line in gt.read_text().splitlines() if line.strip()]
                    gt = args.mailbox / 'smoke_gt.jsonl'
                    gt.write_text(''.join(json.dumps(row) + '\n' for row in rows[:args.smoke_count]))
                methods = args.mailbox / 'methods.json'
                config = json.loads((repo / 'formal_evaluation/config/methods_v1.json').read_text())
                write_json(methods, {'methods': {'egofound3r': config['methods']['egofound3r']}})
                report = args.output_root / dataset / 'report.json'
                subprocess.run([sys.executable, str(repo / 'formal_evaluation/evaluate_six_dataset.py'),
                                '--gt-index', str(gt), '--prediction-index', str(index),
                                '--methods-config', str(methods), '--report-path', str(report)], check=True)
                reports[dataset] = str(report)
        for receipt in pending:
            wait_for(receipt, failure)
        if args.role == 'consumer':
            write_json(args.output_root / 'summary.json', {'status': 'complete', 'reports': reports,
                       'windows': completed, 'global_stride': args.stride, 'global_anchor_phase': args.stride // 2})
            (args.output_root / 'COMPLETE').write_text('complete\n')
        (args.mailbox / (args.role.upper() + '_COMPLETE')).write_text('complete\n')
    except Exception as error:
        write_json(failure, {'role': args.role, 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
