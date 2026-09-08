#!/usr/bin/env python3
"""Frozen latest-stride5 diagnostics: joint MPJPE/RR P95 and joint-eight P97.5."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from egocentric_metrics import world_aligned_mpjpe
from egocentric_metrics.mano import _procrustes_per_frame
from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.evaluate_six_dataset import _camera_to_world, _prediction_geometry_camera, _target_geometry
GRANULARITIES = (
    ("joints", "joint", "joints", "", "mpjpe", "mpjve", "mpjae"),
    ("markers", "marker", "markers", "marker_", "mpmpe", "mpmve", "mpmae"),
    ("vertices", "vertex", "vertices", "vertex_", "mpvpe", "mpvve", "mpvae"),
)
DATASETS = ("h2o", "hot3d", "arctic", "oakink_v2", "taco", "hoi4d")
FRAME_METRICS = ("MPJPE", "RR", "PA", "Sim3", "W", "WA")
FILTER_METRICS = FRAME_METRICS + ("MPJVE", "MPJAE")
SCHEMES = {"joint2_p95": (0.95, (0, 1)), "joint8_p97_5": (0.975, tuple(range(8)))}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_path(index: Path, row: dict) -> Path:
    path = Path(row["array_path"])
    return path if path.is_file() else index.parent / str(path).split("/gt_cache/", 1)[1]


def load_window(prediction_dir: Path, gt_path: Path) -> tuple[dict, dict, dict]:
    metadata = json.loads((prediction_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(prediction_dir / "predictions.npz", allow_pickle=False) as archive:
        prediction = {
            key: archive[key]
            for key in archive.files
            if key.startswith("hand_") or key in ("camera_c2w", "camera_valid")
        }
    with np.load(gt_path, allow_pickle=False) as archive:
        target = {
            key: archive[key]
            for key in archive.files
            if key.startswith("hand_") or key in ("camera_c2w", "camera_valid")
        }
    return metadata, prediction, target


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def weighted_window_metrics(results: list[dict]) -> dict:
    metrics = aggregate_windows(results, method="egofound3r")
    for key in list(metrics):
        if not key.endswith("_mean"):
            continue
        stem = key[:-5]
        suffix = (
            "_frame_count"
            if "_w_" in stem and stem.endswith(("mpjpe", "mpmpe", "mpvpe"))
            else "_pair_count"
            if stem.endswith(("mpjve", "mpmve", "mpvve"))
            else "_triplet_count"
            if stem.endswith(("mpjae", "mpmae", "mpvae"))
            else None
        )
        if suffix is None:
            continue
        samples = [
            (row[stem], row.get(stem + suffix, 0))
            for row in results
            if stem in row and np.isfinite(row[stem]) and row.get(stem + suffix, 0) > 0
        ]
        count = sum(n for _, n in samples)
        metrics[stem + "_sample_count"] = count
        metrics[stem + "_window_macro_mean"] = metrics[key]
        metrics[key] = sum(value * n for value, n in samples) / count if count else float("nan")
    return metrics


def combined_rows(metrics: dict) -> dict:
    tables = {}
    for table, _, _, prefix, position, velocity, acceleration in GRANULARITIES:
        row = {}
        for label, metric, weight_suffix, scale in (
            (position.upper(), position, "_count", 1.0),
            ("RR", "rr_" + position, "_count", 1.0),
            ("PA", "pa_" + position, "_count", 1.0),
            ("Sim3", "global_sim3_" + position, "_count", 1.0),
            ("W", "w_" + position, "_sample_count", 1.0),
            ("WA", "wa_" + position, "_count", 1.0),
            (velocity.upper(), velocity, "_sample_count", 1.0),
            (acceleration.upper(), acceleration, "_sample_count", 0.001),
        ):
            pairs = []
            for side in ("left", "right"):
                stem = "hand_" + side + "_" + prefix + metric
                value, count = metrics.get(stem + "_mean"), metrics.get(stem + weight_suffix, 0)
                if isinstance(value, (int, float)) and np.isfinite(value) and count > 0:
                    pairs.append((float(value), int(count)))
            count = sum(n for _, n in pairs)
            row[label] = {
                "value": sum(value * n for value, n in pairs) / count * scale if count else None,
                "valid_count": count,
            }
        tables[table] = row
    return tables


def _per_frame(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    errors = np.linalg.norm(prediction - target, axis=-1)
    output = np.full(len(valid), np.nan)
    for frame in np.flatnonzero(valid):
        output[frame] = float(errors[frame].mean() * 1000.0)
    return output


def _pair_triplet(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pair = np.full(max(len(valid) - 1, 0), np.nan)
    triplet = np.full(max(len(valid) - 2, 0), np.nan)
    if len(valid) >= 2:
        good = valid[:-1] & valid[1:]
        delta = ((prediction[1:] - prediction[:-1]) - (target[1:] - target[:-1])) * 30.0
        pair[good] = np.linalg.norm(delta[good], axis=-1).mean(axis=-1) * 1000.0
    if len(valid) >= 3:
        good = valid[:-2] & valid[1:-1] & valid[2:]
        delta = (
            (prediction[2:] - 2.0 * prediction[1:-1] + prediction[:-2])
            - (target[2:] - 2.0 * target[1:-1] + target[:-2])
        ) * (30.0**2)
        triplet[good] = np.linalg.norm(delta[good], axis=-1).mean(axis=-1) * 1000.0
    return pair, triplet


def _incident_pair_max(pair: np.ndarray, frame_count: int) -> np.ndarray:
    output = np.full(frame_count, np.nan)
    for frame in range(frame_count):
        values = []
        if frame > 0 and np.isfinite(pair[frame - 1]):
            values.append(pair[frame - 1])
        if frame < frame_count - 1 and np.isfinite(pair[frame]):
            values.append(pair[frame])
        if values:
            output[frame] = max(values)
    return output


def _nanmax(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    output = np.max(np.where(finite, values, -np.inf), axis=axis)
    output[~np.any(finite, axis=axis)] = np.nan
    return output


def freeze_window(prediction: dict, target: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return filter scores plus frozen frame, pair, and triplet errors."""
    pred_joints, _ = _prediction_geometry_camera(prediction, "joint")
    gt_joints = target["hand_joints_camera"]
    hand_valid = prediction["hand_valid"].astype(bool) & target["hand_valid"].astype(bool)
    camera_valid = target["camera_valid"].astype(bool)
    frame_count = pred_joints.shape[0]
    assert frame_count == 60
    frozen_frame = np.full((3, 2, len(FRAME_METRICS), frame_count), np.nan)
    frozen_pair = np.full((3, 2, frame_count - 1), np.nan)
    frozen_triplet = np.full((3, 2, frame_count - 2), np.nan)

    for granularity_index, (_, granularity, field, _, _, _, _) in enumerate(GRANULARITIES):
        pred_points, _ = _prediction_geometry_camera(prediction, granularity)
        gt_points = _target_geometry(target, field, "camera")
        world_pred = _camera_to_world(pred_points, target["camera_c2w"])
        world_gt = _target_geometry(target, field, "world")
        for side in range(2):
            raw_valid = (
                hand_valid[:, side]
                & np.isfinite(pred_points[:, side]).all(axis=(1, 2))
                & np.isfinite(gt_points[:, side]).all(axis=(1, 2))
            )
            point_mask = np.broadcast_to(raw_valid[:, None], pred_points[:, side].shape[:2])
            raw = _per_frame(pred_points[:, side], gt_points[:, side], raw_valid)
            aligned = _procrustes_per_frame(pred_points[:, side], gt_points[:, side], point_mask)
            pa_valid = raw_valid & np.isfinite(aligned).all(axis=(1, 2))
            pa = _per_frame(aligned, gt_points[:, side], pa_valid)
            roots_valid = (
                np.isfinite(pred_joints[:, side, 0]).all(axis=-1)
                & np.isfinite(gt_joints[:, side, 0]).all(axis=-1)
            )
            relative_valid = raw_valid & roots_valid
            relative_pred = pred_points[:, side] - pred_joints[:, side, :1]
            relative_gt = gt_points[:, side] - gt_joints[:, side, :1]
            rr = _per_frame(relative_pred, relative_gt, relative_valid)
            sim3 = world_aligned_mpjpe(
                pred_points[:, side], gt_points[:, side], joint_mask=point_mask,
                mode="all", chunk_length=frame_count, unit_scale=1000.0,
            )
            world_valid = (
                raw_valid
                & camera_valid
                & np.isfinite(world_pred[:, side]).all(axis=(1, 2))
                & np.isfinite(world_gt[:, side]).all(axis=(1, 2))
            )
            world_mask = np.broadcast_to(world_valid[:, None], world_pred[:, side].shape[:2])
            w = world_aligned_mpjpe(
                world_pred[:, side], world_gt[:, side], joint_mask=world_mask,
                mode="first2", chunk_length=frame_count, unit_scale=1000.0,
            )
            wa = world_aligned_mpjpe(
                world_pred[:, side], world_gt[:, side], joint_mask=world_mask,
                mode="all", chunk_length=frame_count, unit_scale=1000.0,
            )
            pair, triplet = _pair_triplet(relative_pred, relative_gt, relative_valid)
            frozen_frame[granularity_index, side] = np.stack((raw, rr, pa, sim3, w, wa))
            frozen_pair[granularity_index, side] = pair
            frozen_triplet[granularity_index, side] = triplet

    joint = frozen_frame[0]
    velocity = np.stack([_incident_pair_max(frozen_pair[0, side], frame_count) for side in range(2)])
    acceleration = np.full((2, frame_count), np.nan)
    acceleration[:, 1:-1] = frozen_triplet[0]
    filter_scores = np.concatenate((joint, velocity[:, None], acceleration[:, None]), axis=1)
    return _nanmax(filter_scores, axis=0), frozen_frame, frozen_pair, frozen_triplet


def exclusion_mask(filter_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.any(filter_scores > thresholds[:, None], axis=0)


def _mean_or_nan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def aggregate_scheme(
    frozen_frame: np.ndarray,
    frozen_pair: np.ndarray,
    frozen_triplet: np.ndarray,
    excluded: np.ndarray,
) -> tuple[dict, list[dict]]:
    results = []
    for window in range(len(excluded)):
        keep = ~excluded[window]
        pair_keep = keep[:-1] & keep[1:]
        triplet_keep = keep[:-2] & keep[1:-1] & keep[2:]
        result = {"excluded_frame_count": int(excluded[window].sum())}
        for granularity_index, (_, _, _, prefix, position, velocity, acceleration) in enumerate(GRANULARITIES):
            for side, name in enumerate(("left", "right")):
                key = "hand_" + name + "_" + prefix
                frame_arrays = frozen_frame[window, granularity_index, side]
                result[key + "valid_frame_count"] = int(np.count_nonzero(keep & np.isfinite(frame_arrays[0])))
                for metric_index, metric in enumerate((
                    position, "rr_" + position, "pa_" + position,
                    "global_sim3_" + position, "w_" + position, "wa_" + position,
                )):
                    values = frame_arrays[metric_index, keep]
                    result[key + metric] = _mean_or_nan(values)
                    if metric_index == 4:
                        result[key + metric + "_frame_count"] = int(np.isfinite(values).sum())
                pair_values = frozen_pair[window, granularity_index, side]
                pair_valid = pair_keep & np.isfinite(pair_values)
                result[key + velocity] = _mean_or_nan(pair_values[pair_valid])
                result[key + velocity + "_pair_count"] = int(pair_valid.sum())
                triplet_values = frozen_triplet[window, granularity_index, side]
                triplet_valid = triplet_keep & np.isfinite(triplet_values)
                result[key + acceleration] = _mean_or_nan(triplet_values[triplet_valid])
                result[key + acceleration + "_triplet_count"] = int(triplet_valid.sum())
        results.append(result)
    metrics = weighted_window_metrics(results)
    return combined_rows(metrics), results


def masks_for_scheme(scores, quantile, selected):
    thresholds = np.full(8, np.nan)
    for index in selected:
        values = scores[:, index].ravel()
        values = values[np.isfinite(values)]
        if len(values):
            thresholds[index] = np.quantile(values, quantile, method="linear")
    reasons = scores > thresholds[None, :, None]
    return thresholds, reasons, reasons.any(axis=1)


def render_tables(summary, baselines, destination):
    labels = dict(zip(DATASETS, ("H2O", "HOT3D", "ARCTIC", "OakInk-v2", "TACO", "HOI4D")))
    sections = ["筛选诊断：预测误差驱动的帧筛选；基线为指定 CSV 中未筛选值，非同 mask 公平比较。",
                "位置 mm；速度 mm/s；加速度 m/s²。W/WA† 使用 GT-camera oracle。所有拟合在完整窗口冻结。"]
    for scheme in SCHEMES:
        for table, _, _, _, pos, vel, acc in GRANULARITIES:
            metrics = (pos.upper(), "RR", "PA", "Sim3", "W", "WA", vel.upper(), acc.upper())
            lines = [f"## {scheme} — {table}", "", "| 数据集 | " + " | ".join(m + ("†" if m in ("W", "WA") else "") for m in metrics) + " |", "|---|" + "---|" * 8]
            for ds in DATASETS:
                cells = []
                for metric in metrics:
                    value = summary['schemes'][scheme]['datasets'][ds]['tables'][table][metric]['value']
                    best = baselines[table][ds].get(metric)
                    ego = f"{value:.3f}" if value is not None else "—"
                    if best and value is not None and value < best['value']:
                        ego = f"**{ego}**"
                    cells.append(ego + (f" / {best['value']:.3f} {best['initial']}" if best else " / —"))
                lines.append("| " + labels[ds] + " | " + " | ".join(cells) + " |")
            text = '\n'.join(lines) + '\n'
            (destination / f'{scheme}_{table}.md').write_text(text)
            sections.append(text)
    (destination / 'tables.md').write_text('\n\n'.join(sections) + '\n')


def run(spec, output, scratch):
    output.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    summary = {
        'status': 'running', 'checkpoint_sha256': spec['checkpoint_sha256'],
        'inference_commit': spec['inference_commit'], 'source_commit': spec['source_commit'],
        'baseline_sources': spec['baseline_sources'], 'inputs': {}, 'frozen_windows': 0,
        'scheme_window_evaluations': 0,
        'protocol': {'global_stride': 5, 'global_anchor_phase': 2, 'frames_per_window': 60,
            'filter_granularity': 'joint_only', 'world_pose_source': 'gt_camera_c2w_oracle',
            'fit_on_full_unfiltered_window': True, 'no_refit_after_filter': True,
            'excluded_gaps_are_never_bridged': True, 'shared_frozen_scores_between_schemes': True,
            'hand_reduction': 'maximum of finite valid hand scores',
            'threshold_scope': 'dataset separately, finite window-frame scores, numpy linear quantile',
            'filter_combination': 'exclude if any selected joint metric strictly exceeds its threshold',
            'invalid_scores': 'NaN does not trigger; invalid support remains invalid',
            'downstream_mask': 'shared frame mask for joints markers vertices and future Contact3R',
            'baseline_comparison': 'fixed unfiltered CSV baselines; diagnostic only'},
        'schemes': {name: {'quantile': q, 'filter_metrics': [FILTER_METRICS[i] for i in ids], 'datasets': {}}
                    for name, (q, ids) in SCHEMES.items()}}
    for catalog in spec['catalogs']:
        ds = catalog['dataset']; gt_index = Path(catalog['gt_index']); pred_index = Path(catalog['prediction_index'])
        deadline = time.monotonic() + 43200
        while not pred_index.is_file():
            assert time.monotonic() < deadline, f'timed out waiting for {pred_index}'
            print(json.dumps({'stage': 'waiting_for_complete_prediction_index', 'dataset': ds}), flush=True)
            time.sleep(60)
        gt_list, pred_list = read_jsonl(gt_index), read_jsonl(pred_index)
        gt = {r['window_id']: r for r in gt_list}; pred = {r['window_id']: r for r in pred_list}
        assert len(gt) == len(gt_list) == len(pred) == len(pred_list) == catalog['expected_windows']
        assert gt.keys() == pred.keys()
        hashes = {'gt_index_sha256': sha256(gt_index), 'prediction_index_sha256': sha256(pred_index)}
        ids = sorted(gt); frames = []; arrays = [[], [], [], []]
        for number, wid in enumerate(ids, 1):
            entry = pred[wid]; directory = Path(entry['prediction_dir'])
            assert entry['method'] == 'egofound3r' and entry['dataset'] == ds
            metadata, prediction, target = load_window(directory, target_path(gt_index, gt[wid]))
            assert metadata['window_id'] == wid and metadata['dataset'] == ds
            assert metadata['frame_ids'] == gt[wid]['frame_ids'] and len(metadata['frame_ids']) == 60
            assert metadata['global_stride'] == 5 and metadata['global_anchor_phase'] == 2
            for key in ('checkpoint_sha256', 'inference_commit', 'source_commit'):
                assert metadata[key] == spec[key], (ds, wid, key)
            assert json.loads((directory/'run.json').read_text())['status'] == 'success'
            frames.append(metadata['frame_ids'])
            frozen = freeze_window(prediction, target)
            for values, value in zip(arrays, frozen): values.append(value)
            if number == 1 or number % 25 == 0 or number == len(ids):
                print(json.dumps({'stage': 'freeze', 'dataset': ds, 'processed': number, 'total': len(ids)}), flush=True)
        assert hashes == {'gt_index_sha256': sha256(gt_index), 'prediction_index_sha256': sha256(pred_index)}
        score, frame, pair, triplet = [np.stack(x) for x in arrays]
        cache = scratch / f'{ds}_frozen.npz'
        np.savez_compressed(cache, window_ids=np.array(ids), frame_ids=np.array(frames), filter=score, frame=frame, pair=pair, triplet=triplet)
        summary['inputs'][ds] = {**catalog, **hashes, 'windows': len(ids), 'frozen_cache': str(cache)}
        summary['frozen_windows'] += len(ids)
        dest = output/ds; dest.mkdir()
        full_tables, _ = aggregate_scheme(frame, pair, triplet, np.zeros((len(ids), 60), dtype=bool))
        (dest/'unfiltered_frozen_tables.json').write_text(json.dumps(json_safe(full_tables), allow_nan=False))
        for name, (quantile, selected) in SCHEMES.items():
            thresholds, reasons, excluded = masks_for_scheme(score, quantile, selected)
            tables, rows = aggregate_scheme(frame, pair, triplet, excluded)
            evaluable = np.isfinite(score).any(axis=1)
            n = int(evaluable.sum()); removed = int((excluded & evaluable).sum())
            supports = {key: sum(r.get(key, 0) for r in rows) for key in rows[0]
                        if key.endswith(('_frame_count', '_pair_count', '_triplet_count')) and key != 'excluded_frame_count'}
            summary['schemes'][name]['datasets'][ds] = {
                'windows': len(ids), 'total_window_frames': len(ids)*60,
                'evaluable_frame_count': n, 'invalid_frame_count': int((~evaluable).sum()),
                'thresholds': {FILTER_METRICS[i]: thresholds[i] for i in selected},
                'per_metric_above_count': {FILTER_METRICS[i]: int(reasons[:, i].sum()) for i in selected},
                'excluded_union_count': removed, 'excluded_union_rate': removed/n if n else None,
                'retained_intersection_count': n-removed, 'retained_intersection_rate': (n-removed)/n if n else None,
                'support_counts': supports, 'tables': tables}
            with (dest/f'{name}_mask.jsonl').open('x') as handle:
                for w, wid in enumerate(ids):
                    for f, frame_id in enumerate(frames[w]):
                        row = {'dataset': ds, 'window_id': wid, 'frame_index': f, 'frame_id': frame_id,
                               'excluded': bool(excluded[w,f]), 'retained': not bool(excluded[w,f]),
                               'evaluable': bool(evaluable[w,f]),
                               'reason_codes': [FILTER_METRICS[i] for i in selected if reasons[w,i,f]]}
                        handle.write(json.dumps(row)+'\n')
            (dest/f'{name}_window_metrics.jsonl').write_text(''.join(json.dumps(json_safe({'window_id': wid, **row}), allow_nan=False)+'\n' for wid,row in zip(ids,rows)))
            summary['scheme_window_evaluations'] += len(ids)
            print(json.dumps({'stage': 'aggregate', 'dataset': ds, 'scheme': name, 'windows': len(ids), 'excluded': removed}), flush=True)
        (dest/'COMPLETE').write_text('complete\n')
        (output/'progress_summary.json').write_text(json.dumps(json_safe(summary), allow_nan=False, indent=2)+'\n')
    assert summary['frozen_windows'] == 2378 and summary['scheme_window_evaluations'] == 4756
    summary['status'] = 'complete'
    render_tables(summary, spec['baselines'], output)
    (output/'summary.json').write_text(json.dumps(json_safe(summary), allow_nan=False, indent=2)+'\n')
    (output/'COMPLETE').write_text('complete\n')
    print(json.dumps({'status': 'complete', 'frozen_windows': 2378, 'scheme_window_evaluations': 4756, 'tables': 6}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scratch-root', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()), args.output_root, args.scratch_root)
