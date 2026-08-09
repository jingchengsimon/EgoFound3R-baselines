#!/usr/bin/env python3
"""Run S²Contact or ContactOpt and retain its 21-point contact probabilities.

The supplied baseline cache is right-hand only.  The adapter deliberately keeps
the left-hand ``hand_valid`` flag false rather than inventing a left prediction.
Joint projection follows the legacy H2O evaluator exactly: nearest MANO vertex
for each joint, with the baseline's five explicit fingertip vertex overrides.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from formal_evaluation.common.io import load_manifest, write_comparison_output
from formal_evaluation.common.schema import SCHEMA_VERSION


TIP_JOINT_IDS = (4, 8, 12, 16, 20)
# These are the baseline cache's MANO mesh indexing, retained from its old evaluator.
BASELINE_TIP_VERTEX_IDS = (745, 317, 444, 556, 673)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("s2contact", "contactopt"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="formal manifest JSON or preserved frame_index.jsonl")
    parser.add_argument("--materialized-manifest", type=Path, help="write reconstructed formal manifest when --manifest is JSONL")
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("formal",), default="formal")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_baseline(baseline: str, source_root: Path):
    """Import each vendored package using the compatibility shims from old eval."""
    sys.path.insert(0, str(source_root))
    import torch_cluster
    import torch_geometric.nn as tgn

    tgn.PointConv = tgn.PointNetConv
    tgn.knn_graph = torch_cluster.knn_graph
    if baseline == "contactopt":
        pool = importlib.import_module("torch_geometric.nn.pool")
        kui = importlib.import_module("torch_geometric.nn.unpool.knn_interpolate")
        tgn.fps = torch_cluster.fps
        tgn.radius = torch_cluster.radius
        pool.fps = torch_cluster.fps
        pool.radius = torch_cluster.radius
        pool.knn = torch_cluster.knn
        kui.knn = torch_cluster.knn
        from contactopt.deepcontact_net import DeepContactNet
        from contactopt.loader import ContactDBDataset

        return ContactDBDataset, DeepContactNet()
    from network.deepcontact_net import DeepContactNet
    from network.loader import ContactDBDataset

    return ContactDBDataset, DeepContactNet(model="incep")


def _formal_windows(manifest: Mapping[str, object]) -> dict[str, tuple[str, list[str]]]:
    formal = manifest.get("formal_test")
    sequences = formal.get("sequences") if isinstance(formal, Mapping) else None
    if not isinstance(sequences, list):
        raise ValueError("manifest.formal_test.sequences 格式错误")
    result: dict[str, tuple[str, list[str]]] = {}
    for entry in sequences:
        if not isinstance(entry, Mapping):
            continue
        sequence = entry.get("sequence")
        windows = entry.get("windows")
        if not isinstance(sequence, str) or not isinstance(windows, list):
            continue
        for window in windows:
            if not isinstance(window, Mapping):
                continue
            window_id = window.get("window_id")
            frame_ids = window.get("frame_ids")
            if not isinstance(window_id, str) or not isinstance(frame_ids, list):
                raise ValueError("formal window 缺少 id 或 frame_ids")
            result[window_id] = (sequence, [str(frame_id) for frame_id in frame_ids])
    if not result:
        raise ValueError("formal manifest 不含 window")
    return result


def _formal_windows_from_frame_index(path: Path) -> dict[str, tuple[str, list[str]]]:
    """Restore the exact 2579 window membership retained in ``frame_index.jsonl``."""
    grouped: dict[str, tuple[str, set[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sequence = row.get("sequence")
        frame_id = row.get("frame_id")
        window_ids = row.get("window_ids")
        if not isinstance(sequence, str) or frame_id is None or not isinstance(window_ids, list):
            raise ValueError(f"invalid frame-index record: {row}")
        for window_id in window_ids:
            if not isinstance(window_id, str):
                raise ValueError(f"invalid window id: {window_id!r}")
            previous = grouped.get(window_id)
            if previous is None:
                grouped[window_id] = (sequence, {str(frame_id)})
            elif previous[0] != sequence:
                raise ValueError(f"window {window_id} spans sequences: {previous[0]} vs {sequence}")
            else:
                previous[1].add(str(frame_id))
    result = {
        window_id: (sequence, sorted(frame_ids, key=lambda frame_id: int(frame_id)))
        for window_id, (sequence, frame_ids) in grouped.items()
    }
    if len(result) != 2579 or any(len(frame_ids) != 12 for _, frame_ids in result.values()):
        bad = [window_id for window_id, (_, frame_ids) in result.items() if len(frame_ids) != 12]
        raise ValueError(f"frame_index must restore 2579 12-frame windows; got {len(result)}, bad={bad[:3]}")
    return result


def _load_windows(path: Path, materialized_path: Path | None) -> dict[str, tuple[str, list[str]]]:
    if path.suffix.lower() != ".jsonl":
        return _formal_windows(load_manifest(path))
    windows = _formal_windows_from_frame_index(path)
    if materialized_path is not None:
        sequences: dict[str, list[dict[str, object]]] = {}
        for window_id, (sequence, frame_ids) in windows.items():
            sequences.setdefault(sequence, []).append({"window_id": window_id, "frame_ids": frame_ids})
        payload = {
            "manifest_version": "h2o_formal_reconstructed_from_preserved_frame_index_v1",
            "source_frame_index": str(path),
            "formal_test": {"sequences": [{"sequence": sequence, "windows": windows} for sequence, windows in sequences.items()]},
        }
        materialized_path.parent.mkdir(parents=True, exist_ok=True)
        materialized_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return windows


def _joint_vertex_indices(record: Mapping[str, object]) -> np.ndarray:
    ho = record["ho_gt"]
    vertices = np.asarray(ho.hand_verts)
    joints = np.asarray(ho.hand_joints)
    indices = np.argmin(((vertices[:, None, :] - joints[None, :, :]) ** 2).sum(axis=-1), axis=0)
    indices[list(TIP_JOINT_IDS)] = np.asarray(BASELINE_TIP_VERTEX_IDS)
    return indices


def _new_window_arrays(windows: Mapping[str, tuple[str, list[str]]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        window_id: {
            "joint_contact_probability": np.zeros((len(frame_ids), 2, 21), dtype=np.float32),
            "hand_valid": np.zeros((len(frame_ids), 2), dtype=bool),
        }
        for window_id, (_, frame_ids) in windows.items()
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S²Contact/ContactOpt formal prediction requires CUDA")
    windows = _load_windows(args.manifest, args.materialized_manifest)
    arrays_by_window = _new_window_arrays(windows)
    method_config = json.loads(args.methods_config.read_text(encoding="utf-8"))["methods"][args.baseline]
    Dataset, model = _load_baseline(args.baseline, args.source_root)
    dataset = Dataset(str(args.cache), min_num_cont=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=Dataset.collate_fn)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(args.device)
    model.to(device).eval()
    start = time.perf_counter()
    matched_samples = 0
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            size = batch["hand_verts_aug"].shape[0]
            output = model(
                batch["hand_verts_aug"].to(device),
                batch["hand_feats_aug"].to(device),
                batch["obj_sampled_verts_aug"].to(device),
                batch["obj_feats_aug"].to(device),
            )
            # Old binary condition was argmax(class) >= 5.  This keeps the
            # corresponding soft probability P(class in {5,...,9}) for AP/F1.
            probability = torch.softmax(output["contact_hand"], dim=-1)[..., 5:].sum(dim=-1).cpu().numpy()
            for batch_index in range(size):
                record = dataset.dataset[cursor + batch_index]
                if int(record.get("h2o_hand_index", 1)) != 1:
                    continue
                sequence = str(record["h2o_sequence"])
                frame_number = int(record["h2o_frame_id"])
                joint_probability = probability[batch_index, _joint_vertex_indices(record)]
                for window_id in record.get("h2o_window_ids", []):
                    descriptor = windows.get(str(window_id))
                    if descriptor is None or descriptor[0] != sequence:
                        continue
                    try:
                        time_index = [int(frame_id) for frame_id in descriptor[1]].index(frame_number)
                    except ValueError:
                        continue
                    arrays = arrays_by_window[str(window_id)]
                    arrays["joint_contact_probability"][time_index, 1] = joint_probability
                    arrays["hand_valid"][time_index, 1] = True
                    matched_samples += 1
            cursor += size
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    missing_windows = [window_id for window_id, arrays in arrays_by_window.items() if not arrays["hand_valid"][:, 1].all()]
    if missing_windows:
        raise RuntimeError(f"{args.baseline} cache 未覆盖 {len(missing_windows)} formal windows (e.g. {missing_windows[:3]})")
    for window_id, (sequence, frame_ids) in windows.items():
        arrays = arrays_by_window[window_id]
        write_comparison_output(
            args.output_root / args.baseline / args.phase / window_id,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "method": args.baseline,
                "source": method_config["source"],
                "checkpoint": str(args.checkpoint),
                "phase": args.phase,
                "sequence": sequence,
                "window_id": window_id,
                "frame_ids": frame_ids,
                "capabilities": {name: True for name in arrays},
                "scale_type": "not_applicable",
                "runner_detail": {
                    "prediction": "softmax(contact_hand logits)[classes 5:10].sum",
                    "joint_projection": "legacy nearest-MANO-vertex plus baseline fingertip overrides",
                    "hand_coverage": "right hand only; left hand_valid=false",
                },
            },
            arrays=arrays,
            run={
                "status": "success",
                "elapsed_seconds_total": elapsed,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                "device": str(device),
                "cache_samples": len(dataset),
                "matched_window_samples": matched_samples,
            },
            native_metadata={"native_logits": "contact_hand, 10 distance-bin logits"},
        )
    print(json.dumps({"baseline": args.baseline, "windows": len(windows), "cache_samples": len(dataset), "matched_window_samples": matched_samples, "elapsed_seconds": elapsed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
