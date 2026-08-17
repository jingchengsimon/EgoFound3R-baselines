#!/usr/bin/env python3
"""Convert neutral MANO/object window geometry into a S²Contact/ContactOpt cache.

The cache contains *inputs only*: camera-space right MANO mesh/joints and
object mesh.  Contact targets are deliberately zero placeholders, never
dataset GT labels, because the released models only consume PointNet features.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from formal_evaluation.datasets.window_inputs import load_window_input


def _hand_object(baseline: str, source_root: Path, mano_right: Path):
    source_root = source_root.resolve(strict=True)
    mano_right = mano_right.resolve(strict=True)
    if mano_right.name != "MANO_RIGHT.pkl":
        raise ValueError(f"--mano-right must name MANO_RIGHT.pkl: {mano_right}")
    from manopth import manolayer
    original_init = manolayer.ManoLayer.__init__

    def absolute_mano_init(self, *args, **kwargs):
        if kwargs.get("mano_root", "mano/models") == "mano/models":
            kwargs["mano_root"] = str(mano_right.parent)
        return original_init(self, *args, **kwargs)

    manolayer.ManoLayer.__init__ = absolute_mano_init
    sys.path.insert(0, str(source_root))
    if baseline == "contactopt":
        return importlib.import_module("contactopt.hand_object").HandObject
    return importlib.import_module("network.hand_object").HandObject


def _sample_indices(vertex_count: int, count: int, seed_text: str) -> np.ndarray:
    if vertex_count <= 0:
        raise ValueError("object geometry has no vertices")
    seed = int.from_bytes(seed_text.encode("utf-8"), "little", signed=False) % (2 ** 32)
    return np.random.default_rng(seed).choice(vertex_count, size=count, replace=vertex_count < count).astype(np.int64)


def _sample(HandObject: object, geometry_path: Path, *, sample_id: str, points: int) -> dict[str, object] | None:
    with np.load(geometry_path, allow_pickle=False) as geometry:
        valid = np.asarray(geometry["hand_valid"], dtype=bool)
        vertices = np.asarray(geometry["hand_vertices"], dtype=np.float32)
        joints = np.asarray(geometry["hand_joints"], dtype=np.float32)
        obj_vertices = np.asarray(geometry["object_vertices"], dtype=np.float32)
        obj_faces = np.asarray(geometry["object_faces"], dtype=np.int64)
    # Released caches/model are right-hand-only.  Invalid/no-object frames are
    # skipped and become undefined (not false-negative) metric entries.
    if not valid[1] or obj_vertices.ndim != 2 or obj_vertices.shape[1:] != (3,) or len(obj_vertices) == 0:
        return None
    if obj_faces.ndim != 2 or obj_faces.shape[1:] != (3,) or len(obj_faces) == 0:
        return None
    hand = HandObject()
    hand.load_from_verts(vertices[1], obj_faces, obj_vertices)
    hand.hand_joints = joints[1]
    hand.hand_pose = np.zeros(48, dtype=np.float32)
    hand.hand_beta = np.zeros(10, dtype=np.float32)
    hand.hand_mTc = np.eye(4, dtype=np.float32)
    hand.hand_contact = np.zeros((778, 1), dtype=np.float32)
    hand.obj_contact = np.zeros((len(obj_vertices), 1), dtype=np.float32)
    indices = _sample_indices(len(obj_vertices), points, sample_id)
    hand_features, object_features = hand.generate_pointnet_features(indices)
    # ContactDBDataset reads both `ho_gt` and `ho_aug`.  They intentionally
    # share the same neutral geometry; no augmentation/GT contact is injected.
    return {
        "ho_gt": hand,
        "ho_aug": hand,
        "obj_sampled_idx": indices,
        "hand_feats_aug": np.asarray(hand_features, dtype=np.float32),
        "obj_feats_aug": np.asarray(object_features, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("s2contact", "contactopt"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mano-right", type=Path, required=True)
    parser.add_argument("--window-input-index", type=Path, required=True,
                        help="window_inputs.jsonl written by materialize_six_dataset_window_inputs.py")
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--object-points", type=int, default=2048)
    args = parser.parse_args()
    if args.output_cache.exists() or args.output_index.exists():
        raise FileExistsError("refusing to overwrite a contact geometry cache/index")
    HandObject = _hand_object(args.baseline, args.source_root, args.mano_right)
    cache: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for line in args.window_input_index.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        record_path = Path(str(entry["window_input"]))
        record = load_window_input(record_path)
        for time_index, geometry in enumerate(record["geometry_paths"]):
            identity = f"{record['cache_id']}:{time_index}"
            sample = _sample(HandObject, Path(str(geometry)), sample_id=identity, points=args.object_points)
            if sample is None:
                continue
            cache_index = len(cache)
            cache.append(sample)
            index_rows.append({
                "cache_index": cache_index,
                "cache_id": record["cache_id"],
                "dataset": record["dataset"],
                "sequence_id": record["sequence_id"],
                "window_id": record["window_id"],
                "time_index": time_index,
                "frame_id": record["frame_ids"][time_index],
                "geometry_path": geometry,
            })
    if not cache:
        raise RuntimeError("no valid right-hand/object geometry frames for contact cache")
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.output_cache.open("xb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    with args.output_index.open("x", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"baseline": args.baseline, "cache_samples": len(cache), "cache": str(args.output_cache), "index": str(args.output_index)}))


if __name__ == "__main__":
    main()
