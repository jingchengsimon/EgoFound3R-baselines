"""Export normalized sequence/frame manifests from the existing read-only dataset indexes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DATASETS = {
    "h2o": "h2o",
    "hot3d": "hot3d_aria",
    "oakink_v2": "oakink_v2",
    "arctic": "egoforce_arctic",
    "hoi4d": "hoi4d",
    "taco": "taco",
}


def _ordered_frames(dataset: Any) -> dict[str, list[str]]:
    record_index = getattr(dataset, "_record_index", None)
    if record_index is not None:
        return {
            str(sequence_id): [str(record_index[int(index)]["frame_id"]) for index in indices]
            for sequence_id, indices in dataset.sequence_to_indices.items()
        }

    entries = getattr(dataset, "_sequence_entries", None)
    if entries is None:
        raise TypeError(f"unsupported dataset index: {type(dataset).__name__}")
    result: dict[str, list[str]] = {}
    for entry in entries:
        sequence_id = str(entry["sequence_id"])
        if "frame_ids" in entry:
            frame_ids = entry["frame_ids"]
        elif entry.get("index_mode") == "manifest_bounds":
            frame_ids = range(int(entry["seq_beg"]), int(entry["seq_end"]) + 1)
        else:
            frame_ids = [item["frame_id"] for item in dataset._oakink_sequence_cache_entries(entry)]
        result[sequence_id] = [str(frame_id) for frame_id in frame_ids]
    return result


def _h2o_manifest(root: Path) -> dict[str, object]:
    sequences: list[dict[str, object]] = []
    for rgb_dir in sorted(root.glob("subject*_ego/*/*/cam4/rgb")):
        frame_ids = sorted(path.stem for path in rgb_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        sequences.append({
            "sequence_id": rgb_dir.parent.parent.relative_to(root).as_posix(),
            "frame_ids": frame_ids,
        })

    official_splits = {
        split: [line for line in (root / "label_split" / f"pose_{split}.txt").read_text().splitlines() if line]
        for split in ("train", "val", "test")
    }
    return {"sequences": sequences, "official_splits": official_splits}


def _taco_splits(path: Path, sequence_ids: set[str]) -> dict[str, list[str]]:
    entry = json.loads(path.read_text(encoding="utf-8"))["datasets"]["taco"]
    result = {
        "train": list(entry["train_sequence_ids"]),
        "test": list(entry["test_sequence_ids"]),
    }
    frozen_ids = set(result["train"]) | set(result["test"])
    if set(result["train"]) & set(result["test"]):
        raise ValueError("taco: frozen train/test split overlaps")
    if frozen_ids != sequence_ids:
        raise ValueError("taco: frozen train/test split does not exactly cover the dataset index")
    return result


def export(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.egofound3r_dev))
    from egohandmetric_prompt.data.stages import build_named_frame_dataset

    roots = dict(value.split("=", 1) for value in args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output_name, loader_name in DATASETS.items():
        if output_name == "h2o":
            manifest = _h2o_manifest(Path(roots[output_name]))
        else:
            dataset = build_named_frame_dataset(
                loader_name,
                root_override=roots[output_name],
                split="all",
                load_rgb=False,
                load_depth=False,
            )
            frames = _ordered_frames(dataset)
            manifest = {"sequences": [
                {"sequence_id": sequence_id, "frame_ids": frame_ids}
                for sequence_id, frame_ids in sorted(frames.items())
            ]}
            if output_name == "taco":
                manifest["official_splits"] = _taco_splits(args.sequence_splits, set(frames))
        path = args.output_dir / f"{output_name}.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(json.dumps({
            "dataset": output_name,
            "sequences": len(manifest["sequences"]),
            "frames": sum(len(item["frame_ids"]) for item in manifest["sequences"]),
            "path": str(path),
        }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egofound3r-dev", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--sequence-splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
