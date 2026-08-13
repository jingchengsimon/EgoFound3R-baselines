#!/usr/bin/env python3
"""Time S²Contact or ContactOpt on 500 prepared H2O geometry samples."""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from pathlib import Path


def _frames(data_root: Path) -> tuple[str, list[str]]:
    candidates = sorted(
        rgb.parent.parent.relative_to(data_root).as_posix()
        for rgb in data_root.glob("subject4_ego/*/*/cam4/rgb")
        if sum(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in rgb.iterdir()) >= 500
    )
    if not candidates:
        raise RuntimeError("no H2O test sequence has 500 RGB frames")
    sequence = random.Random(0).choice(candidates)
    frame_ids = [path.stem for path in sorted((data_root / sequence / "cam4/rgb").iterdir())
                 if path.suffix.lower() in {".png", ".jpg", ".jpeg"}][:500]
    return sequence, frame_ids


def _parameters(model) -> int:
    seen, total = set(), 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        storage = parameter.untyped_storage()
        key = (parameter.device.type, parameter.device.index, storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += parameter.numel()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("s2contact", "contactopt"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    import torch
    from torch.utils.data import DataLoader, Subset
    from formal_evaluation.contact.adapters.run_s2_contactopt import _load_baseline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sequence, frame_ids = _frames(args.data_root)
    wanted = {(sequence, int(frame_id)) for frame_id in frame_ids}
    Dataset, model = _load_baseline(args.baseline, args.source_root)
    dataset = Dataset(str(args.cache), min_num_cont=1)
    indices = [index for index, row in enumerate(dataset.dataset)
               if (str(row.get("h2o_sequence")), int(row.get("h2o_frame_id"))) in wanted]
    if len(indices) != 500:
        raise RuntimeError(f"cache has {len(indices)} of the selected 500 H2O samples")
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=Dataset.collate_fn)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    # Cache loading, collation and CPU-to-GPU preparation end before timing.
    batches = [{key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
               for batch in loader]

    def forward() -> None:
        for batch in batches:
            with torch.inference_mode():
                output = model(batch["hand_verts_aug"], batch["hand_feats_aug"],
                               batch["obj_sampled_verts_aug"], batch["obj_feats_aug"])
            del output

    for _ in range(2):
        forward(); torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    trials = []
    for _ in range(5):
        torch.cuda.synchronize(device); start = time.perf_counter()
        forward(); torch.cuda.synchronize(device)
        trials.append(time.perf_counter() - start)
    ordered = sorted(trials)
    median = statistics.median(trials)
    parameters = _parameters(model)
    report = {
        "status": "success: real checkpoint forward on prepared H2O geometry",
        "baseline": args.baseline, "seed": 0, "sequence": sequence, "frame_ids": frame_ids,
        "timing_boundary": "prepared H2O GT hand/object geometry tensors in GPU memory to contact logits; excludes cache read, collation, GPU transfer, metrics and saves",
        "strategy": "500 independent frames; no RGB input because these official contact models consume hand/object geometry",
        "model_input_frames": 500, "output_frames": 500, "trial_seconds": trials,
        "median_seconds": median, "mean_seconds": statistics.fmean(trials),
        "p90_seconds": ordered[4], "output_fps": 500 / median,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "parameter_count": {"DeepContactNet": parameters, "unique_total": parameters},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / f"benchmark_{args.baseline}_500.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    del model, batches
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
