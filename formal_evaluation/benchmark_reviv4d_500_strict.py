#!/usr/bin/env python3
"""Strict in-memory 500-output-frame benchmark for the official ReViV pipeline.

Unlike ``benchmark_reviv4d_500.py``, this runner neither starts demo
subprocesses nor creates MP4/NPY files while timing.  It reuses the demos'
official model-loading, tokenization, generation and detokenization functions.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path


TARGET_FRAMES = 500
CONTEXT_FRAMES = 40
CLIP_FRAMES = 60
CLIP_COUNT = 9


def _select(data_root: Path) -> tuple[str, list[Path]]:
    candidates = sorted(
        rgb.parent.parent.relative_to(data_root).as_posix()
        for rgb in data_root.glob("subject4_ego/*/*/cam4/rgb")
        if sum(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in rgb.iterdir()) >= TARGET_FRAMES
    )
    if not candidates:
        raise RuntimeError("no H2O test sequence has 500 RGB frames")
    sequence = random.Random(0).choice(candidates)
    paths = sorted(
        path for path in (data_root / sequence / "cam4/rgb").iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    needed = TARGET_FRAMES + CONTEXT_FRAMES
    if len(paths) < needed:
        raise RuntimeError(f"{sequence} has {len(paths)} RGB frames, needs {needed} for 500 ReViV outputs")
    return sequence, paths[:needed]


def _resize_crop(frames, size: int):
    import cv2
    import numpy as np

    height, width = frames[0].shape[:2]
    scale = size / min(height, width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    top, left = (new_h - size) // 2, (new_w - size) // 2
    return np.stack([
        cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)[top:top + size, left:left + size]
        for frame in frames
    ])[None]


def _prepare_clips(paths: list[Path]):
    import cv2
    import numpy as np

    decoded = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"cv2 failed to decode {path}")
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    scene, hand = [], []
    scene_indices = np.round(np.arange(32) * 30 / 16).astype(int)
    hand_indices = np.round(np.arange(16) * 30 / 8).astype(int)
    for start in range(0, TARGET_FRAMES, CLIP_FRAMES):
        clip = decoded[start:start + CLIP_FRAMES]
        if len(clip) != CLIP_FRAMES:
            raise RuntimeError(f"clip at {start} has {len(clip)}, expected {CLIP_FRAMES}")
        scene.append(_resize_crop([clip[index] for index in scene_indices], 512))
        hand.append(_resize_crop([clip[index] for index in hand_indices], 256))
    if len(scene) != CLIP_COUNT or len(hand) != CLIP_COUNT:
        raise RuntimeError("unexpected ReViV clip plan")
    return scene, hand


def _parameter_counts(modules: dict[str, object]) -> tuple[dict[str, int], int]:
    counts, seen, total = {}, set(), 0
    for name, module in modules.items():
        count = 0
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            count += parameter.numel()
            storage = parameter.untyped_storage()
            key = (parameter.device.type, parameter.device.index, storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += parameter.numel()
        counts[name] = count
    return counts, total


def _median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--scene-cosmos-dir", type=Path, required=True)
    parser.add_argument("--hand-cosmos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for path in (args.data_root, args.source_root, args.checkpoint_root, args.scene_cosmos_dir, args.hand_cosmos_dir):
        path.resolve(strict=True)

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    from reviv.models.generate import GenerationSampler
    from demo_infer import (
        PATHWAYS, apply_ckpt_root, build_sample as build_scene_sample, build_schedule as build_scene_schedule,
        generation_context, load_main_model, load_motion_tokenizer, resolve_targets, select_pathway,
    )
    from demo_hand import build_sample as build_hand_sample, build_schedule as build_hand_schedule, select_hand_pathway

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sequence, paths = _select(args.data_root)
    scene_clips, hand_clips = _prepare_clips(paths)

    class Paths:
        checkpoint = cam_tokenizer = lhand_tokenizer = rhand_tokenizer = None
        body_tokenizer = gaze_tokenizer = None

    model_paths = Paths()
    model_paths.ckpt_root = str(args.checkpoint_root)
    apply_ckpt_root(model_paths, {
        "checkpoint": "reviv_main.pth", "cam_tokenizer": "reviv_tok_cam.pth",
        "lhand_tokenizer": "reviv_tok_lhand.pth", "rhand_tokenizer": "reviv_tok_rhand.pth",
    })
    load_start = time.perf_counter()
    text_tokenizer = Tokenizer.from_file(str(source_root / "reviv/utils/tokenizer/trained/text_tokenizer_reviv_wordpiece_30k.json"))
    main_model, all_domains = load_main_model(model_paths.checkpoint)
    sampler = GenerationSampler(main_model)
    scene_pathway = select_pathway(all_domains)
    hand_pathway = select_hand_pathway(all_domains)
    scene_pw, hand_pw = PATHWAYS[scene_pathway], PATHWAYS[hand_pathway]
    scene_targets = resolve_targets(["tok_cam", "tok_depth_512"], scene_pathway, all_domains)
    if scene_pathway != "512" or hand_pathway != "256" or set(scene_targets) != {"tok_cam", "tok_depth_512"}:
        raise RuntimeError(f"unexpected metric_depth pathways/targets: scene={scene_pathway}, hand={hand_pathway}, targets={scene_targets}")
    scene_encoder = CausalVideoTokenizer(checkpoint_enc=str(args.scene_cosmos_dir / "encoder.jit"), device=str(device))
    scene_decoder = CausalVideoTokenizer(checkpoint_dec=str(args.scene_cosmos_dir / "decoder.jit"), device=str(device))
    hand_encoder = CausalVideoTokenizer(checkpoint_enc=str(args.hand_cosmos_dir / "encoder.jit"), device=str(device))
    tokenizers = {
        "tok_cam": load_motion_tokenizer(model_paths.cam_tokenizer),
        "tok_lhand": load_motion_tokenizer(model_paths.lhand_tokenizer),
        "tok_rhand": load_motion_tokenizer(model_paths.rhand_tokenizer),
    }
    stats = {
        name: torch.as_tensor(np.load(args.checkpoint_root / "norm_stats" / f"{name}.npy"), device=device)
        for name in ("cam_mean", "cam_std", "lhand_mean", "lhand_std", "rhand_mean", "rhand_std")
    }
    load_seconds = time.perf_counter() - load_start
    scene_schedules = {
        target: build_scene_schedule(scene_pw["cond_domains"], target, scene_pw["target_tokens"][target])
        for target in scene_targets
    }
    hand_schedule = build_hand_schedule(hand_pw)

    def scene_once(clip):
        encoded = scene_encoder(clip, temporal_window=scene_pw["num_frames"])
        cond = {scene_pw["cond_domains"][0]: torch.as_tensor(encoded, dtype=torch.int64).reshape(1, -1)}
        decoded = []
        for target in scene_targets:
            sample = build_scene_sample(cond, target, scene_pw["target_tokens"][target], text_tokenizer)
            with generation_context(args.amp_dtype):
                output = sampler.generate(sample, scene_schedules[target], text_tokenizer=text_tokenizer,
                                          verbose=False, seed=0, top_p=0.8, top_k=0.0)
            tokens = output[target]["tensor"]
            if target == "tok_cam":
                decoded.append(tokenizers[target].decode_tokens(tokens) * stats["cam_std"] + stats["cam_mean"])
            else:
                decoded.append(scene_decoder.decode(tokens.reshape(1, 5, 32, 32)))
            del sample, output
        return decoded

    def hand_once(clip):
        encoded = hand_encoder(clip, temporal_window=hand_pw["num_frames"])
        prepared = {
            "tokens": torch.as_tensor(encoded, dtype=torch.int64).reshape(1, -1),
            "clip": torch.from_numpy(clip).float().div(255).mul(2).sub(1),
        }
        sample = build_hand_sample(prepared, text_tokenizer, hand_pw)
        with generation_context(args.amp_dtype):
            output = sampler.generate(sample, hand_schedule, text_tokenizer=text_tokenizer,
                                      verbose=False, seed=0, top_p=0.8, top_k=0.0)
        decoded = []
        for side in ("lhand", "rhand"):
            tokens = output[f"tok_{side}"]["tensor"]
            if tokens.ndim == 2:
                tokens = tokens.reshape(1, 30, 7)
            decoded.append(tokenizers[f"tok_{side}"].decode_tokens(tokens) * stats[f"{side}_std"] + stats[f"{side}_mean"])
        del sample, output
        return decoded

    def run_once() -> dict[str, float]:
        stages = {"scene_pipeline": 0.0, "hand_pipeline": 0.0}
        for scene_clip, hand_clip in zip(scene_clips, hand_clips, strict=True):
            start = time.perf_counter()
            scene_output = scene_once(scene_clip)
            torch.cuda.synchronize(device)
            stages["scene_pipeline"] += time.perf_counter() - start
            start = time.perf_counter()
            hand_output = hand_once(hand_clip)
            torch.cuda.synchronize(device)
            stages["hand_pipeline"] += time.perf_counter() - start
            if not scene_output or not hand_output:
                raise RuntimeError("ReViV produced no decoded outputs")
        return stages

    for _ in range(2):
        run_once()
    torch.cuda.reset_peak_memory_stats(device)
    trials, stage_trials = [], {}
    for _ in range(5):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        stages = run_once()
        torch.cuda.synchronize(device)
        trials.append(time.perf_counter() - start)
        for name, seconds in stages.items():
            stage_trials.setdefault(name, []).append(seconds)
    counts, unique_total = _parameter_counts({
        "reviv_main_shared_scene_hand": main_model,
        "cosmos_dv8_encoder": scene_encoder,
        "cosmos_dv8_decoder": scene_decoder,
        "cosmos_dv4_hand_encoder": hand_encoder,
        "camera_tokenizer": tokenizers["tok_cam"],
        "left_hand_tokenizer": tokenizers["tok_lhand"],
        "right_hand_tokenizer": tokenizers["tok_rhand"],
    })
    median = _median(trials)
    report = {
        "status": "success: strict loaded-once ReViV scene + hand pipeline",
        "seed": 0, "sequence": sequence, "frame_ids": [path.stem for path in paths[:TARGET_FRAMES]],
        "output_frames": TARGET_FRAMES,
        "model_input_frames": {"source_rgb_context_frames": TARGET_FRAMES + CONTEXT_FRAMES,
                               "scene_dv8_clips": CLIP_COUNT * 32, "hand_dv4_clips": CLIP_COUNT * 16},
        "strategy": "9 contiguous 60-frame source clips; clip 8 contributes 20 targets and 40 context frames; scene uses 9x32 DV8 inputs, hands use 9x16 DV4 inputs",
        "timing_boundary": "models loaded once and official resized/cropped RGB clips held in memory through Cosmos encoding, ReViV generation and in-memory scene/hand detokenization; excludes checkpoint load, source RGB IO/decode, resize/crop, video encode/decode, output writes, metrics and visualization",
        "checkpoint_and_model_load_seconds_excluded": load_seconds,
        "trial_seconds": trials, "median_seconds": median, "mean_seconds": statistics.fmean(trials),
        "p90_seconds": max(trials), "output_fps": TARGET_FRAMES / median,
        "stage_trial_seconds": stage_trials,
        "stage_median_seconds": {name: _median(values) for name, values in stage_trials.items()},
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "parameter_count": {"modules": counts, "unique_total": unique_total,
                            "rule": "requires_grad parameters; shared storage deduplicated; shared ReViV main model counted once"},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "benchmark_reviv4d_500_strict.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    del main_model, sampler, scene_encoder, scene_decoder, hand_encoder, tokenizers
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
