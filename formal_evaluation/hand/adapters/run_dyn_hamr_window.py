#!/usr/bin/env python3
"""Run the released Dyn-HaMR RGB pipeline for one neutral window input.

This is a smoke/pilot runner, intentionally separate from the fixed 500-frame
speed benchmark.  It consumes only the materialized RGB paths and emits world
MANO joints reconstructed from Dyn-HaMR's official optimized parameters.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from formal_evaluation.common.io import write_comparison_output
from formal_evaluation.common.schema import SCHEMA_VERSION
from formal_evaluation.datasets.window_inputs import load_window_input


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_joints(
    mano_output: dict[str, object], frame_count: int, start_idx: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Place Dyn-HaMR's optimized joints back on the full input clip.

    Dyn-HaMR trims the sequence to ``[start_idx, end_idx)`` -- the frames its
    surviving tracks actually cover -- so its output is shorter than the clip
    that was fed in. Frames outside that span carry no official prediction and
    stay invalid rather than being filled in.
    """
    joints = mano_output["joints"].detach().cpu().numpy()
    sides = mano_output["is_right"].detach().cpu().numpy().astype(bool)
    output = np.zeros((frame_count, 2, 21, 3), dtype=np.float32)
    valid = np.zeros((frame_count, 2), dtype=bool)
    span = min(joints.shape[1], frame_count - start_idx) if joints.shape[0] else 0
    if span <= 0:
        return output, valid
    window = slice(start_idx, start_idx + span)
    for track in range(joints.shape[0]):
        slot = 1 if bool(sides[track, 0]) else 0
        if valid[:, slot].any():
            continue  # one canonical slot per side; keep official first track deterministically
        values = np.asarray(joints[track, :span], dtype=np.float32)
        mask = np.isfinite(values).all(axis=(1, 2))
        output[window, slot] = np.nan_to_num(values)
        valid[window, slot] = mask
    return output, valid


def _load_context(window_input: Path, record: dict[str, object], context_frames: int) -> tuple[list[Path], list[int]]:
    """Resolve the longer RGB context Dyn-HaMR needs, plus the scoring positions.

    Dyn-HaMR's official MultiPeopleDataset keeps only tracks with
    ``track_len > MIN_TRACK_LEN`` (60), so a 60-frame scoring window is always
    discarded. The context is fed to the official pipeline unchanged; only the
    manifest's scoring frames are read back out of the result.
    """
    directory = window_input.parent / f"context_{context_frames}f"
    mapping_path = directory / "mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"missing {mapping_path}; materialize it with "
            f"materialize_six_dataset_video_contexts.py --context-frames {context_frames}"
        )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    scoring_ids = [str(value) for value in record["frame_ids"]]
    if [str(value) for value in mapping["frame_ids"]] != scoring_ids:
        raise ValueError("context mapping scoring frames do not match the window input")
    if mapping.get("sequence") != record["sequence_id"] or mapping.get("window_id") != record["window_id"]:
        raise ValueError("context mapping identity does not match the window input")
    context_ids = [str(value) for value in mapping["context_frame_ids"]]
    if len(context_ids) != context_frames:
        raise ValueError(f"context mapping holds {len(context_ids)} frames, expected {context_frames}")
    paths = [(directory / "rgb" / f"{index:03d}_{frame_id}.png").resolve(strict=True)
             for index, frame_id in enumerate(context_ids)]
    scoring_indices = [int(value) for value in mapping["hand_indices_30fps"]]
    if len(scoring_indices) != len(scoring_ids):
        raise ValueError("context mapping provides the wrong number of scoring indices")
    if any(index < 0 or index >= context_frames for index in scoring_indices):
        raise ValueError("context mapping scoring index outside the context clip")
    if [context_ids[index] for index in scoring_indices] != scoring_ids:
        raise ValueError("context scoring indices do not recover the manifest frame identity")
    return paths, scoring_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--window-input", type=Path, required=True)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hamer-checkpoint-root", type=Path, required=True)
    parser.add_argument("--detector-weight", type=Path, required=True)
    parser.add_argument("--droid-weight", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root-iters", type=int, default=1, help="smoke default; use formal policy values for pilot")
    parser.add_argument("--smooth-iters", type=int, default=1, help="smoke default; use formal policy values for pilot")
    parser.add_argument("--context-frames", type=int,
                        help="feed this many contiguous RGB frames to the official pipeline and score only "
                             "the manifest frames; required because Dyn-HaMR drops tracks of 60 frames or fewer")
    args = parser.parse_args()
    if args.output_root.exists() and (args.output_root / "dyn_hamr" / args.phase).exists():
        # Per-window directory below is still checked before writing; this only
        # avoids implying a shared overwrite policy.
        pass

    import cv2
    import torch
    from hydra import compose, initialize_config_dir
    from ultralytics import YOLO

    record = load_window_input(args.window_input)
    scoring_count = len(record["frame_ids"])
    if args.context_frames is None:
        frame_paths = [Path(str(path)).resolve(strict=True) for path in record["rgb_paths"]]
        scoring_indices = list(range(len(frame_paths)))
    else:
        frame_paths, scoring_indices = _load_context(args.window_input, record, args.context_frames)
    frame_count = len(frame_paths)
    if frame_count < 2:
        raise ValueError("Dyn-HaMR requires at least two contiguous RGB frames")
    cache_id = str(record["cache_id"])
    output_dir = args.output_root / "dyn_hamr" / args.phase / cache_id
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_root = args.source_root.resolve(strict=True)
    dyn_root, hamer_root = source_root / "dyn-hamr", source_root / "third-party" / "hamer"
    droid_root = source_root / "third-party" / "DROID-SLAM"
    for path in (dyn_root, hamer_root, droid_root, args.hamer_checkpoint_root, args.detector_weight, args.droid_weight, args.scratch_dir):
        path.resolve(strict=True)
    sys.path[:0] = [str(dyn_root), str(hamer_root), str(droid_root), str(droid_root / "droid_slam")]
    hamer = _module(hamer_root / "run.py", "dyn_hamr_window_hamer")
    from body_model import MANO
    from body_model.utils import run_mano
    from data.dataset import MultiPeopleDataset
    from preproc.export_hamer import export_sequence_results
    from preproc.run_slam import get_slam_parser, run_loaded, save_cameras
    from droid import Droid
    from run_opt import run_opt, set_seed
    from util.loaders import resolve_cfg_paths

    images = [cv2.imread(str(path)) for path in frame_paths]
    if any(image is None for image in images):
        raise RuntimeError("failed to decode a materialized RGB frame")
    height, width = images[0].shape[:2]
    images = [image if image.shape[:2] == (height, width) else cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in images]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    hamer_model, hamer_cfg = hamer.load_hamer(str(args.hamer_checkpoint_root))
    hamer_model = hamer_model.to(device).eval()
    detector = YOLO(str(args.detector_weight)); detector.to(device)
    droid_net = Droid.load_network(str(args.droid_weight))
    sequence_name = f"dyn_hamr_{cache_id}"
    work = Path(tempfile.mkdtemp(prefix="dyn_hamr_window_", dir=args.scratch_dir))
    start = time.perf_counter()
    try:
        image_dir = work / "images" / sequence_name
        image_dir.mkdir(parents=True)
        for index, path in enumerate(frame_paths):
            (image_dir / f"{index:06d}{path.suffix.lower()}").symlink_to(path)
        raw = hamer.extract_raw_bboxes(frame_paths, detector, images=images)
        cleaned = hamer.clean_bbox_sequences(raw)
        hamer_result = hamer.run_hamer_on_cleaned_bboxes(cleaned, hamer_model, hamer_cfg, None, SimpleNamespace(rescale_factor=1.3, render=False, res_folder=None), images=images)
        hamer_pickle = work / "hamer.pkl"
        with hamer_pickle.open("wb") as handle:
            pickle.dump(hamer_result, handle)
        track_dir = work / "dynhamr" / "track_preds" / sequence_name
        shot_path = work / "dynhamr" / "shot_idcs" / f"{sequence_name}.json"
        camera_dir = work / "dynhamr" / "cameras" / sequence_name / "shot-0"
        export_sequence_results(str(hamer_pickle), str(track_dir), str(shot_path))
        focal = 0.5 * (height + width)
        intrins = torch.tensor([focal, focal, width / 2, height / 2, width, height])[None].repeat(frame_count, 1)
        slam_args = get_slam_parser().parse_args([])
        slam_args.weights, slam_args.t0, slam_args.disable_vis, slam_args.stereo = str(args.droid_weight), 0, True, False
        frame_w2c, droid = run_loaded(slam_args, frame_paths, intrins, images, droid_net)
        save_cameras(str(camera_dir), frame_w2c, intrins)
        dataset = MultiPeopleDataset({"images": str(image_dir), "tracks": str(track_dir), "shots": str(shot_path), "cameras": str(camera_dir)}, sequence_name, end_idx=frame_count, is_static=False, split_cameras=True, img_size=(width, height))
        # Dyn-HaMR keeps only tracks longer than MIN_TRACK_LEN and trims the clip
        # to the frames they cover, so a shorter span is an official outcome, not
        # an error. A clip with no surviving track yields no prediction at all.
        kept_start, kept_len = int(dataset.start_idx), int(dataset.seq_len)
        if dataset.n_tracks == 0 or kept_len <= 0:
            blocked = {
                "status": "blocked_no_track_over_min_track_len",
                "elapsed_seconds": time.perf_counter() - start, "device": str(device),
                "context_frames": frame_count, "scored_frames": scoring_count,
                "covered_scored_frames": 0,
                "detail": ("Dyn-HaMR's official MultiPeopleDataset kept no track longer than "
                           "MIN_TRACK_LEN (60) in this clip"),
            }
            joints = np.zeros((scoring_count, 2, 21, 3), dtype=np.float32)
            valid = np.zeros((scoring_count, 2), dtype=bool)
            kept_start, kept_len = 0, 0
        else:
            blocked = None
        if blocked is None:
            with initialize_config_dir(version_base=None, config_dir=str(dyn_root / "confs")):
                cfg = compose(config_name="config", overrides=[
                    "data=video_driod", f"data.seq={sequence_name}", f"data.end_idx={frame_count}",
                    f"optim.root.num_iters={args.root_iters}", f"optim.smooth.num_iters={args.smooth_iters}",
                    "run_prior=False", "run_vis=False", "is_static=False",
                ])
            cfg = resolve_cfg_paths(cfg); cfg.paths.base_dir = str(source_root); cfg.data.frame_opts.fps = 30
            mano_cfg = {key.lower(): value for key, value in dict(cfg.MANO).items()}
            # Size MANO to the span Dyn-HaMR actually kept. run_mano derives
            # seq_len from this batch size and only pads when it disagrees with
            # the data, and its padding helper assumes B x T x D -- it cannot
            # pad the B x T x 3 x 3 root orientation.
            mano_model = MANO(batch_size=len(dataset) * kept_len, pose2rot=True, **mano_cfg).to(device)
            set_seed(cfg.get("seed", 42))
            _, prediction = run_opt(cfg, dataset, str(work / "unused"), device, hand_model=mano_model, save_io=False)
            world = prediction["world"]
            mano = run_mano(mano_model, world["trans"], world["root_orient"], world["pose_body"], world["is_right"], betas=world.get("betas"))
            joints, valid = _canonical_joints(mano, frame_count, kept_start)
            # Score only the manifest frames; the surrounding context is input-only.
            selection = np.asarray(scoring_indices, dtype=np.int64)
            joints, valid = joints[selection], valid[selection]
            torch.cuda.synchronize(device)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    config = json.loads(args.methods_config.read_text(encoding="utf-8"))["methods"]["dyn_hamr"]
    write_comparison_output(output_dir, metadata={
        "schema_version": SCHEMA_VERSION, "method": "dyn_hamr", "source": config["source"],
        "phase": args.phase, "dataset": record["dataset"], "sequence": record["sequence_id"],
        "window_id": record["window_id"], "frame_ids": record["frame_ids"],
        "capabilities": {"hand_joints_world": True, "hand_valid": True},
        "scale_type": config["scale_type"], "coordinate_space": "world",
        "runner_detail": "official RGB->YOLO/HaMeR/DROID-SLAM/Dyn-HaMR; smoke optimization iterations configured explicitly",
        "context_frames": frame_count,
        "scoring_indices": [int(index) for index in scoring_indices],
        "context_policy": (
            "context frames are input-only; Dyn-HaMR's official MultiPeopleDataset drops tracks "
            "with track_len <= 60, so a 60-frame scoring window cannot be run on its own"
        ),
        "official_kept_span": [kept_start, kept_start + kept_len],
    }, arrays={"hand_joints_world": joints, "hand_valid": valid}, run=blocked or {
        "status": "success", "elapsed_seconds": time.perf_counter() - start, "device": str(device),
        "root_iterations": args.root_iters, "smooth_iterations": args.smooth_iters,
        "context_frames": frame_count, "scored_frames": scoring_count,
        "covered_scored_frames": int(valid.any(axis=1).sum()),
    }, native_metadata={"window_input": str(args.window_input)})
    print(json.dumps({"status": (blocked or {}).get("status", "success"), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
