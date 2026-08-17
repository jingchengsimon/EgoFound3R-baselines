#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from formal_evaluation.common.io import (
    load_manifest,
    resolve_rgb_paths,
    select_manifest_window,
    write_comparison_output,
)
from formal_evaluation.common.schema import SCHEMA_VERSION
from formal_evaluation.datasets.window_inputs import load_window_input


SCENE_METHODS = ("vggt", "pi3", "da3_large_1_1", "lingbot_map_long", "vggt_omega")


def _as_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _finite_mask(array: np.ndarray) -> np.ndarray:
    return np.isfinite(array).all(axis=-1)


def _matrix_finite_mask(array: np.ndarray) -> np.ndarray:
    return np.isfinite(array).reshape(array.shape[:-2] + (-1,)).all(axis=-1)


def _invert_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.zeros(extrinsics.shape[:-2] + (1, 4), dtype=extrinsics.dtype)
        bottom[..., 0, 3] = 1.0
        extrinsics = np.concatenate([extrinsics, bottom], axis=-2)
    if extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"相机外参形状错误: {extrinsics.shape}")
    return np.linalg.inv(extrinsics).astype(np.float32)


def _squeeze_batch(array: np.ndarray) -> np.ndarray:
    return array[0] if array.ndim > 0 and array.shape[0] == 1 else array


def _squeeze_last_channel(array: np.ndarray) -> np.ndarray:
    return array[..., 0] if array.ndim > 0 and array.shape[-1] == 1 else array


def _run_vggt(frame_paths: list[Path], source_root: Path, checkpoint: Path, device_name: str):
    sys.path.insert(0, str(source_root))
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device(device_name)
    model = VGGT()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    images = load_and_preprocess_images([str(path) for path in frame_paths], mode="crop").to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        prediction = model(images)

    height, width = images.shape[-2:]
    w2c, intrinsics = pose_encoding_to_extri_intri(
        prediction["pose_enc"], image_size_hw=(height, width)
    )
    depth = _squeeze_last_channel(_squeeze_batch(_as_numpy(prediction["depth"])))
    depth_confidence = _squeeze_batch(_as_numpy(prediction["depth_conf"]))
    world_points = _squeeze_batch(_as_numpy(prediction["world_points"]))
    world_points_confidence = _squeeze_batch(_as_numpy(prediction["world_points_conf"]))
    camera_c2w = _invert_extrinsics(_squeeze_batch(_as_numpy(w2c)))
    intrinsics_np = _squeeze_batch(_as_numpy(intrinsics))

    arrays = {
        "camera_c2w": camera_c2w,
        "camera_valid": _matrix_finite_mask(camera_c2w),
        "intrinsics": intrinsics_np,
        "intrinsics_valid": _matrix_finite_mask(intrinsics_np),
        "depth": depth,
        "depth_valid": np.isfinite(depth) & (depth > 0),
        "depth_confidence": depth_confidence,
        "world_points": world_points,
        "world_points_valid": _finite_mask(world_points),
        "world_points_confidence": world_points_confidence,
    }
    native = {"pose_encoding": _squeeze_batch(_as_numpy(prediction["pose_enc"]))}
    return arrays, native, (height, width), {"preprocess": "official crop, width=518, patch=14"}


def _run_pi3(frame_paths: list[Path], source_root: Path, checkpoint: Path, device_name: str):
    sys.path.insert(0, str(source_root))
    import torch
    from pi3.models.pi3 import Pi3
    from safetensors.torch import load_file

    device = torch.device(device_name)
    model = Pi3()
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    model = model.to(device).eval()

    # 官方 loader 接收目录；为避免复制/重命名输入帧，复用同一公式直接处理显式路径。
    original = ImageOps.exif_transpose(Image.open(frame_paths[0])).convert("RGB")
    original_width, original_height = original.size
    pixel_limit = 255_000
    scale = (pixel_limit / (original_width * original_height)) ** 0.5
    target_width, target_height = original_width * scale, original_height * scale
    grid_width, grid_height = round(target_width / 14), round(target_height / 14)
    while (grid_width * 14) * (grid_height * 14) > pixel_limit:
        if grid_width / grid_height > target_width / target_height:
            grid_width -= 1
        else:
            grid_height -= 1
    width, height = max(1, grid_width) * 14, max(1, grid_height) * 14
    from torchvision.transforms.functional import pil_to_tensor

    images = torch.stack(
        [
            pil_to_tensor(
                ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            ).float()
            / 255.0
            for path in frame_paths
        ]
    ).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        prediction = model(images[None])

    camera_c2w = _squeeze_batch(_as_numpy(prediction["camera_poses"]))
    camera_points = _squeeze_batch(_as_numpy(prediction["local_points"]))
    world_points = _squeeze_batch(_as_numpy(prediction["points"]))
    confidence = 1.0 / (1.0 + np.exp(-_squeeze_batch(_as_numpy(prediction["conf"]))[..., 0]))
    point_valid = _finite_mask(camera_points) & (camera_points[..., 2] > 0)
    arrays = {
        "camera_c2w": camera_c2w,
        "camera_valid": _matrix_finite_mask(camera_c2w),
        "camera_points": camera_points,
        "camera_points_valid": point_valid,
        "camera_points_confidence": confidence,
        "world_points": world_points,
        "world_points_valid": _finite_mask(world_points) & point_valid,
        "world_points_confidence": confidence.copy(),
    }
    native = {"confidence_logits": _squeeze_batch(_as_numpy(prediction["conf"]))[..., 0]}
    detail = {
        "preprocess": "official Pi3 pixel-limit resize (255000 pixels), divisible by 14",
        "loader_reference": "pi3.utils.basic.load_images_as_tensor",
    }
    return arrays, native, (height, width), detail


def _run_da3(frame_paths: list[Path], source_root: Path, checkpoint: Path, device_name: str):
    sys.path.insert(0, str(source_root / "src"))
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.io.input_processor import InputProcessor

    config_path = checkpoint.parent / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))["config"]
    model = create_object(OmegaConf.create(model_config))
    wrapper_state = load_file(str(checkpoint), device="cpu")
    state = {key.removeprefix("model."): value for key, value in wrapper_state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = tuple(
        f"head.scratch.output_conv2_aux.{index}.2." for index in (1, 2, 3)
    )
    disallowed_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"DA3 checkpoint 不兼容: disallowed_missing={disallowed_missing}, unexpected={unexpected}"
        )
    device = torch.device(device_name)
    model = model.to(device).eval()

    images_cpu, _, _ = InputProcessor()(
        [str(path) for path in frame_paths],
        process_res=504,
        process_res_method="upper_bound_resize",
        num_workers=min(8, len(frame_paths)),
    )
    images = images_cpu.unsqueeze(0).to(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        prediction = model(images)

    depth = _squeeze_last_channel(_squeeze_batch(_as_numpy(prediction["depth"])))
    depth_confidence = _squeeze_batch(_as_numpy(prediction["depth_conf"]))
    native_extrinsics = _squeeze_batch(_as_numpy(prediction["extrinsics"]))
    camera_c2w = _invert_extrinsics(native_extrinsics)
    intrinsics = _squeeze_batch(_as_numpy(prediction["intrinsics"]))
    arrays = {
        "camera_c2w": camera_c2w,
        "camera_valid": _matrix_finite_mask(camera_c2w),
        "intrinsics": intrinsics,
        "intrinsics_valid": _matrix_finite_mask(intrinsics),
        "depth": depth,
        "depth_valid": np.isfinite(depth) & (depth > 0),
        "depth_confidence": depth_confidence,
    }
    native = {"camera_w2c": native_extrinsics}
    detail = {
        "preprocess": "official InputProcessor, process_res=504, upper_bound_resize",
        "api_import_policy": "official core model and InputProcessor; exporter-only dependencies not imported",
        "checkpoint_allowed_missing_aux_ray_keys": list(missing),
    }
    return arrays, native, tuple(images.shape[-2:]), detail


def _run_lingbot(frame_paths: list[Path], source_root: Path, checkpoint: Path, device_name: str):
    sys.path.insert(0, str(source_root))
    import torch
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device(device_name)
    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=4,
    )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data.get("model", checkpoint_data)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    images = load_and_preprocess_images(
        [str(path) for path in frame_paths], mode="crop", image_size=518, patch_size=14
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    model.aggregator = model.aggregator.to(dtype=dtype)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        prediction = model.inference_streaming(
            images.to(device), num_scale_frames=8, keyframe_interval=1, output_device=torch.device("cpu")
        )

    height, width = images.shape[-2:]
    w2c, intrinsics = pose_encoding_to_extri_intri(
        prediction["pose_enc"], image_size_hw=(height, width)
    )
    depth = _squeeze_last_channel(_squeeze_batch(_as_numpy(prediction["depth"])))
    depth_confidence = _squeeze_batch(_as_numpy(prediction["depth_conf"]))
    camera_c2w = _invert_extrinsics(_squeeze_batch(_as_numpy(w2c)))
    intrinsics_np = _squeeze_batch(_as_numpy(intrinsics))
    arrays = {
        "camera_c2w": camera_c2w,
        "camera_valid": _matrix_finite_mask(camera_c2w),
        "intrinsics": intrinsics_np,
        "intrinsics_valid": _matrix_finite_mask(intrinsics_np),
        "depth": depth,
        "depth_valid": np.isfinite(depth) & (depth > 0),
        "depth_confidence": depth_confidence,
    }
    native = {"pose_encoding": _squeeze_batch(_as_numpy(prediction["pose_enc"]))}
    detail = {
        "preprocess": "official crop, width=518, patch=14",
        "inference": "official streaming, 8 scale frames, keyframe interval 1, SDPA backend",
        "checkpoint_missing_keys": list(missing),
        "checkpoint_unexpected_keys": list(unexpected),
    }
    return arrays, native, (height, width), detail


def _run_vggt_omega(frame_paths: list[Path], source_root: Path, checkpoint: Path, device_name: str):
    sys.path.insert(0, str(source_root))
    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = torch.device(device_name)
    model = VGGTOmega().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    images = load_and_preprocess_images(
        [str(path) for path in frame_paths], mode="balanced", image_resolution=512, patch_size=16
    ).to(device)
    with torch.inference_mode():
        prediction = model(images)

    height, width = images.shape[-2:]
    w2c, intrinsics = encoding_to_camera(prediction["pose_enc"], (height, width))
    depth = _squeeze_last_channel(_squeeze_batch(_as_numpy(prediction["depth"])))
    depth_confidence = _squeeze_last_channel(
        _squeeze_batch(_as_numpy(prediction["depth_conf"]))
    )
    camera_c2w = _invert_extrinsics(_squeeze_batch(_as_numpy(w2c)))
    intrinsics_np = _squeeze_batch(_as_numpy(intrinsics))
    arrays = {
        "camera_c2w": camera_c2w,
        "camera_valid": _matrix_finite_mask(camera_c2w),
        "intrinsics": intrinsics_np,
        "intrinsics_valid": _matrix_finite_mask(intrinsics_np),
        "depth": depth,
        "depth_valid": np.isfinite(depth) & (depth > 0),
        "depth_confidence": depth_confidence,
    }
    native = {"pose_encoding": _squeeze_batch(_as_numpy(prediction["pose_enc"]))}
    detail = {
        "preprocess": "official balanced mode, image_resolution=512, patch=16",
    }
    return arrays, native, (height, width), detail


RUNNERS = {
    "vggt": _run_vggt,
    "pi3": _run_pi3,
    "da3_large_1_1": _run_da3,
    "lingbot_map_long": _run_lingbot,
    "vggt_omega": _run_vggt_omega,
}


def _original_resolution(frame_path: Path) -> tuple[int, int]:
    with Image.open(frame_path) as image:
        width, height = ImageOps.exif_transpose(image).size
    return height, width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行场景 baseline 并写入 EgoFound3R canonical 输出")
    parser.add_argument("--method", choices=SCENE_METHODS, required=True)
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--window-input", type=Path,
                        help="method-neutral record from materialize_six_dataset_window_inputs.py")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--window-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rgb-dir-template", default="{sequence}/cam4/rgb",
                        help="dataset-relative RGB directory; {sequence} is replaced from the manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_input is not None:
        if args.manifest is not None or args.data_root is not None:
            raise ValueError("--window-input cannot be combined with --manifest/--data-root")
        window_input = load_window_input(args.window_input)
        sequence = str(window_input["sequence_id"])
        window_id = str(window_input["window_id"])
        frame_ids = [str(item) for item in window_input["frame_ids"]]
        frame_paths = [Path(str(item)) for item in window_input["rgb_paths"]]
        dataset = str(window_input["dataset"])
        output_window_id = str(window_input["cache_id"])
    else:
        if args.manifest is None or args.data_root is None:
            raise ValueError("provide --window-input or both --manifest and --data-root")
        manifest = load_manifest(args.manifest)
        sequence, window_id, frame_ids = select_manifest_window(
            manifest, phase=args.phase, sequence=args.sequence, window_id=args.window_id,
        )
        if args.rgb_dir_template == "{sequence}/cam4/rgb":
            frame_paths = resolve_rgb_paths(args.data_root, sequence, frame_ids)
        else:
            rgb_dir = args.data_root / args.rgb_dir_template.format(sequence=sequence)
            by_stem = {path.stem: path for path in rgb_dir.iterdir()
                       if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
            missing = [frame_id for frame_id in frame_ids if frame_id not in by_stem]
            if missing:
                raise FileNotFoundError(f"RGB frames missing from {rgb_dir}: {missing[:3]}")
            frame_paths = [by_stem[frame_id] for frame_id in frame_ids]
        dataset = "h2o"
        output_window_id = window_id
    methods = json.loads(args.methods_config.read_text(encoding="utf-8"))["methods"]
    method_config = methods[args.method]
    output_dir = args.output_root / args.method / args.phase / output_window_id

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("场景 baseline smoke/pilot 需要 CUDA")
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    arrays, native_arrays, processed_resolution, runner_detail = RUNNERS[args.method](
        frame_paths, args.source_root, args.checkpoint, args.device
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated()

    capabilities = {name: True for name in arrays}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": args.method,
        "source": method_config["source"],
        "source_commit": method_config["source_commit"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_id": method_config.get("checkpoint_id"),
        "phase": args.phase,
        "dataset": dataset,
        "sequence": sequence,
        "window_id": window_id,
        "frame_ids": frame_ids,
        "capabilities": capabilities,
        "scale_type": method_config["scale_type"],
        "camera_convention": "OpenCV x-right y-down z-forward; camera_c2w maps camera to world",
        "units": "method-native relative scale unless scale_type is metric",
        "source_resolution_hw": list(_original_resolution(frame_paths[0])),
        "processed_resolution_hw": list(processed_resolution),
        "runner_detail": runner_detail,
    }
    run = {
        "status": "success",
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
        "device": args.device,
        "frame_count": len(frame_ids),
    }
    native_metadata = {
        "official_output_keys_retained": sorted(native_arrays),
        "canonical_derivations": {
            "camera_c2w": "inverse of official world-to-camera output when applicable",
            "valid_masks": "finite values and positive z/depth only; no GT used",
        },
    }
    write_comparison_output(
        output_dir,
        metadata=metadata,
        arrays=arrays,
        run=run,
        native_metadata=native_metadata,
        native_arrays=native_arrays,
    )
    print(json.dumps({"output_dir": str(output_dir), **run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
