#!/usr/bin/env python3
"""Benchmark official InteractVLM hcontact inference on 500 prepared H2O RGB frames."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path


H2O_OBJECTS = {
    1: "book", 2: "espresso", 3: "lotion", 4: "spray",
    5: "milk", 6: "cocoa", 7: "chips", 8: "cappuccino",
}


def _select(data_root: Path, frame_count: int):
    from formal_evaluation.datasets import get_dataset_adapter
    selected = get_dataset_adapter("h2o", data_root).select_contiguous(
        "test", frame_count, seed=0
    )
    object_names = []
    for path in selected.frame_paths:
        object_pose = path.parent.parent / "obj_pose" / f"{path.stem}.txt"
        object_id = int(float(object_pose.read_text().split()[0]))
        try:
            object_names.append(H2O_OBJECTS[object_id])
        except KeyError as error:
            raise RuntimeError(
                f"H2O frame {path} has unsupported/background object class {object_id}"
            ) from error
    return selected, object_names


def _parameter_counts(model) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {}
    seen, total = set(), 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        pieces = name.split(".")
        module = ".".join(pieces[:2]) if pieces[0] == "model" and len(pieces) > 1 else pieces[0]
        counts[module] = counts.get(module, 0) + parameter.numel()
        storage = parameter.untyped_storage()
        key = (parameter.device.type, parameter.device.index, storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += parameter.numel()
    return counts, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14",
                        help="official CLIP ViT-L/14 identifier or fully local snapshot directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-count", type=int, default=500)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.frame_count != 500 or args.warmups < 2 or args.trials < 5:
        raise ValueError("formal InteractVLM benchmark requires 500 frames, >=2 warmups and >=5 trials")

    source_root = args.source_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    sys.path.insert(0, str(source_root))
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoTokenizer, CLIPImageProcessor
    from model.InteractVLM import InteractVLMForCausalLM
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from model.segment_anything.utils.transforms import ResizeLongestSide
    from datasets.base_contact_dataset import normalize_cam_params
    from preprocess_data.constants import HUMAN_VIEW_DICT
    from utils.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("InteractVLM formal speed benchmark requires CUDA")
    torch.cuda.set_device(device)
    selected, object_names = _select(args.data_root, args.frame_count)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), model_max_length=512, padding_side="right", use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    model = InteractVLMForCausalLM.from_pretrained(
        str(checkpoint), low_cpu_mem_usage=True, vision_tower=args.vision_tower,
        torch_dtype=dtype, train_from_LISA=False, train_from_LLAVA=False,
        local_files_only=True,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model = model.to(device=device, dtype=dtype).eval()
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=device, dtype=dtype)
    clip_image_processor = CLIPImageProcessor.from_pretrained(
        model.config.vision_tower, local_files_only=True
    )
    load_seconds = time.perf_counter() - load_started

    def preprocess_sam(image: np.ndarray, image_size: int = 1024):
        value = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        value = (value - torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1))
        value = value / torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        return F.pad(value, (0, image_size - value.shape[-1], 0, image_size - value.shape[-2]))

    prep_started = time.perf_counter()
    view_names = list(HUMAN_VIEW_DICT[model.config.hC_sam_view_type]["cam_params"])
    cam_params = torch.stack([
        normalize_cam_params(HUMAN_VIEW_DICT[model.config.hC_sam_view_type]["cam_params"][view])
        for view in view_names
    ]).unsqueeze(0).to(device=device, dtype=dtype)
    transform = ResizeLongestSide(1024)
    static_root = source_root / "data/hcontact_vitruvian"
    sam_images = []
    resize_hw = None
    for view in view_names:
        image = np.asarray(Image.open(static_root / f"body_render_norm_{view}.png"))
        resized = transform.apply_image(image)
        resize_hw = resized.shape[:2]
        sam_images.append(preprocess_sam(resized))
    sam_multiview = torch.stack(sam_images).unsqueeze(0).to(device=device, dtype=dtype)
    resize_list = [resize_hw]

    prepared = []
    for path, object_name in zip(selected.frame_paths, object_names, strict=True):
        rgb_bgr = cv2.imread(str(path))
        if rgb_bgr is None:
            raise RuntimeError(f"failed to decode selected RGB frame: {path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        image_clip = clip_image_processor.preprocess(rgb, return_tensors="pt")["pixel_values"][0]
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + (
            f"Which body parts are in contact with the {object_name}? Segment these contact areas."
        )
        prompt = prompt.replace(
            DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt")
        prepared.append((
            image_clip.unsqueeze(0).to(device=device, dtype=dtype),
            input_ids.unsqueeze(0).to(device=device),
        ))
    torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prep_started

    def forward_once() -> None:
        predictions = []
        with torch.inference_mode():
            for image_clip, input_ids in prepared:
                output = model.evaluate(
                    image_clip, sam_multiview, input_ids, cam_params, resize_list,
                    original_size_list=resize_list, lift2d_dict_path=None, contact_type="hcontact",
                    max_new_tokens=512, tokenizer=tokenizer,
                )
                prediction = output["pred_contact_3d"]
                if prediction is None or prediction.shape[-1] != 6890:
                    raise RuntimeError(f"unexpected InteractVLM hcontact output: {None if prediction is None else tuple(prediction.shape)}")
                predictions.append(prediction.detach())
        if len(predictions) != args.frame_count:
            raise RuntimeError(f"expected {args.frame_count} predictions, received {len(predictions)}")

    for _ in range(args.warmups):
        forward_once()
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    trials = []
    for _ in range(args.trials):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        forward_once()
        torch.cuda.synchronize(device)
        trials.append(time.perf_counter() - started)
    median = statistics.median(trials)
    counts, unique_total = _parameter_counts(model)
    report = {
        "status": "success: official loaded-once InteractVLM hcontact inference",
        "seed": 0, "sequence": selected.sequence_id, "frame_ids": list(selected.frame_ids),
        "object_names": object_names, "output_frames": args.frame_count, "model_input_frames": args.frame_count,
        "strategy": "500 independent official hcontact RGB inferences; InteractVLM has no temporal window or chunking",
        "timing_boundary": "all RGB decoding and official CLIP/SAM/text preprocessing complete in GPU memory before timing; times InteractVLM evaluate through 6890-vertex SMPL-H contact predictions; excludes checkpoint/model load, disk I/O, preprocessing, metrics and saves",
        "checkpoint_and_model_load_seconds_excluded": load_seconds,
        "input_preparation_seconds_excluded": preparation_seconds,
        "trial_seconds": trials, "median_seconds": median, "mean_seconds": statistics.fmean(trials),
        "p90_seconds": sorted(trials)[min(len(trials) - 1, int(0.9 * len(trials)))],
        "output_fps": args.frame_count / median,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "stage_trial_seconds": {"official_model_evaluate": trials},
        "parameter_count": {"modules": counts, "unique_total": unique_total,
                            "rule": "requires_grad parameters; shared storage deduplicated"},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "benchmark_interactvlm_500.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    del model, prepared, sam_multiview
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
