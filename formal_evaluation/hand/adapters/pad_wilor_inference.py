#!/usr/bin/env python3
"""Run PAD-Hand's WiLoR front end without its visualization-only renderer."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wilor_root = args.source_root / "WiLoR"
    sys.path.insert(0, str(wilor_root))
    import wilor.models as models
    from wilor.configs import get_config

    def load_wilor_without_renderer(checkpoint_path: str, cfg_path: str):
        config = get_config(cfg_path, update_cachedir=True)
        if "vit" in config.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in config.MODEL:
            config.defrost()
            if config.MODEL.IMAGE_SIZE != 256:
                raise ValueError(f"WiLoR expects a 256px ViT input, got {config.MODEL.IMAGE_SIZE}")
            config.MODEL.BBOX_SHAPE = [192, 256]
            config.freeze()
        if "PRETRAINED_WEIGHTS" in config.MODEL.BACKBONE:
            config.defrost()
            config.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
            config.freeze()
        if "DATA_DIR" in config.MANO:
            config.defrost()
            config.MANO.DATA_DIR = "./mano_data/"
            config.MANO.MODEL_PATH = "./mano_data/"
            config.MANO.MEAN_PARAMS = "./mano_data/mano_mean_params.npz"
            config.freeze()
        return models.WiLoR.load_from_checkpoint(
            checkpoint_path, strict=False, cfg=config, init_renderer=False
        ), config

    models.load_wilor = load_wilor_without_renderer
    sys.argv = [str(args.source_root / "wilor_inference.py"), "--video", args.video, "--output", args.output]
    runpy.run_path(str(args.source_root / "wilor_inference.py"), run_name="__main__")


if __name__ == "__main__":
    main()
