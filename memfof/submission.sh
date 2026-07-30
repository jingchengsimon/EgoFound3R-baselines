#!/bin/bash
set -e
export WANDB_MODE=disabled
python3 -m scripts.ckpts
python3 -m scripts.submission --cfg config/eval/kitti.json
python3 -m scripts.submission --cfg config/eval/sintel.json
python3 -m scripts.submission --cfg config/eval/spring.json
