#!/bin/bash
set -e
python3 -m scripts.ckpts
python3 -m scripts.evaluate --cfg config/eval/kitti.json
python3 -m scripts.evaluate --cfg config/eval/sintel.json
python3 -m scripts.evaluate --cfg config/eval/spring.json
