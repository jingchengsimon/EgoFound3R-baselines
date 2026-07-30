#!/bin/bash
set -e
python3 -m scripts.train --cfg config/train/Tartan.json
python3 -m scripts.train --cfg config/train/Tartan-T.json
python3 -m scripts.train --cfg config/train/Tartan-T-TSKH.json
python3 -m scripts.train --cfg config/train/Tartan-T-TSKH-kitti.json
python3 -m scripts.train --cfg config/train/Tartan-T-TSKH-sintel.json
python3 -m scripts.train --cfg config/train/Tartan-T-TSKH-spring.json
