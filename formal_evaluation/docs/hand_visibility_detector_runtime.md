# Hand Visibility Detector baseline

The user-authorized extension `hand_visibility_detector` uses the official default WiLoR visibility head. It does not change the historical baseline14 roster.

## Frozen upstream assets

- Source: https://github.com/ryhara/hand_visibility_detector/commit/6321d62fb7617cf504d1de6e4d9ecda7aadfb989
- WiLoR-mini source: `ebec42f94c389070cdd7dda6fd1bf0b4a659c960`, as pinned by the upstream `uv.lock`.
- Visibility weights: https://huggingface.co/ryhara/hand-visibility-detector/tree/941b791bcba4a0bb381c325c225f56e0a80cf98f (`best.pt`).
- Backbone/detector/MANO assets: https://huggingface.co/warmshao/WiLoR-mini/tree/b00adea9a6843bbb4c9042109c5eb29ab2a59dea

Both model repositories were public and ungated at setup. Training datasets are not required for inference. HaMeR and `best_without_ego4d.pt` are distinct optional variants, not this baseline. Exact runtime paths, sizes and hashes are owned by `config/baseline_runtime_registry_dsw.json`.

## Adapter contract

`hand/adapters/run_hand_visibility_detector.py` consumes the existing six-dataset `window_input.json`. No dataset-specific adapters are needed. Only RGB paths and original frame identity are consumed; GT geometry and camera extrinsics are not used. Each full frame is resized with PIL bilinear to 256 by 256, followed by the official 256-square hand crop and its internal WiLoR 256-by-192 center crop. RGB is converted to BGR only at the Ultralytics detector predict boundary (including tracking); pose and visibility crops continue to consume the original RGB image.

The adapter emits only `hand_visibility[T,2,21]` probabilities and `hand_valid[T,2]`. Canonical slots are left then right; joints are wrist followed by thumb/index/middle/ring/pinky, four per finger. Highest detector confidence selects among multiple hands of the same side. Missing detections stay invalid with NaN probabilities. Existing output directories are rejected.

This supports **Joint visibility only**. The 21 probabilities cannot supply 195-marker visibility or dense vertex visibility. No geometric reconstruction capabilities are advertised. The official training visibility labels and our geometric GT visibility protocol differ; report that distinction when comparing scores. The current evaluator masks by valid predictions, so detection coverage must accompany visibility scores.

## Readiness boundary

Setup checks asset hashes, dependency imports, official pipeline CPU initialization and visibility-head strict loading. Adapter tests check side selection and canonical schema. The subsequent user-authorized 2026-09-10 validation executed one exact 60-frame window for each of H2O, HOT3D, ARCTIC, OakInk-v2, TACO and HOI4D on node 6001 GPU6. All six outputs passed frame-identity, 256-by-256 preprocessing and canonical-schema checks. Five windows had valid detections; HOI4D had none, so the strict nonempty-hand smoke gate remains failed for that window. Its visibility F1 remains NaN with one undefined window and zero valid joint samples. This is a single-window runtime check, not a full benchmark result.

Inference provenance: `smoke-six-egoforce-hvd-single60-452d029fb23f`. The six-window metric report completed under `metrics-six-egoforce-hvd-single60-eabe22b3d3b7`, at `/mnt/workspace/sjc/DATA/eval_artifacts/egoforce_hvd_metrics_20260910/hand_visibility_detector_report.json`. Every dataset has exactly one report window and there are no missing prediction windows. The same run validates EgoForce Joint, 195-marker and 778-vertex camera-space metrics; world-camera W/WA metrics remain unsupported without predicted camera extrinsics.

```bash
/mnt/workspace/sjc/envs/hand_visibility_6321d62/bin/python \
  formal_evaluation/hand/adapters/run_hand_visibility_detector.py \
  --phase smoke --window-input /absolute/path/window_input.json \
  --checkpoint /mnt/workspace/sjc/models/hand_visibility_941b791/best.pt \
  --output-root /new/unique/output --device cuda:0
```

Any future execution requires taskctl registration and the usual resource gates.

## RGB detector correction follow-up

Commit `831f314` corrects RGB-to-BGR only at the YOLO boundary. The same HOI4D 60-frame window was rerun at 256-by-256 on node 5001 GPU4. HVD and EgoForce each recovered 34 valid right-hand predictions from zero; their metric reports completed. Run: `rerun-hoi4d-egoforce-hvd-rgbfix60-3d50ef87a758`, output `/mnt/workspace/sjc/DATA/eval_artifacts/egoforce_hvd_rgbfix_20260910`. The earlier six-window results above remain historical evidence from the pre-fix adapter; the corrected rerun covers HOI4D only.
