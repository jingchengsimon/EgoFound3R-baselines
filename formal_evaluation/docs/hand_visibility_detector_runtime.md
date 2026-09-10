# Hand Visibility Detector baseline

The user-authorized extension `hand_visibility_detector` uses the official default WiLoR visibility head. It does not change the historical baseline14 roster.

## Frozen upstream assets

- Source: https://github.com/ryhara/hand_visibility_detector/commit/6321d62fb7617cf504d1de6e4d9ecda7aadfb989
- WiLoR-mini source: `ebec42f94c389070cdd7dda6fd1bf0b4a659c960`, as pinned by the upstream `uv.lock`.
- Visibility weights: https://huggingface.co/ryhara/hand-visibility-detector/tree/941b791bcba4a0bb381c325c225f56e0a80cf98f (`best.pt`).
- Backbone/detector/MANO assets: https://huggingface.co/warmshao/WiLoR-mini/tree/b00adea9a6843bbb4c9042109c5eb29ab2a59dea

Both model repositories were public and ungated at setup. Training datasets are not required for inference. HaMeR and `best_without_ego4d.pt` are distinct optional variants, not this baseline. Exact runtime paths, sizes and hashes are owned by `config/baseline_runtime_registry_dsw.json`.

## Adapter contract

`hand/adapters/run_hand_visibility_detector.py` consumes the existing six-dataset `window_input.json`. No dataset-specific adapters are needed. Only RGB paths and original frame identity are consumed; GT geometry and camera extrinsics are not used. Each full frame is resized with PIL bilinear to 256 by 256, followed by the official 256-square hand crop and its internal WiLoR 256-by-192 center crop.

The adapter emits only `hand_visibility[T,2,21]` probabilities and `hand_valid[T,2]`. Canonical slots are left then right; joints are wrist followed by thumb/index/middle/ring/pinky, four per finger. Highest detector confidence selects among multiple hands of the same side. Missing detections stay invalid with NaN probabilities. Existing output directories are rejected.

This supports **Joint visibility only**. The 21 probabilities cannot supply 195-marker visibility or dense vertex visibility. No geometric reconstruction capabilities are advertised. The official training visibility labels and our geometric GT visibility protocol differ; report that distinction when comparing scores. The current evaluator masks by valid predictions, so detection coverage must accompany visibility scores.

## Readiness boundary

Setup checks asset hashes, dependency imports, official pipeline CPU initialization and visibility-head strict loading. Adapter tests check side selection and canonical schema. Six real dataset windows and GPU forward validation are deferred at the user's request; CPU readiness is not a formal evaluation result.

```bash
/mnt/workspace/sjc/envs/hand_visibility_6321d62/bin/python \
  formal_evaluation/hand/adapters/run_hand_visibility_detector.py \
  --phase smoke --window-input /absolute/path/window_input.json \
  --checkpoint /mnt/workspace/sjc/models/hand_visibility_941b791/best.pt \
  --output-root /new/unique/output --device cuda:0
```

Any future execution requires taskctl registration and the usual resource gates.
