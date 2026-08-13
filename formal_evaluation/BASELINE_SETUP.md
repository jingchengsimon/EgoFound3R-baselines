# Pending baseline setup

## Canonical DSW runtime registry

Before any baseline setup, resolve the method from
[`config/baseline_runtime_registry_dsw.json`](config/baseline_runtime_registry_dsw.json).
It is the single source of truth for the fixed 14-method roster, exact DSW
paths, expected consumer mappings, Python environments, and current blockers.
Do not replace a registered path with a recursive/fuzzy-match result. On DSW,
check the registered paths with:

```bash
python formal_evaluation/validate_runtime_registry.py --method METHOD --strict
```

## Current verified runtime notes

- HaWoR: use `benchmark_hawor_500.py` for the loaded-once 500-frame pipeline.
  It reports detector/tracker, HaWoR, DROID-SLAM, Metric3D, infiller/world
  conversion and MANO stages separately. The manifest adapter accepts
  `--rgb-dir-template` for datasets whose RGB layout differs from H2O.
- Dyn-HaMR: use `/mnt/workspace/sjc/envs/dyn_hamr/bin/python` and add Dyn's
  DROID Python directories plus the two HaWoR sm_90 extension directories from
  the registry to `PYTHONPATH`. A 128-frame H2O smoke completed HaMeR,
  DROID-SLAM, camera export and reduced-iteration optimization. The sole speed
  entry is `benchmark_dyn_hamr_500.py`: it accepts raw H2O RGB only and runs
  loaded-once YOLO, HaMeR, DROID-SLAM and Dyn optimization for every warm-up
  and trial. It reports only whole-pipeline trial times, median seconds and
  500-output-frame FPS; it does not publish component-stage FPS. It rejects
  frame counts other than 500 and has no tracks/cameras input, so a
  prepared-cache stage benchmark cannot be mistaken for full FPS.
- S²Contact/ContactOpt: `benchmark_contact_500.py` times real checkpoint
  forwards from prepared hand/object geometry already in GPU memory. The large
  H2O pickle files are input geometry, not cached model predictions. New
  datasets need a builder that emits the same geometry contract.
- LingBot-Map: use `/mnt/workspace/sjc/envs/lingbot_map/bin/python`. The offline
  Torch and torchvision wheels and the exact long checkpoint are registered;
  the runner uses SDPA, so FlashInfer is optional.
- PAD-Hand speed: `benchmark_pad_hand_500.py` runs detector, bundled WiLoR and
  PAD refinement in one process with all three models loaded once. Launch it
  with the registered WiLoR interpreter and expose only PAD's `openmesh`
  site-packages directory through `PYTHONPATH`. For 500 outputs, PAD processes
  32 non-overlapping 16-frame windows (512 PAD inputs); the final input frame is
  repeated 12 times and the padded outputs are discarded.

The three adapters below only prepare or convert inference outputs. They do not
download licensed assets, submit jobs, or run formal metrics.

## Dyn-HaMR

1. Clone the official repository with submodules and run its `prepare.sh`.
2. Add the separately licensed `MANO_RIGHT.pkl`; generate the eight BMC arrays
   required by the official project.
3. Make an H2O context video with
   `python formal_evaluation/prepare_h2o_context.py ... --context-frames 128`;
   use `frame_opts.fps=30` when invoking Dyn-HaMR.
4. Run Dyn-HaMR on that RGB context, then save a normalized `.npz`
   with `hand_joints_world` (`T,2,21,3`), `hand_valid` (`T,2`), and
   `camera_c2w` (`T,4,4`).
5. Import a window with
   `formal_evaluation/hand/adapters/run_dyn_hamr_baseline.py --native-predictions ... --joints-key hand_joints_world --coordinate-space world --camera-key camera_c2w ...`.

The final conversion command must provide `--frame-indices` when the native
sequence has more frames than the formal 12-frame window.

## PAD-Hand

The released PAD-Hand tree must include `checkpoints/pad_hand.pt`,
`assets/processed_MANO_{RIGHT,LEFT}.pkl`, and the WiLoR detector/model
checkpoints required by `wilor_inference.py`. It needs two compatible Python
environments: PAD-Hand itself and WiLoR. Prepare at least 16 contiguous frames
(60 is convenient when sharing the ReViV context), then run:

```bash
python formal_evaluation/hand/adapters/run_pad_hand_baseline.py --phase smoke \
  --manifest MANIFEST.json --methods-config formal_evaluation/config/methods_v1.json \
  --prepared-dir PREPARED_DIR --source-root PAD-Hand \
  --wilor-python /mnt/workspace/sjc/envs/egofound3r/bin/python --output-root OUTPUT_ROOT
```

The adapter invokes the official WiLoR front end and PAD reverse diffusion,
then exports its refined camera-space 21-joint prediction. It does not treat a
plain WiLoR result as PAD-Hand. Run the adapter itself with
`/mnt/workspace/sjc/miniconda3/envs/pad_hand_h20/bin/python`; the old named
`wilor` environment is not used because it has no Torch installation.

## ReViV4D: 3R plus hand only

1. Obtain the official ReViV checkpoint set and matching Cosmos tokenizer;
   accept the NVIDIA tokenizer license before downloading it.
2. Produce one two-second RGB context per requested H2O window:

   ```bash
   python formal_evaluation/prepare_h2o_context.py --phase pilot --manifest MANIFEST.json \
     --data-root H2O_ROOT --sequence SEQUENCE --window-id WINDOW_ID --output-dir PREPARED_DIR
   ```

3. Run and convert only `tok_cam`, `tok_depth`, `tok_lhand`, and `tok_rhand`:

   ```bash
   python formal_evaluation/scene/adapters/run_reviv4d_baseline.py --phase pilot \
     --manifest MANIFEST.json --methods-config formal_evaluation/config/methods_v1.json \
     --prepared-dir PREPARED_DIR --source-root REVIV_ROOT --checkpoint-root REVIV_CKPT_ROOT \
     --cosmos-dir COSMOS_DV8_ROOT --hand-cosmos-dir COSMOS_DV4_ROOT \
     --sequence SEQUENCE --window-id WINDOW_ID --output-root OUTPUT_ROOT
   ```

`tok_body` and `tok_gaze` are explicitly excluded. ReViV's camera remains in
its RGB-predicted canonical frame: the adapter never reads a GT first-camera
pose sidecar for anchoring.

For ReViV-native 500-target-frame speed, use
`formal_evaluation/benchmark_reviv4d_500.py`. It runs nine contiguous 60-frame
clips (540 model-input frames, of which the final 40 are context only), not 42
independent H2O 12-frame windows. Resolve every path from the registry entry
below; the `--pythonpath` overlay is required until a persistent ReViV
environment is provisioned.

```bash
PYTHONPATH=/tmp/reviv4d_smoke_deps_20260813 \
/mnt/workspace/sjc/envs/egofound3r/bin/python formal_evaluation/benchmark_reviv4d_500.py \
  --data-root /mnt/workspace/sjc/DATA/H2O/h2o_data \
  --output-dir /mnt/workspace/sjc/artifacts/reviv4d_h2o500_TIMESTAMP \
  --source-root /mnt/workspace/sjc/external/reviv4d \
  --checkpoint-root /mnt/workspace/sjc/external/reviv4d/reviv_checkpoints/metric_depth \
  --scene-cosmos-dir /mnt/workspace/sjc/external/reviv4d/Cosmos/checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16 \
  --hand-cosmos-dir /mnt/workspace/sjc/external/reviv4d/Cosmos/checkpoints/Cosmos-0.1-Tokenizer-DV4x8x8 \
  --python /mnt/workspace/sjc/envs/egofound3r/bin/python --cuda-visible-devices 7
```
