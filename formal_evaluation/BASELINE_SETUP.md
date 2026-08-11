# Pending baseline setup

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
  --conda-executable CONDA --output-root OUTPUT_ROOT
```

The adapter invokes the official WiLoR front end and PAD reverse diffusion,
then exports its refined camera-space 21-joint prediction. It does not treat a
plain WiLoR result as PAD-Hand.

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
     --cosmos-dir COSMOS_ROOT --sequence SEQUENCE --window-id WINDOW_ID --output-root OUTPUT_ROOT
   ```

`tok_body` and `tok_gaze` are explicitly excluded. ReViV's camera remains in
its RGB-predicted canonical frame: the adapter never reads a GT first-camera
pose sidecar for anchoring.
