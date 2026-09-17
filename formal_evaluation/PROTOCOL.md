# Formal evaluation protocol

This directory evaluates canonical `predictions.npz` files written by the
adapters.  Baseline inference is separate from metric evaluation; window length
is defined exclusively by the supplied manifest.

## Shared rules

- Coordinates and depth passed to `egocentric_metrics` are metres.
- A canonical output must pass `common/schema.py` before it is evaluated.
- Undefined metrics remain `NaN`; aggregation excludes them and records the
  corresponding `*_undefined_window_count`.  It never converts an undefined
  precision, recall, or F1 into zero.
- EgoFound3R verifies its checkpoint SHA-256 before inference. Managed runs
  are registered through taskctl before submission; a task status of `done`
  alone does not prove that a physical `COMPLETE` file exists.

## Final EgoFound3R stride comparison

- Training tag: `final-root-depth-fusion-v2-dynamic-multirate-20260904`;
  source commit: `e73dcd8a51b0a06c1790fc18e1b1aa7b5180aea8`;
  inference commit: `8b0c44a721bded978373a3d9a8222b096d8c9930`.
- Use the registered step-1599 checkpoint with native BF16 weights and
  SHA-256 `f358de97ff0c9f9ba4f35f48a68f62582403580a2095f08a9362b94ad2416a6f`.
- The adapter defaults to `--global-stride 5`, independently of training
  config defaults. Evaluate five separate fixed strides, 1 through 5, with
  `global_anchor_phase = global_stride // 2`; there is no mixed evaluation.
  Both metadata and run provenance record the stride, phase, and model identity.
- Decode the typed multi-rate output using its native H/G fields. Refined H-axis
  cameras are already metric and clip-local. G-axis depth is scaled once with
  the runtime's scene factor and is valid only at the registered global anchors;
  non-anchor depth/intrinsics remain NaN, with false validity masks.
- Each stride uses the same 2,378 windows: H2O 283, HOT3D 400, ARCTIC 434,
  OakInk-v2 400, TACO 400, HOI4D 461. Keep checkpoint, GT cache, preprocessing,
  and metric code identical. Reuse each prediction for all applicable metrics;
  do not fabricate native vertex predictions from this model's marker output.
- Before formal submission, verify the default smoke's `COMPLETE` and
  `smoke_summary.json`, then validate a stride-1 single-window smoke. Each
  stride needs its own registered logical task and unique output root. Run
  independent single-GPU jobs only on taskctl-confirmed idle resources on
  5000/5001/6001. Deployment and formal results remain separate from local tests.

## Hand

- GT is reconstructed from `hand_pose_mano`; hand slots are left then right.
  It caches 21 OpenPose joints, 195 fixed MeshGraphormer level-1 MANO markers,
  and the full 778-vertex MANO mesh.
- Each available granularity reports raw, root-relative, per-frame PA, and
  window-global Sim(3) position errors, plus velocity and acceleration errors.
  Names are `MPJPE/MPJVE/MPJAE`, `MPMPE/MPMVE/MPMAE`, and
  `MPVPE/MPVVE/MPVAE`; velocities are mm/s and accelerations are mm/s² at the
  recorded `temporal_fps` (30 by default). Invalid neighbours never form a
  temporal difference.
- Raw joint `MPJPE` and marker `MPMPE` are always evaluated in camera space.
  Native world-space outputs such as HaWoR are transformed back with that
  method's own predicted `camera_c2w`; GT camera poses are not used for this
  conversion. If a world-space output has no predicted camera pose, its raw
  joint/marker point metrics are undefined instead of mixing coordinates.
- W/WA are hand-aligned world-space metrics. Camera-space predictions are first
  transformed to world space with the method's predicted `camera_c2w` when it
  exists. Camera-only methods without a predicted trajectory use GT
  `camera_c2w` as an explicit oracle input; reports and table labels must record
  that provenance. Native world-space methods use their native world hand
  geometry and must not substitute an identity camera fallback for a failed
  native world reconstruction.
- W fits one Sim(3) from predicted to GT hand points using the first two valid
  frames of the complete window, then applies that fixed transform to the full
  window. WA fits one Sim(3) using all valid hand points in the complete window.
  Fitting is performed before the frozen P95 mask; masking changes aggregation
  only and never triggers a refit. The same definition is applied independently
  at joint, marker, and vertex granularity when that geometry is available.
- WiLoR, HaWoR, PAD-Hand, and Dyn-HaMR provide native MANO-778 geometry.
  EgoFound3R's vertex metrics are explicitly marked `derived_from_195_markers`
  through the fixed MeshGraphormer upsampling matrix. ReViV4D remains 21-joint
  only.
- Presence uses `binary_metrics` on every frame, including frames with absent
  GT hands.  Undefined denominators yield `NaN`.

### Archived camera-trajectory alignment experiment

The superseded experiment fitted predicted camera-centre trajectories to GT
camera-centre trajectories, using fixed-scale SE(3) for W and Sim(3) for WA,
then applied the camera transform to the hands. Its empirical results were poor,
so this camera-trajectory alignment is diagnostic history rather than the formal
W/WA protocol. Results produced by that experiment must not replace the
first-two-frame W or full-window WA values defined above.

## Scene

- Poses are `camera_c2w` in the OpenCV convention.  Aligned camera ATE is
  `ate(..., alignment="similarity")`, i.e. trajectory RMSE after window-level
  Sim(3), not the legacy mean Euclidean trajectory error.
- Rotation is the mean geodesic error in degrees after expressing predicted
  and GT poses relative to their first valid frame, so a constant global world
  rotation does not dominate the score. The previous absolute-frame value is
  retained only as `camera_rot_absolute_error_deg` for diagnosis.
- `camera_pose_auc_30` is AUC@30° of the maximum of rotation and translation-
  direction errors after expressing all poses relative to the first valid frame.
- Relative-depth baselines use one explicit `median(GT/pred)` scale over all
  finite, strictly positive depth pairs in a window; metric-scale methods use
  scale one.  This scale policy is external because the metrics library does
  not infer scale.
- Depth is evaluated per available frame by `depth_metrics` and then averaged
  across finite frame values.  There is no legacy `GT > 0.01` cutoff or
  100-pixel minimum.

## Contact

- EgoFound3R joint and marker contact use frozen H2O scheme-2 object/interhand
  union targets, then intersect their supervision mask with `hand_valid`.
- Contact precision, recall, F1, counts, and AP call `binary_metrics` and
  `average_precision`; scores are thresholded at 0.5 for binary metrics.
- The registered contact baselines are S²Contact, ContactOpt, and InteractVLM.
  The joint-21 table contains EgoFound3R/S²Contact/ContactOpt; the marker-195
  table is EgoFound3R-only; InteractVLM is a separate right-hand vertex-778
  hcontact table. These are distinct protocols and are never cross-ranked.
