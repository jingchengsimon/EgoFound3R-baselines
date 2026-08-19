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
- No SHA/digest, smoke run, or scheduler submission is performed by this tree.

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
- W is unaligned world-coordinate error. WA2 fits one Sim(3) from the first
  two valid frames in a window; WA fits one Sim(3) from all valid window points.
  Camera-space predictions are transformed with predicted `camera_c2w` when
  present, otherwise explicitly with GT `camera_c2w` and marked as such in the
  window result. Native world outputs remain native world outputs.
- WiLoR, HaWoR, PAD-Hand, and Dyn-HaMR provide native MANO-778 geometry.
  EgoFound3R's vertex metrics are explicitly marked `derived_from_195_markers`
  through the fixed MeshGraphormer upsampling matrix. ReViV4D remains 21-joint
  only.
- Presence uses `binary_metrics` on every frame, including frames with absent
  GT hands.  Undefined denominators yield `NaN`.

## Scene

- Poses are `camera_c2w` in the OpenCV convention.  Aligned camera ATE is
  `ate(..., alignment="similarity")`, i.e. trajectory RMSE after window-level
  Sim(3), not the legacy mean Euclidean trajectory error.
- Rotation is `extrinsic_metrics` geodesic rotation converted from radians to
  degrees.
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
