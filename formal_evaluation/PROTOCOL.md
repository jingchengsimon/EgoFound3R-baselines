# Formal evaluation protocol

This directory evaluates canonical `predictions.npz` files written by the
adapters.  Baseline inference is separate from metric evaluation.  Every
window is twelve frames unless the supplied manifest explicitly says otherwise.

## Shared rules

- Coordinates and depth passed to `egocentric_metrics` are metres.
- A canonical output must pass `common/schema.py` before it is evaluated.
- Undefined metrics remain `NaN`; aggregation excludes them and records the
  corresponding `*_undefined_window_count`.  It never converts an undefined
  precision, recall, or F1 into zero.
- No SHA/digest, smoke run, or scheduler submission is performed by this tree.

## Hand

- GT is reconstructed only from `hand_pose_mano`; hand slots are left then
  right, and joints use the 21-point OpenPose ordering.
- All hand metrics are compared in camera coordinates.  World-space baseline
  outputs are converted with their predicted `camera_c2w`.
- MPJPE, root-relative MPJPE, and PA-MPJPE call `mano_metrics` with root index
  zero.  PA is one Sim(3) per valid frame.
- `hand_*_sim3_mpjpe` calls `world_aligned_mpjpe(mode="all",
  chunk_length=T)`: exactly one Sim(3) fit over every valid joint in the
  evaluation window.
- Presence uses `binary_metrics` on every frame, including frames with absent
  GT hands.  Undefined denominators yield `NaN`.

## Scene

- Poses are `camera_c2w` in the OpenCV convention.  Aligned camera ATE is
  `ate(..., alignment="similarity")`, i.e. trajectory RMSE after window-level
  Sim(3), not the legacy mean Euclidean trajectory error.
- Rotation is `extrinsic_metrics` geodesic rotation converted from radians to
  degrees.
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
- S²Contact/ContactOpt (21 points) and InteractVLM (778 vertices) retain their
  own GT projection adapters.  Their predictions must be serialized to the
  same probability/target/mask form before this metric layer is used; their
  F1 values are not cross-protocol rankable.
