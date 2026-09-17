# 2D overlay: visibility-60 maximum non-overlap 114

Status: **verified**

- Frozen selection: 114 center frames; selection SHA-256 `6a95a1c01288fbe91589345e013bd7c96bcb038ebef2414bea140240824f4d2f`.
- Criterion: both hands valid at the first and last display frame; each internal combined missing run is at most 60 frames; then maximum non-overlapping interval selection per dataset/sequence.
- Outputs: 114 center-frame PNGs and 114 actual-length MP4s at 30 FPS; 12559 decoded display frames in total.
- Dataset clips: ARCTIC 9, H2O 26, HOI4D 0, HOT3D 1, OakInk-v2 18, TACO 60.
- Interpolation: 8 clips, 310 hand-side frames. Commit `8fc061a615895bd3b5a556f7387bae306e32d9db` visualization postprocess (`_camera_space_smooth_fill` plus root-relative `[1,2,1]/4` local-shape smoothing). Source predictions and strict metric validity remain unchanged; no inference rerun or root-UV re-solve.
- Center PNG interpolation: 2 centers (arctic rank 48, oakink_v2 rank 53).
- Layout: 2x4, top GT and bottom Ego stride5 + display fill; columns geometry, contact distance, contact, visibility.
- Contact distance: `vertex_contact_distance`, Turbo scale 0-50 mm. Visibility: predicted `marker_visibility` spatially mapped 195 to 778, threshold 0.5; it is not RGB-observed visibility.
- All videos were re-probed for decoded frame count and 30 FPS; frame sidecars match the frozen source-frame order, center, and first/last identities; non-overlap is verified.

See `selected_manifest.jsonl`, `selection_summary.json`, `render_manifest.jsonl`, and `completeness_audit.json` for machine-readable provenance.
