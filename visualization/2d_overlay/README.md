# 2D overlay index

This directory is the entry point for the current 2D visualization work.

- `HANDOFF.md`: protocol, current status, artifact map, known gaps, and the
  next batch procedure.
- `manifest_registry.json`: machine-readable identities, counts, checksums,
  and tracked/local-only boundaries.
- `build_visibility60_selected114.py`: exact selector snapshot that produced
  the frozen 114-window identity from the 600 audited candidates.
- `bracketable600_20260914.json.gz`: exact 600-row completeness-audited input
  used by the visibility and non-overlap selector.
- `interactvlm_center_overlap_audit_20260915.json`: exact overlap audit between
  the frozen 114 centers and registered InteractVLM windows.

Current result roles:

1. `../overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/` is the
   verified frozen 114-clip **legacy 2x4** gallery. Its selection is
   authoritative; its point-style layout is superseded for future rendering.
2. `../overlay_2d_contact_face_3x4_example_20260917/` contains the current
   **3x4 face-level** review pilot. Only the H2O rank-17 PNG is the review
   target. No 114-clip 3x4 batch has been started.
3. `../overlay_2d_redesign_pilot_20260915/` and
   `../contact_vertex_four_method_example_20260915/` are superseded design
   experiments and must not be used as final effect references.

Rendered PNG/MP4 galleries, frame sidecars, per-clip reports, and ZIP bundles
remain local/artifact-storage outputs and are intentionally excluded from Git.
