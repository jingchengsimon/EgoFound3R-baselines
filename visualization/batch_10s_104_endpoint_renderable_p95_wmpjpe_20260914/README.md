# 10s endpoint-renderable 3D gallery v5

This directory freezes the selection, source alignment, renderer, and runtime inputs for the canonical 104-segment 3D gallery.

- Result summary: [`../3D_VISUALIZATION_RESULT_SUMMARY_20260916.md`](../3D_VISUALIZATION_RESULT_SUMMARY_20260916.md)
- Handoff: [`../3D_VISUALIZATION_HANDOFF_20260916.md`](../3D_VISUALIZATION_HANDOFF_20260916.md)

The 104 segments are a maximum non-overlapping set selected from 602 10-second candidates. A candidate passes only when both hands have renderable Ego and GT predictions at the first and last frame: `hand_valid` is true and all 195 markers are finite. Interior gaps are allowed because they have valid anchors on both sides. The endpoint gate reads complete prediction caches and never applies the Joint8 P95 per-frame evaluation mask.

`endpoint_audit.jsonl` records all 602 candidates, `eligible_manifest.jsonl` contains the 319 passing candidates, and `selected_manifest.jsonl` contains the 104 selected segments. `selected_manifest_hydrated.jsonl.gz` is the deployed frozen input for the 520 constituent 60-frame windows. Its uncompressed SHA-256 is `e88ac84845a5a61e6e472b12d160816ef0d8a48eaeb0e2085985335ff33c92a5`.

The v5 renderer is `render_aux_pilot.py`. The historical filename is retained because registered runtime specs reference it; it supports both pilot and full batches.
