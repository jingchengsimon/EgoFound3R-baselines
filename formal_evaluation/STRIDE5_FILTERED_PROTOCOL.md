# Latest stride5 frozen filtering diagnostics

Run `python -m formal_evaluation.analyze_stride5_filtered --spec SPEC --output-root NEW_RESULTS --scratch-root NEW_CPFS_SCRATCH` on CPU.

The two schemes are joint MPJPE/RR P95 and joint-eight P97.5. P99 is disabled. Each dataset uses its own newly computed linear empirical quantiles. Left/right filtering scores use the maximum finite valid hand score; NaN never triggers exclusion. A single frame mask is applied to all three hand granularities.

The 2378 new stride5/phase2 60-frame windows are frozen once. Each geometry's PA/Sim3/W/WA fits use full unfiltered supports; aggregation cannot refit. W uses original window frames 0 and 1, not the first surviving frames. W/WA use GT-camera oracle coordinates and must be marked †. The filtered tables are diagnostics, not replacements for full formal metrics. Fixed unfiltered baseline CSVs are comparison references, not matched-mask baselines.

Per-dataset output includes the two masks with frame identity and reason codes, window metrics, unfiltered frozen reference, and support counts. Summary records index hashes, checkpoint provenance, thresholds and coverage. Frozen NPZ goes to CPFS scratch. COMPLETE requires 2378 frozen windows, 4756 scheme-window aggregations and all six comparison tables.

A missing final prediction index waits up to 12 hours. Put datasets already ready first in the spec. Existing predictions, reports, and previous diagnostic output are read-only. Launch through the registered taskctl interface; use fresh results and scratch paths on a real rerun.
