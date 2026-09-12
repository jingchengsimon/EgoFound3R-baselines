# Registered artifact access

Read only for artifact reuse, coverage audits, or historical path recovery.

## Current artifact paths (2026-09-05)

- For reusable GT and prediction paths, use the exact catalog task
  `artifacts:<dataset>:baseline14:60f`, then `taskctl.py path --run-id <resolved-id>`.
  Narrow the second step with `--artifact gt-cache`, or
  `--artifact predictions --method <exact registry method>`.
- These catalogs are in `evaluation_task_registry.json`, cover all six datasets
  and all 14 registered methods, and explicitly record composite main/backfill or
  shard roots. Their scope is available historical 60-frame baseline predictions;
  pending final EgoFound3R stride evaluations are not completed prediction assets.
- `path` reads only exact catalog paths and checks GT identity, readable nonempty
  files, and complete prediction-window coverage. Blocked entries do not return
  usable prediction roots. Do not use a historical `done` state, parent directory,
  file timestamp, or newest submission as proof of a reusable complete artifact.
- Old GT run IDs are retained for provenance and rejected as superseded by path
  lookup. Legacy formal path queries point to the required catalog; resolve that
  exact catalog before use. Never retry fuzzy discovery, guessed alternate roots,
  or recursive searches to get around a missing or blocked catalog entry.
- Updating a catalog requires exact migration/run/recompute provenance plus a live
  coverage audit. Preserve old runs and remote assets. Missing methods stay explicitly
  blocked; do not substitute another method or an incomplete newer run.

- Before declaring a method's results missing, read the registered formal
  `report.json`, its validation report when present, and its exact prediction
  index. Audit every main/backfill/shard reference from that index. A partial
  inference directory is not proof that the completed result set is missing.
- `path --artifact reports --method <method>` is available for report-path checks.
  Distinguish completed report coverage, live prediction-file coverage, and an
  unresolved final-report reference. A label such as `v3` on a historical task
  does not establish a new model version or a changed scientific protocol.

## Authorized historical discovery

- Historical result audit exception (authorized 2026-08-26): for **H2O, HOT3D,
  ARCTIC, OakInk-v2, TACO, and HOI4D**, an operator may use read-only, bounded
  remote discovery below `/mnt/workspace/sjc/eval_artifacts` and the six exact
  `/mnt/workspace/sjc/DATA/<dataset>_contact_baseline/cache` roots. The audit
  may locate canonical prediction pairs (`predictions.npz` plus `metadata.json`),
  GT-cache indexes, geometry caches, and report JSON files. It must first print
  exact candidate roots and counts; it must not launch, stop, modify, delete,
  or register a task until candidates are verified. Every accepted historical
  root must then be registered idempotently in taskctl before reuse.
