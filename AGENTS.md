# EgoFound3R-baselines collaboration rules

## Baseline runtime registry

- The fixed 14-method roster, including the deliberate exclusion of CHOI, and
  every DSW method path, environment, checkpoint, consumer mapping, and
  blocker are authoritative only in
  `formal_evaluation/config/baseline_runtime_registry_dsw.json`. Do not add,
  remove, rename, or substitute methods or use recursive/fuzzy path discovery
  to bypass a missing registered asset.
- Before a baseline setup or run, resolve its registry entry and execute
  `python formal_evaluation/validate_runtime_registry.py --method <method>
  --strict`. A missing registered requirement is a blocker until an explicitly
  authorized, reversible setup action resolves it.

## Results and resource protection

- Use a new, uniquely named output directory for every run. Do not overwrite,
  delete, or modify checkpoints, formal results, existing baseline outputs, or
  another task's logs. Keep benchmark JSON, logs, and artifacts outside Git.
- Before a GPU run, verify the selected physical GPU is idle and identify its
  owner/processes. Use only confirmed-free resources and release models/CUDA
  cache after a method; a failed method must be recorded precisely without
  stopping independent methods.

## Dataset and evaluation contract

- Dataset adapters are read-only: they own split discovery, original sequence
  and frame identity, RGB paths, and resolution audit, but never resize or
  rewrite dataset files. Method adapters own their official preprocessing.
- A canonical prediction must pass `formal_evaluation/common/schema.py` before
  evaluation. Keep undefined metric values as `NaN`, exclude them only through
  the documented aggregation, and record their undefined-window counts; never
  coerce undefined precision, recall, or F1 to zero.
