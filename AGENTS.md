# EgoFound3R-baselines collaboration rules

## Git and DSW synchronization

- Local edits do not authorize `git commit`, `git push`, or a remote Git
  update. Only perform those operations when a human explicitly asks for a
  commit. A request to modify, test, run, deploy, or synchronize is not by
  itself commit authorization.
- Without explicit commit authorization, keep source changes local. Do not use
  a commit plus DSW `pull` as an implicit synchronization mechanism, and do not
  replace it with ad-hoc project-root copies. Report that DSW still has the
  previous source revision and request the exact promotion instruction when a
  remote run depends on the local change.
- When a human explicitly authorizes a commit, group work into one atomic,
  DSW-runnable change set: complete local fixes, static checks, and a small
  smoke test first; amend locally if needed and do not push intermediate work.
- At a reproducible experiment boundary, create one scope-specific commit and
  push it. DSW checkouts must be clean and may advance only with
  `git pull --ff-only` to that commit.
- Never rewrite a pushed commit. Follow-up defects use a small new fix commit
  so the actual experiment history remains traceable.
- Benchmark JSON, logs, checkpoints, and result directories stay outside Git.
  Every result must record the exact source commit SHA used to produce it.

## DSW connections

- DSW compute nodes share host `39.106.218.186`; connect as `root` with an
  explicit compute port: `ssh -o BatchMode=yes -p <5000|5001|6001>
  root@39.106.218.186`. The local `6001` SSH alias is equivalent only for that
  port. These ports are compute nodes, not OSS endpoints.
- Before any DSW state change, use a read-only probe on the selected node to
  confirm `hostname`, checkout branch and worktree cleanliness, GPU ownership
  and memory, destination free space, and the exact target path. Do not assume
  that a reachable node shares another node's GPU availability or process
  state.
- Keep DSW checkouts execution-only: change source and configuration on macOS,
  then use a clean checkout and `git pull --ff-only` only after an explicitly
  authorized commit/push. Do not edit, commit, merge, reset, or reconcile
  source on DSW. Do not remount or reconfigure OSSFS; report an unhealthy or
  stale mount instead.

## Large-file transfer via OSS

- For local macOS-to-DSW data, checkpoints, archives, and other large files,
  default to the authorized OSS relay: upload from macOS with `ossutil` to a
  unique object, then copy from the healthy DSW OSSFS mount to the resolved
  final destination. Do not use this rule for Git source synchronization.
- Before a transfer, verify the exact OSS bucket/prefix authorization, local
  endpoint, DSW OSSFS mount health, source size, destination free space, and
  that the final target will not be overwritten. Use a unique temporary object
  or incoming path; promote only after the expected byte size is present.
- If OSS availability, authorization, or a short throughput probe is worse
  than direct transfer, retain partial data and explicitly fall back to a
  resumable direct transfer. Never store OSS credentials in the repository,
  scripts, logs, or reports.

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
