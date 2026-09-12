# EgoFound3R-baselines collaboration rules

## Scope and completion

Read only the instructions relevant to the task. Complete the requested change and necessary
validation, then stop. Do not expand into unrelated cleanup, testing, or protocol changes.
A status query returns one snapshot; waiting or ongoing monitoring requires an explicit request.

## DSW access and resource checks

- Use the user's external terminal and `formal_evaluation/taskctl.py` for DSW/OSS access.
  With autonomous remote authorization and no terminal connector, approved sandbox-external
  execution (`require_escalated`) is the fallback. Never use ordinary sandbox SSH or interpret
  sandbox network failures as remote state.
- Before the first remote query, identify the target node from the exact registry/task or user's
  target. Run `python3 formal_evaluation/taskctl.py resources --nodes <target-node>` and retain
  its JSON as the handshake. Probe multiple nodes only for an explicit fleet query or placement
  decision. If the node cannot be resolved, report the ambiguity rather than probing the fleet.
- Reuse a successful connection within the conversation. Only repair a failed connection with the
  selected configured SSH alias; do not always check/connect to `pai-5000` for another node.
  If the external channel is unavailable, stop remote work and report the missing channel.
- For registered task state, resources, paths, and control, taskctl is the sole entrypoint. Do not
  bypass it with raw SSH/process/filesystem searches. Separately authorized non-task operations
  use the verified project connection; no remounting or reconfiguration of OSSFS.
- Entrances: `5000`, `5001`, `6001` use `/mnt/workspace`; `4091` uses `/mnt/cpfs`.
  If a probe conflicts with this mapping, report the conflict and do not discover artifacts via
  the conflicting mount. Do not infer one node's resources from another node.
- Before a state change, refresh the relevant checks on the selected node: branch/cleanliness
  for Git updates, GPU ownership for GPU runs, and exact destination/free space for writes.

## Runtime and scientific invariants

- The fixed 14-method roster (excluding CHOI), runtime environments, checkpoints, and consumer
  mappings are authoritative in `formal_evaluation/config/baseline_runtime_registry_dsw.json`.
  Before setup/run, resolve the method and run
  `python formal_evaluation/validate_runtime_registry.py --method <method> --strict`.
  Missing registered requirements block that method; do not guess substitutes or fuzzy paths.
- Every run uses a unique output directory. Never overwrite/delete checkpoints, formal results,
  existing outputs, or another task's logs. Benchmark JSON, logs, and artifacts stay outside Git.
- Before a GPU run, verify the physical GPU exists and has no occupying processes; record owners.
  Free VRAM is observational, not a fixed threshold. Release models/CUDA cache after a method;
  record individual failures without stopping independent methods.
- Dataset adapters are read-only: preserve splits, sequence/frame identity, RGB files and original
  resolution. Official preprocessing belongs to method adapters; never rewrite dataset files.
- Canonical predictions must pass `formal_evaluation/common/schema.py`. Preserve undefined
  metrics as `NaN`, use documented aggregation, and record undefined-window counts.

## Registered tasks

- Runtime assets use the runtime registry; task identity and paths use
  `formal_evaluation/config/evaluation_task_registry.json`. Live status comes from the exact
  registered state file, not a directory timestamp, inferred PID, or partial output.
- First resolve an exact logical task ID (or exact dataset/method-set fields) with taskctl;
  retain the returned run ID and path, then use that run ID for `inspect` or `path`.
  Reuse this verified ID for subsequent read-only queries in the conversation. Re-resolve when
  identity changes, a handle is stale, or before `start`, `pause`, or `resume`.
- `inspect` currently includes a registered-node capacity probe; there is no skip-storage CLI
  option. Do not invent one or replace it with another command to avoid the probe. Avoid extra
  capacity probes for the same read-only question. Use `watch` only when monitoring is requested.
- Register every managed submission before launch, without an extra permission round. Reuse the
  returned ID for identical task/output/state/scheduler/job keys. `created=false` never authorizes
  duplicate launch. A real rerun needs a new output root and ID. Read/pause/resume do not register.
- Missing/non-unique tasks, stale/missing handles, and missing paths are blockers. Report the
  structured error once; do not guess paths, PIDs, or methods. State changes use exact run IDs.

## Artifact reuse and audits

For GT/prediction/report reuse, coverage checks, or historical recovery, read
[artifact-access.md](agent_policy/artifact-access.md). It owns catalog lookup, composite shard
coverage, superseded assets, and the bounded historical-audit exception. Preserve formal coverage,
protocol, and completion evidence; a running process or a partial directory is not completion.

Personal host configuration is maintained in `agent_policy/global-AGENTS.md`; it is an installation
source for the user's host, not an instruction to collaborators to change their configuration.
