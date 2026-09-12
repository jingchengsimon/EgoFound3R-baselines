# Personal instruction sources

This directory versions the user's host instructions and narrow patches to installed skills.
It does not change training/evaluation code, protocols, results, or collaborator configuration.

- `global-AGENTS.md`: source for the user's `~/.codex/AGENTS.md`.
- `ponytail-config.json`: merge `defaultMode: off` into the user's Ponytail config; preserve other
  settings. This disables default activation, not explicit skill use.
- `skill-updates/*.patch`: exact patches for the installed versions named in each patch header.
  The Ponytail skill is explicit-only and task-scoped. Other patches narrow unnecessary reading,
  rendering, or waiting while retaining tool contracts and correctness checks.
- `artifact-access.md`: project artifact lookup reference, loaded only when needed.

After plugin updates, check whether the upstream fix supersedes a patch before applying it.
Do not blindly apply patches to different versions or synchronize whole plugin caches. Existing
sessions may retain previously loaded instructions; use a new task to load updated descriptions.
Remote synchronization of this directory alone does not install host-level instructions there.

Validation for this change is limited to file scope, skill metadata, references, and preserving
scientific/operational invariants. No comparative benchmark or training/evaluation run is required.
