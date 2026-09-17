#!/usr/bin/env python3
"""Verify one exact detached execution checkout and record bounded evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def git(worktree: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(worktree), *args], text=True, timeout=90
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    head = git(args.worktree, "rev-parse", "HEAD")
    branch = git(args.worktree, "branch", "--show-current")
    status = git(args.worktree, "status", "--porcelain=v1")
    if head != args.expected_commit:
        raise ValueError(f"HEAD mismatch: {head}")
    if status:
        raise ValueError("worktree is dirty")
    payload = {
        "status": "complete",
        "worktree": str(args.worktree),
        "head": head,
        "branch": branch or None,
        "detached": not branch,
        "clean": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "COMPLETE").write_text("clean worktree verified\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
