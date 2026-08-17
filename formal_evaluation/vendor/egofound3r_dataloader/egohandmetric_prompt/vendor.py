from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VendorRoots:
    repo_root: Path
    vggt_omega: Path
    wilor: Path


def get_repo_root() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    git_marker = package_root / ".git"
    if git_marker.is_file():
        gitdir_line = git_marker.read_text().strip()
        if gitdir_line.startswith("gitdir: "):
            gitdir = Path(gitdir_line.removeprefix("gitdir: ").strip())
            if not gitdir.is_absolute():
                gitdir = (git_marker.parent / gitdir).resolve()
            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                return gitdir.parents[2]
    return package_root


def get_vendor_roots() -> VendorRoots:
    repo_root = get_repo_root()
    return VendorRoots(
        repo_root=repo_root,
        vggt_omega=repo_root / "other" / "vggt-omega",
        wilor=repo_root / "other" / "WiLoR",
    )


def ensure_vendor_paths() -> VendorRoots:
    roots = get_vendor_roots()
    for path in (roots.vggt_omega, roots.wilor):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return roots
