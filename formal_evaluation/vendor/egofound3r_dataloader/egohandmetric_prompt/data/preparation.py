from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from zipfile import ZipFile


def validate_required_paths(root: Path, required: dict[str, str]) -> None:
    missing = [name for name, relative_path in required.items() if not (root / relative_path).exists()]
    if missing:
        raise ValueError(f"missing required TACO/HoloAssist component(s): {', '.join(missing)}")


def extract_verified_zip(archive: Path, destination: Path, *, required: dict[str, str]) -> None:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    temporary = destination.with_name(f".{destination.name}.extract-{uuid.uuid4().hex}")
    try:
        with ZipFile(archive) as handle:
            bad_member = handle.testzip()
            if bad_member is not None:
                raise ValueError(f"corrupt zip member: {bad_member}")
            handle.extractall(temporary)
        validate_required_paths(temporary, required)
        if destination.exists():
            raise FileExistsError(destination)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
