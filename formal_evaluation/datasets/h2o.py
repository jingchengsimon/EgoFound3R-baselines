from __future__ import annotations

import random
import struct
from pathlib import Path

from .base import FrameSequence


def _image_size(path: Path) -> tuple[int, int]:
    """Read PNG/JPEG dimensions without decoding pixels or adding dependencies."""
    with path.open("rb") as handle:
        head = handle.read(24)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", head[16:24])
        if head[:2] != b"\xff\xd8":
            raise ValueError(f"unsupported RGB image header: {path}")
        handle.seek(2)
        while True:
            marker = handle.read(2)
            if len(marker) != 2:
                break
            if marker[0] != 0xFF:
                continue
            while marker[1] == 0xFF:
                marker = bytes((0xFF, handle.read(1)[0]))
            if marker[1] in {0xD8, 0xD9}:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if marker[1] in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                             0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height, width = struct.unpack(">HH", handle.read(5)[1:5])
                return width, height
            handle.seek(length - 2, 1)
    raise ValueError(f"could not read image dimensions: {path}")


class H2ODatasetAdapter:
    """Read-only H2O split/sequence/RGB adapter; method preprocessing stays external."""

    name = "h2o"
    splits = {"test": "subject4_ego"}

    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)

    @staticmethod
    def _rgb_frames(rgb_dir: Path) -> list[Path]:
        return sorted(path for path in rgb_dir.iterdir()
                      if path.suffix.lower() in {".png", ".jpg", ".jpeg"})

    def sequence_ids(self, split: str, *, min_frames: int = 1) -> list[str]:
        try:
            subject = self.splits[split]
        except KeyError as error:
            raise ValueError(f"unsupported H2O split: {split}") from error
        sequences = []
        for rgb_dir in (self.root / subject).glob("*/*/cam4/rgb"):
            if len(self._rgb_frames(rgb_dir)) >= min_frames:
                sequences.append(rgb_dir.parent.parent.relative_to(self.root).as_posix())
        return sorted(sequences)

    def select_contiguous(self, split: str, count: int, *, seed: int = 0) -> FrameSequence:
        if count <= 0:
            raise ValueError("frame count must be positive")
        candidates = self.sequence_ids(split, min_frames=count)
        if not candidates:
            raise RuntimeError(f"no H2O {split} sequence has {count} frames")
        sequence = random.Random(seed).choice(candidates)
        paths = self._rgb_frames(self.root / sequence / "cam4/rgb")[:count]
        resolutions = [_image_size(path) for path in paths]
        return FrameSequence(
            dataset=self.name, split=split, sequence_id=sequence,
            frame_paths=tuple(paths), frame_ids=tuple(path.stem for path in paths),
            resolutions=tuple(resolutions),
        )
