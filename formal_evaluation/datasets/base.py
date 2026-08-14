from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameSequence:
    dataset: str
    split: str
    sequence_id: str
    frame_paths: tuple[Path, ...]
    frame_ids: tuple[str, ...]
    resolutions: tuple[tuple[int, int], ...]

    @property
    def resolution_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for width, height in self.resolutions:
            key = f"{width}x{height}"
            counts[key] = counts.get(key, 0) + 1
        return counts
