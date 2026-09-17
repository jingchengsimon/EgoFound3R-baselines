"""H.264 mp4 writer through an ffmpeg rawvideo pipe."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class VideoWriter:
    def __init__(self, path: Path, fps: int = 30):
        self.path = Path(path)
        self.fps = fps
        self.process = None
        self.frames = 0

    def _start(self, width: int, height: int):
        width += width % 2
        height += height % 2
        self.size = (width, height)
        self.process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
             "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
             "-preset", "medium", "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE)

    def add(self, image):
        import numpy as np
        array = np.asarray(image.convert("RGB"))
        if self.process is None:
            self._start(array.shape[1], array.shape[0])
        if (array.shape[1], array.shape[0]) != self.size:
            from PIL import Image
            image = image.resize(self.size, Image.Resampling.LANCZOS)
            array = np.asarray(image.convert("RGB"))
        self.process.stdin.write(array.tobytes())
        self.frames += 1

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            self.process = None
        return self.frames
