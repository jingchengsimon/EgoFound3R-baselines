"""Publish complete RGB cache files and tolerate transient shared-file reads."""
import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Capture before model imports can replace Image.open with an auto-install hook.
_PIL_OPEN = Image.open


def read_rgb(path, attempts=4):
    path = Path(path)
    for attempt in range(attempts):
        try:
            with _PIL_OPEN(io.BytesIO(path.read_bytes())) as image:
                return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
        except (OSError, ValueError) as error:
            if attempt + 1 == attempts:
                raise OSError(f"RGB cache decode failed after {attempts} reads: {path}") from error
            time.sleep(0.25 * (2 ** attempt))


def atomic_publish(path, writer):
    """Keep an existing cache readable until its replacement is fully closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".incoming-", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
