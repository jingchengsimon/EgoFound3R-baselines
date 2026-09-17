"""Stream locally rendered RGB frames into an atomically published MP4."""

import json
import os
import subprocess
import tempfile
from pathlib import Path


def encode_mp4(frames, target: Path, *, width: int, height: int, frame_count: int = 300, fps: int = 30):
    """Encode exactly frame_count PIL images; leave target untouched on failure."""
    if width % 2 or height % 2 or min(width, height, frame_count, fps) <= 0:
        raise ValueError("MP4 dimensions must be positive and even; count/fps must be positive")
    target = Path(target)
    if target.suffix.lower() != ".mp4" or target.exists():
        raise ValueError("target must be a new .mp4 file")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.stem + ".", suffix=".partial.mp4", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "23",
               "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    try:
        for image in frames:
            if count >= frame_count:
                raise ValueError("renderer produced too many frames")
            if image.mode != "RGB" or image.size != (width, height):
                raise ValueError(f"frame {count} is not RGB {width}x{height}")
            proc.stdin.write(image.tobytes())
            count += 1
        proc.stdin.close()
        error = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error[-1000:]}")
        if count != frame_count:
            raise ValueError(f"renderer produced {count}/{frame_count} frames")
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "stream=nb_frames,r_frame_rate,duration,width,height",
                                "-of", "json", str(temporary)], capture_output=True, text=True, check=True)
        stream = json.loads(probe.stdout)["streams"][0]
        if (int(stream["nb_frames"]) != frame_count or int(stream["width"]) != width
                or int(stream["height"]) != height or abs(float(stream["duration"]) - frame_count / fps) > 0.02):
            raise ValueError(f"MP4 validation failed: {stream}")
        os.replace(temporary, target)
        return {"path": str(target), "frames": frame_count, "fps": fps,
                "duration_seconds": float(stream["duration"]), "bytes": target.stat().st_size}
    except Exception:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        raise
    finally:
        temporary.unlink(missing_ok=True)
