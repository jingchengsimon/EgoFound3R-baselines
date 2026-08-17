from __future__ import annotations

import atexit
import io
import json
import os
import subprocess
from collections import OrderedDict
import tarfile
from functools import lru_cache
from typing import Any

import cv2
import h5py
import numpy as np
import torch

from egohandmetric_prompt.data.schema import MediaRef


_HOT3D_FRAME_CACHE_LIMIT = 256
_TAR_FILE_CACHE_LIMIT = 16
_H5_FILE_CACHE_LIMIT = 16
_VIDEO_CAPTURE_CACHE_LIMIT = 16
_FFV1_DEPTH_WINDOW_FRAMES = 8
_FFV1_DEPTH_WINDOW_CACHE_LIMIT = 4
_HOT3D_FRAME_CACHE: OrderedDict[tuple[str, str, int], torch.Tensor] = OrderedDict()
_TAR_FILE_CACHE: OrderedDict[str, tarfile.TarFile] = OrderedDict()
_H5_FILE_CACHE: OrderedDict[str, h5py.File] = OrderedDict()
_VIDEO_CAPTURE_CACHE: OrderedDict[str, Any] = OrderedDict()
_FFV1_DEPTH_WINDOW_CACHE: OrderedDict[tuple[str, str, int], np.ndarray] = OrderedDict()
_MEDIA_CACHE_PID = os.getpid()


@lru_cache(maxsize=128)
def _video_stream_info(path: str) -> tuple[str, str, int, int]:
    _ensure_process_local_media_caches()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height",
            "-of",
            "json",
            path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"无法探测视频流：{path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"视频没有可读取的图像流：{path}")
    stream = streams[0]
    return (
        str(stream.get("codec_name", "")),
        str(stream.get("pix_fmt", "")),
        int(stream["width"]),
        int(stream["height"]),
    )


def _close_media_handle(handle: Any) -> None:
    if hasattr(handle, "release"):
        handle.release()
        return
    handle.close()


def _cached_handle(cache: OrderedDict[str, Any], key: str, factory, *, limit: int) -> Any:
    _ensure_process_local_media_caches()
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached
    handle = factory()
    cache[key] = handle
    cache.move_to_end(key)
    while len(cache) > limit:
        _, stale = cache.popitem(last=False)
        _close_media_handle(stale)
    return handle


def _clear_media_caches() -> None:
    for cache in (_TAR_FILE_CACHE, _H5_FILE_CACHE, _VIDEO_CAPTURE_CACHE):
        while cache:
            _, handle = cache.popitem(last=False)
            _close_media_handle(handle)
    _HOT3D_FRAME_CACHE.clear()
    _FFV1_DEPTH_WINDOW_CACHE.clear()
    _get_hot3d_vrs_provider.cache_clear()


def _ensure_process_local_media_caches() -> None:
    """Discard inherited handles and decoded frames after a DataLoader fork."""
    global _MEDIA_CACHE_PID
    pid = os.getpid()
    if _MEDIA_CACHE_PID == pid:
        return
    _clear_media_caches()
    _MEDIA_CACHE_PID = pid


atexit.register(_clear_media_caches)


def _ensure_readable(array: np.ndarray, is_rgb: bool) -> torch.Tensor:
    if array.ndim == 2:
        return torch.from_numpy(array.copy())
    if is_rgb:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def _ensure_rgb_readable(array: np.ndarray) -> torch.Tensor:
    if array.ndim == 2:
        return torch.from_numpy(array.copy()).float() / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).float() / 255.0


def _load_from_path(path: str, is_rgb: bool) -> torch.Tensor:
    flag = cv2.IMREAD_COLOR if is_rgb else cv2.IMREAD_UNCHANGED
    array = cv2.imread(path, flag)
    if array is None:
        raise FileNotFoundError(path)
    return _ensure_readable(array, is_rgb=is_rgb)


def _load_from_video(path: str, frame_index: int, time_msec: float, is_rgb: bool) -> torch.Tensor:
    capture = _cached_handle(
        _VIDEO_CAPTURE_CACHE,
        path,
        lambda: cv2.VideoCapture(path),
        limit=_VIDEO_CAPTURE_CACHE_LIMIT,
    )
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    if time_msec >= 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_msec)
    else:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"无法从视频读取帧：{path} frame_index={frame_index} time_msec={time_msec}")
    if not is_rgb and frame.ndim == 3:
        if not np.array_equal(frame[:, :, 0], frame[:, :, 1]) or not np.array_equal(frame[:, :, 1], frame[:, :, 2]):
            raise ValueError(f"非 RGB 视频深度帧必须是单通道或等值三通道：{path}")
        frame = frame[:, :, 0]
    return _ensure_readable(frame, is_rgb=is_rgb)


def _load_ffv1_depth_window(path: str, frame_index: int, *, dataset_name: str) -> torch.Tensor:
    """Decode an official gray16 FFV1 window once and reuse it within a worker.

    The official exports use a 15 Hz, all-intra FFV1 stream.  We retain their
    exact ``gray16le`` ffmpeg output, but read a small aligned window so a
    temporal chunk does not spawn one ffmpeg process per depth frame.
    """
    _ensure_process_local_media_caches()
    if frame_index < 0:
        raise ValueError(f"{dataset_name} depth frame_index must be non-negative: {frame_index}")
    codec_name, pix_fmt, width, height = _video_stream_info(path)
    if codec_name != "ffv1" or not pix_fmt.startswith("gray16"):
        raise RuntimeError(f"{dataset_name} depth video must be FFV1 gray16: {path}; got {codec_name}/{pix_fmt}")

    window_start = (frame_index // _FFV1_DEPTH_WINDOW_FRAMES) * _FFV1_DEPTH_WINDOW_FRAMES
    cache_key = (dataset_name, path, window_start)
    window = _FFV1_DEPTH_WINDOW_CACHE.get(cache_key)
    if window is None:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{float(window_start) / 15.0:.9f}",
                "-i",
                path,
                "-frames:v",
                str(_FFV1_DEPTH_WINDOW_FRAMES),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray16le",
                "pipe:1",
            ],
            check=False,
            capture_output=True,
        )
        frame_bytes = width * height * np.dtype("<u2").itemsize
        if result.returncode != 0 or not result.stdout or len(result.stdout) % frame_bytes:
            raise RuntimeError(
                f"无法无损解码 {dataset_name} gray16 深度窗口：{path} start={window_start}; "
                f"bytes={len(result.stdout)} frame_bytes={frame_bytes}; "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        window = np.frombuffer(result.stdout, dtype="<u2").reshape(-1, height, width).copy()
        _FFV1_DEPTH_WINDOW_CACHE[cache_key] = window
        _FFV1_DEPTH_WINDOW_CACHE.move_to_end(cache_key)
        while len(_FFV1_DEPTH_WINDOW_CACHE) > _FFV1_DEPTH_WINDOW_CACHE_LIMIT:
            _FFV1_DEPTH_WINDOW_CACHE.popitem(last=False)
    else:
        _FFV1_DEPTH_WINDOW_CACHE.move_to_end(cache_key)

    relative_index = frame_index - window_start
    if relative_index >= len(window):
        raise RuntimeError(
            f"无法读取 {dataset_name} depth frame_index={frame_index}："
            f"window [{window_start}, {window_start + len(window)}) 之外"
        )
    return torch.from_numpy(window[relative_index].copy())


def _load_taco_depth_video(path: str, frame_index: int) -> torch.Tensor:
    codec_name, pix_fmt, _, _ = _video_stream_info(path)
    if codec_name != "ffv1" or not pix_fmt.startswith("gray16"):
        return _load_from_video(path, frame_index=frame_index, time_msec=-1.0, is_rgb=False)
    return _load_ffv1_depth_window(path, frame_index, dataset_name="TACO")


def _load_hoi4d_depth_video(path: str, frame_index: int) -> torch.Tensor:
    """Lazily decode HOI4D exactly as the official ffmpeg export does."""
    return _load_ffv1_depth_window(path, frame_index, dataset_name="HOI4D")


def _load_from_tar(path: str, member: str, is_rgb: bool) -> torch.Tensor:
    handle = _cached_handle(
        _TAR_FILE_CACHE,
        path,
        lambda: tarfile.open(path, "r"),
        limit=_TAR_FILE_CACHE_LIMIT,
    )
    file_obj = handle.extractfile(member)
    if file_obj is None:
        raise FileNotFoundError(f"{path}:{member}")
    payload = np.frombuffer(file_obj.read(), dtype=np.uint8)
    flag = cv2.IMREAD_COLOR if is_rgb else cv2.IMREAD_UNCHANGED
    array = cv2.imdecode(payload, flag)
    if array is None:
        raise RuntimeError(f"无法解码 tar 成员：{path}:{member}")
    return _ensure_readable(array, is_rgb=is_rgb)


def _load_from_h5(path: str, member: str, frame_index: int) -> torch.Tensor:
    handle = _cached_handle(
        _H5_FILE_CACHE,
        path,
        lambda: h5py.File(path, "r"),
        limit=_H5_FILE_CACHE_LIMIT,
    )
    array = handle[member][frame_index]
    return _ensure_readable(np.asarray(array), is_rgb=False)


@lru_cache(maxsize=8)
def _get_hot3d_vrs_provider(vrs_path: str):
    from projectaria_tools.core import data_provider

    return data_provider.create_vrs_data_provider(vrs_path)


def _load_from_hot3d_vrs(path: str, stream_id: str, timestamp_ns: int) -> torch.Tensor:
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId

    provider = _get_hot3d_vrs_provider(path)
    image_data = provider.get_image_data_by_time_ns(
        StreamId(stream_id),
        int(timestamp_ns),
        TimeDomain.TIME_CODE,
        TimeQueryOptions.CLOSEST,
    )
    if image_data is None:
        raise RuntimeError(f"无法从 HOT3D VRS 读取图像：{path} stream_id={stream_id} timestamp_ns={timestamp_ns}")
    array = image_data[0].to_numpy_array()
    return _ensure_rgb_readable(array)


def _load_from_hot3d_vrs_cached(path: str, stream_id: str, timestamp_ns: int) -> torch.Tensor:
    cache_key = (path, stream_id, int(timestamp_ns))
    cached = _HOT3D_FRAME_CACHE.get(cache_key)
    if cached is not None:
        _HOT3D_FRAME_CACHE.move_to_end(cache_key)
        return cached.clone()
    tensor = _load_from_hot3d_vrs(path, stream_id, timestamp_ns)
    _HOT3D_FRAME_CACHE[cache_key] = tensor
    _HOT3D_FRAME_CACHE.move_to_end(cache_key)
    while len(_HOT3D_FRAME_CACHE) > _HOT3D_FRAME_CACHE_LIMIT:
        _HOT3D_FRAME_CACHE.popitem(last=False)
    return tensor.clone()


def prewarm_media_ref(media_ref: MediaRef) -> None:
    if media_ref.kind == "path":
        return
    if media_ref.kind == "video_frame":
        _cached_handle(
            _VIDEO_CAPTURE_CACHE,
            media_ref.path,
            lambda: cv2.VideoCapture(media_ref.path),
            limit=_VIDEO_CAPTURE_CACHE_LIMIT,
        )
        return
    if media_ref.kind in {"taco_depth_video", "hoi4d_depth_video"}:
        _video_stream_info(media_ref.path)
        return
    if media_ref.kind == "tar_member":
        _cached_handle(
            _TAR_FILE_CACHE,
            media_ref.path,
            lambda: tarfile.open(media_ref.path, "r"),
            limit=_TAR_FILE_CACHE_LIMIT,
        )
        return
    if media_ref.kind == "h5_dataset":
        _cached_handle(
            _H5_FILE_CACHE,
            media_ref.path,
            lambda: h5py.File(media_ref.path, "r"),
            limit=_H5_FILE_CACHE_LIMIT,
        )
        return
    if media_ref.kind == "hot3d_vrs_frame":
        _get_hot3d_vrs_provider(media_ref.path)
        return
    raise ValueError(f"未知 MediaRef.kind: {media_ref.kind}")


def prewarm_media_refs(media_refs: list[MediaRef]) -> None:
    for media_ref in media_refs:
        prewarm_media_ref(media_ref)


def load_media_ref(media_ref: MediaRef, is_rgb: bool) -> torch.Tensor:
    _ensure_process_local_media_caches()
    if media_ref.kind == "path":
        return _load_from_path(media_ref.path, is_rgb=is_rgb)
    if media_ref.kind == "video_frame":
        return _load_from_video(
            media_ref.path,
            frame_index=media_ref.frame_index,
            time_msec=media_ref.time_msec,
            is_rgb=is_rgb,
        )
    if media_ref.kind == "taco_depth_video":
        if is_rgb:
            raise ValueError("taco_depth_video 只能作为深度加载")
        return _load_taco_depth_video(media_ref.path, frame_index=media_ref.frame_index)
    if media_ref.kind == "hoi4d_depth_video":
        if is_rgb:
            raise ValueError("hoi4d_depth_video 只能作为深度加载")
        return _load_hoi4d_depth_video(media_ref.path, frame_index=media_ref.frame_index)
    if media_ref.kind == "tar_member":
        return _load_from_tar(media_ref.path, media_ref.member, is_rgb=is_rgb)
    if media_ref.kind == "h5_dataset":
        return _load_from_h5(media_ref.path, media_ref.member, media_ref.frame_index)
    if media_ref.kind == "hot3d_vrs_frame":
        return _load_from_hot3d_vrs_cached(media_ref.path, media_ref.member, media_ref.timestamp_ns)
    raise ValueError(f"未知 MediaRef.kind: {media_ref.kind}")
