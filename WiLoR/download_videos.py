import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import json
import os
import shutil
import socket
import subprocess
import sys
import threading

import cv2
import numpy as np
from pytubefix import YouTube


PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
DEFAULT_PROXY = "http://127.0.0.1:7890"
PRINT_LOCK = threading.Lock()
PROGRESS_BUCKETS = {}


class StreamUnavailable(RuntimeError):
    pass


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def proxy_configured():
    return any(os.environ.get(key) for key in PROXY_ENV_KEYS)


def proxy_available(proxy):
    host, port = proxy.removeprefix("http://").removeprefix("https://").split(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except OSError:
        return False


def set_proxy(proxy):
    for key in PROXY_ENV_KEYS:
        os.environ[key] = proxy


def stream_res(stream):
    if not stream.resolution or not stream.resolution.endswith("p"):
        return -1
    return int(stream.resolution[:-1])


def is_h264_stream(stream):
    return getattr(stream, "video_codec", "").startswith("avc1")


def stream_ext(stream):
    return getattr(stream, "subtype", None) or "mp4"


def stream_codec_rank(stream):
    codec = getattr(stream, "video_codec", "")
    if codec.startswith("avc1"):
        return 0
    if codec.startswith("vp9") or codec.startswith("vp09"):
        return 1
    if codec.startswith("av01"):
        return 2
    return 3


def stream_container_rank(stream):
    return 0 if stream_ext(stream) == "mp4" else 1


def available_streams(streams):
    return sorted(
        {(s.resolution, getattr(s, "video_codec", ""), stream_ext(s)) for s in streams},
        key=lambda x: (x[0] or "", x[1], x[2]),
    )


def sort_same_resolution(streams):
    return sorted(streams, key=lambda s: (stream_codec_rank(s), stream_container_rank(s), -(getattr(s, "fps", 0) or 0)))


def candidate_streams(streams, target_res, exact_res=False):
    streams = [s for s in streams if stream_res(s) > 0]
    exact = [s for s in streams if stream_res(s) == target_res]
    if exact:
        return sort_same_resolution(exact)
    if exact_res:
        raise StreamUnavailable(f"no video stream at {target_res}p; available={available_streams(streams)}")

    lower = [s for s in streams if stream_res(s) <= target_res]
    if lower:
        grouped = []
        for res in sorted({stream_res(s) for s in lower}, reverse=True):
            grouped.extend(sort_same_resolution([s for s in lower if stream_res(s) == res]))
        return grouped

    raise StreamUnavailable(f"no video stream at or below {target_res}p; available={available_streams(streams)}")


def select_stream(streams, target_res, exact_res=False):
    return candidate_streams(streams, target_res, exact_res=exact_res)[0]


def progress_bar(label, done, total):
    done = int(done)
    total = int(total)
    if total <= 0:
        log(f"{label}: {done}")
        return
    width = 28
    ratio = min(max(done / total, 0), 1)
    bucket = int(ratio * 20)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    with PRINT_LOCK:
        if done < total and PROGRESS_BUCKETS.get(label) == bucket:
            return
        if PROGRESS_BUCKETS.get(label) == bucket:
            return
        PROGRESS_BUCKETS[label] = bucket
        print(f"{label}: [{bar}] {done}/{total} {ratio * 100:5.1f}%", flush=True)


def video_codec(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def video_dimensions(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed probing {video_path}")
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def video_integrity_error(video_path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            video_path,
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return None
    errors = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    return errors[0] if errors else "ffmpeg decode check failed"


def validate_video(video_path):
    codec = video_codec(video_path)
    if codec != "h264":
        raise RuntimeError(f"downloaded codec is {codec}, need h264")
    integrity_error = video_integrity_error(video_path)
    if integrity_error:
        raise RuntimeError(f"downloaded file is damaged: {integrity_error}")


def transcode_to_h264(input_path, output_path):
    temp_output = output_path + ".tmp.mp4"
    if os.path.exists(temp_output):
        os.remove(temp_output)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        temp_output,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise RuntimeError(result.stderr.strip() or "ffmpeg transcode failed")
    validate_video(temp_output)
    os.replace(temp_output, output_path)


def normalize_video(raw_path, video_path, stream):
    codec = video_codec(raw_path)
    if codec == "h264" and stream_ext(stream) == "mp4":
        os.replace(raw_path, video_path)
    else:
        log(f"Transcoding {os.path.basename(raw_path)} from {codec}/{stream_ext(stream)} to h264/mp4")
        transcode_to_h264(raw_path, video_path)
        os.remove(raw_path)
    try:
        validate_video(video_path)
    except Exception:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise


def existing_video_status(video_path, expected_height=None, max_height=None):
    if not os.path.exists(video_path):
        return True, None
    codec = video_codec(video_path)
    integrity_error = video_integrity_error(video_path) if codec == "h264" else None
    wrong_height = False
    too_large = False
    if codec == "h264" and not integrity_error and (expected_height or max_height):
        _, height = video_dimensions(video_path)
        wrong_height = expected_height and height != expected_height
        too_large = max_height and height > max_height
    if codec == "h264" and not integrity_error and not wrong_height and not too_large:
        return True, None
    reason = f"existing codec is {codec}, need h264"
    if integrity_error:
        reason = f"existing file is damaged: {integrity_error}"
    if wrong_height:
        reason = f"existing height is {height}p, need {expected_height}p"
    if too_large:
        reason = f"existing height is {height}p, need <= {max_height}p"
    return False, reason


def existing_video_ok(video_id, video_path, expected_height=None, max_height=None):
    ok, reason = existing_video_status(video_path, expected_height, max_height)
    if ok:
        return True
    log(f"Redownloading {video_id}: {reason}")
    os.remove(video_path)
    return False


def raw_download_path(output_dir, video_id, stream):
    return os.path.join(output_dir, f"{video_id}.download.{stream_ext(stream)}")


def remove_candidate_files(raw_path, video_path):
    if raw_path and os.path.exists(raw_path):
        os.remove(raw_path)
    temp_output = video_path + ".tmp.mp4"
    if os.path.exists(temp_output):
        os.remove(temp_output)


def verify_download_size(raw_path, stream):
    filesize = getattr(stream, "filesize", None)
    if filesize and os.path.getsize(raw_path) != filesize:
        actual_size = os.path.getsize(raw_path)
        raise RuntimeError(f"downloaded size mismatch: {actual_size} != {filesize}")


def stream_description(stream):
    return f"{stream.resolution} {getattr(stream, 'video_codec', '')} {stream_ext(stream)}"


def download_stream_candidates(video_id, streams, target_res, output_dir, exact_res, download_stream):
    video_path = os.path.join(output_dir, video_id + ".mp4")
    candidates = candidate_streams(streams, target_res, exact_res=exact_res)
    last_error = None

    for stream in candidates:
        expected_height = stream_res(stream)
        if os.path.exists(video_path):
            ok, existing_reason = existing_video_status(video_path, expected_height=expected_height)
            if ok:
                return video_path
            log(f"Existing {video_id}.mp4 not reusable for {stream_description(stream)}: {existing_reason}")

        raw_path = raw_download_path(output_dir, video_id, stream)
        remove_candidate_files(raw_path, video_path)
        try:
            log(f"Selected {video_id}: {stream_description(stream)}")
            download_stream(stream, raw_path)
            verify_download_size(raw_path, stream)
            normalize_video(raw_path, video_path, stream)
            return video_path
        except Exception as e:
            last_error = e
            remove_candidate_files(raw_path, video_path)
            log(f"Candidate failed {video_id}: {stream_description(stream)}: {type(e).__name__}: {e}")

    raise StreamUnavailable(f"no usable stream for {video_id} at target {target_res}p: {last_error}")


def download_video_pytubefix(video_id, target_res, output_dir, exact_res):
    yt = YouTube(
        "https://youtu.be/" + video_id,
        on_progress_callback=lambda stream, chunk, bytes_remaining: progress_bar(
            f"download {video_id}",
            (stream.filesize or 0) - bytes_remaining,
            stream.filesize or 0,
        ),
    )
    streams = list(yt.streams.filter(only_video=True).order_by("resolution").desc())

    def download_stream(stream, raw_path):
        stream.download(output_path=output_dir, filename=os.path.basename(raw_path))
        progress_bar(f"download {video_id}", stream.filesize or 1, stream.filesize or 1)

    return download_stream_candidates(video_id, streams, target_res, output_dir, exact_res, download_stream)


def download_video_aria2c(video_id, target_res, output_dir, exact_res):
    if not shutil.which("aria2c"):
        raise RuntimeError("aria2c downloader requires aria2c in PATH")

    yt = YouTube("https://youtu.be/" + video_id)
    streams = list(yt.streams.filter(only_video=True).order_by("resolution").desc())

    def download_stream(stream, raw_path):
        command = [
            "aria2c",
            "-x",
            "8",
            "-s",
            "8",
            "-k",
            "1M",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--summary-interval=1",
            "-d",
            output_dir,
            "-o",
            os.path.basename(raw_path),
            stream.url,
        ]
        target_label = f"target={target_res}p" if exact_res else f"target<={target_res}p"
        log(f"Downloading {video_id} with aria2c {target_label}")
        result = subprocess.run(command)
        if result.returncode != 0:
            raise RuntimeError(f"aria2c failed with exit code {result.returncode}")
        if not os.path.exists(raw_path):
            raise RuntimeError(f"download did not create {raw_path}")

    return download_stream_candidates(video_id, streams, target_res, output_dir, exact_res, download_stream)


def download_video(video_id, target_res, output_dir, downloader, exact_res):
    if downloader == "pytubefix":
        return download_video_pytubefix(video_id, target_res, output_dir, exact_res)
    if downloader == "aria2c":
        return download_video_aria2c(video_id, target_res, output_dir, exact_res)
    raise RuntimeError(f"unknown downloader {downloader}")


def download_video_with_fallback(video_id, target_res, source_height, output_dir, downloader, max_download_res):
    if not max_download_res:
        return download_video(video_id, source_height, output_dir, downloader, exact_res=True), False
    try:
        return download_video(video_id, target_res, output_dir, downloader, exact_res=False), False
    except StreamUnavailable as e:
        log(f"No direct low-res stream for {video_id}: {e}")
        log(f"Falling back to original {source_height}p and resizing frames locally")
        return download_video(video_id, source_height, output_dir, downloader, exact_res=True), True


def remove_video_file(video_id, output_dir):
    video_path = os.path.join(output_dir, video_id + ".mp4")
    if os.path.exists(video_path):
        os.remove(video_path)
    for suffix in (".part", ".ytdl", ".temp.mp4"):
        temp_path = video_path + suffix
        if os.path.exists(temp_path):
            os.remove(temp_path)


def scale_annotation_item(item, scale_x, scale_y):
    scaled = dict(item)
    bbox = item["bbox"].copy()
    bbox[[0, 2]] *= scale_x
    bbox[[1, 3]] *= scale_y
    scaled["bbox"] = bbox

    joints_2d = item["joints_2d"]
    if hasattr(joints_2d, "clone"):
        joints_2d = joints_2d.clone()
    else:
        joints_2d = joints_2d.copy()
    joints_2d[..., 0] *= scale_x
    joints_2d[..., 1] *= scale_y
    scaled["joints_2d"] = joints_2d

    intrinsics = item["K"].copy()
    intrinsics[0, :] *= scale_x
    intrinsics[1, :] *= scale_y
    scaled["K"] = intrinsics
    return scaled


def scale_annotation(annotation, scale_x, scale_y):
    scaled = np.empty(annotation.shape, dtype=object)
    for idx, item in enumerate(annotation.flat):
        scaled.flat[idx] = scale_annotation_item(item, scale_x, scale_y)
    return scaled


def save_scaled_annotation(source_path, target_path, scale_x, scale_y):
    if os.path.exists(target_path):
        return
    annotation = np.load(source_path, allow_pickle=True)
    scaled = scale_annotation(annotation, scale_x, scale_y)
    np.save(target_path, scaled)


def write_resize_info(output_anno_dir, source_res, target_res):
    os.makedirs(output_anno_dir, exist_ok=True)
    info_path = os.path.join(output_anno_dir, ".resize_info.json")
    info = {
        "source_res": [int(source_res[0]), int(source_res[1])],
        "target_res": [int(target_res[0]), int(target_res[1])],
    }
    if os.path.exists(info_path):
        with open(info_path) as f:
            existing = json.load(f)
        if existing != info:
            raise RuntimeError(f"resize metadata mismatch in {info_path}: {existing} != {info}")
        return
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)


def output_anno_dir(root, mode, video_id, max_download_res):
    if not max_download_res:
        return os.path.join(root, "WHIM", mode, "anno", video_id)
    return os.path.join(root, "WHIM", mode, f"anno_resized_max{max_download_res}p", video_id)


def write_frame(output_path, image):
    if not cv2.imwrite(output_path, image):
        raise RuntimeError(f"failed writing {output_path}")


def resize_frame(output_path, image_path, target_res):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"failed reading {image_path}")
    target_height, target_width = target_res
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    write_frame(output_path, resized)


def flush_completed_writes(pending, wait_for_one):
    if not pending:
        return pending
    if wait_for_one:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
    else:
        done = pending
        pending = set()
    for future in done:
        future.result()
    return pending


def extract_frames(video_path, source_anno_dir, output_dir, fps_org, video_id, source_res, write_workers=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_rate = round(fps / fps_org)
    target_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_height, source_width = int(source_res[0]), int(source_res[1])
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    frame_files = sorted(frame for frame in os.listdir(source_anno_dir) if frame.endswith(".npy"))
    write_workers = max(1, write_workers)
    max_pending = max(1, write_workers * 4)
    os.makedirs(output_dir, exist_ok=True)
    if output_dir != source_anno_dir:
        write_resize_info(output_dir, [source_height, source_width], [target_height, target_width])

    try:
        with ThreadPoolExecutor(max_workers=write_workers) as write_pool:
            pending = set()
            for idx, frame in enumerate(frame_files, start=1):
                frame_gt = int(os.path.splitext(frame)[0])
                frame_idx = frame_gt * fps_rate
                output_path = os.path.join(output_dir, os.path.splitext(frame)[0] + ".jpg")
                output_gt_path = os.path.join(output_dir, frame)
                if output_dir != source_anno_dir:
                    save_scaled_annotation(
                        os.path.join(source_anno_dir, frame),
                        output_gt_path,
                        scale_x,
                        scale_y,
                    )
                if os.path.exists(output_path):
                    if idx == len(frame_files) or idx % 50 == 0:
                        progress_bar(f"frames {video_id}", idx, len(frame_files))
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, img_cv2 = cap.read()
                if not ret:
                    pending = flush_completed_writes(pending, wait_for_one=False)
                    raise RuntimeError(f"failed reading frame {frame_gt} from {video_id}")

                pending.add(write_pool.submit(write_frame, output_path, img_cv2))
                if len(pending) >= max_pending:
                    pending = flush_completed_writes(pending, wait_for_one=True)

                if idx == len(frame_files) or idx % 50 == 0:
                    progress_bar(f"frames {video_id}", idx, len(frame_files))
            pending = flush_completed_writes(pending, wait_for_one=False)
    finally:
        cap.release()
    if frame_files:
        progress_bar(f"frames {video_id}", len(frame_files), len(frame_files))


def resized_resolution(source_res, max_height):
    source_height, source_width = int(source_res[0]), int(source_res[1])
    target_height = min(source_height, int(max_height))
    target_width = round(source_width * target_height / source_height)
    return [target_height, target_width]


def resize_extracted_frames(source_anno_dir, output_dir, source_res, target_res, video_id, write_workers=1):
    source_height, source_width = int(source_res[0]), int(source_res[1])
    target_height, target_width = int(target_res[0]), int(target_res[1])
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    frame_files = sorted(frame for frame in os.listdir(source_anno_dir) if frame.endswith(".npy"))
    write_workers = max(1, write_workers)
    max_pending = max(1, write_workers * 4)
    os.makedirs(output_dir, exist_ok=True)
    write_resize_info(output_dir, [source_height, source_width], [target_height, target_width])

    with ThreadPoolExecutor(max_workers=write_workers) as write_pool:
        pending = set()
        for idx, frame in enumerate(frame_files, start=1):
            frame_stem = os.path.splitext(frame)[0]
            source_gt_path = os.path.join(source_anno_dir, frame)
            source_image_path = os.path.join(source_anno_dir, frame_stem + ".jpg")
            output_gt_path = os.path.join(output_dir, frame)
            output_image_path = os.path.join(output_dir, frame_stem + ".jpg")

            save_scaled_annotation(source_gt_path, output_gt_path, scale_x, scale_y)
            if not os.path.exists(output_image_path):
                pending.add(write_pool.submit(resize_frame, output_image_path, source_image_path, target_res))
                if len(pending) >= max_pending:
                    pending = flush_completed_writes(pending, wait_for_one=True)

            if idx == len(frame_files) or idx % 50 == 0:
                progress_bar(f"resize {video_id}", idx, len(frame_files))
        pending = flush_completed_writes(pending, wait_for_one=False)
    if frame_files:
        progress_bar(f"resize {video_id}", len(frame_files), len(frame_files))


def process_video(
    root,
    mode,
    video_dir,
    idx,
    total,
    video_id,
    video_meta,
    write_workers,
    max_retries,
    downloader,
    max_download_res,
):
    source_res = video_meta["res"]
    source_height = int(source_res[0])
    target_res = min(source_height, max_download_res) if max_download_res else source_height
    exact_res = max_download_res is None
    target_label = f"target={target_res}p" if exact_res else f"target<={target_res}p"
    log(f"[{idx}/{total}] {video_id} {target_label} downloader={downloader}")
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            source_anno_dir = os.path.join(root, "WHIM", mode, "anno", video_id)
            video_path, manual_resize = download_video_with_fallback(
                video_id,
                target_res,
                source_height,
                video_dir,
                downloader,
                max_download_res,
            )
            if manual_resize:
                extract_frames(
                    video_path,
                    source_anno_dir,
                    source_anno_dir,
                    video_meta["fps"],
                    video_id,
                    source_res,
                    write_workers,
                )
                resize_extracted_frames(
                    source_anno_dir,
                    output_anno_dir(root, mode, video_id, max_download_res),
                    source_res,
                    resized_resolution(source_res, max_download_res),
                    video_id,
                    write_workers,
                )
            else:
                extract_frames(
                    video_path,
                    source_anno_dir,
                    output_anno_dir(root, mode, video_id, max_download_res),
                    video_meta["fps"],
                    video_id,
                    source_res,
                    write_workers,
                )
            log(f"Done {video_id}")
            return None
        except Exception as e:
            last_error = e
            log(f"Attempt {attempt}/{max_retries} failed {video_id}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                remove_video_file(video_id, video_dir)
                log(f"Retrying {video_id}")

    log(f"Failed {video_id}: {type(last_error).__name__}: {last_error}")
    return video_id


def process_videos(root, mode, video_dict, video_dir, workers, write_workers, max_retries, downloader, max_download_res):
    items = list(video_dict.items())
    total = len(items)
    failed_ids = []
    workers = max(1, workers)

    if workers == 1:
        for idx, (video_id, video_meta) in enumerate(items, start=1):
            failed_id = process_video(
                root,
                mode,
                video_dir,
                idx,
                total,
                video_id,
                video_meta,
                write_workers,
                max_retries,
                downloader,
                max_download_res,
            )
            if failed_id:
                failed_ids.append(failed_id)
        return failed_ids

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_video,
                root,
                mode,
                video_dir,
                idx,
                total,
                video_id,
                video_meta,
                write_workers,
                max_retries,
                downloader,
                max_download_res,
            ): video_id
            for idx, (video_id, video_meta) in enumerate(items, start=1)
        }
        for future in as_completed(futures):
            failed_id = future.result()
            if failed_id:
                failed_ids.append(failed_id)
    return failed_ids


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Directory of WHIM")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Train/Test set")
    parser.add_argument("--proxy", type=str, default=None, help="HTTP(S) proxy URL")
    parser.add_argument("--no-auto-proxy", action="store_true", help="Disable local proxy autodetection")
    parser.add_argument("--workers", type=positive_int, default=1, help="Parallel video download/extraction workers")
    parser.add_argument("--write-workers", type=positive_int, default=4, help="JPEG writing workers per video")
    parser.add_argument("--max-retries", type=positive_int, default=3, help="Max attempts per video")
    parser.add_argument("--downloader", choices=["pytubefix", "aria2c"], default="pytubefix")
    parser.add_argument("--max-download-res", type=positive_int, default=None, help="Max downloaded video height")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.proxy:
        set_proxy(args.proxy)
        log(f"Using proxy {args.proxy}")
    elif not args.no_auto_proxy and not proxy_configured() and proxy_available(DEFAULT_PROXY):
        set_proxy(DEFAULT_PROXY)
        log(f"Using proxy {DEFAULT_PROXY}")

    with open(os.path.join(args.root, "whim", f"{args.mode}_video_ids.json")) as f:
        video_dict = json.load(f)

    video_dir = os.path.join(args.root, "Videos")
    os.makedirs(video_dir, exist_ok=True)

    failed_ids = process_videos(
        args.root,
        args.mode,
        video_dict,
        video_dir,
        args.workers,
        args.write_workers,
        args.max_retries,
        args.downloader,
        args.max_download_res,
    )

    np.save(os.path.join(args.root, f"failed_videos_{args.mode}.npy"), failed_ids)
    np.save(os.path.join(args.root, "failed_videos.npy"), failed_ids)
    if failed_ids:
        log(f"Failed videos: {len(failed_ids)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
