import argparse
import json
import os

import numpy as np

import download_videos


DEFAULT_ROOT = "/media/hanyu/82FE717CFE716975/mnt/DATA/whim"


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def load_failed_ids(path):
    failed_ids = np.load(path, allow_pickle=True).tolist()
    seen = set()
    unique_ids = []
    for video_id in failed_ids:
        video_id = str(video_id)
        if video_id in seen:
            continue
        seen.add(video_id)
        unique_ids.append(video_id)
    return unique_ids


def default_failed_file(root, mode):
    mode_path = os.path.join(root, f"failed_videos_{mode}.npy")
    if os.path.exists(mode_path):
        return mode_path
    return os.path.join(root, "failed_videos.npy")


def save_failed_ids(root, mode, failed_ids):
    mode_path = os.path.join(root, f"failed_videos_{mode}.npy")
    np.save(mode_path, failed_ids)
    np.save(os.path.join(root, "failed_videos.npy"), failed_ids)
    return mode_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT, help="WHIM root directory")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--failed-file", default=None, help="Input .npy failed-id file")
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--write-workers", type=positive_int, default=8)
    parser.add_argument("--max-retries", type=positive_int, default=3)
    parser.add_argument("--downloader", choices=["pytubefix", "aria2c"], default="aria2c")
    parser.add_argument("--max-download-res", type=positive_int, default=480)
    parser.add_argument("--proxy", default=None, help="HTTP(S) proxy URL")
    parser.add_argument("--no-auto-proxy", action="store_true", help="Disable local proxy autodetection")
    parser.add_argument("--dry-run", action="store_true", help="Only print IDs to retry")
    return parser.parse_args()


def main():
    args = parse_args()
    failed_file = args.failed_file or default_failed_file(args.root, args.mode)

    failed_ids = load_failed_ids(failed_file)
    with open(os.path.join(args.root, "whim", f"{args.mode}_video_ids.json")) as f:
        video_dict = json.load(f)

    missing_ids = [video_id for video_id in failed_ids if video_id not in video_dict]
    if missing_ids:
        raise RuntimeError(f"{len(missing_ids)} failed ids are missing from {args.mode}_video_ids.json: {missing_ids}")

    retry_dict = {video_id: video_dict[video_id] for video_id in failed_ids}
    print(f"Input failed file: {failed_file}")
    print(f"Retry videos: {len(retry_dict)}")
    print(f"IDs: {failed_ids}")
    if args.dry_run:
        return

    if args.proxy:
        download_videos.set_proxy(args.proxy)
        download_videos.log(f"Using proxy {args.proxy}")
    elif not args.no_auto_proxy and not download_videos.proxy_configured() and download_videos.proxy_available(download_videos.DEFAULT_PROXY):
        download_videos.set_proxy(download_videos.DEFAULT_PROXY)
        download_videos.log(f"Using proxy {download_videos.DEFAULT_PROXY}")

    video_dir = os.path.join(args.root, "Videos")
    os.makedirs(video_dir, exist_ok=True)
    remaining_failed = download_videos.process_videos(
        args.root,
        args.mode,
        retry_dict,
        video_dir,
        args.workers,
        args.write_workers,
        args.max_retries,
        args.downloader,
        args.max_download_res,
    )

    remaining_failed = [str(video_id) for video_id in remaining_failed]
    succeeded = [video_id for video_id in failed_ids if video_id not in set(remaining_failed)]
    saved_path = save_failed_ids(args.root, args.mode, remaining_failed)

    print(f"Retry succeeded: {len(succeeded)}")
    print(f"Retry failed: {len(remaining_failed)}")
    print(f"Updated: {saved_path}")
    print(f"Updated: {os.path.join(args.root, 'failed_videos.npy')}")


if __name__ == "__main__":
    main()
