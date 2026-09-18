"""Preflight: prove that hands, camera frustums and trajectories share one world frame.

The 3D comparison draws every method into a single levelled world, so three things
have to agree there: the hand geometry, the per-method camera rigs (frustum, axis
triad and trajectory tube) and the calibrated display camera.  Rotating only some of
them silently moves the rigs to a different world - that is exactly how the frustums
ended up ~2.7 m away from their hands until 2026-09-18.

This tool rebuilds the scene through the *production* assembly
(``batch_3d_video.scene_state``, which itself asserts the contract) and then audits it:

* window stitch: the first present predicted window anchors to GT; each contiguous
  next window advances the previous prediction by the GT boundary motion;
* the calibrated camera is levelled (up axis = +y);
* each hand sits in front of its own camera at a plausible capture distance.

Exit code 1 if any segment violates the contract.  CPU is enough for the audit.

Example
-------
``python3 tools/check_3d_world_frame.py --staged-root /tmp/pv3d_batch_staged \
    --src-dir ... --prepared-root ... --hawor-root ... --hawor-index ... \
    --contact-root ... --mapping ... --device cpu``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from batch_3d_video import scene_state                        # noqa: E402
from paper_viz.sequences3d import camera_bundle_report        # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staged-root", type=Path, required=True)
    for flag in ("src-dir", "prepared-root", "hawor-root", "hawor-index", "contact-root", "mapping"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--device", default="cpu",
                        help="torch device used to build the scene; cpu is enough")
    parser.add_argument("--cell", type=int, default=192)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--bin-size", type=int, default=64)
    parser.add_argument("--camera-scale", type=float, default=0.12)
    parser.add_argument("--camera-overlay", choices=("show", "hide"), default="hide")
    parser.add_argument("--fit-margin", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segments = sorted(p for p in args.staged_root.iterdir()
                      if p.is_dir() and (p / "selection.json").is_file())
    if args.limit:
        segments = segments[:args.limit]
    if not segments:
        raise SystemExit(f"no staged segment under {args.staged_root}")

    payload, failed = {"segments": []}, False
    for segment in segments:
        scene = scene_state(args, segment, args.device)
        report = camera_bundle_report(scene["store"], scene["camera"])
        ok = not report["violations"]
        failed = failed or not ok
        print(f"\n=== {segment.name} ({'OK' if ok else 'VIOLATION'}) ===")
        print(f"  display camera up-axis error |up-+y| = {report['up_error']:.2e}")
        print(f"  {'method':11s} {'src':5s} {'cam/hand':>9s} {'|C-H| med':>10s} {'|C-H| max':>10s} "
              f"{'axis max':>9s} {'cone':>6s} {'stitch':>9s}")
        for row in report["rows"]:
            print(f"  {row['method']:11s} {row['source']:5s} "
                  f"{row['camera_frames']:4d}/{row['paired_frames']:<4d} "
                  f"{row['distance_median']:10.3f} {row['distance_max']:10.3f} "
                  f"{row['angle_max']:9.1f} {row['cone_angle']:6.1f} {row['anchor_max']:9.2e}")
        for message in report["violations"]:
            print(f"  ! {message}")
        for message in report["warnings"]:
            print(f"  ~ {message}")
        payload["segments"].append({"segment": segment.name, "ok": ok, **report})

    payload["ok"] = not failed
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{len(segments)} segment(s): {'all in one world frame' if not failed else 'CONTRACT VIOLATED'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
