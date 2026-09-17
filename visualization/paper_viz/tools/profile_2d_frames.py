"""Profile the per-frame 2D render cost of one segment (same flags as paper_viz.cli).

Renders ``--frames`` video-style frames (cell2d-video, full 15-column row) with
cProfile and prints the cumulative time of the hottest functions plus the wall
time per stage (frame load, columns, compose).  Used to size the 2D batch run.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path

from paper_viz import render2d
from paper_viz.cli import (apply_style_args, build_parser, build_registry,
                           compose_figure2, load_segment)


def main() -> None:
    parser = build_parser()
    parser.add_argument("--frames", type=int, default=4, help="frames to profile")
    args = parser.parse_args()
    apply_style_args(args)
    segment = load_segment(build_registry(args), args)
    total_frames = 60 * len(segment.windows)
    indices = [round(i * (total_frames - 1) / max(args.frames - 1, 1)) for i in range(args.frames)]
    timings = {"load": 0.0, "columns": 0.0, "compose": 0.0}
    profiler = cProfile.Profile()
    for t in indices:
        w_index, f_index = divmod(int(t), 60)
        window = segment.windows[w_index]
        start = time.perf_counter()
        frame = render2d.Frame2D(window, f_index, segment.mano, cell_w=args.cell2d_video)
        timings["load"] += time.perf_counter() - start
        profiler.enable()
        cells = {}
        start = time.perf_counter()
        for c, (method, signal) in enumerate(render2d.COLUMNS):
            cells[(0, c)] = render2d.column_cell(frame, method, signal, f_index)
        timings["columns"] += time.perf_counter() - start
        profiler.disable()
        start = time.perf_counter()
        compose_figure2(cells, 1, [f"{window.frame_ids[f_index]}"], segment,
                        subtitle=f"profile frame {t}")
        timings["compose"] += time.perf_counter() - start
        print(f"frame {t}: load={timings['load']:.2f}s columns={timings['columns']:.2f}s "
              f"compose={timings['compose']:.2f}s", flush=True)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(35)
    print(stream.getvalue())
    print("stages:", {key: round(value / len(indices), 3) for key, value in timings.items()},
          "per frame (avg)")
    print("output dir:", Path(args.out))


if __name__ == "__main__":
    main()
