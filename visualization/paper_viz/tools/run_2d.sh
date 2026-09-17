#!/usr/bin/env bash
# Render one already-staged segment with the LOCKED 2D style.
#
# The style knobs are pinned here on purpose: everything else in the pipeline is
# path plumbing, so this wrapper is the single place that defines the figure look
# the user signed off on 2026-09-17.  Override the cells/fps/jobs only if you also
# update USAGE_2D.md, otherwise figures stop matching the frozen gallery.
#
#   ./tools/run_2d.sh --inputs-dir <staged segment> --out <out dir> [--jobs 8]
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(dirname "$HERE")"

PYTHON="${PYTHON:-/usr/local/bin/python3}"
DATASET="${DATASET:-arctic}"
PREPARED_ROOT="${PREPARED_ROOT:-/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_arctic_20260820T000131Z/window_inputs}"
HAWOR_ROOT="${HAWOR_ROOT:-/mnt/workspace/sjc/DATA/eval_artifacts/hawor_native_camera_repair_20260915_v7_5000/datasets}"
CONTACT_ROOT="${CONTACT_ROOT:-/mnt/cpfs/sjc/eval_artifacts/result3_completion_20260910_v1}"
MAPPING="${MAPPING:-/mnt/workspace/sjc/EgoFound3R_viz_8fc061a_20260917/egohandmetric_prompt/data/mano_upsampling/mano_195_to_778.npz}"
SRC_DIR="${SRC_DIR:-/mnt/workspace/sjc/DATA/eval_artifacts/paper_viz_src_20260917}"

INPUTS_DIR=""
OUT=""
JOBS=8
ROWS=5
CELL2D=720
CELL2D_VIDEO=360
FPS=30
STAGES=fig2
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inputs-dir) INPUTS_DIR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --rows) ROWS="$2"; shift 2 ;;
    --cell2d) CELL2D="$2"; shift 2 ;;
    --cell2d-video) CELL2D_VIDEO="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --extra) EXTRA+=("$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$INPUTS_DIR" || -z "$OUT" ]]; then
  echo "usage: $0 --inputs-dir <staged segment> --out <out dir> [--jobs N] [--stages fig2|fig2_frame|all]" >&2
  exit 2
fi

export PYTHONPATH="$PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---- LOCKED STYLE (2026-09-17) -------------------------------------------------
# geometry columns reuse the signal rendering path: radius-1 dots, 1 px edges,
# 0.30 face wash, dots/edges mixed 65% toward white, occluded geometry still drawn
# at 45% opacity, hand-only z-buffer (the object is never used for hand geometry).
STYLE=(
  --palette reference
  --signal-style face
  --face-alpha 0.30
  --face-occluded-factor 0.35
  --geometry-brightness 1.60
  --signal-brightness 1.15
  --geometry-line-width 1
  --geometry-dot-radius 1
  --geometry-face-alpha 0.30
  --geometry-occluded-alpha 0.45
  --geometry-stroke-lighten 0.65
)
# -------------------------------------------------------------------------------

exec "$PYTHON" -m paper_viz.cli \
  --stages "$STAGES" --rows "$ROWS" --cell2d "$CELL2D" --cell2d-video "$CELL2D_VIDEO" \
  --fps "$FPS" --jobs "$JOBS" \
  "${STYLE[@]}" \
  --inputs-dir "$INPUTS_DIR" \
  --src-dir "$SRC_DIR" \
  --prepared-root "$PREPARED_ROOT/$DATASET" \
  --hawor-root "$HAWOR_ROOT/$DATASET" \
  --hawor-index "$HAWOR_ROOT/$DATASET/predictions.jsonl" \
  --contact-root "$CONTACT_ROOT" \
  --mapping "$MAPPING" \
  --out "$OUT" \
  "${EXTRA[@]+"${EXTRA[@]}"}"
