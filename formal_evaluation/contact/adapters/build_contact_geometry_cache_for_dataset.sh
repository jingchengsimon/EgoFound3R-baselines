#!/bin/bash
# Standard post-materialization step: build the S2Contact/ContactOpt geometry
# cache for one dataset from its already-materialized window_inputs shards.
#
# Usage:
#   build_contact_geometry_cache_for_dataset.sh <dataset> <window_cap> <work_root> <shard_jsonl_1> [<shard_jsonl_2> ...]
#
# The shard jsonl files should be passed in canonical order and together
# contain at least <window_cap> lines; only the first <window_cap> lines are
# used (so pass exactly the shards needed, e.g. 8 x 50-window shards for a
# 400-window cap). Produces, under /mnt/workspace/sjc/DATA/<dataset>_contact_baseline/cache/:
#   s2_right_<dataset>_<window_cap>.pkl / _index.jsonl
#   contactopt_right_<dataset>_<window_cap>.pkl / _index.jsonl
#
# Idempotent: skips a baseline whose output cache already exists.
set -euo pipefail

DATASET=$1; CAP=$2; WORK_ROOT=$3; shift 3
SHARDS=("$@")

PY=/mnt/workspace/sjc/envs/contactopt/bin/python
W=/mnt/workspace/sjc/EgoFound3R-baselines_formal_7c79a98
MANO=/mnt/workspace/sjc/models/human/mano/MANO_RIGHT.pkl
SCRIPT=$W/formal_evaluation/contact/adapters/build_s2_contactopt_geometry_cache.py

WORK=$WORK_ROOT/contact_cache_build_${DATASET}_${CAP}
CACHE_DIR=/mnt/workspace/sjc/DATA/${DATASET}_contact_baseline/cache
mkdir -p "$WORK" "$CACHE_DIR"

COMBINED=$WORK/window_inputs_${DATASET}_first${CAP}.jsonl
: > "$COMBINED"
for s in "${SHARDS[@]}"; do
  cat "$s" >> "$COMBINED"
done
head -n "$CAP" "$COMBINED" > "$COMBINED.capped"
mv "$COMBINED.capped" "$COMBINED"
N=$(wc -l < "$COMBINED")
if [ "$N" -ne "$CAP" ]; then
  echo "ERROR: combined index has $N lines, expected $CAP" >&2
  exit 1
fi

for baseline in s2contact contactopt; do
  case "$baseline" in
    s2contact) SRC=/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/S2Contact; PREFIX=s2 ;;
    contactopt) SRC=/mnt/workspace/sjc/EgoFound3R-baselines/contact_src/ContactOpt; PREFIX=contactopt ;;
  esac
  OUT_CACHE=$CACHE_DIR/${PREFIX}_right_${DATASET}_${CAP}.pkl
  OUT_INDEX=$CACHE_DIR/${PREFIX}_right_${DATASET}_${CAP}_index.jsonl
  if [ -f "$OUT_CACHE" ]; then
    echo "skip $baseline: $OUT_CACHE already exists"
    continue
  fi
  echo "building $baseline cache for $DATASET ($CAP windows) -> $OUT_CACHE"
  $PY "$SCRIPT" --baseline "$baseline" --source-root "$SRC" --mano-right "$MANO" \
    --window-input-index "$COMBINED" --output-cache "$OUT_CACHE" --output-index "$OUT_INDEX" \
    > "$WORK/build_${baseline}.log" 2>&1
  echo "done $baseline: $(tail -1 "$WORK/build_${baseline}.log")"
done
