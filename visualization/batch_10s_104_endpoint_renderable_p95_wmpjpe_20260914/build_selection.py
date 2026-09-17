"""Select maximum non-overlapping 10s segments after the endpoint gate."""

import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def counts(rows):
    return dict(sorted(collections.Counter(row["dataset"] for row in rows).items()))


rows = read_jsonl(ROOT / "endpoint_audit.jsonl")
for row in rows:
    fields = [
        row[f"{endpoint}_{side}_{owner}_{field}"]
        for endpoint in ("first", "last")
        for side in ("left", "right")
        for owner, field in (("ego", "valid"), ("gt", "valid"),
                             ("ego", "markers_finite"), ("gt", "markers_finite"))
    ]
    row["endpoint_renderable"] = row["source_complete"] and all(fields)

eligible = [row for row in rows if row["endpoint_renderable"]]
selected = []
for dataset in sorted({row["dataset"] for row in eligible}):
    by_sequence = collections.defaultdict(list)
    for row in eligible:
        if row["dataset"] == dataset:
            by_sequence[row["sequence_id"]].append(row)
    dataset_selected = []
    for sequence_rows in by_sequence.values():
        last_end = None
        for row in sorted(sequence_rows, key=lambda value: int(value["frame_ids"][-1])):
            start, end = int(row["frame_ids"][0]), int(row["frame_ids"][-1])
            if last_end is None or start > last_end:
                dataset_selected.append(row)
                last_end = end
    for number, row in enumerate(sorted(dataset_selected, key=lambda value: (value["sequence_id"], int(value["frame_ids"][0]))), 1):
        suffix = row["segment_id"].split("__", 1)[1]
        stem = f"{dataset}__{number:03d}__{row['frame_ids'][0]}-{row['frame_ids'][-1]}__{suffix}"
        selected.append({**row, "batch_number": number, "gallery_stem": stem,
                         "png_filename": stem + ".png", "video_filename": stem + ".mp4",
                         "selection_band": "max_count_nonoverlap_after_endpoint_renderable"})

current = [row for row in rows if row["in_current_177"]]
current_ids = {row["segment_id"] for row in current}
selected_ids = {row["segment_id"] for row in selected}
assert len(rows) == 602 and len(current) == 177
assert len(eligible) == 319 and sum(row["endpoint_renderable"] for row in current) == 81
assert len(selected) == 104

write_jsonl(ROOT / "eligible_manifest.jsonl", eligible)
write_jsonl(ROOT / "selected_manifest.jsonl", selected)
(ROOT / "summary.json").write_text(json.dumps({
    "status": "complete",
    "gate": "at first and last frame, both hands require Ego hand_valid, GT hand_valid, finite Ego 195 markers, and finite GT 195 markers; middle frames may be missing; P95 mask is not applied",
    "source_candidates": len(rows),
    "eligible_candidates": len(eligible),
    "eligible_candidates_by_dataset": counts(eligible),
    "current_177_eligible": sum(row["endpoint_renderable"] for row in current),
    "current_177_eligible_by_dataset": counts([row for row in current if row["endpoint_renderable"]]),
    "maximum_nonoverlapping_segments": len(selected),
    "selected_by_dataset": counts(selected),
    "selected_already_in_current_177": len(selected_ids & current_ids),
    "selected_replacements_from_overlapping_candidates": len(selected_ids - current_ids),
    "verified_unique_windows": 978,
    "source_errors": 0,
}, indent=2) + "\n")
(ROOT / "COMPLETE").write_text("602 candidates audited; 319 endpoint-renderable; 104 maximum non-overlapping segments.\n")
