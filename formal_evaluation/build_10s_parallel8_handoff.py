"""Freeze the verified completed pairs and eight disjoint continuation shards."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELATIVE = Path("visualization/batch_10s_177_p95_wmpjpe_20260912")
HANDOFF = RELATIVE / "parallel8_handoff_20260913"
LABELS = ("full177", "shard2", "shard3", "shard4")
RUN_IDS = (
    "visualization-five-10s-gallery-5000-full177-20260913",
    "visualization-five-10s-gallery-5000-parallel4_shard2-20260913",
    "visualization-five-10s-gallery-5000-parallel4_shard3-20260913",
    "visualization-five-10s-gallery-5000-parallel4_shard4-20260913",
)
OUTPUT_PREFIX = "/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_joint8_p95_wmpjpe_20260913_parallel8_handoff"


def main():
    directory = ROOT / HANDOFF
    target = directory / "plan.json"
    if target.exists():
        raise FileExistsError(f"FROZEN_PLAN_ALREADY_EXISTS:{target}")
    manifest = (ROOT / RELATIVE / "selected_manifest.jsonl").read_bytes()
    rows = [json.loads(line) for line in manifest.splitlines() if line]
    digest = hashlib.sha256(manifest).hexdigest()
    alignment = json.loads((ROOT / RELATIVE / "source_alignment_5000.json").read_text())
    if len(rows) != 177 or digest != alignment["manifest_sha256"]:
        raise ValueError("FROZEN_MANIFEST_IDENTITY_MISMATCH")
    stem_to_index = {row["gallery_stem"]: index for index, row in enumerate(rows)}
    if len(stem_to_index) != 177:
        raise ValueError("DUPLICATE_GALLERY_STEMS")
    registry = json.loads((ROOT / "formal_evaluation/config/evaluation_task_registry.json").read_text())
    complete = {}
    verification = []
    partial = []
    for label, run_id in zip(LABELS, RUN_IDS):
        run = registry["runs"][run_id]
        identity = run["identity"]
        if identity["manifest_sha256"] != digest or identity["reader_node"] != 5000:
            raise ValueError(f"REGISTERED_SOURCE_IDENTITY_MISMATCH:{run_id}")
        snapshot_path = directory / f"verify_{label}.json"
        snapshot_bytes = snapshot_path.read_bytes()
        snapshot = json.loads(snapshot_bytes)
        if (snapshot["run_id"] != run_id or snapshot["errors"] or snapshot["unexpected"]
                or snapshot["paired_verified"] != len(snapshot["paired_verified_stems"])):
            raise ValueError(f"SOURCE_VERIFICATION_FAILED:{run_id}")
        verification.append({"run_id": run_id, "sha256": hashlib.sha256(snapshot_bytes).hexdigest()})
        partial.extend({"run_id": run_id, "gallery_stem": stem} for stem in snapshot["unpaired"])
        start, end = identity.get("shard_start", 0), identity.get("shard_end", 177)
        for stem in snapshot["paired_verified_stems"]:
            index = stem_to_index[stem]
            if not start <= index < end or index in complete:
                raise ValueError(f"COMPLETED_PAIR_RANGE_OR_OVERLAP:{stem}")
            complete[index] = {"index": index, "gallery_stem": stem, "run_id": run_id,
                               "source_root": run["output_root"]}
    remaining = sorted(set(range(177)) - set(complete))
    shards = [{"number": number, "indices": remaining[number - 1::8],
               "output_root": f"{OUTPUT_PREFIX}_shard{number}"} for number in range(1, 9)]
    assigned = [index for shard in shards for index in shard["indices"]]
    if sorted(assigned) != remaining or len(assigned) != len(set(assigned)):
        raise ValueError("HANDOFF_PARTITION_INCOMPLETE_OR_OVERLAPPING")
    plan = {"schema": "10s_gallery_parallel8_handoff_v1", "manifest_sha256": digest,
            "reader_node": 5000, "source_verification": verification,
            "completed": [complete[index] for index in sorted(complete)], "partial_redo": partial,
            "shards": shards, "aggregate_output_root": OUTPUT_PREFIX + "_gallery",
            "completed_count": len(complete), "remaining_count": len(remaining)}
    target.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(target), "plan_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "completed": len(complete), "remaining": len(remaining),
                      "shard_counts": [len(shard["indices"]) for shard in shards],
                      "partial_redo": len(partial)}))


if __name__ == "__main__":
    main()
