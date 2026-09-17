"""Freeze verified parallel8 pairs and partition remaining gallery segments."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELATIVE = Path("visualization/batch_10s_177_p95_wmpjpe_20260912")
OLD_PLAN = RELATIVE / "parallel8_handoff_20260913/plan.json"
HANDOFF = RELATIVE / "parallel16_handoff_20260913"
OUTPUT_PREFIX = "/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_joint8_p95_wmpjpe_20260913_parallel16_handoff"


def main():
    target = ROOT / HANDOFF / "plan.json"
    if target.exists():
        raise FileExistsError(f"FROZEN_PLAN_ALREADY_EXISTS:{target}")
    manifest = (ROOT / RELATIVE / "selected_manifest.jsonl").read_bytes()
    rows = [json.loads(line) for line in manifest.splitlines() if line]
    digest = hashlib.sha256(manifest).hexdigest()
    alignment = json.loads((ROOT / RELATIVE / "source_alignment_5000.json").read_text())
    if len(rows) != 177 or alignment["manifest_sha256"] != digest:
        raise ValueError("FROZEN_MANIFEST_IDENTITY_MISMATCH")
    stem_to_index = {row["gallery_stem"]: index for index, row in enumerate(rows)}
    if len(stem_to_index) != 177:
        raise ValueError("DUPLICATE_GALLERY_STEMS")
    old_plan_bytes = (ROOT / OLD_PLAN).read_bytes()
    old_plan = json.loads(old_plan_bytes)
    if (old_plan["schema"] != "10s_gallery_parallel8_handoff_v1"
            or old_plan["manifest_sha256"] != digest
            or len(old_plan["shards"]) != 8):
        raise ValueError("PARALLEL8_PLAN_IDENTITY_MISMATCH")
    registry = json.loads((ROOT / "formal_evaluation/config/evaluation_task_registry.json").read_text())
    complete = {}
    for entry in old_plan["completed"]:
        index = entry["index"]
        old = registry["runs"].get(entry["run_id"])
        if (index in complete or rows[index]["gallery_stem"] != entry["gallery_stem"]
                or not old or old["output_root"] != entry["source_root"]):
            raise ValueError("PREVIOUS_COMPLETED_SOURCE_MISMATCH")
        complete[index] = entry
    verification = []
    partial = []
    for shard in old_plan["shards"]:
        number = shard["number"]
        run_id = f"visualization-five-10s-gallery-5000-parallel8_handoff_shard{number}-20260913"
        run = registry["runs"].get(run_id)
        if (not run or run["output_root"] != shard["output_root"]
                or run["identity"]["manifest_sha256"] != digest
                or run["identity"]["selected_indices"] != shard["indices"]
                or run["identity"]["reader_node"] != 5000):
            raise ValueError(f"PARALLEL8_REGISTERED_SOURCE_MISMATCH:{run_id}")
        state = json.loads((ROOT / run["state_file"]).read_text())
        if state.get("jobs", {}).get("five::gallery", {}).get("status") != "paused":
            raise ValueError(f"PARALLEL8_SOURCE_NOT_PAUSED:{run_id}")
        snapshot_bytes = (ROOT / HANDOFF / f"verify_shard{number}.json").read_bytes()
        snapshot = json.loads(snapshot_bytes)
        if (snapshot["run_id"] != run_id or snapshot["errors"] or snapshot["unexpected"]
                or snapshot["paired_verified"] != len(snapshot["paired_verified_stems"])):
            raise ValueError(f"PARALLEL8_SOURCE_VERIFICATION_FAILED:{run_id}")
        verification.append({"run_id": run_id, "sha256": hashlib.sha256(snapshot_bytes).hexdigest()})
        partial.extend({"run_id": run_id, "gallery_stem": stem} for stem in snapshot["unpaired"])
        allowed = set(shard["indices"])
        for stem in snapshot["paired_verified_stems"]:
            index = stem_to_index[stem]
            if index not in allowed or index in complete:
                raise ValueError(f"COMPLETED_PAIR_RANGE_OR_OVERLAP:{stem}")
            complete[index] = {"index": index, "gallery_stem": stem,
                               "run_id": run_id, "source_root": run["output_root"]}
    remaining = sorted(set(range(177)) - set(complete))
    shards = [{"number": number, "indices": remaining[number - 1::16],
               "output_root": f"{OUTPUT_PREFIX}_shard{number}"} for number in range(1, 17)]
    assigned = [index for shard in shards for index in shard["indices"]]
    if sorted(assigned) != remaining or len(assigned) != len(set(assigned)):
        raise ValueError("HANDOFF_PARTITION_INCOMPLETE_OR_OVERLAPPING")
    plan = {"schema": "10s_gallery_parallel16_handoff_v1", "manifest_sha256": digest,
            "reader_node": 5000, "previous_plan_sha256": hashlib.sha256(old_plan_bytes).hexdigest(),
            "previous_source_verification": old_plan["source_verification"],
            "source_verification": verification,
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
