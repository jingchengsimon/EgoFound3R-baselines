import json
from pathlib import Path

from formal_evaluation.dispatch_ready_formal import ready_indices


def test_ready_indices_requires_every_shard(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    paths = []
    for shard in range(2):
        index = root / f"window_inputs_hot3d_shard_{shard:03d}_of_002.jsonl"
        index.write_text("{}\n", encoding="utf-8")
        index.with_suffix(".status.json").write_text(json.dumps({
            "status": "complete", "index": str(index), "shard_index": shard, "shard_count": 2,
        }), encoding="utf-8")
        paths.append(index)
    assert ready_indices(root, "hot3d") == paths
    paths[1].with_suffix(".status.json").unlink()
    assert ready_indices(root, "hot3d") == []
