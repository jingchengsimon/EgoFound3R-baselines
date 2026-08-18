import json
from pathlib import Path

from formal_evaluation.materialize_six_dataset_input_queue import _tasks


def test_tasks_preserve_per_dataset_50_window_bound(tmp_path: Path) -> None:
    rows = []
    for dataset, count in {"h2o": 51, "taco": 1, "hot3d": 1, "oakink_v2": 1, "arctic": 1, "hoi4d": 1}.items():
        for index in range(count):
            rows.append({
                "dataset": dataset,
                "sequence_id": dataset,
                "window_id": f"{dataset}:{index}",
                "frame_ids": [f"{frame:05d}" for frame in range(60)],
                "window_size": 60,
                "window_stride": 60,
                "window_overlap": 0,
            })
    manifest = tmp_path / "windows.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    tasks = _tasks(manifest, 50)

    h2o = [task for task in tasks if task["dataset"] == "h2o"]
    assert [task["window_count"] for task in h2o] == [50, 1]
    assert len(tasks) == 7
