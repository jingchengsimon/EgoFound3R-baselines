import json
from pathlib import Path

from formal_evaluation.materialize_six_dataset_input_queue import _tasks
from formal_evaluation.promote_window_inputs_to_oss import promote_dataset
from formal_evaluation.run_formal_window_queue import _input_tasks


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


def test_formal_queue_requires_completed_input_sentinel(tmp_path: Path) -> None:
    rgb, geometry = tmp_path / "rgb.png", tmp_path / "geometry.npz"
    rgb.write_bytes(b"rgb"); geometry.write_bytes(b"geometry")
    record = {
        "window_input_version": "six_dataset_window_input_v1", "dataset": "h2o", "cache_id": "cache",
        "sequence_id": "subject4_ego/h1/0", "window_id": "window", "frame_ids": ["000000"],
        "rgb_paths": [str(rgb)], "geometry_paths": [str(geometry)],
    }
    window_input = tmp_path / "window_input.json"
    window_input.write_text(json.dumps(record), encoding="utf-8")
    index = tmp_path / "window_inputs_h2o_shard_000_of_001.jsonl"
    index.write_text(json.dumps({"window_input": str(window_input)}) + "\n", encoding="utf-8")
    index.with_suffix(".status.json").write_text(json.dumps({
        "status": "complete", "index": str(index), "window_count": 1,
    }), encoding="utf-8")

    tasks = _input_tasks([index])

    assert tasks[0]["dataset"] == "h2o"
    assert tasks[0]["records"][0]["_path"] == str(window_input)


def test_oss_promotion_rewrites_indexes_records_and_symlinks(tmp_path: Path) -> None:
    source, destination = tmp_path / "cpfs", tmp_path / "oss"
    source.mkdir(); destination.mkdir()
    input_dir = source / "h2o" / "cache"
    rgb = input_dir / "rgb.png"; geometry = input_dir / "geometry.npz"
    input_dir.mkdir(parents=True); rgb.write_bytes(b"rgb"); geometry.write_bytes(b"geometry")
    (input_dir / "rgb_link.png").symlink_to(rgb)
    record = {
        "window_input_version": "six_dataset_window_input_v1", "dataset": "h2o", "cache_id": "cache",
        "sequence_id": "s", "window_id": "w", "frame_ids": ["000000"],
        "rgb_paths": [str(rgb)], "geometry_paths": [str(geometry)], "video_path": str(input_dir / "video.mp4"),
    }
    (input_dir / "video.mp4").write_bytes(b"video")
    window_input = input_dir / "window_input.json"; window_input.write_text(json.dumps(record), encoding="utf-8")
    index = source / "window_inputs_h2o_shard_000_of_001.jsonl"
    index.write_text(json.dumps({"window_input": str(window_input)}) + "\n", encoding="utf-8")
    index.with_suffix(".status.json").write_text(json.dumps({"status": "complete", "index": str(index), "window_count": 1}), encoding="utf-8")

    result = promote_dataset(source, destination, "h2o")

    assert result["window_count"] == 1
    promoted = json.loads((destination / "h2o" / "cache" / "window_input.json").read_text(encoding="utf-8"))
    assert promoted["rgb_paths"] == [str(destination / "h2o" / "cache" / "rgb.png")]
    assert (destination / "h2o" / "cache" / "rgb_link.png").resolve() == destination / "h2o" / "cache" / "rgb.png"
