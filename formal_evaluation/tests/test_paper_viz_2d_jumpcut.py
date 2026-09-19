from formal_evaluation.audit_paper_viz_2d_jumpcut import jump_blocks, stitched_entry


def axis(length=360, gap_after=179):
    result = []
    for index in range(length):
        frame = index if index <= gap_after else index + 40
        result.append({
            "window_id": f"sequence:{index // 60:05d}",
            "cache_id": f"cache-{index // 60}",
            "index": index % 60,
            "frame_id": f"{frame:06d}",
        })
    return result


def entry(dataset="h2o"):
    frames = axis(60, 1000)
    return {
        "dataset": dataset,
        "sequence_id": "sequence",
        "window_id": "sequence:00120-00179",
        "frame_id": "000150",
        "frame_ids": [item["frame_id"] for item in frames],
        "frame_refs": [{key: item[key] for key in ("window_id", "cache_id", "index")}
                       for item in frames],
        "within_window_step_p95": 1,
        "actual_length": 60,
    }


def test_h2o_is_centered_to_300_and_records_jump():
    result = stitched_entry(entry(), axis())
    assert len(result["frame_ids"]) == 300
    assert result["frame_ids"][result["center_index_in_clip"]] == "000150"
    assert result["jump_stitch"]["jump_count"] == 1
    assert len(result["jump_stitch"]["blocks"]) == 2


def test_taco_is_unchanged():
    source = entry("taco")
    result = stitched_entry(source, None)
    assert result["frame_ids"] == source["frame_ids"]
    assert result["jump_stitch"]["policy"] == "unchanged"


def test_jump_blocks_uses_registered_sampling_step():
    frames = [{"frame_id": value, "cache_id": "c", "index": index}
              for index, value in enumerate(("0", "4", "8", "40", "44"))]
    assert jump_blocks(frames, 4) == [
        {"start": 0, "end_exclusive": 3, "first_frame_id": "0", "last_frame_id": "8"},
        {"start": 3, "end_exclusive": 5, "first_frame_id": "40", "last_frame_id": "44"},
    ]
