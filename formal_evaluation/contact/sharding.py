from __future__ import annotations


def select_window_shard(windows, input_records, cache_locations, *, num_shards: int, shard_index: int):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    assigned = {window_id for index, window_id in enumerate(windows) if index % num_shards == shard_index}
    selected = [index for index, (window_id, _) in cache_locations.items() if window_id in assigned]
    return (
        {key: value for key, value in windows.items() if key in assigned},
        {key: value for key, value in input_records.items() if key in assigned},
        selected,
    )
