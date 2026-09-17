import unittest

from formal_evaluation.contact.sharding import select_window_shard


class ContactShardingTest(unittest.TestCase):
    def test_window_shards_are_disjoint_and_complete(self) -> None:
        windows = {f"w{i}": ("sequence", [str(i)]) for i in range(7)}
        records = {key: {"window_id": key} for key in windows}
        locations = {i: (f"w{i}", 0) for i in range(7)}
        shards = [select_window_shard(windows, records, locations, num_shards=3, shard_index=i) for i in range(3)]
        self.assertEqual([len(shard[0]) for shard in shards], [3, 2, 2])
        self.assertEqual(set().union(*(set(shard[0]) for shard in shards)), set(windows))
        self.assertEqual(sum((shard[2] for shard in shards), []), [0, 3, 6, 1, 4, 2, 5])
