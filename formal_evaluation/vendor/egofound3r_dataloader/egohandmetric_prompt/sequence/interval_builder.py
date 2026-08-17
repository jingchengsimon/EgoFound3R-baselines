from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class IntervalBuildResult:
    side_visible: torch.Tensor
    side_intervals: list[list[list[tuple[int, int]]]]
    union_intervals: list[list[tuple[int, int]]]
    attention_mask: torch.Tensor


class IntervalBuilder:
    def __init__(
        self,
        visible_count_threshold: int,
        merge_gap_threshold: int,
        filter_length_threshold: int,
        visibility_threshold: float = 0.5,
    ) -> None:
        self.visible_count_threshold = visible_count_threshold
        self.merge_gap_threshold = merge_gap_threshold
        self.filter_length_threshold = filter_length_threshold
        self.visibility_threshold = visibility_threshold

    def build(self, marker_visibility: torch.Tensor) -> IntervalBuildResult:
        visible_counts = (marker_visibility >= self.visibility_threshold).sum(dim=-1)
        side_visible = (visible_counts >= self.visible_count_threshold).transpose(1, 2).contiguous()
        batch_size, side_count, num_frames = side_visible.shape

        side_intervals: list[list[list[tuple[int, int]]]] = []
        union_intervals: list[list[tuple[int, int]]] = []
        attention_mask = torch.zeros(batch_size, num_frames, num_frames, dtype=torch.bool, device=marker_visibility.device)

        for batch_index in range(batch_size):
            sample_side_intervals: list[list[tuple[int, int]]] = []
            union_visible = torch.zeros(num_frames, dtype=torch.bool, device=marker_visibility.device)
            for side_index in range(side_count):
                intervals = self._intervals_from_bool(side_visible[batch_index, side_index])
                intervals = self._merge_intervals(intervals)
                intervals = self._filter_intervals(intervals)
                sample_side_intervals.append(intervals)
                for start, end in intervals:
                    union_visible[start:end] = True
            sample_union_intervals = self._intervals_from_bool(union_visible)
            side_intervals.append(sample_side_intervals)
            union_intervals.append(sample_union_intervals)
            for start, end in sample_union_intervals:
                attention_mask[batch_index, start:end, start:end] = True

        return IntervalBuildResult(
            side_visible=side_visible,
            side_intervals=side_intervals,
            union_intervals=union_intervals,
            attention_mask=attention_mask,
        )

    @staticmethod
    def _intervals_from_bool(flags: torch.Tensor) -> list[tuple[int, int]]:
        intervals: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(flags.tolist()):
            if value and start is None:
                start = index
            elif not value and start is not None:
                intervals.append((start, index))
                start = None
        if start is not None:
            intervals.append((start, len(flags)))
        return intervals

    def _merge_intervals(self, intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start - last_end <= self.merge_gap_threshold:
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))
        return merged

    def _filter_intervals(self, intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [interval for interval in intervals if interval[1] - interval[0] >= self.filter_length_threshold]
