from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class KeypointHeatmapTargets:
    heatmaps: torch.Tensor
    mask: torch.Tensor
    cell_indices: torch.Tensor
    offsets: torch.Tensor


def build_keypoint_heatmap_targets(
    batch: dict[str, Any],
    *,
    heatmap_height: int,
    heatmap_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KeypointHeatmapTargets:
    batch_size, num_frames = batch["images"].shape[:2]
    joint_count = 21
    heatmaps = torch.zeros(
        batch_size,
        num_frames,
        2,
        joint_count,
        heatmap_height,
        heatmap_width,
        device=device,
        dtype=dtype,
    )
    mask = torch.zeros(batch_size, num_frames, 2, joint_count, device=device, dtype=torch.bool)
    cell_indices = torch.full(
        (batch_size, num_frames, 2, joint_count),
        -1,
        device=device,
        dtype=torch.long,
    )
    offsets = torch.zeros(batch_size, num_frames, 2, joint_count, 2, device=device, dtype=dtype)
    joints_2d = batch.get("joints_2d_targets")
    joints_2d_mask = batch.get("joints_2d_supervision_mask")
    presence_targets = batch.get("presence_targets")
    presence_mask = batch.get("presence_supervision_mask")
    if joints_2d is None or joints_2d_mask is None:
        return KeypointHeatmapTargets(heatmaps, mask, cell_indices, offsets)

    joints_2d = joints_2d.to(device=device, dtype=dtype)
    joints_2d_mask = joints_2d_mask.to(device=device, dtype=torch.bool)
    if presence_targets is not None:
        presence_targets = presence_targets.to(device=device, dtype=dtype) > 0.5
    else:
        presence_targets = torch.ones(batch_size, num_frames, 2, device=device, dtype=torch.bool)
    if presence_mask is not None:
        presence_mask = presence_mask.to(device=device, dtype=torch.bool)
        presence_targets = presence_targets & presence_mask

    image_height, image_width = batch["images"].shape[-2:]
    scale_x = float(max(heatmap_width - 1, 1)) / float(max(image_width - 1, 1))
    scale_y = float(max(heatmap_height - 1, 1)) / float(max(image_height - 1, 1))
    valid_side = joints_2d_mask & presence_targets
    for batch_index in range(batch_size):
        for frame_index in range(num_frames):
            for side_index in range(2):
                if not bool(valid_side[batch_index, frame_index, side_index].item()):
                    continue
                for joint_index in range(min(joint_count, joints_2d.shape[3])):
                    uv = joints_2d[batch_index, frame_index, side_index, joint_index]
                    if not bool(torch.isfinite(uv).all().item()):
                        continue
                    x = float(uv[0].item())
                    y = float(uv[1].item())
                    if x < 0.0 or x > image_width - 1 or y < 0.0 or y > image_height - 1:
                        continue
                    continuous_x = x * scale_x
                    continuous_y = y * scale_y
                    cell_x = min(max(int(round(continuous_x)), 0), heatmap_width - 1)
                    cell_y = min(max(int(round(continuous_y)), 0), heatmap_height - 1)
                    heatmaps[batch_index, frame_index, side_index, joint_index, cell_y, cell_x] = 1.0
                    mask[batch_index, frame_index, side_index, joint_index] = True
                    cell_indices[batch_index, frame_index, side_index, joint_index] = cell_y * heatmap_width + cell_x
                    offsets[batch_index, frame_index, side_index, joint_index, 0] = continuous_x - float(cell_x)
                    offsets[batch_index, frame_index, side_index, joint_index, 1] = continuous_y - float(cell_y)

    return KeypointHeatmapTargets(heatmaps, mask, cell_indices, offsets)
