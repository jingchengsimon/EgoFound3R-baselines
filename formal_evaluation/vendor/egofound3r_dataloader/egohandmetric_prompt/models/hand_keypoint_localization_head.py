from __future__ import annotations

import torch
from torch import nn


class HandKeypointLocalizationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        joint_count: int = 21,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.joint_count = joint_count
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.heatmap_head = nn.Conv2d(hidden_channels, 2 * joint_count, kernel_size=1)
        self.offset_head = nn.Conv2d(hidden_channels, 2 * joint_count * 2, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_frames, channels, height, width = feature_map.shape
        flattened = feature_map.reshape(batch_size * num_frames, channels, height, width)
        features = self.stem(flattened)
        logits = self.heatmap_head(features)
        offsets = self.offset_head(features)
        logits = logits.reshape(batch_size, num_frames, 2, self.joint_count, height, width)
        offsets = offsets.reshape(batch_size, num_frames, 2, self.joint_count, 2, height, width)
        return {
            "keypoint_heatmap_logits": logits,
            "keypoint_offset": offsets,
        }
