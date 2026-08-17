from __future__ import annotations

import math

import torch
from torch import nn


class MaskedSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除。")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, _ = x.shape
        qkv = self.qkv(x).view(batch_size, num_frames, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attention_mask[:, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(batch_size, num_frames, self.hidden_dim)
        return self.proj(attended)


class FlowBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MaskedSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        return x + self.mlp(self.norm2(x))


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        num_markers: int,
        state_dim: int,
        camera_pose_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        condition_dim = (num_markers * 3 * 2) + (num_markers * 2) + camera_pose_dim
        self.state_proj = nn.Linear(state_dim * 2, hidden_dim)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([FlowBlock(hidden_dim, num_heads) for _ in range(num_layers)])
        self.output = nn.Linear(hidden_dim, state_dim * 2)

    def forward(
        self,
        x_t: torch.Tensor,
        marker_xyz: torch.Tensor,
        marker_visibility: torch.Tensor,
        camera_pose: torch.Tensor,
        attention_mask: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_frames, side_count, _ = x_t.shape
        if side_count != 2:
            raise ValueError("FlowMatchingModel 当前只支持双手状态输入，要求 side_count=2。")

        flattened_state = x_t.reshape(batch_size, num_frames, side_count * self.state_dim)
        flattened_markers = marker_xyz.reshape(batch_size, num_frames, -1)
        flattened_visibility = marker_visibility.reshape(batch_size, num_frames, -1)
        condition = torch.cat([flattened_markers, flattened_visibility, camera_pose], dim=-1)
        hidden = self.state_proj(flattened_state) + self.condition_proj(condition)
        hidden = hidden + self.time_proj(time[:, None]).unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        velocity = self.output(hidden).reshape(batch_size, num_frames, side_count, self.state_dim)
        return velocity
