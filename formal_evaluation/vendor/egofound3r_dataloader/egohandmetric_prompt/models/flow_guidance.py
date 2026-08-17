from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class FlowGuidanceOutput:
    dino_feature_map: torch.Tensor
    dino_tokens: torch.Tensor
    mid_tokens: torch.Tensor
    flow_pred: torch.Tensor
    flow_pair_mask: torch.Tensor
    gate: torch.Tensor


def _add_2d_sincos_position(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    dim = tokens.shape[-1]
    if dim % 4 != 0:
        raise ValueError(f"flow_motion_dim 必须能被 4 整除，当前为 {dim}")
    axis_dim = dim // 4
    exponent = torch.arange(axis_dim, device=tokens.device, dtype=torch.float32)
    exponent = exponent / max(axis_dim - 1, 1)
    omega = (1.0 / (10000.0**exponent)).to(dtype=tokens.dtype)
    y = torch.arange(height, device=tokens.device, dtype=tokens.dtype)
    x = torch.arange(width, device=tokens.device, dtype=tokens.dtype)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    position = torch.cat(
        [
            torch.sin(x_grid.reshape(-1, 1) * omega),
            torch.cos(x_grid.reshape(-1, 1) * omega),
            torch.sin(y_grid.reshape(-1, 1) * omega),
            torch.cos(y_grid.reshape(-1, 1) * omega),
        ],
        dim=-1,
    )
    return tokens + position.view(1, 1, height * width, dim)


class _MotionTransformerBlock(nn.Module):
    """标准 Transformer block: Cross-Attention (source→target) + FFN + 2×LayerNorm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_target = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attention_output = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """source: (N, T, D), target: (N, T, D) → output: (N, T, D)"""
        n, t, _ = source.shape
        # cross-attention: source attends to target
        x = self.norm1(source)
        target_normed = self.norm_target(target)
        q = self.query(x).reshape(n, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(target_normed).reshape(n, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(target_normed).reshape(n, t, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(n, t, self.dim)
        x = source + self.attention_output(attn_out)
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class FlowGuidanceBranch(nn.Module):
    def __init__(
        self,
        *,
        dino_dim: int,
        aggregator_dim: int,
        motion_dim: int = 768,
        num_heads: int = 12,
        num_motion_blocks: int = 6,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        gate_init_logit: float = -6.0,
    ) -> None:
        super().__init__()
        if motion_dim % num_heads != 0:
            raise ValueError("motion_dim 必须能被 num_heads 整除")
        if motion_dim % 4 != 0:
            raise ValueError("motion_dim 必须能被 4 整除以加入二维位置编码")
        if patch_size <= 0:
            raise ValueError("patch_size 必须为正整数")
        if num_motion_blocks <= 0:
            raise ValueError("num_motion_blocks 必须为正整数")
        self.motion_dim = int(motion_dim)
        self.num_heads = int(num_heads)
        self.patch_size = int(patch_size)
        # 输入投影
        self.dino_input = nn.Linear(dino_dim, motion_dim)
        self.aggregator_input = nn.Linear(aggregator_dim, motion_dim)
        self.input_fuse = nn.Sequential(nn.Linear(2 * motion_dim, motion_dim), nn.LayerNorm(motion_dim))
        # Transformer blocks
        self.motion_blocks = nn.ModuleList(
            [_MotionTransformerBlock(motion_dim, num_heads, mlp_ratio) for _ in range(num_motion_blocks)]
        )
        # 输出投影
        self.flow_head = nn.Conv2d(motion_dim, 2 * patch_size * patch_size, kernel_size=1)
        self.dino_residual = nn.Linear(motion_dim, dino_dim)
        self.aggregator_residual = nn.Linear(motion_dim, aggregator_dim)
        # 零初始化两个残差输出投影：使 init 时 dino_delta = agg_delta = 0，
        # 从而 guided = 原特征，无论 gate 多大都不扰动预训练 backbone。
        nn.init.zeros_(self.dino_residual.weight)
        nn.init.zeros_(self.dino_residual.bias)
        nn.init.zeros_(self.aggregator_residual.weight)
        nn.init.zeros_(self.aggregator_residual.bias)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init_logit)))

    def forward(
        self,
        *,
        dino_feature_map: torch.Tensor,
        dino_tokens: torch.Tensor,
        mid_tokens: torch.Tensor,
        flow_pair_mask: torch.Tensor | None = None,
    ) -> FlowGuidanceOutput:
        batch_size, num_frames, _, grid_height, grid_width = dino_feature_map.shape
        token_count = grid_height * grid_width
        if dino_tokens.shape[:3] != (batch_size, num_frames, token_count):
            raise ValueError("dino_tokens 与 dino_feature_map 的 patch 网格不一致")
        if mid_tokens.shape[:3] != (batch_size, num_frames, token_count):
            raise ValueError("mid_tokens 与 dino_feature_map 的 patch 网格不一致")
        pair_shape = (batch_size, max(num_frames - 1, 0))
        if flow_pair_mask is None:
            pair_mask = torch.ones(pair_shape, device=dino_tokens.device, dtype=torch.bool)
        else:
            pair_mask = flow_pair_mask.to(device=dino_tokens.device, dtype=torch.bool)
            if pair_mask.shape != pair_shape:
                raise ValueError(f"flow_pair_mask 形状应为 {pair_shape}，当前为 {tuple(pair_mask.shape)}")
        if num_frames < 2:
            flow_pred = dino_tokens.new_zeros(
                batch_size,
                0,
                2,
                grid_height * self.patch_size,
                grid_width * self.patch_size,
            )
            return FlowGuidanceOutput(
                dino_feature_map=dino_feature_map,
                dino_tokens=dino_tokens,
                mid_tokens=mid_tokens,
                flow_pred=flow_pred,
                flow_pair_mask=pair_mask,
                gate=torch.sigmoid(self.gate_logit),
            )

        projected = self.input_fuse(
            torch.cat([self.dino_input(dino_tokens), self.aggregator_input(mid_tokens)], dim=-1)
        )
        projected = _add_2d_sincos_position(projected, grid_height, grid_width)
        source = projected[:, :-1].reshape(-1, token_count, self.motion_dim)
        target = projected[:, 1:].reshape(-1, token_count, self.motion_dim)
        # 多层 Transformer：source attend to target
        for block in self.motion_blocks:
            source = block(source, target)
        motion = source.reshape(batch_size, num_frames - 1, token_count, self.motion_dim)
        pair_weight = pair_mask.to(dtype=motion.dtype).unsqueeze(-1).unsqueeze(-1)
        motion = motion * pair_weight

        motion_grid = motion.permute(0, 1, 3, 2).reshape(
            batch_size * (num_frames - 1), self.motion_dim, grid_height, grid_width
        )
        flow_pred = F.pixel_shuffle(self.flow_head(motion_grid), self.patch_size).reshape(
            batch_size,
            num_frames - 1,
            2,
            grid_height * self.patch_size,
            grid_width * self.patch_size,
        )
        flow_pred = flow_pred * pair_mask[:, :, None, None, None].to(dtype=flow_pred.dtype)

        dino_delta = self.dino_residual(motion) * pair_weight
        agg_delta = self.aggregator_residual(motion) * pair_weight
        dino_delta = torch.cat([dino_delta, torch.zeros_like(dino_delta[:, :1])], dim=1)
        agg_delta = torch.cat([agg_delta, torch.zeros_like(agg_delta[:, :1])], dim=1)
        gate = torch.sigmoid(self.gate_logit).to(dtype=dino_tokens.dtype)
        guided_dino_tokens = dino_tokens + gate * dino_delta.to(dtype=dino_tokens.dtype)
        guided_mid_tokens = mid_tokens + gate * agg_delta.to(dtype=mid_tokens.dtype)
        guided_dino_map = guided_dino_tokens.transpose(-1, -2).reshape_as(dino_feature_map)
        return FlowGuidanceOutput(
            dino_feature_map=guided_dino_map,
            dino_tokens=guided_dino_tokens,
            mid_tokens=guided_mid_tokens,
            flow_pred=flow_pred,
            flow_pair_mask=pair_mask,
            gate=gate,
        )
