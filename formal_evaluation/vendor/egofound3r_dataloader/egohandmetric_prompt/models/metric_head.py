from __future__ import annotations

import torch
from torch import nn


class MetricHead(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        hand_feature_dim: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.hand_feature_dim = self.prompt_dim if hand_feature_dim is None else int(hand_feature_dim)
        self.hidden_dim = self.prompt_dim if hidden_dim is None else int(hidden_dim)
        self.hand_feature_proj = (
            nn.Identity()
            if self.hand_feature_dim == self.prompt_dim
            else nn.Sequential(
                nn.LayerNorm(self.hand_feature_dim),
                nn.Linear(self.hand_feature_dim, self.prompt_dim),
            )
        )
        self.frame_fuse = nn.Sequential(
            nn.LayerNorm(self.prompt_dim * 2),
            nn.Linear(self.prompt_dim * 2, self.hidden_dim),
            nn.GELU(),
        )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        final_metric_prompt: torch.Tensor,
        *,
        metric_feature: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if final_metric_prompt.ndim == 4:
            if final_metric_prompt.shape[2] != 1:
                raise ValueError("final_metric_prompt with 4 dims must have one metric prompt")
            final_metric_prompt = final_metric_prompt.squeeze(2)
        if final_metric_prompt.ndim != 3:
            raise ValueError("final_metric_prompt must have shape [B, T, C] or [B, T, 1, C]")
        projected_feature = self.hand_feature_proj(metric_feature)
        if projected_feature.shape != final_metric_prompt.shape:
            raise ValueError("metric_feature must match final_metric_prompt after projection")
        fused = self.frame_fuse(torch.cat([final_metric_prompt, projected_feature], dim=-1))
        if valid_mask is None:
            valid_mask = torch.ones(fused.shape[:2], dtype=torch.bool, device=fused.device)
        valid = valid_mask.to(device=fused.device, dtype=torch.bool)
        weights = valid.to(dtype=fused.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        pooled = (fused * weights).sum(dim=1) / denominator
        log_metric_value = self.scale_head(pooled).squeeze(-1)
        return {
            "log_metric_value": log_metric_value,
            "metric_value": torch.exp(log_metric_value),
        }
