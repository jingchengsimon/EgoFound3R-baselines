from __future__ import annotations

import torch
from torch import nn

from .transformer_blocks import TransformerDecoderBlock


class HandPromptAdapterV2(nn.Module):
    prompt_type_count = 3

    def __init__(
        self,
        dino_dim: int,
        agg_dim: int,
        prompt_dim: int,
        hand_feature_dim: int | None = None,
        num_layers: int = 6,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        joint_count: int = 21,
        enable_presence_head: bool = False,
    ) -> None:
        super().__init__()
        self.prompt_dim = prompt_dim
        self.joint_count = joint_count
        self.enable_presence_head = bool(enable_presence_head)
        self.hand_feature_dim = int(hand_feature_dim) if hand_feature_dim is not None else prompt_dim * 2
        self.dino_proj = nn.Linear(dino_dim, self.hand_feature_dim)
        self.agg_proj = nn.Linear(agg_dim, self.hand_feature_dim)
        self.token_fuse = nn.Linear(self.hand_feature_dim * 2, self.hand_feature_dim)
        self.side_type_queries = nn.Parameter(
            torch.randn(1, 1, 2, self.prompt_type_count, self.hand_feature_dim) * 0.02
        )
        self.query_norm = nn.LayerNorm(self.hand_feature_dim)
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(self.hand_feature_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.hand_feature_dim)
        self.keypoint_feature_proj = nn.Linear(dino_dim, self.hand_feature_dim)
        self.keypoint_coord_proj = nn.Linear(3, self.hand_feature_dim)
        self.keypoint_type_embed = nn.Parameter(torch.randn(1, 1, 1, joint_count, self.hand_feature_dim) * 0.02)
        self.missing_keypoint_token = nn.Parameter(torch.randn(1, 1, 1, joint_count, self.hand_feature_dim) * 0.02)
        self.loc_context_proj = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim),
            nn.Linear(self.hand_feature_dim, self.hand_feature_dim),
        )
        self.prompt_projector = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim),
            nn.Linear(self.hand_feature_dim, prompt_dim),
        )
        self.keypoint_prompt_projector = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim),
            nn.Linear(self.hand_feature_dim, prompt_dim),
        )
        self.scene_metric_feature_projector = nn.Sequential(
            nn.LayerNorm(agg_dim),
            nn.Linear(agg_dim, self.hand_feature_dim),
        )
        self.scene_metric_prompt_projector = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim),
            nn.Linear(self.hand_feature_dim, prompt_dim),
        )
        self.hand_metric_residual_projector = nn.Sequential(
            nn.LayerNorm(self.hand_feature_dim),
            nn.Linear(self.hand_feature_dim, prompt_dim),
        )
        if self.enable_presence_head:
            self.presence_head = nn.Sequential(
                nn.LayerNorm(self.hand_feature_dim),
                nn.Linear(self.hand_feature_dim, 1),
            )

    def _keypoint_tokens(
        self,
        *,
        keypoint_features: torch.Tensor,
        keypoint_uv: torch.Tensor,
        keypoint_conf: torch.Tensor,
        keypoint_mask: torch.Tensor,
    ) -> torch.Tensor:
        coord_input = torch.cat([keypoint_uv, keypoint_conf.unsqueeze(-1)], dim=-1)
        tokens = (
            self.keypoint_feature_proj(keypoint_features)
            + self.keypoint_coord_proj(coord_input)
            + self.keypoint_type_embed.to(device=keypoint_features.device, dtype=keypoint_features.dtype)
        )
        missing_tokens = (
            self.missing_keypoint_token.to(device=keypoint_features.device, dtype=keypoint_features.dtype)
            + self.keypoint_type_embed.to(device=keypoint_features.device, dtype=keypoint_features.dtype)
        )
        missing_tokens = missing_tokens.expand_as(tokens)
        return torch.where(keypoint_mask.unsqueeze(-1), tokens, missing_tokens)

    def forward(
        self,
        dino_tokens: torch.Tensor,
        agg_tokens: torch.Tensor,
        *,
        keypoint_features: torch.Tensor,
        keypoint_uv: torch.Tensor,
        keypoint_conf: torch.Tensor,
        keypoint_mask: torch.Tensor,
        presence_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if dino_tokens.ndim != 4 or agg_tokens.ndim != 4:
            raise ValueError("HandPromptAdapterV2 expects [B, T, N, C] DINO and aggregator tokens")
        if dino_tokens.shape[:3] != agg_tokens.shape[:3]:
            raise ValueError("DINO and aggregator tokens must share [B, T, N]")
        batch_size, num_frames, token_count = dino_tokens.shape[:3]
        flat_dino_tokens = dino_tokens.reshape(batch_size * num_frames, token_count, dino_tokens.shape[-1])
        flat_agg_tokens = agg_tokens.reshape(batch_size * num_frames, token_count, agg_tokens.shape[-1])
        dino_features = self.dino_proj(flat_dino_tokens)
        agg_features = self.agg_proj(flat_agg_tokens)
        memory = self.token_fuse(torch.cat([dino_features, agg_features], dim=-1))
        query = self.side_type_queries.to(device=memory.device, dtype=memory.dtype).expand(
            batch_size,
            num_frames,
            -1,
            -1,
            -1,
        )
        query = self.query_norm(
            query.reshape(batch_size * num_frames, 2 * self.prompt_type_count, self.hand_feature_dim)
        )
        for block in self.blocks:
            query = block(query, memory)
        hand_features = self.output_norm(query).reshape(
            batch_size,
            num_frames,
            2,
            self.prompt_type_count,
            self.hand_feature_dim,
        )

        keypoint_mask = keypoint_mask.to(device=hand_features.device, dtype=torch.bool)
        keypoint_tokens = self._keypoint_tokens(
            keypoint_features=keypoint_features.to(device=hand_features.device, dtype=hand_features.dtype),
            keypoint_uv=keypoint_uv.to(device=hand_features.device, dtype=hand_features.dtype),
            keypoint_conf=keypoint_conf.to(device=hand_features.device, dtype=hand_features.dtype),
            keypoint_mask=keypoint_mask,
        )
        root_context = keypoint_tokens[..., 0, :]
        joint_context = keypoint_tokens.mean(dim=3)
        loc_context = torch.stack([root_context, joint_context, joint_context], dim=3)
        hand_features = hand_features + self.loc_context_proj(loc_context)
        hand_prompt_set = self.prompt_projector(hand_features)
        keypoint_prompt_tokens = self.keypoint_prompt_projector(keypoint_tokens)

        side_features = hand_features.mean(dim=3)
        if self.enable_presence_head:
            presence_logits = self.presence_head(side_features).squeeze(-1)
            if presence_targets is not None:
                if presence_targets.shape != presence_logits.shape:
                    raise ValueError("presence_targets must have shape [B, T, 2]")
                presence_mask_2d = presence_targets.to(device=presence_logits.device, dtype=torch.bool)
            else:
                presence_mask_2d = presence_logits > 0
            presence_weight = torch.where(
                presence_mask_2d.unsqueeze(-1),
                presence_logits.sigmoid().unsqueeze(-1),
                torch.zeros_like(side_features[..., :1]),
            )
        else:
            if presence_targets is not None:
                if presence_targets.shape != side_features.shape[:-1]:
                    raise ValueError("presence_targets must have shape [B, T, 2]")
                presence_mask_2d = presence_targets.to(device=side_features.device, dtype=torch.bool)
            else:
                presence_mask_2d = keypoint_mask.any(dim=-1)
            positive_logits = torch.full(side_features.shape[:-1], 20.0, device=side_features.device, dtype=side_features.dtype)
            presence_logits = torch.where(presence_mask_2d, positive_logits, -positive_logits)
            presence_weight = presence_mask_2d.to(dtype=side_features.dtype).unsqueeze(-1)
        presence_mask = presence_mask_2d.unsqueeze(-1)

        scene_summary = flat_agg_tokens.mean(dim=1)
        scene_metric_feature = self.scene_metric_feature_projector(scene_summary).reshape(
            batch_size,
            num_frames,
            self.hand_feature_dim,
        )
        scene_metric_prompt = self.scene_metric_prompt_projector(scene_metric_feature)
        metric_denominator = presence_weight.sum(dim=2).clamp_min(1e-6)
        pooled_hand_metric_feature = (side_features.detach() * presence_weight).sum(dim=2) / metric_denominator
        has_hand = presence_mask.any(dim=2)
        hand_metric_feature = torch.where(
            has_hand.expand_as(pooled_hand_metric_feature),
            pooled_hand_metric_feature,
            torch.zeros_like(pooled_hand_metric_feature),
        )
        hand_metric_gate = has_hand.to(dtype=hand_features.dtype)
        hand_metric_residual = self.hand_metric_residual_projector(hand_metric_feature)
        metric_feature = scene_metric_feature + hand_metric_gate * hand_metric_feature
        metric_prompt = (scene_metric_prompt + hand_metric_gate * hand_metric_residual).unsqueeze(2)
        return {
            "hand_feature_set": hand_features,
            "hand_prompt_set": hand_prompt_set,
            "keypoint_prompt_tokens": keypoint_prompt_tokens,
            "metric_feature": metric_feature,
            "scene_metric_feature": scene_metric_feature,
            "hand_metric_feature": hand_metric_feature,
            "hand_observed_ratio": has_hand.squeeze(-1).to(dtype=hand_features.dtype).mean(dim=1),
            "metric_prompt": metric_prompt,
            "presence_logits": presence_logits,
            "presence_mask": presence_mask_2d,
        }
