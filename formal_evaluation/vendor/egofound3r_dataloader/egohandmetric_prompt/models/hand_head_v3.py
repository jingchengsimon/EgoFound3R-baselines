from __future__ import annotations

import torch
from torch import nn

from egohandmetric_prompt.data.marker_vertex_joint_map import validate_vertex_joint_weights
from egohandmetric_prompt.losses.contact_losses import contact_confidence_from_log

from .transformer_blocks import TransformerDecoderBlock


class HandHeadV3(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        num_markers: int,
        num_joints: int = 21,
        num_queries: int | None = None,
        hand_feature_dim: int | None = None,
        root_num_layers: int = 1,
        joint_num_layers: int = 5,
        vertex_num_layers: int = 5,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        vertex_joint_weights: torch.Tensor | None = None,
        random_init_contact_confidence_heads: bool = False,
    ) -> None:
        super().__init__()
        if int(num_joints) != 21:
            raise ValueError("HandHeadV3 currently requires 21 joints including root.")
        if num_queries is not None and int(num_queries) != int(num_markers):
            raise ValueError("HandHeadV3 requires num_queries == num_markers.")
        if vertex_joint_weights is None:
            raise ValueError("HandHeadV3 requires fixed vertex_joint_weights.")
        self.prompt_dim = int(prompt_dim)
        self.hand_feature_dim = self.prompt_dim if hand_feature_dim is None else int(hand_feature_dim)
        self.num_vertices = int(num_markers)
        self.num_joints = int(num_joints)
        validate_vertex_joint_weights(
            vertex_joint_weights,
            marker_count=self.num_vertices,
            joint_count=self.num_joints,
        )
        self.register_buffer("vertex_joint_weights", vertex_joint_weights.detach().to(dtype=torch.float32))

        self.root_query = nn.Parameter(torch.randn(1, 1, self.prompt_dim) * 0.02)
        self.joint_queries = nn.Parameter(torch.randn(1, self.num_joints - 1, self.prompt_dim) * 0.02)
        self.vertex_queries = nn.Parameter(torch.randn(1, self.num_vertices, self.prompt_dim) * 0.02)

        self.root_query_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.joint_query_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.vertex_query_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.root_keypoint_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.joint_keypoint_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.vertex_keypoint_proj = nn.Linear(self.prompt_dim, self.prompt_dim)
        self.root_2d_proj = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, self.prompt_dim))
        self.joint_2d_proj = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, self.prompt_dim))
        self.hand_feature_proj = (
            nn.Identity()
            if self.hand_feature_dim == self.prompt_dim
            else nn.Sequential(
                nn.LayerNorm(self.hand_feature_dim),
                nn.Linear(self.hand_feature_dim, self.prompt_dim),
            )
        )
        self.metric_feature_proj = (
            nn.Identity()
            if self.hand_feature_dim == self.prompt_dim
            else nn.Sequential(
                nn.LayerNorm(self.hand_feature_dim),
                nn.Linear(self.hand_feature_dim, self.prompt_dim),
            )
        )

        self.root_blocks = nn.ModuleList(
            [TransformerDecoderBlock(self.prompt_dim, num_heads, mlp_ratio) for _ in range(root_num_layers)]
        )
        self.joint_blocks = nn.ModuleList(
            [TransformerDecoderBlock(self.prompt_dim, num_heads, mlp_ratio) for _ in range(joint_num_layers)]
        )
        self.vertex_blocks = nn.ModuleList(
            [TransformerDecoderBlock(self.prompt_dim, num_heads, mlp_ratio) for _ in range(vertex_num_layers)]
        )
        self.root_output_norm = nn.LayerNorm(self.prompt_dim)
        self.joint_output_norm = nn.LayerNorm(self.prompt_dim)
        self.vertex_output_norm = nn.LayerNorm(self.prompt_dim)

        self.root_xyz_head = nn.Linear(self.prompt_dim, 3)
        self.root_visibility_head = nn.Linear(self.prompt_dim, 1)
        self.root_contact_head = nn.Linear(self.prompt_dim, 1)
        self.root_contact_distance_head = nn.Linear(self.prompt_dim, 1)
        self.root_contact_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.root_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.joint_xyz_head = nn.Linear(self.prompt_dim, 3)
        self.joint_visibility_head = nn.Linear(self.prompt_dim, 1)
        self.joint_contact_head = nn.Linear(self.prompt_dim, 1)
        self.joint_contact_distance_head = nn.Linear(self.prompt_dim, 1)
        self.joint_contact_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.joint_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_xyz_head = nn.Linear(self.prompt_dim, 3)
        self.vertex_visibility_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_contact_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_contact_distance_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_contact_log_conf_head = nn.Linear(self.prompt_dim, 1)
        self.vertex_log_conf_head = nn.Linear(self.prompt_dim, 1)
        confidence_heads = [
            self.root_log_conf_head,
            self.joint_log_conf_head,
            self.vertex_log_conf_head,
        ]
        if not random_init_contact_confidence_heads:
            confidence_heads.extend(
                [
                    self.root_contact_log_conf_head,
                    self.joint_contact_log_conf_head,
                    self.vertex_contact_log_conf_head,
                ]
            )
        for conf_head in confidence_heads:
            nn.init.zeros_(conf_head.weight)
            nn.init.zeros_(conf_head.bias)

    def _checked_keypoint_inputs(
        self,
        *,
        final_prompt_set: torch.Tensor,
        keypoint_prompt_tokens: torch.Tensor,
        keypoint_uv: torch.Tensor,
        keypoint_conf: torch.Tensor,
        keypoint_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = final_prompt_set.shape[0]
        expected_token_shape = (batch_size, self.num_joints, self.prompt_dim)
        if tuple(keypoint_prompt_tokens.shape) != expected_token_shape:
            raise ValueError(f"keypoint_prompt_tokens must have shape {expected_token_shape}.")
        if tuple(keypoint_uv.shape) != (batch_size, self.num_joints, 2):
            raise ValueError("keypoint_uv must have shape [N, 21, 2].")
        if tuple(keypoint_conf.shape) != (batch_size, self.num_joints):
            raise ValueError("keypoint_conf must have shape [N, 21].")
        if tuple(keypoint_mask.shape) != (batch_size, self.num_joints):
            raise ValueError("keypoint_mask must have shape [N, 21].")
        dtype = final_prompt_set.dtype
        device = final_prompt_set.device
        keypoint_mask = keypoint_mask.to(device=device, dtype=torch.bool)
        keypoint_prompt_tokens = keypoint_prompt_tokens.to(device=device, dtype=dtype)
        keypoint_uv = keypoint_uv.to(device=device, dtype=dtype)
        keypoint_conf = keypoint_conf.to(device=device, dtype=dtype)
        keypoint_uv = torch.where(keypoint_mask.unsqueeze(-1), keypoint_uv, torch.zeros_like(keypoint_uv))
        keypoint_conf = torch.where(keypoint_mask, keypoint_conf, torch.zeros_like(keypoint_conf))
        keypoint_2d = torch.cat([keypoint_uv, keypoint_conf.unsqueeze(-1)], dim=-1)
        return keypoint_prompt_tokens, keypoint_2d, keypoint_conf, keypoint_mask

    def _project_hand_features(self, final_prompt_set: torch.Tensor, hand_features: torch.Tensor | None) -> list[torch.Tensor]:
        root_prompt, joint_prompt, vertex_prompt = final_prompt_set.unbind(dim=1)
        if hand_features is None:
            return [root_prompt[:, None], joint_prompt[:, None], vertex_prompt[:, None]]
        projected_features = self.hand_feature_proj(hand_features)
        if projected_features.ndim != 3 or tuple(projected_features.shape[:2]) != (final_prompt_set.shape[0], 3):
            raise ValueError("hand_features must have shape [N, 3, C].")
        return [projected_features[:, 0:1], projected_features[:, 1:2], projected_features[:, 2:3]]

    def _metric_memory(
        self,
        *,
        final_prompt_set: torch.Tensor,
        metric_prompt: torch.Tensor | None,
        metric_feature: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        dtype = final_prompt_set.dtype
        device = final_prompt_set.device
        memory = []
        if metric_prompt is not None:
            if tuple(metric_prompt.shape) != (final_prompt_set.shape[0], 1, self.prompt_dim):
                raise ValueError("metric_prompt must have shape [N, 1, C].")
            memory.append(metric_prompt.to(device=device, dtype=dtype))
        if metric_feature is not None:
            if metric_feature.ndim != 2 or metric_feature.shape[0] != final_prompt_set.shape[0]:
                raise ValueError("metric_feature must have shape [N, C].")
            projected_metric = self.metric_feature_proj(metric_feature.to(device=device, dtype=dtype))
            memory.append(projected_metric[:, None])
        return memory

    def forward(
        self,
        final_prompt_set: torch.Tensor,
        *,
        hand_features: torch.Tensor | None = None,
        keypoint_prompt_tokens: torch.Tensor,
        keypoint_uv: torch.Tensor,
        keypoint_conf: torch.Tensor,
        keypoint_mask: torch.Tensor,
        metric_prompt: torch.Tensor | None = None,
        metric_feature: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if final_prompt_set.ndim != 3 or tuple(final_prompt_set.shape[1:]) != (3, self.prompt_dim):
            raise ValueError("final_prompt_set must have shape [N, 3, C].")
        root_prompt = final_prompt_set[:, 0]
        joint_prompt = final_prompt_set[:, 1]
        vertex_prompt = final_prompt_set[:, 2]
        keypoint_prompt_tokens, keypoint_2d, _, _ = self._checked_keypoint_inputs(
            final_prompt_set=final_prompt_set,
            keypoint_prompt_tokens=keypoint_prompt_tokens,
            keypoint_uv=keypoint_uv,
            keypoint_conf=keypoint_conf,
            keypoint_mask=keypoint_mask,
        )
        root_feature, joint_feature, vertex_feature = self._project_hand_features(final_prompt_set, hand_features)
        metric_memory = self._metric_memory(
            final_prompt_set=final_prompt_set,
            metric_prompt=metric_prompt,
            metric_feature=metric_feature,
        )

        root_query = (
            self.root_query.to(device=final_prompt_set.device, dtype=final_prompt_set.dtype).expand(final_prompt_set.shape[0], -1, -1)
            + self.root_query_proj(root_prompt)[:, None]
            + self.root_keypoint_proj(keypoint_prompt_tokens[:, 0])[:, None]
            + self.root_2d_proj(keypoint_2d[:, 0])[:, None]
        )
        root_memory = torch.cat(
            [
                root_prompt[:, None],
                root_feature,
                keypoint_prompt_tokens[:, 0:1],
                *metric_memory,
            ],
            dim=1,
        )
        for block in self.root_blocks:
            root_query = block(root_query, root_memory)
        root_summary = self.root_output_norm(root_query[:, 0])
        root_xyz = self.root_xyz_head(root_summary)
        root_visibility_logits = self.root_visibility_head(root_summary).squeeze(-1)
        root_log_conf = self.root_log_conf_head(root_summary).squeeze(-1)

        joint_queries = (
            self.joint_queries.to(device=final_prompt_set.device, dtype=final_prompt_set.dtype).expand(final_prompt_set.shape[0], -1, -1)
            + self.joint_query_proj(joint_prompt).unsqueeze(1)
            + self.joint_keypoint_proj(keypoint_prompt_tokens[:, 1:])
            + self.joint_2d_proj(keypoint_2d[:, 1:])
        )
        joint_memory = torch.cat(
            [
                root_summary[:, None],
                root_prompt[:, None],
                joint_prompt[:, None],
                root_feature,
                joint_feature,
                keypoint_prompt_tokens,
            ],
            dim=1,
        )
        for block in self.joint_blocks:
            joint_queries = block(joint_queries, joint_memory)
        joint_summary = self.joint_output_norm(joint_queries)
        joint_offset_nonroot = self.joint_xyz_head(joint_summary)
        joint_offset = root_xyz.new_zeros(root_xyz.shape[0], self.num_joints, 3)
        joint_offset[:, 1:] = joint_offset_nonroot
        joint_xyz = root_xyz[:, None, :] + joint_offset
        joint_visibility_logits = root_visibility_logits.new_empty(root_xyz.shape[0], self.num_joints)
        joint_visibility_logits[:, 0] = root_visibility_logits
        joint_visibility_logits[:, 1:] = self.joint_visibility_head(joint_summary).squeeze(-1)
        joint_contact_logits = root_visibility_logits.new_empty(root_xyz.shape[0], self.num_joints)
        joint_contact_logits[:, 0] = self.root_contact_head(root_summary).squeeze(-1)
        joint_contact_logits[:, 1:] = self.joint_contact_head(joint_summary).squeeze(-1)
        joint_contact_log_distance = root_visibility_logits.new_empty(root_xyz.shape[0], self.num_joints)
        joint_contact_log_distance[:, 0] = self.root_contact_distance_head(root_summary).squeeze(-1)
        joint_contact_log_distance[:, 1:] = self.joint_contact_distance_head(joint_summary).squeeze(-1)
        joint_contact_log_conf = root_visibility_logits.new_empty(root_xyz.shape[0], self.num_joints)
        joint_contact_log_conf[:, 0] = self.root_contact_log_conf_head(root_summary).squeeze(-1)
        joint_contact_log_conf[:, 1:] = self.joint_contact_log_conf_head(joint_summary).squeeze(-1)
        joint_log_conf = root_log_conf.new_empty(root_xyz.shape[0], self.num_joints)
        joint_log_conf[:, 0] = root_log_conf
        joint_log_conf[:, 1:] = self.joint_log_conf_head(joint_summary).squeeze(-1)

        vertex_weights = self.vertex_joint_weights.to(device=final_prompt_set.device, dtype=final_prompt_set.dtype)
        vertex_keypoint_context = torch.einsum("vj,njc->nvc", vertex_weights, keypoint_prompt_tokens)
        vertex_queries = (
            self.vertex_queries.to(device=final_prompt_set.device, dtype=final_prompt_set.dtype).expand(final_prompt_set.shape[0], -1, -1)
            + self.vertex_query_proj(vertex_prompt).unsqueeze(1)
            + self.vertex_keypoint_proj(vertex_keypoint_context)
        )
        vertex_memory = torch.cat(
            [
                root_summary[:, None],
                root_prompt[:, None],
                vertex_prompt[:, None],
                root_feature,
                vertex_feature,
                keypoint_prompt_tokens,
            ],
            dim=1,
        )
        for block in self.vertex_blocks:
            vertex_queries = block(vertex_queries, vertex_memory)
        vertex_summary = self.vertex_output_norm(vertex_queries)
        vertex_offset = self.vertex_xyz_head(vertex_summary)
        vertex_xyz = root_xyz[:, None, :] + vertex_offset
        vertex_contact_logits = self.vertex_contact_head(vertex_summary).squeeze(-1)
        vertex_contact_log_distance = self.vertex_contact_distance_head(vertex_summary).squeeze(-1)
        vertex_contact_log_conf = self.vertex_contact_log_conf_head(vertex_summary).squeeze(-1)
        vertex_log_conf = self.vertex_log_conf_head(vertex_summary).squeeze(-1)
        return {
            "vertex_xyz": vertex_xyz,
            "vertex_offset": vertex_offset,
            "vertex_visibility_logits": self.vertex_visibility_head(vertex_summary).squeeze(-1),
            "vertex_contact_logits": vertex_contact_logits,
            "vertex_contact_log_distance": vertex_contact_log_distance,
            "vertex_contact_probability": torch.sigmoid(vertex_contact_logits),
            "vertex_contact_log_conf": vertex_contact_log_conf,
            "vertex_contact_conf": contact_confidence_from_log(vertex_contact_log_conf),
            "joint_xyz": joint_xyz,
            "joint_offset": joint_offset,
            "joint_visibility_logits": joint_visibility_logits,
            "joint_contact_logits": joint_contact_logits,
            "joint_contact_log_distance": joint_contact_log_distance,
            "joint_contact_probability": torch.sigmoid(joint_contact_logits),
            "joint_contact_log_conf": joint_contact_log_conf,
            "joint_contact_conf": contact_confidence_from_log(joint_contact_log_conf),
            "joint_log_conf": joint_log_conf,
            "joint_conf": torch.exp(joint_log_conf),
            "vertex_log_conf": vertex_log_conf,
            "vertex_conf": torch.exp(vertex_log_conf),
        }
