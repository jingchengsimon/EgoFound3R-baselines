from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from egohandmetric_prompt.heatmap_targets import KeypointHeatmapTargets, build_keypoint_heatmap_targets

from .hand_head_v3 import HandHeadV3
from .flow_guidance import FlowGuidanceBranch
from .hand_keypoint_localization_head import HandKeypointLocalizationHead
from .hand_prompt_adapter_v2 import HandPromptAdapterV2
from .metric_head import MetricHead
from .prompt_injected_aggregator_runner import PromptInjectedAggregatorRunner
from .temporal_hand_prompt_adapter import TemporalHandPromptAdapter


@dataclass(slots=True)
class MarkerBackboneFeatures:
    dino_feature_map: torch.Tensor
    dino_tokens: torch.Tensor
    mid_tokens: torch.Tensor
    camera_pose: torch.Tensor
    depth: torch.Tensor
    intrinsics: torch.Tensor
    camera_pose_encoding: torch.Tensor | None = None
    depth_conf: torch.Tensor | None = None


class SyntheticFrozenBackbone(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        feature_dim: int = 8,
        agg_dim: int = 10,
        patch_grid: tuple[int, int] = (4, 4),
        depth_size: tuple[int, int] = (16, 16),
    ) -> None:
        super().__init__()
        self.image_proj = nn.Conv2d(image_channels, feature_dim, kernel_size=3, padding=1)
        self.mid_proj = nn.Conv2d(feature_dim, agg_dim, kernel_size=1)
        self.depth_head = nn.Conv2d(feature_dim, 1, kernel_size=1)
        self.patch_grid = patch_grid
        self.depth_size = depth_size
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> MarkerBackboneFeatures:
        batch_size, num_frames, channels, height, width = images.shape
        flattened = images.reshape(batch_size * num_frames, channels, height, width)
        features = self.image_proj(flattened)
        features = torch.nn.functional.adaptive_avg_pool2d(features, self.patch_grid)
        mid = self.mid_proj(features)
        depth = torch.nn.functional.interpolate(features[:, :1], size=self.depth_size, mode="bilinear", align_corners=False)
        dino_feature_map = features.reshape(batch_size, num_frames, features.shape[1], *self.patch_grid)
        dino_tokens = dino_feature_map.flatten(-2).transpose(-1, -2)
        mid_tokens = mid.reshape(batch_size, num_frames, mid.shape[1], *self.patch_grid).flatten(-2).transpose(-1, -2)
        depth = depth.reshape(batch_size, num_frames, *self.depth_size)
        camera_pose = torch.zeros(batch_size, num_frames, 9, device=images.device, dtype=images.dtype)
        intrinsics = torch.eye(3, device=images.device, dtype=images.dtype).view(1, 1, 3, 3).expand(batch_size, num_frames, -1, -1).clone()
        intrinsics[..., 0, 2] = (self.depth_size[1] - 1) / 2.0
        intrinsics[..., 1, 2] = (self.depth_size[0] - 1) / 2.0
        return MarkerBackboneFeatures(
            dino_feature_map=dino_feature_map,
            dino_tokens=dino_tokens,
            mid_tokens=mid_tokens,
            camera_pose=camera_pose,
            depth=depth,
            intrinsics=intrinsics,
            camera_pose_encoding=camera_pose,
            depth_conf=torch.ones_like(depth),
        )


class MarkerModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        keypoint_head: HandKeypointLocalizationHead,
        hand_adapter: HandPromptAdapterV2,
        temporal_adapter: TemporalHandPromptAdapter,
        hand_head: HandHeadV3,
        metric_head: MetricHead | None = None,
        flow_guidance: FlowGuidanceBranch | None = None,
        teacher_feature_dim: int | None = None,
        keypoint_peak_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head
        self.hand_adapter = hand_adapter
        self.temporal_adapter = temporal_adapter
        self.hand_head = hand_head
        self.flow_guidance = flow_guidance
        self.keypoint_peak_threshold = float(keypoint_peak_threshold)
        self.teacher_feature_dim = teacher_feature_dim
        self.hand_feature_dim = getattr(hand_adapter, "hand_feature_dim", hand_head.prompt_dim)
        self.metric_head = (
            metric_head
            if metric_head is not None
            else MetricHead(prompt_dim=hand_head.prompt_dim, hand_feature_dim=self.hand_feature_dim)
        )
        self.teacher_projector = (
            nn.Linear(self.hand_feature_dim, teacher_feature_dim)
            if teacher_feature_dim is not None
            else None
        )
        self.inject_layer_idx = getattr(backbone, "mid_layer_index", 11)
        backbone_token_dim = keypoint_head.in_channels
        self.prompt_to_backbone = nn.Linear(hand_head.prompt_dim, backbone_token_dim)
        self.backbone_to_prompt = nn.Linear(backbone_token_dim, hand_head.prompt_dim)
        self.empty_hand_slot_token = nn.Parameter(torch.zeros(1, 1, 1, 3, hand_head.prompt_dim))

    def _guided_features(
        self,
        *,
        dino_feature_map: torch.Tensor,
        dino_tokens: torch.Tensor,
        mid_tokens: torch.Tensor,
        prompt_target_batch: dict[str, Any] | None,
        enable_flow_guidance: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.flow_guidance is None or not enable_flow_guidance:
            return dino_feature_map, dino_tokens, mid_tokens, {}
        pair_mask = None if prompt_target_batch is None else prompt_target_batch.get("flow_pair_mask")
        guided = self.flow_guidance(
            dino_feature_map=dino_feature_map,
            dino_tokens=dino_tokens,
            mid_tokens=mid_tokens,
            flow_pair_mask=pair_mask,
        )
        return (
            guided.dino_feature_map,
            guided.dino_tokens,
            guided.mid_tokens,
            {
                "flow_pred": guided.flow_pred,
                "flow_pair_mask": guided.flow_pair_mask,
                "flow_gate": guided.gate,
            },
        )

    def _supports_prompt_injection(self) -> bool:
        return all(
            hasattr(self.backbone, name)
            for name in ("run_prefix", "run_suffix", "decode_camera", "decode_depth_with_conf", "build_intrinsics")
        )

    def _route_prompt_slots(
        self,
        hand_prompt_set: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> torch.Tensor:
        empty_prompt = self.empty_hand_slot_token.to(device=hand_prompt_set.device, dtype=hand_prompt_set.dtype).expand_as(
            hand_prompt_set
        )
        presence_mask = presence_mask.to(device=hand_prompt_set.device, dtype=torch.bool)
        return torch.where(presence_mask.unsqueeze(-1).unsqueeze(-1), hand_prompt_set, empty_prompt)

    def _decode_fixed_hand_outputs(
        self,
        *,
        final_prompt_set: torch.Tensor,
        hand_feature_set: torch.Tensor,
        keypoint_prompt_tokens: torch.Tensor,
        keypoint_uv: torch.Tensor,
        keypoint_conf: torch.Tensor,
        keypoint_mask: torch.Tensor,
        final_metric_prompt: torch.Tensor,
        metric_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_frames, side_count = final_prompt_set.shape[:3]
        flat_final_tokens = final_prompt_set.reshape(
            batch_size * num_frames * side_count,
            final_prompt_set.shape[-2],
            final_prompt_set.shape[-1],
        )
        flat_hand_features = hand_feature_set.reshape(
            batch_size * num_frames * side_count,
            hand_feature_set.shape[-2],
            hand_feature_set.shape[-1],
        )
        flat_keypoint_tokens = keypoint_prompt_tokens.reshape(
            batch_size * num_frames * side_count,
            keypoint_prompt_tokens.shape[-2],
            keypoint_prompt_tokens.shape[-1],
        )
        flat_keypoint_uv = keypoint_uv.reshape(
            batch_size * num_frames * side_count,
            keypoint_uv.shape[-2],
            keypoint_uv.shape[-1],
        )
        flat_keypoint_conf = keypoint_conf.reshape(batch_size * num_frames * side_count, keypoint_conf.shape[-1])
        flat_keypoint_mask = keypoint_mask.reshape(batch_size * num_frames * side_count, keypoint_mask.shape[-1])
        flat_metric_prompt = final_metric_prompt[:, :, None].expand(
            batch_size,
            num_frames,
            side_count,
            final_metric_prompt.shape[-2],
            final_metric_prompt.shape[-1],
        ).reshape(
            batch_size * num_frames * side_count,
            final_metric_prompt.shape[-2],
            final_metric_prompt.shape[-1],
        )
        flat_metric_feature = metric_feature[:, :, None].expand(
            batch_size,
            num_frames,
            side_count,
            metric_feature.shape[-1],
        ).reshape(batch_size * num_frames * side_count, metric_feature.shape[-1])
        hand_outputs = self.hand_head(
            flat_final_tokens,
            hand_features=flat_hand_features,
            keypoint_prompt_tokens=flat_keypoint_tokens,
            keypoint_uv=flat_keypoint_uv,
            keypoint_conf=flat_keypoint_conf,
            keypoint_mask=flat_keypoint_mask,
            metric_prompt=flat_metric_prompt,
            metric_feature=flat_metric_feature,
        )
        return {
            key: value.reshape(batch_size, num_frames, side_count, *value.shape[1:])
            for key, value in hand_outputs.items()
        }

    def _decode_metric_outputs(
        self,
        *,
        final_metric_prompt: torch.Tensor,
        metric_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.metric_head(
            final_metric_prompt,
            metric_feature=metric_feature,
        )

    def _adapter_presence_targets(
        self,
        prompt_target_batch: dict[str, Any] | None,
        keypoint_inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if not self.hand_adapter.enable_presence_head:
            return keypoint_inputs["keypoint_mask"].any(dim=-1)
        if prompt_target_batch is None:
            return None
        return prompt_target_batch.get("presence_targets")

    def _fixed_teacher_features(self, hand_feature_set: torch.Tensor) -> torch.Tensor | None:
        if self.teacher_projector is None:
            return None
        return self.teacher_projector(hand_feature_set.mean(dim=3))

    def _keypoint_targets(
        self,
        prompt_target_batch: dict[str, Any] | None,
        *,
        heatmap_logits: torch.Tensor,
    ) -> KeypointHeatmapTargets | None:
        if prompt_target_batch is None:
            return None
        return build_keypoint_heatmap_targets(
            prompt_target_batch,
            heatmap_height=heatmap_logits.shape[-2],
            heatmap_width=heatmap_logits.shape[-1],
            device=heatmap_logits.device,
            dtype=heatmap_logits.dtype,
        )

    def _keypoint_inputs(
        self,
        *,
        dino_feature_map: torch.Tensor,
        heatmap_logits: torch.Tensor,
        keypoint_offset: torch.Tensor,
        prompt_target_batch: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        if keypoint_offset.shape != (*heatmap_logits.shape[:4], 2, *heatmap_logits.shape[-2:]):
            raise ValueError("keypoint_offset must have shape [B, T, 2, joints, 2, H, W]")
        targets = self._keypoint_targets(prompt_target_batch, heatmap_logits=heatmap_logits)
        if targets is None:
            conditioning_heatmap = heatmap_logits.sigmoid()
            flat_heatmap = conditioning_heatmap.reshape(*conditioning_heatmap.shape[:4], -1)
            keypoint_conf = flat_heatmap.max(dim=-1).values
            keypoint_mask = keypoint_conf > self.keypoint_peak_threshold
            cell_indices = flat_heatmap.argmax(dim=-1)
            target_offsets = torch.zeros(*conditioning_heatmap.shape[:4], 2, device=heatmap_logits.device, dtype=heatmap_logits.dtype)
            source_is_gt = heatmap_logits.new_zeros(())
        else:
            conditioning_heatmap = targets.heatmaps
            keypoint_mask = targets.mask
            cell_indices = targets.cell_indices
            target_offsets = targets.offsets
            source_is_gt = heatmap_logits.new_ones(())
        batch_size, num_frames, side_count, joint_count, height, width = conditioning_heatmap.shape
        flat_heatmap = conditioning_heatmap.reshape(batch_size, num_frames, side_count, joint_count, height * width)
        safe_cell_indices = cell_indices.clamp_min(0)
        keypoint_conf = flat_heatmap.gather(-1, safe_cell_indices.unsqueeze(-1)).squeeze(-1)
        cell_y = torch.div(safe_cell_indices, width, rounding_mode="floor")
        cell_x = safe_cell_indices.remainder(width)
        flat_offset = keypoint_offset.reshape(batch_size, num_frames, side_count, joint_count, 2, height * width)
        offset_index = safe_cell_indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size,
            num_frames,
            side_count,
            joint_count,
            2,
            1,
        )
        pred_offsets = 0.5 * torch.tanh(flat_offset.gather(-1, offset_index).squeeze(-1))
        selected_offsets = pred_offsets if targets is None else target_offsets.to(device=heatmap_logits.device, dtype=heatmap_logits.dtype)
        uv = torch.stack(
            [
                (cell_x.to(dtype=heatmap_logits.dtype) + selected_offsets[..., 0]) / float(max(width - 1, 1)),
                (cell_y.to(dtype=heatmap_logits.dtype) + selected_offsets[..., 1]) / float(max(height - 1, 1)),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        flat_features = dino_feature_map.flatten(-2).transpose(-1, -2)
        feature_dim = flat_features.shape[-1]
        feature_index = safe_cell_indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size,
            num_frames,
            side_count,
            joint_count,
            1,
            feature_dim,
        )
        keypoint_features = (
            flat_features[:, :, None, None, :, :]
            .expand(batch_size, num_frames, side_count, joint_count, height * width, feature_dim)
            .gather(4, feature_index)
            .squeeze(4)
        )
        keypoint_conf = torch.where(keypoint_mask, keypoint_conf, torch.zeros_like(keypoint_conf))
        return {
            "keypoint_features": keypoint_features,
            "keypoint_uv": uv,
            "keypoint_conf": keypoint_conf,
            "keypoint_mask": keypoint_mask,
            "keypoint_offset": keypoint_offset,
            "keypoint_offset_targets": target_offsets,
            "keypoint_offset_mask": keypoint_mask,
            "keypoint_cell_indices": cell_indices,
            "keypoint_heatmap_targets": (
                torch.zeros_like(conditioning_heatmap) if targets is None else targets.heatmaps
            ),
            "keypoint_heatmap_mask": keypoint_mask,
            "keypoint_conditioning_is_gt": source_is_gt,
        }

    def forward(
        self,
        images: torch.Tensor,
        inference_egocentric: bool = True,
        dense_slots_per_side: int | None = None,
        prompt_target_batch: dict[str, Any] | None = None,
        enable_flow_guidance: bool = True,
    ) -> dict[str, torch.Tensor]:
        _ = inference_egocentric, dense_slots_per_side
        if self._supports_prompt_injection():
            return self._forward_prompt_injected(
                images,
                prompt_target_batch=prompt_target_batch,
                enable_flow_guidance=enable_flow_guidance,
            )
        return self._forward_basic(
            images,
            prompt_target_batch=prompt_target_batch,
            enable_flow_guidance=enable_flow_guidance,
        )

    def _forward_basic(
        self,
        images: torch.Tensor,
        *,
        prompt_target_batch: dict[str, Any] | None,
        enable_flow_guidance: bool,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        dino_feature_map, dino_tokens, mid_tokens, flow_outputs = self._guided_features(
            dino_feature_map=features.dino_feature_map,
            dino_tokens=features.dino_tokens,
            mid_tokens=features.mid_tokens,
            prompt_target_batch=prompt_target_batch,
            enable_flow_guidance=enable_flow_guidance,
        )
        keypoint_outputs = self.keypoint_head(dino_feature_map)
        heatmap_logits = keypoint_outputs["keypoint_heatmap_logits"]
        keypoint_offset = keypoint_outputs["keypoint_offset"]
        keypoint_inputs = self._keypoint_inputs(
            dino_feature_map=dino_feature_map,
            heatmap_logits=heatmap_logits,
            keypoint_offset=keypoint_offset,
            prompt_target_batch=prompt_target_batch,
        )
        adapter_outputs = self.hand_adapter(
            dino_tokens=dino_tokens,
            agg_tokens=mid_tokens,
            keypoint_features=keypoint_inputs["keypoint_features"],
            keypoint_uv=keypoint_inputs["keypoint_uv"],
            keypoint_conf=keypoint_inputs["keypoint_conf"],
            keypoint_mask=keypoint_inputs["keypoint_mask"],
            presence_targets=self._adapter_presence_targets(prompt_target_batch, keypoint_inputs),
        )
        hand_feature_set = adapter_outputs["hand_feature_set"]
        hand_prompt_set = adapter_outputs["hand_prompt_set"]
        metric_feature = adapter_outputs["metric_feature"]
        metric_prompt = adapter_outputs["metric_prompt"]
        refined_prompt_set = self.temporal_adapter(hand_prompt_set)
        prompt_slots = self._route_prompt_slots(refined_prompt_set, adapter_outputs["presence_mask"])
        hand_outputs = self._decode_fixed_hand_outputs(
            final_prompt_set=prompt_slots,
            hand_feature_set=hand_feature_set,
            keypoint_prompt_tokens=adapter_outputs["keypoint_prompt_tokens"],
            keypoint_uv=keypoint_inputs["keypoint_uv"],
            keypoint_conf=keypoint_inputs["keypoint_conf"],
            keypoint_mask=keypoint_inputs["keypoint_mask"],
            final_metric_prompt=metric_prompt,
            metric_feature=metric_feature,
        )
        metric_outputs = self._decode_metric_outputs(
            final_metric_prompt=metric_prompt,
            metric_feature=metric_feature,
        )
        outputs = self._format_outputs(
            camera_pose=features.camera_pose,
            camera_pose_encoding=features.camera_pose_encoding,
            depth=features.depth,
            depth_conf=features.depth_conf,
            intrinsics=features.intrinsics,
            heatmap_logits=heatmap_logits,
            keypoint_inputs=keypoint_inputs,
            adapter_outputs=adapter_outputs,
            hand_outputs=hand_outputs,
            metric_outputs=metric_outputs,
            prompt_set=hand_prompt_set,
            prompt_slots=prompt_slots,
            final_prompt_set=prompt_slots,
            final_metric_prompt=metric_prompt,
        )
        outputs.update(flow_outputs)
        return outputs

    def _format_outputs(
        self,
        *,
        camera_pose: torch.Tensor,
        camera_pose_encoding: torch.Tensor | None,
        depth: torch.Tensor,
        depth_conf: torch.Tensor | None,
        intrinsics: torch.Tensor,
        heatmap_logits: torch.Tensor,
        keypoint_inputs: dict[str, torch.Tensor],
        adapter_outputs: dict[str, torch.Tensor],
        hand_outputs: dict[str, torch.Tensor],
        metric_outputs: dict[str, torch.Tensor],
        prompt_set: torch.Tensor,
        prompt_slots: torch.Tensor,
        final_prompt_set: torch.Tensor,
        final_metric_prompt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        metric_feature = adapter_outputs["metric_feature"]
        hand_feature_set = adapter_outputs["hand_feature_set"]
        metric_prompt = adapter_outputs["metric_prompt"]
        presence_logits = adapter_outputs["presence_logits"]
        presence_mask = adapter_outputs["presence_mask"]
        hand_feature_teacher_features = self._fixed_teacher_features(hand_feature_set)
        flat_prompt_set = prompt_set.reshape(-1, prompt_set.shape[-1])
        flat_final_prompts = final_prompt_set.reshape(-1, final_prompt_set.shape[-1])
        flat_vertex_xyz = hand_outputs["vertex_xyz"].reshape(-1, self.hand_head.num_vertices, 3)
        flat_vertex_offset = hand_outputs["vertex_offset"].reshape(-1, self.hand_head.num_vertices, 3)
        flat_vertex_visibility_logits = hand_outputs["vertex_visibility_logits"].reshape(-1, self.hand_head.num_vertices)
        flat_vertex_contact_logits = hand_outputs["vertex_contact_logits"].reshape(-1, self.hand_head.num_vertices)
        flat_vertex_contact_probability = hand_outputs["vertex_contact_probability"].reshape(-1, self.hand_head.num_vertices)
        flat_vertex_contact_log_conf = hand_outputs["vertex_contact_log_conf"].reshape(-1, self.hand_head.num_vertices)
        flat_vertex_contact_conf = hand_outputs["vertex_contact_conf"].reshape(-1, self.hand_head.num_vertices)
        flat_joint_xyz = hand_outputs["joint_xyz"].reshape(-1, self.hand_head.num_joints, 3)
        flat_joint_offset = hand_outputs["joint_offset"].reshape(-1, self.hand_head.num_joints, 3)
        flat_joint_visibility_logits = hand_outputs["joint_visibility_logits"].reshape(-1, self.hand_head.num_joints)
        flat_joint_contact_logits = hand_outputs["joint_contact_logits"].reshape(-1, self.hand_head.num_joints)
        flat_joint_contact_probability = hand_outputs["joint_contact_probability"].reshape(-1, self.hand_head.num_joints)
        flat_joint_contact_log_conf = hand_outputs["joint_contact_log_conf"].reshape(-1, self.hand_head.num_joints)
        flat_joint_contact_conf = hand_outputs["joint_contact_conf"].reshape(-1, self.hand_head.num_joints)
        flat_joint_log_conf = hand_outputs["joint_log_conf"].reshape(-1, self.hand_head.num_joints)
        flat_joint_conf = hand_outputs["joint_conf"].reshape(-1, self.hand_head.num_joints)
        flat_vertex_log_conf = hand_outputs["vertex_log_conf"].reshape(-1, self.hand_head.num_vertices)
        flat_vertex_conf = hand_outputs["vertex_conf"].reshape(-1, self.hand_head.num_vertices)
        dense_hand_root_xyz = hand_outputs["joint_xyz"][..., 0, :]
        dense_hand_root_log_conf = hand_outputs["joint_log_conf"][..., :1]
        dense_hand_root_conf = hand_outputs["joint_conf"][..., :1]
        flat_hand_root_xyz = dense_hand_root_xyz.reshape(-1, 3)
        flat_hand_root_log_conf = dense_hand_root_log_conf.reshape(-1, 1)
        flat_hand_root_conf = dense_hand_root_conf.reshape(-1, 1)
        return {
            "camera_pose": camera_pose,
            "camera_pose_encoding": camera_pose_encoding,
            "depth": depth,
            "depth_conf": depth_conf,
            "intrinsics": intrinsics,
            "heatmap_logits": heatmap_logits,
            "keypoint_heatmap_logits": heatmap_logits,
            "keypoint_offset": keypoint_inputs["keypoint_offset"],
            "keypoint_heatmap_targets": keypoint_inputs["keypoint_heatmap_targets"],
            "keypoint_heatmap_mask": keypoint_inputs["keypoint_heatmap_mask"],
            "keypoint_offset_targets": keypoint_inputs["keypoint_offset_targets"],
            "keypoint_offset_mask": keypoint_inputs["keypoint_offset_mask"],
            "keypoint_cell_indices": keypoint_inputs["keypoint_cell_indices"],
            "keypoint_uv": keypoint_inputs["keypoint_uv"],
            "keypoint_conf": keypoint_inputs["keypoint_conf"],
            "keypoint_mask": keypoint_inputs["keypoint_mask"],
            "keypoint_conditioning_is_gt": keypoint_inputs["keypoint_conditioning_is_gt"],
            "keypoint_prompt_tokens": adapter_outputs["keypoint_prompt_tokens"],
            "groups": [],
            "hand_feature_set": hand_feature_set,
            "hand_prompt_set": prompt_set,
            "metric_feature": metric_feature,
            "scene_metric_feature": adapter_outputs.get("scene_metric_feature", metric_feature),
            "hand_metric_feature": adapter_outputs.get("hand_metric_feature", metric_feature),
            "hand_observed_ratio": adapter_outputs.get(
                "hand_observed_ratio",
                torch.ones(metric_feature.shape[0], device=metric_feature.device, dtype=metric_feature.dtype),
            ),
            "metric_prompt": metric_prompt,
            "presence_logits": presence_logits,
            "presence_mask": presence_mask,
            "presence_loss_enabled": heatmap_logits.new_tensor(
                1.0 if self.hand_adapter.enable_presence_head else 0.0
            ),
            "prompts": flat_prompt_set,
            "prompt_slots": prompt_slots,
            "final_prompt_slots": final_prompt_set,
            "final_hand_prompts": final_prompt_set,
            "final_metric_prompt": final_metric_prompt,
            "log_metric_value": metric_outputs["log_metric_value"],
            "metric_value": metric_outputs["metric_value"],
            "final_prompt_teacher_features": None,
            "final_prompt_slot_teacher_features": None,
            "hand_feature_teacher_features": hand_feature_teacher_features,
            "prompt_teacher_features": (
                None if hand_feature_teacher_features is None else hand_feature_teacher_features.reshape(-1, hand_feature_teacher_features.shape[-1])
            ),
            "final_prompt_flat": flat_final_prompts,
            "prompt_vertex_xyz": flat_vertex_xyz,
            "prompt_vertex_offset": flat_vertex_offset,
            "prompt_vertex_visibility_logits": flat_vertex_visibility_logits,
            "prompt_vertex_contact_logits": flat_vertex_contact_logits,
            "prompt_vertex_contact_probability": flat_vertex_contact_probability,
            "prompt_vertex_contact_log_conf": flat_vertex_contact_log_conf,
            "prompt_vertex_contact_conf": flat_vertex_contact_conf,
            "prompt_hand_root_xyz": flat_hand_root_xyz,
            "prompt_joint_xyz": flat_joint_xyz,
            "prompt_joint_offset": flat_joint_offset,
            "prompt_joint_visibility_logits": flat_joint_visibility_logits,
            "prompt_joint_contact_logits": flat_joint_contact_logits,
            "prompt_joint_contact_probability": flat_joint_contact_probability,
            "prompt_joint_contact_log_conf": flat_joint_contact_log_conf,
            "prompt_joint_contact_conf": flat_joint_contact_conf,
            "prompt_hand_root_log_conf": flat_hand_root_log_conf,
            "prompt_hand_root_conf": flat_hand_root_conf,
            "prompt_joint_log_conf": flat_joint_log_conf,
            "prompt_joint_conf": flat_joint_conf,
            "prompt_vertex_log_conf": flat_vertex_log_conf,
            "prompt_vertex_conf": flat_vertex_conf,
            "dense_vertex_xyz": hand_outputs["vertex_xyz"],
            "dense_vertex_offset": hand_outputs["vertex_offset"],
            "dense_vertex_visibility_logits": hand_outputs["vertex_visibility_logits"],
            "dense_vertex_contact_logits": hand_outputs["vertex_contact_logits"],
            "dense_vertex_contact_log_distance": hand_outputs["vertex_contact_log_distance"],
            "dense_vertex_contact_probability": hand_outputs["vertex_contact_probability"],
            "dense_vertex_contact_log_conf": hand_outputs["vertex_contact_log_conf"],
            "dense_vertex_contact_conf": hand_outputs["vertex_contact_conf"],
            "dense_hand_root_xyz": dense_hand_root_xyz,
            "dense_joint_xyz": hand_outputs["joint_xyz"],
            "dense_joint_offset": hand_outputs["joint_offset"],
            "dense_joint_visibility_logits": hand_outputs["joint_visibility_logits"],
            "dense_joint_contact_logits": hand_outputs["joint_contact_logits"],
            "dense_joint_contact_log_distance": hand_outputs["joint_contact_log_distance"],
            "dense_joint_contact_probability": hand_outputs["joint_contact_probability"],
            "dense_joint_contact_log_conf": hand_outputs["joint_contact_log_conf"],
            "dense_joint_contact_conf": hand_outputs["joint_contact_conf"],
            "dense_hand_root_log_conf": dense_hand_root_log_conf,
            "dense_hand_root_conf": dense_hand_root_conf,
            "dense_joint_log_conf": hand_outputs["joint_log_conf"],
            "dense_joint_conf": hand_outputs["joint_conf"],
            "dense_vertex_log_conf": hand_outputs["vertex_log_conf"],
            "dense_vertex_conf": hand_outputs["vertex_conf"],
        }

    def _forward_prompt_injected(
        self,
        images: torch.Tensor,
        *,
        prompt_target_batch: dict[str, Any] | None,
        enable_flow_guidance: bool,
    ) -> dict[str, torch.Tensor]:
        prefix = self.backbone.run_prefix(images, inject_layer_idx=self.inject_layer_idx)
        batch_size, num_frames = images.shape[:2]
        dino_feature_map, dino_tokens, mid_tokens, flow_outputs = self._guided_features(
            dino_feature_map=prefix.dino_feature_map,
            dino_tokens=prefix.dino_tokens,
            mid_tokens=prefix.mid_tokens,
            prompt_target_batch=prompt_target_batch,
            enable_flow_guidance=enable_flow_guidance,
        )
        keypoint_outputs = self.keypoint_head(dino_feature_map)
        heatmap_logits = keypoint_outputs["keypoint_heatmap_logits"]
        keypoint_offset = keypoint_outputs["keypoint_offset"]
        keypoint_inputs = self._keypoint_inputs(
            dino_feature_map=dino_feature_map,
            heatmap_logits=heatmap_logits,
            keypoint_offset=keypoint_offset,
            prompt_target_batch=prompt_target_batch,
        )
        adapter_outputs = self.hand_adapter(
            dino_tokens=dino_tokens,
            agg_tokens=mid_tokens,
            keypoint_features=keypoint_inputs["keypoint_features"],
            keypoint_uv=keypoint_inputs["keypoint_uv"],
            keypoint_conf=keypoint_inputs["keypoint_conf"],
            keypoint_mask=keypoint_inputs["keypoint_mask"],
            presence_targets=self._adapter_presence_targets(prompt_target_batch, keypoint_inputs),
        )
        hand_feature_set = adapter_outputs["hand_feature_set"]
        hand_prompt_set = adapter_outputs["hand_prompt_set"]
        metric_feature = adapter_outputs["metric_feature"]
        metric_prompt = adapter_outputs["metric_prompt"]
        refined_prompt_set = self.temporal_adapter(hand_prompt_set)
        prompt_slots = self._route_prompt_slots(refined_prompt_set, adapter_outputs["presence_mask"])
        flat_prompt_slots = prompt_slots.reshape(batch_size, num_frames, 2 * self.hand_adapter.prompt_type_count, -1)
        prompt_slots_with_metric = torch.cat([flat_prompt_slots, metric_prompt], dim=2)
        packed_backbone_prompts = self.prompt_to_backbone(prompt_slots_with_metric)
        prefix_layout = getattr(prefix, "layout", None)
        prompt_run = PromptInjectedAggregatorRunner.append_prompts(
            prefix.public_tokens,
            packed_backbone_prompts,
            patch_token_start=prefix.patch_token_start,
            hand_prompt_count=flat_prompt_slots.shape[2],
            metric_prompt_count=metric_prompt.shape[2],
            patch_grid_hw=getattr(prefix_layout, "patch_grid_hw", None),
        )
        suffix = self.backbone.run_suffix(
            prompt_run.internal_tokens,
            layout=prompt_run.layout,
            start_layer_idx=self.inject_layer_idx + 1,
            prefix_aggregated_tokens_list=prefix.aggregated_tokens_list,
        )
        final_prompt_slots = self.backbone_to_prompt(suffix.final_hand_prompts).reshape(
            batch_size,
            num_frames,
            2,
            self.hand_adapter.prompt_type_count,
            -1,
        )
        final_metric_prompt = self.backbone_to_prompt(suffix.final_metric_prompts)
        hand_outputs = self._decode_fixed_hand_outputs(
            final_prompt_set=final_prompt_slots,
            hand_feature_set=hand_feature_set,
            keypoint_prompt_tokens=adapter_outputs["keypoint_prompt_tokens"],
            keypoint_uv=keypoint_inputs["keypoint_uv"],
            keypoint_conf=keypoint_inputs["keypoint_conf"],
            keypoint_mask=keypoint_inputs["keypoint_mask"],
            final_metric_prompt=final_metric_prompt,
            metric_feature=metric_feature,
        )
        metric_outputs = self._decode_metric_outputs(
            final_metric_prompt=final_metric_prompt,
            metric_feature=metric_feature,
        )
        camera_pose_encoding = None
        if hasattr(self.backbone, "decode_camera_pose_encoding"):
            camera_pose_encoding = self.backbone.decode_camera_pose_encoding(
                suffix.aggregated_tokens_list,
                patch_token_start=prefix.patch_token_start,
            )
            camera_pose = self.backbone.decode_camera_from_pose_encoding(
                camera_pose_encoding,
                image_size_hw=images.shape[-2:],
            )
            intrinsics = self.backbone.build_intrinsics(
                batch_size=batch_size,
                num_frames=num_frames,
                image_size_hw=images.shape[-2:],
                device=images.device,
                dtype=images.dtype,
                pose_encoding=camera_pose_encoding,
            )
        else:
            camera_pose = self.backbone.decode_camera(
                suffix.aggregated_tokens_list,
                patch_token_start=prefix.patch_token_start,
                image_size_hw=images.shape[-2:],
            )
            intrinsics = self.backbone.build_intrinsics(
                batch_size=batch_size,
                num_frames=num_frames,
                image_size_hw=images.shape[-2:],
                device=images.device,
                dtype=images.dtype,
                aggregated_tokens_list=suffix.aggregated_tokens_list,
                patch_token_start=prefix.patch_token_start,
            )
        depth, depth_conf = self.backbone.decode_depth_with_conf(
            suffix.aggregated_tokens_list,
            images=images,
            patch_token_start=prefix.patch_token_start,
        )
        depth = depth.to(device=images.device, dtype=images.dtype)
        depth_conf = depth_conf.to(device=images.device, dtype=images.dtype)
        outputs = self._format_outputs(
            camera_pose=camera_pose.to(device=images.device, dtype=images.dtype),
            camera_pose_encoding=(
                None if camera_pose_encoding is None else camera_pose_encoding.to(device=images.device, dtype=images.dtype)
            ),
            depth=depth,
            depth_conf=depth_conf,
            intrinsics=intrinsics,
            heatmap_logits=heatmap_logits,
            keypoint_inputs=keypoint_inputs,
            adapter_outputs=adapter_outputs,
            hand_outputs=hand_outputs,
            metric_outputs=metric_outputs,
            prompt_set=hand_prompt_set,
            prompt_slots=prompt_slots,
            final_prompt_set=final_prompt_slots,
            final_metric_prompt=final_metric_prompt,
        )
        outputs.update(flow_outputs)
        return outputs
