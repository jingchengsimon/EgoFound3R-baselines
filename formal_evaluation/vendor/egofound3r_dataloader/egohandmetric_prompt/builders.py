from __future__ import annotations

import torch.nn as nn

from egohandmetric_prompt.configs import (
    FlowModelConfig,
    MarkerModelConfig,
    ProjectConfig,
    SmokeBackboneConfig,
)
from egohandmetric_prompt.data.marker_vertex_joint_map import marker_vertex_joint_weights_for_count
from egohandmetric_prompt.models import (
    FlowGuidanceBranch,
    FlowMatchingModel,
    HandHeadV3,
    HandKeypointLocalizationHead,
    HandPromptAdapterV2,
    MarkerModel,
    MetricHead,
    SyntheticFrozenBackbone,
    TemporalHandPromptAdapter,
    VggtOmegaFrozenWrapper,
)


def _build_flow_guidance(model_config: MarkerModelConfig) -> FlowGuidanceBranch | None:
    if not model_config.enable_flow_guidance:
        return None
    return FlowGuidanceBranch(
        dino_dim=model_config.dino_dim,
        aggregator_dim=model_config.aggregator_dim,
        motion_dim=model_config.flow_motion_dim,
        num_heads=model_config.flow_num_heads,
        num_motion_blocks=model_config.flow_num_motion_blocks,
        mlp_ratio=model_config.flow_mlp_ratio,
        patch_size=model_config.flow_patch_size,
        gate_init_logit=model_config.flow_gate_init_logit,
    )


def build_marker_model_from_config(
    model_config: MarkerModelConfig,
    backbone_config: SmokeBackboneConfig,
) -> MarkerModel:
    backbone = SyntheticFrozenBackbone(
        feature_dim=backbone_config.feature_dim,
        agg_dim=backbone_config.aggregator_dim,
        patch_grid=(backbone_config.patch_grid_height, backbone_config.patch_grid_width),
        depth_size=(backbone_config.depth_height, backbone_config.depth_width),
    )
    keypoint_head = HandKeypointLocalizationHead(
        in_channels=model_config.dino_dim,
        hidden_channels=model_config.detection_hidden_dim,
    )
    hand_adapter = HandPromptAdapterV2(
        dino_dim=model_config.dino_dim,
        agg_dim=model_config.aggregator_dim,
        prompt_dim=model_config.prompt_dim,
        hand_feature_dim=model_config.hand_feature_dim,
        num_layers=model_config.adapter_num_layers,
        num_heads=model_config.adapter_num_heads,
        mlp_ratio=model_config.adapter_mlp_ratio,
        enable_presence_head=model_config.enable_presence_head,
    )
    temporal_adapter = TemporalHandPromptAdapter(
        prompt_dim=model_config.prompt_dim,
        num_layers=model_config.temporal_prompt_layers,
        num_heads=model_config.temporal_prompt_heads,
        mlp_ratio=model_config.temporal_prompt_mlp_ratio,
    )
    hand_head = HandHeadV3(
        prompt_dim=model_config.prompt_dim,
        num_markers=model_config.num_markers,
        num_queries=model_config.marker_query_count,
        hand_feature_dim=hand_adapter.hand_feature_dim,
        root_num_layers=model_config.root_head_num_layers,
        joint_num_layers=model_config.joint_head_num_layers,
        vertex_num_layers=model_config.vertex_head_num_layers,
        num_heads=model_config.hand_head_num_heads,
        mlp_ratio=model_config.hand_head_mlp_ratio,
        vertex_joint_weights=marker_vertex_joint_weights_for_count(model_config.num_markers),
        random_init_contact_confidence_heads=model_config.random_init_contact_confidence_heads,
    )
    metric_head = MetricHead(
        prompt_dim=model_config.prompt_dim,
        hand_feature_dim=hand_adapter.hand_feature_dim,
    )
    return MarkerModel(
        backbone,
        keypoint_head,
        hand_adapter,
        temporal_adapter,
        hand_head,
        metric_head=metric_head,
        flow_guidance=_build_flow_guidance(model_config),
        keypoint_peak_threshold=model_config.peak_threshold,
    )


def build_flow_model_from_config(model_config: FlowModelConfig) -> FlowMatchingModel:
    return FlowMatchingModel(
        num_markers=model_config.num_markers,
        state_dim=model_config.state_dim,
        camera_pose_dim=model_config.camera_pose_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
    )


def build_smoke_marker_model(project_config: ProjectConfig) -> MarkerModel:
    return build_marker_model_from_config(project_config.smoke_marker_model, project_config.smoke_backbone)


def build_runtime_marker_model(project_config: ProjectConfig) -> MarkerModel:
    runtime_config = project_config.marker_runtime
    teacher_feature_dim = runtime_config.wilor_teacher_feature_dim if runtime_config.enable_wilor_teacher else None
    if runtime_config.backend == "smoke":
        model = build_smoke_marker_model(project_config)
        model.teacher_feature_dim = teacher_feature_dim
        if teacher_feature_dim is not None and model.teacher_projector is None:
            model.teacher_projector = nn.Linear(model.hand_feature_dim, teacher_feature_dim)
        return model
    if runtime_config.backend != "vggt_omega":
        raise ValueError(f"未知 marker runtime backend: {runtime_config.backend}")
    if not runtime_config.vggt_checkpoint_path:
        raise ValueError("marker_runtime.vggt_checkpoint_path 不能为空")
    backbone = VggtOmegaFrozenWrapper.from_vendor(
        checkpoint_path=runtime_config.vggt_checkpoint_path,
        image_height=runtime_config.image_height,
        image_width=runtime_config.image_width,
        mid_layer_index=project_config.marker_model.prompt_injection_layer - 1,
        suffix_activation_checkpointing=project_config.marker_train.vggt_suffix_activation_checkpointing,
    )
    keypoint_head = HandKeypointLocalizationHead(
        in_channels=project_config.marker_model.dino_dim,
        hidden_channels=project_config.marker_model.detection_hidden_dim,
    )
    hand_adapter = HandPromptAdapterV2(
        dino_dim=project_config.marker_model.dino_dim,
        agg_dim=project_config.marker_model.aggregator_dim,
        prompt_dim=project_config.marker_model.prompt_dim,
        hand_feature_dim=project_config.marker_model.hand_feature_dim,
        num_layers=project_config.marker_model.adapter_num_layers,
        num_heads=project_config.marker_model.adapter_num_heads,
        mlp_ratio=project_config.marker_model.adapter_mlp_ratio,
        enable_presence_head=project_config.marker_model.enable_presence_head,
    )
    temporal_adapter = TemporalHandPromptAdapter(
        prompt_dim=project_config.marker_model.prompt_dim,
        num_layers=project_config.marker_model.temporal_prompt_layers,
        num_heads=project_config.marker_model.temporal_prompt_heads,
        mlp_ratio=project_config.marker_model.temporal_prompt_mlp_ratio,
    )
    hand_head = HandHeadV3(
        prompt_dim=project_config.marker_model.prompt_dim,
        num_markers=project_config.marker_model.num_markers,
        num_queries=project_config.marker_model.marker_query_count,
        hand_feature_dim=hand_adapter.hand_feature_dim,
        root_num_layers=project_config.marker_model.root_head_num_layers,
        joint_num_layers=project_config.marker_model.joint_head_num_layers,
        vertex_num_layers=project_config.marker_model.vertex_head_num_layers,
        num_heads=project_config.marker_model.hand_head_num_heads,
        mlp_ratio=project_config.marker_model.hand_head_mlp_ratio,
        vertex_joint_weights=marker_vertex_joint_weights_for_count(project_config.marker_model.num_markers),
        random_init_contact_confidence_heads=project_config.marker_model.random_init_contact_confidence_heads,
    )
    metric_head = MetricHead(
        prompt_dim=project_config.marker_model.prompt_dim,
        hand_feature_dim=hand_adapter.hand_feature_dim,
    )
    return MarkerModel(
        backbone,
        keypoint_head,
        hand_adapter,
        temporal_adapter,
        hand_head,
        metric_head=metric_head,
        flow_guidance=_build_flow_guidance(project_config.marker_model),
        teacher_feature_dim=teacher_feature_dim,
        keypoint_peak_threshold=project_config.marker_model.peak_threshold,
    )


def build_smoke_flow_model(project_config: ProjectConfig) -> FlowMatchingModel:
    return build_flow_model_from_config(project_config.smoke_flow_model)
