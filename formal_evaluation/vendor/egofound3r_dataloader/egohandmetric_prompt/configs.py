from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


VGGT_OMEGA_AGGREGATOR_LAYER_COUNT = 24


@dataclass(slots=True)
class ProjectInfoConfig:
    default_name: str = "egofound3r"
    seed: int = 0


@dataclass(slots=True)
class MarkerModelConfig:
    dino_dim: int = 1024
    aggregator_dim: int = 2048
    prompt_dim: int = 768
    hand_feature_dim: int = 768
    detection_hidden_dim: int = 256
    num_markers: int = 195
    marker_query_count: int = 195
    prompt_injection_layer: int = 18
    adapter_num_layers: int = 6
    adapter_num_heads: int = 12
    adapter_mlp_ratio: int = 4
    temporal_prompt_layers: int = 2
    temporal_prompt_heads: int = 12
    temporal_prompt_mlp_ratio: int = 4
    root_head_num_layers: int = 1
    joint_head_num_layers: int = 5
    vertex_head_num_layers: int = 5
    hand_head_num_heads: int = 12
    hand_head_mlp_ratio: int = 4
    peak_threshold: float = 0.3
    enable_presence_head: bool = False
    inference_egocentric: bool = True
    enable_flow_guidance: bool = False
    flow_motion_dim: int = 1232
    flow_num_heads: int = 12
    flow_num_motion_blocks: int = 6
    flow_mlp_ratio: int = 4
    flow_patch_size: int = 16
    flow_gate_init_logit: float = 0.0
    flow_local_correlation_radius: int = 3
    flow_local_correlation_dim: int = 64
    random_init_contact_confidence_heads: bool = False


@dataclass(slots=True)
class FlowModelConfig:
    num_markers: int = 195
    state_dim: int = 61
    camera_pose_dim: int = 9
    hidden_dim: int = 768
    num_layers: int = 8
    num_heads: int = 12


@dataclass(slots=True)
class SmokeBackboneConfig:
    feature_dim: int = 8
    aggregator_dim: int = 10
    patch_grid_height: int = 4
    patch_grid_width: int = 4
    depth_height: int = 16
    depth_width: int = 16


@dataclass(slots=True)
class IntervalConfig:
    visible_count_threshold: int = 2
    merge_gap_threshold: int = 1
    filter_length_threshold: int = 1
    visibility_threshold: float = 0.5


@dataclass(slots=True)
class SmokeMarkerTrainConfig:
    batch_size: int = 2
    num_frames: int = 3
    image_height: int = 32
    image_width: int = 32
    learning_rate: float = 1e-3


@dataclass(slots=True)
class SmokeFlowTrainConfig:
    batch_size: int = 2
    num_frames: int = 5
    learning_rate: float = 1e-3


@dataclass(slots=True)
class MarkerDataCommonConfig:
    split: str = "train"
    exclude_h2o_test_split_from_training: bool = False
    dataset_sampling_weights: dict[str, float] = field(default_factory=dict)
    num_frames: int = 2
    min_num_frames: int | None = None
    max_num_frames: int | None = None
    window_stride: int = 1
    min_frame_stride: int = 1
    max_frame_stride: int = 1
    group_frame_stride_in_batch: bool = False
    sequence_sampling_mode: str = "window_proportional"
    batch_size: int = 1
    num_workers: int = 0
    data_loader_prefetch_factor: int = 2
    data_loader_pin_memory: bool = True
    data_loader_worker_threads: int = 1
    streaming_epoch_sampler: bool = False
    container_local_sampling_enabled: bool = False
    container_local_sampling_dataset_names: list[str] = field(default_factory=list)
    container_local_sampling_block_batches: int = 64
    container_local_sampling_start_order: str = "random_cursor"
    container_local_prefetch_enabled: bool = False
    container_local_prefetch_margin_batches: int = 16
    hot3d_rectified_rgb_cache_root: str = ""
    hot3d_rectified_rgb_cache_max_bytes: int = 0
    hot3d_rectified_rgb_cache_worker_count: int = 1
    missing_sample_replacement_attempts: int = 0
    missing_sample_registry_dir: str = ""


@dataclass(slots=True)
class RandomCropResizeConfig:
    enabled: bool = False
    crop_probability: float = 0.0
    target_shapes: list[list[int]] = field(default_factory=list)


@dataclass(slots=True)
class MarkerDataStageConfig:
    marker_dataset_names: list[str] = field(default_factory=list)
    three_r_dataset_names: list[str] = field(default_factory=list)
    marker_dataset_sampling_weights: dict[str, float] = field(default_factory=dict)
    three_r_dataset_sampling_weights: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class MarkerDataConfig:
    common: MarkerDataCommonConfig
    midtrain: MarkerDataStageConfig
    posttrain: MarkerDataStageConfig


@dataclass(slots=True)
class MarkerRuntimeConfig:
    backend: str = "smoke"
    image_height: int = 512
    image_width: int = 512
    random_crop_resize: RandomCropResizeConfig = field(default_factory=RandomCropResizeConfig)
    vggt_checkpoint_path: str = ""
    marker_visibility_loss_weight: float = 0.1
    vertex_visibility_mesh_mode: str = "marker"
    include_scene_occlusion_in_visibility: bool = True
    joint_visibility_loss_weight: float = 0.1
    contact_loss_weight: float = 0.1
    marker_contact_loss_weight: float = 0.1
    contact_distance_loss_weight: float = 0.0
    marker_contact_distance_loss_weight: float = 0.0
    contact_supervision: dict[str, str] = field(default_factory=dict)
    contact_confidence_warmup_steps: int = 200
    contact_confidence_alpha: float = 0.05
    depth_consistency_loss_weight: float = 0.1
    depth_supervision_loss_weight: float = 1.0
    intrinsics_supervision_loss_weight: float = 1.0
    camera_translation_loss_weight: float = 1.0
    camera_absolute_translation_loss_weight: float = 0.0
    camera_rotation_loss_weight: float = 1.0
    camera_translation_max_sample_loss: float = 100.0
    keypoint_heatmap_loss_weight: float = 1.0
    keypoint_heatmap_positive_weight: float = 0.0
    keypoint_offset_loss_weight: float = 1.0
    scene_scale_min_value: float = 1e-3
    scale_normalized_max_sample_loss: float = 100.0
    depth_confidence_alpha: float = 0.2
    depth_relative_weight_max: float = 20.0
    depth_robust_quantile: float = 0.98
    point_supervision_loss_weight: float = 0.25
    point_robust_quantile: float = 0.98
    vertex_depth_pred_self_loss_weight: float = 0.05
    vertex_depth_pred_self_warmup_steps: int = 0
    hand_representation: str = "root_offsets"
    hand_vertex_offset_loss_weight: float = 1.0
    hand_joint_offset_loss_weight: float = 1.0
    hand_root_scene_loss_weight: float = 0.25
    hand_root_metric_loss_weight: float = 0.25
    hand_root_metric_warmup_steps: int = 1000
    hand_joint_scene_loss_weight: float = 0.25
    hand_joint_metric_loss_weight: float = 0.25
    hand_vertex_scene_loss_weight: float = 0.25
    hand_vertex_metric_loss_weight: float = 0.25
    hand_2d_reprojection_loss_weight: float = 2.2
    hand_root_2d_reprojection_loss_weight: float = 11.0
    hand_vertex_2d_reprojection_loss_weight: float = 2.2
    hand_joint_2d_bone_length_loss_weight: float = 5.0
    hand_joint_2d_bbox_size_loss_weight: float = 2.5
    hand_vertex_2d_bbox_size_loss_weight: float = 1.0
    hand_2d_reprojection_warmup_steps: int = 0
    hand_2d_reprojection_error_max: float = 0.0
    hand_confidence_warmup_steps: int = 1000
    hand_conf_log_min: float = 0.0
    hand_conf_log_max: float = 2.0
    hand_conf_local_alpha: float = 0.05
    hand_conf_root_alpha: float = 0.05
    metric_scale_loss_weight: float = 1.0
    metric_hand_joint_loss_weight: float = 0.5
    metric_hand_vertex_loss_weight: float = 0.5
    metric_hand_scale_warmup_steps: int = 0
    metric_scale_min_value: float = 0.01
    metric_scale_max_value: float = 100.0
    enable_wilor_teacher: bool = False
    wilor_teacher_loss_weight: float = 0.1
    wilor_teacher_feature_dim: int = 1280
    wilor_teacher_input: str = "gt_bbox"
    wilor_checkpoint_path: str = ""
    wilor_config_path: str = ""
    flow_loss_weight: float = 0.1
    flow_loss_peak_weight: float = 0.5
    flow_loss_warmup_steps: int = 200
    flow_loss_peak_steps: int = 1000
    flow_loss_decay_steps: int = 3000
    flow_motion_weight: float = 0.0
    flow_motion_threshold_px: float = 1.0
    flow_background_weight: float = 0.05
    flow_hand_region_weight: float = 1.0
    flow_fingertip_weight: float = 3.0
    flow_fingertip_sigma_px: float = 12.0
    flow_multiscale_half_weight: float = 0.25
    flow_smoothness_weight: float = 0.005
    flow_pseudo_cache_root: str = ""
    flow_max_supervised_frame_stride: int = 1


@dataclass(slots=True)
class MarkerTrainConfig:
    output_dir: str = "outputs/marker_train"
    resume_from: str = ""
    resume_mode: str = "full"
    control_mode: str = "steps"
    max_epochs: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    log_interval: int = 10
    find_unused_parameters: bool = True
    posttrain_three_r_stream_ratio: float = 1.0
    posttrain_marker_stream_steps: int = 1
    posttrain_three_r_stream_steps: int = 1
    posttrain_stream_schedule: dict[str, int] = field(default_factory=dict)
    vggt_suffix_activation_checkpointing: bool = False
    eval_interval: int = 100
    checkpoint_interval: int = 100
    keep_last_k: int = 3
    mixed_precision: str = "bf16"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 0
    min_learning_rate: float = 1e-6
    grad_clip_norm: float = 1.0
    eval_stage: str = "posttrain"
    eval_stream: str = "marker"
    eval_steps: int = 4
    eval_max_samples: int = 0
    eval_sample_seed: int = 0
    use_official_h2o_eval: bool = True
    official_h2o_eval_samples: int = 50
    official_h2o_eval_seed: int = 0
    best_metric_name: str = "mano_joint_mpjpe"
    best_metric_mode: str = "min"
    visualization_interval_steps: int = 0
    visualization_sequences_per_step: int = 1


@dataclass(slots=True)
class PathsConfig:
    data_root: str = ""
    human_model_root: str = ""
    dataset_root_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectConfig:
    project: ProjectInfoConfig
    paths: PathsConfig
    marker_model: MarkerModelConfig
    flow_model: FlowModelConfig
    smoke_backbone: SmokeBackboneConfig
    smoke_marker_model: MarkerModelConfig
    smoke_flow_model: FlowModelConfig
    interval: IntervalConfig
    smoke_marker_train: SmokeMarkerTrainConfig
    marker_data: MarkerDataConfig
    marker_runtime: MarkerRuntimeConfig
    marker_train: MarkerTrainConfig
    smoke_flow_train: SmokeFlowTrainConfig


def repo_root() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    git_marker = package_root / ".git"
    if git_marker.is_file():
        gitdir_line = git_marker.read_text().strip()
        if gitdir_line.startswith("gitdir: "):
            gitdir = Path(gitdir_line.removeprefix("gitdir: ").strip())
            if not gitdir.is_absolute():
                gitdir = (git_marker.parent / gitdir).resolve()
            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                return gitdir.parents[2]
    return package_root


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    return _package_root() / "configs" / "default.toml"


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _merge_nested_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _from_section(section_cls, section_data: dict | None):
    base = asdict(section_cls())
    if section_data:
        base.update(section_data)
    valid_keys = {field.name for field in fields(section_cls)}
    unknown_keys = set(base) - valid_keys
    if unknown_keys:
        raise ValueError(f"{section_cls.__name__} 包含未知字段：{sorted(unknown_keys)}")
    return section_cls(**base)


def _validate_marker_model_config(config: MarkerModelConfig) -> None:
    if not 1 <= config.prompt_injection_layer <= VGGT_OMEGA_AGGREGATOR_LAYER_COUNT:
        raise ValueError(
            "marker_model.prompt_injection_layer 必须在 "
            f"[1, {VGGT_OMEGA_AGGREGATOR_LAYER_COUNT}] 范围内，当前为 {config.prompt_injection_layer}。"
        )
    if float(config.peak_threshold) < 0.0 or float(config.peak_threshold) > 1.0:
        raise ValueError(f"marker_model.peak_threshold 必须在 [0, 1] 范围内，当前为 {config.peak_threshold}。")
    if int(config.temporal_prompt_layers) <= 0:
        raise ValueError("marker_model.temporal_prompt_layers 必须为正整数。")
    if int(config.temporal_prompt_heads) <= 0:
        raise ValueError("marker_model.temporal_prompt_heads 必须为正整数。")
    if int(config.temporal_prompt_mlp_ratio) <= 0:
        raise ValueError("marker_model.temporal_prompt_mlp_ratio 必须为正整数。")
    if int(config.root_head_num_layers) <= 0:
        raise ValueError("marker_model.root_head_num_layers 必须为正整数。")
    if int(config.joint_head_num_layers) <= 0:
        raise ValueError("marker_model.joint_head_num_layers 必须为正整数。")
    if int(config.vertex_head_num_layers) <= 0:
        raise ValueError("marker_model.vertex_head_num_layers 必须为正整数。")
    if int(config.flow_motion_dim) <= 0 or int(config.flow_num_heads) <= 0:
        raise ValueError("marker_model flow_motion_dim 和 flow_num_heads 必须为正整数。")
    if int(config.flow_motion_dim) % int(config.flow_num_heads) != 0:
        raise ValueError("marker_model.flow_motion_dim 必须能被 flow_num_heads 整除。")
    if int(config.flow_motion_dim) % 4 != 0:
        raise ValueError("marker_model.flow_motion_dim 必须能被 4 整除。")
    if int(config.flow_num_motion_blocks) <= 0:
        raise ValueError("marker_model.flow_num_motion_blocks 必须为正整数。")
    if int(config.flow_mlp_ratio) <= 0:
        raise ValueError("marker_model.flow_mlp_ratio 必须为正整数。")
    if int(config.flow_patch_size) <= 0:
        raise ValueError("marker_model.flow_patch_size 必须为正整数。")
    if int(config.flow_local_correlation_radius) < 0:
        raise ValueError("marker_model.flow_local_correlation_radius 必须为非负整数。")
    if not 1 <= int(config.flow_local_correlation_dim) <= int(config.flow_motion_dim):
        raise ValueError("marker_model.flow_local_correlation_dim 必须在 [1, flow_motion_dim] 范围内。")


def _validate_marker_train_config(config: MarkerTrainConfig) -> None:
    if int(config.visualization_interval_steps) < 0:
        raise ValueError("marker_train.visualization_interval_steps 必须 >= 0。")
    if int(config.visualization_sequences_per_step) <= 0:
        raise ValueError("marker_train.visualization_sequences_per_step 必须为正整数。")
    if config.resume_mode not in {"weights", "weights_optimizer", "full"}:
        raise ValueError("marker_train.resume_mode 必须是 weights、weights_optimizer 或 full。")
    for name in ("posttrain_marker_stream_steps", "posttrain_three_r_stream_steps"):
        if int(getattr(config, name)) < 0:
            raise ValueError(f"marker_train.{name} 必须 >= 0。")
    if config.posttrain_stream_schedule:
        unknown = set(config.posttrain_stream_schedule) - {"marker", "three_r"}
        if unknown:
            raise ValueError(f"marker_train.posttrain_stream_schedule 包含未知 stream: {sorted(unknown)}")
        if any(int(value) < 0 for value in config.posttrain_stream_schedule.values()):
            raise ValueError("marker_train.posttrain_stream_schedule 的周期必须 >= 0。")
        if not any(int(value) > 0 for value in config.posttrain_stream_schedule.values()):
            raise ValueError("marker_train.posttrain_stream_schedule 至少需要一个正周期。")


def _validate_marker_data_common_config(config: MarkerDataCommonConfig) -> None:
    if int(config.num_workers) < 0:
        raise ValueError("marker_data.common.num_workers 必须为非负整数。")
    if int(config.data_loader_prefetch_factor) <= 0:
        raise ValueError("marker_data.common.data_loader_prefetch_factor 必须为正整数。")
    if int(config.data_loader_worker_threads) <= 0:
        raise ValueError("marker_data.common.data_loader_worker_threads 必须为正整数。")
    if int(config.hot3d_rectified_rgb_cache_max_bytes) < 0:
        raise ValueError("marker_data.common.hot3d_rectified_rgb_cache_max_bytes 必须为非负整数。")
    if int(config.hot3d_rectified_rgb_cache_worker_count) <= 0:
        raise ValueError("marker_data.common.hot3d_rectified_rgb_cache_worker_count 必须为正整数。")
    if bool(config.hot3d_rectified_rgb_cache_root) != bool(config.hot3d_rectified_rgb_cache_max_bytes):
        raise ValueError(
            "HOT3D rectified RGB 缓存需要同时设置 root 与正的 max_bytes，或同时禁用。"
        )
    if int(config.container_local_sampling_block_batches) <= 0:
        raise ValueError("marker_data.common.container_local_sampling_block_batches 必须为正整数。")
    if config.container_local_sampling_start_order not in {"sequential", "random_cursor"}:
        raise ValueError("marker_data.common.container_local_sampling_start_order 必须是 sequential 或 random_cursor。")
    if bool(config.container_local_prefetch_enabled):
        if int(config.container_local_prefetch_margin_batches) <= 0:
            raise ValueError("marker_data.common.container_local_prefetch_margin_batches 必须为正整数。")
        if int(config.container_local_prefetch_margin_batches) >= int(config.container_local_sampling_block_batches):
            raise ValueError(
                "marker_data.common.container_local_prefetch_margin_batches 必须小于 container_local_sampling_block_batches。"
            )


def _validate_flow_guidance_sampling_config(
    marker_model: MarkerModelConfig,
    marker_data_common: MarkerDataCommonConfig,
) -> None:
    if not bool(marker_model.enable_flow_guidance):
        return
    if int(marker_data_common.num_frames) < 2:
        raise ValueError("启用 flow guidance 时 marker_data.common.num_frames 至少为 2。")
    min_num_frames = (
        marker_data_common.num_frames
        if marker_data_common.min_num_frames is None
        else marker_data_common.min_num_frames
    )
    max_num_frames = (
        marker_data_common.num_frames
        if marker_data_common.max_num_frames is None
        else marker_data_common.max_num_frames
    )
    if int(min_num_frames) < 2:
        raise ValueError("启用 flow guidance 时 marker_data.common.min_num_frames 至少为 2。")
    if int(max_num_frames) < 2:
        raise ValueError("启用 flow guidance 时 marker_data.common.max_num_frames 至少为 2。")
    if int(marker_data_common.min_frame_stride) != 1 or int(marker_data_common.max_frame_stride) != 1:
        raise ValueError("启用 flow guidance 时必须使用相邻帧监督，marker_data.common frame stride=1。")


def _validate_marker_runtime_config(config: MarkerRuntimeConfig) -> None:
    if config.vertex_visibility_mesh_mode not in {"full", "marker"}:
        raise ValueError("marker_runtime.vertex_visibility_mesh_mode 必须是 full 或 marker。")
    if config.hand_representation != "root_offsets":
        raise ValueError("marker_runtime.hand_representation 目前必须为 'root_offsets'。")
    if float(config.keypoint_heatmap_loss_weight) < 0.0:
        raise ValueError("marker_runtime.keypoint_heatmap_loss_weight 必须 >= 0。")
    if float(config.keypoint_heatmap_positive_weight) < 0.0:
        raise ValueError("marker_runtime.keypoint_heatmap_positive_weight 必须 >= 0。")
    if float(config.keypoint_offset_loss_weight) < 0.0:
        raise ValueError("marker_runtime.keypoint_offset_loss_weight 必须 >= 0。")
    if float(config.camera_absolute_translation_loss_weight) < 0.0:
        raise ValueError("marker_runtime.camera_absolute_translation_loss_weight 必须 >= 0。")
    if float(config.hand_vertex_offset_loss_weight) < 0.0:
        raise ValueError("marker_runtime.hand_vertex_offset_loss_weight 必须 >= 0。")
    if float(config.hand_joint_offset_loss_weight) < 0.0:
        raise ValueError("marker_runtime.hand_joint_offset_loss_weight 必须 >= 0。")
    if float(config.contact_loss_weight) < 0.0:
        raise ValueError("marker_runtime.contact_loss_weight 必须 >= 0。")
    if float(config.marker_contact_loss_weight) < 0.0:
        raise ValueError("marker_runtime.marker_contact_loss_weight 必须 >= 0。")
    valid_contact_modes = {"object_and_interhand", "interhand_only", "disabled"}
    unknown_contact_modes = set(config.contact_supervision) - {
        "h2o", "hot3d_aria", "hoi4d", "oakink_v2", "whim", "forehoi", "reinterhand", "stera_10m", "egoforce_arctic", "taco", "oakink_v1"
    }
    if unknown_contact_modes:
        raise ValueError(f"marker_runtime.contact_supervision 包含未知数据集: {sorted(unknown_contact_modes)}")
    invalid_contact_modes = {
        name: mode for name, mode in config.contact_supervision.items() if mode not in valid_contact_modes
    }
    if invalid_contact_modes:
        raise ValueError(f"marker_runtime.contact_supervision 模式非法: {invalid_contact_modes}")
    for dataset_name in ("h2o", "hot3d_aria", "hoi4d", "oakink_v2", "egoforce_arctic", "taco"):
        contact_mode = config.contact_supervision.get(dataset_name)
        if contact_mode is not None and contact_mode != "object_and_interhand":
            raise ValueError(
                f"{dataset_name} 是 HOI 数据集，marker_runtime.contact_supervision.{dataset_name} "
                "必须为 object_and_interhand，不能降级为 interhand_only 或 disabled。"
            )
    reinterhand_contact_mode = config.contact_supervision.get("reinterhand")
    if reinterhand_contact_mode is not None and reinterhand_contact_mode != "interhand_only":
        raise ValueError(
            "ReInterHand 只提供双手交互，marker_runtime.contact_supervision.reinterhand "
            "必须为 interhand_only，不能使用 object_and_interhand 或 disabled。"
        )
    for name in (
        "contact_distance_loss_weight",
        "marker_contact_distance_loss_weight",
    ):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"marker_runtime.{name} 必须 >= 0。")
    if int(config.contact_confidence_warmup_steps) < 0:
        raise ValueError("marker_runtime.contact_confidence_warmup_steps 必须 >= 0。")
    if float(config.contact_confidence_alpha) < 0.0:
        raise ValueError("marker_runtime.contact_confidence_alpha 必须 >= 0。")
    if int(config.vertex_depth_pred_self_warmup_steps) < 0:
        raise ValueError("marker_runtime.vertex_depth_pred_self_warmup_steps 必须 >= 0。")
    if float(config.scene_scale_min_value) < 0.0:
        raise ValueError("marker_runtime.scene_scale_min_value 必须 >= 0。")
    if float(config.scale_normalized_max_sample_loss) < 0.0:
        raise ValueError("marker_runtime.scale_normalized_max_sample_loss 必须 >= 0。")
    if int(config.metric_hand_scale_warmup_steps) < 0:
        raise ValueError("marker_runtime.metric_hand_scale_warmup_steps 必须 >= 0。")
    if int(config.hand_confidence_warmup_steps) < 0:
        raise ValueError("marker_runtime.hand_confidence_warmup_steps 必须 >= 0。")
    if float(config.hand_conf_log_min) > float(config.hand_conf_log_max):
        raise ValueError("marker_runtime.hand_conf_log_min 必须 <= hand_conf_log_max。")
    if float(config.hand_conf_local_alpha) < 0.0:
        raise ValueError("marker_runtime.hand_conf_local_alpha 必须 >= 0。")
    if float(config.hand_conf_root_alpha) < 0.0:
        raise ValueError("marker_runtime.hand_conf_root_alpha 必须 >= 0。")
    if int(config.hand_root_metric_warmup_steps) < 0:
        raise ValueError("marker_runtime.hand_root_metric_warmup_steps 必须 >= 0。")
    if float(config.hand_root_2d_reprojection_loss_weight) < 0.0:
        raise ValueError("marker_runtime.hand_root_2d_reprojection_loss_weight 必须 >= 0。")
    if float(config.hand_vertex_2d_reprojection_loss_weight) < 0.0:
        raise ValueError("marker_runtime.hand_vertex_2d_reprojection_loss_weight 必须 >= 0。")
    if float(config.hand_joint_2d_bone_length_loss_weight) < 0.0:
        raise ValueError("marker_runtime.hand_joint_2d_bone_length_loss_weight 必须 >= 0。")
    if float(config.hand_joint_2d_bbox_size_loss_weight) < 0.0:
        raise ValueError("marker_runtime.hand_joint_2d_bbox_size_loss_weight 必须 >= 0。")
    if float(config.hand_vertex_2d_bbox_size_loss_weight) < 0.0:
        raise ValueError("marker_runtime.hand_vertex_2d_bbox_size_loss_weight 必须 >= 0。")
    if int(config.hand_2d_reprojection_warmup_steps) < 0:
        raise ValueError("marker_runtime.hand_2d_reprojection_warmup_steps 必须 >= 0。")
    if float(config.hand_2d_reprojection_error_max) < 0.0:
        raise ValueError("marker_runtime.hand_2d_reprojection_error_max 必须 >= 0。")
    if float(config.flow_loss_weight) < 0.0:
        raise ValueError("marker_runtime.flow_loss_weight 必须 >= 0。")
    if float(config.flow_loss_peak_weight) < 0.0:
        raise ValueError("marker_runtime.flow_loss_peak_weight 必须 >= 0。")
    for field_name in (
        "flow_loss_warmup_steps",
        "flow_loss_peak_steps",
        "flow_loss_decay_steps",
    ):
        if int(getattr(config, field_name)) < 0:
            raise ValueError(f"marker_runtime.{field_name} 必须 >= 0。")
    if float(config.flow_motion_weight) < 0.0:
        raise ValueError("marker_runtime.flow_motion_weight 必须 >= 0。")
    if float(config.flow_motion_threshold_px) <= 0.0:
        raise ValueError("marker_runtime.flow_motion_threshold_px 必须 > 0。")
    for field_name in (
        "flow_background_weight",
        "flow_hand_region_weight",
        "flow_fingertip_weight",
        "flow_multiscale_half_weight",
        "flow_smoothness_weight",
    ):
        if float(getattr(config, field_name)) < 0.0:
            raise ValueError(f"marker_runtime.{field_name} 必须 >= 0。")
    if float(config.flow_fingertip_sigma_px) <= 0.0:
        raise ValueError("marker_runtime.flow_fingertip_sigma_px 必须 > 0。")
    if int(config.flow_max_supervised_frame_stride) != 1:
        raise ValueError("首版 marker_runtime.flow_max_supervised_frame_stride 必须为 1。")


def _default_midtrain_marker_data() -> MarkerDataStageConfig:
    return MarkerDataStageConfig(
        marker_dataset_names=["h2o", "hot3d_aria", "hoi4d", "oakink_v2", "whim", "forehoi", "reinterhand", "stera_10m", "egoforce_arctic"],
        three_r_dataset_names=[],
    )


def _default_posttrain_marker_data() -> MarkerDataStageConfig:
    return MarkerDataStageConfig(
        marker_dataset_names=["h2o"],
        three_r_dataset_names=["h2o"],
    )


def load_project_config(config_path: str | Path | None = None) -> ProjectConfig:
    raw = _load_toml(default_config_path())
    if config_path is not None:
        override = _load_toml(Path(config_path))
        raw = _merge_nested_dict(raw, override)
    marker_data_raw = raw.get("marker_data", {})
    marker_runtime_raw = raw.get("marker_runtime", {}) or {}
    marker_model = _from_section(MarkerModelConfig, raw.get("marker_model"))
    _validate_marker_model_config(marker_model)
    smoke_marker_model = _from_section(MarkerModelConfig, raw.get("smoke_marker_model"))
    _validate_marker_model_config(smoke_marker_model)

    marker_train = _from_section(MarkerTrainConfig, raw.get("marker_train"))
    _validate_marker_train_config(marker_train)

    marker_runtime = _from_section(
        MarkerRuntimeConfig,
        {
            key: value
            for key, value in marker_runtime_raw.items()
            if key != "random_crop_resize"
        },
    )
    _validate_marker_runtime_config(marker_runtime)
    marker_runtime.random_crop_resize = _from_section(
        RandomCropResizeConfig,
        marker_runtime_raw.get("random_crop_resize"),
    )

    marker_data_common = _from_section(MarkerDataCommonConfig, marker_data_raw.get("common"))
    _validate_marker_data_common_config(marker_data_common)
    _validate_flow_guidance_sampling_config(marker_model, marker_data_common)

    return ProjectConfig(
        project=_from_section(ProjectInfoConfig, raw.get("project")),
        paths=_from_section(PathsConfig, raw.get("paths")),
        marker_model=marker_model,
        flow_model=_from_section(FlowModelConfig, raw.get("flow_model")),
        smoke_backbone=_from_section(SmokeBackboneConfig, raw.get("smoke_backbone")),
        smoke_marker_model=smoke_marker_model,
        smoke_flow_model=_from_section(FlowModelConfig, raw.get("smoke_flow_model")),
        interval=_from_section(IntervalConfig, raw.get("interval")),
        smoke_marker_train=_from_section(SmokeMarkerTrainConfig, raw.get("smoke_marker_train")),
        marker_data=MarkerDataConfig(
            common=marker_data_common,
            midtrain=_from_section(
                MarkerDataStageConfig,
                _merge_nested_dict(asdict(_default_midtrain_marker_data()), marker_data_raw.get("midtrain", {})),
            ),
            posttrain=_from_section(
                MarkerDataStageConfig,
                _merge_nested_dict(asdict(_default_posttrain_marker_data()), marker_data_raw.get("posttrain", {})),
            ),
        ),
        marker_runtime=marker_runtime,
        marker_train=marker_train,
        smoke_flow_train=_from_section(SmokeFlowTrainConfig, raw.get("smoke_flow_train")),
    )


def default_data_root_path(config_path: str | Path | None = None) -> Path:
    config = load_project_config(config_path)
    if config.paths.data_root:
        return Path(config.paths.data_root)
    return repo_root().parent / "mnt" / "DATA"


def default_human_model_root_path(config_path: str | Path | None = None) -> Path:
    config = load_project_config(config_path)
    if config.paths.human_model_root:
        return Path(config.paths.human_model_root)
    return repo_root().parent / "mnt" / "MODELS" / "HUMAN"


def default_dataset_root_overrides(config_path: str | Path | None = None) -> dict[str, Path]:
    config = load_project_config(config_path)
    return {name: Path(path) for name, path in config.paths.dataset_root_overrides.items()}
