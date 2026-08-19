"""Independent metrics for egocentric 3D vision experiments."""

from .common import as_numpy, prepare_mask, safe_mean, safe_rmse
from .binary import binary_metrics
from .advanced_geometry import (
    add_metrics,
    ate,
    contact_coverage,
    pck_auc,
    point_pck_auc,
    pose_auc,
    relative_rotation_angle,
    relative_translation_angle,
    rpe,
)
from .camera import extrinsic_metrics, intrinsic_metrics, relative_pose_metrics
from .contact import contact_distance_metrics
from .depth import depth_metrics
from .detection import (
    average_precision,
    box_iou,
    detection_average_precision,
    mean_average_precision,
    roc_auc,
    soft_similarity,
)
from .efficiency import efficiency_metrics
from .global_metrics import compute_global_metrics
from .runner import available_metrics, evaluate
from .labels import (
    joint_contact_distance_metrics,
    joint_contact_metrics,
    joint_visibility_metrics,
    vertex_contact_distance_metrics,
    vertex_contact_metrics,
    vertex_visibility_metrics,
)
from .mano import (
    hand_scale_error,
    keypoint_mpjpe,
    mano_metrics,
    sequence_chunk_mpjpe,
    vertex_metrics,
    world_aligned_mpjpe,
    world_mpjpe,
)
from .pointcloud import pointcloud_metrics
from .rendering import image_metrics, psnr, ssim
from .temporal import acceleration, acceleration_error, contact_sliding, frame_difference_error, jitter, mpfje, mpfve, rte, temporal_point_errors
from .tracking import tracking_metrics

__all__ = [
    "as_numpy",
    "prepare_mask",
    "safe_mean",
    "safe_rmse",
    "depth_metrics",
    "pck_auc",
    "point_pck_auc",
    "ate",
    "rpe",
    "pose_auc",
    "relative_rotation_angle",
    "relative_translation_angle",
    "add_metrics",
    "contact_coverage",
    "compute_global_metrics",
    "evaluate",
    "available_metrics",
    "joint_visibility_metrics",
    "vertex_visibility_metrics",
    "joint_contact_metrics",
    "vertex_contact_metrics",
    "joint_contact_distance_metrics",
    "vertex_contact_distance_metrics",
    "intrinsic_metrics",
    "extrinsic_metrics",
    "relative_pose_metrics",
    "pointcloud_metrics",
    "average_precision",
    "roc_auc",
    "soft_similarity",
    "box_iou",
    "detection_average_precision",
    "mean_average_precision",
    "mano_metrics",
    "hand_scale_error",
    "keypoint_mpjpe",
    "vertex_metrics",
    "sequence_chunk_mpjpe",
    "world_mpjpe",
    "world_aligned_mpjpe",
    "acceleration",
    "acceleration_error",
    "frame_difference_error",
    "mpfje",
    "mpfve",
    "temporal_point_errors",
    "jitter",
    "rte",
    "contact_sliding",
    "binary_metrics",
    "contact_distance_metrics",
    "image_metrics",
    "psnr",
    "ssim",
    "tracking_metrics",
    "efficiency_metrics",
]
