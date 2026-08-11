"""Metric specifications and adapters for the public evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .advanced_geometry import (
    add_metrics,
    ate,
    contact_coverage,
    point_pck_auc,
    pose_auc,
    relative_rotation_angle,
    relative_translation_angle,
    rpe,
)
from .binary import binary_metrics
from .camera import extrinsic_metrics, intrinsic_metrics
from .contact import contact_distance_metrics
from .depth import depth_metrics
from .detection import average_precision, detection_average_precision, mean_average_precision, roc_auc, soft_similarity
from .efficiency import efficiency_metrics
from .mano import hand_scale_error, mano_metrics, vertex_metrics, world_aligned_mpjpe, world_mpjpe
from .pointcloud import pointcloud_metrics
from .rendering import image_metrics, psnr, ssim
from .temporal import acceleration, acceleration_error, jitter, mpfje, mpfve, rte
from .tracking import tracking_metrics


MetricCompute = Callable[[dict[str, object], dict[str, object]], object]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    required_inputs: tuple[str, ...]
    compute: MetricCompute | None = None
    placeholder_reason: str | None = None


def _finite_mean(values: object) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _w_mpjpe(inputs: dict[str, object], config: dict[str, object]) -> float:
    return _finite_mean(
        world_mpjpe(
            inputs["pred_joints"],
            inputs["gt_joints"],
            inputs.get("joint_mask"),
            unit_scale=float(config.get("unit_scale", 1000.0)),
        )
    )


def _mano_metric(metric_name: str, inputs: dict[str, object], config: dict[str, object]) -> object:
    result = mano_metrics(
        inputs["pred_joints"],
        inputs["gt_joints"],
        joint_mask=inputs.get("joint_mask"),
        root_index=int(config.get("root_index", 0)),
        unit_scale=float(config.get("unit_scale", 1000.0)),
    )
    return result[metric_name]


def _vertex_metric(metric_name: str, inputs: dict[str, object], config: dict[str, object]) -> object:
    result = vertex_metrics(
        inputs["pred_vertices"],
        inputs["gt_vertices"],
        vertex_mask=inputs.get("vertex_mask"),
        root_index=int(config.get("vertex_root_index", 0)),
        unit_scale=float(config.get("unit_scale", 1000.0)),
    )
    return result[metric_name]


def _wa_mpjpe(inputs: dict[str, object], config: dict[str, object]) -> float:
    return _finite_mean(world_aligned_mpjpe(
        inputs["pred_joints"], inputs["gt_joints"], inputs.get("joint_mask"),
        mode="all", chunk_length=int(config.get("chunk_length", 100)),
        unit_scale=float(config.get("unit_scale", 1000.0)),
    ))


def _wa2_mpjpe(inputs: dict[str, object], config: dict[str, object]) -> float:
    return _finite_mean(world_aligned_mpjpe(
        inputs["pred_joints"], inputs["gt_joints"], inputs.get("joint_mask"),
        mode="first2", chunk_length=int(config.get("chunk_length", 100)),
        unit_scale=float(config.get("unit_scale", 1000.0)),
    ))


def _joint_auc(inputs, config):
    return point_pck_auc(inputs["pred_joints"], inputs["gt_joints"], inputs.get("joint_mask"), max_threshold=float(config.get("auc_max_threshold", 50.0)), num_thresholds=int(config.get("auc_num_thresholds", 100)), unit_scale=float(config.get("auc_unit_scale", 1000.0)))


def _vertex_auc(inputs, config):
    return point_pck_auc(inputs["pred_vertices"], inputs["gt_vertices"], inputs.get("vertex_mask"), max_threshold=float(config.get("auc_max_threshold", 50.0)), num_thresholds=int(config.get("auc_num_thresholds", 100)), unit_scale=float(config.get("auc_unit_scale", 1000.0)))


def _rte_from_joints(inputs, config):
    root = int(config.get("root_index", 0))
    return _finite_mean(rte(inputs["gt_joints"][:, root], inputs["pred_joints"][:, root], percent=bool(config.get("rte_percent", True))))


def _ate_from_joints(inputs, config):
    root = int(config.get("root_index", 0))
    return ate(inputs["pred_joints"][:, root], inputs["gt_joints"][:, root], alignment=str(config.get("alignment", "none")))


def _rpe(inputs, config):
    return rpe(inputs["pred_poses"], inputs["gt_poses"], delta=int(config.get("rpe_delta", 1)))


def _pose_auc(inputs, config):
    pred = np.asarray(inputs["pred_poses"], dtype=float)
    gt = np.asarray(inputs["gt_poses"], dtype=float)
    rotation = relative_rotation_angle(pred[..., :3, :3], gt[..., :3, :3])
    translation = relative_translation_angle(pred[..., :3, 3], gt[..., :3, 3])
    return pose_auc(rotation, translation, max_threshold_deg=float(config.get("pose_auc_max_deg", 30.0)), num_thresholds=int(config.get("pose_auc_num_thresholds", 100)))


def _depth(inputs, config):
    return depth_metrics(inputs["pred_depth"], inputs["gt_depth"], inputs.get("depth_mask"), unit_scale=float(config.get("depth_unit_scale", 1.0)))


def _depth_component(name: str, inputs, config):
    return _depth(inputs, config)[name]


def _intrinsic(inputs, config):
    return intrinsic_metrics(inputs["pred_intrinsics"], inputs["gt_intrinsics"], inputs.get("camera_mask"))


def _extrinsic(inputs, config):
    return extrinsic_metrics(inputs["pred_extrinsics"], inputs["gt_extrinsics"], inputs.get("camera_mask"), translation_unit_scale=float(config.get("translation_unit_scale", 1.0)))


def _pointcloud(inputs, config):
    return pointcloud_metrics(inputs["pred_pointcloud"], inputs["gt_pointcloud"], thresholds=tuple(config.get("fscore_thresholds", (0.005, 0.01, 0.02))))


def _pointcloud_component(name: str, inputs, config):
    return _pointcloud(inputs, config)[name]


def _binary(kind: str, inputs, config):
    return binary_metrics(inputs[f"pred_{kind}"], inputs[f"gt_{kind}"], inputs.get(f"{kind}_mask"), prediction_type=str(config.get(f"{kind}_prediction_type", "label")), threshold=float(config.get(f"{kind}_threshold", 0.5)))


def _contact_distance(kind: str, inputs, config):
    return contact_distance_metrics(inputs[f"pred_{kind}_distance"], inputs[f"gt_{kind}_distance"], inputs.get(f"{kind}_distance_mask"), unit_scale=float(config.get("contact_distance_unit_scale", 1000.0)))


def _image(inputs, config):
    return image_metrics(inputs["pred_image"], inputs["gt_image"], data_range=float(config.get("image_data_range", 1.0)), window_size=int(config.get("ssim_window_size", 7)))


def _jitter(inputs, config):
    return _finite_mean(jitter(inputs["pred_joints"], fps=float(config.get("fps", 30.0))))


def _acceleration(inputs, config):
    return _finite_mean(acceleration(inputs["pred_joints"], fps=float(config.get("fps", 30.0)), unit_scale=float(config.get("temporal_unit_scale", 1.0))))


def _acceleration_error(inputs, config):
    return acceleration_error(inputs["pred_joints"], inputs["gt_joints"], fps=float(config.get("fps", 30.0)), unit_scale=float(config.get("temporal_unit_scale", 1.0)))


def _mpfje(inputs, config):
    return mpfje(inputs["pred_joints"], inputs["gt_joints"], unit_scale=float(config.get("temporal_unit_scale", 1.0)))


def _mpfve(inputs, config):
    return mpfve(inputs["pred_vertices"], inputs["gt_vertices"], unit_scale=float(config.get("temporal_unit_scale", 1.0)))


def _add(inputs, config):
    return add_metrics(inputs["object_model_points"], inputs["pred_object_pose"], inputs["gt_object_pose"], object_diameter=float(inputs["object_diameter"]))


def _hand_scale(inputs, config):
    return hand_scale_error(inputs["pred_joints"], inputs["gt_joints"], joint_pair=tuple(config.get("hand_scale_joint_pair", (0, 9))), unit_scale=float(config.get("unit_scale", 1000.0)))


def _rra(inputs, config):
    errors = relative_rotation_angle(inputs["pred_relative_rotation"], inputs["gt_relative_rotation"])
    threshold = np.deg2rad(float(config.get("rra_threshold_deg", 30.0)))
    valid = errors[np.isfinite(errors)]
    return float(np.mean(valid <= threshold)) if valid.size else float("nan")


def _rta(inputs, config):
    errors = relative_translation_angle(inputs["pred_relative_translation"], inputs["gt_relative_translation"])
    threshold = np.deg2rad(float(config.get("rta_threshold_deg", 30.0)))
    valid = errors[np.isfinite(errors)]
    return float(np.mean(valid <= threshold)) if valid.size else float("nan")


def _ap(inputs, config):
    return average_precision(inputs["prediction_scores"], inputs["target_labels"], inputs.get("label_mask"))


def _roc_auc(inputs, config):
    return roc_auc(inputs["prediction_scores"], inputs["target_labels"], inputs.get("label_mask"))


def _sim(inputs, config):
    return soft_similarity(inputs["pred_distribution"], inputs["gt_distribution"])


def _detection_ap(inputs, config):
    return detection_average_precision(inputs["pred_boxes"], inputs["pred_scores"], inputs["gt_boxes"], iou_threshold=float(config.get("detection_iou_threshold", 0.5)))


def _detection_map(inputs, config):
    return mean_average_precision(inputs["pred_boxes_by_class"], inputs["pred_scores_by_class"], inputs["gt_boxes_by_class"], iou_threshold=float(config.get("detection_iou_threshold", 0.5)))


def _psnr(inputs, config):
    return psnr(inputs["pred_image"], inputs["gt_image"], data_range=float(config.get("image_data_range", 1.0)))


def _ssim(inputs, config):
    return ssim(inputs["pred_image"], inputs["gt_image"], data_range=float(config.get("image_data_range", 1.0)), window_size=int(config.get("ssim_window_size", 7)))


def _tracking_component(name: str, inputs, config):
    return _tracking(inputs, config)[name]


def _tracking(inputs, config):
    return tracking_metrics(inputs["pred_track_xy"], inputs["gt_track_xy"], inputs["pred_track_visible"], inputs["gt_track_visible"], thresholds=tuple(config.get("track_thresholds", (1.0, 2.0, 4.0, 8.0, 16.0))))


def _efficiency(inputs, config):
    return efficiency_metrics(elapsed_seconds=float(inputs["elapsed_seconds"]), processed_frames=int(inputs["processed_frames"]), parameters=inputs.get("parameters"), peak_memory_bytes=inputs.get("peak_memory_bytes"))


METRIC_REGISTRY: dict[str, MetricSpec] = {
    "mpjpe": MetricSpec("mpjpe", ("pred_joints", "gt_joints"), lambda i, c: _mano_metric("mpjpe", i, c)),
    "pa_mpjpe": MetricSpec("pa_mpjpe", ("pred_joints", "gt_joints"), lambda i, c: _mano_metric("pa_mpjpe", i, c)),
    "root_relative_mpjpe": MetricSpec("root_relative_mpjpe", ("pred_joints", "gt_joints"), lambda i, c: _mano_metric("root_relative_mpjpe", i, c)),
    "pve": MetricSpec("pve", ("pred_vertices", "gt_vertices"), lambda i, c: _vertex_metric("pve", i, c)),
    "pa_pve": MetricSpec("pa_pve", ("pred_vertices", "gt_vertices"), lambda i, c: _vertex_metric("pa_pve", i, c)),
    "root_relative_pve": MetricSpec("root_relative_pve", ("pred_vertices", "gt_vertices"), lambda i, c: _vertex_metric("root_relative_pve", i, c)),
    "w_mpjpe": MetricSpec("w_mpjpe", ("pred_joints", "gt_joints"), _w_mpjpe),
    "wa_mpjpe": MetricSpec("wa_mpjpe", ("pred_joints", "gt_joints"), _wa_mpjpe),
    "waa_mpjpe": MetricSpec("waa_mpjpe", ("pred_joints", "gt_joints"), _wa_mpjpe),
    "wa2_mpjpe": MetricSpec("wa2_mpjpe", ("pred_joints", "gt_joints"), _wa2_mpjpe),
    "joint_auc": MetricSpec("joint_auc", ("pred_joints", "gt_joints"), _joint_auc),
    "vertex_auc": MetricSpec("vertex_auc", ("pred_vertices", "gt_vertices"), _vertex_auc),
    "rte": MetricSpec("rte", ("pred_joints", "gt_joints"), _rte_from_joints),
    "acceleration": MetricSpec("acceleration", ("pred_joints",), _acceleration),
    "acceleration_error": MetricSpec("acceleration_error", ("pred_joints", "gt_joints"), _acceleration_error),
    "jitter": MetricSpec("jitter", ("pred_joints",), _jitter),
    "mpfje": MetricSpec("mpfje", ("pred_joints", "gt_joints"), _mpfje),
    "mpfve": MetricSpec("mpfve", ("pred_vertices", "gt_vertices"), _mpfve),
    "hand_scale_error": MetricSpec("hand_scale_error", ("pred_joints", "gt_joints"), _hand_scale),
    "ate": MetricSpec("ate", ("pred_joints", "gt_joints"), _ate_from_joints),
    "rpe": MetricSpec("rpe", ("pred_poses", "gt_poses"), _rpe),
    "pose_auc": MetricSpec("pose_auc", ("pred_poses", "gt_poses"), _pose_auc),
    "rra": MetricSpec("rra", ("pred_relative_rotation", "gt_relative_rotation"), _rra),
    "rta": MetricSpec("rta", ("pred_relative_translation", "gt_relative_translation"), _rta),
    "depth": MetricSpec("depth", ("pred_depth", "gt_depth"), _depth),
    "mae_depth": MetricSpec("mae_depth", ("pred_depth", "gt_depth"), lambda i, c: _depth_component("mae", i, c)),
    "rmse_depth": MetricSpec("rmse_depth", ("pred_depth", "gt_depth"), lambda i, c: _depth_component("rmse", i, c)),
    "abs_rel": MetricSpec("abs_rel", ("pred_depth", "gt_depth"), lambda i, c: _depth_component("abs_rel", i, c)),
    "sq_rel": MetricSpec("sq_rel", ("pred_depth", "gt_depth"), lambda i, c: _depth_component("sq_rel", i, c)),
    "delta1": MetricSpec("delta1", ("pred_depth", "gt_depth"), lambda i, c: _depth_component("delta1", i, c)),
    "intrinsics": MetricSpec("intrinsics", ("pred_intrinsics", "gt_intrinsics"), _intrinsic),
    "extrinsics": MetricSpec("extrinsics", ("pred_extrinsics", "gt_extrinsics"), _extrinsic),
    "pointcloud": MetricSpec("pointcloud", ("pred_pointcloud", "gt_pointcloud"), _pointcloud),
    "pointcloud_accuracy": MetricSpec("pointcloud_accuracy", ("pred_pointcloud", "gt_pointcloud"), lambda i, c: _pointcloud_component("pred_to_target", i, c)),
    "pointcloud_completeness": MetricSpec("pointcloud_completeness", ("pred_pointcloud", "gt_pointcloud"), lambda i, c: _pointcloud_component("target_to_pred", i, c)),
    "chamfer_l1": MetricSpec("chamfer_l1", ("pred_pointcloud", "gt_pointcloud"), lambda i, c: _pointcloud_component("chamfer_l1", i, c)),
    "chamfer_l2": MetricSpec("chamfer_l2", ("pred_pointcloud", "gt_pointcloud"), lambda i, c: _pointcloud_component("chamfer_l2", i, c)),
    "contact_coverage": MetricSpec("contact_coverage", ("hand_points", "object_points"), lambda i, c: contact_coverage(i["hand_points"], i["object_points"], threshold=float(c.get("contact_threshold", 0.01)))),
    "add": MetricSpec("add", ("object_model_points", "pred_object_pose", "gt_object_pose", "object_diameter"), _add),
    "adds": MetricSpec("adds", ("object_model_points", "pred_object_pose", "gt_object_pose", "object_diameter"), lambda i, c: _add(i, c)["adds"]),
    "add_0_1d": MetricSpec("add_0_1d", ("object_model_points", "pred_object_pose", "gt_object_pose", "object_diameter"), lambda i, c: _add(i, c)["add_0_1d"]),
    "average_precision": MetricSpec("average_precision", ("prediction_scores", "target_labels"), _ap),
    "roc_auc": MetricSpec("roc_auc", ("prediction_scores", "target_labels"), _roc_auc),
    "sim": MetricSpec("sim", ("pred_distribution", "gt_distribution"), _sim),
    "detection_ap": MetricSpec("detection_ap", ("pred_boxes", "pred_scores", "gt_boxes"), _detection_ap),
    "map": MetricSpec("map", ("pred_boxes_by_class", "pred_scores_by_class", "gt_boxes_by_class"), _detection_map),
    "joint_visibility": MetricSpec("joint_visibility", ("pred_joint_visibility", "gt_joint_visibility"), lambda i, c: _binary("joint_visibility", i, c)),
    "vertex_visibility": MetricSpec("vertex_visibility", ("pred_vertex_visibility", "gt_vertex_visibility"), lambda i, c: _binary("vertex_visibility", i, c)),
    "joint_contact": MetricSpec("joint_contact", ("pred_joint_contact", "gt_joint_contact"), lambda i, c: _binary("joint_contact", i, c)),
    "vertex_contact": MetricSpec("vertex_contact", ("pred_vertex_contact", "gt_vertex_contact"), lambda i, c: _binary("vertex_contact", i, c)),
    "joint_contact_distance": MetricSpec("joint_contact_distance", ("pred_joint_contact_distance", "gt_joint_contact_distance"), lambda i, c: _contact_distance("joint", i, c)),
    "vertex_contact_distance": MetricSpec("vertex_contact_distance", ("pred_vertex_contact_distance", "gt_vertex_contact_distance"), lambda i, c: _contact_distance("vertex", i, c)),
    "image": MetricSpec("image", ("pred_image", "gt_image"), _image),
    "psnr": MetricSpec("psnr", ("pred_image", "gt_image"), _psnr),
    "ssim": MetricSpec("ssim", ("pred_image", "gt_image"), _ssim),
    "tracking": MetricSpec("tracking", ("pred_track_xy", "gt_track_xy", "pred_track_visible", "gt_track_visible"), _tracking),
    "average_jaccard": MetricSpec("average_jaccard", ("pred_track_xy", "gt_track_xy", "pred_track_visible", "gt_track_visible"), lambda i, c: _tracking_component("average_jaccard", i, c)),
    "delta_avg_visible": MetricSpec("delta_avg_visible", ("pred_track_xy", "gt_track_xy", "pred_track_visible", "gt_track_visible"), lambda i, c: _tracking_component("delta_avg_visible", i, c)),
    "occlusion_accuracy": MetricSpec("occlusion_accuracy", ("pred_track_xy", "gt_track_xy", "pred_track_visible", "gt_track_visible"), lambda i, c: _tracking_component("occlusion_accuracy", i, c)),
    "efficiency": MetricSpec("efficiency", ("elapsed_seconds", "processed_frames"), _efficiency),
    "fid": MetricSpec(
        "fid",
        (),
        placeholder_reason="requires a caller-selected feature extractor and feature preprocessing protocol",
    ),
    "lpips": MetricSpec("lpips", (), placeholder_reason="requires a caller-selected perceptual network and preprocessing protocol"),
    "intersection_volume": MetricSpec("intersection_volume", (), placeholder_reason="requires a mesh boolean or voxelization protocol"),
    "penetration_depth": MetricSpec("penetration_depth", (), placeholder_reason="requires an object signed-distance function"),
    "geodesic_contact_distance": MetricSpec("geodesic_contact_distance", (), placeholder_reason="requires mesh face topology and a surface geodesic implementation"),
    "euler_lagrange_residual": MetricSpec("euler_lagrange_residual", (), placeholder_reason="requires a hand dynamics model and external forces"),
    "success_rate": MetricSpec("success_rate", (), placeholder_reason="requires an environment-specific success callback"),
}
