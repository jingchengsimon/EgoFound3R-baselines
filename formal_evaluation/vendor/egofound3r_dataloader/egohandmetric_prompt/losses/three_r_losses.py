from __future__ import annotations

import torch

from egohandmetric_prompt.losses.metric_scale_losses import (
    transform_camera_points_to_first_frame,
    unproject_depth_to_camera_points,
)


def _zero_like_loss(anchor: torch.Tensor) -> torch.Tensor:
    return anchor.sum() * 0.0


def _raw_error_keep_mask(error: torch.Tensor, valid: torch.Tensor, robust_quantile: float) -> torch.Tensor:
    if robust_quantile <= 0.0 or robust_quantile >= 1.0:
        return valid
    values = error[valid]
    if values.numel() == 0:
        return valid
    threshold = torch.quantile(values.detach(), float(robust_quantile))
    return valid & (error <= threshold)


def _mean_valid_loss_with_sample_drop(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_sample_loss: float,
) -> tuple[torch.Tensor, int]:
    valid = valid.to(device=values.device, dtype=torch.bool) & torch.isfinite(values)
    if not torch.any(valid):
        return _zero_like_loss(torch.nan_to_num(values)), 0
    if float(max_sample_loss) <= 0.0:
        return values[valid].mean(), 0

    batch_size = values.shape[0]
    flat_valid = valid.reshape(batch_size, -1)
    flat_values = torch.where(valid, torch.nan_to_num(values), torch.zeros_like(values)).reshape(batch_size, -1)
    counts = flat_valid.sum(dim=1)
    per_sample_valid = counts > 0
    per_sample_loss = flat_values.sum(dim=1) / counts.clamp_min(1).to(dtype=values.dtype)
    keep = per_sample_valid & torch.isfinite(per_sample_loss) & (
        per_sample_loss.detach() <= float(max_sample_loss)
    )
    dropped = int((per_sample_valid & ~keep).sum().item())
    if not torch.any(keep):
        return _zero_like_loss(torch.nan_to_num(values)), dropped
    keep_mask = keep.view(batch_size, *((1,) * (values.ndim - 1)))
    return values[valid & keep_mask].mean(), dropped


def depth_confidence_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    depth_conf: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    relative_weight_max: float,
    robust_quantile: float,
    max_sample_loss: float = 0.0,
    return_dropped: bool = False,
    min_depth: float = 1e-6,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    if pred_depth.shape != target_depth.shape:
        raise ValueError("pred_depth and target_depth must have the same shape")
    if depth_conf.shape != pred_depth.shape:
        raise ValueError("depth_conf must match pred_depth shape")
    if mask.shape != pred_depth.shape:
        raise ValueError("mask must match pred_depth shape")

    valid = mask.to(device=pred_depth.device, dtype=torch.bool)
    valid &= torch.isfinite(pred_depth) & torch.isfinite(target_depth) & torch.isfinite(depth_conf)
    if not torch.any(valid):
        return _zero_like_loss(pred_depth)

    error = torch.abs(pred_depth - target_depth)
    error = torch.where(valid, error, torch.zeros_like(error))
    valid = _raw_error_keep_mask(error, valid, robust_quantile)
    if not torch.any(valid):
        return _zero_like_loss(pred_depth)

    conf = depth_conf.clamp_min(1.0 + eps)
    safe_target_depth = torch.where(valid, target_depth, torch.ones_like(target_depth))
    relative_weight = 1.0 + 1.0 / safe_target_depth.clamp_min(min_depth)
    relative_weight = relative_weight.clamp_max(float(relative_weight_max))
    pixel_loss = conf * relative_weight * error - float(alpha) * torch.log(conf)
    loss, dropped = _mean_valid_loss_with_sample_drop(
        pixel_loss,
        valid,
        max_sample_loss=max_sample_loss,
    )
    if return_dropped:
        return loss, dropped
    return loss


def point_confidence_loss(
    *,
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
    depth_conf: torch.Tensor,
    pred_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    pred_camera_pose: torch.Tensor,
    target_camera_pose: torch.Tensor,
    pred_scene_scale: torch.Tensor,
    target_scene_scale: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    relative_weight_max: float,
    robust_quantile: float,
    max_sample_loss: float = 0.0,
    return_dropped: bool = False,
    min_depth: float = 1e-6,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    if pred_depth.shape != target_depth.shape:
        raise ValueError("pred_depth and target_depth must have the same shape")
    if depth_conf.shape != pred_depth.shape:
        raise ValueError("depth_conf must match pred_depth shape")
    if mask.shape != pred_depth.shape:
        raise ValueError("mask must match pred_depth shape")
    if pred_scene_scale.shape != (pred_depth.shape[0],):
        raise ValueError("pred_scene_scale must have shape [B]")
    if target_scene_scale.shape != (pred_depth.shape[0],):
        raise ValueError("target_scene_scale must have shape [B]")

    valid = mask.to(device=pred_depth.device, dtype=torch.bool)
    valid &= torch.isfinite(pred_depth) & torch.isfinite(target_depth) & torch.isfinite(depth_conf)
    valid &= torch.isfinite(pred_intrinsics).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    valid &= torch.isfinite(target_intrinsics).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    valid &= torch.isfinite(pred_camera_pose).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    valid &= torch.isfinite(target_camera_pose).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    valid &= torch.isfinite(pred_scene_scale).view(-1, 1, 1, 1)
    valid &= torch.isfinite(target_scene_scale).view(-1, 1, 1, 1)
    if not torch.any(valid):
        return _zero_like_loss(pred_depth)

    pred_points = unproject_depth_to_camera_points(pred_depth, pred_intrinsics)
    target_points = unproject_depth_to_camera_points(target_depth, target_intrinsics)
    pred_points = transform_camera_points_to_first_frame(pred_points, pred_camera_pose)
    target_points = transform_camera_points_to_first_frame(target_points, target_camera_pose)
    pred_points = pred_points / pred_scene_scale.detach().clamp_min(eps).view(-1, 1, 1, 1, 1)
    target_points = target_points / target_scene_scale.detach().clamp_min(eps).view(-1, 1, 1, 1, 1)

    point_diff = pred_points - target_points
    point_diff = torch.where(
        valid.unsqueeze(-1),
        point_diff,
        torch.zeros_like(point_diff),
    )
    error = torch.linalg.norm(point_diff, dim=-1)
    valid = _raw_error_keep_mask(error, valid, robust_quantile)
    if not torch.any(valid):
        return _zero_like_loss(pred_depth)

    conf = depth_conf.clamp_min(1.0 + eps)
    safe_target_depth = torch.where(valid, target_depth, torch.ones_like(target_depth))
    relative_weight = 1.0 + 1.0 / safe_target_depth.clamp_min(min_depth)
    relative_weight = relative_weight.clamp_max(float(relative_weight_max))
    pixel_loss = conf * relative_weight * error - float(alpha) * torch.log(conf)
    loss, dropped = _mean_valid_loss_with_sample_drop(
        pixel_loss,
        valid,
        max_sample_loss=max_sample_loss,
    )
    if return_dropped:
        return loss, dropped
    return loss
