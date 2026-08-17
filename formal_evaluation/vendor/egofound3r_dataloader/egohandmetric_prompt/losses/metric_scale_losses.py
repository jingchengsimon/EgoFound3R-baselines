from __future__ import annotations

import torch
import torch.nn.functional as F


def _expand_batch_scale(scale: torch.Tensor, target_ndim: int) -> torch.Tensor:
    return scale.reshape(scale.shape[0], *((1,) * (target_ndim - 1)))


def masked_average_distance(
    points: torch.Tensor,
    mask: torch.Tensor,
    *,
    fallback: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if points.shape[:-1] != mask.shape:
        raise ValueError("points and mask shapes must match except for the xyz dimension")
    if points.shape[-1] != 3:
        raise ValueError("points must end with xyz dimension")
    batch_size = points.shape[0]
    flat_points = points.reshape(batch_size, -1, 3)
    flat_mask = mask.reshape(batch_size, -1).to(dtype=torch.bool)
    finite = torch.isfinite(flat_points).all(dim=-1)
    valid = flat_mask & finite
    distances = torch.linalg.norm(torch.nan_to_num(flat_points), dim=-1)
    counts = valid.sum(dim=1)
    summed = torch.where(valid, distances, torch.zeros_like(distances)).sum(dim=1)
    scale = summed / counts.clamp_min(1).to(dtype=points.dtype)
    fallback_tensor = torch.full_like(scale, float(fallback))
    return torch.where(counts > 0, scale.clamp_min(eps), fallback_tensor)


def closed_form_scale_target(
    pred_norm: torch.Tensor,
    target_metric: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_value: float,
    max_value: float,
    fallback: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred_norm.shape != target_metric.shape:
        raise ValueError("pred_norm and target_metric must have the same shape")
    if pred_norm.shape[:-1] != mask.shape:
        raise ValueError("mask must match pred_norm except for xyz dimension")
    if pred_norm.shape[-1] != 3:
        raise ValueError("point tensors must end with xyz dimension")
    batch_size = pred_norm.shape[0]
    flat_pred = pred_norm.reshape(batch_size, -1, 3)
    flat_target = target_metric.reshape(batch_size, -1, 3)
    flat_mask = mask.reshape(batch_size, -1).to(dtype=torch.bool)
    finite = torch.isfinite(flat_pred).all(dim=-1) & torch.isfinite(flat_target).all(dim=-1)
    valid = flat_mask & finite
    pred = torch.nan_to_num(flat_pred)
    target = torch.nan_to_num(flat_target)
    numerator = torch.where(valid, (pred * target).sum(dim=-1), torch.zeros_like(pred[..., 0])).sum(dim=1)
    denominator = torch.where(valid, (pred * pred).sum(dim=-1), torch.zeros_like(pred[..., 0])).sum(dim=1)
    raw_scale = numerator / denominator.clamp_min(eps)
    fallback_tensor = torch.full_like(raw_scale, float(fallback))
    scale = torch.where((valid.sum(dim=1) > 0) & (denominator > eps) & (raw_scale > 0), raw_scale, fallback_tensor)
    return scale.clamp(min=float(min_value), max=float(max_value))


def normalize_points_by_scale(points: torch.Tensor, scale: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    if points.shape[0] != scale.shape[0]:
        raise ValueError("points and scale batch dimensions must match")
    return points / _expand_batch_scale(scale.clamp_min(eps), points.ndim)


def depth_anchored_hand_scale_target(
    pred_depth: torch.Tensor,
    pred_intrinsics: torch.Tensor,
    pred_scene_scale: torch.Tensor,
    points_2d: torch.Tensor,
    target_points_3d: torch.Tensor,
    point_mask: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    min_value: float,
    max_value: float,
    fallback: float = 1.0,
    min_depth: float = 1e-6,
    min_scene_scale: float = 1e-6,
    min_valid_points: int = 1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_depth.ndim == 5:
        if pred_depth.shape[-1] != 1:
            raise ValueError("pred_depth with 5 dims must end with a singleton channel")
        pred_depth = pred_depth.squeeze(-1)
    if pred_depth.ndim != 4:
        raise ValueError("pred_depth must have shape [B, T, H, W]")
    if pred_intrinsics.shape[:2] != pred_depth.shape[:2] or pred_intrinsics.shape[-2:] != (3, 3):
        raise ValueError("pred_intrinsics must have shape [B, T, 3, 3]")
    if pred_scene_scale.shape != (pred_depth.shape[0],):
        raise ValueError("pred_scene_scale must have shape [B]")
    if points_2d.shape[:-1] != target_points_3d.shape[:-1] or points_2d.shape[-1] != 2:
        raise ValueError("points_2d and target_points_3d must match except for xy/xyz dimensions")
    if target_points_3d.shape[-1] != 3:
        raise ValueError("target_points_3d must end with xyz dimension")
    if point_mask.shape != points_2d.shape[:-1]:
        raise ValueError("point_mask must match points_2d except for xy dimension")
    if points_2d.shape[:2] != pred_depth.shape[:2]:
        raise ValueError("points_2d must share pred_depth batch and frame dimensions")

    with torch.no_grad():
        depth = pred_depth.detach()
        intrinsics = pred_intrinsics.detach()
        scene_scale = pred_scene_scale.detach()
        points_2d = points_2d.detach().to(device=depth.device, dtype=depth.dtype)
        target_points_3d = target_points_3d.detach().to(device=depth.device, dtype=depth.dtype)
        point_mask = point_mask.to(device=depth.device, dtype=torch.bool)

        batch_size, num_frames, height, width = depth.shape
        u = points_2d[..., 0] * (float(width) / max(float(image_width), 1.0))
        v = points_2d[..., 1] * (float(height) / max(float(image_height), 1.0))
        u_norm = (u / max(width - 1, 1)) * 2.0 - 1.0
        v_norm = (v / max(height - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1)
        sampled_depth = F.grid_sample(
            depth.reshape(batch_size * num_frames, 1, height, width),
            torch.nan_to_num(grid).reshape(
                batch_size * num_frames,
                grid.shape[2],
                grid.shape[3],
                2,
            ),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)
        sampled_depth = sampled_depth.reshape(
            batch_size,
            num_frames,
            grid.shape[2],
            grid.shape[3],
        )

        finite_intrinsics = torch.isfinite(intrinsics).all(dim=(-2, -1))
        finite_intrinsics &= intrinsics[..., 0, 0] > eps
        finite_intrinsics &= intrinsics[..., 1, 1] > eps
        scene_scale_min = max(float(min_scene_scale), float(eps))
        finite_scene_scale = torch.isfinite(scene_scale) & (scene_scale >= scene_scale_min)
        finite_points = torch.isfinite(points_2d).all(dim=-1)
        finite_points &= torch.isfinite(target_points_3d).all(dim=-1)
        in_bounds = (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
        valid = point_mask & finite_points & in_bounds
        valid &= torch.isfinite(sampled_depth) & (sampled_depth > min_depth)
        valid &= finite_intrinsics.unsqueeze(-1).unsqueeze(-1)
        valid &= finite_scene_scale.view(batch_size, 1, 1, 1)

        safe_intrinsics = torch.nan_to_num(intrinsics)
        fx = safe_intrinsics[..., 0, 0].clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
        fy = safe_intrinsics[..., 1, 1].clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
        cx = safe_intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
        cy = safe_intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
        z = torch.nan_to_num(sampled_depth)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        pred_points_norm = torch.stack([x, y, z], dim=-1)
        pred_points_norm = pred_points_norm / scene_scale.clamp_min(eps).view(
            batch_size,
            1,
            1,
            1,
            1,
        )

        flat_pred = torch.nan_to_num(pred_points_norm).reshape(batch_size, -1, 3)
        flat_target = torch.nan_to_num(target_points_3d).reshape(batch_size, -1, 3)
        flat_valid = valid.reshape(batch_size, -1)
        numerator = torch.where(
            flat_valid,
            (flat_pred * flat_target).sum(dim=-1),
            torch.zeros_like(flat_pred[..., 0]),
        ).sum(dim=1)
        denominator = torch.where(
            flat_valid,
            (flat_pred * flat_pred).sum(dim=-1),
            torch.zeros_like(flat_pred[..., 0]),
        ).sum(dim=1)
        raw_scale = numerator / denominator.clamp_min(eps)
        target_mask = (flat_valid.sum(dim=1) >= int(min_valid_points)) & (denominator > eps)
        target_mask &= torch.isfinite(raw_scale) & (raw_scale > 0)
        fallback_tensor = torch.full_like(raw_scale, float(fallback))
        target_scale = torch.where(target_mask, raw_scale, fallback_tensor)
        return target_scale.clamp(min=float(min_value), max=float(max_value)), target_mask


def unproject_depth_to_camera_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if depth.ndim == 5:
        if depth.shape[-1] != 1:
            raise ValueError("depth with 5 dims must end with a singleton channel")
        depth = depth.squeeze(-1)
    if depth.ndim != 4:
        raise ValueError("depth must have shape [B, T, H, W]")
    if intrinsics.shape[:2] != depth.shape[:2] or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B, T, 3, 3]")
    height, width = depth.shape[-2:]
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = intrinsics[..., 0, 0].clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[..., 1, 1].clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
    z = depth
    x = (xs.view(1, 1, height, width) - cx) / fx * z
    y = (ys.view(1, 1, height, width) - cy) / fy * z
    return torch.stack([x, y, z], dim=-1)


def transform_camera_points_to_first_frame(points: torch.Tensor, camera_pose: torch.Tensor) -> torch.Tensor:
    if points.ndim != 5 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, T, N, 3] or [B, T, H, W, 3]")
    if camera_pose.shape[:2] != points.shape[:2] or camera_pose.shape[-2:] != (4, 4):
        raise ValueError("camera_pose must have shape [B, T, 4, 4]")
    point_shape = points.shape
    flat_points = points.reshape(points.shape[0], points.shape[1], -1, 3)
    with torch.autocast(device_type=points.device.type, enabled=False):
        camera_pose_f = camera_pose.float()
        relative_pose = torch.matmul(camera_pose_f, torch.linalg.pinv(camera_pose_f[:, :1]))
        frame_to_first = torch.linalg.pinv(relative_pose)
    rotation = frame_to_first[..., :3, :3].to(dtype=flat_points.dtype)
    translation = frame_to_first[..., :3, 3].to(dtype=flat_points.dtype)
    transformed = torch.einsum("btij,btnj->btni", rotation, flat_points) + translation.unsqueeze(-2)
    return transformed.reshape(point_shape)


def scene_average_distance_from_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pose: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    fallback: float = 1.0,
) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth_for_mask = depth.squeeze(-1)
    else:
        depth_for_mask = depth
    points = unproject_depth_to_camera_points(depth, intrinsics)
    points_first = transform_camera_points_to_first_frame(points, camera_pose)
    valid = torch.isfinite(depth_for_mask) & (depth_for_mask > 0)
    valid &= torch.isfinite(intrinsics).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    valid &= torch.isfinite(camera_pose).all(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
    pose_det = torch.linalg.det(camera_pose.float()).abs().to(device=depth.device)
    valid &= (pose_det > 1e-6).unsqueeze(-1).unsqueeze(-1)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=depth.device, dtype=torch.bool)
    return masked_average_distance(
        points_first.reshape(points_first.shape[0], -1, 3),
        valid.reshape(valid.shape[0], -1),
        fallback=fallback,
    )


def log_scale_loss(
    log_metric_value: torch.Tensor,
    target_scale: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool)
    valid &= torch.isfinite(log_metric_value)
    valid &= torch.isfinite(target_scale) & (target_scale > 0)
    if not torch.any(valid):
        return log_metric_value.sum() * 0.0
    target_log = torch.log(target_scale[valid].to(dtype=log_metric_value.dtype))
    return F.smooth_l1_loss(log_metric_value[valid], target_log, beta=beta)
