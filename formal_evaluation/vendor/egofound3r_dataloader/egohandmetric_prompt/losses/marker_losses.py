from __future__ import annotations

import torch
import torch.nn.functional as F


def hand_confidence_weighted_loss(
    error: torch.Tensor,
    log_conf: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    log_min: float,
    log_max: float,
    use_confidence: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if error.shape != log_conf.shape or error.shape != mask.shape:
        raise ValueError("error, log_conf and mask must share the same shape")
    valid = mask.to(device=error.device, dtype=torch.bool)
    valid &= torch.isfinite(error) & torch.isfinite(log_conf)
    if not torch.any(valid):
        return error[torch.isfinite(error)].sum() * 0.0, {}
    loss_dtype = error.dtype
    error = error.float()
    log_conf = log_conf.float()
    error = torch.where(valid, error, torch.zeros_like(error))
    log_conf = torch.where(valid, log_conf, torch.zeros_like(log_conf))
    if not use_confidence:
        log_conf = torch.zeros_like(log_conf)
    log_conf = log_conf.clamp(float(log_min), float(log_max))
    conf = torch.exp(log_conf)
    weighted = conf * error - float(alpha) * log_conf
    loss = weighted[valid].mean().to(loss_dtype)
    with torch.no_grad():
        conf_valid = conf[valid].detach().float()
        stats = {
            "raw": float(error[valid].detach().float().mean().item()),
            "conf": float(weighted[valid].detach().float().mean().item()),
            "conf_mean": float(conf_valid.mean().item()),
            "conf_p10": float(torch.quantile(conf_valid, 0.10).item()),
            "conf_p50": float(torch.quantile(conf_valid, 0.50).item()),
            "conf_p90": float(torch.quantile(conf_valid, 0.90).item()),
            "log_conf_mean": float(log_conf[valid].detach().float().mean().item()),
        }
    return loss, stats


def vertex_depth_consistency_loss(
    vertex_xyz: torch.Tensor,
    vertex_visibility: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_map: torch.Tensor,
    depth_valid_mask: torch.Tensor | None = None,
    visibility_threshold: float = 0.5,
    min_valid_z: float = 1e-4,
    max_sample_loss: float = 0.0,
    return_dropped: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    if depth_map.ndim == 5:
        if depth_map.shape[-1] != 1:
            raise ValueError(
                "depth_map with 5 dimensions must have a singleton channel dimension"
            )
        depth_map = depth_map.squeeze(-1)
    if depth_map.ndim != 4:
        raise ValueError("depth_map must have shape (batch, num_frames, height, width)")
    batch_size, num_frames, side_count, num_vertices, _ = vertex_xyz.shape
    _, _, height, width = depth_map.shape

    xyz = vertex_xyz.reshape(batch_size * num_frames, side_count, num_vertices, 3)
    visibility = vertex_visibility.reshape(batch_size * num_frames, side_count, num_vertices)
    intrinsics = intrinsics.reshape(batch_size * num_frames, 3, 3)
    depth = depth_map.reshape(batch_size * num_frames, 1, height, width)
    finite_intrinsics = torch.isfinite(intrinsics).all(dim=(-2, -1))
    intrinsics = torch.nan_to_num(intrinsics, nan=0.0, posinf=0.0, neginf=0.0)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid_depth = None
    if depth_valid_mask is not None:
        valid_depth = depth_valid_mask.reshape(batch_size * num_frames, 1, height, width).to(dtype=depth.dtype)

    z = xyz[..., 2].clamp_min(min_valid_z)
    fx = intrinsics[:, None, None, 0, 0]
    fy = intrinsics[:, None, None, 1, 1]
    cx = intrinsics[:, None, None, 0, 2]
    cy = intrinsics[:, None, None, 1, 2]
    u = fx * (xyz[..., 0] / z) + cx
    v = fy * (xyz[..., 1] / z) + cy

    u_norm = (u / max(width - 1, 1)) * 2.0 - 1.0
    v_norm = (v / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1)
    sample_grid = grid.detach()

    sampled_depth = F.grid_sample(
        depth,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(1)
    sampled_valid_depth = None
    if valid_depth is not None:
        sampled_valid_depth = F.grid_sample(
            valid_depth,
            sample_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)

    valid = visibility >= visibility_threshold
    valid &= finite_intrinsics[:, None, None]
    valid &= xyz[..., 2] >= min_valid_z
    valid &= (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
    if sampled_valid_depth is not None:
        valid &= sampled_valid_depth > 0.5
    valid &= torch.isfinite(sampled_depth)

    if not torch.any(valid):
        loss = vertex_xyz.sum() * 0.0
        return (loss, 0) if return_dropped else loss

    diff = torch.abs(sampled_depth - xyz[..., 2])
    if float(max_sample_loss) <= 0.0:
        loss = diff[valid].mean()
        return (loss, 0) if return_dropped else loss

    diff_by_sequence = diff.reshape(batch_size, num_frames, side_count, num_vertices)
    valid_by_sequence = valid.reshape(batch_size, num_frames, side_count, num_vertices)
    counts = valid_by_sequence.reshape(batch_size, -1).sum(dim=1)
    per_sequence_valid = counts > 0
    safe_diff = torch.where(
        valid_by_sequence,
        torch.nan_to_num(diff_by_sequence),
        torch.zeros_like(diff_by_sequence),
    )
    per_sequence_loss = safe_diff.reshape(batch_size, -1).sum(dim=1) / counts.clamp_min(1).to(dtype=diff.dtype)
    keep = per_sequence_valid & torch.isfinite(per_sequence_loss) & (
        per_sequence_loss.detach() <= float(max_sample_loss)
    )
    dropped = int((per_sequence_valid & ~keep).sum().item())
    if not torch.any(keep):
        loss = vertex_xyz.sum() * 0.0
        return (loss, dropped) if return_dropped else loss
    keep_mask = keep.view(batch_size, 1, 1, 1)
    loss = diff_by_sequence[valid_by_sequence & keep_mask].mean()
    return (loss, dropped) if return_dropped else loss


def marker_depth_consistency_loss(
    marker_xyz: torch.Tensor,
    marker_visibility: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_map: torch.Tensor,
    depth_valid_mask: torch.Tensor | None = None,
    visibility_threshold: float = 0.5,
    min_valid_z: float = 1e-4,
    max_sample_loss: float = 0.0,
    return_dropped: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    return vertex_depth_consistency_loss(
        vertex_xyz=marker_xyz,
        vertex_visibility=marker_visibility,
        intrinsics=intrinsics,
        depth_map=depth_map,
        depth_valid_mask=depth_valid_mask,
        visibility_threshold=visibility_threshold,
        min_valid_z=min_valid_z,
        max_sample_loss=max_sample_loss,
        return_dropped=return_dropped,
    )
