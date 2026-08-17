from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from egohandmetric_prompt.configs import ProjectConfig, default_data_root_path
from egohandmetric_prompt.data import H2OFrameDataset, TemporalChunkDataset
from egohandmetric_prompt.losses.contact_losses import (
    contact_classification_metrics,
    masked_contact_bce_with_logits,
)
from egohandmetric_prompt.marker_runtime import (
    MarkerRuntimeCollator,
    build_marker_batch_collator_from_config,
    reconstruct_metric_scale_outputs,
    _fixed_side_tensor,
    _output_flag_enabled,
)


@dataclass(slots=True)
class MarkerEvalSummary:
    sample_count: int
    presence_loss: float
    presence_accuracy: float
    presence_precision: float
    presence_recall: float
    contact_loss: float
    contact_accuracy: float
    contact_precision: float
    contact_recall: float
    contact_f1: float
    contact_valid_count: int
    marker_contact_loss: float
    marker_contact_accuracy: float
    marker_contact_precision: float
    marker_contact_recall: float
    marker_contact_f1: float
    marker_contact_valid_count: int
    mano_joint_mpjpe: float
    mano_joint_visibility_bce: float
    mano_vertex_mae: float
    mano_vertex_visibility_bce: float
    depth_mae: float
    intrinsics_l1: float
    camera_translation_l2: float
    camera_rotation_fro: float


def build_h2o_official_eval_loader(
    project_config: ProjectConfig,
    *,
    max_samples: int | None = None,
    seed: int | None = None,
    split: str = "test",
) -> DataLoader:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported H2O evaluation split: {split}")
    common = project_config.marker_data.common
    if project_config.paths.dataset_root_overrides.get("h2o"):
        h2o_root = Path(project_config.paths.dataset_root_overrides["h2o"])
    elif project_config.paths.data_root:
        h2o_root = Path(project_config.paths.data_root) / "H2O" / "h2o_data"
    else:
        h2o_root = default_data_root_path() / "H2O" / "h2o_data"
    dataset = H2OFrameDataset(
        h2o_root,
        split=split,
        load_rgb=True,
        load_depth=True,
    )
    chunk_dataset = TemporalChunkDataset(
        dataset,
        num_frames=common.num_frames,
        window_stride=common.window_stride,
        drop_last=False,
    )
    if max_samples is not None and max_samples > 0 and len(chunk_dataset) > max_samples:
        generator = torch.Generator()
        generator.manual_seed(project_config.project.seed if seed is None else seed)
        indices = torch.randperm(len(chunk_dataset), generator=generator)[:max_samples].tolist()
        chunk_dataset = Subset(chunk_dataset, indices)
    collator: MarkerRuntimeCollator = build_marker_batch_collator_from_config(project_config, stage="posttrain")
    return DataLoader(
        chunk_dataset,
        batch_size=common.batch_size,
        shuffle=False,
        num_workers=common.num_workers,
        collate_fn=collator,
    )


def summarize_marker_eval_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not metric_dicts:
        return {
            "contact_loss": 0.0,
            "contact_accuracy": 0.0,
            "contact_precision": 0.0,
            "contact_recall": 0.0,
            "contact_f1": 0.0,
            "contact_valid_count": 0.0,
            "contact_true_positive_count": 0.0,
            "contact_false_positive_count": 0.0,
            "contact_false_negative_count": 0.0,
            "contact_true_negative_count": 0.0,
            "marker_contact_loss": 0.0,
            "marker_contact_accuracy": 0.0,
            "marker_contact_precision": 0.0,
            "marker_contact_recall": 0.0,
            "marker_contact_f1": 0.0,
            "marker_contact_valid_count": 0.0,
            "marker_contact_true_positive_count": 0.0,
            "marker_contact_false_positive_count": 0.0,
            "marker_contact_false_negative_count": 0.0,
            "marker_contact_true_negative_count": 0.0,
            "mano_joint_mpjpe": 0.0,
            "mano_joint_visibility_bce": 0.0,
            "mano_vertex_mae": 0.0,
            "mano_vertex_visibility_bce": 0.0,
            "depth_mae": 0.0,
            "intrinsics_l1": 0.0,
            "camera_translation_l2": 0.0,
            "camera_rotation_fro": 0.0,
        }
    keys = metric_dicts[0].keys()
    return {
        key: float(sum(metric[key] for metric in metric_dicts) / len(metric_dicts))
        for key in keys
    }


def _masked_l2_mean(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.ndim == pred.ndim:
        mask = mask[..., 0]
    if not torch.any(mask):
        return 0.0
    diff = torch.linalg.norm(pred - target, dim=-1)
    return float(diff[mask].mean().item())


def _masked_l1_mean(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.ndim == pred.ndim:
        mask = mask[..., 0]
    if not torch.any(mask):
        return 0.0
    diff = torch.abs(pred - target)
    return float(diff[mask.unsqueeze(-1).expand_as(diff)].mean().item())


def _masked_scalar_l1_mean(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not torch.any(mask):
        return 0.0
    diff = torch.abs(pred - target)
    return float(diff[mask].mean().item())


def _masked_matrix_fro_mean(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not torch.any(mask):
        return 0.0
    diff = pred - target
    fro = torch.linalg.norm(diff.reshape(diff.shape[0], diff.shape[1], -1), dim=-1)
    return float(fro[mask].mean().item())


def _masked_bce_mean(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    if not torch.any(mask):
        return 0.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask],
        targets[mask],
    )
    return float(loss.item())


def compute_marker_eval_metrics(
    outputs: dict[str, torch.Tensor | list[Any]],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    outputs = reconstruct_metric_scale_outputs(outputs)
    presence_loss = 0.0
    presence_accuracy = 0.0
    presence_precision = 0.0
    presence_recall = 0.0
    presence_logits = outputs.get("presence_logits")
    presence_loss_enabled = _output_flag_enabled(outputs, "presence_loss_enabled", True)
    presence_targets = batch.get("presence_targets")
    presence_mask = batch.get("presence_supervision_mask")
    if presence_logits is not None and presence_targets is not None and presence_mask is not None:
        presence_targets = presence_targets.to(device=presence_logits.device, dtype=presence_logits.dtype)
        presence_mask = presence_mask.to(device=presence_logits.device)
        if torch.any(presence_mask):
            if presence_loss_enabled:
                presence_loss = float(F.binary_cross_entropy_with_logits(presence_logits[presence_mask], presence_targets[presence_mask]).item())
            pred_presence = presence_logits[presence_mask] > 0
            target_presence = presence_targets[presence_mask] > 0.5
            presence_accuracy = float((pred_presence == target_presence).to(dtype=torch.float32).mean().item())
            true_positive = (pred_presence & target_presence).sum().to(dtype=torch.float32)
            false_positive = (pred_presence & ~target_presence).sum().to(dtype=torch.float32)
            false_negative = (~pred_presence & target_presence).sum().to(dtype=torch.float32)
            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative
            if float(precision_denominator.item()) > 0.0:
                presence_precision = float((true_positive / precision_denominator).item())
            if float(recall_denominator.item()) > 0.0:
                presence_recall = float((true_positive / recall_denominator).item())

    contact_loss = 0.0
    contact_loss_sum = 0.0
    contact_metrics = {
        "contact_accuracy": 0.0,
        "contact_precision": 0.0,
        "contact_recall": 0.0,
        "contact_f1": 0.0,
        "contact_valid_count": 0.0,
        "contact_true_positive_count": 0.0,
        "contact_false_positive_count": 0.0,
        "contact_false_negative_count": 0.0,
        "contact_true_negative_count": 0.0,
    }
    contact_logits = outputs.get("dense_joint_contact_logits")
    contact_targets = batch.get("contact_targets")
    contact_mask = batch.get("contact_supervision_mask")
    if (
        isinstance(contact_logits, torch.Tensor)
        and isinstance(contact_targets, torch.Tensor)
        and isinstance(contact_mask, torch.Tensor)
    ):
        contact_targets = contact_targets.to(device=contact_logits.device, dtype=contact_logits.dtype)
        contact_mask = contact_mask.to(device=contact_logits.device, dtype=torch.bool)
        contact_loss = float(masked_contact_bce_with_logits(contact_logits, contact_targets, contact_mask).item())
        contact_metrics = contact_classification_metrics(contact_logits, contact_targets, contact_mask)
        if torch.any(contact_mask):
            contact_loss_sum = float(
                F.binary_cross_entropy_with_logits(
                    contact_logits[contact_mask],
                    contact_targets[contact_mask],
                    reduction="sum",
                ).item()
            )

    marker_contact_loss = 0.0
    marker_contact_loss_sum = 0.0
    marker_contact_metrics = {
        "marker_contact_accuracy": 0.0,
        "marker_contact_precision": 0.0,
        "marker_contact_recall": 0.0,
        "marker_contact_f1": 0.0,
        "marker_contact_valid_count": 0.0,
        "marker_contact_true_positive_count": 0.0,
        "marker_contact_false_positive_count": 0.0,
        "marker_contact_false_negative_count": 0.0,
        "marker_contact_true_negative_count": 0.0,
    }
    marker_contact_logits = outputs.get("dense_vertex_contact_logits")
    marker_contact_targets = batch.get("marker_contact_targets")
    marker_contact_mask = batch.get("marker_contact_supervision_mask")
    if (
        isinstance(marker_contact_logits, torch.Tensor)
        and isinstance(marker_contact_targets, torch.Tensor)
        and isinstance(marker_contact_mask, torch.Tensor)
    ):
        marker_contact_targets = marker_contact_targets.to(
            device=marker_contact_logits.device,
            dtype=marker_contact_logits.dtype,
        )
        marker_contact_mask = marker_contact_mask.to(device=marker_contact_logits.device, dtype=torch.bool)
        marker_contact_loss = float(
            masked_contact_bce_with_logits(
                marker_contact_logits,
                marker_contact_targets,
                marker_contact_mask,
            ).item()
        )
        marker_contact_metrics = {
            f"marker_{name}": value
            for name, value in contact_classification_metrics(
                marker_contact_logits,
                marker_contact_targets,
                marker_contact_mask,
            ).items()
        }
        if torch.any(marker_contact_mask):
            marker_contact_loss_sum = float(
                F.binary_cross_entropy_with_logits(
                    marker_contact_logits[marker_contact_mask],
                    marker_contact_targets[marker_contact_mask],
                    reduction="sum",
                ).item()
            )

    vertex_targets = _fixed_side_tensor(batch, "vertex_xyz_targets").to(device=outputs["dense_vertex_xyz"].device, dtype=outputs["dense_vertex_xyz"].dtype)
    effective_marker_vertex_mask = _fixed_side_tensor(batch, "vertex_xyz_supervision_mask").to(device=outputs["dense_vertex_xyz"].device)
    mano_vertex_mae = _masked_l1_mean(
        outputs["dense_vertex_xyz"],
        vertex_targets,
        effective_marker_vertex_mask.unsqueeze(-1).expand_as(outputs["dense_vertex_xyz"]),
    )
    vertex_visibility_targets = _fixed_side_tensor(batch, "vertex_visibility_targets").to(device=outputs["dense_vertex_visibility_logits"].device)
    vertex_visibility_mask = _fixed_side_tensor(batch, "vertex_visibility_supervision_mask").to(device=outputs["dense_vertex_visibility_logits"].device)
    mano_vertex_visibility_bce = _masked_bce_mean(
        outputs["dense_vertex_visibility_logits"],
        vertex_visibility_targets.to(dtype=outputs["dense_vertex_visibility_logits"].dtype),
        vertex_visibility_mask,
    )

    joint_targets = _fixed_side_tensor(batch, "joints_3d_targets").to(device=outputs["dense_joint_xyz"].device, dtype=outputs["dense_joint_xyz"].dtype)
    joint_xyz_mask = _fixed_side_tensor(batch, "raw_joint_supervision_mask").to(device=outputs["dense_joint_xyz"].device).unsqueeze(-1).unsqueeze(-1)
    joint_mpjpe = _masked_l2_mean(outputs["dense_joint_xyz"], joint_targets, joint_xyz_mask.expand_as(outputs["dense_joint_xyz"]))
    joint_visibility_targets = _fixed_side_tensor(batch, "joint_visibility_targets").to(device=outputs["dense_joint_visibility_logits"].device)
    joint_visibility_mask = _fixed_side_tensor(batch, "joint_visibility_supervision_mask").to(device=outputs["dense_joint_visibility_logits"].device)
    joint_visibility_bce = _masked_bce_mean(
        outputs["dense_joint_visibility_logits"],
        joint_visibility_targets.to(dtype=outputs["dense_joint_visibility_logits"].dtype),
        joint_visibility_mask,
    )

    depth = batch.get("depth")
    depth_mask = batch.get("depth_supervision_mask")
    depth_valid_mask = batch.get("depth_valid_mask")
    depth_mae = 0.0
    if depth is not None and depth_mask is not None:
        depth = depth.to(device=outputs["depth"].device, dtype=outputs["depth"].dtype)
        batch_size, num_frames, source_height, source_width = depth.shape
        target_height, target_width = outputs["depth"].shape[-2:]
        if (source_height, source_width) != (target_height, target_width):
            depth = F.interpolate(
                depth.reshape(batch_size * num_frames, 1, source_height, source_width),
                size=(target_height, target_width),
                mode="nearest",
            ).reshape(batch_size, num_frames, target_height, target_width)
            if depth_valid_mask is not None:
                depth_valid_mask = (
                    F.interpolate(
                        depth_valid_mask.to(device=outputs["depth"].device, dtype=torch.float32).reshape(
                            batch_size * num_frames, 1, source_height, source_width
                        ),
                        size=(target_height, target_width),
                        mode="nearest",
                    ).reshape(batch_size, num_frames, target_height, target_width)
                    > 0.5
                )
        depth_mask_expanded = depth_mask.to(device=outputs["depth"].device).unsqueeze(-1).unsqueeze(-1).expand_as(outputs["depth"])
        if depth_valid_mask is not None:
            depth_mask_expanded = depth_mask_expanded & depth_valid_mask.to(device=outputs["depth"].device)
        depth_mae = _masked_scalar_l1_mean(outputs["depth"], depth, depth_mask_expanded)

    intrinsics = batch["intrinsics"].to(device=outputs["intrinsics"].device, dtype=outputs["intrinsics"].dtype)
    intrinsics_mask = batch["intrinsics_supervision_mask"].to(device=outputs["intrinsics"].device).unsqueeze(-1).unsqueeze(-1)
    intrinsics_l1 = _masked_l1_mean(outputs["intrinsics"], intrinsics, intrinsics_mask.expand_as(outputs["intrinsics"]))

    translation_l2 = 0.0
    rotation_fro = 0.0
    if outputs["camera_pose"].ndim >= 4 and outputs["camera_pose"].shape[-2:] == (4, 4):
        camera_pose = batch["camera_pose"].to(device=outputs["camera_pose"].device, dtype=outputs["camera_pose"].dtype)
        camera_pose_mask = batch["camera_pose_supervision_mask"].to(device=outputs["camera_pose"].device)
        translation_pred = outputs["camera_pose"][..., :3, 3]
        translation_gt = camera_pose[..., :3, 3]
        translation_l2 = _masked_l2_mean(
            translation_pred,
            translation_gt,
            camera_pose_mask.unsqueeze(-1).expand_as(translation_pred),
        )
        rotation_pred = outputs["camera_pose"][..., :3, :3]
        rotation_gt = camera_pose[..., :3, :3]
        rotation_fro = _masked_matrix_fro_mean(rotation_pred, rotation_gt, camera_pose_mask)

    return {
        "presence_loss": presence_loss,
        "presence_accuracy": presence_accuracy,
        "presence_precision": presence_precision,
        "presence_recall": presence_recall,
        "contact_loss": contact_loss,
        "contact_loss_sum": contact_loss_sum,
        **contact_metrics,
        "marker_contact_loss": marker_contact_loss,
        "marker_contact_loss_sum": marker_contact_loss_sum,
        **marker_contact_metrics,
        "mano_joint_mpjpe": joint_mpjpe,
        "mano_joint_visibility_bce": joint_visibility_bce,
        "mano_vertex_mae": mano_vertex_mae,
        "mano_vertex_visibility_bce": mano_vertex_visibility_bce,
        "depth_mae": depth_mae,
        "intrinsics_l1": intrinsics_l1,
        "camera_translation_l2": translation_l2,
        "camera_rotation_fro": rotation_fro,
    }
