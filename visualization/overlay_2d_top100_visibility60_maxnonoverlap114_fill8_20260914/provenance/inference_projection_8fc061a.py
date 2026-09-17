"""Display-only calibration and camera-space Hand projection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .data.marker_mesh import marker_edges_for_count, marker_faces_for_count
from .hand_upsampling import build_mano778_hand
from .inference_gt_visualization import gt_hand_arrays
from .inference_visualization import INFERENCE_COLOR_MODES

K_LABELS = {"pred": "Pred K", "gt": "GT K", "root": "Root K"}
K_SOURCES = {label: source for source, label in K_LABELS.items()}


def visualization_intrinsics(payload: dict, source: str) -> tuple[torch.Tensor, torch.Tensor]:
    if source == "gt":
        section = payload.get("ground_truth", {}).get("intrinsics", {})
        K, valid = section.get("K"), section.get("valid")
    elif source == "root":
        K, valid = payload.get("hand_projection_K_full"), payload.get("hand_projection_K_valid_full")
    elif source == "pred":
        section = payload.get("metric_predictions", {}).get("intrinsics", {})
        K, valid = section.get("K_full"), section.get("K_full_valid")
        if K is None:
            K = next((payload[name] for name in ("intrinsics_K_full", "K_full", "intrinsics_full", "intrinsics")
                      if isinstance(payload.get(name), torch.Tensor)), None)
            valid = payload.get("intrinsics_K_full_valid")
    else:
        raise ValueError(f"unknown visualization K source: {source}")
    if not isinstance(K, torch.Tensor):
        raise ValueError(f"{K_LABELS[source]} is unavailable")
    K = K.detach().cpu().float()
    frames = len(payload.get("source_frame_indices", K))
    if K.shape != (frames, 3, 3):
        raise ValueError("visualization K must align with the complete R axis")
    if valid is None:
        valid = torch.ones(frames, dtype=torch.bool)
    if not isinstance(valid, torch.Tensor) or valid.shape != (frames,) or valid.dtype != torch.bool:
        raise ValueError("visualization K validity must be bool [T_R]")
    valid = valid.detach().cpu() & torch.isfinite(K).all(dim=(-2, -1)) & (K[:, 0, 0] > 0) & (K[:, 1, 1] > 0)
    valid &= (K[:, 0, 1].abs() < 1e-6) & (K[:, 1, 0].abs() < 1e-6)
    valid &= (K[:, 2] - torch.tensor([0., 0., 1.])).abs().amax(-1) < 1e-6
    return torch.where(valid[:, None, None], K, torch.eye(3).expand_as(K)), valid


def available_visualization_intrinsics(payload: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result = {"pred": visualization_intrinsics(payload, "pred")}
    if isinstance(payload.get("hand_projection_K_full"), torch.Tensor):
        result["root"] = visualization_intrinsics(payload, "root")
    if isinstance(payload.get("ground_truth", {}).get("intrinsics", {}).get("K"), torch.Tensor):
        result["gt"] = visualization_intrinsics(payload, "gt")
    return {key: value for key, value in result.items() if key != "gt" or bool(value[1].any())}


def visualization_extrinsics(payload: dict, source: str) -> tuple[np.ndarray, np.ndarray]:
    from .inference_ground_truth import align_gt_camera_to_prediction

    pred = payload.get("camera_pose_metric_full", payload.get("camera_pose"))
    if not isinstance(pred, torch.Tensor) or pred.ndim != 3 or pred.shape[-2:] != (4, 4):
        raise ValueError("visualization extrinsics require R-rate W2C poses")
    pred = pred.detach().cpu().float()
    pred_valid = payload.get("camera_pose_metric_valid_full", torch.isfinite(pred).all(dim=(-2, -1)))
    pred_valid = pred_valid.detach().cpu().bool()
    if source == "pred":
        pose, valid = pred, pred_valid
    elif source == "gt":
        camera = payload["ground_truth"]["camera"]
        pose, _, reference = align_gt_camera_to_prediction(camera["pose_w2c_metric"], camera["valid"], pred, pred_valid)
        valid = camera["valid"].detach().cpu().bool() & (reference is not None)
    else:
        raise ValueError(f"unknown visualization extrinsics source: {source}")
    valid = (valid & torch.isfinite(pose).all(dim=(-2, -1))).numpy()
    c2w = np.tile(np.eye(4, dtype=np.float32), (len(pose), 1, 1))
    c2w[valid] = np.linalg.inv(pose.cpu().numpy()[valid])
    return c2w, valid


def predicted_hand_arrays(payload: dict, count: int = 195) -> dict[str, np.ndarray]:
    hand = payload.get("metric_predictions", {}).get("hand", payload)
    if count not in (195, 778):
        raise ValueError("projection marker count must be 195 or 778")
    vertices = hand
    if count == 778:
        vertices = payload.get("hand_778")
        if not isinstance(vertices, dict):
            vertices = build_mano778_hand(hand, source_vertex_ids=payload.get("vertex_ids"))
    result = {}
    for prefix in ("joint", "vertex"):
        values = vertices if prefix == "vertex" else hand
        xyz = values.get(prefix + "_xyz_metric", payload.get(prefix + "_xyz"))
        if not isinstance(xyz, torch.Tensor):
            raise ValueError(f"missing metric {prefix} geometry")
        xyz = xyz.detach().cpu().float()
        valid = payload.get("hand_display_valid_full", hand.get("valid", payload.get("hand_metric_valid_full")))
        if valid is None:
            valid = torch.ones(xyz.shape[:2], dtype=torch.bool)
        result[prefix + "_xyz"] = xyz.numpy()
        result[prefix + "_valid"] = (valid.cpu().bool()[..., None] & torch.isfinite(xyz).all(-1)).numpy()
        for attr in ("visibility_probability", "contact_probability", "contact_distance_metric"):
            value = values.get(prefix + "_" + attr)
            if value is None:
                log_name = attr.replace("_probability", "_logits") if attr.endswith("probability") else "contact_log_distance"
                logits = values.get(prefix + "_" + log_name, payload.get(prefix + "_" + log_name))
                if isinstance(logits, torch.Tensor):
                    value = torch.sigmoid(logits.float()) if attr.endswith("probability") else torch.expm1(logits.float()).clamp_min(0) / 1000
            result[prefix + "_" + attr] = (np.full(xyz.shape[:-1], np.nan, dtype=np.float32) if value is None
                                            else value.detach().cpu().float().numpy())
    result["faces"] = (vertices["marker_faces"] if count == 778 else marker_faces_for_count(195)).cpu().numpy()
    result["edges"] = (vertices["marker_edges"] if count == 778 else marker_edges_for_count(195)).cpu().numpy()
    return result


def render_hand_projection(arrays: dict, frame: int, rgb: np.ndarray, K: np.ndarray | torch.Tensor,
                           valid: bool, mode: str, *, title: str | None = None,
                           geometry_color_rgb=None, draw_title: bool = True, **display) -> np.ndarray:
    from infer_marker_video import _project_points, _render_semantic_overlay_frame

    matrix = torch.as_tensor(K, dtype=torch.float32).reshape(1, 3, 3)
    projected = {}
    for prefix in ("joint", "vertex"):
        uv, mask = _project_points(torch.from_numpy(arrays[prefix + "_xyz"][frame:frame + 1]), matrix)
        mask &= torch.from_numpy(arrays[prefix + "_valid"][frame:frame + 1]) & bool(valid)
        projected[prefix + "_uv"], projected[prefix + "_valid"] = uv[0].numpy(), mask[0].numpy()
    return _render_semantic_overlay_frame(
        rgb, mode=mode, marker_uv=projected["vertex_uv"], marker_valid=projected["vertex_valid"],
        marker_visibility=arrays["vertex_visibility_probability"][frame], marker_contact=arrays["vertex_contact_probability"][frame],
        marker_distance_m=arrays["vertex_contact_distance_metric"][frame], joint_uv=projected["joint_uv"],
        joint_valid=projected["joint_valid"], joint_visibility=arrays["joint_visibility_probability"][frame],
        joint_contact=arrays["joint_contact_probability"][frame], joint_distance_m=arrays["joint_contact_distance_metric"][frame],
        marker_edges=arrays["edges"], marker_faces=arrays["faces"], title=title,
        geometry_color_rgb=geometry_color_rgb, draw_title=draw_title, **display,
    )


def letterbox_rgb(rgb: np.ndarray, size_hw: tuple[int, int], *, display_size_hw=None) -> tuple[np.ndarray, np.ndarray]:
    height, width = size_hw
    native_h, native_w = rgb.shape[:2] if display_size_hw is None else display_size_hw
    scale = min(width / native_w, height / native_h)
    affine = np.array([[scale * native_w / rgb.shape[1], 0., (width - scale * native_w) / 2],
                       [0., scale * native_h / rgb.shape[0], (height - scale * native_h) / 2], [0., 0., 1.]])
    image = cv2.warpPerspective(rgb, affine, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(242, 242, 242))
    return image, affine


def overlay_projected_root_trajectory(image: np.ndarray, arrays: dict, K: torch.Tensor, K_valid: torch.Tensor,
                                     affine: np.ndarray, timestamps: np.ndarray, frame: int,
                                     mode: str, trail_seconds: float, colors) -> np.ndarray:
    from infer_marker_video import _project_points, _draw_segment
    from visualize_marker_inference import _trajectory_segments

    root = torch.from_numpy(arrays["joint_xyz"][:, :, :1])
    uv, valid = _project_points(root, K)
    valid = valid[:, :, 0].numpy() & arrays["joint_valid"][:, :, 0] & K_valid.numpy()[:, None]
    pixels = np.concatenate((uv[:, :, 0].numpy(), np.ones((*uv.shape[:2], 1))), -1) @ affine.T
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for side in range(2):
        for segment in _trajectory_segments(pixels[:, side], valid[:, side], timestamps, frame, mode, trail_seconds):
            _draw_segment(canvas, segment[0, :2], segment[1, :2], tuple(colors[side][::-1]), 1)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def iter_intrinsics_comparison_frames(payload: dict, rgb_frames, *, marker_count: int = 195, show_gt: bool = True):

    sources = available_visualization_intrinsics(payload)
    if "gt" not in sources or "pred" not in sources:
        return
    owners = [("Pred", predicted_hand_arrays(payload, marker_count))]
    if show_gt and isinstance(payload.get("ground_truth"), dict):
        owners.append(("GT", gt_hand_arrays(payload["ground_truth"], marker_count)))
    for frame, rgb in enumerate(rgb_frames):
        geometry_rows, semantic_rows = [], []
        for owner, arrays in owners:
            blocks = []
            for source in ("pred", "gt"):
                K, valid = sources[source]
                panels = [render_hand_projection(
                    arrays, frame, rgb, K[frame], bool(valid[frame]), mode,
                    title=f"{owner} / {K_LABELS[source]} / {mode}" if valid[frame] else f"{owner} / {K_LABELS[source]} unavailable",
                ) for mode in INFERENCE_COLOR_MODES]
                blocks.append(panels)
            geometry_rows.append(np.concatenate([block[0] for block in blocks], axis=1))
            semantic_rows.append(np.concatenate([
                np.concatenate((np.concatenate(block[:2], 1), np.concatenate(block[2:], 1)), 0) for block in blocks
            ], axis=1))
        yield {"overlay_k_compare": np.concatenate(geometry_rows, axis=0),
               "overlay_multimode_k_compare": np.concatenate(semantic_rows, axis=0)}

def write_intrinsics_comparisons(payload: dict, rgb_frames, output_dir: Path, fps: float,
                                *, marker_count: int = 195, show_gt: bool = True,
                                save_frame_images: bool = True) -> bool:
    from egohandmetric_prompt.inference_video_io import write_overlay_streams

    sources = available_visualization_intrinsics(payload)
    if "gt" not in sources or "pred" not in sources:
        return False
    write_overlay_streams(output_dir, iter_intrinsics_comparison_frames(
        payload, rgb_frames, marker_count=marker_count, show_gt=show_gt,
    ), fps, save_frame_images=save_frame_images)
    return True
