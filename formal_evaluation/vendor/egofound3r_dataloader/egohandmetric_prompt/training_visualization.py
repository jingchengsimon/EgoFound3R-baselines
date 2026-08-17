from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from loguru import logger

from egohandmetric_prompt.data.marker_mesh import marker_faces_for_count


MANO_JOINT_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

JOINT_COLORS = (
    (255, 245, 0, 255),
    (255, 0, 255, 255),
)
MARKER_VISIBLE_RGB = (0, 255, 0)
MARKER_HIDDEN_RGB = (255, 0, 0)
MARKER_MIXED_RGB = (255, 165, 0)
MARKER_UNSUPERVISED_RGB = (190, 190, 190)


@dataclass(slots=True)
class SequenceCandidate:
    stream_name: str
    batch_index: int
    score: int
    dataset_name: str
    sequence_id: str


def tensor_rgb_to_image(rgb: torch.Tensor) -> Image.Image:
    array = rgb.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


def _tensor_mask_has_signal(batch: dict, key: str, batch_index: int) -> bool:
    value = batch.get(key)
    if value is None:
        return False
    tensor = torch.as_tensor(value)
    if tensor.shape[0] <= batch_index:
        return False
    return bool(tensor[batch_index].to(dtype=torch.bool).any().item())


def select_sequence_candidates(stream_batches: dict[str, dict], *, limit: int) -> list[SequenceCandidate]:
    candidates: list[SequenceCandidate] = []
    for stream_name, batch in stream_batches.items():
        batch_size = int(batch["images"].shape[0])
        sources = list(batch.get("batch_sources", []))
        for batch_index in range(batch_size):
            has_marker = (
                _tensor_mask_has_signal(batch, "marker_supervision_mask", batch_index)
                or _tensor_mask_has_signal(batch, "raw_joint_supervision_mask", batch_index)
            )
            has_depth = _tensor_mask_has_signal(batch, "depth_supervision_mask", batch_index)
            has_camera = _tensor_mask_has_signal(batch, "camera_pose_supervision_mask", batch_index)
            score = (4 if has_marker else 0) + (2 if has_depth else 0) + (1 if has_camera else 0)
            if score <= 0:
                continue
            source = sources[batch_index] if batch_index < len(sources) else {}
            candidates.append(
                SequenceCandidate(
                    stream_name=stream_name,
                    batch_index=batch_index,
                    score=score,
                    dataset_name=str(source.get("dataset_name", "")),
                    sequence_id=str(source.get("sequence_id", "")),
                )
            )
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.stream_name, candidate.batch_index))
    return candidates[:limit]


def _image_to_chw_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.transpose(array, (2, 0, 1))


def write_visualization_panels(
    *,
    output_dir: str | Path,
    writer,
    source_name: str,
    step: int,
    sequence_name: str,
    panels: dict[str, Image.Image],
) -> int:
    sequence_dir = Path(output_dir) / "visualizations" / f"step_{step:08d}" / source_name / sequence_name
    sequence_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for panel_name, image in panels.items():
        image.save(sequence_dir / f"{panel_name}.png")
        if writer is not None:
            writer.add_image(
                f"visualization/{source_name}/{sequence_name}/{panel_name}",
                _image_to_chw_array(image),
                global_step=step,
            )
        written += 1
    return written


def safe_run_training_visualization(run: Callable[[], None], *, logger_name: str = "training_visualization") -> bool:
    try:
        run()
    except Exception as exc:
        logger.bind(component=logger_name).warning("跳过本次训练可视化: {}", exc)
        return False
    return True


def _fixed_side_tensor(batch: dict, key: str) -> torch.Tensor:
    tensor = batch[key]
    if tensor.shape[2] == 2:
        return tensor
    slots_per_side = int(batch.get("hand_slots_per_side", max(tensor.shape[2] // 2, 1)))
    indices = torch.tensor([0, slots_per_side], dtype=torch.long, device=tensor.device)
    return tensor.index_select(2, indices)


def _sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"


def _sequence_output_name(sequence_index: int, candidate: SequenceCandidate) -> str:
    dataset_name = _sanitize_name(candidate.dataset_name)
    sequence_id = _sanitize_name(candidate.sequence_id)
    stream_name = _sanitize_name(candidate.stream_name)
    return f"seq_{sequence_index:03d}_{stream_name}_{dataset_name}_{sequence_id}"


def _candidate_marker_panels(batch: dict, outputs: dict, candidate: SequenceCandidate) -> dict[str, Image.Image]:
    panels: dict[str, Image.Image] = {}
    if "intrinsics" not in batch:
        return panels
    batch_index = candidate.batch_index
    rgb = batch["images"][batch_index].detach().cpu()
    intrinsics = batch["intrinsics"][batch_index].detach().cpu()
    if not bool(torch.isfinite(intrinsics).all().item()):
        return panels

    gt_vertices = None
    if batch.get("vertex_xyz_targets") is not None:
        gt_vertices = _fixed_side_tensor(batch, "vertex_xyz_targets")[batch_index].detach().cpu()
    gt_vertex_visibility = None
    if batch.get("vertex_visibility_targets") is not None:
        gt_vertex_visibility = _fixed_side_tensor(batch, "vertex_visibility_targets")[batch_index].detach().cpu()
    gt_vertex_visibility_mask = None
    if batch.get("vertex_visibility_supervision_mask") is not None:
        gt_vertex_visibility_mask = _fixed_side_tensor(batch, "vertex_visibility_supervision_mask")[batch_index].detach().cpu()
    gt_joints = None
    if batch.get("joints_3d_targets") is not None:
        gt_joints = _fixed_side_tensor(batch, "joints_3d_targets")[batch_index].detach().cpu()
    if gt_vertices is not None or gt_joints is not None:
        panels["markers_gt"] = render_marker_strip(
            rgb,
            intrinsics,
            gt_vertices,
            gt_joints,
            title="gt",
            vertex_visibility=gt_vertex_visibility,
            vertex_visibility_mask=gt_vertex_visibility_mask,
        )

    pred_vertices = outputs.get("dense_vertex_xyz")
    pred_joints = outputs.get("dense_joint_xyz")
    pred_vertex_visibility_logits = outputs.get("dense_vertex_visibility_logits")
    pred_presence_mask = outputs.get("presence_mask")
    if pred_vertices is not None or pred_joints is not None:
        pred_vertices_for_batch = None if pred_vertices is None else pred_vertices[batch_index].detach().cpu()
        pred_joints_for_batch = None if pred_joints is None else pred_joints[batch_index].detach().cpu()
        pred_vertex_visibility_for_batch = (
            None
            if pred_vertex_visibility_logits is None
            else torch.sigmoid(pred_vertex_visibility_logits[batch_index]).detach().cpu()
        )
        pred_presence_for_batch = None if pred_presence_mask is None else pred_presence_mask[batch_index].detach().cpu()
        panels["markers_pred"] = render_marker_strip(
            rgb,
            intrinsics,
            pred_vertices_for_batch,
            pred_joints_for_batch,
            title="pred",
            vertex_visibility=pred_vertex_visibility_for_batch,
            hand_presence_mask=pred_presence_for_batch,
        )
    return panels


def _resize_depth_to_prediction(batch: dict, outputs: dict, batch_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    depth = batch.get("depth")
    pred_depth = outputs.get("depth")
    if depth is None or pred_depth is None:
        return None
    gt_depth = depth[batch_index : batch_index + 1].detach().cpu().to(dtype=torch.float32)
    pred = pred_depth[batch_index : batch_index + 1].detach().cpu().to(dtype=torch.float32)
    resized_gt = F.interpolate(
        gt_depth.reshape(gt_depth.shape[0] * gt_depth.shape[1], 1, gt_depth.shape[-2], gt_depth.shape[-1]),
        size=pred.shape[-2:],
        mode="nearest",
    ).reshape(gt_depth.shape[0], gt_depth.shape[1], pred.shape[-2], pred.shape[-1])
    valid_mask = torch.isfinite(resized_gt) & (resized_gt > 0.0)
    depth_valid_mask = batch.get("depth_valid_mask")
    if depth_valid_mask is not None:
        source_valid = depth_valid_mask[batch_index : batch_index + 1].detach().cpu().to(dtype=torch.float32)
        resized_valid = F.interpolate(
            source_valid.reshape(source_valid.shape[0] * source_valid.shape[1], 1, source_valid.shape[-2], source_valid.shape[-1]),
            size=pred.shape[-2:],
            mode="nearest",
        ).reshape(gt_depth.shape[0], gt_depth.shape[1], pred.shape[-2], pred.shape[-1]) > 0.5
        valid_mask = valid_mask & resized_valid
    return resized_gt[0], pred[0], valid_mask[0]


def _candidate_depth_panels(batch: dict, outputs: dict, candidate: SequenceCandidate) -> dict[str, Image.Image]:
    if not _tensor_mask_has_signal(batch, "depth_supervision_mask", candidate.batch_index):
        return {}
    resized = _resize_depth_to_prediction(batch, outputs, candidate.batch_index)
    if resized is None:
        return {}
    gt_depth, pred_depth, valid_mask = resized
    return {"depth_gt_pred_absdiff": render_depth_panel(gt_depth, pred_depth, valid_mask)}


def _predicted_camera_pose(outputs: dict) -> torch.Tensor | None:
    pred_pose = outputs.get("camera_pose")
    if pred_pose is not None and pred_pose.ndim >= 4 and pred_pose.shape[-2:] == (4, 4):
        return pred_pose
    pose_encoding = outputs.get("camera_pose_encoding")
    if pose_encoding is not None and pose_encoding.shape[-1] >= 7:
        from egohandmetric_prompt.marker_runtime import _vggt_pose_matrix_from_encoding

        return _vggt_pose_matrix_from_encoding(pose_encoding)
    return None


def _candidate_camera_panels(batch: dict, outputs: dict, candidate: SequenceCandidate) -> dict[str, Image.Image]:
    if not _tensor_mask_has_signal(batch, "camera_pose_supervision_mask", candidate.batch_index):
        return {}
    gt_pose = batch.get("camera_pose")
    pred_pose = _predicted_camera_pose(outputs)
    if gt_pose is None or pred_pose is None:
        return {}
    batch_index = candidate.batch_index
    valid_mask = batch.get("camera_pose_supervision_mask")
    image = render_camera_pose_panel(
        gt_pose[batch_index].detach().cpu(),
        pred_pose[batch_index].detach().cpu(),
        None if valid_mask is None else valid_mask[batch_index].detach().cpu(),
    )
    return {} if image is None else {"camera_pose_3d": image}


def _flow_to_rgb(flow: torch.Tensor, valid: torch.Tensor, *, magnitude_scale: float) -> Image.Image:
    flow_np = flow.detach().cpu().to(dtype=torch.float32).numpy()
    valid_np = valid.detach().cpu().to(dtype=torch.bool).numpy()
    magnitude, angle = cv2.cartToPolar(flow_np[0], flow_np[1], angleInDegrees=True)
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(
        magnitude / max(float(magnitude_scale), 1e-6) * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[~valid_np] = 0
    return Image.fromarray(rgb)


def render_flow_panel(
    rgb: torch.Tensor,
    flow_pred: torch.Tensor,
    flow_target: torch.Tensor,
    flow_valid: torch.Tensor,
    flow_pair_mask: torch.Tensor,
    *,
    max_pairs: int = 6,
) -> Image.Image | None:
    pair_indices = [
        index
        for index in range(min(int(flow_pred.shape[0]), int(flow_pair_mask.shape[0])))
        if bool(flow_pair_mask[index].item()) and bool(flow_valid[index].any().item())
    ][:max_pairs]
    if not pair_indices:
        return None
    height, width = [int(value) for value in flow_pred.shape[-2:]]
    label_height = 20
    columns = ("source", "target", "flow target", "flow pred", "endpoint error")
    panel = Image.new("RGB", (width * len(columns), (height + label_height) * len(pair_indices)), "black")
    draw = ImageDraw.Draw(panel)
    for row, pair_index in enumerate(pair_indices):
        valid = flow_valid[pair_index].to(dtype=torch.bool)
        target = flow_target[pair_index]
        pred = flow_pred[pair_index]
        magnitudes = torch.cat(
            [target.square().sum(dim=0).sqrt()[valid], pred.square().sum(dim=0).sqrt()[valid]]
        )
        magnitude_scale = float(torch.quantile(magnitudes, 0.95).item()) if magnitudes.numel() else 1.0
        error = (pred - target).square().sum(dim=0).sqrt()
        error_values = error[valid]
        error_scale = float(torch.quantile(error_values, 0.95).item()) if error_values.numel() else 1.0
        error_normalized = (error / max(error_scale, 1e-6)).clamp(0.0, 1.0).numpy()
        error_rgb = (plt.get_cmap("inferno")(error_normalized)[..., :3] * 255.0).astype(np.uint8)
        error_rgb[~valid.numpy()] = 0
        images = (
            tensor_rgb_to_image(rgb[pair_index]).resize((width, height)),
            tensor_rgb_to_image(rgb[pair_index + 1]).resize((width, height)),
            _flow_to_rgb(target, valid, magnitude_scale=magnitude_scale),
            _flow_to_rgb(pred, valid, magnitude_scale=magnitude_scale),
            Image.fromarray(error_rgb),
        )
        y = row * (height + label_height)
        for column, (label, image) in enumerate(zip(columns, images, strict=True)):
            x = column * width
            panel.paste(image, (x, y + label_height))
            draw.text((x + 4, y + 3), f"{label} t={pair_index}", fill="white")
    return panel


def _candidate_flow_panels(batch: dict, outputs: dict, candidate: SequenceCandidate) -> dict[str, Image.Image]:
    flow_pred = outputs.get("flow_pred")
    flow_target = batch.get("flow_pseudo_target")
    flow_valid = batch.get("flow_pseudo_valid")
    flow_pair_mask = batch.get("flow_pair_mask")
    if not all(isinstance(value, torch.Tensor) for value in (flow_pred, flow_target, flow_valid, flow_pair_mask)):
        return {}
    batch_index = candidate.batch_index
    image = render_flow_panel(
        batch["images"][batch_index].detach().cpu(),
        flow_pred[batch_index].detach().cpu().to(dtype=torch.float32),
        flow_target[batch_index].detach().cpu().to(dtype=torch.float32),
        flow_valid[batch_index].detach().cpu().to(dtype=torch.bool),
        flow_pair_mask[batch_index].detach().cpu().to(dtype=torch.bool),
    )
    return {} if image is None else {"flow_gt_pred_epe": image}


def _candidate_panels(batch: dict, outputs: dict, candidate: SequenceCandidate) -> dict[str, Image.Image]:
    panels: dict[str, Image.Image] = {}
    panels.update(_candidate_marker_panels(batch, outputs, candidate))
    panels.update(_candidate_depth_panels(batch, outputs, candidate))
    panels.update(_candidate_camera_panels(batch, outputs, candidate))
    panels.update(_candidate_flow_panels(batch, outputs, candidate))
    return panels


def run_training_visualization(
    *,
    stream_batches_by_source: dict[str, dict[str, dict]],
    forward_batch: Callable[[str, dict], dict],
    writer,
    output_dir: str | Path,
    step: int,
    sequences_per_source: int,
) -> int:
    total_written = 0
    for source_name, stream_batches in stream_batches_by_source.items():
        candidates = select_sequence_candidates(stream_batches, limit=sequences_per_source)
        outputs_by_stream: dict[str, dict] = {}
        for sequence_index, candidate in enumerate(candidates):
            batch = stream_batches[candidate.stream_name]
            outputs = outputs_by_stream.get(candidate.stream_name)
            if outputs is None:
                outputs = forward_batch(candidate.stream_name, batch)
                outputs_by_stream[candidate.stream_name] = outputs
            panels = _candidate_panels(batch, outputs, candidate)
            if not panels:
                continue
            total_written += write_visualization_panels(
                output_dir=output_dir,
                writer=writer,
                source_name=source_name,
                step=step,
                sequence_name=_sequence_output_name(sequence_index, candidate),
                panels=panels,
            )
    if total_written == 0:
        logger.info("本次训练可视化没有可写出的面板: step={}", step)
    return total_written


def project_points(points_xyz: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = points_xyz.detach().cpu().to(dtype=torch.float32)
    camera_intrinsics = intrinsics.detach().cpu().to(dtype=torch.float32)
    z = points[..., 2]
    valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(camera_intrinsics).all() & (z > 1e-6)
    safe_z = z.clamp_min(1e-6)
    u = camera_intrinsics[0, 0] * (points[..., 0] / safe_z) + camera_intrinsics[0, 2]
    v = camera_intrinsics[1, 1] * (points[..., 1] / safe_z) + camera_intrinsics[1, 2]
    return torch.stack([u, v], dim=-1), valid


def _draw_points(
    draw: ImageDraw.ImageDraw,
    points_uv: torch.Tensor,
    valid: torch.Tensor,
    *,
    color: tuple[int, int, int, int],
    radius: int,
) -> None:
    for point in points_uv[valid]:
        x, y = [float(value) for value in point.tolist()]
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def _draw_joints(
    draw: ImageDraw.ImageDraw,
    joints_uv: torch.Tensor,
    valid: torch.Tensor,
    *,
    color: tuple[int, int, int, int],
) -> None:
    for start, end in MANO_JOINT_EDGES:
        if start >= joints_uv.shape[0] or end >= joints_uv.shape[0]:
            continue
        if not bool((valid[start] & valid[end]).item()):
            continue
        x0, y0 = [float(value) for value in joints_uv[start].tolist()]
        x1, y1 = [float(value) for value in joints_uv[end].tolist()]
        draw.line([x0, y0, x1, y1], fill=color, width=2)
    _draw_points(draw, joints_uv, valid, color=color, radius=2)


def _mesh_edges_from_faces(faces: torch.Tensor) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces.detach().cpu().to(dtype=torch.long).tolist():
        if len(face) != 3:
            continue
        a, b, c = [int(value) for value in face]
        for start, end in ((a, b), (b, c), (c, a)):
            if start == end:
                continue
            edges.add((min(start, end), max(start, end)))
    return sorted(edges)


def _point_inside(point: np.ndarray, width: int, height: int) -> bool:
    return bool(0.0 <= float(point[0]) < float(width) and 0.0 <= float(point[1]) < float(height))


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(color[2]), int(color[1]), int(color[0])


def _marker_point_rgb(
    vertex_index: int,
    vertex_visibility: torch.Tensor | None,
    vertex_visibility_mask: torch.Tensor | None,
) -> tuple[int, int, int]:
    if vertex_visibility_mask is not None and not bool(vertex_visibility_mask[vertex_index].item()):
        return MARKER_UNSUPERVISED_RGB
    if vertex_visibility is None:
        return MARKER_VISIBLE_RGB
    return MARKER_VISIBLE_RGB if float(vertex_visibility[vertex_index].item()) >= 0.5 else MARKER_HIDDEN_RGB


def _marker_edge_rgb(
    start_idx: int,
    end_idx: int,
    vertex_visibility: torch.Tensor | None,
    vertex_visibility_mask: torch.Tensor | None,
) -> tuple[int, int, int]:
    start_color = _marker_point_rgb(start_idx, vertex_visibility, vertex_visibility_mask)
    end_color = _marker_point_rgb(end_idx, vertex_visibility, vertex_visibility_mask)
    if start_color == end_color:
        return start_color
    return MARKER_MIXED_RGB


def _draw_marker_mesh(
    image: Image.Image,
    points_uv: torch.Tensor,
    valid: torch.Tensor,
    faces: torch.Tensor,
    vertex_visibility: torch.Tensor | None,
    vertex_visibility_mask: torch.Tensor | None,
    *,
    radius: int,
    thickness: int,
) -> Image.Image:
    canvas = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    height, width = canvas.shape[:2]
    points = points_uv.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    for start_idx, end_idx in _mesh_edges_from_faces(faces):
        if start_idx >= len(points) or end_idx >= len(points):
            continue
        if not (valid_np[start_idx] and valid_np[end_idx]):
            continue
        if not (_point_inside(points[start_idx], width, height) and _point_inside(points[end_idx], width, height)):
            continue
        cv2.line(
            canvas,
            tuple(int(round(float(value))) for value in points[start_idx]),
            tuple(int(round(float(value))) for value in points[end_idx]),
            _rgb_to_bgr(_marker_edge_rgb(start_idx, end_idx, vertex_visibility, vertex_visibility_mask)),
            thickness,
            lineType=cv2.LINE_AA,
        )
    for vertex_index, point in enumerate(points):
        if not valid_np[vertex_index] or not _point_inside(point, width, height):
            continue
        cv2.circle(
            canvas,
            tuple(int(round(float(value))) for value in point),
            radius,
            _rgb_to_bgr(_marker_point_rgb(vertex_index, vertex_visibility, vertex_visibility_mask)),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return Image.fromarray(canvas[:, :, ::-1].copy())


def render_marker_strip(
    rgb: torch.Tensor,
    intrinsics: torch.Tensor,
    vertices_xyz: torch.Tensor | None,
    joints_xyz: torch.Tensor | None,
    *,
    title: str,
    marker_faces: torch.Tensor | None = None,
    vertex_visibility: torch.Tensor | None = None,
    vertex_visibility_mask: torch.Tensor | None = None,
    hand_presence_mask: torch.Tensor | None = None,
) -> Image.Image:
    del title
    if rgb.ndim != 4:
        raise ValueError("rgb must have shape (frames, channels, height, width)")
    frame_count, _, height, width = rgb.shape
    hand_presence = None
    if hand_presence_mask is not None:
        hand_presence = hand_presence_mask.detach().cpu().to(dtype=torch.bool)
        if hand_presence.ndim != 2 or hand_presence.shape[0] != frame_count:
            raise ValueError("hand_presence_mask must have shape (frames, hands)")
    strip = Image.new("RGB", (width * frame_count, height))
    for frame_index in range(frame_count):
        frame = tensor_rgb_to_image(rgb[frame_index]).convert("RGB")
        frame_intrinsics = intrinsics[frame_index]
        frame_presence = None if hand_presence is None else hand_presence[frame_index]
        if vertices_xyz is not None:
            vertices_per_hand = int(vertices_xyz.shape[2])
            faces = marker_faces
            if faces is None:
                faces = marker_faces_for_count(vertices_per_hand)
            for hand_index in range(vertices_xyz.shape[1]):
                if frame_presence is not None and not bool(frame_presence[hand_index].item()):
                    continue
                points_uv, valid = project_points(vertices_xyz[frame_index, hand_index], frame_intrinsics)
                hand_vertex_visibility = None if vertex_visibility is None else vertex_visibility[frame_index, hand_index]
                hand_vertex_visibility_mask = None if vertex_visibility_mask is None else vertex_visibility_mask[frame_index, hand_index]
                frame = _draw_marker_mesh(
                    frame,
                    points_uv,
                    valid,
                    faces,
                    hand_vertex_visibility,
                    hand_vertex_visibility_mask,
                    radius=1,
                    thickness=1,
                )
        if joints_xyz is not None:
            frame = frame.convert("RGBA")
            overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay, "RGBA")
            for hand_index in range(joints_xyz.shape[1]):
                if frame_presence is not None and not bool(frame_presence[hand_index].item()):
                    continue
                color = JOINT_COLORS[hand_index % len(JOINT_COLORS)]
                joints_uv, valid = project_points(joints_xyz[frame_index, hand_index], frame_intrinsics)
                _draw_joints(draw, joints_uv, valid, color=color)
            frame = Image.alpha_composite(frame, overlay).convert("RGB")
        strip.paste(frame, (frame_index * width, 0))
    return strip


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().to(dtype=torch.float32).numpy()


def render_depth_panel(
    gt_depth: torch.Tensor,
    pred_depth: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> Image.Image:
    if gt_depth.shape != pred_depth.shape:
        raise ValueError("gt_depth and pred_depth must have matching shapes")
    gt_np = _tensor_to_numpy(gt_depth)
    pred_np = _tensor_to_numpy(pred_depth)
    valid_np = np.isfinite(gt_np) & np.isfinite(pred_np)
    if valid_mask is not None:
        valid_np &= valid_mask.detach().cpu().numpy().astype(bool)
    diff_np = np.abs(gt_np - pred_np)
    gt_plot = np.where(valid_np, gt_np, np.nan)
    pred_plot = np.where(valid_np, pred_np, np.nan)
    diff_plot = np.where(valid_np, diff_np, np.nan)
    depth_values = np.concatenate([gt_plot[np.isfinite(gt_plot)], pred_plot[np.isfinite(pred_plot)]])
    depth_min = float(np.nanmin(depth_values)) if depth_values.size else 0.0
    depth_max = float(np.nanmax(depth_values)) if depth_values.size else 1.0
    if depth_max <= depth_min:
        depth_max = depth_min + 1.0
    diff_values = diff_plot[np.isfinite(diff_plot)]
    diff_max = float(np.nanmax(diff_values)) if diff_values.size else 1.0
    if diff_max <= 0.0:
        diff_max = 1.0

    frame_count = int(gt_depth.shape[0])
    fig, axes = plt.subplots(frame_count, 3, figsize=(7.5, max(2.0, 2.0 * frame_count)), squeeze=False)
    for frame_index in range(frame_count):
        panels: tuple[tuple[str, np.ndarray, str, dict[str, float]], ...] = (
            ("GT depth", gt_plot[frame_index], "viridis", {"vmin": depth_min, "vmax": depth_max}),
            ("Pred depth", pred_plot[frame_index], "viridis", {"vmin": depth_min, "vmax": depth_max}),
            ("Abs diff", diff_plot[frame_index], "magma", {"vmin": 0.0, "vmax": diff_max}),
        )
        for axis, (name, values, color_map, limits) in zip(axes[frame_index], panels, strict=True):
            axis.imshow(values, cmap=color_map, **limits)
            axis.set_title(f"f{frame_index} {name}", fontsize=8)
            axis.axis("off")
    fig.tight_layout(pad=0.3)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")


def relative_pose_to_first_frame(pose: torch.Tensor) -> torch.Tensor:
    pose = pose.to(dtype=torch.float32)
    return torch.matmul(pose, torch.linalg.inv(pose[:, :1]))


def render_camera_pose_panel(
    gt_pose: torch.Tensor,
    pred_pose: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> Image.Image | None:
    if gt_pose.shape != pred_pose.shape or gt_pose.ndim != 3 or gt_pose.shape[-2:] != (4, 4):
        raise ValueError("gt_pose and pred_pose must have shape (frames, 4, 4)")
    valid = torch.isfinite(gt_pose).all(dim=(-2, -1)) & torch.isfinite(pred_pose).all(dim=(-2, -1))
    if valid_mask is not None:
        valid = valid & valid_mask.detach().cpu().to(dtype=torch.bool)
    if not bool(valid.any().item()):
        return None

    gt_relative = relative_pose_to_first_frame(gt_pose.unsqueeze(0))[0].detach().cpu()
    pred_relative = relative_pose_to_first_frame(pred_pose.unsqueeze(0))[0].detach().cpu()
    gt_centers = gt_relative[:, :3, 3]
    pred_centers = pred_relative[:, :3, 3]
    valid_indices = torch.nonzero(valid.detach().cpu(), as_tuple=False).flatten()

    fig = plt.figure(figsize=(5.5, 4.5))
    axis = fig.add_subplot(111, projection="3d")
    gt_points = gt_centers[valid_indices].numpy()
    pred_points = pred_centers[valid_indices].numpy()
    axis.plot(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], color="#00a6d6", marker="o", label="GT")
    axis.plot(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], color="#f97316", marker="o", label="Pred")

    def _draw_orientation(pose: torch.Tensor, indices: torch.Tensor, base_color: str) -> None:
        for index in indices.tolist():
            center = pose[index, :3, 3].numpy()
            forward = pose[index, :3, 2].numpy()
            up = pose[index, :3, 1].numpy()
            axis.quiver(center[0], center[1], center[2], forward[0], forward[1], forward[2], length=0.08, color=base_color)
            axis.quiver(center[0], center[1], center[2], up[0], up[1], up[2], length=0.05, color=base_color, alpha=0.5)

    _draw_orientation(gt_relative, valid_indices, "#00a6d6")
    _draw_orientation(pred_relative, valid_indices, "#f97316")
    all_points = np.concatenate([gt_points, pred_points], axis=0)
    center = all_points.mean(axis=0)
    radius = max(float(np.abs(all_points - center).max()), 0.1)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.legend(loc="upper left")
    axis.set_title("Camera pose relative to first frame")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")
