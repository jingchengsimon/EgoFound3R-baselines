from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import smplx
import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, quaternion_to_matrix
from smplx.vertex_ids import vertex_ids
from torch.utils.data import ConcatDataset

from egohandmetric_prompt.data.base import TemporalChunkDataset
from egohandmetric_prompt.data.contact import lazy_contact_targets_from_scene
from egohandmetric_prompt.data.marker_mesh import marker_faces_for_count
from egohandmetric_prompt.data.marker_vertices import MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195
from egohandmetric_prompt.data.scene_accel import (
    ObjectMeshAccelCache,
    ObjectMeshRawCache,
    SceneObjectAccel,
    SceneObjectMesh,
    SceneObjectPart,
)
from egohandmetric_prompt.data.stages import (
    MIDTRAIN_DATASET_NAMES,
    POSTTRAIN_MARKER_DATASET_NAMES,
    POSTTRAIN_3R_DATASET_NAMES,
    build_named_frame_dataset,
)
from egohandmetric_prompt.configs import default_human_model_root_path


POSTTRAIN_MARKER_BASE_DATASET_NAMES = {"h2o"}
POSTTRAIN_3R_BASE_DATASET_NAMES = {"h2o"}
HAND_MARKER_ONLY_DATASET_NAMES: set[str] = set()
DEFAULT_HUMAN_MODEL_ROOT = default_human_model_root_path()
VERTEX_VISIBILITY_RASTER_MAX_SIZE = 64
SCENE_VISIBILITY_RASTER_MAX_FACES_PER_BIN = 200_000
# PyTorch3D's CUDA rasterizer can issue an illegal-memory-access error for a
# 60-frame Meshes batch even when every individual mesh is finite and indexed
# correctly.  Bound the batch only at the rasterization boundary; the per-frame
# z-buffer semantics and returned order remain unchanged.
SCENE_VISIBILITY_RASTER_MAX_BATCH_SIZE = 8
SCENE_VISIBILITY_OBJECT_RASTER_MAX_FACES = 100_000
SCENE_VISIBILITY_MARKER_DEPTH_ATOL = 1e-4
SCENE_VISIBILITY_MARKER_DEPTH_RTOL = 2e-3
SCENE_VISIBILITY_RAY_EPSILON = 1e-6
SCENE_VISIBILITY_RAY_ENDPOINT_EPSILON = 1e-4
SCENE_VISIBILITY_RAY_DEDUP_TOLERANCE = 1e-4
SCENE_VISIBILITY_MARKER_DEPTH_OCCLUSION_THRESHOLD_METERS = 0.05
SCENE_VISIBILITY_JOINT_DEPTH_OCCLUSION_THRESHOLD_METERS = 0.07
SCENE_VISIBILITY_RAY_FACE_BLOCK_SIZE = 2048
MANO_EXTRA_JOINT_VERTEX_IDS = tuple(vertex_ids["mano"].values())
MANO_OPENPOSE_JOINT_ORDER = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
H2O_MARKER_CONTACT_COUNT = len(MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195)


def union_contact_targets(
    first_targets: torch.Tensor,
    first_mask: torch.Tensor,
    second_targets: torch.Tensor,
    second_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (
        first_targets.shape == first_mask.shape == second_targets.shape == second_mask.shape
    ):
        raise ValueError("contact union 的 targets/masks 形状必须一致。")
    positive = (first_mask & (first_targets > 0.5)) | (second_mask & (second_targets > 0.5))
    supervision_mask = positive | (first_mask & second_mask)
    return positive.to(dtype=first_targets.dtype), supervision_mask


def _contact_supervision_mode(
    sample: dict[str, Any],
    modes_by_dataset: dict[str, str],
) -> str:
    dataset_name = str(sample.get("dataset_name", ""))
    base_dataset_name = str(sample.get("base_dataset_name", ""))
    object_and_interhand_datasets = {"h2o", "hot3d_aria", "hot3d", "hoi4d", "oakink_v2", "egoforce_arctic", "taco"}
    if dataset_name in object_and_interhand_datasets or base_dataset_name in object_and_interhand_datasets:
        return "object_and_interhand"
    if dataset_name == "reinterhand" or base_dataset_name == "reinterhand":
        return "interhand_only"
    if dataset_name in modes_by_dataset:
        return modes_by_dataset[dataset_name]
    if base_dataset_name in modes_by_dataset:
        return modes_by_dataset[base_dataset_name]
    return "disabled"


def _enable_legacy_chumpy_compatibility() -> None:
    numpy_aliases = {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "unicode": np.str_,
        "str": np.str_,
    }
    for name, value in numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def _matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    rotation_tensor = torch.as_tensor(rotation, dtype=torch.float32).reshape(1, 3, 3)
    return matrix_to_axis_angle(rotation_tensor)[0].detach().cpu().numpy().astype(np.float32)


def _quat_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quat_tensor = torch.as_tensor(quaternion_wxyz, dtype=torch.float32).reshape(1, 4)
    quat_norm = torch.linalg.norm(quat_tensor, dim=-1, keepdim=True).clamp_min(1e-8)
    return quaternion_to_matrix(quat_tensor / quat_norm)[0].detach().cpu().numpy().astype(np.float32)


def _quat16_to_axis_angle_full(quat16: torch.Tensor | np.ndarray) -> np.ndarray:
    quat_array = np.asarray(quat16, dtype=np.float32).reshape(16, 4)
    rotvecs = [_matrix_to_axis_angle(_quat_to_matrix(quaternion)) for quaternion in quat_array]
    return np.concatenate(rotvecs, axis=0)


def _rotation_mats_to_axis_angle_full(global_orient: torch.Tensor | np.ndarray, hand_pose: torch.Tensor | np.ndarray) -> np.ndarray:
    global_orient_np = np.asarray(global_orient, dtype=np.float32).reshape(3, 3)
    hand_pose_np = np.asarray(hand_pose, dtype=np.float32).reshape(-1, 3, 3)
    rotvecs = [_matrix_to_axis_angle(global_orient_np)]
    rotvecs.extend(_matrix_to_axis_angle(matrix) for matrix in hand_pose_np)
    return np.concatenate(rotvecs, axis=0)


def _optional_finite_tensor(value: Any, shape: tuple[int, ...]) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32)
    expected_numel = int(np.prod(shape))
    if tensor.numel() != expected_numel:
        return None
    tensor = tensor.reshape(shape)
    if not torch.isfinite(tensor).all().item():
        return None
    return tensor


def _mano_output_joints(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return output["joints"]
    return output.joints


def combine_rt(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    batch_size = rotation.shape[0]
    transform = torch.eye(4, dtype=rotation.dtype, device=rotation.device).unsqueeze(0).repeat(batch_size, 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    return transform


def eliminate_external_matrix_mano(
    R: np.ndarray | torch.Tensor,
    T: np.ndarray | torch.Tensor,
    mano_model: torch.nn.Module,
    hand_pose: np.ndarray | torch.Tensor,
    global_orient: np.ndarray | torch.Tensor | None = None,
    transl: np.ndarray | torch.Tensor | None = None,
    betas: np.ndarray | torch.Tensor | None = None,
    transl_is_root_position: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    hand_pose = torch.as_tensor(hand_pose, dtype=torch.float32)
    batch_size = hand_pose.shape[0]
    if getattr(mano_model, "batch_size", batch_size) != batch_size:
        mano_model.batch_size = batch_size
    device = hand_pose.device
    R = torch.as_tensor(R, dtype=hand_pose.dtype, device=device)
    T = torch.as_tensor(T, dtype=hand_pose.dtype, device=device)
    if R.ndim == 2:
        R = R.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if T.ndim == 1:
        T = T.unsqueeze(0).expand(batch_size, -1).clone()
    if global_orient is None:
        global_orient = torch.zeros(batch_size, 3, dtype=hand_pose.dtype, device=device)
    else:
        global_orient = torch.as_tensor(global_orient, dtype=hand_pose.dtype, device=device)
    if transl is None:
        transl = torch.zeros(batch_size, 3, dtype=hand_pose.dtype, device=device)
    else:
        transl = torch.as_tensor(transl, dtype=hand_pose.dtype, device=device)
    betas = None if betas is None else torch.as_tensor(betas, dtype=hand_pose.dtype, device=device)
    mano_input = {"return_verts": True, **kwargs}
    global_orient_cam = matrix_to_axis_angle(torch.bmm(R, axis_angle_to_matrix(global_orient)))
    if transl_is_root_position:
        transl_root = transl
    else:
        mano_output = mano_model(
            global_orient=global_orient,
            transl=transl,
            hand_pose=hand_pose,
            betas=betas,
            **mano_input,
        )
        transl_root = _mano_output_joints(mano_output)[:, 0]
    transl_homo = torch.cat([transl_root, torch.ones_like(transl[:, :1])], dim=1)
    transl_homo = torch.bmm(combine_rt(R, T), transl_homo[..., None])
    transl_cam = transl_homo[:, :3, 0] / transl_homo[:, 3:4, 0]
    params_cam = {
        "betas": betas,
        "global_orient": global_orient_cam,
        "hand_pose": hand_pose,
    }
    mano_output = mano_model(**params_cam, **mano_input)
    pelvis_shift = _mano_output_joints(mano_output)[:, 0]
    transl_cam = transl_cam - pelvis_shift
    return global_orient_cam, transl_cam


def normalize_hand_target(hand: dict[str, Any] | None) -> dict[str, Any] | None:
    if hand is None:
        return None

    extras = hand.get("extras", {})
    normalized = {
        "side": hand.get("side"),
        "visible": bool(hand.get("visible", True)),
        "bbox_xyxy": hand.get("bbox_xyxy"),
        "joints_3d": hand.get("joints_3d"),
        "joints_2d": hand.get("joints_2d"),
        "mano_betas": hand.get("mano_betas"),
        "mano_trans": hand.get("mano_trans"),
        "mano_pose_axis_angle": None,
        "mano_pose_pca": None,
        "mano_global_orient_axis_angle": None,
        "mano_flat_hand_mean": bool(extras.get("mano_flat_hand_mean", hand.get("mano_flat_hand_mean", True))),
        "mano_flip_left_shapedirs": bool(extras.get("mano_flip_left_shapedirs", hand.get("mano_flip_left_shapedirs", False))),
        "mano_use_right_hand_layer": bool(
            extras.get("mano_use_right_hand_layer", hand.get("mano_use_right_hand_layer", False))
        ),
        "mano_mirror_local_x": bool(extras.get("mano_mirror_local_x", hand.get("mano_mirror_local_x", False))),
        "mano_trans_is_root_position": bool(
            extras.get("mano_trans_is_root_position", hand.get("mano_trans_is_root_position", False))
        ),
        "mano_align_root_to_joints_3d": bool(
            extras.get("mano_align_root_to_joints_3d", hand.get("mano_align_root_to_joints_3d", False))
        ),
        "mano_pose_format": hand.get("mano_pose_format"),
        "marker_vertices": None,
        "marker_valid": False,
        "mano_valid": False,
        "joints_valid": hand.get("joints_3d") is not None,
        "extras": extras,
    }

    pose_format = hand.get("mano_pose_format")
    mano_pose = hand.get("mano_pose")
    if pose_format == "axis_angle_full" and mano_pose is not None:
        axis_angle = _optional_finite_tensor(mano_pose, (48,))
        if axis_angle is not None:
            normalized["mano_pose_axis_angle"] = axis_angle
            normalized["mano_global_orient_axis_angle"] = axis_angle[:3]
            normalized["mano_valid"] = True
    elif pose_format == "axis_angle_pose45":
        global_orient = _optional_finite_tensor(hand.get("mano_global_orient"), (3,))
        local_pose = _optional_finite_tensor(hand.get("mano_hand_pose"), (45,))
        if global_orient is not None and local_pose is not None:
            axis_angle = torch.cat([global_orient, local_pose])
            normalized["mano_pose_axis_angle"] = axis_angle
            normalized["mano_global_orient_axis_angle"] = axis_angle[:3]
            normalized["mano_valid"] = True
    elif pose_format == "rotation_matrix":
        global_orient = _optional_finite_tensor(hand.get("mano_global_orient"), (3, 3))
        local_pose = _optional_finite_tensor(hand.get("mano_hand_pose"), (15, 3, 3))
        if global_orient is not None and local_pose is not None:
            axis_angle = _rotation_mats_to_axis_angle_full(global_orient, local_pose)
            normalized["mano_pose_axis_angle"] = torch.from_numpy(axis_angle)
            normalized["mano_global_orient_axis_angle"] = normalized["mano_pose_axis_angle"][:3]
            normalized["mano_valid"] = True
    elif pose_format == "quat16" and mano_pose is not None:
        quat16 = _optional_finite_tensor(mano_pose, (16, 4))
        if quat16 is not None:
            axis_angle = _quat16_to_axis_angle_full(quat16)
            normalized["mano_pose_axis_angle"] = torch.from_numpy(axis_angle)
            normalized["mano_global_orient_axis_angle"] = normalized["mano_pose_axis_angle"][:3]
            normalized["mano_valid"] = True
    elif pose_format in {"mano_pca", "hot3d_mano_pca"} and mano_pose is not None:
        pca_pose = _optional_finite_tensor(mano_pose, (15,))
        global_orient = _optional_finite_tensor(hand.get("mano_global_orient"), (3,))
        if pca_pose is not None and global_orient is not None:
            normalized["mano_pose_pca"] = pca_pose
            normalized["mano_global_orient_axis_angle"] = global_orient
            normalized["mano_valid"] = True

    if normalized["bbox_xyxy"] is not None:
        normalized["bbox_xyxy"] = torch.as_tensor(normalized["bbox_xyxy"], dtype=torch.float32)
    if normalized["joints_3d"] is not None:
        normalized["joints_3d"] = torch.as_tensor(normalized["joints_3d"], dtype=torch.float32)
    if normalized["joints_2d"] is not None:
        normalized["joints_2d"] = torch.as_tensor(normalized["joints_2d"], dtype=torch.float32)
    if normalized["mano_betas"] is not None:
        normalized["mano_betas"] = _optional_finite_tensor(normalized["mano_betas"], (10,))
    if normalized["mano_trans"] is not None:
        normalized["mano_trans"] = _optional_finite_tensor(normalized["mano_trans"], (3,))
    return normalized


class ManoVertexBuilder:
    def __init__(
        self,
        model_root: str | Path = DEFAULT_HUMAN_MODEL_ROOT,
        *,
        device: str = "cpu",
        axis_angle_flat_hand_mean: bool = True,
        pca_flat_hand_mean: bool = False,
        pca_components: int = 15,
    ) -> None:
        _enable_legacy_chumpy_compatibility()
        model_root = Path(model_root)
        mano_root = model_root / "mano"
        if not (mano_root / "MANO_LEFT.pkl").exists() or not (mano_root / "MANO_RIGHT.pkl").exists():
            raise FileNotFoundError(f"缺少 MANO 模型文件：{mano_root}")
        self.device = torch.device(device)
        self.default_axis_angle_flat_hand_mean = axis_angle_flat_hand_mean
        self.axis_angle_layers_by_flat_hand_mean = {
            flat_hand_mean: {
                "left": smplx.create(
                    str(model_root),
                    model_type="mano",
                    is_rhand=False,
                    use_pca=False,
                    flat_hand_mean=flat_hand_mean,
                    num_pca_comps=pca_components,
                ).to(self.device),
                "right": smplx.create(
                    str(model_root),
                    model_type="mano",
                    is_rhand=True,
                    use_pca=False,
                    flat_hand_mean=flat_hand_mean,
                    num_pca_comps=pca_components,
                ).to(self.device),
            }
            for flat_hand_mean in (True, False)
        }
        self.axis_angle_layers = self.axis_angle_layers_by_flat_hand_mean[axis_angle_flat_hand_mean]
        self.axis_angle_left_shapedirs_layers_by_flat_hand_mean = {
            flat_hand_mean: smplx.create(
                str(model_root),
                model_type="mano",
                is_rhand=False,
                use_pca=False,
                flat_hand_mean=flat_hand_mean,
                num_pca_comps=pca_components,
            ).to(self.device)
            for flat_hand_mean in (True, False)
        }
        for layer in self.axis_angle_left_shapedirs_layers_by_flat_hand_mean.values():
            layer.shapedirs[:, 0, :] *= -1
        self.pca_layers = {
            "left": smplx.create(
                str(model_root),
                model_type="mano",
                is_rhand=False,
                use_pca=True,
                flat_hand_mean=pca_flat_hand_mean,
                num_pca_comps=pca_components,
            ).to(self.device),
            "right": smplx.create(
                str(model_root),
                model_type="mano",
                is_rhand=True,
                use_pca=True,
                flat_hand_mean=pca_flat_hand_mean,
                num_pca_comps=pca_components,
            ).to(self.device),
        }
        self.pca_left_shapedirs_layer = smplx.create(
            str(model_root),
            model_type="mano",
            is_rhand=False,
            use_pca=True,
            flat_hand_mean=pca_flat_hand_mean,
            num_pca_comps=pca_components,
        ).to(self.device)
        self.pca_left_shapedirs_layer.shapedirs[:, 0, :] *= -1
        axis_angle_layers = [
            layer
            for layers_by_side in self.axis_angle_layers_by_flat_hand_mean.values()
            for layer in layers_by_side.values()
        ]
        flipped_left_layers = [
            *self.axis_angle_left_shapedirs_layers_by_flat_hand_mean.values(),
            self.pca_left_shapedirs_layer,
        ]
        for layer in [*axis_angle_layers, *self.pca_layers.values(), *flipped_left_layers]:
            layer.eval()
            for parameter in layer.parameters():
                parameter.requires_grad = False
        self.faces = torch.as_tensor(self.axis_angle_layers["right"].faces.astype(np.int64))
        self.extra_joint_vertex_ids = torch.as_tensor(MANO_EXTRA_JOINT_VERTEX_IDS, dtype=torch.long)
        self.openpose_joint_order = torch.as_tensor(MANO_OPENPOSE_JOINT_ORDER, dtype=torch.long)

    def _axis_angle_layer(self, hand_target: dict[str, Any]):
        side = hand_target["side"]
        layer_side = "right" if bool(hand_target.get("mano_use_right_hand_layer", False)) else side
        flat_hand_mean = bool(hand_target.get("mano_flat_hand_mean", self.default_axis_angle_flat_hand_mean))
        flip_left_shapedirs = layer_side == "left" and bool(hand_target.get("mano_flip_left_shapedirs", False))
        if flip_left_shapedirs:
            return self.axis_angle_left_shapedirs_layers_by_flat_hand_mean[flat_hand_mean]
        return self.axis_angle_layers_by_flat_hand_mean[flat_hand_mean][layer_side]

    def prepare_hand_target(self, hand_target: dict[str, Any]) -> dict[str, Any]:
        extras = hand_target.get("extras", {})
        transform = extras.get("mano_external_transform")
        if transform is None or hand_target.get("mano_pose_axis_angle") is None:
            return hand_target
        pose = hand_target["mano_pose_axis_angle"].to(self.device).reshape(48)
        betas = hand_target.get("mano_betas")
        transl = hand_target.get("mano_trans")
        if betas is None or transl is None:
            return hand_target
        transform_tensor = torch.as_tensor(transform, dtype=torch.float32, device=self.device).reshape(4, 4)
        global_orient_cam, transl_cam = eliminate_external_matrix_mano(
            transform_tensor[:3, :3].unsqueeze(0),
            transform_tensor[:3, 3].unsqueeze(0),
            self._axis_angle_layer(hand_target),
            pose[3:].reshape(1, 45),
            global_orient=pose[:3].reshape(1, 3),
            transl=transl.to(self.device).reshape(1, 3),
            betas=betas.to(self.device).reshape(1, 10),
            transl_is_root_position=bool(hand_target.get("mano_trans_is_root_position", False)),
        )
        prepared = dict(hand_target)
        prepared_extras = dict(extras)
        prepared_extras.pop("mano_external_transform", None)
        prepared_extras.pop("mano_trans_is_root_position", None)
        prepared["extras"] = prepared_extras
        prepared["mano_trans_is_root_position"] = False
        prepared["mano_pose_axis_angle"] = torch.cat([global_orient_cam[0].cpu(), pose[3:].cpu()]).to(dtype=torch.float32)
        prepared["mano_global_orient_axis_angle"] = global_orient_cam[0].detach().cpu().to(dtype=torch.float32)
        prepared["mano_trans"] = transl_cam[0].detach().cpu().to(dtype=torch.float32)
        return prepared

    def _openpose_joints(self, vertices: torch.Tensor, mano_joints: torch.Tensor) -> torch.Tensor:
        fingertip_joints = vertices[self.extra_joint_vertex_ids]
        joints = torch.cat([mano_joints, fingertip_joints], dim=0)
        return joints[self.openpose_joint_order]

    def build_vertices(
        self,
        hand_target: dict[str, Any],
        *,
        marker_vertex_ids: Sequence[int] | None = None,
    ) -> torch.Tensor | None:
        vertices, _ = self.build_vertices_and_joints(hand_target, marker_vertex_ids=marker_vertex_ids)
        return vertices

    def build_vertices_and_joints(
        self,
        hand_target: dict[str, Any],
        *,
        marker_vertex_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        hand_target = self.prepare_hand_target(hand_target)
        if not hand_target["mano_valid"]:
            return None, None
        side = hand_target["side"]
        betas = hand_target["mano_betas"]
        transl = hand_target["mano_trans"]
        if betas is None or transl is None:
            return None, None
        betas_tensor = betas.to(self.device).reshape(1, 10)
        layer_side = "right" if bool(hand_target.get("mano_use_right_hand_layer", False)) else side
        flip_left_shapedirs = layer_side == "left" and bool(hand_target.get("mano_flip_left_shapedirs", False))
        mirror_local_x = bool(hand_target.get("mano_mirror_local_x", False))
        mano_trans_is_root_position = bool(hand_target.get("mano_trans_is_root_position", False))

        def _mirror_local_x(
            vertices: torch.Tensor,
            joints: torch.Tensor,
            transl_tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not mirror_local_x:
                return vertices, joints
            offset = transl_tensor.detach().cpu().reshape(1, 3)
            vertices = vertices.clone()
            joints = joints.clone()
            vertices = vertices - offset
            joints = joints - offset
            vertices[:, 0] *= -1
            joints[:, 0] *= -1
            vertices = vertices + offset
            joints = joints + offset
            return vertices, joints

        def _build_with_local_convention(
            build_fn,
            transl_tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            vertices_local, joints_local = build_fn(transl_tensor)
            return _mirror_local_x(vertices_local, joints_local, transl_tensor)

        def _maybe_root_position_transl(
            build_fn,
            transl_tensor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not mano_trans_is_root_position:
                return _build_with_local_convention(build_fn, transl_tensor)
            zero_transl = torch.zeros_like(transl_tensor)
            _, zero_joints = _build_with_local_convention(build_fn, zero_transl)
            corrected_transl = transl_tensor - zero_joints[0]
            return _build_with_local_convention(build_fn, corrected_transl)

        if hand_target["mano_pose_axis_angle"] is not None:
            pose = hand_target["mano_pose_axis_angle"].to(self.device)
            global_orient = pose[:3].reshape(1, 3)
            hand_pose = pose[3:].reshape(1, 45)
            flat_hand_mean = bool(hand_target.get("mano_flat_hand_mean", self.default_axis_angle_flat_hand_mean))
            layer = self._axis_angle_layer(hand_target)

            def _build_axis_angle(transl_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                with torch.no_grad():
                    output = layer(
                        betas=betas_tensor,
                        global_orient=global_orient,
                        hand_pose=hand_pose,
                        transl=transl_tensor.reshape(1, 3),
                        return_verts=True,
                    )
                vertices_local = output.vertices[0].detach().cpu()
                joints_local = output.joints[0].detach().cpu()
                return vertices_local, joints_local

            vertices, joints = _maybe_root_position_transl(
                _build_axis_angle,
                transl.to(self.device).reshape(3),
            )
        elif hand_target["mano_pose_pca"] is not None:
            layer = self.pca_left_shapedirs_layer if flip_left_shapedirs else self.pca_layers[layer_side]
            global_orient = hand_target["mano_global_orient_axis_angle"]
            if global_orient is None:
                global_orient = torch.zeros(3, dtype=torch.float32)

            def _build_pca(transl_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                with torch.no_grad():
                    output = layer(
                        betas=betas_tensor,
                        global_orient=global_orient.to(self.device).reshape(1, 3),
                        hand_pose=hand_target["mano_pose_pca"].to(self.device).reshape(1, -1),
                        transl=transl_tensor.reshape(1, 3),
                        return_verts=True,
                    )
                vertices_local = output.vertices[0].detach().cpu()
                joints_local = output.joints[0].detach().cpu()
                return vertices_local, joints_local

            vertices, joints = _maybe_root_position_transl(
                _build_pca,
                transl.to(self.device).reshape(3),
            )
        else:
            return None, None
        joints = self._openpose_joints(vertices, joints)
        post_mano_transform = hand_target.get("extras", {}).get("post_mano_transform")
        if post_mano_transform is not None:
            transform = torch.as_tensor(post_mano_transform, dtype=vertices.dtype)
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            vertices = (rotation @ vertices.T).T + translation
            joints = (rotation @ joints.T).T + translation
        if bool(hand_target.get("mano_align_root_to_joints_3d", False)) and hand_target.get("joints_3d") is not None:
            target_joints = hand_target["joints_3d"].to(dtype=joints.dtype)
            if target_joints.shape[0] > 0 and bool(torch.isfinite(target_joints[0]).all().item()):
                delta = target_joints[0] - joints[0]
                vertices = vertices + delta
                joints = joints + delta
        if marker_vertex_ids:
            vertices = vertices[torch.as_tensor(marker_vertex_ids, dtype=torch.long)]
        return vertices, joints

    def build_faces(self) -> torch.Tensor:
        return self.faces.clone()


def sample_allows_marker_supervision(
    sample: dict[str, Any],
    *,
    stage: str,
    stream_name: str | None = None,
) -> bool:
    dataset_name = sample["dataset_name"]
    base_dataset_name = sample["base_dataset_name"]
    if stage == "midtrain":
        return dataset_name in MIDTRAIN_DATASET_NAMES and _sample_has_marker_signal(sample)
    if stage == "posttrain":
        if stream_name is not None:
            if stream_name not in {"marker", "three_r"}:
                raise ValueError(f"未知 stream：{stream_name}")
            return stream_name == "marker" and _sample_has_marker_signal(sample)
        return (
            base_dataset_name in POSTTRAIN_MARKER_BASE_DATASET_NAMES
            and sample["is_egocentric"]
            and sample["max_left_count"] <= 1
            and sample["max_right_count"] <= 1
            and _sample_has_marker_signal(sample)
        )
    raise ValueError(f"未知阶段：{stage}")


def sample_allows_three_r_supervision(
    sample: dict[str, Any],
    *,
    stage: str,
    stream_name: str | None = None,
) -> bool:
    dataset_name = sample["dataset_name"]
    base_dataset_name = sample["base_dataset_name"]
    if dataset_name in HAND_MARKER_ONLY_DATASET_NAMES or base_dataset_name in HAND_MARKER_ONLY_DATASET_NAMES:
        return False
    if stage == "midtrain":
        return dataset_name in MIDTRAIN_DATASET_NAMES and _sample_has_three_r_signal(sample)
    if stage == "posttrain":
        if stream_name is not None:
            if stream_name not in {"marker", "three_r"}:
                raise ValueError(f"未知 stream：{stream_name}")
            return _sample_has_three_r_signal(sample)
        return base_dataset_name in POSTTRAIN_3R_BASE_DATASET_NAMES and _sample_has_three_r_signal(sample)
    raise ValueError(f"未知阶段：{stage}")


def _sample_has_marker_signal(sample: dict[str, Any]) -> bool:
    for hand in sample.get("hand_annos", []):
        if hand is None:
            continue
        if (
            _has_valid_tensor(hand.get("bbox_xyxy"))
            or _has_valid_tensor(hand.get("joints_2d"))
            or _has_valid_tensor(hand.get("joints_3d"))
            or _has_valid_tensor(hand.get("mano_pose"))
            or _has_valid_tensor(hand.get("mano_global_orient"))
            or _has_valid_tensor(hand.get("mano_hand_pose"))
        ):
            return True
    return False


def _sample_has_three_r_signal(sample: dict[str, Any]) -> bool:
    return bool(
        sample.get("depth_ref") is not None
        or _has_valid_depth_tensor(sample.get("depth"))
        or _has_valid_tensor(sample.get("camera_pose"))
        or _has_valid_tensor(sample.get("intrinsics"))
    )


def _normalized_hands_by_side(hands: Sequence[dict[str, Any] | None]) -> list[list[dict[str, Any]]]:
    outputs: list[list[dict[str, Any]]] = [[], []]
    for hand in hands:
        normalized = normalize_hand_target(hand)
        if normalized is None:
            continue
        if normalized["side"] == "left":
            outputs[0].append(normalized)
        elif normalized["side"] == "right":
            outputs[1].append(normalized)
    return outputs


def _stack_optional_sample_tensors(
    chunks: Sequence[dict[str, Any]],
    *,
    key: str,
    shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(chunks)
    num_frames = len(chunks[0]["samples"])
    stacked = torch.full((batch_size, num_frames, *shape), float("nan"), dtype=torch.float32)
    mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
    for batch_index, chunk in enumerate(chunks):
        for frame_index, sample in enumerate(chunk["samples"]):
            value = sample.get(key)
            if value is None:
                continue
            stacked[batch_index, frame_index] = torch.as_tensor(value, dtype=torch.float32)
            mask[batch_index, frame_index] = True
    return stacked, mask


def _stack_optional_chunk_depths(chunks: Sequence[dict[str, Any]]) -> torch.Tensor | None:
    depth_shape = None
    for chunk in chunks:
        chunk_depth = chunk.get("depth")
        if chunk_depth is not None:
            depth_shape = tuple(chunk_depth.shape)
            break
        for sample in chunk["samples"]:
            sample_depth = sample.get("depth")
            if sample_depth is not None:
                depth_shape = (len(chunk["samples"]), *sample_depth.shape)
                break
        if depth_shape is not None:
            break
    if depth_shape is None:
        return None

    depths = torch.full((len(chunks), *depth_shape), float("nan"), dtype=torch.float32)
    for batch_index, chunk in enumerate(chunks):
        chunk_depth = chunk.get("depth")
        if chunk_depth is not None:
            depths[batch_index] = torch.as_tensor(chunk_depth, dtype=torch.float32)
            continue

        sample_depths = []
        for sample in chunk["samples"]:
            sample_depth = sample.get("depth")
            if sample_depth is None:
                sample_depths = []
                break
            sample_depths.append(torch.as_tensor(sample_depth, dtype=torch.float32))
        if sample_depths:
            depths[batch_index] = torch.stack(sample_depths)
    return depths


def _build_depth_valid_mask(depths: torch.Tensor | None) -> torch.Tensor | None:
    if depths is None:
        return None
    if depths.ndim != 4:
        return None
    return torch.isfinite(depths) & (depths > 0)


def _project_points(points_xyz: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = points_xyz[..., 2]
    valid = torch.isfinite(points_xyz).all(dim=-1) & torch.isfinite(intrinsics).all() & (z > 1e-6)
    safe_z = z.clamp_min(1e-6)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u = fx * (points_xyz[..., 0] / safe_z) + cx
    v = fy * (points_xyz[..., 1] / safe_z) + cy
    return torch.stack([u, v], dim=-1), valid


def _bbox_from_points_2d(points_2d: torch.Tensor | None) -> torch.Tensor | None:
    if points_2d is None:
        return None
    points = torch.as_tensor(points_2d, dtype=torch.float32)
    if points.ndim != 2 or points.shape[-1] != 2:
        return None
    valid = torch.isfinite(points).all(dim=-1)
    if not bool(valid.any().item()):
        return None
    selected = points[valid]
    bbox = torch.stack(
        [
            selected[:, 0].min(),
            selected[:, 1].min(),
            selected[:, 0].max(),
            selected[:, 1].max(),
        ]
    )
    if not bool(((bbox[2] > bbox[0]) & (bbox[3] > bbox[1])).item()):
        return None
    return bbox


def _bbox_from_projected_points_3d(points_3d: torch.Tensor | None, intrinsics: torch.Tensor | None) -> torch.Tensor | None:
    if points_3d is None or intrinsics is None:
        return None
    points = torch.as_tensor(points_3d, dtype=torch.float32)
    camera_intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)
    if points.ndim != 2 or points.shape[-1] != 3 or camera_intrinsics.shape != (3, 3):
        return None
    projected, valid = _project_points(points, camera_intrinsics)
    return _bbox_from_points_2d(projected[valid])


def _derive_hand_bbox_xyxy(hand_target: dict[str, Any], intrinsics: torch.Tensor | None) -> torch.Tensor | None:
    if hand_target["bbox_xyxy"] is not None:
        return hand_target["bbox_xyxy"]
    bbox = _bbox_from_points_2d(hand_target["joints_2d"])
    if bbox is not None:
        return bbox
    return _bbox_from_projected_points_3d(hand_target["joints_3d"], intrinsics)


def _points_in_image(uv: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    return (
        (uv[..., 0] >= 0)
        & (uv[..., 0] <= image_width - 1)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] <= image_height - 1)
    )


def _any_2d_point_in_image(points_2d: torch.Tensor | None, image_height: int, image_width: int) -> bool:
    if points_2d is None:
        return False
    points = torch.as_tensor(points_2d, dtype=torch.float32)
    if points.ndim != 2 or points.shape[-1] != 2:
        return False
    finite = torch.isfinite(points).all(dim=-1)
    visible = finite & _points_in_image(points, image_height, image_width)
    return bool(visible.any().item())


def _any_projected_point_in_image(
    points_3d: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    image_height: int,
    image_width: int,
) -> bool:
    if points_3d is None or intrinsics is None:
        return False
    points = torch.as_tensor(points_3d, dtype=torch.float32)
    if points.ndim != 2 or points.shape[-1] != 3:
        return False
    projected_uv, projected_valid = _project_points(points, torch.as_tensor(intrinsics, dtype=torch.float32))
    visible = projected_valid & _points_in_image(projected_uv, image_height, image_width)
    return bool(visible.any().item())


def _hand_has_reliable_visible_evidence(
    hand_target: dict[str, Any],
    *,
    intrinsics: torch.Tensor | None,
    image_height: int,
    image_width: int,
    full_vertices: torch.Tensor | None,
    mano_joints: torch.Tensor | None,
) -> bool:
    return (
        _any_projected_point_in_image(full_vertices, intrinsics, image_height, image_width)
        or _any_projected_point_in_image(mano_joints, intrinsics, image_height, image_width)
        or _any_projected_point_in_image(hand_target.get("joints_3d"), intrinsics, image_height, image_width)
        or _any_2d_point_in_image(hand_target.get("joints_2d"), image_height, image_width)
    )


def _visibility_raster_size(image_height: int, image_width: int) -> tuple[int, int]:
    max_size = max(int(image_height), int(image_width), 1)
    scale = min(1.0, float(VERTEX_VISIBILITY_RASTER_MAX_SIZE) / float(max_size))
    return (
        max(int(round(float(image_height) * scale)), 1),
        max(int(round(float(image_width) * scale)), 1),
    )


def _scale_intrinsics_for_image_size(
    intrinsics: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    scaled = intrinsics.clone().to(dtype=torch.float32)
    scaled[0, 0] *= float(target_width) / max(float(source_width), 1.0)
    scaled[0, 2] *= float(target_width) / max(float(source_width), 1.0)
    scaled[1, 1] *= float(target_height) / max(float(source_height), 1.0)
    scaled[1, 2] *= float(target_height) / max(float(source_height), 1.0)
    return scaled


def _depth_map_occludes_points(
    points_3d: torch.Tensor,
    intrinsics: torch.Tensor | None,
    depth: torch.Tensor | None,
    *,
    image_height: int,
    image_width: int,
    threshold_meters: float,
) -> torch.Tensor:
    """Return points that are farther than the observed scene depth."""
    points = torch.as_tensor(points_3d, dtype=torch.float32)
    point_count = int(points.shape[0]) if points.ndim >= 1 else 0
    occluded = torch.zeros(point_count, dtype=torch.bool, device=points.device)
    if (
        points.ndim != 2
        or points.shape[-1] != 3
        or intrinsics is None
        or depth is None
    ):
        return occluded
    depth_map = torch.as_tensor(depth, dtype=torch.float32, device=points.device)
    if depth_map.ndim != 2 or depth_map.numel() == 0:
        return occluded
    uv, projected_valid = _project_points(
        points, torch.as_tensor(intrinsics, dtype=torch.float32, device=points.device)
    )
    depth_height, depth_width = depth_map.shape
    depth_uv = uv.clone()
    depth_uv[..., 0] *= float(depth_width) / max(float(image_width), 1.0)
    depth_uv[..., 1] *= float(depth_height) / max(float(image_height), 1.0)
    sampleable = projected_valid & _points_in_image(depth_uv, depth_height, depth_width)
    if not bool(sampleable.any().item()):
        return occluded
    pixel_x = depth_uv[:, 0].round().to(dtype=torch.long).clamp(0, depth_width - 1)
    pixel_y = depth_uv[:, 1].round().to(dtype=torch.long).clamp(0, depth_height - 1)
    sampled_depth = depth_map[pixel_y, pixel_x]
    valid_observation = sampleable & torch.isfinite(sampled_depth) & (sampled_depth > 0)
    return valid_observation & ((points[:, 2] - sampled_depth) > float(threshold_meters))


def _opencv_camera_vertices_to_pytorch3d(vertices: torch.Tensor) -> torch.Tensor:
    converted = vertices.clone()
    converted[..., 0:2] = -converted[..., 0:2]
    return converted


def _joint_visibility_targets_from_projection(
    joints_3d: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    *,
    image_height: int,
    image_width: int,
    depth: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros(21, dtype=torch.bool)
    supervision_mask = torch.zeros(21, dtype=torch.bool)
    if joints_3d is None or intrinsics is None:
        return targets, supervision_mask
    projected_uv, projected_valid = _project_points(joints_3d, intrinsics)
    visible = projected_valid & _points_in_image(projected_uv, image_height, image_width)
    depth_occluded = _depth_map_occludes_points(
        joints_3d,
        intrinsics,
        depth,
        image_height=image_height,
        image_width=image_width,
        threshold_meters=SCENE_VISIBILITY_JOINT_DEPTH_OCCLUSION_THRESHOLD_METERS,
    )
    targets[:] = visible & ~depth_occluded
    supervision_mask[:] = projected_valid
    return targets, supervision_mask


def _joint_2d_targets_from_projection(
    joints_3d: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if joints_3d is None or intrinsics is None:
        return None, None
    joints = torch.as_tensor(joints_3d, dtype=torch.float32)
    if joints.ndim != 2 or joints.shape != (21, 3):
        return None, None
    projected_uv, projected_valid = _project_points(joints, torch.as_tensor(intrinsics, dtype=torch.float32))
    targets = torch.full((21, 2), float("nan"), dtype=torch.float32)
    targets = torch.where(projected_valid.unsqueeze(-1), projected_uv, targets)
    return targets, projected_valid


def _merge_joint_2d_targets(
    existing_targets: torch.Tensor,
    projected_targets: torch.Tensor | None,
    projected_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    if projected_targets is None or projected_valid is None:
        return existing_targets, bool(torch.isfinite(existing_targets).all(dim=-1).any().item())
    existing_valid = torch.isfinite(existing_targets).all(dim=-1)
    merged = torch.where(existing_valid.unsqueeze(-1), existing_targets, projected_targets)
    valid = existing_valid | projected_valid
    return merged, bool(valid.any().item())


def _first_joint_depth(points_3d: torch.Tensor | None) -> torch.Tensor:
    if points_3d is None:
        return torch.tensor(float("nan"), dtype=torch.float32)
    points = torch.as_tensor(points_3d, dtype=torch.float32)
    if (
        points.ndim >= 2
        and points.shape[0] > 0
        and points.shape[-1] == 3
        and bool(torch.isfinite(points[0, 2]).item())
    ):
        return points[0, 2].to(dtype=torch.float32)
    return torch.tensor(float("nan"), dtype=torch.float32)


def _wrist_uv_depth_from_points(
    points_3d: torch.Tensor | None,
    points_2d: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    *,
    image_height: int,
    image_width: int,
    require_in_image: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if points_2d is not None:
        points = torch.as_tensor(points_2d, dtype=torch.float32)
        if points.ndim >= 2 and points.shape[0] > 0 and points.shape[-1] == 2:
            uv = points[0]
            valid = torch.isfinite(uv).all()
            if require_in_image:
                valid = valid & _points_in_image(uv, image_height, image_width)
            if bool(valid.item()):
                return uv, _first_joint_depth(points_3d)

    if points_3d is not None and intrinsics is not None:
        points = torch.as_tensor(points_3d, dtype=torch.float32)
        if points.ndim >= 2 and points.shape[0] > 0 and points.shape[-1] == 3:
            projected_uv, projected_valid = _project_points(points[:1], torch.as_tensor(intrinsics, dtype=torch.float32))
            valid = projected_valid[0]
            if require_in_image:
                valid = valid & _points_in_image(projected_uv[0], image_height, image_width)
            if bool(valid.item()):
                return projected_uv[0], points[0, 2].to(dtype=torch.float32)

    return None, None


def _vertex_visibility_targets_from_mesh(
    full_vertices_list: Sequence[torch.Tensor | None],
    faces: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    *,
    image_height: int,
    image_width: int,
    vertex_ids_list: Sequence[Sequence[int]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    outputs = [
        (
            torch.zeros(len(vertex_ids), dtype=torch.bool),
            torch.zeros(len(vertex_ids), dtype=torch.bool),
        )
        for vertex_ids in vertex_ids_list
    ]
    if faces is None or intrinsics is None:
        return outputs

    raster_height, raster_width = _visibility_raster_size(image_height, image_width)
    raster_intrinsics = _scale_intrinsics_for_image_size(
        intrinsics,
        source_height=image_height,
        source_width=image_width,
        target_height=raster_height,
        target_width=raster_width,
    )
    scene_meshes: list[torch.Tensor] = []
    supervised_slots: list[int] = []
    slot_global_vertex_ids: list[torch.Tensor | None] = [None] * len(full_vertices_list)
    device = None
    global_vertex_offset = 0
    for slot_index, (full_vertices, vertex_ids) in enumerate(zip(full_vertices_list, vertex_ids_list, strict=True)):
        targets, supervision_mask = outputs[slot_index]
        if full_vertices is None:
            continue
        scene_meshes.append(full_vertices)
        if len(vertex_ids) > 0:
            subset_indices = torch.as_tensor(vertex_ids, dtype=torch.long)
            subset_vertices = full_vertices[subset_indices]
            subset_uv, subset_valid = _project_points(subset_vertices, raster_intrinsics)
            subset_in_image = _points_in_image(subset_uv, raster_height, raster_width)
            # A front-facing marker outside the image is a valid negative
            # visibility target, rather than an unsupervised point.
            supervision_mask[:] = subset_valid
            if torch.any(supervision_mask):
                supervised_slots.append(slot_index)
                slot_global_vertex_ids[slot_index] = subset_indices + global_vertex_offset
        if device is None:
            device = full_vertices.device
        global_vertex_offset += int(full_vertices.shape[0])

    if not scene_meshes or device is None or not supervised_slots:
        return outputs

    scene_vertices = torch.cat(
        [
            _opencv_camera_vertices_to_pytorch3d(vertices.to(dtype=torch.float32, device=device))
            for vertices in scene_meshes
        ],
        dim=0,
    )
    scene_faces_list = []
    face_vertex_offset = 0
    faces_device = faces.to(dtype=torch.int64, device=device)
    for vertices in scene_meshes:
        scene_faces_list.append(faces_device + face_vertex_offset)
        face_vertex_offset += int(vertices.shape[0])
    scene_faces = torch.cat(scene_faces_list, dim=0)

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[raster_intrinsics[0, 0], raster_intrinsics[1, 1]]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[raster_intrinsics[0, 2], raster_intrinsics[1, 2]]], dtype=torch.float32, device=device),
        R=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, dtype=torch.float32, device=device),
        device=device,
        in_ndc=False,
        image_size=torch.tensor([[raster_height, raster_width]], dtype=torch.int64, device=device),
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(raster_height, raster_width),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            max_faces_per_bin=SCENE_VISIBILITY_RASTER_MAX_FACES_PER_BIN,
        ),
    )
    fragments = rasterizer(
        Meshes(
            verts=[scene_vertices],
            faces=[scene_faces],
        )
    )
    pix_to_face = fragments.pix_to_face[0, ..., 0]
    visible_faces = torch.unique(pix_to_face[pix_to_face >= 0])
    if visible_faces.numel() == 0:
        return outputs
    visible_vertices = torch.unique(scene_faces[visible_faces].reshape(-1))
    visible_vertex_mask = torch.zeros(scene_vertices.shape[0], dtype=torch.bool, device=device)
    visible_vertex_mask[visible_vertices] = True
    for slot_index in supervised_slots:
        targets, supervision_mask = outputs[slot_index]
        subset_global_indices = slot_global_vertex_ids[slot_index]
        assert subset_global_indices is not None
        subset_visible = visible_vertex_mask[subset_global_indices]
        targets[:] = subset_visible.cpu() & supervision_mask
    return outputs


def _batched_vertex_visibility_targets_from_mesh(
    frame_full_vertices_list: Sequence[Sequence[torch.Tensor | None]],
    faces: torch.Tensor | None,
    frame_intrinsics_list: Sequence[torch.Tensor | None],
    *,
    image_height: int,
    image_width: int,
    frame_vertex_ids_list: Sequence[Sequence[Sequence[int]]],
    visibility_mesh_mode: str = "full",
    marker_faces: torch.Tensor | None = None,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    outputs = [
        [
            (
                torch.zeros(len(vertex_ids), dtype=torch.bool),
                torch.zeros(len(vertex_ids), dtype=torch.bool),
            )
            for vertex_ids in vertex_ids_list
        ]
        for vertex_ids_list in frame_vertex_ids_list
    ]
    if faces is None:
        return outputs
    if visibility_mesh_mode not in {"full", "marker"}:
        raise ValueError("visibility_mesh_mode must be 'full' or 'marker'")
    if visibility_mesh_mode == "marker" and (marker_faces is None or marker_faces.numel() == 0):
        raise ValueError("marker visibility mesh mode requires non-empty marker_faces")

    raster_height, raster_width = _visibility_raster_size(image_height, image_width)
    scene_vertices_list: list[torch.Tensor] = []
    scene_faces_list: list[torch.Tensor] = []
    active_frame_indices: list[int] = []
    active_slot_global_vertex_ids: list[list[torch.Tensor | None]] = []
    focal_length_list: list[torch.Tensor] = []
    principal_point_list: list[torch.Tensor] = []
    faces_device = None
    device = None

    for frame_index, (full_vertices_list, intrinsics, vertex_ids_list) in enumerate(
        zip(frame_full_vertices_list, frame_intrinsics_list, frame_vertex_ids_list, strict=True)
    ):
        if intrinsics is None:
            continue
        raster_intrinsics = _scale_intrinsics_for_image_size(
            intrinsics,
            source_height=image_height,
            source_width=image_width,
            target_height=raster_height,
            target_width=raster_width,
        )
        scene_meshes: list[torch.Tensor] = []
        supervised_slots: list[int] = []
        slot_global_vertex_ids: list[torch.Tensor | None] = [None] * len(full_vertices_list)
        global_vertex_offset = 0
        for slot_index, (full_vertices, vertex_ids) in enumerate(zip(full_vertices_list, vertex_ids_list, strict=True)):
            targets, supervision_mask = outputs[frame_index][slot_index]
            if full_vertices is None:
                continue
            if len(vertex_ids) > 0:
                subset_indices = torch.as_tensor(vertex_ids, dtype=torch.long)
                subset_vertices = full_vertices[subset_indices]
                subset_uv, subset_valid = _project_points(subset_vertices, raster_intrinsics)
                subset_in_image = _points_in_image(subset_uv, raster_height, raster_width)
                # Keep out-of-image, front-facing markers as final visibility=0
                # supervision so marker and joint semantics are identical.
                supervision_mask[:] = subset_valid
                scene_mesh = subset_vertices if visibility_mesh_mode == "marker" else full_vertices
                scene_meshes.append(scene_mesh)
                if torch.any(supervision_mask):
                    supervised_slots.append(slot_index)
                    slot_global_vertex_ids[slot_index] = (
                        torch.arange(len(vertex_ids), dtype=torch.long) + global_vertex_offset
                        if visibility_mesh_mode == "marker"
                        else subset_indices + global_vertex_offset
                    )
            if device is None:
                device = full_vertices.device
                faces_device = (
                    marker_faces.to(dtype=torch.int64, device=device)
                    if visibility_mesh_mode == "marker"
                    else faces.to(dtype=torch.int64, device=device)
                )
            global_vertex_offset += int(len(vertex_ids) if visibility_mesh_mode == "marker" else full_vertices.shape[0])
        if not scene_meshes or not supervised_slots:
            continue
        assert device is not None
        assert faces_device is not None
        scene_vertices = torch.cat(
            [
                _opencv_camera_vertices_to_pytorch3d(vertices.to(dtype=torch.float32, device=device))
                for vertices in scene_meshes
            ],
            dim=0,
        )
        face_parts = []
        face_vertex_offset = 0
        for vertices in scene_meshes:
            face_parts.append(faces_device + face_vertex_offset)
            face_vertex_offset += int(vertices.shape[0])
        scene_vertices_list.append(scene_vertices)
        scene_faces_list.append(torch.cat(face_parts, dim=0))
        active_frame_indices.append(frame_index)
        active_slot_global_vertex_ids.append(slot_global_vertex_ids)
        focal_length_list.append(torch.tensor([raster_intrinsics[0, 0], raster_intrinsics[1, 1]], dtype=torch.float32, device=device))
        principal_point_list.append(torch.tensor([raster_intrinsics[0, 2], raster_intrinsics[1, 2]], dtype=torch.float32, device=device))

    if not scene_vertices_list or device is None:
        return outputs

    active_count = len(scene_vertices_list)
    cameras = PerspectiveCameras(
        focal_length=torch.stack(focal_length_list, dim=0),
        principal_point=torch.stack(principal_point_list, dim=0),
        R=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(active_count, -1, -1).clone(),
        T=torch.zeros(active_count, 3, dtype=torch.float32, device=device),
        device=device,
        in_ndc=False,
        image_size=torch.tensor([[raster_height, raster_width]], dtype=torch.int64, device=device).expand(active_count, -1),
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(raster_height, raster_width),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            max_faces_per_bin=SCENE_VISIBILITY_RASTER_MAX_FACES_PER_BIN,
        ),
    )
    meshes = Meshes(
        verts=scene_vertices_list,
        faces=scene_faces_list,
    )
    fragments = rasterizer(meshes)
    face_offsets = meshes.mesh_to_faces_packed_first_idx()

    for active_index, frame_index in enumerate(active_frame_indices):
        pix_to_face = fragments.pix_to_face[active_index, ..., 0]
        visible_faces = torch.unique(pix_to_face[pix_to_face >= 0])
        if visible_faces.numel() == 0:
            continue
        local_visible_faces = visible_faces - face_offsets[active_index]
        local_visible_faces = local_visible_faces[
            (local_visible_faces >= 0) & (local_visible_faces < scene_faces_list[active_index].shape[0])
        ]
        if local_visible_faces.numel() == 0:
            continue
        visible_vertices = torch.unique(scene_faces_list[active_index][local_visible_faces].reshape(-1))
        visible_vertex_mask = torch.zeros(scene_vertices_list[active_index].shape[0], dtype=torch.bool, device=device)
        visible_vertex_mask[visible_vertices] = True
        for slot_index, subset_global_indices in enumerate(active_slot_global_vertex_ids[active_index]):
            if subset_global_indices is None:
                continue
            targets, supervision_mask = outputs[frame_index][slot_index]
            subset_visible = visible_vertex_mask[subset_global_indices]
            targets[:] = subset_visible.cpu() & supervision_mask
    return outputs


def _scene_marker_visibility_targets_from_mesh(
    full_vertices_list: Sequence[torch.Tensor | None],
    faces: torch.Tensor,
    intrinsics: torch.Tensor | None,
    *,
    image_height: int,
    image_width: int,
    vertex_ids_list: Sequence[Sequence[int]],
    object_mesh: tuple[torch.Tensor, torch.Tensor] | None,
    object_mesh_factory: Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    object_face_count: int | None = None,
    object_ray_accelerator: SceneObjectAccel | None = None,
    depth: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Full-resolution, per-marker z-buffer visibility for one frame."""
    outputs = [
        (torch.zeros(len(vertex_ids), dtype=torch.bool), torch.zeros(len(vertex_ids), dtype=torch.bool))
        for vertex_ids in vertex_ids_list
    ]
    if intrinsics is None:
        return outputs

    device = next((vertices.device for vertices in full_vertices_list if vertices is not None), None)
    if device is None:
        return outputs
    scene_vertices_parts: list[torch.Tensor] = []
    scene_faces_parts: list[torch.Tensor] = []
    scene_face_owners: list[torch.Tensor] = []
    marker_points: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    vertex_offset = 0
    faces_device = faces.to(dtype=torch.int64, device=device)
    intrinsics_device = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    for slot_index, (vertices, vertex_ids) in enumerate(zip(full_vertices_list, vertex_ids_list, strict=True)):
        if vertices is None:
            continue
        vertices = vertices.to(dtype=torch.float32, device=device)
        subset_indices = torch.as_tensor(vertex_ids, dtype=torch.long, device=device)
        if subset_indices.numel():
            subset_vertices = vertices[subset_indices]
            subset_uv, subset_valid = _project_points(subset_vertices, intrinsics_device)
            subset_in_image = _points_in_image(subset_uv, image_height, image_width)
            # Projection validity gates supervision; image membership gates only
            # visibility=1. Out-of-image markers retain the final 0 target.
            outputs[slot_index][1][:] = subset_valid.cpu()
            marker_points.append((slot_index, subset_vertices, subset_uv, subset_valid & subset_in_image))
        scene_vertices_parts.append(_opencv_camera_vertices_to_pytorch3d(vertices))
        scene_faces_parts.append(faces_device + vertex_offset)
        scene_face_owners.append(torch.full((faces_device.shape[0],), slot_index, dtype=torch.long, device=device))
        vertex_offset += int(vertices.shape[0])
    effective_object_face_count = object_face_count
    if effective_object_face_count is None and object_mesh is not None:
        effective_object_face_count = int(object_mesh[1].shape[0])
    object_routed_to_bvh = (
        object_ray_accelerator is not None
        and effective_object_face_count is not None
        and effective_object_face_count > SCENE_VISIBILITY_OBJECT_RASTER_MAX_FACES
    )
    if object_mesh is None and not object_routed_to_bvh and object_mesh_factory is not None:
        object_mesh = object_mesh_factory()
    if object_mesh is not None:
        object_vertices, object_faces = object_mesh
        object_vertices = object_vertices.to(dtype=torch.float32, device=device)
        object_faces = object_faces.to(dtype=torch.int64, device=device)
        if object_vertices.numel() and object_faces.numel() and not object_routed_to_bvh:
            scene_vertices_parts.append(_opencv_camera_vertices_to_pytorch3d(object_vertices))
            scene_faces_parts.append(object_faces + vertex_offset)
            scene_face_owners.append(torch.full((object_faces.shape[0],), -1, dtype=torch.long, device=device))
    if not scene_vertices_parts or not marker_points:
        return outputs

    scene_vertices = torch.cat(scene_vertices_parts, dim=0)
    scene_faces = torch.cat(scene_faces_parts, dim=0)
    face_owners = torch.cat(scene_face_owners, dim=0)
    cameras = PerspectiveCameras(
        focal_length=intrinsics_device[[0, 1], [0, 1]].reshape(1, 2),
        principal_point=intrinsics_device[[0, 1], [2, 2]].reshape(1, 2),
        R=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0),
        T=torch.zeros((1, 3), dtype=torch.float32, device=device),
        device=device,
        in_ndc=False,
        image_size=torch.tensor([[image_height, image_width]], dtype=torch.int64, device=device),
    )
    fragments = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(image_height, image_width),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            max_faces_per_bin=SCENE_VISIBILITY_RASTER_MAX_FACES_PER_BIN,
        ),
    )(Meshes(verts=[scene_vertices], faces=[scene_faces]))
    pix_to_face = fragments.pix_to_face[0, ..., 0]
    zbuf = fragments.zbuf[0, ..., 0]
    for slot_index, vertices, uv, in_image in marker_points:
        targets, supervision_mask = outputs[slot_index]
        if not bool(in_image.any().item()):
            continue
        pixel_x = uv[:, 0].round().to(dtype=torch.long).clamp(0, image_width - 1)
        pixel_y = uv[:, 1].round().to(dtype=torch.long).clamp(0, image_height - 1)
        front_faces = pix_to_face[pixel_y, pixel_x]
        front_depths = zbuf[pixel_y, pixel_x]
        front_owner = torch.full_like(front_faces, -2)
        valid_faces = front_faces >= 0
        front_owner[valid_faces] = face_owners[front_faces[valid_faces]]
        depth_matches = torch.isclose(front_depths, vertices[:, 2], atol=SCENE_VISIBILITY_MARKER_DEPTH_ATOL, rtol=SCENE_VISIBILITY_MARKER_DEPTH_RTOL)
        mesh_visible = in_image & (front_owner == slot_index) & depth_matches
        if object_routed_to_bvh:
            try:
                object_occluded = torch.as_tensor(
                    object_ray_accelerator.segment_occluded_camera(
                        vertices,
                        endpoint_epsilon=SCENE_VISIBILITY_RAY_ENDPOINT_EPSILON,
                    ),
                    dtype=torch.bool,
                    device=device,
                )
                if object_occluded.shape != (int(vertices.shape[0]),):
                    raise ValueError("object ray accelerator returned an unexpected shape")
                mesh_visible &= ~object_occluded
            except Exception:
                fallback_object_mesh = object_mesh if object_mesh is not None else (
                    None if object_mesh_factory is None else object_mesh_factory()
                )
                if fallback_object_mesh is None:
                    raise ValueError("object marker visibility fallback mesh is unavailable")
                return _scene_marker_visibility_targets_from_mesh(
                    full_vertices_list,
                    faces,
                    intrinsics,
                    image_height=image_height,
                    image_width=image_width,
                    vertex_ids_list=vertex_ids_list,
                    object_mesh=fallback_object_mesh,
                    object_ray_accelerator=None,
                    depth=depth,
                )
        depth_occluded = _depth_map_occludes_points(
            vertices, intrinsics_device, depth,
            image_height=image_height, image_width=image_width,
            threshold_meters=SCENE_VISIBILITY_MARKER_DEPTH_OCCLUSION_THRESHOLD_METERS,
        )
        targets[:] = mesh_visible.cpu() & supervision_mask & ~depth_occluded.cpu()
    return outputs


def _batched_scene_marker_visibility_targets_from_mesh(
    frames: Sequence[
        tuple[
            Sequence[torch.Tensor | None],
            torch.Tensor | None,
            Sequence[Sequence[int]],
            tuple[torch.Tensor, torch.Tensor] | None,
        ]
    ],
    faces: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    device: torch.device,
    depths: Sequence[torch.Tensor | None] | None = None,
    object_ray_accelerators: Sequence[SceneObjectAccel | None] | None = None,
    object_face_counts: Sequence[int | None] | None = None,
    object_mesh_factories: Sequence[Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None] | None = None,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Batch the unchanged per-frame marker z-buffer calculation on ``device``."""
    all_outputs = [
        [(torch.zeros(len(vertex_ids), dtype=torch.bool), torch.zeros(len(vertex_ids), dtype=torch.bool)) for vertex_ids in vertex_ids_list]
        for _, _, vertex_ids_list, _ in frames
    ]
    if depths is not None and len(depths) != len(frames):
        raise ValueError("scene visibility depth count must match frame count")
    if object_ray_accelerators is not None and len(object_ray_accelerators) != len(frames):
        raise ValueError("scene visibility object accelerator count must match frame count")
    if object_face_counts is not None and len(object_face_counts) != len(frames):
        raise ValueError("scene visibility object face count must match frame count")
    if object_mesh_factories is not None and len(object_mesh_factories) != len(frames):
        raise ValueError("scene visibility object mesh factory count must match frame count")
    active: list[
        tuple[
            int,
            list[torch.Tensor],
            list[torch.Tensor],
            list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]],
            torch.Tensor,
            SceneObjectAccel | None,
            Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None,
        ]
    ] = []
    faces_device = faces.to(dtype=torch.int64, device=device)
    for frame_index, (full_vertices_list, intrinsics, vertex_ids_list, object_mesh) in enumerate(frames):
        if intrinsics is None:
            continue
        object_ray_accelerator = (
            None if object_ray_accelerators is None else object_ray_accelerators[frame_index]
        )
        object_mesh_factory = (
            None if object_mesh_factories is None else object_mesh_factories[frame_index]
        )
        effective_object_face_count = (
            int(object_mesh[1].shape[0])
            if object_face_counts is None and object_mesh is not None
            else None if object_face_counts is None else object_face_counts[frame_index]
        )
        object_routed_to_bvh = (
            object_ray_accelerator is not None
            and effective_object_face_count is not None
            and int(effective_object_face_count) > SCENE_VISIBILITY_OBJECT_RASTER_MAX_FACES
        )
        if object_mesh is None and not object_routed_to_bvh and object_mesh_factory is not None:
            object_mesh = object_mesh_factory()
        scene_vertices_parts: list[torch.Tensor] = []
        scene_faces_parts: list[torch.Tensor] = []
        face_owners: list[torch.Tensor] = []
        marker_points: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        vertex_offset = 0
        intrinsics_device = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
        for slot_index, (vertices, vertex_ids) in enumerate(zip(full_vertices_list, vertex_ids_list, strict=True)):
            if vertices is None:
                continue
            vertices = vertices.to(dtype=torch.float32, device=device)
            subset_indices = torch.as_tensor(vertex_ids, dtype=torch.long, device=device)
            if subset_indices.numel():
                subset_vertices = vertices[subset_indices]
                subset_uv, subset_valid = _project_points(subset_vertices, intrinsics_device)
                in_image = subset_valid & _points_in_image(subset_uv, image_height, image_width)
                all_outputs[frame_index][slot_index][1][:] = subset_valid.cpu()
                marker_points.append((slot_index, subset_vertices, subset_uv, in_image))
            scene_vertices_parts.append(_opencv_camera_vertices_to_pytorch3d(vertices))
            scene_faces_parts.append(faces_device + vertex_offset)
            face_owners.append(torch.full((faces_device.shape[0],), slot_index, dtype=torch.long, device=device))
            vertex_offset += int(vertices.shape[0])
        if object_mesh is not None and not object_routed_to_bvh:
            object_vertices, object_faces = object_mesh
            object_vertices = object_vertices.to(dtype=torch.float32, device=device)
            object_faces = object_faces.to(dtype=torch.int64, device=device)
            if object_vertices.numel() and object_faces.numel():
                scene_vertices_parts.append(_opencv_camera_vertices_to_pytorch3d(object_vertices))
                scene_faces_parts.append(object_faces + vertex_offset)
                face_owners.append(torch.full((object_faces.shape[0],), -1, dtype=torch.long, device=device))
        if scene_vertices_parts and marker_points:
            active.append(
                (
                    frame_index,
                    scene_vertices_parts,
                    scene_faces_parts,
                    marker_points,
                    torch.cat(face_owners),
                    object_ray_accelerator if object_routed_to_bvh else None,
                    object_mesh_factory if object_routed_to_bvh else None,
                )
            )
    if not active:
        return all_outputs
    meshes = Meshes(
        verts=[torch.cat(parts, dim=0) for _, parts, _, _, _, _, _ in active],
        faces=[torch.cat(parts, dim=0) for _, _, parts, _, _, _, _ in active],
    )
    intrinsics_batch = torch.stack([
        torch.as_tensor(frames[frame_index][1], dtype=torch.float32, device=device)
        for frame_index, _, _, _, _, _, _ in active
    ])
    active_count = len(active)
    cameras = PerspectiveCameras(
        focal_length=intrinsics_batch[:, [0, 1], [0, 1]],
        principal_point=intrinsics_batch[:, [0, 1], [2, 2]],
        R=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(active_count, -1, -1),
        T=torch.zeros((active_count, 3), dtype=torch.float32, device=device),
        device=device,
        in_ndc=False,
        image_size=torch.tensor([[image_height, image_width]], dtype=torch.int64, device=device).expand(active_count, -1),
    )
    fragments = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(image_height, image_width),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            max_faces_per_bin=SCENE_VISIBILITY_RASTER_MAX_FACES_PER_BIN,
        ),
    )(meshes)
    face_offsets = meshes.mesh_to_faces_packed_first_idx()
    for active_index, (
        frame_index,
        _,
        _,
        marker_points,
        face_owners,
        object_ray_accelerator,
        object_mesh_factory,
    ) in enumerate(active):
        pix_to_face = fragments.pix_to_face[active_index, ..., 0]
        zbuf = fragments.zbuf[active_index, ..., 0]
        local_faces = pix_to_face.clone()
        valid_faces = local_faces >= 0
        local_faces[valid_faces] -= face_offsets[active_index]
        for slot_index, vertices, uv, in_image in marker_points:
            if not bool(in_image.any().item()):
                continue
            pixel_x = uv[:, 0].round().to(dtype=torch.long).clamp(0, image_width - 1)
            pixel_y = uv[:, 1].round().to(dtype=torch.long).clamp(0, image_height - 1)
            front_faces = local_faces[pixel_y, pixel_x]
            front_depths = zbuf[pixel_y, pixel_x]
            front_owner = torch.full_like(front_faces, -2)
            valid = front_faces >= 0
            front_owner[valid] = face_owners[front_faces[valid]]
            depth_matches = torch.isclose(front_depths, vertices[:, 2], atol=SCENE_VISIBILITY_MARKER_DEPTH_ATOL, rtol=SCENE_VISIBILITY_MARKER_DEPTH_RTOL)
            targets, supervision_mask = all_outputs[frame_index][slot_index]
            mesh_visible = in_image & (front_owner == slot_index) & depth_matches
            if object_ray_accelerator is not None:
                try:
                    object_occluded = torch.as_tensor(
                        object_ray_accelerator.segment_occluded_camera(
                            vertices,
                            endpoint_epsilon=SCENE_VISIBILITY_RAY_ENDPOINT_EPSILON,
                        ),
                        dtype=torch.bool,
                        device=device,
                    )
                    if object_occluded.shape != (int(vertices.shape[0]),):
                        raise ValueError("object ray accelerator returned an unexpected shape")
                    mesh_visible &= ~object_occluded
                except Exception:
                    full_vertices_list, intrinsics, vertex_ids_list, object_mesh = frames[frame_index]
                    fallback_object_mesh = object_mesh if object_mesh is not None else (
                        None if object_mesh_factory is None else object_mesh_factory()
                    )
                    if fallback_object_mesh is None:
                        raise ValueError("object marker visibility fallback mesh is unavailable")
                    all_outputs[frame_index] = _scene_marker_visibility_targets_from_mesh(
                        full_vertices_list,
                        faces,
                        intrinsics,
                        image_height=image_height,
                        image_width=image_width,
                        vertex_ids_list=vertex_ids_list,
                        object_mesh=fallback_object_mesh,
                        object_ray_accelerator=None,
                        depth=None if depths is None else depths[frame_index],
                    )
                    break
            depth_occluded = _depth_map_occludes_points(
                vertices,
                frames[frame_index][1],
                None if depths is None else depths[frame_index],
                image_height=image_height,
                image_width=image_width,
                threshold_meters=SCENE_VISIBILITY_MARKER_DEPTH_OCCLUSION_THRESHOLD_METERS,
            )
            targets[:] = mesh_visible.cpu() & supervision_mask & ~depth_occluded.cpu()
    return all_outputs


def _bounded_scene_marker_visibility_targets_from_mesh(
    frames: Sequence[
        tuple[
            Sequence[torch.Tensor | None],
            torch.Tensor | None,
            Sequence[Sequence[int]],
            tuple[torch.Tensor, torch.Tensor] | None,
        ]
    ],
    faces: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    device: torch.device,
    depths: Sequence[torch.Tensor | None] | None = None,
    object_ray_accelerators: Sequence[SceneObjectAccel | None] | None = None,
    object_face_counts: Sequence[int | None] | None = None,
    object_mesh_factories: Sequence[Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None] | None = None,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Preserve framewise visibility while bounding each CUDA rasterizer batch."""
    outputs = []
    for start in range(0, len(frames), SCENE_VISIBILITY_RASTER_MAX_BATCH_SIZE):
        stop = start + SCENE_VISIBILITY_RASTER_MAX_BATCH_SIZE
        outputs.extend(_batched_scene_marker_visibility_targets_from_mesh(
            frames[start:stop], faces,
            image_height=image_height,
            image_width=image_width,
            device=device,
            depths=None if depths is None else depths[start:stop],
            object_ray_accelerators=None if object_ray_accelerators is None else object_ray_accelerators[start:stop],
            object_face_counts=None if object_face_counts is None else object_face_counts[start:stop],
            object_mesh_factories=None if object_mesh_factories is None else object_mesh_factories[start:stop],
        ))
    return outputs


def _segment_triangle_intersection_depths(
    endpoint: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Return normalized camera-origin segment intersection depths in (0, 1)."""
    if faces.numel() == 0 or not bool(torch.isfinite(endpoint).all().item()):
        return torch.empty(0, dtype=torch.float32, device=endpoint.device)
    endpoint = endpoint.to(dtype=torch.float32)
    vertices = vertices.to(dtype=torch.float32, device=endpoint.device)
    faces = faces.to(dtype=torch.long, device=endpoint.device)
    values: list[torch.Tensor] = []
    for start in range(0, int(faces.shape[0]), SCENE_VISIBILITY_RAY_FACE_BLOCK_SIZE):
        triangles = vertices[faces[start : start + SCENE_VISIBILITY_RAY_FACE_BLOCK_SIZE]]
        edge_one = triangles[:, 1] - triangles[:, 0]
        edge_two = triangles[:, 2] - triangles[:, 0]
        pvec = torch.cross(endpoint.expand_as(edge_two), edge_two, dim=-1)
        determinant = (edge_one * pvec).sum(dim=-1)
        valid = determinant.abs() > SCENE_VISIBILITY_RAY_EPSILON
        inverse_determinant = torch.where(valid, determinant.reciprocal(), torch.zeros_like(determinant))
        tvec = -triangles[:, 0]
        bary_u = (tvec * pvec).sum(dim=-1) * inverse_determinant
        qvec = torch.cross(tvec, edge_one, dim=-1)
        bary_v = (endpoint.expand_as(qvec) * qvec).sum(dim=-1) * inverse_determinant
        depth = (edge_two * qvec).sum(dim=-1) * inverse_determinant
        valid &= bary_u >= -SCENE_VISIBILITY_RAY_EPSILON
        valid &= bary_v >= -SCENE_VISIBILITY_RAY_EPSILON
        valid &= (bary_u + bary_v) <= 1.0 + SCENE_VISIBILITY_RAY_EPSILON
        valid &= depth > SCENE_VISIBILITY_RAY_ENDPOINT_EPSILON
        valid &= depth < 1.0 - SCENE_VISIBILITY_RAY_ENDPOINT_EPSILON
        if bool(valid.any().item()):
            values.append(depth[valid])
    return torch.cat(values) if values else torch.empty(0, dtype=torch.float32, device=endpoint.device)


def _segment_triangle_intersection_depths_batched(
    endpoints: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Return ray depths for every endpoint and triangle; invalid entries are +inf."""
    if faces.numel() == 0:
        return torch.empty((endpoints.shape[0], 0), dtype=torch.float32, device=endpoints.device)
    endpoints = endpoints.to(dtype=torch.float32)
    vertices = vertices.to(dtype=torch.float32, device=endpoints.device)
    faces = faces.to(dtype=torch.long, device=endpoints.device)
    chunks: list[torch.Tensor] = []
    for start in range(0, int(faces.shape[0]), SCENE_VISIBILITY_RAY_FACE_BLOCK_SIZE):
        triangles = vertices[faces[start : start + SCENE_VISIBILITY_RAY_FACE_BLOCK_SIZE]]
        edge_one = triangles[:, 1] - triangles[:, 0]
        edge_two = triangles[:, 2] - triangles[:, 0]
        endpoint_expanded = endpoints[:, None, :]
        pvec = torch.cross(endpoint_expanded.expand(-1, edge_two.shape[0], -1), edge_two[None], dim=-1)
        determinant = (edge_one[None] * pvec).sum(dim=-1)
        valid = determinant.abs() > SCENE_VISIBILITY_RAY_EPSILON
        inverse_determinant = torch.where(valid, determinant.reciprocal(), torch.zeros_like(determinant))
        tvec = -triangles[None, :, 0]
        bary_u = (tvec * pvec).sum(dim=-1) * inverse_determinant
        qvec = torch.cross(tvec, edge_one[None], dim=-1)
        bary_v = (endpoint_expanded * qvec).sum(dim=-1) * inverse_determinant
        depth = (edge_two[None] * qvec).sum(dim=-1) * inverse_determinant
        valid &= bary_u >= -SCENE_VISIBILITY_RAY_EPSILON
        valid &= bary_v >= -SCENE_VISIBILITY_RAY_EPSILON
        valid &= (bary_u + bary_v) <= 1.0 + SCENE_VISIBILITY_RAY_EPSILON
        valid &= depth > SCENE_VISIBILITY_RAY_ENDPOINT_EPSILON
        valid &= depth < 1.0 - SCENE_VISIBILITY_RAY_ENDPOINT_EPSILON
        chunks.append(torch.where(valid, depth, torch.full_like(depth, float("inf"))))
    return torch.cat(chunks, dim=1)


def _joint_visibility_targets_from_scene_reference(
    mano_joints_list: Sequence[torch.Tensor | None],
    full_vertices_list: Sequence[torch.Tensor | None],
    faces: torch.Tensor,
    intrinsics: torch.Tensor | None,
    *,
    image_height: int,
    image_width: int,
    object_mesh: tuple[torch.Tensor, torch.Tensor] | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-optimization reference used only for parity tests."""
    outputs = [(torch.zeros(21, dtype=torch.bool), torch.zeros(21, dtype=torch.bool)) for _ in mano_joints_list]
    if intrinsics is None:
        return outputs
    device = next((joints.device for joints in mano_joints_list if joints is not None), None)
    if device is None:
        return outputs
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    hand_meshes = [None if vertices is None else vertices.to(dtype=torch.float32, device=device) for vertices in full_vertices_list]
    for target_slot, joints in enumerate(mano_joints_list):
        if joints is None or hand_meshes[target_slot] is None:
            continue
        joints = joints.to(dtype=torch.float32, device=device)
        uv, projected_valid = _project_points(joints, intrinsics)
        in_image = projected_valid & _points_in_image(uv, image_height, image_width)
        outputs[target_slot][1][:] = projected_valid.cpu()
        for joint_index, endpoint in enumerate(joints):
            if not bool(in_image[joint_index].item()):
                continue
            if object_mesh is not None:
                object_depths = _segment_triangle_intersection_depths(endpoint, object_mesh[0].to(device=device), object_mesh[1].to(device=device))
                if object_depths.numel():
                    continue
            unique_surfaces: list[tuple[float, set[int]]] = []
            hit_depths: list[tuple[float, int]] = []
            for owner, vertices in enumerate(hand_meshes):
                if vertices is not None:
                    hit_depths.extend((float(depth.item()), owner) for depth in _segment_triangle_intersection_depths(endpoint, vertices, faces))
            for depth, owner in sorted(hit_depths):
                if not unique_surfaces or abs(depth - unique_surfaces[-1][0]) > SCENE_VISIBILITY_RAY_DEDUP_TOLERANCE:
                    unique_surfaces.append((depth, {owner}))
                else:
                    unique_surfaces[-1][1].add(owner)
            if not unique_surfaces or (len(unique_surfaces) == 1 and unique_surfaces[0][1] == {target_slot}):
                outputs[target_slot][0][joint_index] = True
    return outputs


def _joint_visibility_targets_from_scene_vectorized(
    mano_joints_list: Sequence[torch.Tensor | None],
    full_vertices_list: Sequence[torch.Tensor | None],
    faces: torch.Tensor,
    intrinsics: torch.Tensor | None,
    *,
    image_height: int,
    image_width: int,
    object_mesh: tuple[torch.Tensor, torch.Tensor] | None,
    object_mesh_factory: Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    object_ray_accelerator: SceneObjectAccel | None = None,
    depth: torch.Tensor | None = None,
    scene_device: torch.device | str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    outputs = [(torch.zeros(21, dtype=torch.bool), torch.zeros(21, dtype=torch.bool)) for _ in mano_joints_list]
    if intrinsics is None:
        return outputs
    device = torch.device(scene_device) if scene_device is not None else next((joints.device for joints in mano_joints_list if joints is not None), None)
    if device is None:
        return outputs
    faces = faces.to(dtype=torch.long, device=device)
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    hand_meshes = [None if vertices is None else vertices.to(dtype=torch.float32, device=device) for vertices in full_vertices_list]
    fallback_object_mesh = object_mesh

    def resolve_fallback_object_mesh() -> tuple[torch.Tensor, torch.Tensor] | None:
        nonlocal fallback_object_mesh
        if fallback_object_mesh is None and object_mesh_factory is not None:
            fallback_object_mesh = object_mesh_factory()
        return fallback_object_mesh

    for target_slot, joints in enumerate(mano_joints_list):
        if joints is None or hand_meshes[target_slot] is None:
            continue
        joints = joints.to(dtype=torch.float32, device=device)
        uv, projected_valid = _project_points(joints, intrinsics)
        in_image = projected_valid & _points_in_image(uv, image_height, image_width)
        outputs[target_slot][1][:] = projected_valid.cpu()
        visible = torch.zeros(21, dtype=torch.bool, device=device)
        active = in_image
        if object_mesh is not None or object_ray_accelerator is not None:
            object_occluded = None
            if object_ray_accelerator is not None:
                try:
                    accelerated = torch.as_tensor(
                        object_ray_accelerator.segment_occluded_camera(
                            joints,
                            endpoint_epsilon=SCENE_VISIBILITY_RAY_ENDPOINT_EPSILON,
                        ),
                        dtype=torch.bool,
                        device=device,
                    )
                    if accelerated.shape != (int(joints.shape[0]),):
                        raise ValueError("object ray accelerator returned an unexpected shape")
                    object_occluded = accelerated
                except Exception:
                    object_occluded = None
            if object_occluded is None:
                fallback_mesh = resolve_fallback_object_mesh()
                if fallback_mesh is None:
                    raise ValueError("object ray fallback mesh is unavailable")
                object_depths = _segment_triangle_intersection_depths_batched(
                    joints, fallback_mesh[0].to(device=device), fallback_mesh[1].to(device=device)
                )
                object_occluded = torch.isfinite(object_depths).any(dim=1)
            active = active & ~object_occluded
        depth_parts: list[torch.Tensor] = []
        owner_parts: list[torch.Tensor] = []
        for owner, vertices in enumerate(hand_meshes):
            if vertices is None:
                continue
            depths = _segment_triangle_intersection_depths_batched(joints, vertices, faces)
            depth_parts.append(depths)
            owner_parts.append(torch.full((depths.shape[1],), owner, dtype=torch.long, device=device))
        if depth_parts:
            all_depths = torch.cat(depth_parts, dim=1)
            all_owners = torch.cat(owner_parts)
            sorted_depths, sorted_indices = all_depths.sort(dim=1)
            sorted_valid = torch.isfinite(sorted_depths)
            new_surface = sorted_valid.clone()
            new_surface[:, 1:] &= (sorted_depths[:, 1:] - sorted_depths[:, :-1]).abs() > SCENE_VISIBILITY_RAY_DEDUP_TOLERANCE
            surface_count = new_surface.sum(dim=1)
            sorted_owners = all_owners[sorted_indices]
            only_target_owner = torch.where(
                sorted_valid,
                sorted_owners == target_slot,
                torch.ones_like(sorted_valid),
            ).all(dim=1)
            visible = (surface_count == 0) | ((surface_count == 1) & only_target_owner)
        else:
            visible[:] = True
        depth_occluded = _depth_map_occludes_points(
            joints, intrinsics, depth,
            image_height=image_height, image_width=image_width,
            threshold_meters=SCENE_VISIBILITY_JOINT_DEPTH_OCCLUSION_THRESHOLD_METERS,
        )
        outputs[target_slot][0][:] = (visible & active & ~depth_occluded).cpu()
    return outputs


def _joint_visibility_targets_from_scene(
    mano_joints_list: Sequence[torch.Tensor | None],
    full_vertices_list: Sequence[torch.Tensor | None],
    faces: torch.Tensor,
    intrinsics: torch.Tensor | None,
    *,
    image_height: int,
    image_width: int,
    object_mesh: tuple[torch.Tensor, torch.Tensor] | None,
    object_mesh_factory: Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    object_ray_accelerator: SceneObjectAccel | None = None,
    depth: torch.Tensor | None = None,
    scene_device: torch.device | str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Vectorized equivalent of the reference scene joint visibility calculation."""
    return _joint_visibility_targets_from_scene_vectorized(
        mano_joints_list, full_vertices_list, faces, intrinsics,
        image_height=image_height,
        image_width=image_width,
        object_mesh=object_mesh,
        object_mesh_factory=object_mesh_factory,
        object_ray_accelerator=object_ray_accelerator,
        depth=depth,
        scene_device=scene_device,
    )


def _has_valid_tensor(value: Any) -> bool:
    if value is None:
        return False
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        return False
    return bool(torch.isfinite(tensor).all().item())


def _has_valid_depth_tensor(value: Any) -> bool:
    if value is None:
        return False
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        return False
    finite_mask = torch.isfinite(tensor)
    return bool((finite_mask & (tensor > 0)).any().item())


def _load_scene_mesh(mesh_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    if Path(mesh_path).suffix.lower() == ".obj":
        vertices, faces, _ = load_obj(mesh_path, load_textures=False)
        return vertices.to(dtype=torch.float32), faces.verts_idx.to(dtype=torch.long)

    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"无法读取有效 object mesh: {mesh_path}")
    return (
        torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32),
        torch.as_tensor(np.asarray(mesh.faces), dtype=torch.long),
    )


class MarkerBatchCollator:
    def __init__(
        self,
        *,
        stage: str,
        marker_vertex_ids: Sequence[int] | None = None,
        mano_vertex_builder: ManoVertexBuilder | None = None,
        vertex_visibility_mesh_mode: str = "full",
        include_scene_occlusion_in_visibility: bool = True,
        scene_visibility_device: torch.device | str | None = None,
        interhand_contact_compute_device: torch.device | str | None = None,
        contact_supervision_by_dataset: dict[str, str] | None = None,
        stream_name: str | None = None,
    ) -> None:
        if vertex_visibility_mesh_mode not in {"full", "marker"}:
            raise ValueError("vertex_visibility_mesh_mode must be 'full' or 'marker'")
        if stream_name not in {None, "marker", "three_r"}:
            raise ValueError(f"未知 stream：{stream_name}")
        self.stage = stage
        self.stream_name = stream_name
        self.hand_supervision_enabled = not (stage == "posttrain" and stream_name == "three_r")
        self.marker_vertex_ids = list(marker_vertex_ids) if marker_vertex_ids is not None else []
        self.mano_vertex_builder = mano_vertex_builder
        self.vertex_visibility_mesh_mode = vertex_visibility_mesh_mode
        self.include_scene_occlusion_in_visibility = bool(include_scene_occlusion_in_visibility)
        self.scene_visibility_device = torch.device(scene_visibility_device) if scene_visibility_device is not None else None
        if interhand_contact_compute_device is not None:
            self.interhand_contact_compute_device = torch.device(interhand_contact_compute_device)
        elif self.scene_visibility_device is not None:
            self.interhand_contact_compute_device = self.scene_visibility_device
        elif torch.cuda.is_available():
            self.interhand_contact_compute_device = torch.device("cuda")
        else:
            self.interhand_contact_compute_device = torch.device("cpu")
        self.contact_supervision_by_dataset = dict(contact_supervision_by_dataset or {})
        valid_contact_modes = {"object_and_interhand", "interhand_only", "disabled"}
        invalid_modes = {
            name: mode
            for name, mode in self.contact_supervision_by_dataset.items()
            if mode not in valid_contact_modes
        }
        if invalid_modes:
            raise ValueError(f"非法 contact supervision mode: {invalid_modes}")
        self._object_mesh_cache = ObjectMeshRawCache(max_entries=64, max_bytes=1 << 30)
        self._object_accel_cache = ObjectMeshAccelCache(max_entries=64)
        self.marker_visibility_faces = marker_faces_for_count(len(self.marker_vertex_ids))
        if self.vertex_visibility_mesh_mode == "marker" and self.marker_visibility_faces.numel() == 0:
            raise ValueError("marker vertex visibility mesh mode requires marker faces for the configured marker count")

    def _object_scene_for_sample(
        self,
        sample: dict[str, Any],
        *,
        require_visibility: bool = False,
    ) -> SceneObjectMesh | None:
        if require_visibility and not self.include_scene_occlusion_in_visibility:
            return None
        extras = sample.get("extras", {})
        descriptors = extras.get("scene_objects")
        if descriptors is None:
            object_id = extras.get("h2o_object_id")
            if object_id is None or int(torch.as_tensor(object_id).item()) == 0:
                return None
            descriptors = [{
                "object_id": object_id,
                "mesh_path": extras.get("h2o_object_mesh_path"),
                "object_to_camera": extras.get("h2o_object_to_camera"),
            }]
        if not isinstance(descriptors, (list, tuple)):
            raise ValueError("extras.scene_objects 必须是 object descriptor 列表。")
        parts: list[SceneObjectPart] = []
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                raise ValueError("scene object descriptor 必须是字典。")
            mesh_path = descriptor.get("mesh_path")
            object_to_camera = descriptor.get("object_to_camera")
            if not isinstance(mesh_path, str) or object_to_camera is None:
                raise ValueError("scene object 必须提供 mesh_path 和 object_to_camera。")
            vertices, faces = self._object_mesh_cache.get_or_load(mesh_path, _load_scene_mesh)
            mesh_scale = float(descriptor.get("mesh_scale", 1.0))
            if not np.isfinite(mesh_scale) or mesh_scale <= 0:
                raise ValueError("scene object mesh_scale must be finite and > 0.")
            transform = torch.as_tensor(object_to_camera, dtype=torch.float32)
            if transform.shape != (4, 4) or not torch.isfinite(transform).all():
                raise ValueError("object_to_camera 必须是有限的 (4, 4) object-to-camera 变换。")
            parts.append(
                SceneObjectPart(
                    mesh_path=mesh_path,
                    local_vertices=vertices,
                    local_faces=faces,
                    object_to_camera=transform,
                    mesh_scale=mesh_scale,
                )
            )
        if not parts:
            return None
        return SceneObjectMesh(tuple(parts))

    def _object_mesh_for_sample(
        self,
        sample: dict[str, Any],
        *,
        require_visibility: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        scene_object = self._object_scene_for_sample(
            sample,
            require_visibility=require_visibility,
        )
        return None if scene_object is None else scene_object.camera_tri_mesh()

    def _object_accelerator_for_scene(
        self,
        scene_object: SceneObjectMesh | None,
    ) -> SceneObjectAccel | None:
        if scene_object is None:
            return None
        try:
            return SceneObjectAccel.from_scene_object(
                scene_object,
                cache=self._object_accel_cache,
            )
        except (OSError, RuntimeError, ValueError):
            return None

    @staticmethod
    def _object_mesh_factory_for_scene(
        scene_object: SceneObjectMesh | None,
    ) -> Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None:
        if scene_object is None:
            return None
        materialized: tuple[torch.Tensor, torch.Tensor] | None = None

        def materialize_camera_mesh() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal materialized
            if materialized is None:
                materialized = scene_object.camera_tri_mesh()
            return materialized

        return materialize_camera_mesh

    def __call__(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        if not chunks:
            raise ValueError("空 batch")
        batch_size = len(chunks)
        num_frames = len(chunks[0]["samples"])
        images = torch.stack([chunk["rgb"] for chunk in chunks])
        depths = _stack_optional_chunk_depths(chunks)
        depth_valid_mask = _build_depth_valid_mask(depths)
        intrinsics, intrinsics_supervision_mask = _stack_optional_sample_tensors(chunks, key="intrinsics", shape=(3, 3))
        camera_pose, camera_pose_supervision_mask = _stack_optional_sample_tensors(chunks, key="camera_pose", shape=(4, 4))
        normalized_hands_by_frame: list[list[list[list[dict[str, Any]]]]] = []
        hand_slots_per_side = 1
        for chunk in chunks:
            chunk_normalized: list[list[list[dict[str, Any]]]] = []
            for sample in chunk["samples"]:
                normalized_by_side = (
                    _normalized_hands_by_side(sample["hand_annos"])
                    if self.hand_supervision_enabled
                    else [[], []]
                )
                if self.hand_supervision_enabled:
                    hand_slots_per_side = max(
                        hand_slots_per_side,
                        len(normalized_by_side[0]),
                        len(normalized_by_side[1]),
                    )
                chunk_normalized.append(normalized_by_side)
            normalized_hands_by_frame.append(chunk_normalized)
        hand_slot_count = 2 * hand_slots_per_side

        marker_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        three_r_supervision_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
        depth_supervision_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
        marker_stage_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
        three_r_stage_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
        fixed_side_hand_count = torch.zeros(batch_size, num_frames, 2, dtype=torch.long)
        fixed_side_invalid_mask = torch.zeros(batch_size, num_frames, 2, dtype=torch.bool)
        presence_targets = torch.zeros(batch_size, num_frames, 2, dtype=torch.float32)
        presence_supervision_mask = torch.zeros(batch_size, num_frames, 2, dtype=torch.bool)
        hand_valid_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        hand_slot_side = torch.full((batch_size, num_frames, hand_slot_count), -1, dtype=torch.long)
        hand_slot_side_offset = torch.full((batch_size, num_frames, hand_slot_count), -1, dtype=torch.long)
        mano_valid_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        joints_valid_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        bbox_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        joints_2d_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        raw_joint_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        joint_visibility_targets = torch.zeros(batch_size, num_frames, hand_slot_count, 21, dtype=torch.bool)
        joint_visibility_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, 21, dtype=torch.bool)
        contact_targets = torch.zeros(batch_size, num_frames, 2, 21, dtype=torch.float32)
        contact_supervision_mask = torch.zeros(batch_size, num_frames, 2, 21, dtype=torch.bool)
        contact_marker_ids = self.marker_vertex_ids or list(MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195)
        marker_contact_count = len(contact_marker_ids)
        marker_contact_targets = torch.zeros(batch_size, num_frames, 2, marker_contact_count, dtype=torch.float32)
        marker_contact_supervision_mask = torch.zeros(
            batch_size,
            num_frames,
            2,
            marker_contact_count,
            dtype=torch.bool,
        )
        contact_distance_targets = torch.zeros(batch_size, num_frames, 2, 21, dtype=torch.float32)
        contact_distance_supervision_mask = torch.zeros(batch_size, num_frames, 2, 21, dtype=torch.bool)
        marker_contact_distance_targets = torch.zeros(batch_size, num_frames, 2, marker_contact_count, dtype=torch.float32)
        marker_contact_distance_supervision_mask = torch.zeros(batch_size, num_frames, 2, marker_contact_count, dtype=torch.bool)
        mano_pose_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        mano_shape_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        mano_trans_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        bbox_targets = torch.full((batch_size, num_frames, hand_slot_count, 4), float("nan"), dtype=torch.float32)
        joints_2d_targets = torch.full((batch_size, num_frames, hand_slot_count, 21, 2), float("nan"), dtype=torch.float32)
        joints_3d_targets = torch.full((batch_size, num_frames, hand_slot_count, 21, 3), float("nan"), dtype=torch.float32)
        mano_pose_axis_angle = torch.full((batch_size, num_frames, hand_slot_count, 48), float("nan"), dtype=torch.float32)
        mano_pose_pca = torch.full((batch_size, num_frames, hand_slot_count, 15), float("nan"), dtype=torch.float32)
        mano_global_orient_axis_angle = torch.full((batch_size, num_frames, hand_slot_count, 3), float("nan"), dtype=torch.float32)
        mano_betas = torch.full((batch_size, num_frames, hand_slot_count, 10), float("nan"), dtype=torch.float32)
        mano_trans = torch.full((batch_size, num_frames, hand_slot_count, 3), float("nan"), dtype=torch.float32)
        wrist_uv_targets = torch.full((batch_size, num_frames, hand_slot_count, 2), float("nan"), dtype=torch.float32)
        wrist_depth_targets = torch.full((batch_size, num_frames, hand_slot_count), float("nan"), dtype=torch.float32)
        wrist_uv_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        wrist_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        marker_vertices = None
        marker_vertex_mask = torch.zeros(batch_size, num_frames, hand_slot_count, dtype=torch.bool)
        vertex_ids = list(self.marker_vertex_ids)
        vertex_2d_targets = torch.full(
            (batch_size, num_frames, hand_slot_count, len(vertex_ids), 2),
            float("nan"),
            dtype=torch.float32,
        )
        vertex_xyz_targets = torch.full((batch_size, num_frames, hand_slot_count, len(vertex_ids), 3), float("nan"), dtype=torch.float32)
        vertex_xyz_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, len(vertex_ids), dtype=torch.bool)
        vertex_visibility_targets = torch.zeros(batch_size, num_frames, hand_slot_count, len(vertex_ids), dtype=torch.bool)
        vertex_visibility_supervision_mask = torch.zeros(batch_size, num_frames, hand_slot_count, len(vertex_ids), dtype=torch.bool)
        mano_faces = self.mano_vertex_builder.build_faces() if self.mano_vertex_builder is not None else torch.zeros((0, 3), dtype=torch.long)
        if self.mano_vertex_builder is not None:
            marker_count = len(self.marker_vertex_ids) if self.marker_vertex_ids else 778
            marker_vertices = torch.full((batch_size, num_frames, hand_slot_count, marker_count, 3), float("nan"), dtype=torch.float32)
        frame_visibility_request_indices: list[tuple[int, int]] = []
        frame_visibility_full_vertices: list[list[torch.Tensor | None]] = []
        frame_visibility_intrinsics: list[torch.Tensor | None] = []
        frame_visibility_vertex_ids: list[list[list[int]]] = []
        scene_visibility_frames: list[
            tuple[
                int,
                int,
                list[torch.Tensor | None],
                list[torch.Tensor | None],
                torch.Tensor | None,
                list[list[int]],
                tuple[torch.Tensor, torch.Tensor] | None,
                SceneObjectAccel | None,
                Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None,
                int | None,
            ]
        ] = []
        lazy_contact_frames: list[
            tuple[
                int,
                int,
                str,
                list[torch.Tensor | None],
                list[torch.Tensor | None],
                tuple[torch.Tensor, torch.Tensor] | None,
                SceneObjectAccel | None,
                Callable[[], tuple[torch.Tensor, torch.Tensor] | None] | None,
            ]
        ] = []

        normalized_hands: list[list[list[dict[str, Any] | None]]] = []
        for batch_index, chunk in enumerate(chunks):
            chunk_normalized: list[list[dict[str, Any] | None]] = []
            for frame_index, sample in enumerate(chunk["samples"]):
                normalized_by_side = normalized_hands_by_frame[batch_index][frame_index]
                sample_normalized: list[dict[str, Any] | None] = [None for _ in range(hand_slot_count)]
                image_height, image_width = sample["rgb"].shape[-2:]
                intrinsics_for_vis = sample.get("intrinsics")
                frame_full_vertices: list[torch.Tensor | None] = [None for _ in range(hand_slot_count)]
                frame_mano_joints: list[torch.Tensor | None] = [None for _ in range(hand_slot_count)]
                frame_vertex_id_lists: list[list[int]] = [list(vertex_ids) for _ in range(hand_slot_count)]
                if sample_allows_three_r_supervision(
                    sample,
                    stage=self.stage,
                    stream_name=self.stream_name,
                ):
                    three_r_stage_mask[batch_index, frame_index] = True
                if sample.get("depth") is not None:
                    depth_supervision_mask[batch_index, frame_index] = True
                if sample_allows_marker_supervision(
                    sample,
                    stage=self.stage,
                    stream_name=self.stream_name,
                ):
                    marker_stage_mask[batch_index, frame_index] = True
                if not self.hand_supervision_enabled:
                    chunk_normalized.append(sample_normalized)
                    continue
                for side_index, side_hands in enumerate(normalized_by_side):
                    fixed_side_hand_count[batch_index, frame_index, side_index] = len(side_hands)
                    fixed_side_invalid_mask[batch_index, frame_index, side_index] = len(side_hands) > 1
                    for side_offset, hand_target in enumerate(side_hands[:hand_slots_per_side]):
                        if self.mano_vertex_builder is not None:
                            hand_target = self.mano_vertex_builder.prepare_hand_target(hand_target)
                        hand_slot = side_index * hand_slots_per_side + side_offset
                        sample_normalized[hand_slot] = hand_target
                        hand_slot_side[batch_index, frame_index, hand_slot] = side_index
                        hand_slot_side_offset[batch_index, frame_index, hand_slot] = side_offset
                        hand_valid_mask[batch_index, frame_index, hand_slot] = True
                        mano_valid_mask[batch_index, frame_index, hand_slot] = hand_target["mano_valid"]
                        joints_valid_mask[batch_index, frame_index, hand_slot] = hand_target["joints_valid"]
                        contact_mode = _contact_supervision_mode(
                            sample,
                            self.contact_supervision_by_dataset,
                        )
                        bbox_xyxy = _derive_hand_bbox_xyxy(hand_target, intrinsics_for_vis)
                        if hand_target["joints_2d"] is not None:
                            joints_2d_supervision_mask[batch_index, frame_index, hand_slot] = True
                            joints_2d_targets[batch_index, frame_index, hand_slot] = hand_target["joints_2d"]
                        if hand_target["joints_3d"] is not None:
                            raw_joint_supervision_mask[batch_index, frame_index, hand_slot] = True
                            joints_3d_targets[batch_index, frame_index, hand_slot] = hand_target["joints_3d"]
                        projected_joint_vis, projected_joint_vis_mask = _joint_visibility_targets_from_projection(
                            hand_target["joints_3d"],
                            intrinsics_for_vis,
                            image_height=image_height,
                            image_width=image_width,
                            depth=sample.get("depth"),
                        )
                        joint_visibility_targets[batch_index, frame_index, hand_slot] = projected_joint_vis
                        joint_visibility_supervision_mask[batch_index, frame_index, hand_slot] = projected_joint_vis_mask
                        if hand_target["mano_pose_axis_angle"] is not None:
                            mano_pose_supervision_mask[batch_index, frame_index, hand_slot] = True
                            mano_pose_axis_angle[batch_index, frame_index, hand_slot] = hand_target["mano_pose_axis_angle"]
                        if hand_target["mano_pose_pca"] is not None:
                            mano_pose_supervision_mask[batch_index, frame_index, hand_slot] = True
                            mano_pose_pca[batch_index, frame_index, hand_slot] = hand_target["mano_pose_pca"]
                        if hand_target["mano_global_orient_axis_angle"] is not None:
                            mano_global_orient_axis_angle[batch_index, frame_index, hand_slot] = hand_target[
                                "mano_global_orient_axis_angle"
                            ]
                        if hand_target["mano_betas"] is not None:
                            mano_shape_supervision_mask[batch_index, frame_index, hand_slot] = True
                            mano_betas[batch_index, frame_index, hand_slot] = hand_target["mano_betas"]
                        if hand_target["mano_trans"] is not None:
                            mano_trans_supervision_mask[batch_index, frame_index, hand_slot] = True
                            mano_trans[batch_index, frame_index, hand_slot] = hand_target["mano_trans"]
                        full_vertices = None
                        mano_joints = None
                        if self.mano_vertex_builder is not None:
                            full_vertices, mano_joints = self.mano_vertex_builder.build_vertices_and_joints(
                                hand_target,
                                marker_vertex_ids=None,
                            )
                            if full_vertices is not None:
                                frame_full_vertices[hand_slot] = full_vertices
                                frame_vertex_id_lists[hand_slot] = (
                                    vertex_ids if vertex_ids else list(range(int(full_vertices.shape[0])))
                                )
                                vertices = (
                                    full_vertices[torch.as_tensor(vertex_ids, dtype=torch.long)]
                                    if vertex_ids
                                    else full_vertices
                                )
                                vertex_xyz_targets[batch_index, frame_index, hand_slot] = vertices
                                finite_vertex_mask = torch.isfinite(vertices).all(dim=-1)
                                vertex_xyz_supervision_mask[batch_index, frame_index, hand_slot] = finite_vertex_mask
                                if intrinsics_for_vis is not None:
                                    projected_vertices, projected_valid = _project_points(
                                        vertices,
                                        torch.as_tensor(intrinsics_for_vis, dtype=torch.float32),
                                    )
                                    vertex_2d_targets[batch_index, frame_index, hand_slot] = torch.where(
                                        projected_valid.unsqueeze(-1),
                                        projected_vertices,
                                        torch.full_like(projected_vertices, float("nan")),
                                    )
                                marker_vertices[batch_index, frame_index, hand_slot] = vertices
                                marker_vertex_mask[batch_index, frame_index, hand_slot] = True
                            if mano_joints is not None and tuple(mano_joints.shape) == (21, 3):
                                frame_mano_joints[hand_slot] = mano_joints.to(dtype=torch.float32)
                            if (
                                hand_target["joints_3d"] is None
                                and mano_joints is not None
                                and tuple(mano_joints.shape) == (21, 3)
                                and bool(torch.isfinite(mano_joints).all(dim=-1).any().item())
                            ):
                                mano_joints = mano_joints.to(dtype=torch.float32)
                                joints_valid_mask[batch_index, frame_index, hand_slot] = True
                                raw_joint_supervision_mask[batch_index, frame_index, hand_slot] = True
                                joints_3d_targets[batch_index, frame_index, hand_slot] = mano_joints
                                projected_joint_vis, projected_joint_vis_mask = _joint_visibility_targets_from_projection(
                                    mano_joints,
                                    intrinsics_for_vis,
                                    image_height=image_height,
                                    image_width=image_width,
                                    depth=sample.get("depth"),
                                )
                                joint_visibility_targets[batch_index, frame_index, hand_slot] = projected_joint_vis
                                joint_visibility_supervision_mask[batch_index, frame_index, hand_slot] = projected_joint_vis_mask
                        projected_joints, projected_joint_mask = _joint_2d_targets_from_projection(
                            hand_target["joints_3d"],
                            intrinsics_for_vis,
                        )
                        if projected_joints is None:
                            projected_joints, projected_joint_mask = _joint_2d_targets_from_projection(
                                mano_joints,
                                intrinsics_for_vis,
                            )
                        merged_joints_2d, has_joint_2d = _merge_joint_2d_targets(
                            joints_2d_targets[batch_index, frame_index, hand_slot],
                            projected_joints,
                            projected_joint_mask,
                        )
                        if has_joint_2d:
                            joints_2d_targets[batch_index, frame_index, hand_slot] = merged_joints_2d
                            joints_2d_supervision_mask[batch_index, frame_index, hand_slot] = True
                        if (
                            len(side_hands) == 1
                            and side_offset == 0
                            and _hand_has_reliable_visible_evidence(
                                hand_target,
                                intrinsics=intrinsics_for_vis,
                                image_height=image_height,
                                image_width=image_width,
                                full_vertices=full_vertices,
                                mano_joints=mano_joints,
                            )
                        ):
                            presence_targets[batch_index, frame_index, side_index] = 1.0
                        if marker_stage_mask[batch_index, frame_index]:
                            wrist_uv, wrist_depth = _wrist_uv_depth_from_points(
                                hand_target["joints_3d"],
                                hand_target["joints_2d"],
                                intrinsics_for_vis,
                                image_height=image_height,
                                image_width=image_width,
                                require_in_image=False,
                            )
                            if wrist_uv is None:
                                wrist_uv, wrist_depth = _wrist_uv_depth_from_points(
                                    mano_joints,
                                    None,
                                    intrinsics_for_vis,
                                    image_height=image_height,
                                    image_width=image_width,
                                    require_in_image=False,
                                )
                            if wrist_uv is not None and wrist_depth is not None:
                                wrist_uv_targets[batch_index, frame_index, hand_slot] = wrist_uv
                                wrist_depth_targets[batch_index, frame_index, hand_slot] = wrist_depth
                                wrist_uv_supervision_mask[batch_index, frame_index, hand_slot] = True
                                if bool(_points_in_image(wrist_uv, image_height, image_width).item()):
                                    wrist_supervision_mask[batch_index, frame_index, hand_slot] = True
                        if bbox_xyxy is None and full_vertices is not None:
                            bbox_xyxy = _bbox_from_projected_points_3d(full_vertices, intrinsics_for_vis)
                        if bbox_xyxy is not None:
                            bbox_supervision_mask[batch_index, frame_index, hand_slot] = True
                            bbox_targets[batch_index, frame_index, hand_slot] = bbox_xyxy
                scene_visibility_enabled = self.mano_vertex_builder is not None
                contact_mode = _contact_supervision_mode(
                    sample,
                    self.contact_supervision_by_dataset,
                )
                needs_scene_object = self.mano_vertex_builder is not None and (
                    contact_mode != "disabled" or self.include_scene_occlusion_in_visibility
                )
                scene_object = self._object_scene_for_sample(sample) if needs_scene_object else None
                object_accelerator = self._object_accelerator_for_scene(scene_object)
                object_face_count = None if scene_object is None else scene_object.face_count
                object_mesh_factory = self._object_mesh_factory_for_scene(scene_object)

                # Low-face objects still use the exact z-buffer path.  Contact
                # and joint visibility keep this factory cold while BVH works.
                object_mesh = None
                if (
                    self.include_scene_occlusion_in_visibility
                    and scene_object is not None
                    and (
                        object_accelerator is None
                        or object_face_count is None
                        or object_face_count <= SCENE_VISIBILITY_OBJECT_RASTER_MAX_FACES
                    )
                ):
                    object_mesh = object_mesh_factory()
                elif (
                    object_accelerator is None
                    and contact_mode != "disabled"
                    and object_mesh_factory is not None
                ):
                    object_mesh = object_mesh_factory()
                if self.mano_vertex_builder is not None and contact_mode != "disabled":
                    lazy_contact_frames.append(
                        (
                            batch_index,
                            frame_index,
                            contact_mode,
                            [
                                frame_full_vertices[side * hand_slots_per_side]
                                if int(fixed_side_hand_count[batch_index, frame_index, side].item()) == 1
                                else None
                                for side in range(2)
                            ],
                            [
                                frame_mano_joints[side * hand_slots_per_side]
                                if int(fixed_side_hand_count[batch_index, frame_index, side].item()) == 1
                                else None
                                for side in range(2)
                            ],
                            object_mesh,
                            object_accelerator,
                            object_mesh_factory,
                        )
                    )
                if self.mano_vertex_builder is not None and scene_visibility_enabled:
                    scene_visibility_frames.append(
                        (
                            batch_index,
                            frame_index,
                            frame_full_vertices,
                            frame_mano_joints,
                            intrinsics_for_vis,
                            frame_vertex_id_lists,
                            object_mesh if self.include_scene_occlusion_in_visibility else None,
                            object_accelerator if self.include_scene_occlusion_in_visibility else None,
                            object_mesh_factory if self.include_scene_occlusion_in_visibility else None,
                            object_face_count if self.include_scene_occlusion_in_visibility else None,
                        )
                    )
                elif self.mano_vertex_builder is not None:
                    frame_visibility_request_indices.append((batch_index, frame_index))
                    frame_visibility_full_vertices.append(frame_full_vertices)
                    frame_visibility_intrinsics.append(intrinsics_for_vis)
                    frame_visibility_vertex_ids.append(frame_vertex_id_lists)
                chunk_normalized.append(sample_normalized)
            normalized_hands.append(chunk_normalized)

        if self.mano_vertex_builder is not None and frame_visibility_request_indices:
            batched_vertex_visibility = _batched_vertex_visibility_targets_from_mesh(
                frame_visibility_full_vertices,
                mano_faces,
                frame_visibility_intrinsics,
                image_height=images.shape[-2],
                image_width=images.shape[-1],
                frame_vertex_ids_list=frame_visibility_vertex_ids,
                visibility_mesh_mode=self.vertex_visibility_mesh_mode,
                marker_faces=self.marker_visibility_faces,
            )
            for (batch_index, frame_index), frame_outputs in zip(
                frame_visibility_request_indices,
                batched_vertex_visibility,
                strict=True,
            ):
                for hand_slot, (vertex_vis_targets, vertex_vis_mask) in enumerate(frame_outputs):
                    vertex_visibility_targets[batch_index, frame_index, hand_slot] = vertex_vis_targets
                    vertex_visibility_supervision_mask[batch_index, frame_index, hand_slot] = vertex_vis_mask

        if self.mano_vertex_builder is not None and scene_visibility_frames:
            scene_device = self.scene_visibility_device
            if scene_device is None:
                scene_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            scene_marker_outputs = _bounded_scene_marker_visibility_targets_from_mesh(
                [
                    (full_vertices, intrinsics_for_vis, frame_vertex_ids, object_mesh)
                    for _, _, full_vertices, _, intrinsics_for_vis, frame_vertex_ids, object_mesh, _, _, _ in scene_visibility_frames
                ],
                mano_faces,
                image_height=images.shape[-2],
                image_width=images.shape[-1],
                device=scene_device,
                depths=[
                    None if depths is None else depths[batch_index, frame_index]
                    for batch_index, frame_index, *_ in scene_visibility_frames
                ],
                object_ray_accelerators=[
                    object_ray_accelerator
                    for *_, object_ray_accelerator, _, _ in scene_visibility_frames
                ],
                object_mesh_factories=[
                    object_mesh_factory
                    for *_, object_mesh_factory, _ in scene_visibility_frames
                ],
                object_face_counts=[
                    object_face_count
                    for *_, object_face_count in scene_visibility_frames
                ],
            )
            for (
                (
                    batch_index,
                    frame_index,
                    full_vertices,
                    mano_joints,
                    intrinsics_for_vis,
                    frame_vertex_ids,
                    object_mesh,
                    object_ray_accelerator,
                    object_mesh_factory,
                    object_face_count,
                ),
                scene_vertex_visibility,
            ) in zip(scene_visibility_frames, scene_marker_outputs, strict=True):
                for hand_slot, (targets, supervision_mask) in enumerate(scene_vertex_visibility):
                    vertex_visibility_targets[batch_index, frame_index, hand_slot] = targets
                    vertex_visibility_supervision_mask[batch_index, frame_index, hand_slot] = supervision_mask
                scene_joint_visibility = _joint_visibility_targets_from_scene(
                    mano_joints,
                    full_vertices,
                    mano_faces,
                    intrinsics_for_vis,
                    image_height=images.shape[-2],
                    image_width=images.shape[-1],
                    object_mesh=object_mesh,
                    object_mesh_factory=object_mesh_factory,
                    object_ray_accelerator=object_ray_accelerator,
                    depth=None if depths is None else depths[batch_index, frame_index],
                    scene_device=scene_device,
                )
                for hand_slot, (targets, supervision_mask) in enumerate(scene_joint_visibility):
                    if mano_joints[hand_slot] is not None:
                        joint_visibility_targets[batch_index, frame_index, hand_slot] = targets
                        joint_visibility_supervision_mask[batch_index, frame_index, hand_slot] = supervision_mask

        for (
            batch_index,
            frame_index,
            contact_mode,
            side_vertices,
            side_joints,
            object_mesh,
            object_accelerator,
            object_mesh_factory,
        ) in lazy_contact_frames:
            lazy_contact = lazy_contact_targets_from_scene(
                side_vertices=side_vertices,
                side_joints=side_joints,
                mano_faces=mano_faces,
                marker_vertex_ids=contact_marker_ids,
                object_mesh=object_mesh,
                mode=contact_mode,
                object_distance_accelerator=object_accelerator,
                object_mesh_factory=object_mesh_factory,
                interhand_compute_device=self.interhand_contact_compute_device,
            )
            contact_targets[batch_index, frame_index] = lazy_contact.joint_targets
            contact_supervision_mask[batch_index, frame_index] = lazy_contact.joint_supervision_mask
            contact_distance_targets[batch_index, frame_index] = lazy_contact.joint_distances
            contact_distance_supervision_mask[batch_index, frame_index] = lazy_contact.joint_distance_supervision_mask
            marker_contact_targets[batch_index, frame_index] = lazy_contact.marker_targets
            marker_contact_supervision_mask[batch_index, frame_index] = lazy_contact.marker_supervision_mask
            marker_contact_distance_targets[batch_index, frame_index] = lazy_contact.marker_distances
            marker_contact_distance_supervision_mask[batch_index, frame_index] = lazy_contact.marker_distance_supervision_mask

        if depth_valid_mask is not None:
            depth_supervision_mask &= depth_valid_mask.any(dim=(-2, -1))
        intrinsics_supervision_mask &= torch.isfinite(intrinsics).all(dim=(-2, -1))
        camera_pose_supervision_mask &= torch.isfinite(camera_pose).all(dim=(-2, -1))
        marker_supervision_mask = (
            bbox_supervision_mask
            | joints_2d_supervision_mask
            | raw_joint_supervision_mask
            | mano_pose_supervision_mask
            | marker_vertex_mask
        ) & marker_stage_mask.unsqueeze(-1)
        presence_supervision_mask = marker_stage_mask.unsqueeze(-1) & ~fixed_side_invalid_mask
        three_r_supervision_mask = (
            depth_supervision_mask
            | intrinsics_supervision_mask
            | camera_pose_supervision_mask
        ) & three_r_stage_mask
        batch_sources = [
            {
                "dataset_name": str(chunk.get("dataset_name", "")),
                "base_dataset_name": str(chunk.get("base_dataset_name", "")),
                "sequence_id": str(chunk.get("sequence_id", "")),
                "view_name": str(chunk.get("view_name", "")),
                "frame_ids": [str(frame_id) for frame_id in chunk.get("frame_ids", [])],
                "temporal_indices": [int(index) for index in chunk.get("temporal_indices", [])],
            }
            for chunk in chunks
        ]

        batch = {
            "stage": self.stage,
            "batch_sources": batch_sources,
            "images": images,
            "depth": depths,
            "depth_valid_mask": depth_valid_mask,
            "intrinsics": intrinsics,
            "camera_pose": camera_pose,
            "normalized_hands": normalized_hands,
            "marker_supervision_mask": marker_supervision_mask,
            "presence_targets": presence_targets,
            "presence_supervision_mask": presence_supervision_mask,
            "fixed_side_hand_count": fixed_side_hand_count,
            "fixed_side_invalid_mask": fixed_side_invalid_mask,
            "three_r_supervision_mask": three_r_supervision_mask,
            "depth_supervision_mask": depth_supervision_mask,
            "camera_pose_supervision_mask": camera_pose_supervision_mask,
            "intrinsics_supervision_mask": intrinsics_supervision_mask,
            "hand_valid_mask": hand_valid_mask,
            "hand_slots_per_side": hand_slots_per_side,
            "hand_slot_count": hand_slot_count,
            "hand_slot_side": hand_slot_side,
            "hand_slot_side_offset": hand_slot_side_offset,
            "mano_valid_mask": mano_valid_mask,
            "joints_valid_mask": joints_valid_mask,
            "bbox_supervision_mask": bbox_supervision_mask,
            "joints_2d_supervision_mask": joints_2d_supervision_mask,
            "raw_joint_supervision_mask": raw_joint_supervision_mask,
            "joint_visibility_targets": joint_visibility_targets,
            "joint_visibility_supervision_mask": joint_visibility_supervision_mask,
            "contact_targets": contact_targets,
            "contact_supervision_mask": contact_supervision_mask,
            "marker_contact_targets": marker_contact_targets,
            "marker_contact_supervision_mask": marker_contact_supervision_mask,
            "contact_distance_targets": contact_distance_targets,
            "contact_distance_supervision_mask": contact_distance_supervision_mask,
            "marker_contact_distance_targets": marker_contact_distance_targets,
            "marker_contact_distance_supervision_mask": marker_contact_distance_supervision_mask,
            "mano_pose_supervision_mask": mano_pose_supervision_mask,
            "mano_shape_supervision_mask": mano_shape_supervision_mask,
            "mano_trans_supervision_mask": mano_trans_supervision_mask,
            "bbox_targets": bbox_targets,
            "wrist_uv_targets": wrist_uv_targets,
            "wrist_depth_targets": wrist_depth_targets,
            "wrist_uv_supervision_mask": wrist_uv_supervision_mask,
            "wrist_supervision_mask": wrist_supervision_mask,
            "joints_2d_targets": joints_2d_targets,
            "joints_3d_targets": joints_3d_targets,
            "mano_pose_axis_angle": mano_pose_axis_angle,
            "mano_pose_pca": mano_pose_pca,
            "mano_global_orient_axis_angle": mano_global_orient_axis_angle,
            "mano_betas": mano_betas,
            "mano_trans": mano_trans,
            "vertex_ids": vertex_ids,
            "vertex_2d_targets": vertex_2d_targets,
            "vertex_xyz_targets": vertex_xyz_targets,
            "vertex_xyz_supervision_mask": vertex_xyz_supervision_mask,
            "vertex_visibility_targets": vertex_visibility_targets,
            "vertex_visibility_supervision_mask": vertex_visibility_supervision_mask,
            "marker_vertex_ids": self.marker_vertex_ids,
            "marker_vertices": marker_vertices,
            "marker_vertex_mask": marker_vertex_mask,
        }
        if chunks and all("flow_pseudo_target" in chunk for chunk in chunks):
            batch["flow_pseudo_target"] = torch.stack([chunk["flow_pseudo_target"] for chunk in chunks])
            batch["flow_pseudo_valid"] = torch.stack([chunk["flow_pseudo_valid"] for chunk in chunks])
            batch["flow_hand_region_mask"] = torch.stack(
                [
                    chunk.get(
                        "flow_hand_region_mask",
                        torch.zeros_like(chunk["flow_pseudo_valid"], dtype=torch.bool),
                    )
                    for chunk in chunks
                ]
            )
            batch["flow_fingertip_weight"] = torch.stack(
                [
                    chunk.get(
                        "flow_fingertip_weight",
                        torch.zeros_like(chunk["flow_pseudo_valid"], dtype=torch.float32),
                    ).to(dtype=torch.float32)
                    for chunk in chunks
                ]
            )
            batch["flow_pair_mask"] = torch.stack([chunk["flow_pair_mask"] for chunk in chunks])
        return batch


class FilteredStageDataset:
    def __init__(self, dataset: Sequence[Any], predicate: Callable[[Any], bool]) -> None:
        self.dataset = dataset
        self.indices = [index for index in range(len(dataset)) if predicate(dataset[index])]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]


class FilteredRandomLengthTemporalChunkDataset:
    def __init__(self, dataset: Any, predicate: Callable[[Any], bool], *, eager: bool = True) -> None:
        self.dataset = dataset
        self.predicate = predicate
        self.eager = eager
        self.frame_dataset = dataset.frame_dataset
        self.min_num_frames = dataset.min_num_frames
        self.max_num_frames = dataset.max_num_frames
        self.window_stride = dataset.window_stride
        self.drop_last = dataset.drop_last
        self._valid_anchor_cache: dict[int, list[int]] = {}
        self._sequence_anchor_position_cache: dict[int, tuple[list[int], list[list[int]]]] = {}

    @property
    def sequence_count(self) -> int:
        return self.dataset.sequence_count

    def sequence_length(self, sequence_index: int) -> int:
        return self.dataset.sequence_length(sequence_index)

    def valid_start_count(self, sequence_index: int, *, num_frames: int, frame_stride: int) -> int:
        return self.dataset.valid_start_count(sequence_index, num_frames=num_frames, frame_stride=frame_stride)

    def _valid_anchor_indices(self, num_frames: int) -> list[int]:
        cached = self._valid_anchor_cache.get(num_frames)
        if cached is not None:
            return cached
        valid_indices = [
            anchor_index
            for anchor_index in range(self.dataset.valid_anchor_count(num_frames))
            if self.predicate(self.dataset[(anchor_index, num_frames)])
        ]
        self._valid_anchor_cache[num_frames] = valid_indices
        return valid_indices

    def valid_anchor_count(self, num_frames: int) -> int:
        if not self.eager:
            return self.dataset.valid_anchor_count(num_frames)
        return len(self._valid_anchor_indices(num_frames))

    def sequence_anchor_counts(self, num_frames: int) -> list[int]:
        if not self.eager:
            return self.dataset.sequence_anchor_counts(num_frames)
        counts, _ = self._sequence_anchor_positions(num_frames)
        return list(counts)

    def _sequence_anchor_positions(self, num_frames: int) -> tuple[list[int], list[list[int]]]:
        cached = self._sequence_anchor_position_cache.get(num_frames)
        if cached is not None:
            return cached
        raw_counts = self.dataset.sequence_anchor_counts(num_frames)
        offsets: list[int] = []
        running_total = 0
        for raw_count in raw_counts:
            offsets.append(running_total)
            running_total += int(raw_count)
        positions = [[] for _ in raw_counts]
        for filtered_anchor_index, original_anchor_index in enumerate(self._valid_anchor_indices(num_frames)):
            sequence_index = bisect_right(offsets, int(original_anchor_index)) - 1
            if sequence_index < 0 or sequence_index >= len(positions):
                raise IndexError(original_anchor_index)
            positions[sequence_index].append(filtered_anchor_index)
        counts = [len(sequence_positions) for sequence_positions in positions]
        cached = (counts, positions)
        self._sequence_anchor_position_cache[num_frames] = cached
        return cached

    def global_anchor_index(self, sequence_index: int, local_anchor_index: int, num_frames: int) -> int:
        if not self.eager:
            return self.dataset.global_anchor_index(sequence_index, local_anchor_index, num_frames)
        sequence_counts, sequence_positions = self._sequence_anchor_positions(num_frames)
        if sequence_index < 0 or sequence_index >= len(sequence_counts):
            raise IndexError(sequence_index)
        if local_anchor_index < 0 or local_anchor_index >= sequence_counts[sequence_index]:
            raise IndexError(local_anchor_index)
        return sequence_positions[sequence_index][local_anchor_index]

    def __len__(self) -> int:
        return self.valid_anchor_count(self.min_num_frames)

    def _next_valid_anchor_chunk(self, anchor_index: int, num_frames: int) -> dict[str, Any]:
        raw_count = self.dataset.valid_anchor_count(num_frames)
        if raw_count <= 0:
            raise IndexError((anchor_index, num_frames))
        if anchor_index < 0:
            anchor_index += raw_count
        if anchor_index < 0 or anchor_index >= raw_count:
            raise IndexError(anchor_index)
        for offset in range(raw_count):
            candidate_index = (anchor_index + offset) % raw_count
            chunk = self.dataset[(candidate_index, num_frames)]
            if self.predicate(chunk):
                return chunk
        raise IndexError("当前数据集没有满足过滤条件的随机长度窗口。")

    def _next_valid_sequence_chunk(
        self,
        sequence_index: int,
        start: int,
        num_frames: int,
        frame_stride: int,
        target_height: int,
        target_width: int,
    ) -> dict[str, Any]:
        sequence_count = self.dataset.sequence_count
        if sequence_count <= 0:
            raise IndexError(sequence_index)
        if sequence_index < 0:
            sequence_index += sequence_count
        if sequence_index < 0 or sequence_index >= sequence_count:
            raise IndexError(sequence_index)
        start_slot = max(0, int(start) // int(self.dataset.window_stride))
        for sequence_offset in range(sequence_count):
            candidate_sequence = (sequence_index + sequence_offset) % sequence_count
            valid_start_count = self.dataset.valid_start_count(
                candidate_sequence,
                num_frames=num_frames,
                frame_stride=frame_stride,
            )
            if valid_start_count <= 0:
                continue
            first_slot = start_slot if sequence_offset == 0 else 0
            for slot_offset in range(valid_start_count):
                candidate_slot = (first_slot + slot_offset) % valid_start_count
                candidate_start = candidate_slot * int(self.dataset.window_stride)
                chunk = self.dataset[
                    (
                        candidate_sequence,
                        candidate_start,
                        num_frames,
                        frame_stride,
                        target_height,
                        target_width,
                    )
                ]
                if self.predicate(chunk):
                    return chunk
        raise IndexError("当前数据集没有满足过滤条件的随机长度窗口。")

    def __getitem__(self, index: int | tuple[int, ...]) -> dict[str, Any]:
        if isinstance(index, tuple) and len(index) == 6:
            sequence_index, start, num_frames, frame_stride, target_height, target_width = (int(value) for value in index)
            if not self.eager:
                return self._next_valid_sequence_chunk(
                    sequence_index,
                    start,
                    num_frames,
                    frame_stride,
                    target_height,
                    target_width,
                )
            chunk = self.dataset[(sequence_index, start, num_frames, frame_stride, target_height, target_width)]
            if self.predicate(chunk):
                return chunk
            return self._next_valid_sequence_chunk(
                sequence_index,
                start,
                num_frames,
                frame_stride,
                target_height,
                target_width,
            )
        if isinstance(index, tuple):
            anchor_index, num_frames = index
        else:
            anchor_index = index
            num_frames = self.min_num_frames
        anchor_index = int(anchor_index)
        num_frames = int(num_frames)
        if not self.eager:
            return self._next_valid_anchor_chunk(anchor_index, num_frames)
        valid_indices = self._valid_anchor_indices(num_frames)
        if anchor_index < 0:
            anchor_index += len(valid_indices)
        if anchor_index < 0 or anchor_index >= len(valid_indices):
            raise IndexError(anchor_index)
        return self.dataset[(valid_indices[anchor_index], num_frames)]

    def epoch_sample_count(self) -> int:
        return sum(
            self.valid_anchor_count(num_frames)
            for num_frames in range(self.min_num_frames, self.max_num_frames + 1)
        )

    def epoch_samples(self) -> list[tuple[int, int]]:
        samples: list[tuple[int, int]] = []
        for num_frames in range(self.min_num_frames, self.max_num_frames + 1):
            samples.extend((anchor_index, num_frames) for anchor_index in range(self.valid_anchor_count(num_frames)))
        return samples


@dataclass(slots=True)
class StageDatasetBundle:
    stage: str
    marker_dataset: Any
    three_r_dataset: Any
    marker_dataset_names: Sequence[str]
    three_r_dataset_names: Sequence[str]

    @property
    def marker_sample_count(self) -> int:
        return len(self.marker_dataset)

    @property
    def three_r_sample_count(self) -> int:
        return len(self.three_r_dataset)


def build_stage_frame_bundle(
    *,
    stage: str,
    marker_dataset_names: Sequence[str],
    three_r_dataset_names: Sequence[str],
    data_root: str | None = None,
    dataset_kwargs_by_name: dict[str, dict[str, Any]] | None = None,
    **dataset_kwargs,
) -> StageDatasetBundle:
    frame_dataset_cache: dict[str, Any] = {}

    def _frame_dataset(name: str) -> Any:
        cached = frame_dataset_cache.get(name)
        if cached is not None:
            return cached
        local_kwargs = dict(dataset_kwargs)
        if dataset_kwargs_by_name is not None and name in dataset_kwargs_by_name:
            local_kwargs.update(dataset_kwargs_by_name[name])
        dataset = build_named_frame_dataset(name, data_root=data_root, **local_kwargs)
        frame_dataset_cache[name] = dataset
        return dataset

    marker_sources = [_frame_dataset(name) for name in marker_dataset_names]
    three_r_sources = [_frame_dataset(name) for name in three_r_dataset_names]
    marker_dataset = ConcatDataset(marker_sources) if marker_sources else []
    three_r_dataset = ConcatDataset(three_r_sources) if three_r_sources else []
    return StageDatasetBundle(
        stage=stage,
        marker_dataset=marker_dataset,
        three_r_dataset=three_r_dataset,
        marker_dataset_names=marker_dataset_names,
        three_r_dataset_names=three_r_dataset_names,
    )


def _chunk_allows_marker_supervision(chunk: dict[str, Any], *, stage: str) -> bool:
    return any(sample_allows_marker_supervision(sample, stage=stage) for sample in chunk["samples"])


def chunk_allows_fixed_query_marker_supervision(chunk: dict[str, Any], *, stage: str) -> bool:
    if not isinstance(chunk, dict) or "samples" not in chunk:
        return True
    samples = chunk["samples"]
    for sample in samples:
        if int(sample.get("max_left_count", 1)) > 1:
            return False
        if int(sample.get("max_right_count", 1)) > 1:
            return False
    return True


def _chunk_allows_three_r_supervision(chunk: dict[str, Any], *, stage: str) -> bool:
    return any(sample_allows_three_r_supervision(sample, stage=stage) for sample in chunk["samples"])


def build_stage_chunk_bundle(
    *,
    stage: str,
    marker_dataset_names: Sequence[str],
    three_r_dataset_names: Sequence[str],
    num_frames: int,
    window_stride: int = 1,
    data_root: str | None = None,
    split_overrides: dict[str, str] | None = None,
    dataset_kwargs_by_name: dict[str, dict[str, Any]] | None = None,
    **dataset_kwargs,
) -> StageDatasetBundle:
    frame_dataset_cache: dict[str, Any] = {}

    def _frame_dataset(name: str) -> Any:
        cached = frame_dataset_cache.get(name)
        if cached is not None:
            return cached
        local_kwargs = dict(dataset_kwargs)
        if split_overrides is not None and name in split_overrides:
            local_kwargs["split"] = split_overrides[name]
        if dataset_kwargs_by_name is not None and name in dataset_kwargs_by_name:
            local_kwargs.update(dataset_kwargs_by_name[name])
        dataset = build_named_frame_dataset(name, data_root=data_root, **local_kwargs)
        frame_dataset_cache[name] = dataset
        return dataset

    marker_sources = [
        TemporalChunkDataset(
            _frame_dataset(name),
            num_frames=num_frames,
            window_stride=window_stride,
        )
        for name in marker_dataset_names
    ]
    three_r_sources = [
        TemporalChunkDataset(
            _frame_dataset(name),
            num_frames=num_frames,
            window_stride=window_stride,
        )
        for name in three_r_dataset_names
    ]
    marker_dataset = ConcatDataset(marker_sources) if marker_sources else []
    three_r_dataset = ConcatDataset(three_r_sources) if three_r_sources else []
    return StageDatasetBundle(
        stage=stage,
        marker_dataset=marker_dataset,
        three_r_dataset=three_r_dataset,
        marker_dataset_names=marker_dataset_names,
        three_r_dataset_names=three_r_dataset_names,
    )


class DualStreamBatchMixer:
    def __init__(self, marker_batches: Iterable[Any], three_r_batches: Iterable[Any]) -> None:
        self.marker_batches = marker_batches
        self.three_r_batches = three_r_batches

    def __iter__(self) -> Iterator[dict[str, Any]]:
        marker_iter = iter(self.marker_batches)
        three_r_iter = iter(self.three_r_batches)
        marker_done = False
        three_r_done = False
        while not (marker_done and three_r_done):
            if not marker_done:
                try:
                    yield {"stream_name": "marker", "batch": next(marker_iter)}
                except StopIteration:
                    marker_done = True
            if not three_r_done:
                try:
                    yield {"stream_name": "three_r", "batch": next(three_r_iter)}
                except StopIteration:
                    three_r_done = True
