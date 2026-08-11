"""H2O ground-truth loading for the formal evaluation protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np


MANO_FINGERTIP_VERTEX_IDS = (744, 320, 443, 554, 671)
MANO_OPENPOSE_JOINT_ORDER = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)


def _load_mano_model(mano_dir: Path, side: str):
    import smplx

    model = smplx.create(
        model_path=str(mano_dir / f"MANO_{side.upper()}.pkl"),
        model_type="mano",
        is_rhand=side == "right",
        use_pca=False,
        flat_hand_mean=True,
    )
    model.eval()
    return model


def _reconstruct_mano(model, root_orient, hand_pose, betas, translation):
    import torch

    batch_size = root_orient.shape[0]
    with torch.no_grad():
        output = model(
            global_orient=torch.from_numpy(root_orient).float().reshape(batch_size, 3),
            hand_pose=torch.from_numpy(hand_pose).float().reshape(batch_size, 45),
            betas=torch.from_numpy(betas).float(),
            transl=torch.from_numpy(translation).float(),
        )
    joints_16 = output.joints[:, :16]
    tips = output.vertices[:, list(MANO_FINGERTIP_VERTEX_IDS)]
    joints = torch.cat([joints_16, tips], dim=1)[:, list(MANO_OPENPOSE_JOINT_ORDER)]
    return joints.numpy(), output.vertices.numpy()


class H2OGTLoader:
    """Read H2O camera-space hand, depth, camera, and contact GT."""

    def __init__(self, data_root: Path, mano_dir: Path):
        self.data_root = data_root
        self.mano_dir = mano_dir
        self._mano_models: dict[str, object] = {}

    def _mano(self, side: str):
        if side not in self._mano_models:
            self._mano_models[side] = _load_mano_model(self.mano_dir, side)
        return self._mano_models[side]

    def _camera_dir(self, sequence: str) -> Path:
        return self.data_root / sequence / "cam4"

    def load_hand_gt(self, sequence: str, frame_ids: list[str]):
        """Return camera-space `(T, 2, 21, 3)`, vertices, and valid hands."""
        count = len(frame_ids)
        joints = np.zeros((count, 2, 21, 3), dtype=np.float32)
        vertices = np.zeros((count, 2, 778, 3), dtype=np.float32)
        valid = np.zeros((count, 2), dtype=bool)
        hand_dir = self._camera_dir(sequence) / "hand_pose_mano"
        for frame_index, frame_id in enumerate(frame_ids):
            path = hand_dir / f"{frame_id}.txt"
            if not path.exists():
                continue
            data = np.loadtxt(path)
            if data.size != 124:
                continue
            for hand_index, side in enumerate(("left", "right")):
                offset = hand_index * 62
                if data[offset] < 0.5:
                    continue
                frame_joints, frame_vertices = _reconstruct_mano(
                    self._mano(side),
                    data[offset + 1 : offset + 4][None],
                    data[offset + 4 : offset + 49][None],
                    data[offset + 49 : offset + 59][None],
                    data[offset + 59 : offset + 62][None],
                )
                joints[frame_index, hand_index] = frame_joints[0]
                vertices[frame_index, hand_index] = frame_vertices[0]
                valid[frame_index, hand_index] = True
        return joints, vertices, valid

    def load_camera_gt(self, sequence: str, frame_ids: list[str]):
        camera_dir = self._camera_dir(sequence)
        poses = np.full((len(frame_ids), 4, 4), np.nan, dtype=np.float32)
        for frame_index, frame_id in enumerate(frame_ids):
            path = camera_dir / "cam_pose" / f"{frame_id}.txt"
            if path.exists():
                values = np.loadtxt(path)
                if values.size == 16:
                    poses[frame_index] = values.reshape(4, 4)
        intrinsics_path = camera_dir / "cam_intrinsics.txt"
        intrinsics = tuple(float(value) for value in np.loadtxt(intrinsics_path)) if intrinsics_path.exists() else None
        return (poses if np.isfinite(poses).any() else None), intrinsics

    def load_depth_gt(self, sequence: str, frame_ids: list[str]):
        from PIL import Image

        depth_dir = self._camera_dir(sequence) / "depth"
        depths = []
        for frame_id in frame_ids:
            path = depth_dir / f"{frame_id}.png"
            if not path.exists():
                depths.append(None)
                continue
            depth = np.asarray(Image.open(path), dtype=np.float32) / 1000.0
            depth[depth == 0.0] = np.nan
            depths.append(depth)
        return None if all(depth is None for depth in depths) else depths

    def load_contact_gt(self, sequence: str, frame_ids: list[str]):
        """Return frozen scheme-2 object/inter-hand union labels and masks."""
        relative = Path(sequence)

        def read(root: str, key: str):
            path = self.data_root.parent / root / relative / "cam4.npz"
            if not path.exists():
                return None
            payload = np.load(path, allow_pickle=False)
            positions = {int(frame_id): index for index, frame_id in enumerate(payload["frame_ids"])}
            try:
                indices = [positions[int(frame_id)] for frame_id in frame_ids]
            except KeyError:
                return None
            return payload[key][indices].astype(bool), payload["valid"][indices].astype(bool)

        def union(object_labels, interhand_labels):
            if object_labels is None or interhand_labels is None:
                return None
            object_target, object_valid = object_labels
            interhand_target, interhand_valid = interhand_labels
            object_mask = object_valid[..., None]
            interhand_mask = interhand_valid[..., None]
            positive = (object_mask & object_target) | (interhand_mask & interhand_target)
            supervised = positive | (object_mask & interhand_mask)
            return positive.astype(np.uint8), supervised.astype(bool)

        return {
            "joint": union(read("h2o_contact_scheme2_v1", "hard_contact"), read("h2o_interhand_contact_scheme2_v1", "joint_hard_contact")),
            "marker": union(read("h2o_marker_contact_scheme2_v1", "hard_contact"), read("h2o_interhand_contact_scheme2_v1", "marker_hard_contact")),
        }
