from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class MediaRef:
    kind: str
    path: str
    member: str = ""
    frame_index: int = -1
    time_msec: float = -1.0
    timestamp_ns: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "member": self.member,
            "frame_index": self.frame_index,
            "time_msec": self.time_msec,
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass(slots=True)
class HandAnnotation:
    side: str | None = None
    hand_index: int | None = None
    visible: bool | None = None
    bbox_xyxy: np.ndarray | None = None
    joints_3d: np.ndarray | None = None
    joints_2d: np.ndarray | None = None
    mano_global_orient: np.ndarray | None = None
    mano_hand_pose: np.ndarray | None = None
    mano_pose: np.ndarray | None = None
    mano_pose_format: str | None = None
    mano_betas: np.ndarray | None = None
    mano_trans: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "hand_index": self.hand_index,
            "visible": self.visible,
            "bbox_xyxy": self.bbox_xyxy,
            "joints_3d": self.joints_3d,
            "joints_2d": self.joints_2d,
            "mano_global_orient": self.mano_global_orient,
            "mano_hand_pose": self.mano_hand_pose,
            "mano_pose": self.mano_pose,
            "mano_pose_format": self.mano_pose_format,
            "mano_betas": self.mano_betas,
            "mano_trans": self.mano_trans,
            "extras": self.extras,
        }


@dataclass(slots=True)
class FrameRecord:
    dataset_name: str
    base_dataset_name: str
    split: str
    sequence_id: str
    frame_id: str
    temporal_index: int
    view_name: str
    is_egocentric: bool
    rgb_ref: MediaRef | None = None
    depth_ref: MediaRef | None = None
    depth_mode: str | None = None
    intrinsics: np.ndarray | None = None
    camera_pose: np.ndarray | None = None
    hand_annos: list[HandAnnotation] = field(default_factory=list)
    has_mano: bool = False
    has_3d_joints: bool = False
    has_depth_gt: bool = False
    has_intrinsics_gt: bool = False
    has_camera_pose_gt: bool = False
    has_bbox_gt: bool = False
    has_joint_3d_gt: bool = False
    has_3r_gt: bool = False
    max_left_count: int = 0
    max_right_count: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def unique_side_hand(self, side: str) -> HandAnnotation | None:
        matched = [hand for hand in self.hand_annos if hand.side == side]
        if len(matched) == 1:
            return matched[0]
        return None
