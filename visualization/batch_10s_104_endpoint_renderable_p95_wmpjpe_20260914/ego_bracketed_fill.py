"""Visualization-only Ego hand completion adapted from EgoFound3R commit 8fc061a."""

import numpy as np


SOURCE_COMMIT = "8fc061a615895bd3b5a556f7387bae306e32d9db"
SOURCE_BRANCH = "codex/hand-depth-scale-fusion-20260908"


def fill_camera_space(joints, markers, hand_valid):
    """Fill only invalid interior frames using bracketing trusted camera-space poses."""
    joints = np.asarray(joints, dtype=float).copy()
    markers = np.asarray(markers, dtype=float).copy()
    hand_valid = np.asarray(hand_valid, dtype=bool)
    if joints.shape[:2] != hand_valid.shape or markers.shape[:2] != hand_valid.shape:
        raise ValueError("hand array shapes disagree")
    trusted = (hand_valid
               & np.isfinite(joints).all(axis=(2, 3))
               & np.isfinite(markers).all(axis=(2, 3)))
    filled = np.zeros_like(trusted)
    gap_runs = [[], []]
    for side in range(2):
        anchors = np.flatnonzero(trusted[:, side])
        if len(anchors) < 2 or anchors[0] != 0 or anchors[-1] != len(trusted) - 1:
            raise ValueError(f"side {side} lacks trusted first/last anchors")
        for left, right in zip(anchors[:-1], anchors[1:]):
            if right == left + 1:
                continue
            gap_runs[side].append([int(left + 1), int(right - 1)])
            root_left, root_right = joints[left, side, 0], joints[right, side, 0]
            joint_local_left = joints[left, side] - root_left
            joint_local_right = joints[right, side] - root_right
            marker_local_left = markers[left, side] - root_left
            marker_local_right = markers[right, side] - root_right
            for frame in range(left + 1, right):
                alpha = (frame - left) / (right - left)
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                root = (1.0 - smooth) * root_left + smooth * root_right
                joints[frame, side] = root + (1.0 - smooth) * joint_local_left + smooth * joint_local_right
                markers[frame, side] = root + (1.0 - smooth) * marker_local_left + smooth * marker_local_right
                filled[frame, side] = True
    render_valid = trusted | filled
    if not render_valid.all():
        raise ValueError("unfilled frames remain; endpoint extrapolation is forbidden")
    return joints, markers, render_valid, {
        "source_commit": SOURCE_COMMIT,
        "source_branch": SOURCE_BRANCH,
        "space": "camera",
        "alpha": "smoothstep",
        "extrapolation": False,
        "trusted_counts_left_right": trusted.sum(0).astype(int).tolist(),
        "filled_counts_left_right": filled.sum(0).astype(int).tolist(),
        "gap_runs_left_right": gap_runs,
        "filled_mask": filled,
    }
