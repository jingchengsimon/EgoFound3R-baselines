"""Recover shared, Hand-independent Omega scene arrays from archived one-way outputs."""
from types import SimpleNamespace

import numpy as np


def recover_raw_scene(prediction, native):
    import torch
    from egohandmetric_prompt.vendor import ensure_vendor_paths
    from egohandmetric_prompt.models.camera_temporal_refiner import DeterministicTemporalInterpolator

    ensure_vendor_paths()
    from vggt_omega.utils.pose_enc import encoding_to_camera

    depth = np.asarray(prediction['depth'], dtype=np.float32)
    count, height, width = depth.shape
    anchors = np.asarray(native['global_anchor_indices'], dtype=np.int64)
    encoding = np.asarray(native['camera_pose_encoding_global'], dtype=np.float32)
    scale = np.asarray(native['metric_scale_factor']).reshape(-1)
    if scale.size != 1 or not np.isfinite(scale[0]) or scale[0] <= 0:
        raise ValueError('archived depth cannot recover raw Omega: invalid Hand scale')
    if not np.array_equal(anchors, np.arange(2, count, 5)) or encoding.shape != (len(anchors), 9):
        raise ValueError('expected complete stride5 phase2 Omega anchors')
    if not np.isfinite(depth[anchors]).reshape(len(anchors), -1).any(axis=1).all():
        raise ValueError('archived depth lost a native Omega anchor; fresh raw inference required')
    # Invert the one documented float32 multiplication, never fit a new scale.
    raw_depth = depth / np.float32(scale[0])
    with torch.inference_mode():
        extrinsics, intrinsics = encoding_to_camera(torch.from_numpy(encoding)[None], (height, width), build_intrinsics=True)
        poses = torch.eye(4).repeat(1, len(anchors), 1, 1)
        poses[..., :3, :4] = extrinsics
        frame_map = SimpleNamespace(global_anchor_indices=torch.from_numpy(anchors)[None], global_frame_present=torch.ones(1, len(anchors), dtype=torch.bool))
        query_map = SimpleNamespace(high_anchor_indices=torch.arange(count)[None], output_present=torch.ones(1, count, dtype=torch.bool))
        # Only parameter-free cubic/SQUAD camera interpolation; no Hand or metric path.
        camera, valid, _, _ = DeterministicTemporalInterpolator()._camera_base(poses, frame_map, query_map, frame_map.global_frame_present)
        c2w = torch.linalg.inv(camera)[0].numpy()
        c2w[~valid[0].numpy()] = np.nan
    dense_k = np.full((count, 3, 3), np.nan, dtype=np.float32)
    dense_k[anchors] = intrinsics[0].numpy()
    return dict(camera_c2w=c2w, camera_valid=np.isfinite(c2w).all(axis=(1, 2)), intrinsics=dense_k,
                depth=raw_depth, depth_valid=np.isfinite(raw_depth) & (raw_depth > 0),
                depth_confidence=np.asarray(prediction['depth_confidence']))
