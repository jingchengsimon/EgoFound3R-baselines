#!/usr/bin/env python3
"""HaWoR canonical adapter for EgoFound3R baseline comparison.

Runs the official HaWoR pipeline: detect/track → motion estimation → DROID-SLAM →
Metric3D scale → infiller → MANO world-space reconstruction.
Outputs world-space hand joints/vertices and camera_c2w in OpenCV convention.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image, ImageOps

from formal_evaluation.common.io import (
    load_manifest,
    resolve_rgb_paths,
    select_manifest_window,
    write_comparison_output,
)
from formal_evaluation.common.marker_vertices import MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195
from formal_evaluation.common.schema import SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Mock non-essential heavy deps before HaWoR imports
# ---------------------------------------------------------------------------
_MOCK_PACKAGES = [
    "evo", "evo.core", "evo.core.trajectory", "evo.core.sync", "evo.core.metrics",
    "evo.tools", "evo.tools.file_interface", "evo.main_ape", "evo.common", "evo.common.logging",
    "mmcv", "mmcv.ops", "mmcv.utils", "mmcv.runner", "mmcv.cnn", "mmcv.parallel",
    "html4vision", "pyrender", "OpenGL", "OpenGL.GL",
]
for _m in _MOCK_PACKAGES:
    if _m not in sys.modules:
        _mock = MagicMock()
        _mock.__package__ = _m.split(".")[0]
        _mock.__path__ = []
        sys.modules[_m] = _mock

MARKER_IDS_195 = np.array(MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195, dtype=np.int64)


# ---------------------------------------------------------------------------
# HaWoR pipeline
# ---------------------------------------------------------------------------


def _run_hawor(
    frame_paths: list[Path],
    source_root: Path,
    checkpoint: Path,
    infiller_weight: Path,
    device_name: str,
    img_focal: float | None = None,
):
    """Run full HaWoR pipeline on a list of frames. Returns canonical arrays + metadata."""
    import torch
    import cv2
    from glob import glob
    from natsort import natsorted

    source_root_abs = source_root.resolve()
    hawor_root = str(source_root_abs)

    # Set up sys.path for HaWoR
    for p in [
        hawor_root,
        str(source_root_abs / "thirdparty" / "DROID-SLAM"),
        str(source_root_abs / "thirdparty" / "DROID-SLAM" / "thirdparty" / "lietorch"),
        str(source_root_abs / "thirdparty" / "Metric3D"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Must cwd to HaWoR root for relative weight paths (./weights/..., _DATA/...)
    prev_cwd = os.getcwd()
    os.chdir(hawor_root)

    try:
        return _run_hawor_inner(
            frame_paths, source_root_abs, checkpoint, infiller_weight, device_name, img_focal
        )
    finally:
        os.chdir(prev_cwd)


def _run_hawor_inner(
    frame_paths: list[Path],
    source_root_abs: Path,
    checkpoint: Path,
    infiller_weight: Path,
    device_name: str,
    img_focal: float | None,
):
    import torch
    import cv2
    import joblib
    from glob import glob
    from natsort import natsorted
    from tqdm import tqdm

    from lib.pipeline.tools import detect_track, parse_chunks, parse_chunks_hand_frame
    from lib.eval_utils.custom_utils import interpolate_bboxes, load_slam_cam, cam2world_convert
    from lib.eval_utils.filling_utils import filling_postprocess, filling_preprocess
    from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
    from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
    from infiller.lib.model.network import TransformerModel

    device = torch.device(device_name)
    T = len(frame_paths)

    # Create temp working directory mimicking HaWoR's expected structure
    tmp_dir = tempfile.mkdtemp(prefix="hawor_comparison_")
    seq_name = "seq"
    seq_folder = os.path.join(tmp_dir, seq_name)
    img_folder = os.path.join(seq_folder, "extracted_images")
    os.makedirs(img_folder, exist_ok=True)

    # Copy frames as jpg
    for i, fp in enumerate(frame_paths):
        dst = os.path.join(img_folder, f"{i:04d}.jpg")
        shutil.copy2(fp, dst)
    imgfiles = np.array(natsorted(glob(os.path.join(img_folder, "*.jpg"))))
    assert len(imgfiles) == T

    # Determine focal and image center
    img0 = cv2.imread(imgfiles[0])
    H_img, W_img = img0.shape[:2]
    if img_focal is None:
        focal = float(max(H_img, W_img))
    else:
        focal = float(img_focal)
    img_center = [W_img / 2.0, H_img / 2.0]

    # --- Stage 1: Detection + Tracking ---
    start_idx, end_idx = 0, T
    track_dir = os.path.join(seq_folder, f"tracks_{start_idx}_{end_idx}")
    os.makedirs(track_dir, exist_ok=True)
    boxes_, tracks_ = detect_track(imgfiles, thresh=0.2)
    np.save(os.path.join(track_dir, "model_boxes.npy"), boxes_)
    np.save(os.path.join(track_dir, "model_tracks.npy"), tracks_)

    # --- Stage 2: HaWoR motion estimation ---
    from scripts.scripts_test_video.hawor_video import load_hawor

    model, model_cfg = load_hawor(str(checkpoint.resolve()))
    model = model.to(device).eval()

    tracks = np.load(os.path.join(track_dir, "model_tracks.npy"), allow_pickle=True).item()
    tid = np.array(list(tracks.keys()))

    # Separate tracks by handedness into left(0) / right(1)
    left_trk, right_trk = [], []
    for idx in tid:
        trk = tracks[idx]
        valid = np.array([t["det"] for t in trk])
        if valid.sum() == 0:
            continue
        is_right = np.concatenate([t["det_handedness"] for t in trk])[valid]
        if is_right.sum() / len(is_right) < 0.5:
            left_trk.extend(trk)
        else:
            right_trk.extend(trk)
    left_trk = sorted(left_trk, key=lambda x: x["frame"])
    right_trk = sorted(right_trk, key=lambda x: x["frame"])
    # final_tracks: 0=left, 1=right (matching HaWoR convention)
    final_tracks = {0: left_trk, 1: right_trk}

    # Run motion estimation per hand
    cam_space_dir = os.path.join(seq_folder, "cam_space")
    frame_chunks_all = {0: [], 1: []}
    # model_masks: used for SLAM masking (zeros = no masking since we can't render headless)
    model_masks = np.zeros((T, H_img, W_img), dtype=bool)

    for hand_idx in [0, 1]:
        hand_trk = final_tracks[hand_idx]
        if len(hand_trk) == 0:
            continue

        # Build frame/box arrays from track
        valid = np.array([t["det"] for t in hand_trk])
        if valid.sum() < 1:
            continue
        boxes_arr = np.concatenate([t["det_box"] for t in hand_trk])
        # Interpolate missing boxes
        non_zero_indices = np.where(np.any(boxes_arr != 0, axis=1))[0]
        if len(non_zero_indices) >= 2:
            first_nz, last_nz = non_zero_indices[0], non_zero_indices[-1]
            boxes_arr[first_nz:last_nz + 1] = interpolate_bboxes(boxes_arr[first_nz:last_nz + 1])
            valid[first_nz:last_nz + 1] = True

        boxes_valid = boxes_arr[valid]
        frames_valid = np.array([t["frame"] for t in hand_trk])[valid]

        # Determine handedness flag
        is_right_arr = np.concatenate([t["det_handedness"] for t in hand_trk])[valid]
        do_flip = (is_right_arr.sum() / len(is_right_arr) < 0.5)  # left hand → flip

        # Parse into contiguous chunks
        frame_chunks, boxes_chunks = parse_chunks(frames_valid, boxes_valid, min_len=1)
        frame_chunks_all[hand_idx] = frame_chunks

        if len(frame_chunks) == 0:
            continue

        hand_dir = os.path.join(cam_space_dir, str(hand_idx))
        os.makedirs(hand_dir, exist_ok=True)

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            img_ck = imgfiles[frame_ck]
            with torch.no_grad():
                results = model.inference(
                    img_ck, boxes_ck,
                    img_focal=focal, img_center=img_center, do_flip=do_flip
                )

            # Build data_out following official code
            data_out = {
                "init_root_orient": results["pred_rotmat"][None, :, 0],   # (1, T_ck, 3, 3)
                "init_hand_pose": results["pred_rotmat"][None, :, 1:],    # (1, T_ck, 15, 3, 3)
                "init_trans": results["pred_trans"][None, :, 0],          # (1, T_ck, 3)
                "init_betas": results["pred_shape"][None, :],             # (1, T_ck, 10)
            }

            # Flip left hand rotation axes back
            init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            if do_flip:
                init_root[..., 1] *= -1
                init_root[..., 2] *= -1
                init_hand_pose[..., 1] *= -1
                init_hand_pose[..., 2] *= -1
            data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
            data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

            # Save camera-space results
            pred_dict = {k: v.tolist() for k, v in data_out.items()}
            pred_path = os.path.join(hand_dir, f"{frame_ck[0]}_{frame_ck[-1]}.json")
            with open(pred_path, "w") as f:
                json.dump(pred_dict, f)

    # --- Stage 3+4: DROID-SLAM + Metric3D scale (with identity fallback) ---
    from lib.pipeline.masked_droid_slam import run_slam
    from lib.pipeline.est_scale import est_scale_hybrid

    calib = np.array([focal, focal, img_center[0], img_center[1]])
    masks_tensor = torch.from_numpy(model_masks)

    slam_failed = False
    try:
        droid, traj = run_slam(img_folder, masks=masks_tensor, calib=calib)
        n = droid.video.counter.value
        if n < 2:
            raise RuntimeError(f"DROID-SLAM produced only {n} keyframes (need >=2)")
        tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
        disps = droid.video.disps_up.cpu().numpy()[:n]
        del droid
        torch.cuda.empty_cache()

        # --- Stage 4: Metric3D scale estimation ---
        from metric import Metric3D

        metric = Metric3D(str(source_root_abs / "thirdparty" / "Metric3D" / "weights" / "metric_depth_vit_large_800k.pth"))
        pred_depths = []
        for t in tstamp:
            pred_depth = metric(imgfiles[t], calib)
            pred_depth = cv2.resize(pred_depth, (W_img, H_img))
            pred_depths.append(pred_depth)

        # Estimate metric scale
        scales_ = []
        min_threshold, max_threshold = 0.4, 0.7
        for i in range(len(tstamp)):
            t = tstamp[i]
            disp = disps[i]
            pred_depth = pred_depths[i]
            slam_depth = 1.0 / (disp + 1e-8)
            # Resize pred_depth to match slam_depth resolution
            if pred_depth.shape != slam_depth.shape:
                pred_depth = cv2.resize(pred_depth, (slam_depth.shape[1], slam_depth.shape[0]))
            msk = model_masks[t].astype(np.uint8)
            scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk,
                                     near_thresh=min_threshold, far_thresh=max_threshold)
            while math.isnan(scale):
                min_threshold -= 0.1
                max_threshold += 0.1
                scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk,
                                         near_thresh=min_threshold, far_thresh=max_threshold)
            scales_.append(scale)
        median_s = float(np.median(scales_))

        # Save SLAM results
        slam_dir = os.path.join(seq_folder, "SLAM")
        os.makedirs(slam_dir, exist_ok=True)
        slam_path = os.path.join(slam_dir, f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        np.savez(slam_path, tstamp=tstamp, disps=disps, traj=traj.astype(np.float32),
                 img_focal=np.float32(focal), img_center=np.array(img_center, dtype=np.float32),
                 scale=np.float32(median_s))

        # Load SLAM cameras
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    except Exception as slam_err:
        print(f"[HaWoR] SLAM failed ({slam_err}), using identity camera fallback")
        slam_failed = True
        median_s = 1.0
        torch.cuda.empty_cache()
        # Identity camera poses for all T frames
        R_c2w_sla_all = torch.eye(3).unsqueeze(0).expand(T, -1, -1).clone()
        t_c2w_sla_all = torch.zeros(T, 3)

    # --- Stage 5: Camera-to-world conversion + Infiller ---
    # After DROID-SLAM multiprocessing, CUDA context may be corrupted.
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    pred_trans = torch.zeros(2, T, 3)
    pred_rot = torch.zeros(2, T, 3)
    pred_hand_pose = torch.zeros(2, T, 45)
    pred_betas = torch.zeros(2, T, 10)
    pred_valid = torch.zeros(2, T)
    cam2world_failed = False

    for hand_idx in [0, 1]:
        chunks = frame_chunks_all[hand_idx]
        if len(chunks) == 0:
            continue
        hand_dir = os.path.join(cam_space_dir, str(hand_idx))
        for frame_ck in chunks:
            pred_path = os.path.join(hand_dir, f"{frame_ck[0]}_{frame_ck[-1]}.json")
            if not os.path.exists(pred_path):
                continue
            with open(pred_path, "r") as f:
                pred_dict_raw = json.load(f)
            data_out = {k: torch.tensor(v) for k, v in pred_dict_raw.items()}

            if not cam2world_failed:
                try:
                    R_c2w_sla = R_c2w_sla_all[frame_ck]
                    t_c2w_sla = t_c2w_sla_all[frame_ck]
                    data_world = cam2world_convert(
                        R_c2w_sla, t_c2w_sla, data_out,
                        "right" if hand_idx > 0 else "left"
                    )
                    pred_trans[hand_idx, frame_ck] = data_world["init_trans"]
                    pred_rot[hand_idx, frame_ck] = data_world["init_root_orient"]
                    pred_hand_pose[hand_idx, frame_ck] = data_world["init_hand_pose"].flatten(-2)
                    pred_betas[hand_idx, frame_ck] = data_out["init_betas"]
                    pred_valid[hand_idx, frame_ck] = 1
                except RuntimeError as e:
                    if "CUDA" in str(e) or "cuda" in str(e):
                        print(f"[HaWoR] cam2world CUDA failed ({e}), camera-space fallback")
                        cam2world_failed = True
                        slam_failed = True
                        median_s = 1.0
                        R_c2w_sla_all = torch.eye(3).unsqueeze(0).expand(T, -1, -1).clone()
                        t_c2w_sla_all = torch.zeros(T, 3)
                    else:
                        raise
            # Fallback: use camera-space predictions directly
            if cam2world_failed:
                rot_aa = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
                pose_aa = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
                pred_trans[hand_idx, frame_ck] = data_out["init_trans"]
                pred_rot[hand_idx, frame_ck] = rot_aa
                pred_hand_pose[hand_idx, frame_ck] = pose_aa.flatten(-2)
                pred_betas[hand_idx, frame_ck] = data_out["init_betas"]
                pred_valid[hand_idx, frame_ck] = 1

    # Run infiller for missing frames
    pred_valid_np = (pred_valid > 0).numpy()
    frame_list = torch.tensor(list(range(T)))
    filling_length = 120

    # Load infiller model
    ckpt = torch.load(str(infiller_weight.resolve()), map_location=device)
    pos_dim, shape_dim, num_joints = 3, 10, 15
    rot_dim = (num_joints + 1) * 6
    repr_dim = 2 * (pos_dim + shape_dim + rot_dim)
    filling_model = TransformerModel(
        seq_len=filling_length, input_dim=repr_dim, d_model=384, nhead=8,
        d_hid=2048, nlayers=8, dropout=0.05, out_dim=repr_dim, masked_attention_stage=True
    )
    filling_model.to(device)
    filling_model.load_state_dict(ckpt["transformer_encoder_state_dict"])
    filling_model.eval()

    idx2hand = ["left", "right"]
    for hand_idx in [1, 0]:
        missing = ~pred_valid_np[hand_idx]
        frame = frame_list[missing]
        if len(frame) == 0:
            continue
        frame_chunks_miss = parse_chunks_hand_frame(frame.numpy())
        for frame_ck in frame_chunks_miss:
            # Find valid neighbor as start
            start_shift = -1
            while frame_ck[0] + start_shift >= 0 and pred_valid_np[:, frame_ck[0] + start_shift].sum() != 2:
                start_shift -= 1
            frame_start = frame_ck[0]
            filling_net_start = max(0, frame_start + start_shift)
            filling_net_end = min(T - 1, filling_net_start + filling_length)
            seq_valid = pred_valid_np[:, filling_net_start:filling_net_end]
            filling_seq = {
                "trans": pred_trans[:, filling_net_start:filling_net_end].numpy(),
                "rot": pred_rot[:, filling_net_start:filling_net_end].numpy(),
                "hand_pose": pred_hand_pose[:, filling_net_start:filling_net_end].numpy(),
                "betas": pred_betas[:, filling_net_start:filling_net_end].numpy(),
                "valid": seq_valid,
            }
            filling_input, transform_w_canon = filling_preprocess(filling_seq)
            if filling_input is None:
                continue

            src_mask = torch.zeros((filling_length, filling_length), device=device, dtype=torch.bool)
            filling_input_t = torch.from_numpy(filling_input).unsqueeze(0).to(device).permute(1, 0, 2)
            T_original = len(filling_input_t)
            if T_original < filling_length:
                pad_length = filling_length - T_original
                padding = filling_input_t[-1:].repeat(pad_length, 1, 1)
                filling_input_t = torch.cat([filling_input_t, padding], dim=0)
                seq_valid_padding = np.ones((2, filling_length - T_original))
                seq_valid_padding = np.concatenate([seq_valid, seq_valid_padding], axis=1)
            else:
                seq_valid_padding = seq_valid

            T_fill, B_fill, _ = filling_input_t.shape
            valid = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).permute(1, 0)
            valid_atten = torch.from_numpy(seq_valid_padding).unsqueeze(0).all(dim=1).unsqueeze(1)
            data_mask = torch.zeros((filling_length, B_fill, 1), device=device, dtype=filling_input_t.dtype)
            data_mask[valid] = 1
            atten_mask = torch.ones((B_fill, 1, filling_length), device=device, dtype=torch.bool)
            atten_mask[valid_atten] = False
            atten_mask = atten_mask.unsqueeze(2).repeat(1, 1, T_fill, 1)

            with torch.no_grad():
                output_ck = filling_model(filling_input_t, src_mask, data_mask, atten_mask)
            output_ck = output_ck.permute(1, 0, 2).reshape(T_fill, 2, -1).cpu().detach()
            output_ck = output_ck[:T_original]
            filling_output = filling_postprocess(output_ck, transform_w_canon)

            filling_seq["trans"][~seq_valid] = filling_output["trans"][~seq_valid]
            filling_seq["rot"][~seq_valid] = filling_output["rot"][~seq_valid]
            filling_seq["hand_pose"][~seq_valid] = filling_output["hand_pose"][~seq_valid]
            filling_seq["betas"][~seq_valid] = filling_output["betas"][~seq_valid]
            pred_trans[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq["trans"])
            pred_rot[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq["rot"])
            pred_hand_pose[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq["hand_pose"])
            pred_betas[:, filling_net_start:filling_net_end] = torch.from_numpy(filling_seq["betas"])
            pred_valid_np[:, filling_net_start:filling_net_end] = 1

    pred_valid = torch.from_numpy(pred_valid_np.astype(np.float32))

    # --- Stage 6: MANO reconstruction ---
    vis_start, vis_end = 0, T

    # Right hand (idx=1)
    hand_idx = 1
    if pred_valid[hand_idx, vis_start:vis_end].sum() > 0:
        pred_glob_r = run_mano(
            pred_trans[hand_idx:hand_idx + 1, vis_start:vis_end],
            pred_rot[hand_idx:hand_idx + 1, vis_start:vis_end],
            pred_hand_pose[hand_idx:hand_idx + 1, vis_start:vis_end],
            betas=pred_betas[hand_idx:hand_idx + 1, vis_start:vis_end],
        )
        right_verts = pred_glob_r["vertices"][0]   # (T, 778, 3)
        right_joints = pred_glob_r["joints"][0]    # (T, 21, 3)
    else:
        right_verts = torch.zeros(T, 778, 3)
        right_joints = torch.zeros(T, 21, 3)

    # Left hand (idx=0)
    hand_idx = 0
    if pred_valid[hand_idx, vis_start:vis_end].sum() > 0:
        pred_glob_l = run_mano_left(
            pred_trans[hand_idx:hand_idx + 1, vis_start:vis_end],
            pred_rot[hand_idx:hand_idx + 1, vis_start:vis_end],
            pred_hand_pose[hand_idx:hand_idx + 1, vis_start:vis_end],
            betas=pred_betas[hand_idx:hand_idx + 1, vis_start:vis_end],
        )
        left_verts = pred_glob_l["vertices"][0]    # (T, 778, 3)
        left_joints = pred_glob_l["joints"][0]     # (T, 21, 3)
    else:
        left_verts = torch.zeros(T, 778, 3)
        left_joints = torch.zeros(T, 21, 3)

    # --- Stage 7: Coordinate conversion DROID/OpenGL → OpenCV ---
    # R_x = diag(1, -1, -1) converts y-up/z-back to y-down/z-forward
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
    R_c2w_cv = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w_cv = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)

    # Flip hand vertices/joints to OpenCV world convention
    left_verts_cv = torch.einsum("ij,tnj->tni", R_x, left_verts.cpu())
    right_verts_cv = torch.einsum("ij,tnj->tni", R_x, right_verts.cpu())
    left_joints_cv = torch.einsum("ij,tnj->tni", R_x, left_joints.cpu())
    right_joints_cv = torch.einsum("ij,tnj->tni", R_x, right_joints.cpu())

    # Build camera_c2w (T, 4, 4)
    camera_c2w = np.zeros((T, 4, 4), dtype=np.float32)
    camera_c2w[:, 3, 3] = 1.0
    camera_c2w[:, :3, :3] = R_c2w_cv.numpy()
    camera_c2w[:, :3, 3] = t_c2w_cv.numpy()

    # Build hand_valid (slot 0=left, 1=right)
    hand_valid = np.zeros((T, 2), dtype=bool)
    hand_valid[:, 0] = pred_valid[0].numpy() > 0
    hand_valid[:, 1] = pred_valid[1].numpy() > 0

    # Build vertices array (slot 0=left, 1=right)
    vertices_out = np.zeros((T, 2, 778, 3), dtype=np.float32)
    vertices_out[:, 0] = left_verts_cv.numpy()
    vertices_out[:, 1] = right_verts_cv.numpy()

    # Build joints array (slot 0=left, 1=right)
    joints_out = np.zeros((T, 2, 21, 3), dtype=np.float32)
    joints_out[:, 0] = left_joints_cv.numpy()
    joints_out[:, 1] = right_joints_cv.numpy()

    # Extract 195 marker subset
    markers_out = vertices_out[:, :, MARKER_IDS_195, :]  # (T, 2, 195, 3)

    # Cleanup temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    arrays = {
        "camera_c2w": camera_c2w,
        "camera_valid": np.ones(T, dtype=bool),
        "hand_joints_world": joints_out,
        "hand_vertices_world": vertices_out,
        "hand_markers_world": markers_out,
        "hand_valid": hand_valid,
    }
    native = {
        "slam_scale": np.array([median_s], dtype=np.float32),
    }
    detail = {
        "pipeline": "official HaWoR: detect_track → motion_estimation → DROID-SLAM → Metric3D scale → infiller → MANO",
        "detector": "official YOLO hand detector, thresh=0.2, with tracking",
        "slam": "DROID-SLAM with sm90 support, hand masking disabled (headless, no renderer)",
        "slam_failed_identity_fallback": slam_failed,
        "scale_estimation": "Metric3D ViT-Large + est_scale_hybrid" if not slam_failed else "N/A (SLAM failed, identity fallback)",
        "infiller": "official TransformerModel infiller for missing frames",
        "coordinate": "OpenCV world frame via R_x=diag(1,-1,-1) from DROID/OpenGL convention",
        "hand_slot_convention": "slot 0=left, slot 1=right",
        "slam_scale": median_s,
        "n_tracked_left": int(pred_valid[0].sum()),
        "n_tracked_right": int(pred_valid[1].sum()),
    }
    return arrays, native, (H_img, W_img), detail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 HaWoR baseline 并写入 canonical 输出")
    parser.add_argument("--phase", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--methods-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--infiller-weight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--window-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--img-focal", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    sequence, window_id, frame_ids = select_manifest_window(
        manifest, phase=args.phase, sequence=args.sequence, window_id=args.window_id
    )
    frame_paths = resolve_rgb_paths(args.data_root, sequence, frame_ids)

    # Read original resolution
    img0 = Image.open(frame_paths[0])
    img0 = ImageOps.exif_transpose(img0)
    orig_hw = (img0.height, img0.width)

    t0 = time.perf_counter()
    if args.device.startswith("cuda"):
        import torch
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()

    arrays, native_arrays, run_hw, detail = _run_hawor(
        frame_paths=frame_paths,
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        infiller_weight=args.infiller_weight,
        device_name=args.device,
        img_focal=args.img_focal,
    )

    elapsed = time.perf_counter() - t0
    peak_vram_gb = 0.0
    if args.device.startswith("cuda"):
        import torch
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Build capabilities from arrays
    capabilities = {k: True for k in arrays}

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "method": "hawor",
        "sequence": sequence,
        "window_id": window_id,
        "frame_ids": frame_ids,
        "original_resolution_hw": list(orig_hw),
        "capabilities": capabilities,
        "detail": detail,
    }
    run_info = {
        "elapsed_seconds": round(elapsed, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "device": args.device,
        "checkpoint": str(args.checkpoint),
        "infiller_weight": str(args.infiller_weight),
        "source_root": str(args.source_root),
        "status": "success",
    }

    output_dir = args.output_root / "hawor" / args.phase / f"{sequence.replace('/', '_')}_{frame_ids[0]}_{frame_ids[-1]}"
    write_comparison_output(
        output_dir,
        metadata=metadata,
        arrays=arrays,
        run=run_info,
        native_metadata={"method": "hawor", "detail": detail},
        native_arrays=native_arrays,
    )
    print(f"[HaWoR] {args.phase} 完成: {output_dir}")
    print(f"  elapsed={elapsed:.1f}s  peak_vram={peak_vram_gb:.2f}GB")
    print(f"  hand_valid: left={arrays['hand_valid'][:,0].sum()}/{len(frame_ids)}, right={arrays['hand_valid'][:,1].sum()}/{len(frame_ids)}")


if __name__ == "__main__":
    main()
