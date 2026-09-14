"""Run WiLoR on a video and save per-frame predictions to a .npz file.
Called automatically by demo.py via: conda run -n wilor python wilor_inference.py
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
_DEMO_DIR = os.environ.get('PAD_WILOR_ASSET_ROOT', os.path.dirname(os.path.abspath(__file__)))
_WILOR_DIR = os.path.join(_DEMO_DIR, 'WiLoR')
sys.path.insert(0, _WILOR_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO
from prediction_slots import pack_frames, select_hands

WILOR_CKPT  = os.path.join(_DEMO_DIR, 'WiLoR/pretrained_models/wilor_final.ckpt')
WILOR_CFG   = os.path.join(_DEMO_DIR, 'WiLoR/pretrained_models/model_config.yaml')
DETECTOR_PT = os.path.join(_DEMO_DIR, 'WiLoR/pretrained_models/detector.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',  required=True)
    parser.add_argument('--output', required=True, help='Path to save .npz results')
    parser.add_argument('--both-hands', action='store_true',
                        help='preserve one left and one right detection per frame for formal evaluation')
    args = parser.parse_args()
    # Resolve to absolute paths before chdir so relative paths keep working
    args.video  = os.path.abspath(args.video)
    args.output = os.path.abspath(args.output)
    os.chdir(_WILOR_DIR)  # WiLoR expects cwd to be its own dir (mano_data/, pretrained_models/)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wilor_model, wilor_cfg = load_wilor(checkpoint_path=WILOR_CKPT, cfg_path=WILOR_CFG)
    detector = YOLO(DETECTOR_PT)
    wilor_model = wilor_model.to(device)
    detector    = detector.to(device)
    wilor_model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames = []

    pbar = tqdm(desc='WiLoR inference')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        detections = detector(frame, conf=0.3, verbose=False)[0]
        bboxes, rights = [], []
        for det in detections:
            bbox_data = det.boxes.data.cpu().detach().squeeze()
            if bbox_data.dim() == 0:
                continue
            rights.append(float(det.boxes.cls.cpu().detach().squeeze().item()))
            bboxes.append(bbox_data[:4].tolist())

        if not bboxes:
            frames.append(select_hands([], args.both_hands))
            continue

        dataset    = ViTDetDataset(wilor_cfg, frame, np.stack(bboxes), np.stack(rights))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        frame_hands = []
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = wilor_model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam'].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            img_size     = batch['img_size'].float()
            scaled_focal = wilor_cfg.EXTRA.FOCAL_LENGTH / wilor_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t   = cam_crop_to_full(
                pred_cam, batch['box_center'].float(), batch['box_size'].float(),
                img_size, scaled_focal,
            ).detach().cpu()

            for n in range(batch['img'].shape[0]):
                is_r  = float(batch['right'][n].cpu().item())
                verts = out['pred_vertices'][n].detach().cpu().clone()
                verts[:, 0] = (2 * is_r - 1) * verts[:, 0]

                frame_hands.append({
                    'vertices':      verts.numpy(),
                    'cam_t':         pred_cam_t[n].numpy(),
                    'global_orient': out['pred_mano_params']['global_orient'][n].detach().cpu().numpy(),
                    'hand_pose':     out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy(),
                    'betas':         out['pred_mano_params']['betas'][n].detach().cpu().numpy(),
                    'is_right':      is_r,
                    'img_size':      img_size[n].cpu().numpy(),
                    'scaled_focal':  float(scaled_focal.cpu()),
                })

        frames.append(select_hands(frame_hands, args.both_hands))

    cap.release()
    pbar.close()

    arrays = pack_frames(frames, args.both_hands)
    np.savez(args.output, **arrays, fps=np.array([fps], dtype=np.float32))
    print(f'Saved WiLoR predictions for {len(frames)} frames → {args.output}')


if __name__ == '__main__':
    main()
