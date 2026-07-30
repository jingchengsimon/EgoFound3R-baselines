import os
import argparse

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description="Download pretrained checkpoints from HuggingFace")
    parser.add_argument("output_dir", type=str, help="Saving path for checkpoints", nargs="?", default="ckpts")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    ckpts = [
        "Tartan-T-TSKH-kitti",
        "Tartan-T-TSKH-sintel",
        "Tartan-T-TSKH-spring",
        "Tartan-T-TSKH",
        "Tartan-T",
        "Tartan",
    ]

    for ckpt in ckpts:
        output_path = os.path.join(args.output_dir, f"{ckpt}.ckpt")
        if os.path.exists(output_path):
            continue
        print(f"Downloading {output_path}")
        snapshot_path = snapshot_download(repo_id=f"egorchistov/optical-flow-MEMFOF-{ckpt}")
        state_dict = load_file(os.path.join(snapshot_path, "model.safetensors"))
        state_dict = {"model." + k: v for k, v in state_dict.items()}
        torch.save({"state_dict": state_dict}, output_path)


if __name__ == "__main__":
    main()
