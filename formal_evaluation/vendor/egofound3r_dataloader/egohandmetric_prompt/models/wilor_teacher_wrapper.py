from __future__ import annotations

from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn

from egohandmetric_prompt.models.vggt_omega_frozen_wrapper import freeze_module
from egohandmetric_prompt.vendor import ensure_vendor_paths, get_vendor_roots


class WiLorTeacherWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, pooled_dim: int | None = None) -> None:
        super().__init__()
        self.backbone = freeze_module(backbone)
        self.pooled_dim = pooled_dim
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        freeze_module(self.backbone)
        return self

    @classmethod
    def from_vendor(
        cls,
        *,
        checkpoint_path: str | Path,
        cfg_path: str | Path | None = None,
        pooled_dim: int | None = None,
    ) -> "WiLorTeacherWrapper":
        ensure_vendor_paths()
        from wilor.configs import get_config

        roots = get_vendor_roots()
        resolved_cfg_path = Path(cfg_path) if cfg_path else roots.wilor / "pretrained_models" / "model_config.yaml"
        if not resolved_cfg_path.exists():
            resolved_cfg_path = roots.wilor / "pretrained_models" / "model_config.yaml"
        model_cfg = get_config(str(resolved_cfg_path), update_cachedir=True)
        mano_data_root = roots.wilor / "mano_data"
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = str(mano_data_root)
        model_cfg.MANO.MODEL_PATH = str(mano_data_root)
        model_cfg.MANO.MEAN_PARAMS = str(mano_data_root / "mano_mean_params.npz")
        model_cfg.freeze()
        wilor_pkg_root = roots.wilor / "wilor"
        if "wilor" not in sys.modules:
            pkg = types.ModuleType("wilor")
            pkg.__path__ = [str(wilor_pkg_root)]
            sys.modules["wilor"] = pkg
        if "wilor.models" not in sys.modules:
            pkg = types.ModuleType("wilor.models")
            pkg.__path__ = [str(wilor_pkg_root / "models")]
            sys.modules["wilor.models"] = pkg
        if "wilor.utils" not in sys.modules:
            pkg = types.ModuleType("wilor.utils")
            pkg.__path__ = [str(wilor_pkg_root / "utils")]
            sys.modules["wilor.utils"] = pkg
        from wilor.models.backbones import create_backbone
        backbone = create_backbone(model_cfg)
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        backbone_state_dict = {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        missing, unexpected = backbone.load_state_dict(backbone_state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"WiLoR backbone checkpoint 含未知字段: {unexpected[:10]}")
        if missing:
            raise RuntimeError(f"WiLoR backbone checkpoint 缺少字段: {missing[:10]}")
        return cls(backbone=backbone, pooled_dim=pooled_dim)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            crops = (crops - self.image_mean.to(device=crops.device, dtype=crops.dtype)) / self.image_std.to(
                device=crops.device,
                dtype=crops.dtype,
            )
            if crops.shape[-1] == crops.shape[-2] and crops.shape[-1] >= 64:
                crops = crops[:, :, :, 32:-32]
            features = self.backbone(crops)
            if isinstance(features, (tuple, list)):
                features = features[-1]
            if isinstance(features, dict):
                if "vit_out" in features:
                    features = features["vit_out"]
                elif "pred_mano_feats" in features:
                    features = features["pred_mano_feats"]
                else:
                    raise ValueError("当前不支持这种 WiLoR backbone 输出字典。")
            if features.ndim == 4:
                pooled = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
            elif features.ndim == 2:
                pooled = features
            else:
                raise ValueError("WiLoR 特征必须是 2 维或 4 维张量。")
            if self.pooled_dim is not None and pooled.shape[-1] != self.pooled_dim:
                raise ValueError(f"期望 pooled_dim={self.pooled_dim}，实际得到 {pooled.shape[-1]}。")
            return pooled
