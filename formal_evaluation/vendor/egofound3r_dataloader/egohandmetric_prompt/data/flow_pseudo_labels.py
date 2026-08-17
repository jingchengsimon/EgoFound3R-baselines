from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as F


FLOW_CACHE_VERSION = 2
FLOW_CACHE_LEGACY_VERSION = 1


@dataclass(slots=True)
class FlowPseudoLabel:
    flow: torch.Tensor
    valid: torch.Tensor
    source_image_size_hw: tuple[int, int]


def _component(value: str | int) -> str:
    return quote(str(value), safe="")


def flow_cache_path(
    root: str | Path,
    *,
    dataset_name: str,
    sequence_id: str,
    view_name: str,
    source_temporal_index: int,
    target_temporal_index: int,
    preprocessing_version: str = "raw-v1",
) -> Path:
    return (
        Path(root)
        / _component(dataset_name)
        / _component(sequence_id)
        / _component(view_name)
        / _component(preprocessing_version)
        / f"{int(source_temporal_index):08d}_{int(target_temporal_index):08d}.npz"
    )


class FlowPseudoLabelStore:
    def __init__(self, root: str | Path, *, preprocessing_version: str = "raw-v1") -> None:
        self.root = Path(root)
        self.preprocessing_version = str(preprocessing_version)

    def load(
        self,
        *,
        dataset_name: str,
        sequence_id: str,
        view_name: str,
        source_temporal_index: int,
        target_temporal_index: int,
    ) -> FlowPseudoLabel | None:
        path = flow_cache_path(
            self.root,
            dataset_name=dataset_name,
            sequence_id=sequence_id,
            view_name=view_name,
            source_temporal_index=source_temporal_index,
            target_temporal_index=target_temporal_index,
            preprocessing_version=self.preprocessing_version,
        )
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as payload:
            version = int(np.asarray(payload["cache_version"]).reshape(()))
            if version == FLOW_CACHE_LEGACY_VERSION:
                flow = torch.from_numpy(np.asarray(payload["flow"], dtype=np.float32).copy())
            elif version == FLOW_CACHE_VERSION:
                scale = float(np.asarray(payload["flow_scale"], dtype=np.float32).reshape(()))
                if not np.isfinite(scale) or scale <= 0.0:
                    raise ValueError(f"flow cache 量化 scale 无效: {path}")
                flow = torch.from_numpy(np.asarray(payload["flow_q"], dtype=np.int16).copy()).to(
                    dtype=torch.float32
                ) / scale
            else:
                raise ValueError(f"不支持的 flow cache 版本 {version}: {path}")
            valid = torch.from_numpy(np.asarray(payload["valid"], dtype=np.bool_).copy())
            source_size = tuple(int(value) for value in np.asarray(payload["source_image_size_hw"]).tolist())
        if flow.ndim != 3 or flow.shape[0] != 2 or valid.shape != flow.shape[-2:]:
            raise ValueError(f"flow cache 形状无效: {path}")
        if len(source_size) != 2 or tuple(flow.shape[-2:]) != source_size:
            raise ValueError(f"flow cache 原图尺寸不一致: {path}")
        finite = torch.isfinite(flow).all(dim=0)
        valid &= finite
        flow = torch.where(finite.unsqueeze(0), flow, torch.zeros_like(flow))
        return FlowPseudoLabel(flow=flow, valid=valid, source_image_size_hw=source_size)


def transport_flow_pseudo_label(
    label: FlowPseudoLabel,
    *,
    crop_box: tuple[int, int, int, int] | None,
    target_height: int,
    target_width: int,
    horizontal_flip: bool = False,
) -> FlowPseudoLabel:
    flow = label.flow.to(dtype=torch.float32)
    valid = label.valid.to(dtype=torch.bool)
    source_height, source_width = label.source_image_size_hw
    if flow.shape[-2:] != (source_height, source_width):
        raise ValueError("flow 与 source_image_size_hw 不一致")
    left, top, right, bottom = crop_box or (0, 0, source_width, source_height)
    left = max(min(int(left), source_width - 1), 0)
    top = max(min(int(top), source_height - 1), 0)
    right = max(min(int(right), source_width), left + 1)
    bottom = max(min(int(bottom), source_height), top + 1)
    y, x = torch.meshgrid(
        torch.arange(source_height, dtype=torch.float32),
        torch.arange(source_width, dtype=torch.float32),
        indexing="ij",
    )
    endpoint_x = x + flow[0]
    endpoint_y = y + flow[1]
    finite = torch.isfinite(flow).all(dim=0)
    valid &= finite
    flow = torch.where(finite.unsqueeze(0), flow, torch.zeros_like(flow))
    valid &= (endpoint_x >= left) & (endpoint_x < right) & (endpoint_y >= top) & (endpoint_y < bottom)
    cropped_flow = flow[:, top:bottom, left:right]
    cropped_valid = valid[top:bottom, left:right]
    crop_height, crop_width = bottom - top, right - left
    resized_flow = F.interpolate(
        cropped_flow.unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    resized_flow[0] *= float(target_width) / float(crop_width)
    resized_flow[1] *= float(target_height) / float(crop_height)
    resized_valid = F.interpolate(
        cropped_valid.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=(target_height, target_width),
        mode="nearest",
    ).squeeze(0).squeeze(0).to(dtype=torch.bool)
    if horizontal_flip:
        resized_flow = torch.flip(resized_flow, dims=(-1,))
        resized_flow[0].neg_()
        resized_valid = torch.flip(resized_valid, dims=(-1,))
    resized_valid &= torch.isfinite(resized_flow).all(dim=0)
    resized_flow = torch.where(resized_valid.unsqueeze(0), resized_flow, torch.zeros_like(resized_flow))
    return FlowPseudoLabel(
        flow=resized_flow,
        valid=resized_valid,
        source_image_size_hw=(int(target_height), int(target_width)),
    )
