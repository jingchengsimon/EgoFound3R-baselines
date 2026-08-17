from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_marker_model_weights(model: torch.nn.Module, checkpoint_path: str | Path) -> int:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint 格式错误: {checkpoint_path}")

    raw_weights: Any
    if "model" in payload:
        raw_weights = payload["model"]
    else:
        raw_weights = payload
    if not isinstance(raw_weights, dict):
        raise ValueError(f"checkpoint 中缺少可加载的 model 权重: {checkpoint_path}")

    state_dict = model.state_dict()
    unexpected_keys = sorted(set(raw_weights) - set(state_dict))
    if unexpected_keys:
        raise ValueError(f"checkpoint 包含未知参数键: {unexpected_keys[:20]}")

    updated_state = dict(state_dict)
    updated_state.update(raw_weights)
    model.load_state_dict(updated_state)
    return len(raw_weights)
