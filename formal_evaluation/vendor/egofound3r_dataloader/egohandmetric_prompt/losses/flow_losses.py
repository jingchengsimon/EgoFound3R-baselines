from __future__ import annotations

import torch


def sample_rectified_flow_state(
    target_state: torch.Tensor,
    noise_state: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if noise_state is None:
        noise_state = torch.randn_like(target_state)
    if time is None:
        time = torch.rand(target_state.shape[0], device=target_state.device, dtype=target_state.dtype)

    while time.ndim < target_state.ndim:
        time = time.unsqueeze(-1)
    x_t = (1.0 - time) * noise_state + time * target_state
    target_velocity = target_state - noise_state
    return x_t, target_velocity, time.squeeze()


def rectified_flow_loss(
    predicted_velocity: torch.Tensor,
    target_state: torch.Tensor,
    noise_state: torch.Tensor,
) -> torch.Tensor:
    target_velocity = target_state - noise_state
    return torch.mean((predicted_velocity - target_velocity) ** 2)
