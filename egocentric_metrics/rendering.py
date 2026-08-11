"""Deterministic full-reference image quality metrics."""

from __future__ import annotations

import numpy as np

from .common import as_numpy


def psnr(prediction, target, *, data_range: float) -> float:
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape or data_range <= 0.0:
        raise ValueError("images must have matching shapes and data_range must be positive")
    mse = float(np.mean(np.square(pred - gt)))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(data_range**2 / mse))


def _ssim_channel(prediction: np.ndarray, target: np.ndarray, data_range: float, window_size: int) -> float:
    height, width = prediction.shape
    size = min(window_size, height, width)
    if size < 2:
        raise ValueError("images must be at least 2 by 2 for SSIM")
    pred_windows = np.lib.stride_tricks.sliding_window_view(prediction, (size, size))
    gt_windows = np.lib.stride_tricks.sliding_window_view(target, (size, size))
    mu_pred = pred_windows.mean(axis=(-2, -1))
    mu_gt = gt_windows.mean(axis=(-2, -1))
    ddof = 1 if size * size > 1 else 0
    var_pred = pred_windows.var(axis=(-2, -1), ddof=ddof)
    var_gt = gt_windows.var(axis=(-2, -1), ddof=ddof)
    covariance = ((pred_windows - mu_pred[..., None, None]) * (gt_windows - mu_gt[..., None, None])).sum(axis=(-2, -1)) / (size * size - ddof)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    value = ((2.0 * mu_pred * mu_gt + c1) * (2.0 * covariance + c2)) / ((np.square(mu_pred) + np.square(mu_gt) + c1) * (var_pred + var_gt + c2))
    return float(np.mean(value))


def ssim(prediction, target, *, data_range: float, window_size: int = 7) -> float:
    pred = as_numpy(prediction, dtype=float)
    gt = as_numpy(target, dtype=float)
    if pred.shape != gt.shape or pred.ndim not in {2, 3} or data_range <= 0.0:
        raise ValueError("images must have matching (H, W) or (H, W, C) shapes and positive data_range")
    if pred.ndim == 2:
        return _ssim_channel(pred, gt, data_range, window_size)
    return float(np.mean([_ssim_channel(pred[..., channel], gt[..., channel], data_range, window_size) for channel in range(pred.shape[-1])]))


def image_metrics(prediction, target, *, data_range: float, window_size: int = 7) -> dict[str, float]:
    return {
        "psnr": psnr(prediction, target, data_range=data_range),
        "ssim": ssim(prediction, target, data_range=data_range, window_size=window_size),
    }
