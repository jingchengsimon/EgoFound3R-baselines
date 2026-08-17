from __future__ import annotations

import torch
import torch.nn.functional as F

from egohandmetric_prompt.losses.marker_losses import hand_confidence_weighted_loss


CONTACT_LOG_CONF_MIN = 0.0
CONTACT_LOG_CONF_MAX = 2.0


def _validate_contact_tensors(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError(
            "contact logits、targets 与 mask 形状必须一致，当前为 "
            f"{tuple(logits.shape)}、{tuple(targets.shape)}、{tuple(mask.shape)}"
        )


def masked_contact_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    _validate_contact_tensors(logits, targets, mask)
    mask = mask.to(device=logits.device, dtype=torch.bool)
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    if not torch.any(mask):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def contact_confidence_from_log(log_conf: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_conf.clamp(CONTACT_LOG_CONF_MIN, CONTACT_LOG_CONF_MAX))


def masked_contact_confidence_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    log_conf: torch.Tensor,
    *,
    alpha: float,
    use_confidence: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    _validate_contact_tensors(logits, targets, mask)
    if log_conf.shape != logits.shape:
        raise ValueError(
            "contact log_conf 必须与 logits 形状一致，当前为 "
            f"{tuple(log_conf.shape)} 与 {tuple(logits.shape)}"
        )
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    per_point_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return hand_confidence_weighted_loss(
        per_point_bce,
        log_conf,
        mask,
        alpha=alpha,
        log_min=CONTACT_LOG_CONF_MIN,
        log_max=CONTACT_LOG_CONF_MAX,
        use_confidence=use_confidence,
    )


def contact_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    _validate_contact_tensors(logits, targets, mask)
    mask = mask.to(device=logits.device, dtype=torch.bool)
    targets = targets.to(device=logits.device)
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        return {
            "contact_accuracy": 0.0,
            "contact_precision": 0.0,
            "contact_recall": 0.0,
            "contact_f1": 0.0,
            "contact_valid_count": 0.0,
            "contact_true_positive_count": 0.0,
            "contact_false_positive_count": 0.0,
            "contact_false_negative_count": 0.0,
            "contact_true_negative_count": 0.0,
        }

    predicted = logits.detach()[mask] >= 0.0
    expected = targets.detach()[mask] >= 0.5
    true_positive = int((predicted & expected).sum().item())
    false_positive = int((predicted & ~expected).sum().item())
    false_negative = int((~predicted & expected).sum().item())
    correct = int((predicted == expected).sum().item())
    true_negative = correct - true_positive
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "contact_accuracy": correct / valid_count,
        "contact_precision": 0.0 if precision_denominator == 0 else true_positive / precision_denominator,
        "contact_recall": 0.0 if recall_denominator == 0 else true_positive / recall_denominator,
        "contact_f1": 0.0 if f1_denominator == 0 else (2 * true_positive) / f1_denominator,
        "contact_valid_count": float(valid_count),
        "contact_true_positive_count": float(true_positive),
        "contact_false_positive_count": float(false_positive),
        "contact_false_negative_count": float(false_negative),
        "contact_true_negative_count": float(true_negative),
    }
