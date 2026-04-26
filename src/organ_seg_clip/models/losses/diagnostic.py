"""Diagnostic supervision losses and metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_binary_diagnostic_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if logits.shape != labels.shape or logits.shape != label_mask.shape:
        raise ValueError("diagnostic logits, labels, and masks must have identical shapes.")
    if not label_mask.any():
        zero = logits.sum() * 0.0
        return zero, {"diagnostic_accuracy": 0.0}
    masked_logits = logits[label_mask]
    masked_labels = labels[label_mask]
    loss = F.binary_cross_entropy_with_logits(masked_logits, masked_labels)
    predictions = (masked_logits.sigmoid() >= 0.5).float()
    accuracy = float((predictions == masked_labels).float().mean().item())
    return loss, {"diagnostic_accuracy": accuracy}
