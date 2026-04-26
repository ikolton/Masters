"""Segmentation supervision losses and metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_IGNORE_INDEX = -1


def _match_spatial_shape(
    logits: torch.Tensor,
    targets: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if targets is None:
        return None, valid_mask
    target_shape = tuple(int(v) for v in targets.shape[-3:])
    logit_shape = tuple(int(v) for v in logits.shape[-3:])
    if target_shape == logit_shape:
        return targets, valid_mask
    resized_targets = F.interpolate(targets.unsqueeze(1).float(), size=logit_shape, mode='nearest').squeeze(1).long()
    resized_valid_mask = valid_mask
    if valid_mask is not None:
        resized_valid_mask = F.interpolate(valid_mask.unsqueeze(1).float(), size=logit_shape, mode='nearest').squeeze(1) > 0.5
    return resized_targets, resized_valid_mask


def segmentation_supervision_loss(
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    *,
    loss_type: str,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if logits is None or targets is None:
        reference = logits if logits is not None else targets
        if reference is None:
            zero = torch.tensor(0.0)
        else:
            zero = reference.sum() * 0.0
        if return_components:
            return zero, zero, zero, zero, 0
        return zero
    prepared_targets, prepared_valid_mask = _match_spatial_shape(logits, targets, valid_mask)
    assert prepared_targets is not None
    prepared_targets = prepared_targets.long()
    if prepared_valid_mask is not None:
        prepared_targets = prepared_targets.masked_fill(~prepared_valid_mask, _IGNORE_INDEX)
    ce_numerator = F.cross_entropy(logits, prepared_targets, ignore_index=_IGNORE_INDEX, reduction="sum")
    ce_denominator = prepared_targets.ne(_IGNORE_INDEX).sum().to(dtype=logits.dtype)
    ce = ce_numerator / ce_denominator.clamp_min(1.0)
    if loss_type == "ce":
        if return_components:
            zero = logits.sum() * 0.0
            return ce, ce_numerator, ce_denominator, zero, 0
        return ce

    probabilities = F.softmax(logits, dim=1)
    valid = prepared_targets.ne(_IGNORE_INDEX)
    clamped_targets = prepared_targets.clamp(min=0)
    one_hot = F.one_hot(clamped_targets, num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
    one_hot = one_hot * valid.unsqueeze(1)
    probabilities = probabilities * valid.unsqueeze(1)
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum(dim=(2, 3, 4))
    denom = probabilities[:, 1:].sum(dim=(2, 3, 4)) + one_hot[:, 1:].sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    dice_loss = 1.0 - dice.mean()
    total = ce + dice_loss
    if return_components:
        return total, ce_numerator, ce_denominator, dice.sum(), int(dice.numel())
    return total


def multiclass_dice_score(
    logits: torch.Tensor | None,
    targets: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    *,
    return_components: bool = False,
) -> float | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits is None or targets is None:
        if return_components:
            zero = torch.zeros((0,), dtype=torch.float32)
            return zero, zero, zero
        return 0.0
    prepared_targets, prepared_valid_mask = _match_spatial_shape(logits, targets, valid_mask)
    assert prepared_targets is not None
    predictions = logits.argmax(dim=1)
    if return_components:
        class_count = int(logits.shape[1])
        intersections = torch.zeros((class_count,), device=logits.device, dtype=torch.float32)
        prediction_sums = torch.zeros_like(intersections)
        target_sums = torch.zeros_like(intersections)
        for class_index in range(1, class_count):
            pred_mask = predictions == class_index
            target_mask = prepared_targets == class_index
            if prepared_valid_mask is not None:
                pred_mask = pred_mask & prepared_valid_mask
                target_mask = target_mask & prepared_valid_mask
            intersections[class_index] = (pred_mask & target_mask).sum().to(dtype=torch.float32)
            prediction_sums[class_index] = pred_mask.sum().to(dtype=torch.float32)
            target_sums[class_index] = target_mask.sum().to(dtype=torch.float32)
        return intersections, prediction_sums, target_sums
    class_scores: list[float] = []
    for class_index in range(1, int(logits.shape[1])):
        pred_mask = predictions == class_index
        target_mask = prepared_targets == class_index
        if prepared_valid_mask is not None:
            pred_mask = pred_mask & prepared_valid_mask
            target_mask = target_mask & prepared_valid_mask
        pred_sum = int(pred_mask.sum().item())
        target_sum = int(target_mask.sum().item())
        if pred_sum == 0 and target_sum == 0:
            continue
        intersection = float((pred_mask & target_mask).sum().item())
        class_scores.append((2.0 * intersection + 1.0) / (pred_sum + target_sum + 1.0))
    if not class_scores:
        return 1.0
    return float(sum(class_scores) / len(class_scores))
