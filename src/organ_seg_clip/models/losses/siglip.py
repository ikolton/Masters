"""SigLIP-style pairwise alignment losses."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .contrastive import (
    _build_positive_mask,
    _build_positive_mask_against_global,
    _gather_bool_mask,
    _gather_embeddings_with_grad,
    _gather_objects,
    _gather_strings,
    _is_distributed,
    _mean_bool_metric,
    _normalize_text_label,
)


def masked_organ_siglip_loss(
    organ_embeddings: torch.Tensor,
    organ_text_embeddings: torch.Tensor,
    organ_mask: torch.Tensor,
    organ_texts: Sequence[Sequence[str]],
    logit_scale: torch.Tensor | float,
    logit_bias: torch.Tensor | float,
    *,
    pair_balance: bool = False,
    positive_weight: float = 1.0,
    same_organ_weight: float = 1.0,
    cross_organ_weight: float = 1.0,
    finding_counts: dict[tuple[int, str], int] | None = None,
    frequency_balance: bool = False,
    frequency_balance_power: float = 0.5,
    frequency_balance_min: float = 0.25,
    frequency_balance_max: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if organ_embeddings.shape[:2] != organ_text_embeddings.shape[:2] or organ_embeddings.shape[:2] != organ_mask.shape:
        raise ValueError("organ embeddings, text embeddings, and mask must agree on [batch, organ].")
    flat_image = organ_embeddings.reshape(-1, organ_embeddings.shape[-1])
    flat_text = organ_text_embeddings.reshape(-1, organ_text_embeddings.shape[-1])
    flat_mask = organ_mask.reshape(-1)
    if not flat_mask.any():
        zero = organ_embeddings.sum() * 0.0
        return zero, {"image_to_text_top1": 0.0, "text_to_image_top1": 0.0}

    flattened_labels: list[str] = []
    organ_ids: list[int] = []
    for sample_texts in organ_texts:
        for organ_index, text in enumerate(sample_texts):
            flattened_labels.append(_normalize_text_label(text))
            organ_ids.append(int(organ_index))
    row_weights = _build_row_weights(
        organ_ids,
        flattened_labels,
        finding_counts=finding_counts,
        enabled=frequency_balance,
        power=float(frequency_balance_power),
        min_weight=float(frequency_balance_min),
        max_weight=float(frequency_balance_max),
        device=flat_image.device,
    )

    image_embeddings = F.normalize(flat_image.float(), dim=-1, eps=1e-6).to(flat_image.dtype)
    text_embeddings = F.normalize(flat_text.float(), dim=-1, eps=1e-6).to(flat_text.dtype)
    scale = logit_scale if isinstance(logit_scale, torch.Tensor) else image_embeddings.new_tensor(float(logit_scale))
    bias = logit_bias if isinstance(logit_bias, torch.Tensor) else image_embeddings.new_tensor(float(logit_bias))
    if _is_distributed():
        global_image_embeddings = _gather_embeddings_with_grad(image_embeddings)
        global_text_embeddings = _gather_embeddings_with_grad(text_embeddings)
        global_mask = _gather_bool_mask(flat_mask)
        global_labels = _gather_strings(flattened_labels)
        global_organ_ids = _gather_objects(organ_ids)
        image_positive_mask = _build_positive_mask_against_global(flattened_labels, organ_ids, global_labels, global_organ_ids, device=image_embeddings.device)
        image_valid_pairs = flat_mask.unsqueeze(1) & global_mask.unsqueeze(0)
        image_positive_mask = image_positive_mask & image_valid_pairs
        text_positive_mask = image_positive_mask
        logits_image_to_text = scale * image_embeddings @ global_text_embeddings.transpose(0, 1) + bias
        logits_text_to_image = scale * text_embeddings @ global_image_embeddings.transpose(0, 1) + bias
        if pair_balance:
            same_organ_mask = _build_same_organ_mask(organ_ids, global_organ_ids, device=image_embeddings.device)
            if float(cross_organ_weight) <= 0.0:
                image_valid_pairs = image_valid_pairs & same_organ_mask
                image_positive_mask = image_positive_mask & image_valid_pairs
                text_positive_mask = text_positive_mask & image_valid_pairs
            return _pairwise_balanced_organ_siglip_loss(
                logits_image_to_text,
                logits_text_to_image,
                image_positive_mask,
                text_positive_mask,
                image_valid_pairs,
                image_valid_pairs,
                same_organ_mask,
                same_organ_mask,
                positive_weight=float(positive_weight),
                same_organ_weight=float(same_organ_weight),
                cross_organ_weight=float(cross_organ_weight),
                image_row_weights=row_weights,
                text_row_weights=row_weights,
            )
        return _pairwise_siglip_loss(
            logits_image_to_text,
            logits_text_to_image,
            image_positive_mask,
            text_positive_mask,
            image_valid_pairs,
            image_valid_pairs,
            image_row_weights=row_weights,
            text_row_weights=row_weights,
        )

    positive_mask = _build_positive_mask(flattened_labels, organ_ids, device=flat_image.device)
    valid_pairs = flat_mask.unsqueeze(1) & flat_mask.unsqueeze(0)
    positive_mask = positive_mask & valid_pairs
    logits = scale * image_embeddings @ text_embeddings.transpose(0, 1) + bias
    if pair_balance:
        same_organ_mask = _build_same_organ_mask(organ_ids, organ_ids, device=flat_image.device)
        if float(cross_organ_weight) <= 0.0:
            valid_pairs = valid_pairs & same_organ_mask
            positive_mask = positive_mask & valid_pairs
        return _pairwise_balanced_organ_siglip_loss(
            logits,
            logits.transpose(0, 1),
            positive_mask,
            positive_mask,
            valid_pairs,
            valid_pairs,
            same_organ_mask,
            same_organ_mask,
            positive_weight=float(positive_weight),
            same_organ_weight=float(same_organ_weight),
            cross_organ_weight=float(cross_organ_weight),
            image_row_weights=row_weights,
            text_row_weights=row_weights,
        )
    return _pairwise_siglip_loss(
        logits,
        logits.transpose(0, 1),
        positive_mask,
        positive_mask,
        valid_pairs,
        valid_pairs,
        image_row_weights=row_weights,
        text_row_weights=row_weights,
    )


def masked_report_siglip_loss(
    report_image_embeddings: torch.Tensor,
    report_text_embeddings: torch.Tensor,
    report_mask: torch.Tensor,
    study_ids: Sequence[str],
    logit_scale: torch.Tensor | float,
    logit_bias: torch.Tensor | float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if report_image_embeddings.shape != report_text_embeddings.shape or report_image_embeddings.shape[0] != report_mask.shape[0]:
        raise ValueError("report embeddings and mask must agree on batch size.")
    if not report_mask.any():
        zero = report_image_embeddings.sum() * 0.0
        return zero, {"image_to_text_top1": 0.0, "text_to_image_top1": 0.0, "valid_count": 0.0}
    image_embeddings = F.normalize(report_image_embeddings.float(), dim=-1, eps=1e-6).to(report_image_embeddings.dtype)
    text_embeddings = F.normalize(report_text_embeddings.float(), dim=-1, eps=1e-6).to(report_text_embeddings.dtype)
    scale = logit_scale if isinstance(logit_scale, torch.Tensor) else image_embeddings.new_tensor(float(logit_scale))
    bias = logit_bias if isinstance(logit_bias, torch.Tensor) else image_embeddings.new_tensor(float(logit_bias))
    local_ids = [_normalize_text_label(study_id) for study_id in study_ids]
    if _is_distributed():
        global_image_embeddings = _gather_embeddings_with_grad(image_embeddings)
        global_text_embeddings = _gather_embeddings_with_grad(text_embeddings)
        global_mask = _gather_bool_mask(report_mask)
        global_ids = _gather_strings(local_ids)
        positive_mask = _build_id_positive_mask(local_ids, global_ids, device=image_embeddings.device)
        valid_pairs = report_mask.unsqueeze(1) & global_mask.unsqueeze(0)
        positive_mask = positive_mask & valid_pairs
        logits_image_to_text = scale * image_embeddings @ global_text_embeddings.transpose(0, 1) + bias
        logits_text_to_image = scale * text_embeddings @ global_image_embeddings.transpose(0, 1) + bias
        loss, metrics = _pairwise_siglip_loss(logits_image_to_text, logits_text_to_image, positive_mask, positive_mask, valid_pairs, valid_pairs)
        metrics["valid_count"] = float(global_mask.sum().item())
        return loss, metrics
    positive_mask = _build_id_positive_mask(local_ids, local_ids, device=image_embeddings.device)
    valid_pairs = report_mask.unsqueeze(1) & report_mask.unsqueeze(0)
    positive_mask = positive_mask & valid_pairs
    logits = scale * image_embeddings @ text_embeddings.transpose(0, 1) + bias
    loss, metrics = _pairwise_siglip_loss(logits, logits.transpose(0, 1), positive_mask, positive_mask, valid_pairs, valid_pairs)
    metrics["valid_count"] = float(report_mask.sum().item())
    return loss, metrics


def _pairwise_siglip_loss(
    logits_image_to_text: torch.Tensor,
    logits_text_to_image: torch.Tensor,
    image_positive_mask: torch.Tensor,
    text_positive_mask: torch.Tensor,
    image_valid_pairs: torch.Tensor,
    text_valid_pairs: torch.Tensor,
    image_row_weights: torch.Tensor | None = None,
    text_row_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    image_valid_rows = image_positive_mask.any(dim=1)
    text_valid_rows = text_positive_mask.any(dim=1)
    if not image_valid_rows.any() and not text_valid_rows.any():
        zero = (logits_image_to_text.sum() + logits_text_to_image.sum()) * 0.0
        return zero, {"image_to_text_top1": 0.0, "text_to_image_top1": 0.0}
    image_loss = _directional_siglip_loss(
        logits_image_to_text,
        image_positive_mask,
        image_valid_pairs,
        image_valid_rows,
        row_weights=image_row_weights,
    )
    text_loss = _directional_siglip_loss(
        logits_text_to_image,
        text_positive_mask,
        text_valid_pairs,
        text_valid_rows,
        row_weights=text_row_weights,
    )
    image_hits = _top1_hits(logits_image_to_text, image_positive_mask, image_valid_pairs, image_valid_rows)
    text_hits = _top1_hits(logits_text_to_image, text_positive_mask, text_valid_pairs, text_valid_rows)
    image_pos_mean, image_neg_mean = _positive_negative_logit_means(logits_image_to_text, image_positive_mask, image_valid_pairs)
    text_pos_mean, text_neg_mean = _positive_negative_logit_means(logits_text_to_image, text_positive_mask, text_valid_pairs)
    return 0.5 * (image_loss + text_loss), {
        "image_to_text_top1": _mean_bool_metric(image_hits),
        "text_to_image_top1": _mean_bool_metric(text_hits),
        "positive_logit_mean": 0.5 * (image_pos_mean + text_pos_mean),
        "negative_logit_mean": 0.5 * (image_neg_mean + text_neg_mean),
        "logit_gap": 0.5 * ((image_pos_mean - image_neg_mean) + (text_pos_mean - text_neg_mean)),
        "row_weight_mean": _row_weight_mean(image_row_weights),
    }



def _pairwise_balanced_organ_siglip_loss(
    logits_image_to_text: torch.Tensor,
    logits_text_to_image: torch.Tensor,
    image_positive_mask: torch.Tensor,
    text_positive_mask: torch.Tensor,
    image_valid_pairs: torch.Tensor,
    text_valid_pairs: torch.Tensor,
    image_same_organ_mask: torch.Tensor,
    text_same_organ_mask: torch.Tensor,
    *,
    positive_weight: float,
    same_organ_weight: float,
    cross_organ_weight: float,
    image_row_weights: torch.Tensor | None = None,
    text_row_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    image_valid_rows = image_positive_mask.any(dim=1)
    text_valid_rows = text_positive_mask.any(dim=1)
    if not image_valid_rows.any() and not text_valid_rows.any():
        zero = (logits_image_to_text.sum() + logits_text_to_image.sum()) * 0.0
        return zero, {"image_to_text_top1": 0.0, "text_to_image_top1": 0.0}
    image_loss = _directional_balanced_organ_siglip_loss(
        logits_image_to_text,
        image_positive_mask,
        image_valid_pairs,
        image_same_organ_mask,
        image_valid_rows,
        positive_weight=positive_weight,
        same_organ_weight=same_organ_weight,
        cross_organ_weight=cross_organ_weight,
        row_weights=image_row_weights,
    )
    text_loss = _directional_balanced_organ_siglip_loss(
        logits_text_to_image,
        text_positive_mask,
        text_valid_pairs,
        text_same_organ_mask,
        text_valid_rows,
        positive_weight=positive_weight,
        same_organ_weight=same_organ_weight,
        cross_organ_weight=cross_organ_weight,
        row_weights=text_row_weights,
    )
    image_hits = _top1_hits(logits_image_to_text, image_positive_mask, image_valid_pairs, image_valid_rows)
    text_hits = _top1_hits(logits_text_to_image, text_positive_mask, text_valid_pairs, text_valid_rows)
    image_means = _organ_pair_type_logit_means(logits_image_to_text, image_positive_mask, image_valid_pairs, image_same_organ_mask)
    text_means = _organ_pair_type_logit_means(logits_text_to_image, text_positive_mask, text_valid_pairs, text_same_organ_mask)
    positive_mean = 0.5 * (image_means["positive"] + text_means["positive"])
    same_negative_mean = 0.5 * (image_means["same_organ_negative"] + text_means["same_organ_negative"])
    cross_negative_mean = 0.5 * (image_means["cross_organ_negative"] + text_means["cross_organ_negative"])
    negative_mean = 0.5 * (image_means["negative"] + text_means["negative"])
    return 0.5 * (image_loss + text_loss), {
        "image_to_text_top1": _mean_bool_metric(image_hits),
        "text_to_image_top1": _mean_bool_metric(text_hits),
        "positive_logit_mean": positive_mean,
        "negative_logit_mean": negative_mean,
        "logit_gap": positive_mean - negative_mean,
        "same_organ_negative_logit_mean": same_negative_mean,
        "cross_organ_negative_logit_mean": cross_negative_mean,
        "same_organ_logit_gap": positive_mean - same_negative_mean,
        "cross_organ_logit_gap": positive_mean - cross_negative_mean,
        "row_weight_mean": _row_weight_mean(image_row_weights),
    }

def _positive_negative_logit_means(logits: torch.Tensor, positive_mask: torch.Tensor, valid_pairs: torch.Tensor) -> tuple[float, float]:
    valid_positive = positive_mask & valid_pairs
    valid_negative = (~positive_mask) & valid_pairs
    positive_mean = _masked_logit_mean(logits, valid_positive)
    negative_mean = _masked_logit_mean(logits, valid_negative)
    return positive_mean, negative_mean


def _masked_logit_mean(logits: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return float(logits.detach()[mask].float().mean().item())



def _organ_pair_type_logit_means(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_pairs: torch.Tensor,
    same_organ_mask: torch.Tensor,
) -> dict[str, float]:
    valid_positive = positive_mask & valid_pairs
    valid_negative = (~positive_mask) & valid_pairs
    same_organ_negative = valid_negative & same_organ_mask
    cross_organ_negative = valid_negative & ~same_organ_mask
    return {
        "positive": _masked_logit_mean(logits, valid_positive),
        "negative": _masked_logit_mean(logits, valid_negative),
        "same_organ_negative": _masked_logit_mean(logits, same_organ_negative),
        "cross_organ_negative": _masked_logit_mean(logits, cross_organ_negative),
    }


def _directional_balanced_organ_siglip_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_pairs: torch.Tensor,
    same_organ_mask: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    positive_weight: float,
    same_organ_weight: float,
    cross_organ_weight: float,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not valid_rows.any():
        return logits.sum() * 0.0
    row_logits = logits[valid_rows]
    row_positive = positive_mask[valid_rows]
    row_valid = valid_pairs[valid_rows]
    row_same_organ = same_organ_mask[valid_rows]
    positive_pairs = row_positive & row_valid
    negative_pairs = (~row_positive) & row_valid
    same_organ_negative_pairs = negative_pairs & row_same_organ
    cross_organ_negative_pairs = negative_pairs & ~row_same_organ

    positive_loss = -F.logsigmoid(row_logits)
    negative_loss = -F.logsigmoid(-row_logits)
    row_loss = _weighted_category_row_loss(
        positive_loss,
        negative_loss,
        positive_pairs,
        same_organ_negative_pairs,
        cross_organ_negative_pairs,
        positive_weight=positive_weight,
        same_organ_weight=same_organ_weight,
        cross_organ_weight=cross_organ_weight,
    )
    if row_weights is None:
        return row_loss.mean()
    return _weighted_mean(row_loss, row_weights[valid_rows])


def _weighted_category_row_loss(
    positive_loss: torch.Tensor,
    negative_loss: torch.Tensor,
    positive_pairs: torch.Tensor,
    same_organ_negative_pairs: torch.Tensor,
    cross_organ_negative_pairs: torch.Tensor,
    *,
    positive_weight: float,
    same_organ_weight: float,
    cross_organ_weight: float,
) -> torch.Tensor:
    positive_counts = positive_pairs.float().sum(dim=1)
    same_counts = same_organ_negative_pairs.float().sum(dim=1)
    cross_counts = cross_organ_negative_pairs.float().sum(dim=1)
    positive_row_loss = (positive_loss * positive_pairs.float()).sum(dim=1) / positive_counts.clamp(min=1.0)
    same_row_loss = (negative_loss * same_organ_negative_pairs.float()).sum(dim=1) / same_counts.clamp(min=1.0)
    cross_row_loss = (negative_loss * cross_organ_negative_pairs.float()).sum(dim=1) / cross_counts.clamp(min=1.0)
    weighted_sum = positive_row_loss * float(positive_weight) * (positive_counts > 0).float()
    weighted_sum = weighted_sum + same_row_loss * float(same_organ_weight) * (same_counts > 0).float()
    weighted_sum = weighted_sum + cross_row_loss * float(cross_organ_weight) * (cross_counts > 0).float()
    weight_sum = float(positive_weight) * (positive_counts > 0).float()
    weight_sum = weight_sum + float(same_organ_weight) * (same_counts > 0).float()
    weight_sum = weight_sum + float(cross_organ_weight) * (cross_counts > 0).float()
    return weighted_sum / weight_sum.clamp(min=1.0)

def _directional_siglip_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_pairs: torch.Tensor,
    valid_rows: torch.Tensor,
    *,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not valid_rows.any():
        return logits.sum() * 0.0
    row_logits = logits[valid_rows]
    row_positive = positive_mask[valid_rows]
    row_valid = valid_pairs[valid_rows]
    positive_pairs = row_positive & row_valid
    negative_pairs = (~row_positive) & row_valid

    positive_loss = -F.logsigmoid(row_logits)
    negative_loss = -F.logsigmoid(-row_logits)
    positive_counts = positive_pairs.float().sum(dim=1)
    negative_counts = negative_pairs.float().sum(dim=1)

    positive_row_loss = (positive_loss * positive_pairs.float()).sum(dim=1) / positive_counts.clamp(min=1.0)
    negative_row_loss = (negative_loss * negative_pairs.float()).sum(dim=1) / negative_counts.clamp(min=1.0)
    row_loss = positive_row_loss + negative_row_loss
    row_loss = torch.where(negative_counts > 0, 0.5 * row_loss, row_loss)
    if row_weights is None:
        return row_loss.mean()
    return _weighted_mean(row_loss, row_weights[valid_rows])


def _top1_hits(logits: torch.Tensor, positive_mask: torch.Tensor, valid_pairs: torch.Tensor, valid_rows: torch.Tensor) -> torch.Tensor:
    if not valid_rows.any():
        return positive_mask.new_empty((0,))
    masked_logits = logits[valid_rows].masked_fill(~valid_pairs[valid_rows], float("-inf"))
    winners = masked_logits.argmax(dim=1, keepdim=True)
    return positive_mask[valid_rows].gather(1, winners).squeeze(1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    normalized_weights = weights.float().clamp(min=0.0)
    return (values * normalized_weights).sum() / normalized_weights.sum().clamp(min=1.0)


def _build_row_weights(
    organ_ids: Sequence[int],
    labels: Sequence[str],
    *,
    finding_counts: dict[tuple[int, str], int] | None,
    enabled: bool,
    power: float,
    min_weight: float,
    max_weight: float,
    device: torch.device,
) -> torch.Tensor:
    if not enabled or not finding_counts:
        return torch.ones((len(labels),), device=device, dtype=torch.float32)
    weights: list[float] = []
    for organ_id, label in zip(organ_ids, labels):
        count = max(1, int(finding_counts.get((int(organ_id), str(label)), 1)))
        weight = count ** (-float(power))
        weight = max(float(min_weight), min(float(max_weight), float(weight)))
        weights.append(weight)
    tensor = torch.tensor(weights, device=device, dtype=torch.float32)
    return tensor / tensor.mean().clamp(min=1.0e-6)


def _row_weight_mean(row_weights: torch.Tensor | None) -> float:
    if row_weights is None or row_weights.numel() == 0:
        return 0.0
    return float(row_weights.detach().float().mean().item())



def _build_same_organ_mask(local_organ_ids: Sequence[int], global_organ_ids: Sequence[int], *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(local_organ_ids), len(global_organ_ids)), device=device, dtype=torch.bool)
    for row_index, organ_id in enumerate(local_organ_ids):
        for col_index, other_organ_id in enumerate(global_organ_ids):
            if int(organ_id) == int(other_organ_id):
                mask[row_index, col_index] = True
    return mask

def _build_id_positive_mask(local_ids: Sequence[str], global_ids: Sequence[str], *, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(local_ids), len(global_ids)), device=device, dtype=torch.bool)
    for row_index, study_id in enumerate(local_ids):
        for col_index, other_study_id in enumerate(global_ids):
            if study_id == other_study_id:
                mask[row_index, col_index] = True
    return mask
