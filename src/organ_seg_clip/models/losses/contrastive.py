"""Contrastive organ alignment losses."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from torch.distributed.nn.functional import all_gather as distributed_all_gather
except Exception:  # pragma: no cover
    distributed_all_gather = None


def masked_organ_clip_loss(
    organ_embeddings: torch.Tensor,
    organ_text_embeddings: torch.Tensor,
    organ_mask: torch.Tensor,
    organ_texts: Sequence[Sequence[str]],
    logit_scale: torch.Tensor | float,
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
    image_embeddings = F.normalize(flat_image.float(), dim=-1, eps=1e-6).to(flat_image.dtype)
    text_embeddings = F.normalize(flat_text.float(), dim=-1, eps=1e-6).to(flat_text.dtype)
    scale = logit_scale if isinstance(logit_scale, torch.Tensor) else image_embeddings.new_tensor(float(logit_scale))
    if _is_distributed():
        global_image_embeddings = _gather_embeddings_with_grad(image_embeddings)
        global_text_embeddings = _gather_embeddings_with_grad(text_embeddings)
        global_mask = _gather_bool_mask(flat_mask)
        global_labels = _gather_strings(flattened_labels)
        global_organ_ids = _gather_objects(organ_ids)
        image_positive_mask = _build_positive_mask_against_global(flattened_labels, organ_ids, global_labels, global_organ_ids, device=image_embeddings.device)
        image_positive_mask = image_positive_mask & flat_mask.unsqueeze(1) & global_mask.unsqueeze(0)
        text_positive_mask = image_positive_mask
        logits_image_to_text = scale * image_embeddings @ global_text_embeddings.transpose(0, 1)
        logits_text_to_image = scale * text_embeddings @ global_image_embeddings.transpose(0, 1)
        return _multi_positive_logits_loss(logits_image_to_text, logits_text_to_image, image_positive_mask, text_positive_mask)
    positive_mask = _build_positive_mask(flattened_labels, organ_ids, device=flat_image.device)
    positive_mask = positive_mask & flat_mask.unsqueeze(1) & flat_mask.unsqueeze(0)
    logits = scale * image_embeddings @ text_embeddings.transpose(0, 1)
    return _multi_positive_logits_loss(logits, logits.transpose(0, 1), positive_mask, positive_mask)


def _multi_positive_logits_loss(
    logits_image_to_text: torch.Tensor,
    logits_text_to_image: torch.Tensor,
    image_positive_mask: torch.Tensor,
    text_positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    image_valid_rows = image_positive_mask.any(dim=1)
    text_valid_rows = text_positive_mask.any(dim=1)
    if not image_valid_rows.any() and not text_valid_rows.any():
        zero = (logits_image_to_text.sum() + logits_text_to_image.sum()) * 0.0
        return zero, {"image_to_text_top1": 0.0, "text_to_image_top1": 0.0}

    image_targets = image_positive_mask.float()
    text_targets = text_positive_mask.float()
    image_targets = image_targets / image_targets.sum(dim=1, keepdim=True).clamp(min=1.0)
    text_targets = text_targets / text_targets.sum(dim=1, keepdim=True).clamp(min=1.0)
    image_loss = (
        _soft_cross_entropy(logits_image_to_text[image_valid_rows], image_targets[image_valid_rows])
        if image_valid_rows.any()
        else logits_image_to_text.sum() * 0.0
    )
    text_loss = (
        _soft_cross_entropy(logits_text_to_image[text_valid_rows], text_targets[text_valid_rows])
        if text_valid_rows.any()
        else logits_text_to_image.sum() * 0.0
    )
    if image_valid_rows.any():
        image_hits = image_positive_mask[image_valid_rows].gather(
            1,
            logits_image_to_text[image_valid_rows].argmax(dim=1, keepdim=True),
        ).squeeze(1)
    else:
        image_hits = image_positive_mask.new_empty((0,))
    if text_valid_rows.any():
        text_hits = text_positive_mask[text_valid_rows].gather(
            1,
            logits_text_to_image[text_valid_rows].argmax(dim=1, keepdim=True),
        ).squeeze(1)
    else:
        text_hits = text_positive_mask.new_empty((0,))
    metrics = {
        "image_to_text_top1": _mean_bool_metric(image_hits),
        "text_to_image_top1": _mean_bool_metric(text_hits),
    }
    return 0.5 * (image_loss + text_loss), metrics


def _mean_bool_metric(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def _build_positive_mask(labels: Sequence[str], organ_ids: Sequence[int], *, device: torch.device) -> torch.Tensor:
    pair_count = len(labels)
    mask = torch.zeros((pair_count, pair_count), device=device, dtype=torch.bool)
    for row_index, (label, organ_id) in enumerate(zip(labels, organ_ids)):
        for col_index, (other_label, other_organ_id) in enumerate(zip(labels, organ_ids)):
            if organ_id == other_organ_id and label == other_label:
                mask[row_index, col_index] = True
    return mask


def _build_positive_mask_against_global(
    local_labels: Sequence[str],
    local_organ_ids: Sequence[int],
    global_labels: Sequence[str],
    global_organ_ids: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((len(local_labels), len(global_labels)), device=device, dtype=torch.bool)
    for row_index, (label, organ_id) in enumerate(zip(local_labels, local_organ_ids)):
        for col_index, (other_label, other_organ_id) in enumerate(zip(global_labels, global_organ_ids)):
            if organ_id == other_organ_id and label == other_label:
                mask[row_index, col_index] = True
    return mask


def _normalize_text_label(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _gather_embeddings_with_grad(embeddings: torch.Tensor) -> torch.Tensor:
    if not _is_distributed():
        return embeddings
    if distributed_all_gather is None:
        raise RuntimeError("Distributed organ CLIP requires torch.distributed.nn.functional.all_gather.")
    gathered = distributed_all_gather(embeddings)
    if isinstance(gathered, torch.Tensor):
        return gathered
    return torch.cat(tuple(gathered), dim=0)


def _gather_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    if not _is_distributed():
        return mask
    if distributed_all_gather is None:
        raise RuntimeError("Distributed organ CLIP requires torch.distributed.nn.functional.all_gather.")
    gathered = distributed_all_gather(mask.to(dtype=torch.float32))
    if isinstance(gathered, torch.Tensor):
        return gathered > 0.5
    return torch.cat(tuple(gathered), dim=0) > 0.5


def _gather_strings(strings: Sequence[str]) -> list[str]:
    if not _is_distributed():
        return list(strings)
    gathered: list[list[str] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, list(strings))
    merged: list[str] = []
    for chunk in gathered:
        if chunk is not None:
            merged.extend(chunk)
    return merged


def _gather_objects(objects: Sequence[int]) -> list[int]:
    if not _is_distributed():
        return list(objects)
    gathered: list[list[int] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, list(objects))
    merged: list[int] = []
    for chunk in gathered:
        if chunk is not None:
            merged.extend(chunk)
    return merged
