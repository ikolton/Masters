"""Shared typed interfaces for OrganSegCLIP."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch


@dataclass(frozen=True)
class EncoderBatch:
    study_ids: list[str]
    images: torch.Tensor
    image_mask: torch.Tensor
    segmentations: torch.Tensor | None
    segmentation_mask: torch.Tensor | None
    report_texts: list[str]
    organ_texts: list[list[str]]
    organ_raw_texts: list[list[str]]
    organ_text_mask: torch.Tensor
    organ_labels: torch.Tensor
    organ_label_mask: torch.Tensor
    lesion_global_labels: torch.Tensor
    lesion_global_mask: torch.Tensor
    lesion_organ_labels: torch.Tensor
    lesion_organ_mask: torch.Tensor
    metadata: list[dict[str, Any]]


@dataclass(frozen=True)
class OrganSegOutput:
    organ_image_embeddings: torch.Tensor
    organ_text_embeddings: torch.Tensor
    report_image_embeddings: torch.Tensor
    report_text_embeddings: torch.Tensor
    diagnostic_logits: torch.Tensor
    lesion_global_logits: torch.Tensor
    lesion_organ_logits: torch.Tensor
    logit_scale: torch.Tensor
    organ_logit_scale: torch.Tensor
    organ_logit_bias: torch.Tensor
    report_logit_scale: torch.Tensor
    report_logit_bias: torch.Tensor
    segmentation_loss: torch.Tensor
    segmentation_dice: float
    segmentation_patch_count: int
    patch_organ_presence_loss: torch.Tensor
    patch_organ_presence_accuracy: float
    patch_organ_presence_count: int
    organ_attention_loss: torch.Tensor
    organ_attention_accuracy: float
    organ_attention_positive_accuracy: float
    organ_attention_negative_accuracy: float
    organ_attention_count: int
    organ_attention_positive_count: int
    organ_attention_negative_count: int
    patches_per_batch_total: int = 0
    patches_per_study_mean: float = 0.0
    patches_per_study_max: int = 0
    segmentation_oom_fallback_count: int = 0


@dataclass(frozen=True)
class RepresentationLossOutput:
    total_loss: torch.Tensor
    organ_clip_loss: torch.Tensor
    report_clip_loss: torch.Tensor
    organ_alignment_loss: torch.Tensor
    report_alignment_loss: torch.Tensor
    segmentation_loss: torch.Tensor
    diagnostic_loss: torch.Tensor
    patch_organ_presence_loss: torch.Tensor
    organ_attention_loss: torch.Tensor
    lesion_global_loss: torch.Tensor
    lesion_organ_loss: torch.Tensor

    def to_dict(self) -> dict[str, float]:
        return {
            field.name: float(getattr(self, field.name).detach().item())
            for field in fields(self)
        }
