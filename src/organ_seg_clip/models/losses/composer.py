"""Loss composer for OrganSegCLIP."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from ...config.schemas import LossConfig
from ..interfaces.types import EncoderBatch, OrganSegOutput, RepresentationLossOutput
from .contrastive import masked_organ_clip_loss
from .siglip import masked_organ_siglip_loss, masked_report_siglip_loss
from .diagnostic import masked_binary_diagnostic_loss


class OrganSegLossComposer(nn.Module):
    def __init__(self, config: LossConfig, *, organ_finding_counts: dict[tuple[int, str], int] | None = None) -> None:
        super().__init__()
        self.config = config
        self.organ_finding_counts = dict(organ_finding_counts or {})

    def forward(self, outputs: OrganSegOutput, batch: EncoderBatch) -> tuple[RepresentationLossOutput, dict[str, float]]:
        # Organ alignment
        if float(self.config.organ_alignment_weight or 0.0) == 0.0:
            organ_clip_loss = outputs.organ_image_embeddings.sum() * 0.0
            organ_metrics = {"image_to_text_top1": 0.0, "text_to_image_top1": 0.0}
        elif self.config.alignment_type == "siglip":
            organ_clip_loss, organ_metrics = masked_organ_siglip_loss(
                outputs.organ_image_embeddings,
                outputs.organ_text_embeddings,
                batch.organ_text_mask,
                batch.organ_raw_texts,
                outputs.organ_logit_scale,
                outputs.organ_logit_bias,
                pair_balance=bool(self.config.organ_pair_balance),
                positive_weight=float(self.config.organ_positive_weight),
                same_organ_weight=float(self.config.organ_same_organ_weight),
                cross_organ_weight=float(self.config.organ_cross_organ_weight),
                finding_counts=self.organ_finding_counts,
                frequency_balance=bool(self.config.organ_frequency_balance),
                frequency_balance_power=float(self.config.organ_frequency_balance_power),
                frequency_balance_min=float(self.config.organ_frequency_balance_min),
                frequency_balance_max=float(self.config.organ_frequency_balance_max),
                soft_positive_threshold=self.config.siglip_soft_positive_threshold,
                hard_negative_weight=float(self.config.siglip_hard_negative_weight),
            )
        else:
            organ_clip_loss, organ_metrics = masked_organ_clip_loss(
                outputs.organ_image_embeddings,
                outputs.organ_text_embeddings,
                batch.organ_text_mask,
                batch.organ_raw_texts,
                outputs.logit_scale,
            )
        # Report alignment
        if float(self.config.report_alignment_weight or 0.0) != 0.0:
            if self.config.alignment_type == "siglip":
                report_mask = torch.tensor([bool(text) for text in batch.report_texts], device=outputs.report_image_embeddings.device, dtype=torch.bool)
                report_clip_loss, report_metrics = masked_report_siglip_loss(
                    outputs.report_image_embeddings,
                    outputs.report_text_embeddings,
                    report_mask,
                    batch.study_ids,
                    outputs.report_logit_scale,
                    outputs.report_logit_bias,
                )
            else:
                report_clip_loss, report_metrics = _masked_report_clip_loss(outputs, batch)
        else:
            report_clip_loss = outputs.report_image_embeddings.sum() * 0.0
            report_metrics = {"image_to_text_top1": 0.0, "text_to_image_top1": 0.0, "valid_count": 0.0}
        diagnostic_loss, diagnostic_metrics = masked_binary_diagnostic_loss(
            outputs.diagnostic_logits,
            batch.organ_labels,
            batch.organ_label_mask,
        )
        if float(self.config.lesion_global_weight) != 0.0:
            lesion_global_loss, lesion_global_metrics = _masked_named_binary_loss(
                outputs.lesion_global_logits,
                batch.lesion_global_labels,
                batch.lesion_global_mask,
                metric_name="lesion_global_accuracy",
            )
        else:
            lesion_global_loss = outputs.lesion_global_logits.sum() * 0.0
            lesion_global_metrics = {"lesion_global_accuracy": 0.0}
        if float(self.config.lesion_organ_weight) != 0.0:
            lesion_organ_loss, lesion_organ_metrics = _masked_named_binary_loss(
                outputs.lesion_organ_logits,
                batch.lesion_organ_labels,
                batch.lesion_organ_mask,
                metric_name="lesion_organ_accuracy",
            )
        else:
            lesion_organ_loss = outputs.lesion_organ_logits.sum() * 0.0
            lesion_organ_metrics = {"lesion_organ_accuracy": 0.0}
        segmentation_loss = outputs.segmentation_loss
        patch_organ_presence_loss = outputs.patch_organ_presence_loss
        organ_attention_loss = outputs.organ_attention_loss
        total_loss = (
            float(self.config.organ_alignment_weight or 0.0) * organ_clip_loss
            + float(self.config.report_alignment_weight or 0.0) * report_clip_loss
            + self.config.segmentation_weight * segmentation_loss
            + self.config.diagnostic_weight * diagnostic_loss
            + self.config.patch_organ_presence_weight * patch_organ_presence_loss
            + self.config.organ_attention_weight * organ_attention_loss
            + self.config.lesion_global_weight * lesion_global_loss
            + self.config.lesion_organ_weight * lesion_organ_loss
        )
        loss_output = RepresentationLossOutput(
            total_loss=total_loss,
            organ_clip_loss=organ_clip_loss,
            report_clip_loss=report_clip_loss,
            organ_alignment_loss=organ_clip_loss,
            report_alignment_loss=report_clip_loss,
            segmentation_loss=segmentation_loss,
            diagnostic_loss=diagnostic_loss,
            patch_organ_presence_loss=patch_organ_presence_loss,
            organ_attention_loss=organ_attention_loss,
            lesion_global_loss=lesion_global_loss,
            lesion_organ_loss=lesion_organ_loss,
        )
        metrics = {
            **{f"organ_{key}": value for key, value in organ_metrics.items()},
            **{f"report_{key}": value for key, value in report_metrics.items()},
            **diagnostic_metrics,
            **lesion_global_metrics,
            **lesion_organ_metrics,
            "patch_organ_presence_accuracy": float(outputs.patch_organ_presence_accuracy),
            "organ_attention_accuracy": float(outputs.organ_attention_accuracy),
            "organ_attention_positive_accuracy": float(outputs.organ_attention_positive_accuracy),
            "organ_attention_negative_accuracy": float(outputs.organ_attention_negative_accuracy),
            "segmentation_dice": float(outputs.segmentation_dice),
            "segmentation_foreground_dice": float(outputs.segmentation_foreground_dice),
        }
        if self.config.alignment_type == "siglip":
            metrics.update(
                {
                    "organ_logit_scale": _scalar_metric(outputs.organ_logit_scale),
                    "organ_logit_bias": _scalar_metric(outputs.organ_logit_bias),
                    "report_logit_scale": _scalar_metric(outputs.report_logit_scale),
                    "report_logit_bias": _scalar_metric(outputs.report_logit_bias),
                }
            )
        return loss_output, metrics


def _masked_report_clip_loss(outputs: OrganSegOutput, batch: EncoderBatch) -> tuple[torch.Tensor, dict[str, float]]:
    report_mask = torch.tensor(
        [[bool(text)] for text in batch.report_texts],
        device=outputs.report_image_embeddings.device,
        dtype=torch.bool,
    )
    valid_report_count = _global_true_count(report_mask)
    if valid_report_count < 2:
        zero = outputs.report_image_embeddings.sum() * 0.0
        return zero, {"image_to_text_top1": 0.0, "text_to_image_top1": 0.0, "valid_count": float(valid_report_count)}
    report_texts = [[text] for text in batch.report_texts]
    loss, metrics = masked_organ_clip_loss(
        outputs.report_image_embeddings.unsqueeze(1),
        outputs.report_text_embeddings.unsqueeze(1),
        report_mask,
        report_texts,
        outputs.logit_scale,
    )
    metrics["valid_count"] = float(valid_report_count)
    return loss, metrics


def _global_true_count(mask: torch.Tensor) -> int:
    count = torch.tensor(float(mask.sum().item()), device=mask.device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return int(count.item())


def _masked_named_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    metric_name: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss, metrics = masked_binary_diagnostic_loss(logits, labels, mask)
    return loss, {metric_name: metrics["diagnostic_accuracy"]}


def _scalar_metric(value: torch.Tensor) -> float:
    return float(value.detach().float().reshape(()).item())
