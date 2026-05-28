"""Semantic auxiliary losses for decoder training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.schemas import DecoderSemanticLossConfig


@dataclass(frozen=True)
class SemanticDiagnosticLossOutput:
    loss: torch.Tensor
    raw_loss: torch.Tensor
    normality_loss: torch.Tensor
    polarity_loss: torch.Tensor
    family_loss: torch.Tensor
    subtype_loss: torch.Tensor
    primary_loss: torch.Tensor
    secondary_loss: torch.Tensor
    sample_count: int
    provisional_sample_count: int

    def to_metrics(self) -> dict[str, float]:
        return {
            "semantic_diagnostic_loss": float(self.raw_loss.detach().cpu().item()),
            "semantic_diagnostic_loss_weighted": float(self.loss.detach().cpu().item()),
            "semantic_normality_loss": float(self.normality_loss.detach().cpu().item()),
            "semantic_polarity_loss": float(self.polarity_loss.detach().cpu().item()),
            "semantic_family_loss": float(self.family_loss.detach().cpu().item()),
            "semantic_subtype_loss": float(self.subtype_loss.detach().cpu().item()),
            "semantic_primary_loss": float(self.primary_loss.detach().cpu().item()),
            "semantic_secondary_loss": float(self.secondary_loss.detach().cpu().item()),
            "semantic_sample_count": float(self.sample_count),
            "semantic_provisional_sample_count": float(self.provisional_sample_count),
        }


class SemanticDiagnosticLoss(nn.Module):
    def __init__(self, config: DecoderSemanticLossConfig, *, hidden_size: int, subtype_count: int, family_count: int) -> None:
        super().__init__()
        self.config = config
        self.subtype_count = int(subtype_count)
        self.family_count = int(family_count)
        enabled = bool(self.config.enabled)
        self.normality_head = nn.Linear(hidden_size, 4) if enabled else None
        self.polarity_head = nn.Linear(hidden_size, 3) if enabled else None
        self.family_head = nn.Linear(hidden_size, self.family_count) if enabled and self.family_count > 0 else None
        self.subtype_head = (
            nn.Linear(hidden_size, self.subtype_count)
            if enabled and self.subtype_count > 0 and self.config.variant == "family_subtype"
            else None
        )
        self.primary_head = (
            nn.Linear(hidden_size, self.subtype_count)
            if enabled and self.subtype_count > 0 and self.config.variant == "primary_secondary"
            else None
        )
        self.secondary_head = (
            nn.Linear(hidden_size, self.subtype_count)
            if enabled and self.subtype_count > 0 and self.config.variant == "primary_secondary"
            else None
        )

    def forward(
        self,
        *,
        pooled_hidden: torch.Tensor,
        semantic_available: torch.Tensor,
        semantic_weights: torch.Tensor,
        semantic_statuses: list[str],
        semantic_normality_targets: torch.Tensor,
        semantic_polarity_targets: torch.Tensor,
        semantic_primary_subtype_targets: torch.Tensor,
        semantic_subtype_targets: torch.Tensor,
        semantic_secondary_subtype_targets: torch.Tensor,
        semantic_allowed_subtype_mask: torch.Tensor,
        semantic_family_targets: torch.Tensor,
        semantic_allowed_family_mask: torch.Tensor,
    ) -> SemanticDiagnosticLossOutput:
        zero = pooled_hidden.sum() * 0.0
        if not self.config.enabled:
            return SemanticDiagnosticLossOutput(zero, zero, zero, zero, zero, zero, zero, zero, 0, 0)
        valid_mask = semantic_available & semantic_weights.gt(0.0)
        if not bool(valid_mask.any().item()):
            zero = zero + self._unused_parameter_anchor(pooled_hidden)
            return SemanticDiagnosticLossOutput(zero, zero, zero, zero, zero, zero, zero, zero, 0, 0)
        weights = semantic_weights[valid_mask]
        assert self.normality_head is not None and self.polarity_head is not None
        norm_loss = _weighted_cross_entropy(
            self.normality_head(pooled_hidden[valid_mask]),
            semantic_normality_targets[valid_mask],
            weights,
        )
        pol_loss = _weighted_cross_entropy(
            self.polarity_head(pooled_hidden[valid_mask]),
            semantic_polarity_targets[valid_mask],
            weights,
        )
        subtype_loss = zero
        family_loss = zero
        primary_loss = zero
        secondary_loss = zero
        if self.family_count > 0:
            assert self.family_head is not None
            family_logits = self.family_head(pooled_hidden[valid_mask])
            family_targets = semantic_family_targets[valid_mask]
            family_row_mask = family_targets.sum(dim=1).gt(0.0)
            if bool(family_row_mask.any().item()):
                family_loss = _masked_weighted_bce(
                    family_logits[family_row_mask],
                    family_targets[family_row_mask],
                    semantic_allowed_family_mask[valid_mask][family_row_mask],
                    weights[family_row_mask],
                )
            else:
                family_loss = family_logits.sum() * 0.0
        if self.subtype_count > 0:
            allowed_mask = semantic_allowed_subtype_mask[valid_mask]
            if self.config.variant == "family_subtype":
                assert self.subtype_head is not None
                subtype_logits = self.subtype_head(pooled_hidden[valid_mask])
                subtype_targets = semantic_subtype_targets[valid_mask]
                subtype_row_mask = subtype_targets.sum(dim=1).gt(0.0)
                if bool(subtype_row_mask.any().item()):
                    subtype_loss = _masked_weighted_bce(
                        subtype_logits[subtype_row_mask],
                        subtype_targets[subtype_row_mask],
                        allowed_mask[subtype_row_mask],
                        weights[subtype_row_mask],
                    )
                else:
                    subtype_loss = subtype_logits.sum() * 0.0
            elif self.config.variant == "primary_secondary":
                assert self.primary_head is not None and self.secondary_head is not None
                primary_logits = self.primary_head(pooled_hidden[valid_mask])
                primary_loss = _masked_weighted_primary_ce(
                    primary_logits,
                    semantic_primary_subtype_targets[valid_mask],
                    allowed_mask,
                    weights,
                )
                secondary_loss = _masked_weighted_bce(
                    self.secondary_head(pooled_hidden[valid_mask]),
                    semantic_secondary_subtype_targets[valid_mask],
                    allowed_mask,
                    weights,
                )
        raw_loss = (
            float(self.config.normality_weight) * norm_loss
            + float(self.config.polarity_weight) * pol_loss
            + float(self.config.family_weight) * family_loss
        )
        if self.config.variant in {"minimal", "family_subtype"}:
            raw_loss = raw_loss + float(self.config.subtype_weight) * subtype_loss
        else:
            raw_loss = raw_loss + float(self.config.primary_weight) * primary_loss + float(self.config.secondary_weight) * secondary_loss
        loss = raw_loss * float(self.config.weight)
        statuses = [semantic_statuses[index] for index, keep in enumerate(valid_mask.tolist()) if keep]
        provisional_count = sum(1 for status in statuses if status == "accepted_provisional")
        return SemanticDiagnosticLossOutput(
            loss=loss,
            raw_loss=raw_loss,
            normality_loss=norm_loss,
            polarity_loss=pol_loss,
            family_loss=family_loss,
            subtype_loss=subtype_loss,
            primary_loss=primary_loss,
            secondary_loss=secondary_loss,
            sample_count=int(valid_mask.sum().item()),
            provisional_sample_count=int(provisional_count),
        )

    def _unused_parameter_anchor(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        """Touch active auxiliary heads with a zero multiplier for DDP edge batches."""
        probe = pooled_hidden[:1]
        zero = pooled_hidden.sum() * 0.0
        for head in (self.normality_head, self.polarity_head, self.family_head, self.subtype_head, self.primary_head, self.secondary_head):
            if head is not None:
                zero = zero + head(probe).sum() * 0.0
        return zero


def _weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1.0e-8)


def _masked_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    allowed_mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    element_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    allowed = allowed_mask.float()
    row_denominator = allowed.sum(dim=1).clamp_min(1.0)
    row_loss = (element_loss * allowed).sum(dim=1) / row_denominator
    return (row_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1.0e-8)


def _masked_weighted_primary_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    allowed_mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    masked_logits = logits.masked_fill(~allowed_mask, -1.0e9)
    losses = F.cross_entropy(masked_logits, targets, reduction="none")
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1.0e-8)
