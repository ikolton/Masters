"""Auxiliary lexical and semantic losses for Merlin ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


@dataclass(frozen=True)
class AuxiliaryLossOutput:
    total: torch.Tensor
    lexical: torch.Tensor
    semantic: torch.Tensor
    semantic_normality: torch.Tensor
    semantic_polarity: torch.Tensor
    semantic_family: torch.Tensor
    semantic_subtype: torch.Tensor
    lexical_count: int
    semantic_count: int

    def metrics(self) -> dict[str, float]:
        return {
            "aux_total_loss": float(self.total.detach().cpu().item()),
            "lexical_loss": float(self.lexical.detach().cpu().item()),
            "semantic_loss": float(self.semantic.detach().cpu().item()),
            "semantic_normality_loss": float(self.semantic_normality.detach().cpu().item()),
            "semantic_polarity_loss": float(self.semantic_polarity.detach().cpu().item()),
            "semantic_family_loss": float(self.semantic_family.detach().cpu().item()),
            "semantic_subtype_loss": float(self.semantic_subtype.detach().cpu().item()),
            "lexical_count": float(self.lexical_count),
            "semantic_count": float(self.semantic_count),
        }


class AuxiliaryDiagnosticLosses(nn.Module):
    def __init__(self, config: LossConfig, *, hidden_size: int, family_count: int, subtype_count: int) -> None:
        super().__init__()
        self.config = config
        self.family_count = int(family_count)
        self.subtype_count = int(subtype_count)
        self.lexical_head = nn.Linear(hidden_size, 1) if config.lexical_weight > 0 and config.lexical_mode == "auxiliary" else None
        self.normality_head = nn.Linear(hidden_size, 4) if config.semantic_weight > 0 else None
        self.polarity_head = nn.Linear(hidden_size, 3) if config.semantic_weight > 0 else None
        self.family_head = nn.Linear(hidden_size, family_count) if config.semantic_weight > 0 and family_count > 0 else None
        self.subtype_head = nn.Linear(hidden_size, subtype_count) if config.semantic_weight > 0 and subtype_count > 0 else None

    def forward(self, pooled_hidden: torch.Tensor, batch: dict[str, object]) -> AuxiliaryLossOutput:
        zero = pooled_hidden.sum() * 0.0
        lexical_loss = zero
        semantic_loss = zero
        normality_loss = zero
        polarity_loss = zero
        family_loss = zero
        subtype_loss = zero
        lexical_count = 0
        semantic_count = 0

        if self.lexical_head is not None:
            labels = _tensor(batch["lexical_label"], pooled_hidden.device).float()
            available = _bool_tensor(batch["lexical_available"], pooled_hidden.device)
            if bool(available.any().item()):
                logits = self.lexical_head(pooled_hidden[available]).squeeze(-1)
                lexical_loss = F.binary_cross_entropy_with_logits(logits, labels[available])
                lexical_count = int(available.sum().item())
            else:
                lexical_loss = self.lexical_head(pooled_hidden[:1]).sum() * 0.0

        if self.normality_head is not None and self.polarity_head is not None:
            available = _bool_tensor(batch["semantic_available"], pooled_hidden.device)
            weights = _tensor(batch["semantic_weight"], pooled_hidden.device).float()
            valid = available & weights.gt(0.0)
            if bool(valid.any().item()):
                semantic_count = int(valid.sum().item())
                normality_targets = _tensor(batch["semantic_normality"], pooled_hidden.device).long()
                polarity_targets = _tensor(batch["semantic_polarity"], pooled_hidden.device).long()
                normality_loss = _weighted_ce(self.normality_head(pooled_hidden[valid]), normality_targets[valid], weights[valid])
                polarity_loss = _weighted_ce(self.polarity_head(pooled_hidden[valid]), polarity_targets[valid], weights[valid])
                semantic_loss = (
                    float(self.config.normality_weight) * normality_loss
                    + float(self.config.polarity_weight) * polarity_loss
                )
                if self.config.semantic_variant in {"family", "family_subtype"} and self.family_head is not None:
                    family_targets = _tensor(batch["semantic_family_targets"], pooled_hidden.device).float()
                    family_allowed = _bool_tensor(batch["semantic_family_allowed"], pooled_hidden.device)
                    family_logits = self.family_head(pooled_hidden[valid])
                    family_loss = _masked_weighted_bce(family_logits, family_targets[valid], family_allowed[valid], weights[valid])
                    semantic_loss = semantic_loss + float(self.config.family_weight) * family_loss
                if self.config.semantic_variant == "family_subtype" and self.subtype_head is not None:
                    subtype_targets = _tensor(batch["semantic_subtype_targets"], pooled_hidden.device).float()
                    subtype_allowed = _bool_tensor(batch["semantic_subtype_allowed"], pooled_hidden.device)
                    subtype_logits = self.subtype_head(pooled_hidden[valid])
                    subtype_loss = _masked_weighted_bce(subtype_logits, subtype_targets[valid], subtype_allowed[valid], weights[valid])
                    semantic_loss = semantic_loss + float(self.config.subtype_weight) * subtype_loss
            else:
                semantic_loss = self._unused_semantic_anchor(pooled_hidden)

        total = float(self.config.lexical_weight) * lexical_loss + float(self.config.semantic_weight) * semantic_loss
        return AuxiliaryLossOutput(
            total=total,
            lexical=lexical_loss,
            semantic=semantic_loss,
            semantic_normality=normality_loss,
            semantic_polarity=polarity_loss,
            semantic_family=family_loss,
            semantic_subtype=subtype_loss,
            lexical_count=lexical_count,
            semantic_count=semantic_count,
        )

    def _unused_semantic_anchor(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        zero = pooled_hidden.sum() * 0.0
        for head in (self.normality_head, self.polarity_head, self.family_head, self.subtype_head):
            if head is not None:
                zero = zero + head(pooled_hidden[:1]).sum() * 0.0
        return zero


def _tensor(value: object, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, torch.Tensor) for item in value):
        tensors = [item.to(device) for item in value]
        if tensors[0].ndim == 0:
            return torch.stack(tensors, dim=0)
        # MONAI's default collation transposes fixed-length vector fields into
        # vocab-length lists of batch tensors. Restore [batch, vocab].
        return torch.stack(tensors, dim=1)
    return torch.as_tensor(value, device=device)


def _bool_tensor(value: object, device: torch.device) -> torch.Tensor:
    return _tensor(value, device).bool()


def _weighted_ce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1.0e-8)


def _masked_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    allowed_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0
    allowed = allowed_mask.float()
    element_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    row_loss = (element_loss * allowed).sum(dim=1) / allowed.sum(dim=1).clamp_min(1.0)
    return (row_loss * weights).sum() / weights.sum().clamp_min(1.0e-8)
