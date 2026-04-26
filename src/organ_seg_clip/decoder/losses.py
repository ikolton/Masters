"""Decoder diagnostic losses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.schemas import DecoderDiagnosticLossConfig


@dataclass(frozen=True)
class DiagnosticLossOutput:
    loss: torch.Tensor
    pathology_positive_loss: torch.Tensor
    pathology_negative_loss: torch.Tensor
    normal_negative_loss: torch.Tensor
    sample_count: int
    positive_sample_count: int
    negative_sample_count: int

    def to_metrics(self) -> dict[str, float]:
        return {
            "diagnostic_loss": float(self.loss.detach().cpu().item()),
            "diagnostic_pathology_positive_loss": float(self.pathology_positive_loss.detach().cpu().item()),
            "diagnostic_pathology_negative_loss": float(self.pathology_negative_loss.detach().cpu().item()),
            "diagnostic_normal_negative_loss": float(self.normal_negative_loss.detach().cpu().item()),
            "diagnostic_sample_count": float(self.sample_count),
            "diagnostic_positive_sample_count": float(self.positive_sample_count),
            "diagnostic_negative_sample_count": float(self.negative_sample_count),
        }


class BinaryDiagnosticLoss(nn.Module):
    """Binary concept loss based on lesion-present/lesion-absent CSV labels."""

    def __init__(self, config: DecoderDiagnosticLossConfig, tokenizer: Any) -> None:
        super().__init__()
        self.config = config
        self.pathology_concepts = tuple(config.pathology_words)
        self.normal_concepts = tuple(config.normal_words)
        self.pathology_token_ids = _concept_token_ids(tokenizer, self.pathology_concepts)
        self.normal_token_ids = _concept_token_ids(tokenizer, self.normal_concepts)

    def forward(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lesion_labels: torch.Tensor,
        lesion_mask: torch.Tensor,
        small_bowel_mask: torch.Tensor,
        target_texts: Sequence[str],
    ) -> DiagnosticLossOutput:
        zero = logits.sum() * 0.0
        if not self.config.enabled:
            return DiagnosticLossOutput(zero, zero, zero, zero, 0, 0, 0)
        prediction_logits = logits[:, :-1, :]
        prediction_labels = labels[:, 1:]
        target_mask = prediction_labels.ne(-100)
        positive_losses: list[torch.Tensor] = []
        negative_losses: list[torch.Tensor] = []
        normal_losses: list[torch.Tensor] = []
        positive_count = 0
        negative_count = 0
        for row_index in range(logits.shape[0]):
            if not bool(lesion_mask[row_index].item()):
                continue
            row_logits = prediction_logits[row_index][target_mask[row_index]]
            if row_logits.numel() == 0:
                continue
            lesion_positive = bool(lesion_labels[row_index].item() > 0.5)
            if lesion_positive:
                positive_count += 1
                any_pathology_probability = _any_token_probability(row_logits, self.pathology_token_ids, self.config.epsilon)
                any_normal_probability = _any_token_probability(row_logits, self.normal_token_ids, self.config.epsilon)
                pos_loss = -torch.log(any_pathology_probability.clamp_min(float(self.config.epsilon)))
                normal_loss = -torch.log((1.0 - any_normal_probability).clamp_min(float(self.config.epsilon)))
                positive_losses.append(pos_loss * float(self.config.positive_pathology_weight))
                normal_losses.append(normal_loss * float(self.config.positive_normal_penalty_weight))
            else:
                negative_count += 1
                concept_ids = _filter_absent_concepts(
                    self.pathology_concepts,
                    self.pathology_token_ids,
                    target_texts[row_index],
                )
                if concept_ids:
                    weight = float(self.config.small_bowel_duodenum_negative_weight) if bool(small_bowel_mask[row_index].item()) else float(self.config.negative_pathology_weight)
                    any_pathology_probability = _any_token_probability(row_logits, concept_ids, self.config.epsilon)
                    negative_losses.append(-torch.log((1.0 - any_pathology_probability).clamp_min(float(self.config.epsilon))) * weight)
        pos = _mean_or_zero(positive_losses, zero)
        neg = _mean_or_zero(negative_losses, zero)
        normal = _mean_or_zero(normal_losses, zero)
        loss = (pos + neg + normal) * float(self.config.weight)
        return DiagnosticLossOutput(
            loss=loss,
            pathology_positive_loss=pos,
            pathology_negative_loss=neg,
            normal_negative_loss=normal,
            sample_count=positive_count + negative_count,
            positive_sample_count=positive_count,
            negative_sample_count=negative_count,
        )


def _concept_token_ids(tokenizer: Any, concepts: Sequence[str]) -> list[tuple[str, tuple[int, ...]]]:
    encoded: list[tuple[str, tuple[int, ...]]] = []
    for concept in concepts:
        ids = tuple(int(value) for value in tokenizer(str(concept), add_special_tokens=False)["input_ids"])
        if ids:
            encoded.append((str(concept), ids))
    return encoded


def _any_token_probability(
    logits: torch.Tensor,
    concept_ids: Sequence[tuple[str, tuple[int, ...]]],
    epsilon: float,
) -> torch.Tensor:
    if not concept_ids:
        return logits.sum() * 0.0
    probs = F.softmax(logits.float(), dim=-1)
    token_ids = sorted({token_id for _, ids in concept_ids for token_id in ids})
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    mass_by_position = probs.index_select(dim=-1, index=token_tensor).sum(dim=-1).clamp(min=0.0, max=1.0 - float(epsilon))
    no_concept_probability = torch.prod(1.0 - mass_by_position)
    return (1.0 - no_concept_probability).clamp(min=float(epsilon), max=1.0 - float(epsilon))


def _filter_absent_concepts(
    concepts: Sequence[str],
    encoded: Sequence[tuple[str, tuple[int, ...]]],
    target_text: str,
) -> list[tuple[str, tuple[int, ...]]]:
    normalized = _normalize_text(target_text)
    present = {
        concept
        for concept in concepts
        if re.search(rf"(?<![a-z0-9]){re.escape(_normalize_text(concept))}(?![a-z0-9])", normalized)
    }
    return [(concept, ids) for concept, ids in encoded if concept not in present]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _mean_or_zero(values: Sequence[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    if not values:
        return zero
    return torch.stack(list(values)).mean()
