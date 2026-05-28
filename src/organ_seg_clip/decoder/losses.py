"""Decoder diagnostic losses."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from ..config.schemas import DecoderDiagnosticLossConfig


@dataclass(frozen=True)
class DiagnosticLossOutput:
    loss: torch.Tensor
    raw_loss: torch.Tensor
    pathology_positive_loss: torch.Tensor
    pathology_negative_loss: torch.Tensor
    normal_negative_loss: torch.Tensor
    sample_count: int
    positive_sample_count: int
    negative_sample_count: int
    positive_concept_count: int = 0
    negative_concept_count: int = 0

    def to_metrics(self) -> dict[str, float]:
        return {
            "diagnostic_loss": float(self.raw_loss.detach().cpu().item()),
            "diagnostic_loss_weighted": float(self.loss.detach().cpu().item()),
            "diagnostic_pathology_positive_loss": float(self.pathology_positive_loss.detach().cpu().item()),
            "diagnostic_pathology_negative_loss": float(self.pathology_negative_loss.detach().cpu().item()),
            "diagnostic_normal_negative_loss": float(self.normal_negative_loss.detach().cpu().item()),
            "diagnostic_sample_count": float(self.sample_count),
            "diagnostic_positive_sample_count": float(self.positive_sample_count),
            "diagnostic_negative_sample_count": float(self.negative_sample_count),
            "diagnostic_positive_concept_count": float(self.positive_concept_count),
            "diagnostic_negative_concept_count": float(self.negative_concept_count),
        }


class BinaryDiagnosticLoss(nn.Module):
    """Binary concept loss based on lesion-present/lesion-absent CSV labels."""

    def __init__(self, config: DecoderDiagnosticLossConfig, tokenizer: Any) -> None:
        super().__init__()
        self.config = config
        self.pathology_concepts = tuple(config.pathology_words)
        self.normal_concepts = tuple(config.normal_words)
        self.lexical_targets = _load_lexical_targets(config.lexical_target_cache) if config.variant == "sample_specific_lexical" else {}
        self.concept_targets = (
            _load_concept_lexical_targets(config.lexical_target_cache, tokenizer)
            if config.variant == "concept_specific_lexical"
            else {}
        )
        self.pathology_token_ids = _concept_token_ids(tokenizer, self.pathology_concepts)
        self.normal_token_ids = _concept_token_ids(tokenizer, self.normal_concepts)
        self.register_buffer(
            "_pathology_token_index",
            _concept_token_index_tensor(self.pathology_token_ids),
            persistent=False,
        )
        self.register_buffer(
            "_normal_token_index",
            _concept_token_index_tensor(self.normal_token_ids),
            persistent=False,
        )

    def forward(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lesion_labels: torch.Tensor,
        lesion_mask: torch.Tensor,
        small_bowel_mask: torch.Tensor,
        target_texts: Sequence[str],
        organ_names: Sequence[str] | None = None,
    ) -> DiagnosticLossOutput:
        zero = logits.sum() * 0.0
        if not self.config.enabled:
            return DiagnosticLossOutput(zero, zero, zero, zero, zero, 0, 0, 0)
        if self.config.variant == "sample_specific_lexical":
            return self._forward_sample_specific_lexical(
                logits=logits,
                labels=labels,
                target_texts=target_texts,
                organ_names=organ_names,
            )
        if self.config.variant == "concept_specific_lexical":
            return self._forward_concept_specific_lexical(
                logits=logits,
                labels=labels,
                target_texts=target_texts,
                organ_names=organ_names,
            )
        prediction_logits = logits[:, :-1, :]
        prediction_labels = labels[:, 1:]
        target_mask = prediction_labels.ne(-100)
        positive_losses: list[torch.Tensor] = []
        negative_losses: list[torch.Tensor] = []
        normal_losses: list[torch.Tensor] = []
        positive_count = 0
        negative_count = 0
        normalized_targets = [_normalize_text(target) for target in target_texts]
        absent_pathology_token_index_cache: dict[str, torch.Tensor] = {}
        for row_index in range(logits.shape[0]):
            if not bool(lesion_mask[row_index].item()):
                continue
            row_logits = prediction_logits[row_index][target_mask[row_index]]
            if row_logits.numel() == 0:
                continue
            lesion_positive = bool(lesion_labels[row_index].item() > 0.5)
            if lesion_positive:
                positive_count += 1
                any_pathology_probability = _any_token_probability(
                    row_logits,
                    self._pathology_token_index,
                    self.config.epsilon,
                )
                any_normal_probability = _any_token_probability(
                    row_logits,
                    self._normal_token_index,
                    self.config.epsilon,
                )
                pos_loss = -torch.log(any_pathology_probability.clamp_min(float(self.config.epsilon)))
                normal_loss = -torch.log((1.0 - any_normal_probability).clamp_min(float(self.config.epsilon)))
                positive_losses.append(pos_loss * float(self.config.positive_pathology_weight))
                normal_losses.append(normal_loss * float(self.config.positive_normal_penalty_weight))
            else:
                negative_count += 1
                normalized_target = normalized_targets[row_index]
                token_index = absent_pathology_token_index_cache.get(normalized_target)
                if token_index is None:
                    concept_ids = _filter_absent_concepts(
                        self.pathology_concepts,
                        self.pathology_token_ids,
                        normalized_target,
                        normalized=True,
                    )
                    token_index = _concept_token_index_tensor(concept_ids)
                    absent_pathology_token_index_cache[normalized_target] = token_index
                token_index = token_index.to(row_logits.device)
                if token_index.numel() > 0:
                    weight = float(self.config.small_bowel_duodenum_negative_weight) if bool(small_bowel_mask[row_index].item()) else float(self.config.negative_pathology_weight)
                    any_pathology_probability = _any_token_probability(
                        row_logits,
                        token_index,
                        self.config.epsilon,
                    )
                    negative_losses.append(-torch.log((1.0 - any_pathology_probability).clamp_min(float(self.config.epsilon))) * weight)
        pos = _mean_or_zero(positive_losses, zero)
        neg = _mean_or_zero(negative_losses, zero)
        normal = _mean_or_zero(normal_losses, zero)
        raw_loss = pos + neg + normal
        loss = raw_loss * float(self.config.weight)
        return DiagnosticLossOutput(
            loss=loss,
            raw_loss=raw_loss,
            pathology_positive_loss=pos,
            pathology_negative_loss=neg,
            normal_negative_loss=normal,
            sample_count=positive_count + negative_count,
            positive_sample_count=positive_count,
            negative_sample_count=negative_count,
        )

    def _forward_concept_specific_lexical(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_texts: Sequence[str],
        organ_names: Sequence[str] | None,
    ) -> DiagnosticLossOutput:
        zero = logits.sum() * 0.0
        if not self.concept_targets or organ_names is None:
            return DiagnosticLossOutput(zero, zero, zero, zero, zero, 0, 0, 0)
        prediction_logits = logits[:, :-1, :]
        prediction_labels = labels[:, 1:]
        target_mask = prediction_labels.ne(-100)
        sample_losses: list[torch.Tensor] = []
        positive_losses: list[torch.Tensor] = []
        negative_losses: list[torch.Tensor] = []
        positive_sample_count = 0
        negative_sample_count = 0
        positive_concept_count = 0
        negative_concept_count = 0

        for row_index in range(logits.shape[0]):
            key = (_normalize_key(organ_names[row_index]), _normalize_key(target_texts[row_index]))
            target = self.concept_targets.get(key)
            if target is None:
                continue
            row_logits = prediction_logits[row_index][target_mask[row_index]]
            if row_logits.numel() == 0:
                continue
            sample_weight = float(target.get("sample_weight", 0.0))
            if sample_weight <= 0.0:
                continue

            row_positive = self._weighted_positive_concept_loss(row_logits, target.get("positive_concepts", []), zero)
            row_negative = self._weighted_negative_concept_loss(row_logits, target.get("negative_concepts", []), zero)
            if row_positive[1] == 0 and row_negative[1] == 0:
                continue
            if row_positive[1] > 0:
                positive_losses.append(row_positive[0])
                positive_sample_count += 1
                positive_concept_count += row_positive[1]
            if row_negative[1] > 0:
                negative_losses.append(row_negative[0])
                negative_sample_count += 1
                negative_concept_count += row_negative[1]
            sample_losses.append(
                sample_weight
                * (
                    float(self.config.positive_pathology_weight) * row_positive[0]
                    + float(self.config.negative_pathology_weight) * row_negative[0]
                )
            )

        pos = _mean_or_zero(positive_losses, zero)
        neg = _mean_or_zero(negative_losses, zero)
        raw_loss = _mean_or_zero(sample_losses, zero)
        loss = raw_loss * float(self.config.weight)
        return DiagnosticLossOutput(
            loss=loss,
            raw_loss=raw_loss,
            pathology_positive_loss=pos,
            pathology_negative_loss=neg,
            normal_negative_loss=zero,
            sample_count=positive_sample_count + negative_sample_count,
            positive_sample_count=positive_sample_count,
            negative_sample_count=negative_sample_count,
            positive_concept_count=positive_concept_count,
            negative_concept_count=negative_concept_count,
        )

    def _weighted_positive_concept_loss(
        self,
        row_logits: torch.Tensor,
        concepts: object,
        zero: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        losses: list[torch.Tensor] = []
        weights: list[float] = []
        for concept in _concept_list(concepts):
            token_index = _token_index_from_ids(concept.get("token_ids", ()), row_logits.device)
            weight = float(concept.get("weight", 0.0) or 0.0)
            if token_index.numel() == 0 or weight <= 0.0:
                continue
            probability = _any_token_probability(row_logits, token_index, self.config.epsilon)
            losses.append(-torch.log(probability.clamp_min(float(self.config.epsilon))) * weight)
            weights.append(weight)
        if not losses:
            return zero, 0
        return torch.stack(losses).sum() / max(sum(weights), float(self.config.epsilon)), len(losses)

    def _weighted_negative_concept_loss(
        self,
        row_logits: torch.Tensor,
        concepts: object,
        zero: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        losses: list[torch.Tensor] = []
        weights: list[float] = []
        for concept in _concept_list(concepts):
            token_index = _token_index_from_ids(concept.get("token_ids", ()), row_logits.device)
            weight = float(concept.get("weight", 0.0) or 0.0)
            if token_index.numel() == 0 or weight <= 0.0:
                continue
            losses.append(_smoothmax_token_probability(row_logits, token_index, self.config.epsilon, self.config.negative_temperature) * weight)
            weights.append(weight)
        if not losses:
            return zero, 0
        return torch.stack(losses).sum() / max(sum(weights), float(self.config.epsilon)), len(losses)

    def _forward_sample_specific_lexical(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        target_texts: Sequence[str],
        organ_names: Sequence[str] | None,
    ) -> DiagnosticLossOutput:
        zero = logits.sum() * 0.0
        if not self.lexical_targets or organ_names is None:
            return DiagnosticLossOutput(zero, zero, zero, zero, zero, 0, 0, 0)
        prediction_logits = logits[:, :-1, :]
        prediction_labels = labels[:, 1:]
        target_mask = prediction_labels.ne(-100)
        positive_losses: list[torch.Tensor] = []
        negative_losses: list[torch.Tensor] = []
        positive_count = 0
        negative_count = 0
        for row_index in range(logits.shape[0]):
            key = (_normalize_key(organ_names[row_index]), _normalize_key(target_texts[row_index]))
            target = self.lexical_targets.get(key)
            if target is None:
                continue
            row_logits = prediction_logits[row_index][target_mask[row_index]]
            if row_logits.numel() == 0:
                continue
            sample_weight = float(target.get("sample_weight", 0.0))
            if sample_weight <= 0.0:
                continue
            positive_token_index = _token_index_from_ids(target.get("positive_token_ids", ()), row_logits.device)
            negative_token_index = _token_index_from_ids(target.get("negative_token_ids", ()), row_logits.device)
            if positive_token_index.numel() > 0:
                positive_count += 1
                probability = _any_token_probability(row_logits, positive_token_index, self.config.epsilon)
                positive_losses.append(
                    -torch.log(probability.clamp_min(float(self.config.epsilon)))
                    * float(self.config.positive_pathology_weight)
                    * sample_weight
                )
            if negative_token_index.numel() > 0:
                negative_count += 1
                probability = _any_token_probability(row_logits, negative_token_index, self.config.epsilon)
                negative_losses.append(
                    -torch.log((1.0 - probability).clamp_min(float(self.config.epsilon)))
                    * float(self.config.negative_pathology_weight)
                    * sample_weight
                )
        pos = _mean_or_zero(positive_losses, zero)
        neg = _mean_or_zero(negative_losses, zero)
        raw_loss = pos + neg
        loss = raw_loss * float(self.config.weight)
        return DiagnosticLossOutput(
            loss=loss,
            raw_loss=raw_loss,
            pathology_positive_loss=pos,
            pathology_negative_loss=neg,
            normal_negative_loss=zero,
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


def _concept_token_index_tensor(concept_ids: Sequence[tuple[str, tuple[int, ...]]]) -> torch.Tensor:
    token_ids = sorted({token_id for _, ids in concept_ids for token_id in ids})
    if not token_ids:
        return torch.empty((0,), dtype=torch.long)
    return torch.tensor(token_ids, dtype=torch.long)


def _any_token_probability(
    logits: torch.Tensor,
    token_index: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if token_index.numel() == 0:
        return logits.sum() * 0.0
    log_z = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    log_probs = logits.float()[:, token_index] - log_z
    mass_by_position = log_probs.exp().sum(dim=-1).clamp(min=0.0, max=1.0 - float(epsilon))
    no_concept_probability = torch.prod(1.0 - mass_by_position)
    return (1.0 - no_concept_probability).clamp(min=float(epsilon), max=1.0 - float(epsilon))


def _smoothmax_token_probability(
    logits: torch.Tensor,
    token_index: torch.Tensor,
    epsilon: float,
    temperature: float,
) -> torch.Tensor:
    if token_index.numel() == 0:
        return logits.sum() * 0.0
    log_z = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
    log_probs = logits.float()[:, token_index] - log_z
    mass_by_position = log_probs.exp().sum(dim=-1).clamp(min=float(epsilon), max=1.0)
    tau = max(float(temperature), float(epsilon))
    return torch.logsumexp(tau * torch.log(mass_by_position), dim=0) / tau


def _filter_absent_concepts(
    concepts: Sequence[str],
    encoded: Sequence[tuple[str, tuple[int, ...]]],
    target_text: str,
    *,
    normalized: bool = False,
) -> list[tuple[str, tuple[int, ...]]]:
    normalized_target = target_text if normalized else _normalize_text(target_text)
    present = {
        concept
        for concept in concepts
        if re.search(rf"(?<![a-z0-9]){re.escape(_normalize_text(concept))}(?![a-z0-9])", normalized_target)
    }
    return [(concept, ids) for concept, ids in encoded if concept not in present]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _normalize_key(value: str) -> str:
    return _normalize_text(value)


def _token_index_from_ids(token_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    if not token_ids:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.tensor(sorted({int(value) for value in token_ids}), dtype=torch.long, device=device)


def _load_lexical_targets(path_value: str) -> dict[tuple[str, str], dict[str, object]]:
    path = Path(path_value).expanduser()
    if not str(path_value).strip():
        raise ValueError("diagnostic_loss.lexical_target_cache is required for sample_specific_lexical variant.")
    if not path.is_file():
        raise FileNotFoundError(f"Lexical diagnostic target cache not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    targets: dict[tuple[str, str], dict[str, object]] = {}
    for row in payload.get("rows", []):
        key = row.get("key")
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            continue
        targets[(_normalize_key(str(key[0])), _normalize_key(str(key[1])))] = {
            "positive_token_ids": tuple(int(value) for value in row.get("positive_token_ids", [])),
            "negative_token_ids": tuple(int(value) for value in row.get("negative_token_ids", [])),
            "sample_weight": float(row.get("sample_weight", 0.0) or 0.0),
        }
    return targets


def _load_concept_lexical_targets(path_value: str, tokenizer: Any | None = None) -> dict[tuple[str, str], dict[str, object]]:
    path = Path(path_value).expanduser()
    if not str(path_value).strip():
        raise ValueError("diagnostic_loss.lexical_target_cache is required for concept_specific_lexical variant.")
    if not path.is_file():
        raise FileNotFoundError(f"Concept lexical diagnostic target cache not found: {path}")
    if path.suffix == ".jsonl":
        if tokenizer is None:
            raise ValueError("A tokenizer is required when loading concept lexical targets from JSONL.")
        return _load_concept_lexical_targets_jsonl(path, tokenizer)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Concept lexical diagnostic target cache must be a dict payload: {path}")
    targets: dict[tuple[str, str], dict[str, object]] = {}
    for row in payload.get("rows", []):
        key = row.get("key")
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            continue
        targets[(_normalize_key(str(key[0])), _normalize_key(str(key[1])))] = {
            "positive_concepts": _normalized_concepts(row.get("positive_concepts", [])),
            "negative_concepts": _normalized_concepts(row.get("negative_concepts", [])),
            "sample_weight": float(row.get("sample_weight", 0.0) or 0.0),
            "review_required": bool(row.get("review_required", False)),
        }
    return targets


def _load_concept_lexical_targets_jsonl(path: Path, tokenizer: Any) -> dict[tuple[str, str], dict[str, object]]:
    targets: dict[tuple[str, str], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            organ = str(row.get("organ", "")).strip()
            text = str(row.get("normalized_text") or row.get("raw_text") or "").strip()
            if not organ or not text:
                continue
            targets[(_normalize_key(organ), _normalize_key(text))] = _tokenized_concept_target_row(row, tokenizer)
    return targets


def _tokenized_concept_target_row(row: dict[str, object], tokenizer: Any) -> dict[str, object]:
    positive = [_tokenized_concept_target(concept, tokenizer) for concept in _concept_list(row.get("positive_concepts", []))]
    positive = [concept for concept in positive if concept["token_ids"]]
    positive_union = {token_id for concept in positive for token_id in concept["token_ids"]}
    negative = []
    mixed_normality = str(row.get("normality", "")).strip() == "mixed"
    for concept in _concept_list(row.get("negative_concepts", [])):
        tokenized = _tokenized_concept_target(concept, tokenizer)
        tokenized["token_ids"] = tuple(token_id for token_id in tokenized["token_ids"] if token_id not in positive_union)
        if mixed_normality and tokenized["source_label"] == "normal_wording":
            tokenized["weight"] = min(float(tokenized["weight"]), 0.05)
        if tokenized["token_ids"] and float(tokenized["weight"]) > 0.0:
            negative.append(tokenized)
    return {
        "positive_concepts": tuple(positive),
        "negative_concepts": tuple(negative),
        "sample_weight": float(row.get("sample_weight", 0.0) or 0.0),
        "review_required": bool(row.get("review_required", False)),
    }


def _tokenized_concept_target(concept: dict[str, object], tokenizer: Any) -> dict[str, object]:
    phrases = _dedupe_text(str(value) for value in concept.get("phrases", []) if str(value).strip())
    return {
        "source_label": str(concept.get("source_label", "")),
        "label_type": str(concept.get("label_type", "")),
        "weight": float(concept.get("weight", 0.0) or 0.0),
        "token_ids": tuple(_phrase_token_ids(tokenizer, phrases)),
    }


def _phrase_token_ids(tokenizer: Any, phrases: Sequence[str]) -> tuple[int, ...]:
    ids: set[int] = set()
    for phrase in phrases:
        encoded = tokenizer(str(phrase), add_special_tokens=False)
        for token_id in encoded.get("input_ids", []):
            ids.add(int(token_id))
    return tuple(sorted(ids))


def _dedupe_text(values: Sequence[str] | Any) -> tuple[str, ...]:
    out = []
    seen = set()
    for value in values:
        text = str(value).strip()
        key = _normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return tuple(out)


def _normalized_concepts(value: object) -> tuple[dict[str, object], ...]:
    out = []
    for concept in _concept_list(value):
        token_ids = tuple(sorted({int(token_id) for token_id in concept.get("token_ids", [])}))
        if not token_ids:
            continue
        out.append(
            {
                "source_label": str(concept.get("source_label", "")),
                "label_type": str(concept.get("label_type", "")),
                "weight": float(concept.get("weight", 0.0) or 0.0),
                "token_ids": token_ids,
            }
        )
    return tuple(out)


def _concept_list(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(concept for concept in value if isinstance(concept, dict))


def _mean_or_zero(values: Sequence[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    if not values:
        return zero
    return torch.stack(list(values)).mean()
