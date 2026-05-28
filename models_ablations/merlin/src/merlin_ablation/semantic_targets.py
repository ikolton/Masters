"""Semantic target lookup for Merlin ablations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


NORMALITY_TO_INDEX = {
    "normal": 0,
    "abnormal": 1,
    "absent_postop": 2,
    "mixed": 3,
}

POLARITY_TO_INDEX = {
    "positive": 0,
    "negative": 1,
    "mixed": 2,
}


@dataclass(frozen=True)
class SemanticTarget:
    normality_index: int
    polarity_index: int
    confidence_weight: float
    sample_weight: float
    family_targets: tuple[float, ...]
    subtype_targets: tuple[float, ...]
    family_allowed: tuple[bool, ...]
    subtype_allowed: tuple[bool, ...]
    review_required: bool


@dataclass(frozen=True)
class SemanticTargetSpec:
    family_vocab: tuple[str, ...]
    subtype_vocab: tuple[str, ...]
    organ_to_family_vocab: Mapping[str, tuple[str, ...]]
    organ_to_subtype_vocab: Mapping[str, tuple[str, ...]]


class SemanticTargetLookup:
    def __init__(self, targets: Mapping[tuple[str, str], SemanticTarget], spec: SemanticTargetSpec) -> None:
        self._targets = dict(targets)
        self.spec = spec

    def get(self, organ: str, raw_text: str) -> SemanticTarget | None:
        return self._targets.get((_key(organ), _key(raw_text)))

    @property
    def size(self) -> int:
        return len(self._targets)


def load_semantic_targets(
    *,
    targets_jsonl: Path | None,
    vocab_json: Path | None,
    organ_names: tuple[str, ...],
    include_review_required: bool,
    confidence_scaling: bool,
    review_required_weight: float,
) -> SemanticTargetLookup | None:
    if targets_jsonl is None or vocab_json is None:
        return None
    if not targets_jsonl.is_file():
        raise FileNotFoundError(f"Semantic targets JSONL not found: {targets_jsonl}")
    if not vocab_json.is_file():
        raise FileNotFoundError(f"Semantic vocab JSON not found: {vocab_json}")

    allowed_organs = {str(organ) for organ in organ_names}
    vocab_payload = json.loads(vocab_json.read_text(encoding="utf-8"))
    subtype_by_organ = _labels_by_organ(vocab_payload, "subtype_labels_by_organ", "organ_to_subtypes")
    family_by_organ = _labels_by_organ(vocab_payload, "family_labels_by_organ", "organ_to_families")
    family_vocab = tuple(sorted({label for labels in family_by_organ.values() for label in labels}))
    subtype_vocab = tuple(sorted({label for labels in subtype_by_organ.values() for label in labels}))
    family_to_index = {label: idx for idx, label in enumerate(family_vocab)}
    subtype_to_index = {label: idx for idx, label in enumerate(subtype_vocab)}
    organ_to_family_vocab = {organ: tuple(family_by_organ.get(organ, ())) for organ in allowed_organs}
    organ_to_subtype_vocab = {organ: tuple(subtype_by_organ.get(organ, ())) for organ in allowed_organs}

    targets: dict[tuple[str, str], SemanticTarget] = {}
    with targets_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            organ = str(payload.get("organ", "")).strip()
            raw_text = str(payload.get("raw_text", "")).strip()
            if organ not in allowed_organs or not raw_text:
                continue
            normality = str(payload.get("normality", "")).strip()
            polarity = str(payload.get("polarity", "")).strip()
            if normality not in NORMALITY_TO_INDEX or polarity not in POLARITY_TO_INDEX:
                continue
            review_required = bool(payload.get("review_required", False))
            if review_required and not include_review_required:
                sample_weight = 0.0
            else:
                sample_weight = 1.0
                if review_required:
                    sample_weight *= float(review_required_weight)
                if confidence_scaling:
                    sample_weight *= float(payload.get("confidence_weight", 0.0) or 0.0)
            family_targets = [0.0] * len(family_vocab)
            subtype_targets = [0.0] * len(subtype_vocab)
            for label, weight in dict(payload.get("family_targets", {})).items():
                if label in family_to_index:
                    family_targets[family_to_index[label]] = float(weight)
            for label, weight in dict(payload.get("subtype_targets", {})).items():
                if label in subtype_to_index:
                    subtype_targets[subtype_to_index[label]] = float(weight)
            family_allowed = [False] * len(family_vocab)
            subtype_allowed = [False] * len(subtype_vocab)
            for label in organ_to_family_vocab.get(organ, ()):
                if label in family_to_index:
                    family_allowed[family_to_index[label]] = True
            for label in organ_to_subtype_vocab.get(organ, ()):
                if label in subtype_to_index:
                    subtype_allowed[subtype_to_index[label]] = True
            targets[(_key(organ), _key(raw_text))] = SemanticTarget(
                normality_index=NORMALITY_TO_INDEX[normality],
                polarity_index=POLARITY_TO_INDEX[polarity],
                confidence_weight=float(payload.get("confidence_weight", 0.0) or 0.0),
                sample_weight=float(sample_weight),
                family_targets=tuple(family_targets),
                subtype_targets=tuple(subtype_targets),
                family_allowed=tuple(family_allowed),
                subtype_allowed=tuple(subtype_allowed),
                review_required=review_required,
            )
    return SemanticTargetLookup(
        targets,
        SemanticTargetSpec(
            family_vocab=family_vocab,
            subtype_vocab=subtype_vocab,
            organ_to_family_vocab=organ_to_family_vocab,
            organ_to_subtype_vocab=organ_to_subtype_vocab,
        ),
    )


def _key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _labels_by_organ(payload: Mapping[str, object], rich_key: str, simple_key: str) -> dict[str, tuple[str, ...]]:
    if rich_key in payload:
        result: dict[str, tuple[str, ...]] = {}
        for organ, rows in dict(payload.get(rich_key, {})).items():
            labels = []
            for row in rows or []:
                if isinstance(row, Mapping):
                    label = str(row.get("label", "")).strip()
                else:
                    label = str(row).strip()
                if label:
                    labels.append(label)
            result[str(organ)] = tuple(sorted(set(labels)))
        return result
    return {
        str(organ): tuple(sorted(str(label).strip() for label in labels if str(label).strip()))
        for organ, labels in dict(payload.get(simple_key, {})).items()
    }
