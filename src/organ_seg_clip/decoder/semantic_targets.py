"""Semantic supervision targets for decoder training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


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


def _normalize_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _normalize_subtype(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


@dataclass(frozen=True)
class SemanticExampleTarget:
    organ_name: str
    raw_text: str
    normality_index: int
    polarity_index: int
    confidence: float
    decision_status: str
    sample_weight: float
    subtype_indices: tuple[int, ...]
    subtype_weights: Mapping[int, float]
    family_indices: tuple[int, ...]
    family_weights: Mapping[int, float]
    primary_subtype_index: int
    secondary_subtype_indices: tuple[int, ...]
    review_required: bool


@dataclass(frozen=True)
class SemanticTargetSpec:
    subtype_vocab: tuple[str, ...]
    subtype_to_index: Mapping[str, int]
    organ_to_subtype_indices: Mapping[str, tuple[int, ...]]
    family_vocab: tuple[str, ...]
    family_to_index: Mapping[str, int]
    organ_to_family_indices: Mapping[str, tuple[int, ...]]


class SemanticTargetLookup:
    def __init__(
        self,
        *,
        targets_by_key: Mapping[tuple[str, str], SemanticExampleTarget],
        spec: SemanticTargetSpec,
    ) -> None:
        self._targets_by_key = dict(targets_by_key)
        self.spec = spec

    def get(self, organ_name: str, target_text: str) -> SemanticExampleTarget | None:
        return self._targets_by_key.get((_normalize_key(organ_name), _normalize_key(target_text)))

    @property
    def size(self) -> int:
        return len(self._targets_by_key)

    @classmethod
    def from_training_targets(
        cls,
        *,
        targets_path: Path,
        vocab_path: Path,
        organ_names: Iterable[str],
        accepted_sample_weight: float,
        provisional_sample_weight: float,
        unresolved_sample_weight: float,
        use_confidence_scaling: bool,
        include_review_required: bool,
        review_required_sample_weight: float,
    ) -> "SemanticTargetLookup":
        allowed_organs = {str(value).strip() for value in organ_names}
        subtype_vocab, organ_to_subtypes, family_vocab, organ_to_families = _load_training_vocab(vocab_path, allowed_organs)
        subtype_to_index = {name: index for index, name in enumerate(subtype_vocab)}
        family_to_index = {name: index for index, name in enumerate(family_vocab)}
        spec = SemanticTargetSpec(
            subtype_vocab=subtype_vocab,
            subtype_to_index=subtype_to_index,
            organ_to_subtype_indices={
                organ: tuple(sorted(subtype_to_index[label] for label in labels if label in subtype_to_index))
                for organ, labels in organ_to_subtypes.items()
            },
            family_vocab=family_vocab,
            family_to_index=family_to_index,
            organ_to_family_indices={
                organ: tuple(sorted(family_to_index[label] for label in labels if label in family_to_index))
                for organ, labels in organ_to_families.items()
            },
        )
        targets_by_key: dict[tuple[str, str], SemanticExampleTarget] = {}
        if not targets_path.is_file():
            raise FileNotFoundError(f"Semantic training targets JSONL not found: {targets_path}")
        with targets_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                organ_name = str(payload.get("organ", "")).strip()
                if organ_name not in allowed_organs:
                    continue
                raw_text = str(payload.get("raw_text", "")).strip()
                if not raw_text:
                    continue
                normality = str(payload.get("normality", "")).strip()
                polarity = str(payload.get("polarity", "")).strip()
                if normality not in NORMALITY_TO_INDEX or polarity not in POLARITY_TO_INDEX:
                    continue
                review_required = bool(payload.get("review_required", False))
                if review_required and not include_review_required:
                    sample_weight = 0.0
                else:
                    sample_weight = _status_weight(
                        str(payload.get("decision_status", "")).strip(),
                        accepted_sample_weight=accepted_sample_weight,
                        provisional_sample_weight=provisional_sample_weight,
                        unresolved_sample_weight=unresolved_sample_weight,
                    )
                    if review_required:
                        sample_weight *= float(review_required_sample_weight)
                    if use_confidence_scaling:
                        sample_weight *= float(payload.get("confidence_weight", 0.0) or 0.0)
                subtype_weights = {
                    int(subtype_to_index[label]): float(weight)
                    for label, weight in dict(payload.get("subtype_targets", {})).items()
                    if label in subtype_to_index
                }
                family_weights = {
                    int(family_to_index[label]): float(weight)
                    for label, weight in dict(payload.get("family_targets", {})).items()
                    if label in family_to_index
                }
                if not subtype_weights and not family_weights:
                    sample_weight = 0.0
                primary_index = next(iter(sorted(subtype_weights)), -100)
                targets_by_key[(_normalize_key(organ_name), _normalize_key(raw_text))] = SemanticExampleTarget(
                    organ_name=organ_name,
                    raw_text=raw_text,
                    normality_index=int(NORMALITY_TO_INDEX[normality]),
                    polarity_index=int(POLARITY_TO_INDEX[polarity]),
                    confidence=float(payload.get("confidence_weight", 0.0) or 0.0),
                    decision_status=str(payload.get("decision_status", "")).strip(),
                    sample_weight=float(sample_weight),
                    subtype_indices=tuple(sorted(subtype_weights)),
                    subtype_weights=subtype_weights,
                    family_indices=tuple(sorted(family_weights)),
                    family_weights=family_weights,
                    primary_subtype_index=int(primary_index),
                    secondary_subtype_indices=tuple(index for index in sorted(subtype_weights) if index != primary_index),
                    review_required=review_required,
                )
        return cls(targets_by_key=targets_by_key, spec=spec)

    @classmethod
    def from_jsonl_paths(
        cls,
        paths: Iterable[Path],
        *,
        organ_names: Iterable[str],
        accepted_sample_weight: float,
        provisional_sample_weight: float,
        unresolved_sample_weight: float,
        use_confidence_scaling: bool,
    ) -> "SemanticTargetLookup | None":
        resolved_paths = [Path(path).expanduser().resolve() for path in paths if str(path).strip()]
        if not resolved_paths:
            return None
        allowed_organs = {str(value).strip() for value in organ_names}
        raw_targets: dict[tuple[str, str], dict[str, object]] = {}
        organ_to_subtypes: dict[str, set[str]] = {organ: set() for organ in allowed_organs}
        for path in resolved_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Semantic target JSONL not found: {path}")
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    organ_name = str(payload.get("organ", "")).strip()
                    if organ_name not in allowed_organs:
                        continue
                    raw_text = str(payload.get("raw_text", "")).strip()
                    if not raw_text:
                        continue
                    primary_subtype = _normalize_subtype(payload.get("primary_subtype"))
                    secondary_subtypes = tuple(
                        subtype
                        for subtype in (_normalize_subtype(value) for value in payload.get("secondary_subtypes", []))
                        if subtype is not None
                    )
                    if primary_subtype is not None:
                        organ_to_subtypes[organ_name].add(primary_subtype)
                    organ_to_subtypes[organ_name].update(secondary_subtypes)
                    raw_targets[(_normalize_key(organ_name), _normalize_key(raw_text))] = {
                        "organ_name": organ_name,
                        "raw_text": raw_text,
                        "normality": str(payload.get("normality", "")).strip(),
                        "polarity": str(payload.get("polarity", "")).strip(),
                        "confidence": float(payload.get("confidence", 0.0) or 0.0),
                        "decision_status": str(payload.get("decision_status", "")).strip(),
                        "primary_subtype": primary_subtype,
                        "secondary_subtypes": secondary_subtypes,
                    }
        subtype_vocab = tuple(sorted({subtype for subtypes in organ_to_subtypes.values() for subtype in subtypes}))
        subtype_to_index = {name: index for index, name in enumerate(subtype_vocab)}
        organ_to_subtype_indices = {
            organ_name: tuple(sorted(subtype_to_index[subtype] for subtype in organ_to_subtypes.get(organ_name, set())))
            for organ_name in sorted(allowed_organs)
        }
        spec = SemanticTargetSpec(
            subtype_vocab=subtype_vocab,
            subtype_to_index=subtype_to_index,
            organ_to_subtype_indices=organ_to_subtype_indices,
            family_vocab=(),
            family_to_index={},
            organ_to_family_indices={organ_name: () for organ_name in sorted(allowed_organs)},
        )
        targets_by_key: dict[tuple[str, str], SemanticExampleTarget] = {}
        for key, payload in raw_targets.items():
            normality = str(payload["normality"]).strip()
            polarity = str(payload["polarity"]).strip()
            if normality not in NORMALITY_TO_INDEX or polarity not in POLARITY_TO_INDEX:
                continue
            status = str(payload["decision_status"]).strip()
            status_weight = {
                "accepted": float(accepted_sample_weight),
                "accepted_provisional": float(provisional_sample_weight),
                "unresolved": float(unresolved_sample_weight),
            }.get(status, 0.0)
            confidence = float(payload["confidence"])
            sample_weight = status_weight * confidence if use_confidence_scaling else status_weight
            primary_subtype = payload["primary_subtype"]
            secondary_subtypes = tuple(payload["secondary_subtypes"])
            active_subtype_names = tuple(
                subtype for subtype in ((primary_subtype,) + secondary_subtypes) if subtype is not None and subtype in subtype_to_index
            )
            if primary_subtype is None or primary_subtype not in subtype_to_index:
                sample_weight = 0.0
                primary_subtype_index = -100
            else:
                primary_subtype_index = int(subtype_to_index[primary_subtype])
            targets_by_key[key] = SemanticExampleTarget(
                organ_name=str(payload["organ_name"]),
                raw_text=str(payload["raw_text"]),
                normality_index=int(NORMALITY_TO_INDEX[normality]),
                polarity_index=int(POLARITY_TO_INDEX[polarity]),
                confidence=confidence,
                decision_status=status,
                sample_weight=float(sample_weight),
                subtype_indices=tuple(sorted({int(subtype_to_index[name]) for name in active_subtype_names})),
                subtype_weights={int(subtype_to_index[name]): 1.0 for name in active_subtype_names},
                family_indices=(),
                family_weights={},
                primary_subtype_index=primary_subtype_index,
                secondary_subtype_indices=tuple(
                    sorted(
                        {
                            int(subtype_to_index[name])
                            for name in secondary_subtypes
                            if name in subtype_to_index and int(subtype_to_index[name]) != primary_subtype_index
                        }
                    )
                ),
                review_required=False,
            )
        return cls(targets_by_key=targets_by_key, spec=spec)


def _load_training_vocab(
    vocab_path: Path,
    allowed_organs: set[str],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], tuple[str, ...], dict[str, tuple[str, ...]]]:
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Semantic training vocab JSON not found: {vocab_path}")
    with vocab_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    organ_to_subtypes: dict[str, tuple[str, ...]] = {}
    organ_to_families: dict[str, tuple[str, ...]] = {}
    subtype_labels: set[str] = set()
    family_labels: set[str] = set()
    for organ_name, rows in dict(payload.get("subtype_labels_by_organ", {})).items():
        if organ_name not in allowed_organs:
            continue
        labels = tuple(str(row.get("label", "")).strip() for row in rows if str(row.get("label", "")).strip())
        organ_to_subtypes[organ_name] = labels
        subtype_labels.update(labels)
    for organ_name, rows in dict(payload.get("family_labels_by_organ", {})).items():
        if organ_name not in allowed_organs:
            continue
        labels = tuple(str(row.get("label", "")).strip() for row in rows if str(row.get("label", "")).strip())
        organ_to_families[organ_name] = labels
        family_labels.update(labels)
    for organ_name in allowed_organs:
        organ_to_subtypes.setdefault(organ_name, ())
        organ_to_families.setdefault(organ_name, ())
    return tuple(sorted(subtype_labels)), organ_to_subtypes, tuple(sorted(family_labels)), organ_to_families


def _status_weight(
    status: str,
    *,
    accepted_sample_weight: float,
    provisional_sample_weight: float,
    unresolved_sample_weight: float,
) -> float:
    return {
        "accepted": float(accepted_sample_weight),
        "accepted_provisional": float(provisional_sample_weight),
        "unresolved": float(unresolved_sample_weight),
    }.get(str(status).strip(), 0.0)
