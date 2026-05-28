#!/usr/bin/env python3
"""Build diagnostic-loss-only lexical artifacts from semantic training targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_DIR = Path(
    "/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/"
    "merlin_converted/consolidation/consolidation_v3/postprocess_v3_clean"
)
DEFAULT_OUTPUT_DIR = Path(
    "/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/diagnostic_lexicon/"
    "merlin_converted/lexical_diag_v1_from_semantic_v3"
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "cm",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "left",
    "no",
    "normal",
    "not",
    "of",
    "on",
    "or",
    "right",
    "the",
    "there",
    "to",
    "unremarkable",
    "with",
    "without",
}

NORMAL_PHRASES = (
    "normal",
    "unremarkable",
    "within normal limits",
    "no abnormality",
    "no focal abnormality",
)

ANATOMY_TERMS = {
    "adrenal",
    "adrenals",
    "bladder",
    "bowel",
    "colon",
    "colonic",
    "gallbladder",
    "gland",
    "glands",
    "hepatic",
    "kidney",
    "kidneys",
    "liver",
    "pancreas",
    "pancreatic",
    "prostate",
    "prostatic",
    "rectal",
    "spleen",
    "splenic",
    "stomach",
    "urinary",
}


@dataclass(frozen=True)
class BuildConfig:
    semantic_targets_jsonl: Path
    training_vocab_json: Path
    output_dir: Path
    include_review_required: bool
    min_subtype_count: int
    max_rows_per_label: int
    max_phrases_per_label: int
    tokenizer_name: str


def main() -> None:
    args = _parse_args()
    config = BuildConfig(
        semantic_targets_jsonl=Path(args.semantic_targets_jsonl),
        training_vocab_json=Path(args.training_vocab_json),
        output_dir=Path(args.output_dir),
        include_review_required=bool(args.include_review_required),
        min_subtype_count=int(args.min_subtype_count),
        max_rows_per_label=int(args.max_rows_per_label),
        max_phrases_per_label=int(args.max_phrases_per_label),
        tokenizer_name=str(args.tokenizer_name or "").strip(),
    )
    run(config)


def run(config: BuildConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "reports").mkdir(parents=True, exist_ok=True)

    rows = list(_read_jsonl(config.semantic_targets_jsonl))
    vocab = json.loads(config.training_vocab_json.read_text(encoding="utf-8"))
    vocab_info = _flatten_vocab(vocab)
    usable_rows = [
        row
        for row in rows
        if _row_sample_weight(row, include_review_required=config.include_review_required) > 0.0
    ]

    examples_by_label = _collect_examples_by_label(usable_rows, max_rows_per_label=config.max_rows_per_label)
    registry = _build_registry(
        vocab_info=vocab_info,
        examples_by_label=examples_by_label,
        min_subtype_count=config.min_subtype_count,
        max_phrases_per_label=config.max_phrases_per_label,
    )
    targets = _build_sample_targets(
        rows=rows,
        registry=registry,
        include_review_required=config.include_review_required,
    )

    registry_path = config.output_dir / "lexicon_registry_v1.json"
    targets_path = config.output_dir / "sample_level_lexical_targets_v1.jsonl"
    write_json(registry_path, registry)
    write_jsonl(targets_path, targets)

    token_cache_path = None
    concept_token_cache_path = None
    if config.tokenizer_name:
        token_cache_path = config.output_dir / "tokenized_lexical_targets_v1.pt"
        concept_token_cache_path = config.output_dir / "tokenized_concept_lexical_targets_v1.pt"
        _write_token_cache(
            tokenizer_name=config.tokenizer_name,
            targets=targets,
            output_path=token_cache_path,
            concept_output_path=concept_token_cache_path,
            source_jsonl_sha256=_sha256(targets_path),
        )

    report = _write_coverage_report(
        config=config,
        all_rows=rows,
        usable_rows=usable_rows,
        registry=registry,
        targets=targets,
        registry_path=registry_path,
        targets_path=targets_path,
        token_cache_path=token_cache_path,
    )
    manifest = {
        "builder": "diagnostic_lexicon.apps.build_diagnostic_loss_artifacts",
        "semantic_targets_jsonl": str(config.semantic_targets_jsonl),
        "semantic_targets_sha256": _sha256(config.semantic_targets_jsonl),
        "training_vocab_json": str(config.training_vocab_json),
        "training_vocab_sha256": _sha256(config.training_vocab_json),
        "outputs": {
            "lexicon_registry": str(registry_path),
            "sample_level_lexical_targets": str(targets_path),
            "tokenized_lexical_targets": None if token_cache_path is None else str(token_cache_path),
            "tokenized_concept_lexical_targets": None if concept_token_cache_path is None else str(concept_token_cache_path),
            "coverage_report": str(report),
        },
        "config": {
            "include_review_required": config.include_review_required,
            "min_subtype_count": config.min_subtype_count,
            "max_rows_per_label": config.max_rows_per_label,
            "max_phrases_per_label": config.max_phrases_per_label,
            "tokenizer_name": config.tokenizer_name,
        },
        "summary": _summary(rows, usable_rows, registry, targets),
    }
    write_json(config.output_dir / "manifest.json", manifest)
    print(f"[diagnostic_lexicon] wrote {registry_path}")
    print(f"[diagnostic_lexicon] wrote {targets_path}")
    if token_cache_path is not None:
        print(f"[diagnostic_lexicon] wrote {token_cache_path}")
    if concept_token_cache_path is not None:
        print(f"[diagnostic_lexicon] wrote {concept_token_cache_path}")
    print(f"[diagnostic_lexicon] wrote {report}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--semantic-targets-jsonl",
        default=str(DEFAULT_SOURCE_DIR / "semantic_training_targets_v3.jsonl"),
        help="Clean semantic training targets JSONL.",
    )
    parser.add_argument(
        "--training-vocab-json",
        default=str(DEFAULT_SOURCE_DIR / "training_vocab_v3_clean.json"),
        help="Clean training vocabulary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for diagnostic lexicon artifacts.",
    )
    parser.add_argument("--include-review-required", action="store_true")
    parser.add_argument("--min-subtype-count", type=int, default=20)
    parser.add_argument("--max-rows-per-label", type=int, default=128)
    parser.add_argument("--max-phrases-per-label", type=int, default=24)
    parser.add_argument(
        "--tokenizer-name",
        default="",
        help="Optional HF tokenizer name/path. If set, writes tokenized_lexical_targets_v1.pt.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _flatten_vocab(vocab: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for label_type, section in (
        ("subtype", "subtype_labels_by_organ"),
        ("family", "family_labels_by_organ"),
    ):
        for organ, entries in dict(vocab.get(section, {})).items():
            for entry in entries or []:
                label = str(entry.get("label", "")).strip()
                if label:
                    out[(str(organ), label_type, label)] = dict(entry)
    return out


def _collect_examples_by_label(
    rows: list[dict[str, Any]],
    *,
    max_rows_per_label: int,
) -> dict[tuple[str, str, str], list[str]]:
    examples: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        organ = str(row.get("organ", "")).strip()
        text = str(row.get("normalized_text") or row.get("raw_text") or "").strip()
        if not organ or not text:
            continue
        for label in dict(row.get("subtype_targets", {})):
            key = (organ, "subtype", str(label))
            if len(examples[key]) < max_rows_per_label:
                examples[key].append(text)
        for label in dict(row.get("family_targets", {})):
            key = (organ, "family", str(label))
            if len(examples[key]) < max_rows_per_label:
                examples[key].append(text)
    return examples


def _build_registry(
    *,
    vocab_info: dict[tuple[str, str, str], dict[str, Any]],
    examples_by_label: dict[tuple[str, str, str], list[str]],
    min_subtype_count: int,
    max_phrases_per_label: int,
) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    for key in sorted(vocab_info):
        organ, label_type, label = key
        info = vocab_info[key]
        count = int(info.get("total_unique_text_count", 0) or 0)
        review_required = bool(info.get("review_required", False))
        if label_type == "subtype" and count < min_subtype_count:
            continue
        examples = examples_by_label.get(key, [])
        seed_phrases = _label_phrases(label, organ=organ, label_type=label_type)
        mined_phrases = _mine_phrases(examples, label=label, limit=max_phrases_per_label)
        positive = _dedupe_phrases([*seed_phrases, *mined_phrases], limit=max_phrases_per_label)
        negative = _negative_phrases_for_label(label=label, label_type=label_type)
        uncertain = [f"possible {phrase}" for phrase in positive[:3] if len(phrase.split()) <= 4]
        registry.append(
            {
                "organ": organ,
                "label_type": label_type,
                "label": label,
                "positive_phrases": positive,
                "negative_phrases": negative,
                "uncertain_phrases": _dedupe_phrases(uncertain, limit=6),
                "confuser_phrases": [],
                "source": {
                    "semantic_artifact": "semantic_training_targets_v3",
                    "example_count": len(examples),
                    "total_unique_text_count": count,
                    "review_required": review_required,
                    "review_flags": sorted(set(info.get("review_flags", []) or [])),
                    "min_loss_weight": float(info.get("min_loss_weight", 0.0) or 0.0),
                    "max_loss_weight": float(info.get("max_loss_weight", 0.0) or 0.0),
                },
            }
        )
    return registry


def _build_sample_targets(
    *,
    rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    include_review_required: bool,
) -> list[dict[str, Any]]:
    by_key = {(entry["organ"], entry["label_type"], entry["label"]): entry for entry in registry}
    normal_by_organ = _normal_phrases_by_organ(registry)
    abnormal_by_organ = _abnormal_phrases_by_organ(registry)
    targets: list[dict[str, Any]] = []
    for row in rows:
        organ = str(row.get("organ", "")).strip()
        if not organ:
            continue
        sample_weight = _row_sample_weight(row, include_review_required=include_review_required)
        subtype_targets = dict(row.get("subtype_targets", {}))
        family_targets = dict(row.get("family_targets", {}))
        positive_concepts: list[dict[str, Any]] = []
        for label, weight in sorted(subtype_targets.items()):
            entry = by_key.get((organ, "subtype", str(label)))
            if entry:
                positive_concepts.append(_concept(entry, weight=float(weight)))
        for label, weight in sorted(family_targets.items()):
            entry = by_key.get((organ, "family", str(label)))
            if entry:
                positive_concepts.append(_concept(entry, weight=float(weight) * 0.75))
        negative_concepts = _negative_concepts(row, normal_by_organ, abnormal_by_organ)
        if sample_weight <= 0.0 or (not positive_concepts and not negative_concepts):
            continue
        targets.append(
            {
                "organ": organ,
                "raw_text": row.get("raw_text"),
                "normalized_text": row.get("normalized_text"),
                "normality": row.get("normality"),
                "polarity": row.get("polarity"),
                "positive_concepts": positive_concepts,
                "negative_concepts": negative_concepts,
                "uncertain_concepts": [],
                "sample_weight": sample_weight,
                "review_required": bool(row.get("review_required", False)),
                "decision_status": row.get("decision_status"),
                "source_targets": {
                    "family_targets": family_targets,
                    "subtype_targets": subtype_targets,
                },
                "provenance": {
                    "source_observed_subtypes": row.get("source_observed_subtypes", []),
                    "review_flags": row.get("review_flags", []),
                },
            }
        )
    return targets


def _concept(entry: dict[str, Any], *, weight: float) -> dict[str, Any]:
    return {
        "source_label": entry["label"],
        "label_type": entry["label_type"],
        "phrases": entry["positive_phrases"],
        "weight": float(weight),
    }


def _negative_concepts(
    row: dict[str, Any],
    normal_by_organ: dict[str, list[str]],
    abnormal_by_organ: dict[str, list[str]],
) -> list[dict[str, Any]]:
    organ = str(row.get("organ", "")).strip()
    normality = str(row.get("normality", "")).strip()
    polarity = str(row.get("polarity", "")).strip()
    concepts: list[dict[str, Any]] = []
    if normality in {"abnormal", "mixed"} or polarity in {"positive", "mixed"}:
        phrases = normal_by_organ.get(organ, list(NORMAL_PHRASES))
        concepts.append(
            {
                "source_label": "normal_wording",
                "label_type": "normal_negative",
                "phrases": phrases[:12],
                "weight": 0.25,
            }
        )
    if normality == "normal" or polarity == "negative":
        phrases = abnormal_by_organ.get(organ, [])
        if phrases:
            concepts.append(
                {
                    "source_label": "same_organ_abnormal_wording",
                    "label_type": "abnormal_negative",
                    "phrases": phrases[:32],
                    "weight": 0.35,
                }
            )
    return concepts


def _normal_phrases_by_organ(registry: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for entry in registry:
        label = str(entry["label"])
        if label == "normal" or label.endswith("_normal") or (str(entry["label_type"]) == "family" and label == "normal"):
            out[str(entry["organ"])].extend(entry["positive_phrases"])
    for organ in list(out):
        out[organ] = _dedupe_phrases([*out[organ], *NORMAL_PHRASES], limit=16)
    return out


def _abnormal_phrases_by_organ(registry: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for entry in registry:
        label = str(entry["label"])
        if label == "normal" or label.endswith("_normal") or label == "absent_postop":
            continue
        out[str(entry["organ"])].extend(entry["positive_phrases"][:4])
    for organ in list(out):
        out[organ] = _dedupe_phrases(out[organ], limit=64)
    return out


def _row_sample_weight(row: dict[str, Any], *, include_review_required: bool) -> float:
    if bool(row.get("review_required", False)) and not include_review_required:
        return 0.0
    confidence = float(row.get("confidence_weight", 0.0) or 0.0)
    status = str(row.get("decision_status", "")).strip()
    status_weight = {"accepted": 1.0, "accepted_provisional": 0.5}.get(status, 0.0)
    if bool(row.get("review_required", False)):
        status_weight *= 0.25
    return max(0.0, confidence * status_weight)


def _label_phrases(label: str, *, organ: str, label_type: str) -> list[str]:
    cleaned = _label_core(label, organ=organ)
    phrase = cleaned.replace("_", " ").strip()
    phrases = [phrase] if phrase else []
    return phrases


def _negative_phrases_for_label(*, label: str, label_type: str) -> list[str]:
    positive = _label_phrases(label, organ="", label_type=label_type)
    negatives = []
    for phrase in positive[:4]:
        if phrase:
            negatives.extend([f"no {phrase}", f"without {phrase}"])
    return _dedupe_phrases(negatives, limit=8)


def _mine_phrases(examples: list[str], *, label: str, limit: int) -> list[str]:
    label_terms = {term for term in _label_core(label, organ="").split("_") if len(term) > 2 and term not in STOPWORDS}
    counts: Counter[str] = Counter()
    for text in examples:
        tokens = _tokens(text)
        for n in (1, 2, 3):
            for i in range(0, max(0, len(tokens) - n + 1)):
                phrase_tokens = tokens[i : i + n]
                if not phrase_tokens:
                    continue
                if not any(token in label_terms for token in phrase_tokens):
                    continue
                phrase = " ".join(phrase_tokens)
                if _good_phrase(phrase):
                    counts[phrase] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
    return [phrase for phrase, _ in ranked[:limit]]


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z]+", str(text).lower())
        if token not in STOPWORDS and token not in ANATOMY_TERMS and len(token) > 2
    ]


def _good_phrase(phrase: str) -> bool:
    if len(phrase) < 3:
        return False
    if any(part.isdigit() for part in phrase.split()):
        return False
    if any(part in {"normal", "unremarkable", "without", "no"} for part in phrase.split()):
        return False
    if all(part in ANATOMY_TERMS for part in phrase.split()):
        return False
    return True


def _dedupe_phrases(phrases: Iterable[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        normalized = re.sub(r"\s+", " ", str(phrase).strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")


def _label_core(label: str, *, organ: str) -> str:
    cleaned = _slug(label)
    organ_slug = _slug(organ)
    prefixes = [
        organ_slug,
        organ_slug.rstrip("s"),
        "adrenal_glands",
        "adrenal",
        "small_bowel",
        "urinary_bladder",
        "kidneys",
        "kidney",
        "gallbladder",
        "colon",
        "pancreas",
        "liver",
        "spleen",
        "stomach",
        "prostate",
    ]
    for prefix in prefixes:
        if prefix and cleaned.startswith(prefix + "_"):
            return cleaned[len(prefix) + 1 :]
    return cleaned


def _write_token_cache(
    *,
    tokenizer_name: str,
    targets: list[dict[str, Any]],
    output_path: Path,
    concept_output_path: Path,
    source_jsonl_sha256: str,
) -> None:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Token cache generation requires torch and transformers.") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    phrase_cache: dict[str, tuple[int, ...]] = {}
    rows = []
    concept_rows = []
    for row in targets:
        positive_phrases = _phrases_from_concepts(row.get("positive_concepts", []))
        negative_phrases = _phrases_from_concepts(row.get("negative_concepts", []))
        rows.append(
            {
                "key": (row["organ"], row["normalized_text"]),
                "positive_token_ids": _phrase_token_ids(tokenizer, positive_phrases, phrase_cache),
                "negative_token_ids": _phrase_token_ids(tokenizer, negative_phrases, phrase_cache),
                "sample_weight": float(row["sample_weight"]),
                "review_required": bool(row["review_required"]),
            }
        )
        concept_rows.append(_tokenized_concept_row(tokenizer, row, phrase_cache))
    payload = {
        "tokenizer_name": tokenizer_name,
        "source_jsonl_sha256": source_jsonl_sha256,
        "rows": rows,
    }
    concept_payload = {
        "tokenizer_name": tokenizer_name,
        "source_jsonl_sha256": source_jsonl_sha256,
        "target_format": "concept_specific_lexical_v1",
        "rows": concept_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    torch.save(concept_payload, concept_output_path)


def _tokenized_concept_row(tokenizer: Any, row: dict[str, Any], phrase_cache: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    positive = [_tokenized_concept(tokenizer, concept, phrase_cache) for concept in row.get("positive_concepts", [])]
    positive = [concept for concept in positive if concept["token_ids"]]
    positive_union = {token_id for concept in positive for token_id in concept["token_ids"]}

    negative = []
    mixed_normality = str(row.get("normality", "")).strip() == "mixed"
    for concept in row.get("negative_concepts", []):
        tokenized = _tokenized_concept(tokenizer, concept, phrase_cache)
        tokenized["token_ids"] = [token_id for token_id in tokenized["token_ids"] if token_id not in positive_union]
        if mixed_normality and tokenized["source_label"] == "normal_wording":
            tokenized["weight"] = min(float(tokenized["weight"]), 0.05)
        if tokenized["token_ids"] and float(tokenized["weight"]) > 0.0:
            negative.append(tokenized)
    return {
        "key": (row["organ"], row["normalized_text"]),
        "positive_concepts": positive,
        "negative_concepts": negative,
        "sample_weight": float(row["sample_weight"]),
        "review_required": bool(row["review_required"]),
        "normality": row.get("normality"),
        "polarity": row.get("polarity"),
    }


def _tokenized_concept(tokenizer: Any, concept: dict[str, Any], phrase_cache: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    phrases = _dedupe_phrases((str(value) for value in concept.get("phrases", []) if str(value).strip()), limit=128)
    return {
        "source_label": str(concept.get("source_label", "")),
        "label_type": str(concept.get("label_type", "")),
        "weight": float(concept.get("weight", 0.0) or 0.0),
        "phrases": phrases,
        "token_ids": _phrase_token_ids(tokenizer, phrases, phrase_cache),
    }


def _phrases_from_concepts(concepts: Iterable[dict[str, Any]]) -> list[str]:
    phrases: list[str] = []
    for concept in concepts:
        phrases.extend(str(value) for value in concept.get("phrases", []) if str(value).strip())
    return _dedupe_phrases(phrases, limit=128)


def _phrase_token_ids(tokenizer: Any, phrases: Iterable[str], phrase_cache: dict[str, tuple[int, ...]] | None = None) -> list[int]:
    token_ids: set[int] = set()
    for phrase in phrases:
        text = str(phrase)
        key = re.sub(r"\s+", " ", text.strip().lower())
        if phrase_cache is not None and key in phrase_cache:
            token_ids.update(phrase_cache[key])
            continue
        encoded = tokenizer(text, add_special_tokens=False).get("input_ids", [])
        phrase_ids = tuple(int(value) for value in encoded)
        if phrase_cache is not None:
            phrase_cache[key] = phrase_ids
        token_ids.update(phrase_ids)
    return sorted(token_ids)


def _write_coverage_report(
    *,
    config: BuildConfig,
    all_rows: list[dict[str, Any]],
    usable_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    registry_path: Path,
    targets_path: Path,
    token_cache_path: Path | None,
) -> Path:
    report_path = config.output_dir / "reports" / "coverage.md"
    subtype_count = sum(1 for entry in registry if entry["label_type"] == "subtype")
    family_count = sum(1 for entry in registry if entry["label_type"] == "family")
    positive_sizes = [_concept_phrase_count(row.get("positive_concepts", [])) for row in targets]
    negative_sizes = [_concept_phrase_count(row.get("negative_concepts", [])) for row in targets]
    labels_without_phrases = [
        f'{entry["organ"]}/{entry["label_type"]}/{entry["label"]}'
        for entry in registry
        if not entry.get("positive_phrases")
    ]
    lines = [
        "# Diagnostic Lexicon Coverage",
        "",
        f"- semantic target rows: `{len(all_rows)}`",
        f"- usable semantic rows: `{len(usable_rows)}`",
        f"- lexical target rows: `{len(targets)}`",
        f"- registry entries: `{len(registry)}`",
        f"- subtype registry entries: `{subtype_count}`",
        f"- family registry entries: `{family_count}`",
        f"- include review required: `{config.include_review_required}`",
        f"- min subtype count: `{config.min_subtype_count}`",
        f"- average positive token phrases per row: `{_mean(positive_sizes):.2f}`",
        f"- average negative token phrases per row: `{_mean(negative_sizes):.2f}`",
        "",
        "## Outputs",
        "",
        f"- registry: `{registry_path}`",
        f"- sample targets: `{targets_path}`",
        f"- token cache: `{token_cache_path or 'not requested'}`",
        "",
        "## Labels Without Positive Phrases",
        "",
    ]
    if labels_without_phrases:
        lines.extend(f"- `{label}`" for label in labels_without_phrases[:100])
    else:
        lines.append("- none")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _concept_phrase_count(concepts: Iterable[dict[str, Any]]) -> int:
    return sum(len(concept.get("phrases", []) or []) for concept in concepts)


def _mean(values: list[int]) -> float:
    return 0.0 if not values else float(sum(values)) / float(len(values))


def _summary(
    rows: list[dict[str, Any]],
    usable_rows: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "semantic_target_rows": len(rows),
        "usable_semantic_rows": len(usable_rows),
        "registry_entries": len(registry),
        "subtype_registry_entries": sum(1 for entry in registry if entry["label_type"] == "subtype"),
        "family_registry_entries": sum(1 for entry in registry if entry["label_type"] == "family"),
        "sample_level_lexical_targets": len(targets),
        "review_required_rows": sum(1 for row in rows if row.get("review_required")),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
