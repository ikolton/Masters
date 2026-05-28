"""Dataset construction for Merlin organ-report ablations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AblationConfig
from .lexical_targets import LexicalTargetLookup
from .semantic_targets import SemanticTargetLookup, load_semantic_targets


PROMPT_ORGAN_ALIASES = {
    "Adrenal glands": "adrenal glands",
    "Colon": "colon",
    "Gallbladder": "gallbladder",
    "Kidneys": "kidneys",
    "Liver": "liver",
    "Pancreas": "pancreas",
    "Prostate": "prostate",
    "Small bowel": "small bowel",
    "Spleen": "spleen",
    "Stomach": "stomach",
    "Urinary bladder": "urinary bladder",
}


@dataclass(frozen=True)
class DatasetBundle:
    train_records: list[dict[str, Any]]
    val_records: list[dict[str, Any]]
    semantic_lookup: SemanticTargetLookup | None
    summary: dict[str, Any]


def build_datasets(config: AblationConfig) -> DatasetBundle:
    lexical_lookup = LexicalTargetLookup(config.paths.metadata_csv, config.data.organ_names)
    semantic_lookup = load_semantic_targets(
        targets_jsonl=config.paths.semantic_targets_jsonl,
        vocab_json=config.paths.semantic_vocab_json,
        organ_names=config.data.organ_names,
        include_review_required=config.data.include_review_required_semantic_targets,
        confidence_scaling=config.losses.confidence_scaling,
        review_required_weight=config.losses.review_required_weight,
    )
    train_records = _build_split_records(
        config=config,
        split=config.data.train_split,
        limit=config.data.train_limit,
        lexical_lookup=lexical_lookup,
        semantic_lookup=semantic_lookup,
    )
    val_records = _build_split_records(
        config=config,
        split=config.data.val_split,
        limit=config.data.val_limit,
        lexical_lookup=lexical_lookup,
        semantic_lookup=semantic_lookup,
    )
    summary = {
        "train_records": len(train_records),
        "val_records": len(val_records),
        "semantic_targets_loaded": 0 if semantic_lookup is None else semantic_lookup.size,
        "family_vocab_size": 0 if semantic_lookup is None else len(semantic_lookup.spec.family_vocab),
        "subtype_vocab_size": 0 if semantic_lookup is None else len(semantic_lookup.spec.subtype_vocab),
        "train_split": config.data.train_split,
        "val_split": config.data.val_split,
        "organ_names": list(config.data.organ_names),
    }
    return DatasetBundle(train_records=train_records, val_records=val_records, semantic_lookup=semantic_lookup, summary=summary)


def _build_split_records(
    *,
    config: AblationConfig,
    split: str,
    limit: int | None,
    lexical_lookup: LexicalTargetLookup,
    semantic_lookup: SemanticTargetLookup | None,
) -> list[dict[str, Any]]:
    split_dir = config.paths.dataset_root / split
    manifest_path = split_dir / "combined.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Merlin-converted split manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    family_count = 0 if semantic_lookup is None else len(semantic_lookup.spec.family_vocab)
    subtype_count = 0 if semantic_lookup is None else len(semantic_lookup.spec.subtype_vocab)
    for item in payload:
        study_id = str(item.get("study_id", "")).strip()
        if not study_id:
            continue
        image_path = split_dir / study_id / f"{study_id}_resampled.nii.gz"
        if not image_path.is_file():
            continue
        findings = dict(item.get("findings", {}))
        labels = dict(item.get("labels", {}))
        for organ in config.data.organ_names:
            raw_text = str(findings.get(organ, "")).strip()
            if not raw_text:
                continue
            prompt_organ = PROMPT_ORGAN_ALIASES.get(organ, organ.lower())
            prompt = f"Generate a radiology report for {prompt_organ}###\n"
            lexical_label, lexical_available = lexical_lookup.get(study_id, organ)
            semantic = semantic_lookup.get(organ, raw_text) if semantic_lookup is not None else None
            family_targets = [0.0] * family_count if semantic is None else list(semantic.family_targets)
            family_allowed = [False] * family_count if semantic is None else [bool(value) for value in semantic.family_allowed]
            subtype_targets = [0.0] * subtype_count if semantic is None else list(semantic.subtype_targets)
            subtype_allowed = [False] * subtype_count if semantic is None else [bool(value) for value in semantic.subtype_allowed]
            records.append(
                {
                    "image": str(image_path),
                    "image_embedding": str(config.paths.image_embedding_cache_dir / split / f"{study_id}.pt"),
                    "study_id": study_id,
                    "organ": organ,
                    "prompt": prompt,
                    "target_text": raw_text,
                    "full_text": prompt + raw_text,
                    "organ_abnormal_label": int(labels.get(organ, -1)) if str(labels.get(organ, "")).strip() in {"0", "1"} else -1,
                    "lexical_label": float(lexical_label),
                    "lexical_available": bool(lexical_available),
                    "semantic_available": bool(semantic is not None and semantic.sample_weight > 0.0),
                    "semantic_weight": 0.0 if semantic is None else float(semantic.sample_weight),
                    "semantic_normality": -100 if semantic is None else int(semantic.normality_index),
                    "semantic_polarity": -100 if semantic is None else int(semantic.polarity_index),
                    "semantic_family_targets": family_targets,
                    "semantic_family_allowed": family_allowed,
                    "semantic_subtype_targets": subtype_targets,
                    "semantic_subtype_allowed": subtype_allowed,
                }
            )
    random.Random(config.data.sample_seed).shuffle(records)
    if limit is not None:
        records = records[: int(limit)]
    return records
