from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .table_store import ParquetTableStore, read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class PostprocessConfig:
    dataset_id: str
    postprocess_id: str
    output_root: Path
    consolidation_id: str
    min_subtype_count_without_review: int
    review_very_rare_family_only: bool
    controlled_families: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def consolidation_dir(self) -> Path:
        return self.output_root / self.dataset_id / "consolidation" / self.consolidation_id

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.dataset_id / "consolidation" / self.consolidation_id / self.postprocess_id


def load_postprocess_config(path: Path) -> PostprocessConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    postprocess = payload.get("postprocess", {})
    return PostprocessConfig(
        dataset_id=str(payload["project"]["dataset_id"]),
        postprocess_id=str(payload["project"]["postprocess_id"]),
        output_root=Path(payload["paths"]["output_root"]),
        consolidation_id=str(payload["source"]["consolidation_id"]),
        min_subtype_count_without_review=int(postprocess.get("min_subtype_count_without_review", 20)),
        review_very_rare_family_only=bool(postprocess.get("review_very_rare_family_only", True)),
        controlled_families=tuple(str(item) for item in postprocess.get("controlled_families", _default_families())),
        raw=payload,
    )


def run_postprocess(config: PostprocessConfig, *, config_path: Path) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = config.consolidation_dir / "llm_consolidation_decisions.jsonl"
    composed_path = config.consolidation_dir / "composed_validated_decisions.jsonl"
    stats_path = config.consolidation_dir / "observed_tag_stats.jsonl"

    decision_rows = read_jsonl(decisions_path)
    composed_rows = read_jsonl(composed_path)
    stats_rows = read_jsonl(stats_path)
    stats_by_key = {(str(row["organ"]), str(row["observed_subtype"])): row for row in stats_rows}

    map_rows = [_postprocess_decision(row, stats_by_key.get((str(row["organ"]), str(row["observed_subtype"])), {}), config) for row in decision_rows]
    map_rows = _resolve_subtype_cycles(map_rows)
    _inject_normal_subtype(composed_rows, map_rows)
    training_vocab = _build_clean_vocab(map_rows)
    review_rows = [row for row in map_rows if row["review_required"]]
    target_rows = _materialize_unique_text_targets(composed_rows, map_rows)

    write_jsonl(config.output_dir / "tag_consolidation_map_v3_clean.jsonl", map_rows)
    write_json(config.output_dir / "training_vocab_v3_clean.json", training_vocab)
    _write_yaml(config.output_dir / "training_vocab_v3_clean.yaml", training_vocab)
    _write_review_csv(config.output_dir / "review_queue_v3.csv", review_rows)
    write_jsonl(config.output_dir / "semantic_training_targets_v3.jsonl", target_rows)
    ParquetTableStore().write_records(config.output_dir / "semantic_training_targets_v3.parquet", target_rows)

    manifest = {
        "dataset_id": config.dataset_id,
        "consolidation_id": config.consolidation_id,
        "postprocess_id": config.postprocess_id,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "inputs": {
            "llm_consolidation_decisions": {"path": str(decisions_path), "sha256": _sha256(decisions_path), "rows": len(decision_rows)},
            "composed_validated_decisions": {"path": str(composed_path), "sha256": _sha256(composed_path), "rows": len(composed_rows)},
            "observed_tag_stats": {"path": str(stats_path), "sha256": _sha256(stats_path), "rows": len(stats_rows)},
        },
        "outputs": {
            "tag_consolidation_map": "tag_consolidation_map_v3_clean.jsonl",
            "training_vocab_json": "training_vocab_v3_clean.json",
            "training_vocab_yaml": "training_vocab_v3_clean.yaml",
            "review_queue": "review_queue_v3.csv",
            "semantic_training_targets_jsonl": "semantic_training_targets_v3.jsonl",
            "semantic_training_targets_parquet": "semantic_training_targets_v3.parquet",
            "report": "reports/deterministic_postprocessing_v3.md",
        },
        "summary": _summarize(map_rows, target_rows, training_vocab),
        "raw_config": config.raw,
    }
    write_json(config.output_dir / "manifest.json", manifest)
    _write_report(config.output_dir / "reports" / "deterministic_postprocessing_v3.md", manifest, map_rows, training_vocab)
    return manifest


def _postprocess_decision(row: dict[str, Any], stats: dict[str, Any], config: PostprocessConfig) -> dict[str, Any]:
    organ = str(row["organ"])
    observed = str(row["observed_subtype"])
    count = int(row.get("unique_text_count") or stats.get("unique_text_count") or 0)
    frequency_tier = str(row.get("frequency_tier") or stats.get("frequency_tier") or "unknown")
    flags: list[str] = []
    repair_source = "llm_valid"

    if row.get("parse_status") != "valid":
        repaired = _repair_invalid_row(row, config)
        if repaired is None:
            repaired = _excluded_payload()
            flags.append("invalid_unrepaired")
        else:
            flags.append("invalid_repaired")
        repair_source = "deterministic_repair"
    else:
        repaired = dict(row)

    use_subtype = bool(repaired.get("use_for_subtype_loss"))
    subtype_label_original = repaired.get("subtype_label")
    subtype_label = None
    subtype_canonicalization = None
    if use_subtype and subtype_label_original:
        subtype_label, subtype_canonicalization = _canonical_subtype_label(organ, str(subtype_label_original))
        if subtype_canonicalization != "unchanged":
            flags.append(f"subtype_canonicalized:{subtype_canonicalization}")

    use_family = bool(repaired.get("use_for_family_loss"))
    family_label = repaired.get("family_label")
    if use_family and family_label not in config.controlled_families:
        inferred = _infer_family(observed)
        if inferred in config.controlled_families:
            family_label = inferred
            flags.append("family_repaired_from_uncontrolled")
        else:
            use_family = False
            family_label = None
            flags.append("family_removed_uncontrolled")

    review_flags = _review_flags(
        organ=organ,
        observed=observed,
        subtype_label=subtype_label,
        family_label=family_label,
        count=count,
        frequency_tier=frequency_tier,
        row=repaired,
        deterministic_flags=flags,
        config=config,
    )

    exclude = bool(repaired.get("exclude_from_loss")) or (not use_subtype and not use_family)
    if exclude:
        use_subtype = False
        use_family = False
        subtype_label = None
        family_label = None

    return {
        "organ": organ,
        "observed_subtype": observed,
        "unique_text_count": count,
        "frequency_tier": frequency_tier,
        "parse_status": row.get("parse_status"),
        "repair_source": repair_source,
        "exclude_from_loss": exclude,
        "use_for_subtype_loss": use_subtype,
        "subtype_mode": repaired.get("subtype_mode", "no_subtype") if use_subtype else "no_subtype",
        "subtype_label_original": subtype_label_original,
        "subtype_label": subtype_label,
        "subtype_loss_weight": float(repaired.get("subtype_loss_weight") or 0.0) if use_subtype else 0.0,
        "use_for_family_loss": use_family,
        "family_label": family_label if use_family else None,
        "family_loss_weight": float(repaired.get("family_loss_weight") or 0.0) if use_family else 0.0,
        "merge_relation": repaired.get("merge_relation", "not_applicable"),
        "review_required": bool(review_flags),
        "review_flags": review_flags,
        "deterministic_flags": flags,
        "llm_rationale": row.get("rationale", ""),
        "validation_error": row.get("validation_error"),
    }


def _repair_invalid_row(row: dict[str, Any], config: PostprocessConfig) -> dict[str, Any] | None:
    observed = str(row["observed_subtype"])
    family = _infer_family(observed)
    if family not in config.controlled_families:
        return None
    return {
        "use_for_subtype_loss": False,
        "subtype_mode": "no_subtype",
        "subtype_label": None,
        "subtype_loss_weight": 0.0,
        "use_for_family_loss": True,
        "family_label": family,
        "family_loss_weight": 0.3,
        "exclude_from_loss": False,
        "merge_relation": "not_applicable",
        "needs_human_review": True,
        "rationale": "Deterministically repaired invalid LLM decision to conservative family-only target.",
    }


def _excluded_payload() -> dict[str, Any]:
    return {
        "use_for_subtype_loss": False,
        "subtype_mode": "no_subtype",
        "subtype_label": None,
        "subtype_loss_weight": 0.0,
        "use_for_family_loss": False,
        "family_label": None,
        "family_loss_weight": 0.0,
        "exclude_from_loss": True,
        "merge_relation": "not_applicable",
        "needs_human_review": True,
        "rationale": "Invalid LLM decision could not be deterministically repaired.",
    }


def _canonical_subtype_label(organ: str, label: str) -> tuple[str, str]:
    label = _slug(label)
    prefix = _organ_prefix(organ)
    alias_key = (prefix, label)
    if alias_key in _SUBTYPE_ALIASES:
        return _SUBTYPE_ALIASES[alias_key], "alias"
    if label.startswith(prefix + "_"):
        return label, "unchanged"
    singular_alias = {
        "kidney": "kidneys",
        "gallstone": "gallbladder_gallstones",
        "gallstones": "gallbladder_gallstones",
    }
    if label in singular_alias:
        value = singular_alias[label]
        if value.startswith(prefix + "_"):
            return value, "alias"
    return f"{prefix}_{label}", "prefixed"


def _organ_prefix(organ: str) -> str:
    return {
        "Adrenal glands": "adrenal",
        "Small bowel": "small_bowel",
        "Urinary bladder": "urinary_bladder",
    }.get(organ, _slug(organ))


def _infer_family(label: str) -> str:
    value = _slug(label)
    checks = [
        (("normal", "unremarkable"), "normal"),
        (("absent", "surgical_absence", "postop_absent"), "absent_postop"),
        (("fistula", "fistulous", "sinus_tract"), "fistula_or_sinus_tract"),
        (("hernia", "herniation", "prolapse"), "hernia_or_prolapse"),
        (("atrophy", "atrophic", "fatty_atrophy", "fatty_infiltration"), "atrophy_or_fatty_change"),
        (("laceration", "trauma", "injury", "hematoma", "hemorrhage"), "trauma_or_injury"),
        (("stone", "calculus", "calcification", "calcified", "gallstones"), "stone_or_calcification"),
        (("cyst", "cystic", "hypodensity", "hypodensities", "fluid_lesion"), "cystic_or_fluid_lesion"),
        (("fluid_collection", "collection", "abscess", "biloma"), "fluid_or_collection"),
        (("mass", "metastasis", "malignan", "tumor", "carcinoma", "implant"), "mass_or_malignancy"),
        (("inflam", "itis", "stranding", "edema", "hyperemia", "phlegmon"), "inflammation"),
        (("wall_thickening", "thickening", "urothelial"), "wall_thickening"),
        (("dilatation", "dilation", "dilated", "distension", "duct_prominence"), "ductal_or_luminal_dilatation"),
        (("obstruction", "narrowing", "stricture", "ileus", "volv", "intussusception"), "obstruction"),
        (("air", "gas", "pneumobilia", "pneumatosis"), "gas_or_air"),
        (("vascular", "varices", "aneurysm", "thrombosis", "stenosis", "shunt"), "vascular"),
        (("post", "surg", "device", "stent", "tube", "catheter", "clip", "anastomosis"), "postoperative_or_device"),
        (("variant", "malrotation", "horseshoe", "divisum", "accessory", "splenule"), "anatomic_variant"),
        (("enlarg", "size", "morphology", "contour", "diminutive", "deform"), "size_or_morphology"),
        (("limited", "artifact", "not_well_visualized", "poorly_visualized", "incompletely"), "limited_assessment"),
        (("indeterminate", "possible", "ill_defined", "or_"), "ambiguous_or_indeterminate"),
    ]
    for needles, family in checks:
        if any(needle in value for needle in needles):
            return family
    return "other_abnormal"


def _review_flags(
    *,
    organ: str,
    observed: str,
    subtype_label: str | None,
    family_label: str | None,
    count: int,
    frequency_tier: str,
    row: dict[str, Any],
    deterministic_flags: list[str],
    config: PostprocessConfig,
) -> list[str]:
    flags: list[str] = []
    if row.get("needs_human_review"):
        flags.append("llm_requested_review")
    if row.get("parse_status") != "valid" or "invalid_repaired" in deterministic_flags:
        flags.append("invalid_or_repaired_llm_output")
    if row.get("merge_relation") == "parent_child":
        flags.append("parent_child_subtype_merge")
    if row.get("use_for_subtype_loss") and count < config.min_subtype_count_without_review:
        flags.append("low_count_subtype_label")
    if family_label == "other_abnormal":
        flags.append("other_abnormal_family")
    text = " ".join(value for value in [observed, subtype_label or ""] if value)
    if any(term in text for term in _AMBIGUITY_TERMS):
        flags.append("ambiguous_label_text")
    if any(flag.startswith("subtype_canonicalized:alias") for flag in deterministic_flags):
        flags.append("nontrivial_subtype_alias")
    if config.review_very_rare_family_only and frequency_tier == "very_rare" and not _is_low_risk_family(row, family_label):
        flags.append("very_rare_non_low_risk_label")
    if organ == "Colon" and family_label == "anatomic_variant" and observed == "colon_diverticulosis":
        flags.append("suspicious_family_assignment")
    return sorted(set(flags))


def _is_low_risk_family(row: dict[str, Any], family_label: str | None) -> bool:
    return family_label in {"normal", "absent_postop", "postoperative_or_device", "limited_assessment"}


def _inject_normal_subtype(composed_rows: list[dict[str, Any]], map_rows: list[dict[str, Any]]) -> None:
    """For texts where the LLM said normality=normal but no subtype was assigned,
    inject {organ}_normal in-place. Catches LLM-tagged normal findings that
    weren't covered by the label_derived pipeline path."""
    norm_text_to_normality: dict[tuple[str, str], str] = {}
    for row in composed_rows:
        key = (str(row["organ"]), str(row["normalized_text"]))
        norm_text_to_normality[key] = str(row.get("normality") or "")

    for row in map_rows:
        if row["use_for_subtype_loss"] or row["exclude_from_loss"]:
            continue
        organ = str(row["organ"])
        observed = str(row["observed_subtype"])
        key = (organ, observed)
        normality = norm_text_to_normality.get(key, "")
        if normality == "normal":
            prefix = _organ_prefix(organ)
            row["use_for_subtype_loss"] = True
            row["subtype_mode"] = "direct"
            row["subtype_label"] = f"{prefix}_normal"
            row["subtype_loss_weight"] = 1.0
            row["use_for_family_loss"] = True
            row["family_label"] = "normal"
            row["family_loss_weight"] = 1.0
            row["review_flags"] = sorted(set(row.get("review_flags", []) + ["normal_injected"]))


def _resolve_subtype_cycles(map_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect and break A→B / B→A circular merge pairs.

    When the LLM concurrently decides that observed label A should merge into
    canonical B and observed label B should merge into canonical A, both labels
    survive as separate vocabulary entries. We resolve each cycle by redirecting
    the lower-count label to the higher-count one.
    """
    # observed (organ, obs) → canonical subtype_label
    obs_to_canonical: dict[tuple[str, str], str] = {}
    obs_count: dict[tuple[str, str], int] = {}
    for row in map_rows:
        if row.get("use_for_subtype_loss") and row.get("subtype_label"):
            key = (str(row["organ"]), str(row["observed_subtype"]))
            obs_to_canonical[key] = str(row["subtype_label"])
            obs_count[key] = int(row.get("unique_text_count") or 0)

    # redirects: (organ, loser_canonical) → winner_canonical
    redirects: dict[tuple[str, str], str] = {}
    seen: set[frozenset[str]] = set()
    for (organ, obs_a), canonical_b in obs_to_canonical.items():
        if canonical_b == obs_a:
            continue
        key_b = (organ, canonical_b)
        if key_b not in obs_to_canonical:
            continue
        canonical_a_via_b = obs_to_canonical[key_b]
        if canonical_a_via_b != obs_a:
            continue
        pair = frozenset([obs_a, canonical_b])
        if pair in seen:
            continue
        seen.add(pair)
        count_a = obs_count.get((organ, obs_a), 0)
        count_b = obs_count.get((organ, canonical_b), 0)
        winner, loser = (obs_a, canonical_b) if count_a >= count_b else (canonical_b, obs_a)
        redirects[(organ, loser)] = winner

    if not redirects:
        return map_rows

    updated: list[dict[str, Any]] = []
    for row in map_rows:
        if row.get("use_for_subtype_loss") and row.get("subtype_label"):
            organ = str(row["organ"])
            canonical = str(row["subtype_label"])
            if (organ, canonical) in redirects:
                row = dict(row)
                row["subtype_label"] = redirects[(organ, canonical)]
                row["review_flags"] = sorted(set(list(row.get("review_flags") or []) + ["cycle_resolved"]))
                row["review_required"] = True
        updated.append(row)
    return updated


def _build_clean_vocab(map_rows: list[dict[str, Any]]) -> dict[str, Any]:
    subtype: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    family: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in map_rows:
        if row["use_for_subtype_loss"] and row["subtype_label"]:
            _add_vocab(subtype, row, label=str(row["subtype_label"]), weight_field="subtype_loss_weight", source_field="subtype_mode")
        if row["use_for_family_loss"] and row["family_label"]:
            _add_vocab(family, row, label=str(row["family_label"]), weight_field="family_loss_weight", source_field=None)
    return {
        "subtype_labels_by_organ": _finalize_vocab(subtype),
        "family_labels_by_organ": _finalize_vocab(family),
    }


def _add_vocab(
    target: dict[str, dict[str, dict[str, Any]]],
    row: dict[str, Any],
    *,
    label: str,
    weight_field: str,
    source_field: str | None,
) -> None:
    organ = str(row["organ"])
    entry = target[organ].setdefault(
        label,
        {
            "label": label,
            "organ": organ,
            "source_observed_subtypes": [],
            "total_unique_text_count": 0,
            "min_loss_weight": 1.0,
            "max_loss_weight": 0.0,
            "review_required": False,
            "review_flags": [],
            "source_modes": [],
        },
    )
    entry["source_observed_subtypes"].append(str(row["observed_subtype"]))
    entry["total_unique_text_count"] += int(row["unique_text_count"])
    weight = float(row[weight_field])
    entry["min_loss_weight"] = min(float(entry["min_loss_weight"]), weight)
    entry["max_loss_weight"] = max(float(entry["max_loss_weight"]), weight)
    entry["review_required"] = bool(entry["review_required"] or row["review_required"])
    entry["review_flags"].extend(row["review_flags"])
    if source_field is not None:
        entry["source_modes"].append(str(row[source_field]))


def _finalize_vocab(target: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    finalized: dict[str, list[dict[str, Any]]] = {}
    for organ, labels in target.items():
        finalized[organ] = []
        for entry in labels.values():
            entry["source_observed_subtypes"] = sorted(set(entry["source_observed_subtypes"]))
            entry["review_flags"] = sorted(set(entry["review_flags"]))
            entry["source_modes"] = sorted(set(entry["source_modes"]))
            finalized[organ].append(entry)
        finalized[organ].sort(key=lambda item: (-int(item["total_unique_text_count"]), str(item["label"])))
    return dict(sorted(finalized.items()))


def _materialize_unique_text_targets(composed_rows: list[dict[str, Any]], map_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    map_by_key = {(str(row["organ"]), str(row["observed_subtype"])): row for row in map_rows}
    targets: list[dict[str, Any]] = []
    for row in composed_rows:
        organ = str(row["organ"])
        source_tags = _source_tags(row)
        subtype_targets: dict[str, float] = {}
        family_targets: dict[str, float] = {}
        review_flags: list[str] = []
        excluded_tags: list[str] = []
        provenance: list[dict[str, Any]] = []
        for tag in source_tags:
            mapped = map_by_key.get((organ, tag))
            if mapped is None:
                review_flags.append(f"missing_consolidation_map:{tag}")
                continue
            if mapped["exclude_from_loss"]:
                excluded_tags.append(tag)
            if mapped["use_for_subtype_loss"] and mapped["subtype_label"]:
                subtype_targets[str(mapped["subtype_label"])] = max(
                    subtype_targets.get(str(mapped["subtype_label"]), 0.0),
                    float(mapped["subtype_loss_weight"]),
                )
            if mapped["use_for_family_loss"] and mapped["family_label"]:
                family_targets[str(mapped["family_label"])] = max(
                    family_targets.get(str(mapped["family_label"]), 0.0),
                    float(mapped["family_loss_weight"]),
                )
            review_flags.extend(str(flag) for flag in mapped["review_flags"])
            provenance.append(
                {
                    "observed_subtype": tag,
                    "subtype_label": mapped["subtype_label"],
                    "family_label": mapped["family_label"],
                    "review_required": mapped["review_required"],
                }
            )
        confidence = float(row.get("confidence") or 0.0)
        status = str(row.get("decision_status"))
        status_weight = 1.0 if status == "accepted" else 0.5 if status == "accepted_provisional" else 0.0
        review_required = bool(review_flags)
        targets.append(
            {
                "organ": organ,
                "raw_text": row.get("raw_text"),
                "normalized_text": row.get("normalized_text"),
                "normality": row.get("normality"),
                "polarity": row.get("polarity"),
                "certainty": row.get("certainty"),
                "subtype_targets": dict(sorted(subtype_targets.items())),
                "family_targets": dict(sorted(family_targets.items())),
                "source_observed_subtypes": source_tags,
                "excluded_observed_subtypes": sorted(set(excluded_tags)),
                "confidence_weight": confidence * status_weight,
                "review_required": review_required,
                "review_flags": sorted(set(review_flags)),
                "decision_status": status,
                "decision_source": row.get("decision_source"),
                "ontology_version": row.get("ontology_version"),
                "target_provenance": provenance,
            }
        )
    return targets


def _source_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if row.get("primary_subtype"):
        tags.append(str(row["primary_subtype"]))
    for subtype in row.get("secondary_subtypes") or []:
        if subtype:
            tags.append(str(subtype))
    return sorted(set(tags))


def _write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "organ",
        "observed_subtype",
        "unique_text_count",
        "frequency_tier",
        "subtype_label",
        "family_label",
        "review_flags",
        "validation_error",
        "llm_rationale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_report(path: Path, manifest: dict[str, Any], map_rows: list[dict[str, Any]], vocab: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# Deterministic Postprocessing V3",
        "",
        f"- dataset: `{manifest['dataset_id']}`",
        f"- consolidation: `{manifest['consolidation_id']}`",
        f"- postprocess: `{manifest['postprocess_id']}`",
        f"- map rows: `{summary['map_rows']}`",
        f"- unique-text targets: `{summary['target_rows']}`",
        f"- review rows: `{summary['review_rows']}`",
        f"- excluded map rows: `{summary['excluded_map_rows']}`",
        f"- subtype labels: `{summary['subtype_label_count']}`",
        f"- organ-family labels: `{summary['family_label_count']}`",
        "",
        "## Decision Modes",
        "",
    ]
    for key, value in sorted(summary["decision_mode_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Repair Sources", ""])
    for key, value in sorted(summary["repair_source_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Top Review Flags", ""])
    for key, value in summary["top_review_flags"]:
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Subtype Labels By Organ", ""])
    for organ, labels in vocab["subtype_labels_by_organ"].items():
        lines.append(f"- `{organ}`: {len(labels)}")
    lines.extend(["", "## Family Labels By Organ", ""])
    for organ, labels in vocab["family_labels_by_organ"].items():
        lines.append(f"- `{organ}`: {len(labels)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize(map_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], vocab: dict[str, Any]) -> dict[str, Any]:
    decision_modes: dict[str, int] = defaultdict(int)
    repair_sources: dict[str, int] = defaultdict(int)
    review_flags: dict[str, int] = defaultdict(int)
    for row in map_rows:
        mode = "exclude" if row["exclude_from_loss"] else "subtype_and_family" if row["use_for_subtype_loss"] and row["use_for_family_loss"] else "subtype_only" if row["use_for_subtype_loss"] else "family_only"
        decision_modes[mode] += 1
        repair_sources[str(row["repair_source"])] += 1
        for flag in row["review_flags"]:
            review_flags[str(flag)] += 1
    subtype_count = sum(len(labels) for labels in vocab["subtype_labels_by_organ"].values())
    family_count = sum(len(labels) for labels in vocab["family_labels_by_organ"].values())
    return {
        "map_rows": len(map_rows),
        "target_rows": len(target_rows),
        "review_rows": sum(1 for row in map_rows if row["review_required"]),
        "excluded_map_rows": sum(1 for row in map_rows if row["exclude_from_loss"]),
        "decision_mode_counts": dict(sorted(decision_modes.items())),
        "repair_source_counts": dict(sorted(repair_sources.items())),
        "top_review_flags": sorted(review_flags.items(), key=lambda item: (-item[1], item[0]))[:20],
        "subtype_label_count": subtype_count,
        "family_label_count": family_count,
    }


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def _default_families() -> list[str]:
    return [
        "normal",
        "absent_postop",
        "focal_lesion",
        "mass_or_malignancy",
        "inflammation",
        "wall_thickening",
        "ductal_or_luminal_dilatation",
        "obstruction",
        "stone_or_calcification",
        "cystic_or_fluid_lesion",
        "fluid_or_collection",
        "vascular",
        "gas_or_air",
        "postoperative_or_device",
        "anatomic_variant",
        "size_or_morphology",
        "limited_assessment",
        "ambiguous_or_indeterminate",
        "trauma_or_injury",
        "fistula_or_sinus_tract",
        "hernia_or_prolapse",
        "atrophy_or_fatty_change",
        "other_abnormal",
    ]


_SUBTYPE_ALIASES = {
    ("gallbladder", "gallstones"): "gallbladder_gallstones",
    ("gallbladder", "gallstone"): "gallbladder_gallstones",
    ("kidneys", "cystic_lesion"): "kidneys_cystic_lesion",
    ("kidneys", "kidney_stone"): "kidneys_stone",
    ("kidneys", "atrophic"): "kidneys_atrophic",
    ("liver", "hepatomegaly"): "liver_hepatomegaly",
    ("pancreas", "mass"): "pancreas_mass",
    ("spleen", "splenomegaly"): "spleen_splenomegaly",
}

_AMBIGUITY_TERMS = {
    "possible",
    "indeterminate",
    "ill_defined",
    "or_",
    "_or",
    "_or_",
    "involvement",
    "encasement",
    "abuts",
    "abutment",
}
