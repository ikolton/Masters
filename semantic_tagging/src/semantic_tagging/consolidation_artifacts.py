from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .table_store import read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class ConsolidationConfig:
    dataset_id: str
    consolidation_id: str
    output_root: Path
    base_run_id: str
    base_decision_file: str
    organ_overrides: dict[str, dict[str, str]]
    min_direct_count: int
    min_review_count: int
    examples_per_tag: int
    candidate_label_limit: int
    include_normal_tags: bool
    raw: dict[str, Any]

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.dataset_id / "consolidation" / self.consolidation_id

    def run_dir(self, run_id: str) -> Path:
        return self.output_root / self.dataset_id / run_id


def load_consolidation_config(path: Path) -> ConsolidationConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = payload["source_runs"]["base"]
    consolidation = payload.get("consolidation", {})
    return ConsolidationConfig(
        dataset_id=str(payload["project"]["dataset_id"]),
        consolidation_id=str(payload["project"]["consolidation_id"]),
        output_root=Path(payload["paths"]["output_root"]),
        base_run_id=str(base["run_id"]),
        base_decision_file=str(base.get("decision_file", "validated_decisions.partial.jsonl")),
        organ_overrides=dict(payload["source_runs"].get("organ_overrides", {})),
        min_direct_count=int(consolidation.get("min_direct_count", 100)),
        min_review_count=int(consolidation.get("min_review_count", 20)),
        examples_per_tag=int(consolidation.get("examples_per_tag", 8)),
        candidate_label_limit=int(consolidation.get("candidate_label_limit", 40)),
        include_normal_tags=bool(consolidation.get("include_normal_tags", True)),
        raw=payload,
    )


def build_consolidation_artifacts(config: ConsolidationConfig, *, config_path: Path) -> dict[str, Any]:
    decisions, source_manifest = _load_composed_decisions(config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "composed_validated_decisions.jsonl", decisions)
    tag_stats = _build_tag_stats(decisions, config=config)
    write_jsonl(output_dir / "observed_tag_stats.jsonl", tag_stats)
    _write_tag_stats_csv(output_dir / "observed_tag_stats.csv", tag_stats)

    items = _build_llm_items(tag_stats, config=config)
    write_jsonl(output_dir / "llm_consolidation_items.jsonl", items)

    manifest = {
        "dataset_id": config.dataset_id,
        "consolidation_id": config.consolidation_id,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_manifest": source_manifest,
        "decision_count": len(decisions),
        "observed_tag_count": len(tag_stats),
        "llm_item_count": len(items),
        "artifacts": {
            "composed_decisions": "composed_validated_decisions.jsonl",
            "observed_tag_stats": "observed_tag_stats.jsonl",
            "observed_tag_stats_csv": "observed_tag_stats.csv",
            "llm_consolidation_items": "llm_consolidation_items.jsonl",
            "report": "reports/consolidation_input_report.md",
        },
        "raw_config": config.raw,
    }
    write_json(output_dir / "manifest.json", manifest)
    _write_report(output_dir / "reports" / "consolidation_input_report.md", tag_stats, manifest)
    return manifest


def _load_composed_decisions(config: ConsolidationConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_path = config.run_dir(config.base_run_id) / config.base_decision_file
    base_rows = read_jsonl(base_path)
    selected_by_key: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["organ"]), str(row["normalized_text"])): row for row in base_rows
    }
    source_manifest = [
        {
            "role": "base",
            "run_id": config.base_run_id,
            "path": str(base_path),
            "sha256": _sha256(base_path),
            "rows": len(base_rows),
        }
    ]

    for organ, override in sorted(config.organ_overrides.items()):
        run_id = str(override["run_id"])
        decision_file = str(override.get("decision_file", "validated_decisions.partial.jsonl"))
        path = config.run_dir(run_id) / decision_file
        rows = [row for row in read_jsonl(path) if row.get("organ") == organ]
        selected_by_key = {key: row for key, row in selected_by_key.items() if key[0] != organ}
        for row in rows:
            selected_by_key[(str(row["organ"]), str(row["normalized_text"]))] = row
        source_manifest.append(
            {
                "role": "organ_override",
                "organ": organ,
                "run_id": run_id,
                "path": str(path),
                "sha256": _sha256(path),
                "rows": len(rows),
            }
        )

    decisions = sorted(selected_by_key.values(), key=lambda row: (str(row["organ"]), str(row["normalized_text"])))
    return decisions, source_manifest


def _build_tag_stats(decisions: list[dict[str, Any]], *, config: ConsolidationConfig) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    organ_label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in decisions:
        organ = str(row["organ"])
        tags = _row_tags(row)
        for subtype, role in tags:
            if not config.include_normal_tags and subtype.endswith("_normal"):
                continue
            organ_label_counts[organ][subtype] += 1
            key = (organ, subtype)
            bucket = buckets.setdefault(
                key,
                {
                    "organ": organ,
                    "observed_subtype": subtype,
                    "unique_text_count": 0,
                    "primary_count": 0,
                    "secondary_count": 0,
                    "normality_counts": Counter(),
                    "polarity_counts": Counter(),
                    "certainty_counts": Counter(),
                    "decision_status_counts": Counter(),
                    "decision_source_counts": Counter(),
                    "examples": [],
                },
            )
            bucket["unique_text_count"] += 1
            bucket[f"{role}_count"] += 1
            bucket["normality_counts"][str(row.get("normality"))] += 1
            bucket["polarity_counts"][str(row.get("polarity"))] += 1
            bucket["certainty_counts"][str(row.get("certainty"))] += 1
            bucket["decision_status_counts"][str(row.get("decision_status"))] += 1
            bucket["decision_source_counts"][str(row.get("decision_source"))] += 1
            if len(bucket["examples"]) < config.examples_per_tag:
                bucket["examples"].append(
                    {
                        "raw_text": row.get("raw_text"),
                        "normality": row.get("normality"),
                        "polarity": row.get("polarity"),
                        "certainty": row.get("certainty"),
                        "decision_status": row.get("decision_status"),
                        "confidence": row.get("confidence"),
                    }
                )

    stats: list[dict[str, Any]] = []
    for bucket in buckets.values():
        total = int(bucket["unique_text_count"])
        bucket["frequency_tier"] = _frequency_tier(total, config=config)
        for key in ("normality_counts", "polarity_counts", "certainty_counts", "decision_status_counts", "decision_source_counts"):
            bucket[key] = dict(bucket[key])
        stats.append(bucket)
    stats.sort(key=lambda row: (str(row["organ"]), -int(row["unique_text_count"]), str(row["observed_subtype"])))
    return stats


def _build_llm_items(tag_stats: list[dict[str, Any]], *, config: ConsolidationConfig) -> list[dict[str, Any]]:
    by_organ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tag_stats:
        by_organ[str(row["organ"])].append(row)

    items: list[dict[str, Any]] = []
    for organ, rows in sorted(by_organ.items()):
        candidate_labels = [
            {
                "observed_subtype": row["observed_subtype"],
                "unique_text_count": row["unique_text_count"],
                "frequency_tier": row["frequency_tier"],
            }
            for row in sorted(rows, key=lambda item: -int(item["unique_text_count"]))[: config.candidate_label_limit]
        ]
        for row in rows:
            items.append(
                {
                    "request_id": f"{_slug(organ)}::{row['observed_subtype']}",
                    "organ": organ,
                    "observed_subtype": row["observed_subtype"],
                    "tag_stats": row,
                    "candidate_training_labels_for_organ": candidate_labels,
                    "allowed_modes": ["direct", "merged", "family_only", "exclude"],
                    "instructions": (
                        "Choose how this observed semantic subtype should be used for diagnostic-loss training. "
                        "Prefer direct for frequent clean labels, merged for wording variants, family_only for real but rare or overly specific labels, "
                        "and exclude for adjacent-organ leakage, parsing artifacts, or labels not medically grounded in the examples."
                    ),
                }
            )
    return items


def _row_tags(row: dict[str, Any]) -> list[tuple[str, str]]:
    tags: list[tuple[str, str]] = []
    primary = row.get("primary_subtype")
    if primary:
        tags.append((str(primary), "primary"))
    for subtype in row.get("secondary_subtypes") or []:
        if subtype:
            tags.append((str(subtype), "secondary"))
    return tags


def _frequency_tier(count: int, *, config: ConsolidationConfig) -> str:
    if count >= config.min_direct_count:
        return "frequent"
    if count >= config.min_review_count:
        return "review"
    if count >= 5:
        return "rare"
    return "very_rare"


def _write_tag_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "organ",
        "observed_subtype",
        "unique_text_count",
        "primary_count",
        "secondary_count",
        "frequency_tier",
        "normality_counts",
        "polarity_counts",
        "certainty_counts",
        "decision_status_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_report(path: Path, tag_stats: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    by_organ: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tier_counts = Counter()
    for row in tag_stats:
        by_organ[str(row["organ"])].append(row)
        tier_counts[str(row["frequency_tier"])] += 1

    lines = [
        "# Consolidation Input Report",
        "",
        f"- dataset: `{manifest['dataset_id']}`",
        f"- consolidation: `{manifest['consolidation_id']}`",
        f"- composed decisions: `{manifest['decision_count']}`",
        f"- observed subtype labels: `{manifest['observed_tag_count']}`",
        f"- LLM consolidation items: `{manifest['llm_item_count']}`",
        "",
        "## Frequency Tiers",
        "",
    ]
    for tier in ("frequent", "review", "rare", "very_rare"):
        lines.append(f"- `{tier}`: {tier_counts[tier]}")
    lines.extend(["", "## Organs", ""])
    for organ, rows in sorted(by_organ.items()):
        top = sorted(rows, key=lambda row: -int(row["unique_text_count"]))[:12]
        lines.append(f"### {organ}")
        lines.append("")
        lines.append(f"- observed labels: `{len(rows)}`")
        lines.append("- top labels:")
        for row in top:
            lines.append(f"  - `{row['observed_subtype']}`: {row['unique_text_count']} ({row['frequency_tier']})")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
