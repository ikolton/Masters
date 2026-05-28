#!/usr/bin/env python3
"""Build label_based_normal_v1 dataset variant.

Replaces every organ finding where labels[organ] == 0 with "unremarkable".
Abnormal findings (labels[organ] == 1) are kept verbatim.

Follows the same materialization pattern as build_normalized_dataset_variants.py:
- Writes identical train/combined.json and val/combined.json (both contain all records).
- Symlinks all study directories from the source split into the target split.
- Writes variant.json and README.md for reproducibility.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path("/net/storage/pr3/plgrid/plggjmiag/Merlin_converted")
OUTPUT_ROOT = Path("/net/scratch/hscra/plgrid/plgikolton/Magisterka/normalized_datasets")
VARIANT_NAME = "label_based_normal_v1"
VARIANT_FAMILY = "labels"

VARIANT_META = {
    "summary": (
        "Replace all organ findings where the binary label == 0 (normal) with 'unremarkable'. "
        "Abnormal findings (label == 1) are preserved verbatim. "
        "Uses ground-truth annotation labels rather than text-pattern heuristics."
    ),
    "intended_use": ["encoder"],
    "strategy": "label_guided_rewrite",
    "text_transforms": ["label_based_normal_collapse"],
    "annotation_transforms": [],
    "risks": [
        "Collapses all normal variants to a single token — loses fine-grained normal-range wording.",
        "14% of label=1 cases mention 'normal' for one side before describing the abnormality; "
        "these are correctly preserved since label=1.",
        "Makes decoder targets uninformative for normal organs if used for generation training.",
    ],
}


def main() -> None:
    variant_root = OUTPUT_ROOT / VARIANT_FAMILY / VARIANT_NAME
    legacy_path = OUTPUT_ROOT / VARIANT_NAME

    records = _load_records(SOURCE_ROOT)
    variant_records, stats = _apply_label_normalization(records)
    _materialize_variant(SOURCE_ROOT, variant_root, variant_records)
    _write_variant_metadata(variant_root, SOURCE_ROOT, stats)
    _ensure_legacy_symlink(legacy_path, variant_root)

    print(f"Built {VARIANT_NAME} at {variant_root}")
    print(f"Changed records: {stats['changed_record_count']}")
    print(f"Changed findings: {stats['total_findings_changed']}")
    print(json.dumps(stats["changed_by_organ"], indent=2))


def _load_records(source_root: Path) -> list[dict[str, Any]]:
    path = source_root / "train" / "combined.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return payload


def _apply_label_normalization(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    changed_records = 0
    total_changed = 0
    changed_by_organ: Counter[str] = Counter()

    variant_records: list[dict[str, Any]] = []
    for record in records:
        findings = record.get("findings")
        labels = record.get("labels")
        if not isinstance(findings, dict) or not isinstance(labels, dict):
            variant_records.append(record)
            continue

        new_findings = dict(findings)
        record_changed = False
        for organ, label_val in labels.items():
            if label_val != 0:
                continue
            finding = findings.get(organ)
            if not isinstance(finding, str):
                continue
            if finding == "unremarkable":
                continue
            new_findings[organ] = "unremarkable"
            record_changed = True
            total_changed += 1
            changed_by_organ[organ] += 1

        if record_changed:
            changed_records += 1
            updated = dict(record)
            updated["findings"] = new_findings
            variant_records.append(updated)
        else:
            variant_records.append(record)

    stats = {
        "changed_record_count": changed_records,
        "total_findings_changed": total_changed,
        "changed_by_organ": dict(sorted(changed_by_organ.items())),
    }
    return variant_records, stats


def _materialize_variant(
    source_root: Path,
    target_root: Path,
    records: list[dict[str, Any]],
) -> None:
    train_target = target_root / "train"
    val_target = target_root / "val"
    train_target.mkdir(parents=True, exist_ok=True)
    val_target.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(records, indent=2, sort_keys=True)
    (train_target / "combined.json").write_text(payload, encoding="utf-8")
    (val_target / "combined.json").write_text(payload, encoding="utf-8")

    _sync_split_symlinks(source_root / "train", train_target)
    _sync_split_symlinks(source_root / "val", val_target)


def _sync_split_symlinks(source_split: Path, target_split: Path) -> None:
    for entry in source_split.iterdir():
        if not entry.is_dir():
            continue
        target = target_split / entry.name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(entry.resolve(), target)


def _ensure_legacy_symlink(legacy_path: Path, target_path: Path) -> None:
    if legacy_path.is_symlink():
        if legacy_path.resolve() == target_path.resolve():
            return
        legacy_path.unlink()
    elif legacy_path.exists():
        return
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target_path.resolve(), legacy_path)


def _write_variant_metadata(
    variant_root: Path,
    source_root: Path,
    stats: dict[str, Any],
) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    variant_json = {
        "name": VARIANT_NAME,
        "family": VARIANT_FAMILY,
        "summary": VARIANT_META["summary"],
        "source_root": str(source_root),
        "canonical_dataset_root": str(variant_root),
        "builder_script": str(Path(__file__).resolve()),
        "built_at_utc": now,
        "strategy": VARIANT_META["strategy"],
        "intended_use": VARIANT_META["intended_use"],
        "text_transforms": VARIANT_META["text_transforms"],
        "annotation_transforms": VARIANT_META["annotation_transforms"],
        "stats": stats,
    }
    (variant_root / "variant.json").write_text(
        json.dumps(variant_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    readme_lines = [
        f"# {VARIANT_NAME}",
        "",
        f"Family: `{VARIANT_FAMILY}`",
        "",
        VARIANT_META["summary"],
        "",
        "Intended use:",
        *[f"- `{u}`" for u in VARIANT_META["intended_use"]],
        "",
        "Text transforms:",
        *[f"- `{t}`" for t in VARIANT_META["text_transforms"]],
        "",
        "Known risks:",
        *[f"- {r}" for r in VARIANT_META["risks"]],
        "",
        "Build stats:",
        f"- changed records: {stats['changed_record_count']}",
        f"- changed findings: {stats['total_findings_changed']}",
        "",
        "Changed findings per organ:",
        *[f"- {organ}: {count}" for organ, count in stats["changed_by_organ"].items()],
    ]
    (variant_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    family_root = variant_root.parent
    family_readme = family_root / "README.md"
    if not family_readme.exists():
        family_readme.write_text(
            "# labels\n\nLabel-guided normalization variants that use binary organ annotations "
            "rather than text-pattern heuristics to decide what constitutes a normal finding.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
