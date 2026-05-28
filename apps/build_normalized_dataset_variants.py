#!/usr/bin/env python3
"""Build sidecar dataset variants with structured normalization metadata."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable


DEFAULT_SOURCE_ROOT = "/net/storage/pr3/plgrid/plggjmiag/Merlin_converted"
DEFAULT_OUTPUT_ROOT = "/net/scratch/hscra/plgrid/plgikolton/Magisterka/normalized_datasets"
DEFAULT_ANALYSIS_ROOT = "/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/analysis/normalization_variants"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    family: str
    version: str
    summary: str
    intended_use: tuple[str, ...]
    risks: tuple[str, ...]
    text_transforms: tuple[str, ...]
    annotation_transforms: tuple[str, ...]
    strategy: str
    normalizer: Callable[[str], str]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--analysis-root", default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "minimal_surface_v1",
            "canonical_normals_absent_surface_v2",
            "canonical_normals_v1",
            "canonical_normals_and_absent_v1",
            "canonical_normal_templates_v2",
            "canonical_normal_templates_and_absent_v2",
        ],
        help="Variant names to build.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only build dataset variants, do not run text-space analysis.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    analysis_root = Path(args.analysis_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    records = _load_records(source_root)
    variant_specs = {spec.name: spec for spec in _variant_specs()}

    selected: list[VariantSpec] = []
    for name in args.variants:
        if name not in variant_specs:
            raise ValueError(
                f"Unsupported variant {name!r}; available: {sorted(variant_specs)}"
            )
        selected.append(variant_specs[name])

    _write_root_readme(output_root, selected)
    _ensure_family_readmes(output_root, variant_specs.values())

    manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "analysis_root": str(analysis_root),
        "builder_script": str(Path(__file__).resolve()),
        "built_at_utc": _utc_now(),
        "variants": {},
    }

    baseline_analysis = analysis_root / "baseline_text_distribution.json"
    if not args.skip_analysis and not baseline_analysis.exists():
        _run_distribution_analysis(source_root, baseline_analysis)

    for spec in selected:
        variant_root = output_root / spec.family / spec.name
        variant_records, stats = _apply_variant(records, spec.normalizer)
        _materialize_variant(
            source_root=source_root,
            target_root=variant_root,
            records=variant_records,
        )
        _write_variant_metadata(
            variant_root=variant_root,
            source_root=source_root,
            spec=spec,
            stats=stats,
        )
        _ensure_legacy_symlink(
            legacy_path=output_root / spec.name,
            target_path=variant_root,
        )
        variant_entry: dict[str, Any] = {
            "name": spec.name,
            "family": spec.family,
            "canonical_dataset_root": str(variant_root),
            "legacy_dataset_root": str(output_root / spec.name),
            "stats": stats,
        }
        if not args.skip_analysis:
            analysis_output = analysis_root / f"{spec.name}_text_distribution.json"
            _run_distribution_analysis(variant_root, analysis_output)
            variant_entry["analysis_output"] = str(analysis_output)
        manifest["variants"][spec.name] = variant_entry

    manifest_path = analysis_root / "variants_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(
            name="minimal_surface_v1",
            family="rules",
            version="v1",
            summary="Apply formatting-only cleanup without collapsing semantic text variants.",
            intended_use=("encoder", "decoder"),
            risks=(
                "May produce only modest gains because it does not reduce semantic fragmentation.",
                "Can still change exact-string targets slightly by removing formatting noise.",
            ),
            text_transforms=("surface_cleanup",),
            annotation_transforms=(),
            strategy="rewrite_text",
            normalizer=_minimal_surface_v1,
        ),
        VariantSpec(
            name="canonical_normals_absent_surface_v2",
            family="rules",
            version="v2",
            summary="Apply conservative surface cleanup plus canonical normal and absent/post-op collapse.",
            intended_use=("encoder", "decoder"),
            risks=(
                "Still reduces target diversity for common benign and post-op findings.",
                "More aggressive than formatting-only cleanup, though much milder than template collapse.",
            ),
            text_transforms=("surface_cleanup", "collapse_normals", "collapse_absent"),
            annotation_transforms=(),
            strategy="rewrite_text",
            normalizer=_canonical_normals_absent_surface_v2,
        ),
        VariantSpec(
            name="canonical_normals_v1",
            family="rules",
            version="v1",
            summary="Collapse obvious normal variants to a single canonical string.",
            intended_use=("encoder", "decoder"),
            risks=(
                "Reduces wording diversity for benign findings.",
                "Can hide subtle but clinically meaningful 'normal-like' phrasing if expanded too aggressively.",
            ),
            text_transforms=("surface_cleanup", "collapse_normals"),
            annotation_transforms=(),
            strategy="rewrite_text",
            normalizer=_canonical_normals_v1,
        ),
        VariantSpec(
            name="canonical_normals_and_absent_v1",
            family="rules",
            version="v1",
            summary="Collapse obvious normal and absent/post-op variants to canonical strings.",
            intended_use=("encoder", "decoder"),
            risks=(
                "Collapses post-surgical wording into one bucket.",
                "May remove stylistic evidence useful for some decoder outputs.",
            ),
            text_transforms=("surface_cleanup", "collapse_normals", "collapse_absent"),
            annotation_transforms=(),
            strategy="rewrite_text",
            normalizer=_canonical_normals_and_absent_v1,
        ),
        VariantSpec(
            name="canonical_normal_templates_v2",
            family="templates",
            version="v2",
            summary="Collapse organ-specific normal templates to a minimal canonical normal string.",
            intended_use=("encoder",),
            risks=(
                "Much stronger text-space compression than rule-only variants.",
                "Can make decoder targets unnaturally bland if used directly for generation training.",
            ),
            text_transforms=(
                "surface_cleanup",
                "collapse_normals",
                "collapse_normal_templates",
            ),
            annotation_transforms=(),
            strategy="rewrite_text",
            normalizer=_canonical_normal_templates_v2,
        ),
        VariantSpec(
            name="canonical_normal_templates_and_absent_v2",
            family="templates",
            version="v2",
            summary="Collapse normal templates and absent/post-op templates to minimal canonical strings.",
            intended_use=("encoder",),
            risks=(
                "Strongly compresses target wording.",
                "Highest risk of over-templating decoder targets.",
            ),
            text_transforms=(
                "surface_cleanup",
                "collapse_normals",
                "collapse_normal_templates",
                "collapse_absent_templates",
            ),
            annotation_transforms=(),
            strategy="rewrite_text",
            normalizer=_canonical_normal_templates_and_absent_v2,
        ),
    ]


def _load_records(source_root: Path) -> list[dict[str, Any]]:
    metadata_path = source_root / "train" / "combined.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {metadata_path}")
    return payload


def _apply_variant(
    records: list[dict[str, Any]],
    normalizer: Callable[[str], str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    changed_records = 0
    total_findings_changed = 0
    changed_by_organ: Counter[str] = Counter()
    mapping_counter: Counter[tuple[str, str]] = Counter()
    variant_records: list[dict[str, Any]] = []

    for record in records:
        findings = record.get("findings")
        if not isinstance(findings, dict):
            variant_records.append(record)
            continue
        new_findings = dict(findings)
        record_changed = False
        for organ_name, value in findings.items():
            if not isinstance(value, str):
                continue
            normalized = normalizer(value)
            if normalized != value:
                new_findings[organ_name] = normalized
                record_changed = True
                total_findings_changed += 1
                changed_by_organ[str(organ_name)] += 1
                mapping_counter[(value, normalized)] += 1
        if record_changed:
            changed_records += 1
            updated_record = dict(record)
            updated_record["findings"] = new_findings
            variant_records.append(updated_record)
        else:
            variant_records.append(record)

    stats = {
        "changed_record_count": int(changed_records),
        "total_findings_changed": int(total_findings_changed),
        "changed_by_organ": dict(sorted(changed_by_organ.items())),
        "top_mappings": [
            {"from": source, "to": target, "count": int(count)}
            for (source, target), count in mapping_counter.most_common(40)
        ],
    }
    return variant_records, stats


def _materialize_variant(
    *, source_root: Path, target_root: Path, records: list[dict[str, Any]]
) -> None:
    train_source = source_root / "train"
    val_source = source_root / "val"
    train_target = target_root / "train"
    val_target = target_root / "val"
    train_target.mkdir(parents=True, exist_ok=True)
    val_target.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(records, indent=2, sort_keys=True)
    (train_target / "combined.json").write_text(payload, encoding="utf-8")
    (val_target / "combined.json").write_text(payload, encoding="utf-8")

    _sync_split_symlinks(train_source, train_target)
    _sync_split_symlinks(val_source, val_target)


def _sync_split_symlinks(source_split: Path, target_split: Path) -> None:
    wanted = {entry.name: entry for entry in source_split.iterdir() if entry.is_dir()}
    for name, entry in wanted.items():
        target = target_split / name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(entry.resolve(), target)


def _ensure_legacy_symlink(*, legacy_path: Path, target_path: Path) -> None:
    if legacy_path.is_symlink():
        if legacy_path.resolve() == target_path.resolve():
            return
        legacy_path.unlink()
    elif legacy_path.exists():
        return
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target_path.resolve(), legacy_path)


def _write_root_readme(output_root: Path, selected: list[VariantSpec]) -> None:
    families = sorted({spec.family for spec in selected})
    text = [
        "# Normalized Dataset Variants",
        "",
        "This directory stores sidecar dataset variants used for controlled text normalization experiments.",
        "",
        "Families:",
    ]
    for family in families:
        text.append(f"- `{family}/`: related variants with a shared normalization strategy")
    text.extend(
        [
            "",
            "Design notes:",
            "- `rules/` variants reduce noisy surface variation conservatively.",
            "- `templates/` variants compress text space aggressively and are usually more suitable for encoder-side experiments.",
            "- `grouped/` is reserved for future text-preserving semantic grouping variants with sidecar metadata.",
            "",
            "Each variant folder should contain:",
            "- `README.md`",
            "- `variant.json`",
            "- `train/` and `val/` sidecar split directories",
        ]
    )
    (output_root / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _ensure_family_readmes(output_root: Path, specs: Any) -> None:
    family_notes = {
        "rules": "Conservative rewrite-text variants that collapse obvious noisy surface forms while preserving most clinical wording diversity.",
        "templates": "Aggressive rewrite-text variants that collapse full normal/absent templates into minimal canonical strings.",
        "grouped": "Reserved for future text-preserving semantic grouping variants that add sidecar concept/group metadata.",
        "experimental": "Reserved for unstable or exploratory variants that should not become default baselines without review.",
    }
    for family in sorted({spec.family for spec in specs} | {"grouped", "experimental"}):
        family_root = output_root / family
        family_root.mkdir(parents=True, exist_ok=True)
        readme = family_root / "README.md"
        if readme.exists():
            continue
        readme.write_text(
            "\n".join(
                [
                    f"# {family}",
                    "",
                    family_notes.get(family, "Normalization family."),
                    "",
                    "Variants in this family should include local `README.md` and `variant.json` files.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def _write_variant_metadata(
    *,
    variant_root: Path,
    source_root: Path,
    spec: VariantSpec,
    stats: dict[str, Any],
) -> None:
    variant_root.mkdir(parents=True, exist_ok=True)
    variant_json = {
        "name": spec.name,
        "family": spec.family,
        "version": spec.version,
        "summary": spec.summary,
        "source_root": str(source_root),
        "canonical_dataset_root": str(variant_root),
        "builder_script": str(Path(__file__).resolve()),
        "built_at_utc": _utc_now(),
        "strategy": spec.strategy,
        "intended_use": list(spec.intended_use),
        "text_transforms": list(spec.text_transforms),
        "annotation_transforms": list(spec.annotation_transforms),
        "stats": stats,
    }
    (variant_root / "variant.json").write_text(
        json.dumps(variant_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    readme_lines = [
        f"# {spec.name}",
        "",
        f"Family: `{spec.family}`",
        f"Version: `{spec.version}`",
        "",
        spec.summary,
        "",
        "Intended use:",
    ]
    for item in spec.intended_use:
        readme_lines.append(f"- `{item}`")
    readme_lines.extend(["", "Text transforms:"])
    for item in spec.text_transforms:
        readme_lines.append(f"- `{item}`")
    readme_lines.extend(["", "Annotation transforms:"])
    if spec.annotation_transforms:
        for item in spec.annotation_transforms:
            readme_lines.append(f"- `{item}`")
    else:
        readme_lines.append("- none")
    readme_lines.extend(["", "Known risks:"])
    for item in spec.risks:
        readme_lines.append(f"- {item}")
    readme_lines.extend(
        [
            "",
            "Future grouping note:",
            "- this variant uses text rewriting only",
            "- future `grouped/` variants should preserve more original text and attach semantic group metadata instead of collapsing the wording itself",
            "",
            "Build stats summary:",
            f"- changed records: {stats['changed_record_count']}",
            f"- changed findings: {stats['total_findings_changed']}",
        ]
    )
    (variant_root / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8"
    )


def _run_distribution_analysis(dataset_root: Path, output_path: Path) -> None:
    script = Path(__file__).with_name("analyze_organ_text_distribution.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


_TRAILING_PUNCTUATION_RE = re.compile(r"[\s\.,;:]+$")
_MULTISPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:.])")
_TRAILING_PERIOD_RUN_RE = re.compile(r"\.{2,}$")


def _normalize_surface(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned


def _minimal_surface_v1(text: str) -> str:
    cleaned = _normalize_surface(text)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _TRAILING_PERIOD_RUN_RE.sub(".", cleaned)
    return cleaned


def _normalized_key(text: str) -> str:
    cleaned = _normalize_surface(text).lower()
    cleaned = _TRAILING_PUNCTUATION_RE.sub("", cleaned)
    return cleaned


def _canonical_normals_v1(text: str) -> str:
    surface = _normalize_surface(text)
    key = _normalized_key(surface)
    if key in {
        "normal",
        "unremarkable",
        "within normal limits",
        "within normal limit",
        "normal appearance",
        "unremarkable appearance",
        "grossly unremarkable",
        "appears normal",
    }:
        return "unremarkable"
    return surface


def _canonical_normals_and_absent_v1(text: str) -> str:
    surface = _canonical_normals_v1(text)
    key = _normalized_key(surface)
    if key in {
        "surgically absent",
        "is surgically absent",
        "are surgically absent",
        "post cholecystectomy",
        "status post cholecystectomy",
        "status-post cholecystectomy",
        "prior cholecystectomy",
        "gallbladder surgically absent",
    }:
        return "surgically absent"
    return surface


def _canonical_normals_absent_surface_v2(text: str) -> str:
    surface = _minimal_surface_v1(text)
    key = _normalized_key(surface)
    if key in {
        "normal",
        "unremarkable",
        "within normal limits",
        "within normal limit",
        "normal appearance",
        "unremarkable appearance",
        "grossly unremarkable",
        "appears normal",
    }:
        return "unremarkable"
    if key in {
        "surgically absent",
        "is surgically absent",
        "are surgically absent",
        "post cholecystectomy",
        "status post cholecystectomy",
        "status-post cholecystectomy",
        "prior cholecystectomy",
        "gallbladder surgically absent",
    }:
        return "surgically absent"
    return surface


_NORMAL_TEMPLATE_TEXTS = (
    "the adrenal glands have normal morphology with no nodules or masses.",
    "the adrenal glands appear normal.",
    "the adrenal glands are normal.",
    "the adrenal glands show normal appearance.",
    "the adrenal glands are within normal limits.",
    "no masses or hyperplasia are present in the adrenal glands.",
    "no masses or hyperplasia in the adrenal glands.",
    "the pancreas demonstrates normal attenuation. no pancreatic duct dilatation or focal masses.",
    "the spleen is normal in size and homogeneous, with no splenomegaly or focal lesions identified.",
    "the urinary bladder is well-distended with no wall thickening or masses.",
    "the urinary bladder is normal.",
    "the urinary bladder is unremarkable.",
    "the bladder appears normal.",
    "the bladder is normal.",
    "the bladder is unremarkable.",
    "the bladder is within normal limits.",
    "the urinary bladder is within normal limits.",
    "the gallbladder has a normal caliber. no wall thickening, pericholecystic fluid, or gallstones.",
    "the kidneys are normal in size and axis. no hydronephrosis, renal calculi, or solid masses.",
    "the small bowel loops are normal in caliber without wall thickening or obstruction.",
    "the liver is normal in size with smooth contours. no focal hepatic lesions, biliary dilatation, or steatosis.",
    "the colon is normal in caliber with no wall thickening, obstruction, or free air.",
    "the prostate is normal in size with no enlargement or focal nodules.",
    "the stomach shows normal wall thickness with no distension or masses.",
)

_ABSENT_TEMPLATE_TEXTS = (
    "surgically absent.",
    "surgically absent",
    "gallbladder surgically absent.",
    "gallbladder surgically absent",
    "the urinary bladder is surgically absent.",
    "status post cholecystectomy.",
    "status post cholecystectomy",
    "status-post cholecystectomy.",
    "prior cholecystectomy.",
    "prior cholecystectomy",
    "post cholecystectomy.",
)

_NORMAL_TEMPLATE_KEYS = {_normalized_key(text) for text in _NORMAL_TEMPLATE_TEXTS}
_ABSENT_TEMPLATE_KEYS = {_normalized_key(text) for text in _ABSENT_TEMPLATE_TEXTS}


def _canonical_normal_templates_v2(text: str) -> str:
    surface = _canonical_normals_v1(text)
    key = _normalized_key(surface)
    if key in _NORMAL_TEMPLATE_KEYS:
        return "unremarkable"
    return surface


def _canonical_normal_templates_and_absent_v2(text: str) -> str:
    surface = _canonical_normal_templates_v2(text)
    key = _normalized_key(surface)
    if key in _ABSENT_TEMPLATE_KEYS:
        return "surgically absent"
    return surface


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
