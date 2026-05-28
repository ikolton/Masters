#!/usr/bin/env python3
"""Compare baseline and normalized organ-text distribution analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variant", nargs="+", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    baseline = _load_json(Path(args.baseline))
    variants = {Path(path).stem.replace("_text_distribution", ""): _load_json(Path(path)) for path in args.variant}

    result = {
        "baseline": {
            "path": str(Path(args.baseline).expanduser().resolve()),
            "global_summary": baseline.get("global_summary", {}),
        },
        "variants": {},
    }

    for name, payload in variants.items():
        result["variants"][name] = _compare_variant(baseline, payload)

    blob = json.dumps(result, indent=2, sort_keys=True)
    print(blob)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(blob, encoding="utf-8")


def _compare_variant(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    baseline_global = baseline.get("global_summary", {})
    variant_global = variant.get("global_summary", {})
    organ_names = sorted(set(baseline.get("organ_text_distribution", {})) | set(variant.get("organ_text_distribution", {})))

    per_organ = {}
    for organ_name in organ_names:
        before = baseline.get("organ_text_distribution", {}).get(organ_name, {})
        after = variant.get("organ_text_distribution", {}).get(organ_name, {})
        per_organ[organ_name] = {
            "unique_text_count_before": before.get("unique_text_count"),
            "unique_text_count_after": after.get("unique_text_count"),
            "unique_text_delta": _delta(after.get("unique_text_count"), before.get("unique_text_count")),
            "top1_fraction_before": before.get("top1_fraction"),
            "top1_fraction_after": after.get("top1_fraction"),
            "top1_fraction_delta": _delta(after.get("top1_fraction"), before.get("top1_fraction")),
            "effective_class_count_before": before.get("effective_class_count"),
            "effective_class_count_after": after.get("effective_class_count"),
            "effective_class_count_delta": _delta(after.get("effective_class_count"), before.get("effective_class_count")),
            "unremarkable_exact_fraction_before": before.get("unremarkable_exact_fraction"),
            "unremarkable_exact_fraction_after": after.get("unremarkable_exact_fraction"),
            "unremarkable_exact_fraction_delta": _delta(after.get("unremarkable_exact_fraction"), before.get("unremarkable_exact_fraction")),
        }

    ranked_unique_reduction = sorted(
        (
            {
                "organ": organ_name,
                "unique_text_delta": per_organ[organ_name]["unique_text_delta"],
                "effective_class_count_delta": per_organ[organ_name]["effective_class_count_delta"],
            }
            for organ_name in organ_names
        ),
        key=lambda row: (row["unique_text_delta"] if row["unique_text_delta"] is not None else 0.0),
    )

    return {
        "global_delta": {
            "global_unique_text_count_before": baseline_global.get("global_unique_text_count"),
            "global_unique_text_count_after": variant_global.get("global_unique_text_count"),
            "global_unique_text_count_delta": _delta(
                variant_global.get("global_unique_text_count"),
                baseline_global.get("global_unique_text_count"),
            ),
            "global_unremarkable_fraction_before": baseline_global.get("global_unremarkable_fraction"),
            "global_unremarkable_fraction_after": variant_global.get("global_unremarkable_fraction"),
            "global_unremarkable_fraction_delta": _delta(
                variant_global.get("global_unremarkable_fraction"),
                baseline_global.get("global_unremarkable_fraction"),
            ),
        },
        "top_organs_by_unique_reduction": ranked_unique_reduction[:5],
        "top_organs_by_unique_increase": list(reversed(ranked_unique_reduction[-5:])),
        "per_organ": per_organ,
    }


def _delta(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
