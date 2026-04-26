#!/usr/bin/env python3
"""Analyze generated per-organ findings for binary pathology/normal words."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path


DEFAULT_PATHOLOGY_WORDS = (
    "lesion",
    "lesions",
    "cyst",
    "cysts",
    "mass",
    "masses",
    "nodule",
    "nodules",
    "metastasis",
    "metastases",
    "tumor",
    "tumour",
)
DEFAULT_NORMAL_WORDS = (
    "unremarkable",
    "normal",
    "within normal limits",
    "no abnormality",
    "no focal abnormality",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON file produced by apps/generate_decoder.py.")
    parser.add_argument("--output", default="", help="Optional JSON summary output path.")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).expanduser().read_text(encoding="utf-8"))
    rows = list(payload.get("generations", []))
    summary = _summarize(rows)
    if args.output:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    totals = defaultdict(int)
    per_organ: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        organ = str(row.get("organ", ""))
        generated = str(row.get("generated", ""))
        target = str(row.get("target", ""))
        lesion_label = row.get("lesion_label")
        has_path = _contains_any(generated, DEFAULT_PATHOLOGY_WORDS)
        has_normal = _contains_any(generated, DEFAULT_NORMAL_WORDS)
        target_has_path = _contains_any(target, DEFAULT_PATHOLOGY_WORDS)
        target_has_normal = _contains_any(target, DEFAULT_NORMAL_WORDS)
        bucket = per_organ[organ]
        for stats in (totals, bucket):
            stats["count"] += 1
            stats["generated_pathology_word_count"] += int(has_path)
            stats["generated_normal_word_count"] += int(has_normal)
            stats["target_pathology_word_count"] += int(target_has_path)
            stats["target_normal_word_count"] += int(target_has_normal)
            if lesion_label is not None:
                positive = float(lesion_label) > 0.5
                stats["csv_labeled_count"] += 1
                stats["csv_positive_count"] += int(positive)
                stats["csv_negative_count"] += int(not positive)
                if positive:
                    stats["csv_positive_generated_pathology_count"] += int(has_path)
                    stats["csv_positive_generated_normal_count"] += int(has_normal)
                else:
                    stats["csv_negative_generated_pathology_count"] += int(has_path)
                    stats["csv_negative_generated_normal_count"] += int(has_normal)
    return {
        "count": totals["count"],
        "overall": _rates(totals),
        "per_organ": {organ: _rates(stats) for organ, stats in sorted(per_organ.items())},
    }


def _rates(stats: dict[str, int]) -> dict[str, float]:
    count = max(1, int(stats["count"]))
    csv_pos = max(1, int(stats["csv_positive_count"]))
    csv_neg = max(1, int(stats["csv_negative_count"]))
    return {
        "count": float(stats["count"]),
        "generated_pathology_word_rate": stats["generated_pathology_word_count"] / count,
        "generated_normal_word_rate": stats["generated_normal_word_count"] / count,
        "target_pathology_word_rate": stats["target_pathology_word_count"] / count,
        "target_normal_word_rate": stats["target_normal_word_count"] / count,
        "csv_labeled_count": float(stats["csv_labeled_count"]),
        "csv_positive_count": float(stats["csv_positive_count"]),
        "csv_negative_count": float(stats["csv_negative_count"]),
        "csv_positive_pathology_recall": stats["csv_positive_generated_pathology_count"] / csv_pos,
        "csv_positive_normal_rate": stats["csv_positive_generated_normal_count"] / csv_pos,
        "csv_negative_pathology_rate": stats["csv_negative_generated_pathology_count"] / csv_neg,
        "csv_negative_normal_rate": stats["csv_negative_generated_normal_count"] / csv_neg,
    }


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    normalized = re.sub(r"\s+", " ", str(text).lower())
    return any(re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", normalized) for word in words)


if __name__ == "__main__":
    main()
