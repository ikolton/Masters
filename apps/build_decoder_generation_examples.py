#!/usr/bin/env python3
"""Build compact side-by-side examples from decoder benchmark generations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, help="Benchmark directory containing runs/*/generations.json.")
    parser.add_argument("--output-json", default="", help="Output JSON path. Defaults under benchmark-dir.")
    parser.add_argument("--output-md", default="", help="Output Markdown path. Defaults under benchmark-dir.")
    parser.add_argument("--max-examples", type=int, default=48, help="Maximum examples to write.")
    parser.add_argument("--per-organ", type=int, default=4, help="Maximum examples per organ.")
    parser.add_argument("--prefer-disagreements", action="store_true", help="Prioritize rows where checkpoints differ.")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else benchmark_dir / "generation_examples.json"
    output_md = Path(args.output_md).expanduser().resolve() if args.output_md else benchmark_dir / "generation_examples.md"
    examples = build_examples(
        benchmark_dir,
        max_examples=int(args.max_examples),
        per_organ=int(args.per_organ),
        prefer_disagreements=bool(args.prefer_disagreements),
    )
    output_json.write_text(json.dumps(examples, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(examples), encoding="utf-8")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")


def build_examples(
    benchmark_dir: Path,
    *,
    max_examples: int,
    per_organ: int,
    prefer_disagreements: bool,
) -> dict[str, Any]:
    run_payloads = _load_run_payloads(benchmark_dir)
    if not run_payloads:
        raise FileNotFoundError(f"No generations found under {benchmark_dir / 'runs'}")
    labels = [label for label, _ in run_payloads]
    aligned: dict[tuple[str, str], dict[str, Any]] = {}
    for label, payload in run_payloads:
        for row in payload.get("generations", []):
            key = (str(row.get("study_id", "")), str(row.get("organ", "")))
            if not key[0] or not key[1]:
                continue
            entry = aligned.setdefault(
                key,
                {
                    "study_id": key[0],
                    "organ": key[1],
                    "target": str(row.get("target", "")),
                    "lesion_label": row.get("lesion_label"),
                    "generations": {},
                },
            )
            entry["generations"][label] = str(row.get("generated", ""))
    complete = [entry for entry in aligned.values() if all(label in entry["generations"] for label in labels)]
    complete.sort(key=lambda entry: (_rank_entry(entry, labels, prefer_disagreements), entry["organ"], entry["study_id"]))
    selected = _select_balanced(complete, max_examples=max_examples, per_organ=per_organ)
    return {
        "benchmark_dir": str(benchmark_dir),
        "run_labels": labels,
        "selection": {
            "max_examples": int(max_examples),
            "per_organ": int(per_organ),
            "prefer_disagreements": bool(prefer_disagreements),
            "available_complete_examples": len(complete),
            "selected_examples": len(selected),
        },
        "examples": selected,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    labels = list(payload.get("run_labels", []))
    lines = [
        "# Decoder Generation Examples",
        "",
        f"- Benchmark: `{payload.get('benchmark_dir')}`",
        f"- Selected examples: `{payload.get('selection', {}).get('selected_examples')}`",
        f"- Runs: {', '.join(f'`{label}`' for label in labels)}",
        "",
        "These examples are aligned by `(study_id, organ)`. They are intended for quick qualitative review, not as a replacement for aggregate metrics.",
        "",
    ]
    for index, example in enumerate(payload.get("examples", []), start=1):
        lesion = example.get("lesion_label")
        lesion_text = "missing" if lesion is None else str(lesion)
        lines.extend(
            [
                f"## {index}. {example.get('organ')} / `{example.get('study_id')}`",
                "",
                f"- Lesion label: `{lesion_text}`",
                "",
                "**Reference**",
                "",
                _quote(example.get("target", "")),
                "",
            ]
        )
        for label in labels:
            lines.extend(
                [
                    f"**{label}**",
                    "",
                    _quote(example.get("generations", {}).get(label, "")),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _load_run_payloads(benchmark_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    for path in sorted((benchmark_dir / "runs").glob("*/generations.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = str(payload.get("label") or path.parent.name)
        payloads.append((label, payload))
    return payloads


def _rank_entry(entry: dict[str, Any], labels: list[str], prefer_disagreements: bool) -> tuple[int, int, str]:
    generations = [entry.get("generations", {}).get(label, "") for label in labels]
    unique_count = len({_normalize_text(text) for text in generations})
    has_pathology = any(_has_pathology_signal(text) for text in generations)
    lesion = entry.get("lesion_label")
    is_positive = isinstance(lesion, (int, float)) and float(lesion) > 0.5
    if not prefer_disagreements:
        return (0, 0, str(entry.get("study_id", "")))
    return (-int(unique_count > 1), -int(is_positive or has_pathology), str(entry.get("study_id", "")))


def _select_balanced(entries: list[dict[str, Any]], *, max_examples: int, per_organ: int) -> list[dict[str, Any]]:
    counts: defaultdict[str, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for entry in entries:
        organ = str(entry.get("organ", ""))
        if counts[organ] >= per_organ:
            continue
        selected.append(entry)
        counts[organ] += 1
        if len(selected) >= max_examples:
            break
    return selected


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _has_pathology_signal(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(word in normalized for word in ("lesion", "mass", "nodule", "cyst", "metasta", "tumor", "tumour"))


def _quote(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return "> _empty_"
    return "\n".join(f"> {line}" for line in cleaned.splitlines())


if __name__ == "__main__":
    main()
