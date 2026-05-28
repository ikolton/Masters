#!/usr/bin/env python
"""Merge sharded Merlin benchmark generations and run metrics once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
APPS_ROOT = ROOT / "apps"
if str(APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(APPS_ROOT))

from evaluate_decoder_generations import DEFAULT_COCO_METRICS, evaluate_file
from benchmark_merlin_ablation_run import _parse_csv, _slugify, _write_single_run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merged-label", required=True)
    parser.add_argument("--shard-label", action="append", required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_COCO_METRICS))
    parser.add_argument("--tokenize", choices=("auto", "java", "none"), default="auto")
    parser.add_argument("--green", action="store_true")
    parser.add_argument("--green-scope", choices=("organ", "study", "both"), default="organ")
    parser.add_argument("--no-study-level", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    merged_dir = output_dir / "runs" / _slugify(args.merged_label)
    merged_dir.mkdir(parents=True, exist_ok=True)
    generations_path = merged_dir / "generations.json"
    evaluation_path = merged_dir / "evaluation.json"

    payloads = [_load_generation(output_dir, label) for label in args.shard_label]
    generations = []
    seen = set()
    for payload in payloads:
        for row in payload.get("generations", []):
            key = (str(row.get("study_id", "")), str(row.get("organ", "")))
            if key in seen:
                raise ValueError(f"Duplicate generated row while merging shards: {key}")
            seen.add(key)
            generations.append(row)
    generations.sort(key=lambda row: (str(row.get("study_id", "")), str(row.get("organ", ""))))

    merged_payload: dict[str, Any] = {
        "format": "merlin_ablation_generations_v1",
        "label": str(args.merged_label),
        "split": "test",
        "merged_from": list(args.shard_label),
        "requested": {
            "merged_from": list(args.shard_label),
            "row_count": len(generations),
        },
        "generations": generations,
    }
    generations_path.write_text(json.dumps(merged_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[merlin-merge] wrote generations={generations_path} rows={len(generations)}", flush=True)

    evaluation = evaluate_file(
        generations_path,
        metrics=_parse_csv(args.metrics),
        tokenize_mode=str(args.tokenize),
        green_scope=str(args.green_scope) if bool(args.green) else "none",
        limit=None,
        include_study_level=not bool(args.no_study_level),
    )
    evaluation["label"] = str(args.merged_label)
    evaluation["generation_path"] = str(generations_path)
    evaluation["merged_from"] = list(args.shard_label)
    evaluation["requested_metrics"] = _parse_csv(args.metrics)
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[merlin-merge] wrote evaluation={evaluation_path}", flush=True)
    _write_single_run_summary(output_dir)


def _load_generation(output_dir: Path, label: str) -> dict[str, Any]:
    path = output_dir / "runs" / _slugify(label) / "generations.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
