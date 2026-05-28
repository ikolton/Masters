#!/usr/bin/env python3
"""Attach post-hoc RadGPT benchmark outputs to run evaluation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, help="Benchmark dir with runs/*/evaluation.json.")
    parser.add_argument("--radgpt-output-dir", required=True, help="Output dir from run_radgpt_benchmark_from_generations.py.")
    parser.add_argument(
        "--target-key",
        default="radgpt_oncology",
        help="Evaluation JSON key to update. Defaults to full-run radgpt_oncology.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    radgpt_output_dir = Path(args.radgpt_output_dir).expanduser().resolve()
    metadata = _read_json(radgpt_output_dir / "summary.json").get("metadata", {})
    attached: list[dict[str, Any]] = []
    missing: list[str] = []

    for comparison_path in sorted((radgpt_output_dir / "runs").glob("*/radgpt/comparison.json")):
        label = comparison_path.parents[1].name
        evaluation_path = benchmark_dir / "runs" / label / "evaluation.json"
        if not evaluation_path.is_file():
            missing.append(str(evaluation_path))
            continue
        comparison = _read_json(comparison_path)
        payload = _read_json(evaluation_path)
        payload[str(args.target_key)] = {
            **comparison,
            "radgpt_output_dir": str(radgpt_output_dir),
            "radgpt_metadata": metadata,
        }
        evaluation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        attached.append(
            {
                "label": label,
                "evaluation_path": str(evaluation_path),
                "comparison_path": str(comparison_path),
            }
        )

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "radgpt_output_dir": str(radgpt_output_dir),
        "target_key": str(args.target_key),
        "attached_count": len(attached),
        "attached": attached,
        "missing": missing,
    }
    out_path = radgpt_output_dir / "attach_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(f"Missing {len(missing)} benchmark evaluation files; see {out_path}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


if __name__ == "__main__":
    main()
