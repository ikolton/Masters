#!/usr/bin/env python3
"""Refresh decoder benchmark comparison files from cached run artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "apps"
if str(APPS) not in sys.path:
    sys.path.insert(0, str(APPS))

from benchmark_decoder_checkpoints import (  # noqa: E402
    RunSpec,
    _build_comparison_row,
    _print_ascii_summary,
    _write_csv,
    _write_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    manifest_path = benchmark_dir / "sample_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing sample manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    comparison_rows = []
    run_summaries = []
    skipped_runs = []
    for generation_path in sorted((benchmark_dir / "runs").glob("*/generations.json")):
        run_dir = generation_path.parent
        label = run_dir.name
        evaluation_path = run_dir / "evaluation.json"
        if not evaluation_path.is_file():
            skipped_runs.append(
                {
                    "label": label,
                    "generation_path": str(generation_path),
                    "reason": f"missing evaluation.json: {evaluation_path}",
                }
            )
            continue
        generation_payload = json.loads(generation_path.read_text(encoding="utf-8"))
        evaluation_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        config_path = benchmark_dir / "configs" / f"{label.replace('-', '_')}.yaml"
        run_spec = RunSpec(label=label, config_path=config_path, checkpoint_path=Path(""))
        row = _build_comparison_row(
            run_spec=run_spec,
            generation_payload=generation_payload,
            evaluation_payload=evaluation_payload,
            manifest=manifest,
        )
        comparison_rows.append(row)
        run_summaries.append(
            {
                "label": label,
                "generation_path": str(generation_path),
                "evaluation_path": str(evaluation_path),
                "row": row,
                "warnings": evaluation_payload.get("warnings", []),
                "unavailable_metrics": evaluation_payload.get("unavailable_metrics", {}),
            }
        )

    summary = {
        "output_dir": str(benchmark_dir),
        "sample_manifest": manifest,
        "runs": run_summaries,
        "skipped_runs": skipped_runs,
        "comparison_rows": comparison_rows,
    }
    (benchmark_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(benchmark_dir / "comparison_summary.csv", comparison_rows)
    _write_markdown(benchmark_dir / "comparison_summary.md", comparison_rows)
    _print_ascii_summary(comparison_rows)


if __name__ == "__main__":
    main()
