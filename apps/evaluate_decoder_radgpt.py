#!/usr/bin/env python3
"""Evaluate decoder generations with RadGPT oncology label extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.evaluation.radgpt_oncology import (  # noqa: E402
    evaluate_generation_file_with_radgpt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Generation JSON from apps/generate_decoder.py or the benchmark run directory.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--cache-dir", default="", help="Optional cache dir. Defaults to sibling '<input stem>_radgpt'.")
    parser.add_argument("--base-url", default="http://0.0.0.0:8000/v1", help="OpenAI-compatible API base URL for RadGPT.")
    parser.add_argument("--radgpt-root", default="/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadGPT", help="Path to local RadGPT clone.")
    parser.set_defaults(fast=True)
    parser.add_argument("--fast", dest="fast", action="store_true", help="Use RadGPT fast prompts.")
    parser.add_argument("--slow", dest="fast", action="store_false", help="Use RadGPT slower, larger prompts.")
    parser.add_argument("--force-reference", action="store_true", help="Recompute shared reference labels.")
    parser.add_argument("--force-generated", action="store_true", help="Recompute generated labels.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else input_path.with_suffix("").with_name(f"{input_path.stem}_radgpt")
    )
    result = evaluate_generation_file_with_radgpt(
        input_path,
        benchmark_cache_dir=cache_dir,
        base_url=str(args.base_url),
        fast=bool(args.fast),
        force_reference=bool(args.force_reference),
        force_generated=bool(args.force_generated),
        radgpt_root=str(args.radgpt_root),
    )
    text = json.dumps(result, indent=args.indent, sort_keys=True)
    if args.output:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
