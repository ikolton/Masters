#!/usr/bin/env python3
"""Evaluate a trained per-organ report decoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.config import load_decoder_config
from organ_seg_clip.training import run_decoder_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the decoder YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Decoder checkpoint path.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()
    config = load_decoder_config(args.config)
    result = run_decoder_evaluation(config, checkpoint_path=args.checkpoint, split=args.split)
    if args.output and int(os.environ.get("RANK", "0")) == 0:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
