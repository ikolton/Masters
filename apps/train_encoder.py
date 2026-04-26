#!/usr/bin/env python3
"""Train the OrganSegCLIP encoder."""

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

from organ_seg_clip.config import load_encoder_config
from organ_seg_clip.training import run_encoder_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    args = parser.parse_args()
    config = load_encoder_config(args.config)
    result = run_encoder_training(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
