#!/usr/bin/env python
"""Train a Merlin ablation run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from merlin_ablation.config import load_config
from merlin_ablation.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Merlin ablation YAML config.")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run_training(config)
    print(f"[merlin-ablation] finished run_id={config.train.run_id} output_dir={config.output_dir}")
    print(f"[merlin-ablation] val={summary['val']}")


if __name__ == "__main__":
    main()

