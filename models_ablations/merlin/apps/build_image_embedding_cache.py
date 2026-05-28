#!/usr/bin/env python
"""Build frozen Merlin image-embedding cache for an ablation config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from merlin_ablation.cache import build_image_embedding_cache
from merlin_ablation.config import load_config
from merlin_ablation.train import _prepare_imports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    _prepare_imports(config)
    build_image_embedding_cache(
        config,
        force=args.force,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
