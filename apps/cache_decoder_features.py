#!/usr/bin/env python3
"""Cache OrganSegCLIP visual features for decoder training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.config import load_decoder_config
from organ_seg_clip.decoder.data import save_feature_store
from organ_seg_clip.decoder.feature_cache import build_feature_store, feature_cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the decoder YAML config.")
    parser.add_argument("--split", choices=("train", "val"), default=None, help="Split to cache. Defaults to train and val.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the feature cache file already exists.")
    args = parser.parse_args()
    config = load_decoder_config(args.config)
    device = torch.device(config.training.device if torch.cuda.is_available() or not config.training.device.startswith("cuda") else "cpu")
    splits = [args.split] if args.split else [config.data.train_split, config.data.val_split]
    result = {}
    for split in splits:
        target = feature_cache_path(config, split)
        if target is None:
            target = config.resolved_output_dir / f"{split}_features.pt"
        # Idempotent: skip the (expensive, GPU+IO) rebuild when a cache already exists.
        # Features depend only on the visual encoder + data, so an existing cache is reusable
        # and re-extraction is pure waste. Also avoids concurrent jobs clobbering a shared
        # cache file. Use --force to rebuild.
        if target.exists() and not args.force:
            result[split] = {"feature_cache": str(target), "built": False, "skipped": True}
            continue
        store, summary = build_feature_store(config, split=split, device=device)
        save_feature_store(target, store)
        result[split] = summary | {"feature_cache": str(target), "built": True}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
