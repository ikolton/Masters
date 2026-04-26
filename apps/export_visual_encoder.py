#!/usr/bin/env python3
"""Export a distilled image-only visual encoder checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.config import load_encoder_config
from organ_seg_clip.models.visual_encoder import build_visual_encoder, load_visual_weights_from_full_checkpoint
from organ_seg_clip.training.checkpointing import unwrap_model
from organ_seg_clip.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Encoder training config used by the source checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Full OrganSegCLIP checkpoint to distill.")
    parser.add_argument("--output", required=True, help="Output path for the distilled visual encoder checkpoint.")
    parser.add_argument("--map-location", default="cpu")
    args = parser.parse_args()

    config = load_encoder_config(args.config)
    visual_encoder = build_visual_encoder(config)
    load_info = load_visual_weights_from_full_checkpoint(visual_encoder, args.checkpoint, map_location=args.map_location)
    state = unwrap_model(visual_encoder).state_dict()
    source_payload = load_info.get("payload", {})
    output_path = Path(args.output).expanduser().resolve()
    ensure_dir(output_path.parent)
    export_payload: dict[str, Any] = {
        "format": "organsegclip_visual_encoder_v1",
        "model_state": state,
        "source_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "source_checkpoint_epoch": source_payload.get("epoch"),
        "source_checkpoint_step": source_payload.get("step"),
        "source_checkpoint_metrics": source_payload.get("metrics", {}),
        "matched_keys": load_info["matched_keys"],
        "missing_keys": load_info["missing_keys"],
        "unexpected_keys": load_info["unexpected_keys"],
        "skipped_full_keys": load_info["skipped_full_keys"],
        "visual_dim": int(config.model.tokenizer.model_dim),
        "organ_names": list(config.data.organ_names),
        "patch_size": list(config.model.patching.patch_size),
        "patch_stride": list(config.model.patching.patch_stride),
        "patch_batch_size": int(config.model.patching.patch_batch_size),
        "config": config.to_dict(),
    }
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    torch.save(export_payload, tmp_path)
    tmp_path.replace(output_path)
    summary = {
        "output": str(output_path),
        "format": export_payload["format"],
        "matched_keys": export_payload["matched_keys"],
        "missing_keys": export_payload["missing_keys"],
        "unexpected_keys": export_payload["unexpected_keys"],
        "skipped_full_keys": export_payload["skipped_full_keys"],
        "source_checkpoint_epoch": export_payload["source_checkpoint_epoch"],
        "source_checkpoint_step": export_payload["source_checkpoint_step"],
        "visual_dim": export_payload["visual_dim"],
        "organ_names": export_payload["organ_names"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
