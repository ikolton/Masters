#!/usr/bin/env python3
"""Run a meaningfully downsized encoder training smoke test from a full config."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.config import load_encoder_config
from organ_seg_clip.training import run_encoder_training
from organ_seg_clip.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Base encoder config.")
    parser.add_argument("--output-dir", default="", help="Optional output directory for the smoke run.")
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--val-limit", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=20)
    parser.add_argument("--max-val-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--enable-wandb", action="store_true")
    args = parser.parse_args()

    base = load_encoder_config(args.config)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ensure_dir(ROOT / "outputs" / "encoder" / "smoke" / base.paths.resolve_output_dir(Path(base.config_dir)).name)
    )
    ensure_dir(output_dir)

    training = replace(
        base.training,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=bool(int(args.num_workers) > 0),
        epochs=int(args.epochs),
        log_every_steps=1,
        save_every_steps=0,
        validation_every_epochs=1,
        fast_val_limit=min(int(args.val_limit), 32),
        fast_val_sampling="fixed",
        profile_timing=True,
        max_train_steps=int(args.max_train_steps),
        max_val_steps=int(args.max_val_steps),
        save_last_checkpoint=False,
        save_best_checkpoint=False,
    )
    data = replace(base.data, train_limit=int(args.train_limit), val_limit=int(args.val_limit))
    patching = replace(base.model.patching, patch_batch_size=int(args.patch_batch_size))
    model = replace(base.model, patching=patching)
    paths = replace(base.paths, output_dir=str(output_dir))
    runtime = replace(base.runtime, compile_model=bool(args.compile_model))
    logging = replace(base.logging, wandb_enabled=bool(args.enable_wandb), wandb_mode=("online" if args.enable_wandb else "disabled"))
    config = replace(base, paths=paths, data=data, model=model, training=training, runtime=runtime, logging=logging)

    summary = run_encoder_training(config)
    history = summary.get("history", [])
    payload = {
        "smoke_output_dir": str(output_dir),
        "history_tail": history[-1] if history else {},
        "epochs_completed": summary.get("epochs_completed", 0),
        "dataset_summary": summary.get("dataset_summary", {}),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
