#!/usr/bin/env python3
"""Run a short profiling sweep for OrganSegCLIP encoder training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.config import load_encoder_config
from organ_seg_clip.training import run_encoder_training
from organ_seg_clip.utils.io import dump_json, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Base encoder config.")
    parser.add_argument("--num-workers", default="2,4,8", help="Comma-separated worker counts.")
    parser.add_argument("--patch-batch-size", default="2,4,6", help="Comma-separated patch batch sizes.")
    parser.add_argument("--batch-size", default="", help="Comma-separated training batch sizes. Defaults to the base config batch size.")
    parser.add_argument("--compile-model", default="false,true", help="Comma-separated booleans.")
    parser.add_argument("--output-dir", default="", help="Optional directory for profile outputs.")
    args = parser.parse_args()

    base = load_encoder_config(args.config)
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ensure_dir(ROOT / "outputs" / "encoder" / "profile_sweeps" / base.paths.resolve_output_dir(Path(base.config_dir)).name)
    )
    ensure_dir(output_root)

    worker_grid = _parse_int_list(args.num_workers)
    patch_grid = _parse_int_list(args.patch_batch_size)
    batch_grid = _parse_int_list(args.batch_size) if str(args.batch_size).strip() else [int(base.training.batch_size)]
    compile_grid = _parse_bool_list(args.compile_model)

    results: list[dict[str, object]] = []
    for num_workers, patch_batch_size, batch_size, compile_model in itertools.product(worker_grid, patch_grid, batch_grid, compile_grid):
        variant_name = f"nw{num_workers}_pb{patch_batch_size}_bs{batch_size}_compile{int(compile_model)}"
        variant_output_dir = output_root / variant_name
        config = _profile_variant(
            base,
            output_dir=variant_output_dir,
            num_workers=num_workers,
            patch_batch_size=patch_batch_size,
            batch_size=batch_size,
            compile_model=compile_model,
        )
        print(f"[profile_encoder] running {variant_name}", flush=True)
        try:
            summary = run_encoder_training(config)
            history = summary.get("history", [])
            final_metrics = history[-1] if history else {}
            train_step_seconds = float(final_metrics.get("train_step_seconds", 0.0))
            train_data_wait_seconds = float(final_metrics.get("train_data_wait_seconds", 0.0))
            data_fraction = 0.0 if train_step_seconds <= 0.0 else train_data_wait_seconds / max(train_step_seconds, 1.0e-6)
            result = {
                "variant": variant_name,
                "status": "ok",
                "num_workers": num_workers,
                "patch_batch_size": patch_batch_size,
                "batch_size": batch_size,
                "compile_model": compile_model,
                "output_dir": str(variant_output_dir),
                "train_step_seconds": train_step_seconds,
                "train_data_wait_seconds": train_data_wait_seconds,
                "train_data_fraction": data_fraction,
                "train_cuda_memory_allocated_gb": float(final_metrics.get("train_cuda_memory_allocated_gb", 0.0)),
                "train_cuda_memory_reserved_gb": float(final_metrics.get("train_cuda_memory_reserved_gb", 0.0)),
                "train_patches_per_batch_total": float(final_metrics.get("train_patches_per_batch_total", 0.0)),
                "train_patches_per_study_mean": float(final_metrics.get("train_patches_per_study_mean", 0.0)),
                "train_patches_per_study_max": float(final_metrics.get("train_patches_per_study_max", 0.0)),
                "train_seconds_per_study_global": float(final_metrics.get("train_seconds_per_study_global", 0.0)),
                "val_total_loss": float(final_metrics.get("val_total_loss", 0.0)),
                "bottleneck_guess": _guess_bottleneck(train_step_seconds, train_data_wait_seconds),
            }
        except Exception as exc:
            result = {
                "variant": variant_name,
                "status": "failed",
                "num_workers": num_workers,
                "patch_batch_size": patch_batch_size,
                "batch_size": batch_size,
                "compile_model": compile_model,
                "output_dir": str(variant_output_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        dump_json(variant_output_dir / "profile_summary.json", result)

    successful = [item for item in results if item.get("status") == "ok"]
    ranked = sorted(successful, key=lambda item: (float(item["train_step_seconds"]), float(item["train_data_wait_seconds"])))
    aggregate = {
        "base_config": str(Path(args.config).expanduser().resolve()),
        "output_dir": str(output_root),
        "results": results,
        "ranked_fastest_first": ranked,
        "best_variant": ranked[0] if ranked else None,
    }
    dump_json(output_root / "profile_sweep_summary.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


def _profile_variant(base, *, output_dir: Path, num_workers: int, patch_batch_size: int, batch_size: int, compile_model: bool):
    output_dir = ensure_dir(output_dir)
    training = replace(
        base.training,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
        epochs=1,
        log_every_steps=1,
        save_every_steps=0,
        profile_timing=True,
        max_train_steps=20,
        max_val_steps=8,
        save_last_checkpoint=False,
        save_best_checkpoint=False,
    )
    data = replace(base.data, train_limit=64, val_limit=16)
    patching = replace(base.model.patching, patch_batch_size=int(patch_batch_size))
    model = replace(base.model, patching=patching)
    paths = replace(base.paths, output_dir=str(output_dir))
    runtime = replace(base.runtime, compile_model=bool(compile_model))
    return replace(base, paths=paths, data=data, model=model, training=training, runtime=runtime)


def _parse_int_list(raw: str) -> list[int]:
    return [int(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip()]


def _parse_bool_list(raw: str) -> list[bool]:
    values: list[bool] = []
    for chunk in str(raw).split(","):
        normalized = chunk.strip().lower()
        if not normalized:
            continue
        if normalized in {"1", "true", "yes", "y"}:
            values.append(True)
        elif normalized in {"0", "false", "no", "n"}:
            values.append(False)
        else:
            raise ValueError(f"Unsupported boolean value: {chunk}")
    return values


def _guess_bottleneck(step_seconds: float, data_wait_seconds: float) -> str:
    if step_seconds <= 0.0:
        return "timing_disabled_or_unavailable"
    ratio = data_wait_seconds / max(step_seconds, 1.0e-6)
    if ratio >= 0.35:
        return "data_pipeline_likely"
    if ratio >= 0.15:
        return "mixed_data_and_model"
    return "model_or_launch_overhead_likely"


if __name__ == "__main__":
    main()
