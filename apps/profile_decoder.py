#!/usr/bin/env python3
"""Profile decoder training variants on a cached feature subset."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import os

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, help="Base decoder config to clone.")
    parser.add_argument("--encoder-config", required=True, help="Encoder config used for visual encoder export.")
    parser.add_argument("--encoder-checkpoint", required=True, help="Full encoder checkpoint to export from.")
    parser.add_argument("--output-dir", required=True, help="Profile sweep output directory.")
    parser.add_argument("--batch-sizes", default="8,16", help="Per-process decoder batch sizes.")
    parser.add_argument("--world-sizes", default="1,2", help="Distributed world sizes to profile.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--val-limit", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--cache-num-workers", type=int, default=4)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--export-map-location", default="cpu")
    parser.add_argument("--torchrun-bin", default="torchrun")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--force-export", action="store_true", help="Rebuild the visual encoder export even if it already exists.")
    parser.add_argument("--force-cache", action="store_true", help="Rebuild feature caches even if train/val cache files already exist.")
    parser.add_argument("--gpu-sample-seconds", type=float, default=1.0, help="Sampling interval for nvidia-smi GPU memory tracking during train variants.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a very small quick sweep preset: world sizes 1,2; batch sizes 8,16; train_limit 128; val_limit 32.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config).expanduser().resolve()
    encoder_config_path = Path(args.encoder_config).expanduser().resolve()
    encoder_checkpoint_path = Path(args.encoder_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    generated_dir = output_dir / "generated_configs"
    feature_cache_dir = output_dir / "features"
    visual_export_path = output_dir / "visual_encoder.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(base_config_path.read_text())
    base_name = base_config_path.stem
    batch_sizes = _parse_int_list(args.batch_sizes)
    world_sizes = _parse_int_list(args.world_sizes)
    train_limit = int(args.train_limit)
    val_limit = int(args.val_limit)
    if args.quick:
        batch_sizes = [8, 16]
        world_sizes = [1, 2]
        train_limit = 128
        val_limit = 32

    should_export = (not args.skip_export) and (args.force_export or not visual_export_path.is_file())
    if should_export:
        _run_command(
            [
                args.python_bin,
                str(ROOT / "apps" / "export_visual_encoder.py"),
                "--config",
                str(encoder_config_path),
                "--checkpoint",
                str(encoder_checkpoint_path),
                "--output",
                str(visual_export_path),
                "--map-location",
                str(args.export_map_location),
            ],
            cwd=ROOT,
            label="export_visual_encoder",
        )
    elif visual_export_path.is_file():
        print(f"[profile_decoder] reusing existing visual export: {visual_export_path}", flush=True)

    cache_config_path = generated_dir / "cache.yaml"
    cache_cfg = _build_variant_config(
        base=base,
        experiment_name=f"{base_name}_cache",
        output_dir=output_dir / "cache_run",
        visual_encoder_checkpoint=visual_export_path,
        feature_cache_dir=feature_cache_dir,
        batch_size=int(args.cache_batch_size),
        epochs=1,
        train_limit=train_limit,
        val_limit=val_limit,
        num_workers=int(args.cache_num_workers),
        log_every_steps=int(args.log_every_steps),
        save_every_steps=0,
        precompute_features_if_missing=True,
    )
    cache_config_path.write_text(yaml.safe_dump(cache_cfg, sort_keys=False))

    train_cache_path = feature_cache_dir / "train_features.pt"
    val_cache_path = feature_cache_dir / "val_features.pt"
    should_cache = (not args.skip_cache) and (
        args.force_cache or not train_cache_path.is_file() or not val_cache_path.is_file()
    )
    if should_cache:
        _run_command(
            [
                args.python_bin,
                str(ROOT / "apps" / "cache_decoder_features.py"),
                "--config",
                str(cache_config_path),
            ],
            cwd=ROOT,
            label="cache_decoder_features",
        )
    elif train_cache_path.is_file() and val_cache_path.is_file():
        print(
            f"[profile_decoder] reusing existing feature caches: {train_cache_path} and {val_cache_path}",
            flush=True,
        )

    results: list[dict[str, Any]] = []
    for world_size in world_sizes:
        for batch_size in batch_sizes:
            variant = f"ws{world_size}_bs{batch_size}"
            variant_output = output_dir / variant
            if variant_output.exists():
                shutil.rmtree(variant_output)
            variant_cfg = _build_variant_config(
                base=base,
                experiment_name=f"{base_name}_{variant}",
                output_dir=variant_output,
                visual_encoder_checkpoint=visual_export_path,
                feature_cache_dir=feature_cache_dir,
                batch_size=int(batch_size),
                epochs=int(args.epochs),
                train_limit=train_limit,
                val_limit=val_limit,
                num_workers=int(args.num_workers),
                log_every_steps=int(args.log_every_steps),
                save_every_steps=int(args.save_every_steps),
                precompute_features_if_missing=False,
            )
            variant_config_path = generated_dir / f"{variant}.yaml"
            variant_config_path.write_text(yaml.safe_dump(variant_cfg, sort_keys=False))
            start = time.time()
            status = "ok"
            error = ""
            try:
                if int(world_size) == 1:
                    monitor = _run_command(
                        [args.python_bin, str(ROOT / "apps" / "train_decoder.py"), "--config", str(variant_config_path)],
                        cwd=ROOT,
                        label=variant,
                        monitor_gpu=True,
                        gpu_sample_seconds=float(args.gpu_sample_seconds),
                    )
                else:
                    monitor = _run_command(
                        [
                            args.torchrun_bin,
                            "--standalone",
                            "--nproc_per_node",
                            str(world_size),
                            str(ROOT / "apps" / "train_decoder.py"),
                            "--config",
                            str(variant_config_path),
                        ],
                        cwd=ROOT,
                        label=variant,
                        monitor_gpu=True,
                        gpu_sample_seconds=float(args.gpu_sample_seconds),
                    )
            except subprocess.CalledProcessError as exc:
                status = "failed"
                error = f"returncode={exc.returncode}"
                monitor = getattr(exc, "monitor", {}) if hasattr(exc, "monitor") else {}
            elapsed = time.time() - start
            result = _summarize_variant(
                variant=variant,
                world_size=int(world_size),
                batch_size=int(batch_size),
                output_dir=variant_output,
                elapsed_seconds=float(elapsed),
                status=status,
                error=error,
                monitor=monitor,
            )
            results.append(result)
            print(json.dumps(result, indent=2, sort_keys=True))

    summary = {
        "base_config": str(base_config_path),
        "encoder_config": str(encoder_config_path),
        "encoder_checkpoint": str(encoder_checkpoint_path),
        "visual_encoder_export": str(visual_export_path),
        "feature_cache_dir": str(feature_cache_dir),
        "train_limit": train_limit,
        "val_limit": val_limit,
        "results": results,
        "ranked_by_train_examples_per_second": sorted(
            results,
            key=lambda row: float(row.get("train_examples_per_second") or -1.0),
            reverse=True,
        ),
    }
    summary_path = output_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"summary_path": str(summary_path)}, indent=2, sort_keys=True))


def _parse_int_list(raw: str) -> list[int]:
    return [int(piece.strip()) for piece in str(raw).split(",") if piece.strip()]


def _build_variant_config(
    *,
    base: dict[str, Any],
    experiment_name: str,
    output_dir: Path,
    visual_encoder_checkpoint: Path,
    feature_cache_dir: Path,
    batch_size: int,
    epochs: int,
    train_limit: int,
    val_limit: int,
    num_workers: int,
    log_every_steps: int,
    save_every_steps: int,
    precompute_features_if_missing: bool,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg.setdefault("paths", {})
    cfg["paths"]["dataset_root"] = ""
    cfg["paths"]["output_dir"] = str(output_dir)
    cfg["paths"]["visual_encoder_checkpoint"] = str(visual_encoder_checkpoint)
    cfg["paths"]["feature_cache_dir"] = str(feature_cache_dir)
    cfg.setdefault("data", {})
    cfg["data"]["train_limit"] = int(train_limit)
    cfg["data"]["val_limit"] = int(val_limit)
    cfg.setdefault("training", {})
    cfg["training"]["batch_size"] = int(batch_size)
    cfg["training"]["epochs"] = int(epochs)
    cfg["training"]["num_workers"] = int(num_workers)
    cfg["training"]["log_every_steps"] = int(log_every_steps)
    cfg["training"]["save_every_steps"] = int(save_every_steps)
    cfg["training"]["precompute_features_if_missing"] = bool(precompute_features_if_missing)
    cfg.setdefault("logging", {})
    cfg["logging"]["experiment_name"] = str(experiment_name)
    cfg["logging"]["wandb_enabled"] = False
    cfg["logging"]["wandb_mode"] = "disabled"
    cfg["training"]["save_last_checkpoint"] = False
    cfg["training"]["save_best_checkpoint"] = False
    return cfg


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    label: str,
    monitor_gpu: bool = False,
    gpu_sample_seconds: float = 1.0,
) -> dict[str, Any]:
    print(f"[profile_decoder] start {label}: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=str(cwd))
    monitor: dict[str, Any] = {}
    try:
        if monitor_gpu:
            monitor = _monitor_process_gpu_memory(process, sample_seconds=max(float(gpu_sample_seconds), 0.2))
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        raise
    if returncode != 0:
        exc = subprocess.CalledProcessError(returncode, command)
        setattr(exc, "monitor", monitor)
        raise exc
    print(f"[profile_decoder] done {label}", flush=True)
    return monitor


def _summarize_variant(
    *,
    variant: str,
    world_size: int,
    batch_size: int,
    output_dir: Path,
    elapsed_seconds: float,
    status: str,
    error: str,
    monitor: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "variant": variant,
        "world_size": int(world_size),
        "per_process_batch_size": int(batch_size),
        "global_batch_size": int(world_size) * int(batch_size),
        "output_dir": str(output_dir),
        "elapsed_seconds": float(elapsed_seconds),
        "status": status,
        "error": error,
        "gpu_monitor": monitor,
    }
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        return result
    rows = json.loads(metrics_path.read_text())
    if not rows:
        return result
    last = rows[-1]
    result["epochs_completed"] = int(len(rows))
    result["final_metrics"] = last
    config_snapshot_path = output_dir / "config_snapshot.json"
    if config_snapshot_path.is_file():
        snapshot = json.loads(config_snapshot_path.read_text())
        dataset_summary = snapshot.get("data", {})
        del dataset_summary
    train_examples = _extract_decoder_examples(output_dir, split="train")
    val_examples = _extract_decoder_examples(output_dir, split="val")
    result["train_examples"] = train_examples
    result["val_examples"] = val_examples
    if train_examples is not None and elapsed_seconds > 0.0:
        result["train_examples_per_second"] = float(train_examples) / float(elapsed_seconds)
    return result


def _monitor_process_gpu_memory(process: subprocess.Popen[Any], *, sample_seconds: float) -> dict[str, Any]:
    max_total_mb = 0
    max_per_gpu_mb: dict[str, int] = {}
    sample_count = 0
    while process.poll() is None:
        sample = _query_gpu_memory_by_pid_tree(process.pid)
        total_mb = int(sample.get("total_mb", 0))
        per_gpu = sample.get("per_gpu_mb", {})
        max_total_mb = max(max_total_mb, total_mb)
        for gpu, value in per_gpu.items():
            max_per_gpu_mb[gpu] = max(int(value), int(max_per_gpu_mb.get(gpu, 0)))
        sample_count += 1
        time.sleep(sample_seconds)
    final_sample = _query_gpu_memory_by_pid_tree(process.pid)
    total_mb = int(final_sample.get("total_mb", 0))
    per_gpu = final_sample.get("per_gpu_mb", {})
    max_total_mb = max(max_total_mb, total_mb)
    for gpu, value in per_gpu.items():
        max_per_gpu_mb[gpu] = max(int(value), int(max_per_gpu_mb.get(gpu, 0)))
    return {
        "sample_count": int(sample_count),
        "max_total_gpu_memory_mb": int(max_total_mb),
        "max_per_gpu_memory_mb": {str(key): int(value) for key, value in sorted(max_per_gpu_mb.items())},
    }


def _query_gpu_memory_by_pid_tree(root_pid: int) -> dict[str, Any]:
    pid_tree = _descendant_pids(int(root_pid))
    if not pid_tree:
        return {"total_mb": 0, "per_gpu_mb": {}}
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return {"total_mb": 0, "per_gpu_mb": {}}
    total_mb = 0
    per_gpu_mb: dict[str, int] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            gpu_uuid = str(parts[1])
            used_mb = int(parts[2])
        except ValueError:
            continue
        if pid not in pid_tree:
            continue
        total_mb += used_mb
        per_gpu_mb[gpu_uuid] = per_gpu_mb.get(gpu_uuid, 0) + used_mb
    return {"total_mb": int(total_mb), "per_gpu_mb": per_gpu_mb}


def _descendant_pids(root_pid: int) -> set[int]:
    try:
        output = subprocess.check_output(["ps", "-e", "-o", "pid=,ppid="], text=True)
    except Exception:
        return {int(root_pid)}
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    seen: set[int] = set()
    stack = [int(root_pid)]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, ()))
    return seen


def _extract_decoder_examples(output_dir: Path, *, split: str) -> int | None:
    config_snapshot_path = output_dir / "config_snapshot.json"
    if not config_snapshot_path.is_file():
        return None
    payload = json.loads(config_snapshot_path.read_text())
    train_limit = payload.get("data", {}).get("train_limit")
    val_limit = payload.get("data", {}).get("val_limit")
    limit = train_limit if split == "train" else val_limit
    if limit is None:
        return None
    # The decoder expands each study to one example per organ before repetition.
    organ_count = len(payload.get("data", {}).get("organ_names", []))
    if organ_count <= 0:
        return None
    return int(limit) * int(organ_count)


if __name__ == "__main__":
    main()
