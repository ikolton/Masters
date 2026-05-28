#!/usr/bin/env python
"""Profile Merlin ablation throughput from a config."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from merlin_ablation.config import load_config
from merlin_ablation.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--mode", choices=["cached", "online"], default="cached")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    results = []
    for batch_size in [int(value) for value in args.batch_sizes.split(",") if value.strip()]:
        raw = copy.deepcopy(base.raw)
        raw.setdefault("train", {})["batch_size"] = batch_size
        raw.setdefault("train", {})["max_steps"] = int(args.max_steps)
        raw.setdefault("train", {})["run_id"] = f"{base.train.run_id}_profile_{args.mode}_bs{batch_size}"
        raw.setdefault("model", {})["image_embedding_mode"] = args.mode
        if args.train_split is not None:
            raw.setdefault("data", {})["train_split"] = str(args.train_split)
        if args.val_split is not None:
            raw.setdefault("data", {})["val_split"] = str(args.val_split)
        if args.train_limit is not None:
            raw.setdefault("data", {})["train_limit"] = int(args.train_limit)
        if args.val_limit is not None:
            raw.setdefault("data", {})["val_limit"] = int(args.val_limit)
        if args.max_length is not None:
            raw.setdefault("model", {})["max_length"] = int(args.max_length)
        if args.grad_accum_steps is not None:
            raw.setdefault("train", {})["grad_accum_steps"] = int(args.grad_accum_steps)
        if args.no_save_checkpoint:
            raw.setdefault("train", {})["save_checkpoint"] = False
            raw.setdefault("train", {})["save_trainable_checkpoint"] = False
        tmp_path = base.output_dir / f"profile_config_{args.mode}_bs{batch_size}.yaml"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        import yaml

        tmp_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        config = load_config(tmp_path)
        gpu_sampler = _GpuSampler()
        gpu_sampler.start()
        started = time.time()
        summary = run_training(config)
        elapsed = max(time.time() - started, 1.0e-6)
        gpu = gpu_sampler.stop()
        metrics_path = config.output_dir / "metrics.jsonl"
        train_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if '"phase": "train"' in line]
        last_rate = train_rows[-1].get("examples_per_second", 0.0) if train_rows else 0.0
        results.append(
            {
                "batch_size": batch_size,
                "run_id": config.train.run_id,
                "elapsed_seconds": elapsed,
                "last_examples_per_second": last_rate,
                "gpu": gpu,
                "summary": summary,
            }
        )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = base.output_dir / f"profile_results_{args.mode}_{timestamp}.json"
    latest_path = base.output_dir / f"profile_results_{args.mode}_latest.json"
    serialized = json.dumps(results, indent=2, sort_keys=True) + "\n"
    result_path.write_text(serialized, encoding="utf-8")
    latest_path.write_text(serialized, encoding="utf-8")
    print(f"[merlin-profile] saved {result_path}")
    print(f"[merlin-profile] latest {latest_path}")
    for row in results:
        gpu = row["gpu"]
        per_gpu = ",".join(
            f"{idx}:{stats.get('util_avg', 0.0):.0f}%/{stats.get('mem_used_peak_gb', 0.0):.1f}GB"
            for idx, stats in sorted(gpu.get("per_gpu", {}).items())
        )
        print(
            f"[merlin-profile] bs={row['batch_size']} rate={row['last_examples_per_second']:.3f}/s "
            f"gpu_avg={gpu.get('util_avg', 0.0):.1f}% gpu_peak={gpu.get('util_peak', 0.0):.1f}% "
            f"mem_peak={gpu.get('mem_used_peak_gb', 0.0):.1f}GB per_gpu={per_gpu} run={row['run_id']}"
        )


class _GpuSampler:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join(timeout=max(self.interval_seconds * 2.0, 2.0))
        if not self._samples:
            return {"samples": 0, "util_avg": 0.0, "util_peak": 0.0, "mem_used_peak_gb": 0.0}
        utils = [sample["util"] for sample in self._samples]
        mems = [sample["mem_used_gb"] for sample in self._samples]
        return {
            "samples": len(self._samples),
            "util_avg": sum(utils) / len(utils),
            "util_peak": max(utils),
            "mem_used_peak_gb": max(mems),
            "per_gpu": self._per_gpu_summary(),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def _sample_once(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return
        utils = []
        mems = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                index = int(parts[0])
                util = float(parts[1])
                mem_used_gb = float(parts[2]) / 1024.0
                utils.append(util)
                mems.append(mem_used_gb)
                self._samples.append({"gpu_index": float(index), "util": util, "mem_used_gb": mem_used_gb})
            except ValueError:
                continue

    def _per_gpu_summary(self) -> dict[str, dict[str, float]]:
        grouped: dict[int, list[dict[str, float]]] = {}
        for sample in self._samples:
            grouped.setdefault(int(sample["gpu_index"]), []).append(sample)
        summary = {}
        for index, samples in grouped.items():
            utils = [sample["util"] for sample in samples]
            mems = [sample["mem_used_gb"] for sample in samples]
            summary[str(index)] = {
                "samples": float(len(samples)),
                "util_avg": sum(utils) / len(utils),
                "util_peak": max(utils),
                "mem_used_peak_gb": max(mems),
            }
        return summary


if __name__ == "__main__":
    main()
