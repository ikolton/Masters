"""Frozen Merlin image-feature cache utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .config import AblationConfig
from .data import build_datasets
from .modeling import MerlinReportTrainingWrapper


def build_image_embedding_cache(
    config: AblationConfig,
    *,
    force: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    """Cache frozen pre-adapter Merlin image features once per study/split."""
    from merlin.data import DataLoader

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.image_embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets(config)
    records_by_split = [
        (config.data.train_split, datasets.train_records),
        (config.data.val_split, datasets.val_records),
    ]
    unique_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for split, records in records_by_split:
        for record in records:
            key = (split, str(record["study_id"]))
            if key in seen:
                continue
            seen.add(key)
            cache_path = config.paths.image_embedding_cache_dir / split / f"{record['study_id']}.pt"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            unique_records.append(
                {
                    "image": record["image"],
                    "study_id": record["study_id"],
                    "split": split,
                    "cache_path": str(cache_path),
                }
            )

    shard_records = [
        record
        for index, record in enumerate(unique_records)
        if index % num_shards == shard_index
    ]
    to_process = [
        record
        for record in shard_records
        if force or not Path(str(record["cache_path"])).is_file()
    ]
    loader = DataLoader(
        datalist=to_process,
        cache_dir=str(config.paths.cache_dir / config.train.run_id / f"image_cache_build_shard{shard_index:03d}_of_{num_shards:03d}"),
        batchsize=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    model = MerlinReportTrainingWrapper(config).to(device)
    model.eval()
    start = time.time()
    written = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            features = model.encode_image_features(images).detach().cpu()
            study_ids = _string_list(batch["study_id"])
            splits = _string_list(batch["split"])
            cache_paths = _string_list(batch["cache_path"])
            for index, cache_path in enumerate(cache_paths):
                payload = {
                    "format": "merlin_ablation_image_feature_cache_v1",
                    "study_id": study_ids[index],
                    "split": splits[index],
                    "image_features": features[index].to(torch.bfloat16),
                }
                tmp_path = f"{cache_path}.tmp.{shard_index}"
                torch.save(payload, tmp_path)
                Path(tmp_path).replace(cache_path)
                written += 1
                if written % 10 == 0:
                    elapsed = max(time.time() - start, 1.0e-6)
                    print(
                        f"[merlin-cache] written={written}/{len(to_process)} "
                        f"rate={written / elapsed:.3f}/s",
                        flush=True,
                    )
    elapsed = max(time.time() - start, 1.0e-6)
    summary = {
        "cache_dir": str(config.paths.image_embedding_cache_dir),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "unique_records": len(unique_records),
        "shard_records": len(shard_records),
        "processed_records": len(to_process),
        "written_records": written,
        "elapsed_seconds": elapsed,
        "records_per_second": written / elapsed if written else 0.0,
    }
    summary_path = output_dir / f"image_embedding_cache_summary_shard{shard_index:03d}_of_{num_shards:03d}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[merlin-cache] summary={summary}", flush=True)
    return summary


def load_cached_image_features(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    paths = _string_list(batch["image_embedding"])
    features = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "image_features" in payload:
            features.append(payload["image_features"])
        elif isinstance(payload, dict) and "embedding" in payload:
            raise ValueError(f"Cache file uses obsolete adapter-projected embedding format: {path}")
        else:
            features.append(payload)
    return torch.stack(features, dim=0).to(device, non_blocking=True)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in list(value)]
