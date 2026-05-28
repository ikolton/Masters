#!/usr/bin/env python
"""Cache Merlin image embeddings from a native image-root directory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from merlin_ablation.config import load_config
from merlin_ablation.modeling import MerlinReportTrainingWrapper
from merlin_ablation.train import _prepare_imports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing Merlin ablation config used only for model/env settings.")
    parser.add_argument("--image-root", required=True, help="Native root: <image-root>/<study_id>/<study_id>.nii.gz")
    parser.add_argument("--cache-dir", required=True, help="Output cache root.")
    parser.add_argument("--split-name", default="test", help="Cache split directory/name to write, default: test.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    _prepare_imports(config)
    summary = build_native_image_embedding_cache(
        config=config,
        image_root=Path(args.image_root).expanduser().resolve(),
        cache_dir=Path(args.cache_dir).expanduser().resolve(),
        split_name=args.split_name,
        force=args.force,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(f"[merlin-native-cache] summary={json.dumps(summary, sort_keys=True)}", flush=True)


def build_native_image_embedding_cache(
    *,
    config,
    image_root: Path,
    cache_dir: Path,
    split_name: str,
    force: bool,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    from merlin.data import DataLoader

    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing native image root: {image_root}")
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    all_records = []
    missing = []
    for study_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        study_id = study_dir.name
        image_path = study_dir / f"{study_id}.nii.gz"
        if not image_path.is_file():
            missing.append(str(image_path))
            continue
        cache_path = cache_dir / split_name / f"{study_id}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        all_records.append(
            {
                "image": str(image_path),
                "study_id": study_id,
                "split": split_name,
                "cache_path": str(cache_path),
            }
        )

    shard_records = [record for index, record in enumerate(all_records) if index % num_shards == shard_index]
    to_process = [record for record in shard_records if force or not Path(str(record["cache_path"])).is_file()]
    loader = DataLoader(
        datalist=to_process,
        cache_dir=str(config.paths.cache_dir / config.train.run_id / f"native_{split_name}_cache_shard{shard_index:03d}_of_{num_shards:03d}"),
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
                    "image_root": str(image_root),
                    "image_features": features[index].to(torch.bfloat16),
                }
                tmp_path = f"{cache_path}.tmp.{shard_index}"
                torch.save(payload, tmp_path)
                Path(tmp_path).replace(cache_path)
                written += 1
                if written % 10 == 0:
                    elapsed = max(time.time() - start, 1.0e-6)
                    print(
                        f"[merlin-native-cache] written={written}/{len(to_process)} "
                        f"rate={written / elapsed:.3f}/s",
                        flush=True,
                    )

    elapsed = max(time.time() - start, 1.0e-6)
    summary = {
        "cache_dir": str(cache_dir),
        "image_root": str(image_root),
        "split_name": split_name,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "found_records": len(all_records),
        "missing_image_count": len(missing),
        "missing_image_examples": missing[:20],
        "shard_records": len(shard_records),
        "processed_records": len(to_process),
        "written_records": written,
        "elapsed_seconds": elapsed,
        "records_per_second": written / elapsed if written else 0.0,
    }
    summary_path = cache_dir / f"native_{split_name}_cache_summary_shard{shard_index:03d}_of_{num_shards:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in list(value)]


if __name__ == "__main__":
    main()
