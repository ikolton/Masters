#!/usr/bin/env python3
"""Create decoder benchmark config copies for the native Merlin test dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RUNS = {
    "nodiag_ce_only": "configs/decoder/train/train_qwen05b_per_organ_binary_diag_strong_best6_gh200_1gpu_bs64_10ep_sched_norepeat_nodiag.yaml",
    "semantic_minimal_v3": "configs/decoder/train/train_qwen05b_per_organ_semantic_minimal_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "semantic_family_subtype_v3": "configs/decoder/train/train_qwen05b_per_organ_semantic_family_subtype_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "lexical_diag_v1_w001": "configs/decoder/train/train_qwen05b_per_organ_lexical_diag_v1_w001_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "lexical_diag_v1_w002": "configs/decoder/train/train_qwen05b_per_organ_lexical_diag_v1_w002_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "lexical_diag_v1_w005": "configs/decoder/train/train_qwen05b_per_organ_lexical_diag_v1_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "lexw002_sem_normality_w002": "configs/decoder/train/train_qwen05b_per_organ_lexw002_sem_normality_w002_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "lexw002_sem_family_w002": "configs/decoder/train/train_qwen05b_per_organ_lexw002_sem_family_w002_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "lexw002_sem_primary_secondary_w002": "configs/decoder/train/train_qwen05b_per_organ_lexw002_sem_primary_secondary_w002_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "sem_normality_w005": "configs/decoder/train/train_qwen05b_per_organ_sem_normality_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "sem_family_w005": "configs/decoder/train/train_qwen05b_per_organ_sem_family_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
    "sem_primary_secondary_w005": "configs/decoder/train/train_qwen05b_per_organ_sem_primary_secondary_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="outputs/datasets/merlin_test_native",
        help="Native test dataset root to inject into each config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/decoder/benchmark_test_full_native/configs",
        help="Directory where benchmark-specific config copies are written.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default="outputs/decoder/feature_cache_merlin_test_native/best6_gh200",
        help="Shared feature cache directory for the native test dataset.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Optional run mapping label::base_config. Repeat to override defaults.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    feature_cache_dir = Path(args.feature_cache_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _parse_runs(args.run) if args.run else dict(DEFAULT_RUNS)
    manifest: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "feature_cache_dir": str(feature_cache_dir),
        "configs": {},
    }
    for label, base_config in runs.items():
        base_path = Path(base_config).expanduser().resolve()
        with base_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected mapping config in {base_path}")

        paths = payload.setdefault("paths", {})
        if not isinstance(paths, dict):
            raise ValueError(f"paths must be a mapping in {base_path}")
        paths["dataset_root"] = str(dataset_root)
        paths["feature_cache_dir"] = str(feature_cache_dir)
        paths["output_dir"] = str(output_dir.parent / "training_outputs_unused" / label)

        logging = payload.setdefault("logging", {})
        if isinstance(logging, dict):
            logging["wandb_enabled"] = False

        target_path = output_dir / f"{label}.yaml"
        with target_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        manifest["configs"][label] = {
            "base_config": str(base_path),
            "eval_config": str(target_path),
        }

    manifest_path = output_dir / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    print(yaml.safe_dump(manifest, sort_keys=False))


def _parse_runs(values: list[str]) -> dict[str, str]:
    runs: dict[str, str] = {}
    for value in values:
        parts = [part.strip() for part in str(value).split("::")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid --run value, expected label::base_config: {value}")
        runs[parts[0]] = parts[1]
    return runs


if __name__ == "__main__":
    main()
