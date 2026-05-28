#!/usr/bin/env python3
"""Shard study-sampled organ-level GREEN for an existing benchmark directory."""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "apps"
if str(APPS) not in sys.path:
    sys.path.insert(0, str(APPS))

from evaluate_decoder_generations import _run_green_metric, _validate_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--study-fraction", type=float, default=0.10)
    parser.add_argument("--study-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--green-batch-size", type=int, default=32)
    parser.add_argument("--green-max-new-tokens", type=int, default=192)
    parser.add_argument("--green-prompt-max-length", type=int, default=2048)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    run_label = str(args.run_label)
    num_shards = int(args.num_shards)
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")

    sampled_dir = benchmark_dir / "sampled_green"
    shard_dir = sampled_dir / "shards"
    sampled_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_shards(
            benchmark_dir=benchmark_dir,
            run_label=run_label,
            num_shards=num_shards,
            sampled_dir=sampled_dir,
            shard_dir=shard_dir,
            green_batch_size=int(args.green_batch_size),
            green_max_new_tokens=int(args.green_max_new_tokens),
            green_prompt_max_length=int(args.green_prompt_max_length),
        )
        return

    if args.shard_index is None:
        raise ValueError("--shard-index is required unless --merge is set")
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    run_path = benchmark_dir / "runs" / run_label / "generations.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing generations for {run_label}: {run_path}")
    all_run_paths = [
        path
        for path in sorted((benchmark_dir / "runs").glob("*/generations.json"))
        if (path.parent / "evaluation.json").is_file()
    ]
    sample_keys, sample_manifest = build_sample_keys(
        all_run_paths,
        study_fraction=float(args.study_fraction),
        study_limit=args.study_limit,
        seed=int(args.seed),
    )
    manifest_path = sampled_dir / "sample_manifest.json"
    locked_write_json(manifest_path, sample_manifest)

    payload = json.loads(run_path.read_text(encoding="utf-8"))
    selected_rows = [row for row in payload.get("generations", []) if row_key(row) in sample_keys]
    selected_rows.sort(key=lambda row: sample_keys[row_key(row)])
    for row in selected_rows:
        row["_sample_index"] = sample_keys[row_key(row)]

    valid_rows, input_summary, warnings = _validate_rows(selected_rows)
    shard_rows = [row for row in valid_rows if int(row["_sample_index"]) % num_shards == shard_index]
    unavailable: dict[str, str] = {}
    _overall, sentence_scores = _run_green_metric(
        shard_rows,
        unavailable=unavailable,
        green_batch_size=int(args.green_batch_size),
        green_max_new_tokens=int(args.green_max_new_tokens),
        green_prompt_max_length=int(args.green_prompt_max_length),
    )
    green_scores = sentence_scores.get("GREEN", [])
    if len(green_scores) != len(shard_rows):
        raise RuntimeError(f"GREEN returned {len(green_scores)} scores for {len(shard_rows)} rows")

    shard_payload = {
        "benchmark_dir": str(benchmark_dir),
        "run_label": run_label,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "sample_manifest_path": str(manifest_path),
        "source_generation_path": str(run_path),
        "input_summary": input_summary,
        "warnings": warnings,
        "unavailable_metrics": unavailable,
        "green_batch_size": int(args.green_batch_size),
        "green_max_new_tokens": int(args.green_max_new_tokens),
        "green_prompt_max_length": int(args.green_prompt_max_length),
        "rows": [
            {
                "sample_index": int(row["_sample_index"]),
                "study_id": str(row.get("study_id", "")),
                "organ": str(row.get("organ", "")),
                "organ_abnormal_label": row.get("organ_abnormal_label"),
                "lesion_label": row.get("lesion_label"),
                "GREEN": float(score),
            }
            for row, score in zip(shard_rows, green_scores)
        ],
    }
    shard_path = shard_dir / f"{run_label}_shard{shard_index}of{num_shards}.json"
    locked_write_json(shard_path, shard_payload)
    print(
        f"[sampled-green-shard] {run_label} shard={shard_index}/{num_shards} "
        f"rows={len(shard_rows)} path={shard_path}",
        flush=True,
    )


def merge_shards(
    *,
    benchmark_dir: Path,
    run_label: str,
    num_shards: int,
    sampled_dir: Path,
    shard_dir: Path,
    green_batch_size: int,
    green_max_new_tokens: int,
    green_prompt_max_length: int,
) -> None:
    shard_paths = [shard_dir / f"{run_label}_shard{index}of{num_shards}.json" for index in range(num_shards)]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing GREEN shard outputs: " + ", ".join(missing))

    rows = []
    shard_payloads = []
    for path in shard_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard_payloads.append(payload)
        rows.extend(payload.get("rows", []))
    rows.sort(key=lambda row: int(row["sample_index"]))
    if len({int(row["sample_index"]) for row in rows}) != len(rows):
        raise RuntimeError(f"Duplicate sample indices found while merging {run_label}")

    overall = mean_metrics(rows)
    by_abnormal = {
        bucket: mean_metrics(bucket_rows) | {"count": len(bucket_rows)}
        for bucket, bucket_rows in group_by_binary(rows, "organ_abnormal_label").items()
        if bucket_rows
    }
    by_lesion = {
        bucket: mean_metrics(bucket_rows) | {"count": len(bucket_rows)}
        for bucket, bucket_rows in group_by_binary(rows, "lesion_label").items()
        if bucket_rows
    }
    by_organ = {
        organ: mean_metrics(bucket_rows) | {"count": len(bucket_rows)}
        for organ, bucket_rows in group_by_text(rows, "organ").items()
        if bucket_rows
    }
    sample_manifest_path = sampled_dir / "sample_manifest.json"
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8")) if sample_manifest_path.is_file() else {}
    sampled_generation_path = sampled_dir / f"{run_label}_sampled_green_sharded_rows.json"
    locked_write_json(
        sampled_generation_path,
        {
            "run_label": run_label,
            "num_shards": num_shards,
            "shards": [str(path) for path in shard_paths],
            "rows": rows,
        },
    )

    sampled_green_payload = {
        "sample_manifest": sample_manifest,
        "sampled_generation_path": str(sampled_generation_path),
        "sharded": True,
        "num_shards": num_shards,
        "green_batch_size": green_batch_size,
        "green_max_new_tokens": green_max_new_tokens,
        "green_prompt_max_length": green_prompt_max_length,
        "count": len(rows),
        "overall": overall,
        "per_organ": by_organ,
        "per_organ_aggregation": "mean_sharded_sentence_scores",
        "by_organ_abnormal_label": by_abnormal,
        "by_organ_abnormal_label_aggregation": "mean_sharded_sentence_scores",
        "by_lesion_label": by_lesion,
        "by_lesion_label_aggregation": "mean_sharded_sentence_scores",
        "abnormal": by_abnormal.get("positive", {}),
        "normal": by_abnormal.get("negative", {}),
        "shard_paths": [str(path) for path in shard_paths],
        "shard_warnings": {str(path): payload.get("warnings", []) for path, payload in zip(shard_paths, shard_payloads)},
        "shard_unavailable_metrics": {
            str(path): payload.get("unavailable_metrics", {}) for path, payload in zip(shard_paths, shard_payloads)
        },
    }
    evaluation_path = benchmark_dir / "runs" / run_label / "evaluation.json"
    locked_update_json(evaluation_path, {"sampled_green": sampled_green_payload})

    summary_path = sampled_dir / "summary_sharded.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"runs": {}}
    summary["runs"][run_label] = {
        "rows": len(rows),
        "evaluation_path": str(evaluation_path),
        "sampled_generation_path": str(sampled_generation_path),
        "green": overall.get("GREEN"),
        "normal_green": by_abnormal.get("negative", {}).get("GREEN"),
        "abnormal_green": by_abnormal.get("positive", {}).get("GREEN"),
        "num_shards": num_shards,
    }
    locked_write_json(summary_path, summary)
    print(
        f"[sampled-green-merge] {run_label} rows={len(rows)} "
        f"GREEN={overall.get('GREEN')} normal_GREEN={by_abnormal.get('negative', {}).get('GREEN')} "
        f"abnormal_GREEN={by_abnormal.get('positive', {}).get('GREEN')}",
        flush=True,
    )


def build_sample_keys(
    run_paths: list[Path],
    *,
    study_fraction: float,
    study_limit: int | None,
    seed: int,
) -> tuple[dict[tuple[str, str], int], dict[str, Any]]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
    row_maps = [{row_key(row): row for row in payload.get("generations", []) if row_key(row) is not None} for payload in payloads]
    common_keys = set(row_maps[0])
    for row_map in row_maps[1:]:
        common_keys &= set(row_map)
    common_study_ids = sorted({study_id for study_id, _organ in common_keys})
    if study_limit is not None:
        requested_study_count = max(0, min(int(study_limit), len(common_study_ids)))
    else:
        fraction = min(max(float(study_fraction), 0.0), 1.0)
        requested_study_count = min(math.ceil(len(common_study_ids) * fraction), len(common_study_ids))
    if requested_study_count <= 0 and common_study_ids:
        requested_study_count = 1
    rng = random.Random(int(seed))
    shuffled = list(common_study_ids)
    rng.shuffle(shuffled)
    selected_study_ids = shuffled[:requested_study_count]
    selected_study_order = {study_id: index for index, study_id in enumerate(selected_study_ids)}
    selected = sorted(
        (key for key in common_keys if key[0] in selected_study_order),
        key=lambda key: (selected_study_order[key[0]], key[1]),
    )
    order = {key: index for index, key in enumerate(selected)}
    first_rows = row_maps[0]
    selected_positive = [key for key in selected if safe_binary_label(first_rows[key].get("organ_abnormal_label")) == 1]
    selected_negative = [key for key in selected if safe_binary_label(first_rows[key].get("organ_abnormal_label")) == 0]
    manifest = {
        "seed": int(seed),
        "sampling_mode": "study_fraction_all_organs",
        "requested_study_fraction": float(study_fraction),
        "requested_study_limit": study_limit,
        "available_common_study_count": len(common_study_ids),
        "selected_study_count": len(selected_study_ids),
        "selected_study_ids": selected_study_ids,
        "selected_positive_count": len(selected_positive),
        "selected_negative_count": len(selected_negative),
        "selected_total_count": len(selected),
        "sample_key_type": ["study_id", "organ"],
        "selected_keys": [{"study_id": key[0], "organ": key[1]} for key in selected],
        "source_runs": [str(path) for path in run_paths],
    }
    return order, manifest


def row_key(row: Any) -> tuple[str, str] | None:
    if not isinstance(row, dict):
        return None
    study_id = str(row.get("study_id", "")).strip()
    organ = str(row.get("organ", "")).strip()
    if not study_id or not organ:
        return None
    return (study_id, organ)


def safe_binary_label(value: Any) -> int | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return 1 if as_float > 0.5 else 0


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(row["GREEN"]) for row in rows if row.get("GREEN") is not None]
    return {"GREEN": sum(values) / len(values)} if values else {}


def group_by_binary(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": []}
    for row in rows:
        label = safe_binary_label(row.get(key))
        if label is None:
            continue
        buckets["positive" if label == 1 else "negative"].append(row)
    return buckets


def group_by_text(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key, "")).strip()
        if value:
            buckets[value].append(row)
    return dict(buckets)


def locked_update_json(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        payload.update(updates)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return payload


def locked_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
