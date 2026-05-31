#!/usr/bin/env python3
"""Compute study-sampled organ-level GREEN for an existing benchmark directory.

This evaluates all organ rows from the same fixed study-level test subset for
every run and stores the result inside each run's evaluation.json under
`sampled_green`.
"""

from __future__ import annotations

import argparse
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

from evaluate_decoder_generations import evaluate_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, help="Benchmark directory with runs/*/generations.json.")
    parser.add_argument("--study-fraction", type=float, default=0.10, help="Fraction of common test studies to sample.")
    parser.add_argument("--study-limit", type=int, default=None, help="Optional exact number of common studies to sample.")
    parser.add_argument(
        "--positive-count",
        type=int,
        default=None,
        help="Deprecated; ignored. GREEN now samples studies, then reports normal/abnormal strata from that same subset.",
    )
    parser.add_argument(
        "--negative-count",
        type=int,
        default=None,
        help="Deprecated; ignored. GREEN now samples studies, then reports normal/abnormal strata from that same subset.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed.")
    parser.add_argument(
        "--sample-manifest",
        default=None,
        help="Pin the GREEN subset to an existing sample_manifest.json (uses its selected_keys). "
        "Skips study-fraction/seed sampling so a later/parallel run scores the IDENTICAL subset "
        "as the runs already in the table. Does not overwrite the existing manifest file.",
    )
    parser.add_argument("--green-batch-size", type=int, default=32)
    parser.add_argument("--green-max-new-tokens", type=int, default=192)
    parser.add_argument("--green-prompt-max-length", type=int, default=2048)
    parser.add_argument("--run-labels", default="", help="Comma-separated run directory names to evaluate. Defaults to all.")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    run_paths = sorted((benchmark_dir / "runs").glob("*/generations.json"))
    requested_labels = {label.strip() for label in str(args.run_labels).split(",") if label.strip()}
    if requested_labels:
        run_paths = [path for path in run_paths if path.parent.name in requested_labels]
    if not run_paths:
        raise FileNotFoundError(f"No generation files found under {benchmark_dir / 'runs'}")

    if args.sample_manifest:
        # Pin the subset to an existing manifest so a parallel/follow-up GREEN run
        # scores the EXACT same rows as the runs already in the table.
        manifest_in = json.loads(Path(args.sample_manifest).expanduser().read_text(encoding="utf-8"))
        selected = manifest_in.get("selected_keys", [])
        if not selected:
            raise ValueError(f"--sample-manifest has no selected_keys: {args.sample_manifest}")
        sample_keys = {
            (str(item["study_id"]).strip(), str(item["organ"]).strip()): idx
            for idx, item in enumerate(selected)
        }
        sample_manifest = manifest_in
    else:
        sample_keys, sample_manifest = _build_sample_keys(
            run_paths,
            study_fraction=float(args.study_fraction),
            study_limit=args.study_limit,
            seed=int(args.seed),
        )
    sampled_dir = benchmark_dir / "sampled_green"
    sampled_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sampled_dir / "sample_manifest.json"
    if not args.sample_manifest:
        manifest_path.write_text(json.dumps(sample_manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary: dict[str, Any] = {
        "benchmark_dir": str(benchmark_dir),
        "sample_manifest": str(manifest_path),
        "runs": {},
    }
    for generation_path in run_paths:
        run_label = generation_path.parent.name
        payload = json.loads(generation_path.read_text(encoding="utf-8"))
        rows = [row for row in payload.get("generations", []) if _row_key(row) in sample_keys]
        rows.sort(key=lambda row: sample_keys[_row_key(row)])

        sampled_generation_path = sampled_dir / f"{run_label}_sampled_generations.json"
        sampled_generation_path.write_text(
            json.dumps(
                {
                    "source_generation_path": str(generation_path),
                    "sample_manifest_path": str(manifest_path),
                    "generations": rows,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        sampled_eval = evaluate_file(
            sampled_generation_path,
            metrics=[],
            tokenize_mode="none",
            green_scope="organ",
            limit=None,
            include_study_level=False,
            green_batch_size=int(args.green_batch_size),
            green_max_new_tokens=int(args.green_max_new_tokens),
            green_prompt_max_length=int(args.green_prompt_max_length),
        )
        evaluation_path = generation_path.parent / "evaluation.json"
        sampled_green_payload = {
            "sample_manifest": sample_manifest,
            "sampled_generation_path": str(sampled_generation_path),
            "green_batch_size": int(args.green_batch_size),
            "green_max_new_tokens": int(args.green_max_new_tokens),
            "green_prompt_max_length": int(args.green_prompt_max_length),
            "count": sampled_eval.get("organ_level", {}).get("count", len(rows)),
            "overall": sampled_eval.get("organ_level", {}).get("overall", {}),
            "by_organ_abnormal_label": sampled_eval.get("organ_level", {}).get("by_organ_abnormal_label", {}),
        }
        label_groups = sampled_green_payload["by_organ_abnormal_label"]
        if isinstance(label_groups, dict):
            sampled_green_payload["abnormal"] = label_groups.get("positive", {})
            sampled_green_payload["normal"] = label_groups.get("negative", {})
        evaluation_payload = _locked_update_json(evaluation_path, {"sampled_green": sampled_green_payload})
        summary["runs"][run_label] = {
            "rows": len(rows),
            "evaluation_path": str(evaluation_path),
            "sampled_generation_path": str(sampled_generation_path),
            "green": sampled_green_payload.get("overall", {}).get("GREEN"),
            "normal_green": sampled_green_payload.get("normal", {}).get("GREEN"),
            "abnormal_green": sampled_green_payload.get("abnormal", {}).get("GREEN"),
        }
        print(
            f"[sampled-green] {run_label} rows={len(rows)} "
            f"GREEN={summary['runs'][run_label]['green']} "
            f"normal_GREEN={summary['runs'][run_label]['normal_green']} "
            f"abnormal_GREEN={summary['runs'][run_label]['abnormal_green']}",
            flush=True,
        )

    summary_path = sampled_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_sample_keys(
    run_paths: list[Path],
    *,
    study_fraction: float,
    study_limit: int | None,
    seed: int,
) -> tuple[dict[tuple[str, str], int], dict[str, Any]]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
    row_maps = []
    for payload in payloads:
        row_maps.append({_row_key(row): row for row in payload.get("generations", []) if _row_key(row) is not None})
    common_keys = set(row_maps[0])
    for row_map in row_maps[1:]:
        common_keys &= set(row_map)

    first_rows = row_maps[0]
    common_study_ids = sorted({study_id for study_id, _organ in common_keys})
    if study_limit is not None:
        requested_study_count = max(0, min(int(study_limit), len(common_study_ids)))
    else:
        fraction = min(max(float(study_fraction), 0.0), 1.0)
        requested_study_count = min(math.ceil(len(common_study_ids) * fraction), len(common_study_ids))
    if requested_study_count <= 0 and common_study_ids:
        requested_study_count = 1

    rng = random.Random(int(seed))
    shuffled_study_ids = list(common_study_ids)
    rng.shuffle(shuffled_study_ids)
    selected_study_ids = shuffled_study_ids[:requested_study_count]
    selected_study_order = {study_id: index for index, study_id in enumerate(selected_study_ids)}
    selected = sorted(
        (key for key in common_keys if key[0] in selected_study_order),
        key=lambda key: (selected_study_order[key[0]], key[1]),
    )
    order = {key: index for index, key in enumerate(selected)}
    selected_positive = [key for key in selected if _safe_binary_label(first_rows[key].get("organ_abnormal_label")) == 1]
    selected_negative = [key for key in selected if _safe_binary_label(first_rows[key].get("organ_abnormal_label")) == 0]
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


def _row_key(row: Any) -> tuple[str, str] | None:
    if not isinstance(row, dict):
        return None
    study_id = str(row.get("study_id", "")).strip()
    organ = str(row.get("organ", "")).strip()
    if not study_id or not organ:
        return None
    return (study_id, organ)


def _safe_binary_label(value: Any) -> int | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return 1 if as_float > 0.5 else 0


def _locked_update_json(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
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


if __name__ == "__main__":
    main()
