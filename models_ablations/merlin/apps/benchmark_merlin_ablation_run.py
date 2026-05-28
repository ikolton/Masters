#!/usr/bin/env python
"""Generate and evaluate one Merlin ablation checkpoint on the native test set."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
APPS_ROOT = ROOT / "apps"
for candidate in (SRC_ROOT, APPS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch
from torch.utils.data import DataLoader
try:
    from transformers.utils import logging as hf_logging
except Exception:  # pragma: no cover - benchmark can run without direct transformers import.
    hf_logging = None

from evaluate_decoder_generations import DEFAULT_COCO_METRICS, evaluate_file
from benchmark_decoder_checkpoints import (
    RunSpec,
    _build_comparison_row,
    _write_csv,
    _write_markdown,
)
from merlin_ablation.cache import load_cached_image_features
from merlin_ablation.config import DEFAULT_ORGANS, load_config
from merlin_ablation.data import PROMPT_ORGAN_ALIASES
from merlin_ablation.modeling import MerlinReportTrainingWrapper
from merlin_ablation.train import _prepare_imports, _simple_collate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-combined-json", default="outputs/datasets/merlin_test_native/test/combined.json")
    parser.add_argument(
        "--image-cache-dir",
        default="outputs/models_ablations/merlin/image_embedding_cache_merlin_test_native",
    )
    parser.add_argument("--output-dir", default="outputs/models_ablations/merlin/benchmark_test_full_basic")
    parser.add_argument("--study-fraction", type=float, default=0.0)
    parser.add_argument("--study-limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--metrics", default=",".join(DEFAULT_COCO_METRICS))
    parser.add_argument("--tokenize", choices=("auto", "java", "none"), default="auto")
    parser.add_argument("--green", action="store_true")
    parser.add_argument("--green-scope", choices=("organ", "study", "both"), default="organ")
    parser.add_argument("--no-study-level", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = output_dir / "runs" / _slugify(args.label)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_manifest.json"
    generation_path = run_dir / "generations.json"
    evaluation_path = run_dir / "evaluation.json"

    test_combined_json = _resolve_path(args.test_combined_json)
    image_cache_dir = _resolve_path(args.image_cache_dir)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    manifest = _load_or_create_manifest(
        manifest_path=manifest_path,
        test_combined_json=test_combined_json,
        seed=int(args.seed),
        study_fraction=float(args.study_fraction),
        study_limit=args.study_limit,
    )
    requested = {
        "label": str(args.label),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "test_combined_json": str(test_combined_json),
        "image_cache_dir": str(image_cache_dir),
        "sample_manifest_digest": manifest["sample_manifest_digest"],
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "num_beams": int(args.num_beams),
        "repetition_penalty": float(args.repetition_penalty),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
    }
    if args.force_generate or not _generation_matches(generation_path, requested):
        print(f"[merlin-benchmark] generating label={args.label}", flush=True)
        payload = generate_merlin_ablation(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            test_combined_json=test_combined_json,
            image_cache_dir=image_cache_dir,
            selected_study_ids=set(manifest["study_ids"]),
            label=str(args.label),
            requested=requested,
            num_shards=int(args.num_shards),
            shard_index=int(args.shard_index),
        )
        generation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[merlin-benchmark] wrote generations={generation_path}", flush=True)

    if not args.skip_eval and (args.force_eval or not _evaluation_matches(evaluation_path, generation_path, manifest)):
        print(f"[merlin-benchmark] evaluating label={args.label}", flush=True)
        evaluation = evaluate_file(
            generation_path,
            metrics=_parse_csv(args.metrics),
            tokenize_mode=str(args.tokenize),
            green_scope=str(args.green_scope) if bool(args.green) else "none",
            limit=None,
            include_study_level=not bool(args.no_study_level),
        )
        evaluation["label"] = str(args.label)
        evaluation["generation_path"] = str(generation_path)
        evaluation["sample_manifest_digest"] = manifest["sample_manifest_digest"]
        evaluation["requested_metrics"] = _parse_csv(args.metrics)
        _locked_update_json(evaluation_path, evaluation)
        print(f"[merlin-benchmark] wrote evaluation={evaluation_path}", flush=True)

    _write_single_run_summary(output_dir)


@torch.no_grad()
def generate_merlin_ablation(
    *,
    config_path: Path,
    checkpoint_path: Path,
    test_combined_json: Path,
    image_cache_dir: Path,
    selected_study_ids: set[str],
    label: str,
    requested: dict[str, Any],
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    config = load_config(config_path)
    _prepare_imports(config)
    if hf_logging is not None:
        hf_logging.set_verbosity_error()
    rows = _build_generation_records(
        test_combined_json=test_combined_json,
        image_cache_dir=image_cache_dir,
        selected_study_ids=selected_study_ids,
    )
    rows = _select_shard(rows, num_shards=num_shards, shard_index=shard_index)
    _assert_cache_exists(rows)
    print(
        f"[merlin-benchmark] rows={len(rows)} shard={shard_index}/{num_shards}",
        flush=True,
    )

    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    model = MerlinReportTrainingWrapper(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _load_model_checkpoint(model, checkpoint)
    del checkpoint
    model.eval()

    loader = DataLoader(rows, batch_size=int(requested["batch_size"]), shuffle=False, num_workers=0, collate_fn=_simple_collate)
    generated_rows: list[dict[str, Any]] = []
    start = time.time()
    for batch_index, batch in enumerate(loader, start=1):
        image_features = load_cached_image_features(batch, device)
        prompts = _string_list(batch["prompt"])
        generations = model.generate(
            image_features=image_features,
            prompts=prompts,
            do_sample=False,
            num_beams=int(requested["num_beams"]),
            repetition_penalty=float(requested["repetition_penalty"]),
            max_new_tokens=int(requested["max_new_tokens"]),
            min_new_tokens=int(requested.get("min_new_tokens", 0)),
        )
        study_ids = _string_list(batch["study_id"])
        organs = _string_list(batch["organ"])
        targets = _string_list(batch["target"])
        lesion_labels = _float_list(batch["lesion_label"])
        organ_labels = _int_list(batch["organ_abnormal_label"])
        for index, generated in enumerate(generations):
            generated_rows.append(
                {
                    "study_id": study_ids[index],
                    "organ": organs[index],
                    "target": targets[index],
                    "generated": generated,
                    "lesion_label": lesion_labels[index],
                    "organ_abnormal_label": organ_labels[index],
                }
            )
        if batch_index % 10 == 0 or len(generated_rows) == len(rows):
            elapsed = max(time.time() - start, 1.0e-6)
            rate = len(generated_rows) / elapsed
            remaining = max(len(rows) - len(generated_rows), 0)
            print(
                f"[merlin-benchmark] batch={batch_index} rows={len(generated_rows)}/{len(rows)} "
                f"rate={rate:.3f}/s eta={_format_duration(remaining / max(rate, 1.0e-9))}",
                flush=True,
            )
    elapsed = max(time.time() - start, 1.0e-6)
    return {
        "format": "merlin_ablation_generations_v1",
        "label": label,
        "split": "test",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "requested": requested,
        "elapsed_seconds": elapsed,
        "rows_per_second": len(generated_rows) / elapsed if generated_rows else 0.0,
        "generations": generated_rows,
    }


def _build_generation_records(
    *,
    test_combined_json: Path,
    image_cache_dir: Path,
    selected_study_ids: set[str],
) -> list[dict[str, Any]]:
    payload = json.loads(test_combined_json.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for item in payload:
        study_id = str(item.get("study_id", "")).strip()
        if not study_id or study_id not in selected_study_ids:
            continue
        findings = dict(item.get("findings", {}))
        labels = dict(item.get("labels", {}))
        for organ in DEFAULT_ORGANS:
            target = str(findings.get(organ, "")).strip()
            if not target:
                continue
            prompt_organ = PROMPT_ORGAN_ALIASES.get(organ, organ.lower())
            label_value = labels.get(organ)
            records.append(
                {
                    "image_embedding": str(image_cache_dir / "test" / f"{study_id}.pt"),
                    "study_id": study_id,
                    "organ": organ,
                    "prompt": f"Generate a radiology report for {prompt_organ}###\n",
                    "target": target,
                    "lesion_label": float(label_value) if str(label_value).strip() in {"0", "1"} else 0.0,
                    "organ_abnormal_label": int(label_value) if str(label_value).strip() in {"0", "1"} else -1,
                }
            )
    return records


def _select_shard(rows: list[dict[str, Any]], *, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    num_shards = int(num_shards)
    shard_index = int(shard_index)
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if num_shards == 1:
        return rows
    return [
        row
        for row in rows
        if (_stable_int(str(row["study_id"])) % num_shards) == shard_index
    ]


def _stable_int(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def _load_or_create_manifest(
    *,
    manifest_path: Path,
    test_combined_json: Path,
    seed: int,
    study_fraction: float,
    study_limit: int | None,
) -> dict[str, Any]:
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            str(existing.get("test_combined_json", "")) == str(test_combined_json)
            and int(existing.get("seed", -1)) == int(seed)
            and float(existing.get("study_fraction", -1.0)) == float(study_fraction)
            and existing.get("study_limit") == (None if study_limit is None else int(study_limit))
        ):
            return existing

    payload = json.loads(test_combined_json.read_text(encoding="utf-8"))
    study_ids = sorted({str(item.get("study_id", "")).strip() for item in payload if str(item.get("study_id", "")).strip()})
    shuffled = list(study_ids)
    random.Random(int(seed)).shuffle(shuffled)
    selected_count = len(shuffled)
    if study_fraction > 0.0:
        selected_count = max(1 if shuffled else 0, int(len(shuffled) * float(study_fraction) + 0.999999))
    if study_limit is not None:
        selected_count = min(selected_count, int(study_limit))
    selected = shuffled[:selected_count]
    digest = hashlib.sha1(
        json.dumps(
            {
                "test_combined_json": str(test_combined_json),
                "seed": int(seed),
                "study_fraction": float(study_fraction),
                "study_limit": None if study_limit is None else int(study_limit),
                "study_ids": selected,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "test_combined_json": str(test_combined_json),
        "seed": int(seed),
        "study_fraction": float(study_fraction),
        "study_limit": None if study_limit is None else int(study_limit),
        "source_study_count": len(study_ids),
        "selected_study_count": len(selected),
        "study_ids": selected,
        "sample_manifest_digest": digest,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _write_single_run_summary(output_dir: Path) -> None:
    manifest_path = output_dir / "sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest.setdefault("split", "test")
    rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for evaluation_path in sorted((output_dir / "runs").glob("*/evaluation.json")):
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        generation_path = Path(str(payload.get("generation_path", "")))
        if not generation_path.is_file():
            continue
        generation_payload = json.loads(generation_path.read_text(encoding="utf-8"))
        requested = generation_payload.get("requested", {})
        run_spec = RunSpec(
            label=str(payload.get("label", evaluation_path.parent.name)),
            config_path=Path(str(requested.get("config", generation_payload.get("config", "")))),
            checkpoint_path=Path(str(generation_payload.get("checkpoint", requested.get("checkpoint", "")))),
        )
        row = _build_comparison_row(
            run_spec=run_spec,
            generation_payload=generation_payload,
            evaluation_payload=payload,
            manifest=manifest,
        )
        rows.append(row)
        run_summaries.append(
            {
                "label": run_spec.label,
                "config": str(run_spec.config_path),
                "checkpoint": str(run_spec.checkpoint_path),
                "generation_path": str(generation_path),
                "evaluation_path": str(evaluation_path),
                "row": row,
                "warnings": payload.get("warnings", []),
                "unavailable_metrics": payload.get("unavailable_metrics", {}),
            }
        )
    summary = {
        "output_dir": str(output_dir),
        "sample_manifest": manifest,
        "runs": run_summaries,
        "comparison_rows": rows,
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "comparison_summary.csv", rows)
    _write_markdown(output_dir / "comparison_summary.md", rows)


def _locked_update_json(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        payload.update(updates)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(path)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return payload


def _assert_cache_exists(rows: list[dict[str, Any]]) -> None:
    missing = []
    seen = set()
    for row in rows:
        path = str(row["image_embedding"])
        if path in seen:
            continue
        seen.add(path)
        if not Path(path).is_file():
            missing.append(path)
            if len(missing) >= 10:
                break
    if missing:
        raise FileNotFoundError("Missing native test image cache files:\n" + "\n".join(missing))


def _load_model_checkpoint(model: MerlinReportTrainingWrapper, checkpoint: dict[str, Any]) -> None:
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        return
    if "model_trainable" in checkpoint:
        result = model.load_state_dict(checkpoint["model_trainable"], strict=False)
        unexpected = list(result.unexpected_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected trainable-only checkpoint keys: {unexpected[:20]}")
        print(
            "[merlin-benchmark] loaded trainable-only checkpoint "
            f"missing_base_keys={len(result.missing_keys)}",
            flush=True,
        )
        return
    raise KeyError("Checkpoint must contain either 'model' or 'model_trainable'.")


def _generation_matches(path: Path, requested: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("requested") == requested and isinstance(payload.get("generations"), list) and bool(payload["generations"])


def _evaluation_matches(path: Path, generation_path: Path, manifest: dict[str, Any]) -> bool:
    if not path.is_file() or not generation_path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        str(payload.get("generation_path", "")) == str(generation_path)
        and str(payload.get("sample_manifest_digest", "")) == str(manifest["sample_manifest_digest"])
        and path.stat().st_mtime >= generation_path.stat().st_mtime
    )


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_i = int(seconds)
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(value).lower()).strip("-") or "run"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in list(value)]


def _float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().flatten().tolist()]
    return [float(item) for item in value]


def _int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().flatten().tolist()]
    return [int(item) for item in value]


if __name__ == "__main__":
    main()
