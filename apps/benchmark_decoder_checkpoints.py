#!/usr/bin/env python3
"""Generate and compare decoder checkpoints on a shared study subset."""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
APPS = ROOT / "apps"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(APPS) not in sys.path:
    sys.path.insert(0, str(APPS))

from organ_seg_clip.config import load_decoder_config
from organ_seg_clip.decoder.data import load_decoder_samples
from organ_seg_clip.evaluation.radgpt_oncology import (
    DEFAULT_LOCAL_JUDGE_MODEL,
    evaluate_generation_file_with_radgpt,
)

from evaluate_decoder_generations import (
    DEFAULT_COCO_METRICS,
    _build_per_organ_summary,
    _build_score_summary,
    evaluate_file,
)
from analyze_decoder_generations import _summarize as summarize_keyword_metrics
from generate_decoder import generate_generations


@dataclass(frozen=True)
class RunSpec:
    label: str
    config_path: Path
    checkpoint_path: Path


@dataclass
class RadGPTServerHandle:
    process: subprocess.Popen[str] | None
    log_path: Path | None
    launched: bool
    base_url: str

    def close(self) -> None:
        if not self.launched or self.process is None:
            return
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help="Run spec in the form 'label::config::checkpoint' or 'config::checkpoint'. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--split", default="val", help="Dataset split to benchmark.")
    parser.add_argument("--output-dir", required=True, help="Directory for manifest, generations, evals, and summaries.")
    parser.add_argument("--study-fraction", type=float, default=0.10, help="Fraction of unique studies to sample.")
    parser.add_argument("--study-limit", type=int, default=None, help="Hard cap on sampled studies after applying fraction.")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed shared across checkpoints.")
    parser.add_argument("--generation-batch-size", type=int, default=None, help="Optional generation batch size override.")
    parser.add_argument("--generation-num-workers", type=int, default=None, help="Optional generation num_workers override.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional generation max_new_tokens override.")
    parser.add_argument("--num-beams", type=int, default=None, help="Optional generation num_beams override.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Optional generation repetition penalty override.")
    parser.set_defaults(green=True, include_study_level=True, do_sample=None)
    parser.add_argument("--no-green", dest="green", action="store_false", help="Disable GREEN evaluation.")
    parser.add_argument("--green-scope", choices=("organ", "study", "both"), default="organ")
    parser.add_argument("--metrics", default=",".join(DEFAULT_COCO_METRICS), help="Comma-separated COCO-style metrics.")
    parser.add_argument("--tokenize", choices=("auto", "java", "none"), default="auto")
    parser.add_argument("--no-study-level", dest="include_study_level", action="store_false", help="Disable reconstructed study-level metrics.")
    parser.add_argument("--green-batch-size", type=int, default=32, help="GREEN judge batch size.")
    parser.add_argument("--green-max-new-tokens", type=int, default=192, help="GREEN judge max_new_tokens.")
    parser.add_argument("--green-prompt-max-length", type=int, default=2048, help="GREEN prompt truncation length.")
    parser.add_argument("--radgpt-eval", action="store_true", help="Run RadGPT oncology-subset evaluation (liver, kidneys, pancreas).")
    parser.add_argument("--radgpt-base-url", default="http://0.0.0.0:8000/v1", help="OpenAI-compatible API base URL for the RadGPT labeler.")
    parser.add_argument("--radgpt-root", default="/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadGPT", help="Path to the local RadGPT clone.")
    parser.add_argument("--radgpt-local-model", default=None, help="Use a local Hugging Face model via transformers instead of an OpenAI-compatible server.")
    parser.add_argument("--radgpt-local-model-dtype", default="bfloat16", help="Torch dtype for the local RadGPT transformers backend.")
    parser.add_argument("--radgpt-local-max-new-tokens", type=int, default=256, help="Generation max_new_tokens for the local RadGPT transformers backend.")
    parser.add_argument("--radgpt-auto-launch", action="store_true", help="Auto-launch a local vLLM OpenAI-compatible server for RadGPT if the endpoint is unavailable.")
    parser.add_argument("--radgpt-cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES to use for auto-launched RadGPT vLLM.")
    parser.add_argument("--radgpt-vllm-model", default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", help="Model name for auto-launched RadGPT vLLM.")
    parser.add_argument("--radgpt-vllm-command", default=None, help="Optional explicit path to the `vllm` executable for RadGPT auto-launch.")
    parser.add_argument("--radgpt-vllm-python", default=None, help="Optional Python interpreter path that has the `vllm` module installed; used for `python -m vllm.entrypoints.openai.api_server` auto-launch.")
    parser.add_argument("--radgpt-vllm-dtype", default="half", help="Dtype for auto-launched RadGPT vLLM.")
    parser.add_argument("--radgpt-vllm-tensor-parallel-size", type=int, default=1, help="Tensor parallel size for auto-launched RadGPT vLLM.")
    parser.add_argument("--radgpt-vllm-gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for auto-launched RadGPT vLLM.")
    parser.add_argument("--radgpt-vllm-max-model-len", type=int, default=60000, help="Max model length for auto-launched RadGPT vLLM.")
    parser.add_argument("--radgpt-vllm-log", default=None, help="Optional log file path for auto-launched RadGPT vLLM.")
    parser.add_argument("--radgpt-vllm-cache-dir", default=None, help="Optional HF cache directory for auto-launched RadGPT vLLM.")
    parser.add_argument("--radgpt-server-timeout", type=int, default=1800, help="Seconds to wait for an auto-launched RadGPT server to become ready.")
    parser.set_defaults(radgpt_fast=True)
    parser.add_argument("--radgpt-fast", dest="radgpt_fast", action="store_true", help="Use RadGPT fast prompts.")
    parser.add_argument("--radgpt-slow", dest="radgpt_fast", action="store_false", help="Use RadGPT slower, larger prompts.")
    parser.add_argument("--force-radgpt-reference", action="store_true", help="Recompute the shared RadGPT reference-label cache.")
    parser.add_argument("--force-radgpt-generated", action="store_true", help="Recompute per-checkpoint RadGPT generated-label caches.")
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", help="Override generation to sampling mode.")
    parser.add_argument("--no-sample", dest="do_sample", action="store_false", help="Override generation to greedy/beam mode.")
    parser.add_argument("--force-sample", action="store_true", help="Rebuild the shared study manifest.")
    parser.add_argument("--force-generate", action="store_true", help="Regenerate generations even if compatible outputs exist.")
    parser.add_argument("--force-eval", action="store_true", help="Recompute evaluations even if compatible outputs exist.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_specs = _parse_run_specs(args.runs)
    manifest = _load_or_create_manifest(
        manifest_path=output_dir / "sample_manifest.json",
        base_config_path=run_specs[0].config_path,
        split=str(args.split),
        seed=int(args.seed),
        study_fraction=float(args.study_fraction),
        study_limit=args.study_limit,
        force=bool(args.force_sample),
    )

    comparison_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    metric_list = _parse_csv(args.metrics)
    requested_green_scope = str(args.green_scope) if bool(args.green) else "none"
    radgpt_server: RadGPTServerHandle | None = None
    if args.radgpt_eval and not args.radgpt_local_model:
        radgpt_server = _ensure_radgpt_server(args, output_dir)
    for run_spec in run_specs:
        run_dir = output_dir / "runs" / _slugify(run_spec.label)
        run_dir.mkdir(parents=True, exist_ok=True)
        generation_path = run_dir / "generations.json"
        evaluation_path = run_dir / "evaluation.json"
        config = load_decoder_config(str(run_spec.config_path))

        if args.force_generate or not _generation_matches(generation_path, run_spec=run_spec, manifest=manifest):
            print(f"[benchmark] generating {run_spec.label}")
            generation_result = generate_generations(
                config,
                checkpoint_path=run_spec.checkpoint_path,
                split=str(args.split),
                study_id_filter=manifest["study_ids"],
                sample_seed=int(manifest["generation_sample_seed"]),
                batch_size=args.generation_batch_size,
                num_workers=args.generation_num_workers,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
            )
            generation_result["label"] = run_spec.label
            generation_result["config"] = str(run_spec.config_path)
            generation_result["sample_manifest_digest"] = manifest["sample_manifest_digest"]
            generation_result["selected_study_count"] = manifest["selected_study_count"]
            generation_path.write_text(json.dumps(generation_result, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[benchmark] generation done {run_spec.label}")
        generation_payload = json.loads(generation_path.read_text(encoding="utf-8"))
        generation_payload = _ensure_organ_abnormal_labels(
            generation_path=generation_path,
            generation_payload=generation_payload,
            config=config,
            split=str(args.split),
            manifest=manifest,
        )

        print(f"[benchmark] evaluating {run_spec.label}")
        evaluation_payload = _load_or_update_evaluation(
            evaluation_path=evaluation_path,
            generation_path=generation_path,
            run_spec=run_spec,
            manifest=manifest,
            metrics=metric_list,
            tokenize_mode=str(args.tokenize),
            green_scope=requested_green_scope,
            include_study_level=bool(args.include_study_level),
            force_eval=bool(args.force_eval),
            green_batch_size=int(args.green_batch_size),
            green_max_new_tokens=int(args.green_max_new_tokens),
            green_prompt_max_length=int(args.green_prompt_max_length),
        )
        if args.radgpt_eval:
            requested_radgpt = {
                "enabled": True,
                "base_url": str(args.radgpt_base_url),
                "fast": bool(args.radgpt_fast),
                "root": str(args.radgpt_root),
                "local_model": str(args.radgpt_local_model or ""),
                "local_model_dtype": str(args.radgpt_local_model_dtype),
                "local_model_max_new_tokens": int(args.radgpt_local_max_new_tokens),
            }
            if _radgpt_matches(
                evaluation_payload,
                generation_path=generation_path,
                requested=requested_radgpt,
            ) and not args.force_radgpt_reference and not args.force_radgpt_generated:
                pass
            else:
                print(f"[benchmark] radgpt {run_spec.label}")
                radgpt_result = evaluate_generation_file_with_radgpt(
                    generation_path,
                    benchmark_cache_dir=run_dir / "radgpt",
                    reference_cache_dir=output_dir / "radgpt_reference",
                    generated_cache_dir=run_dir / "radgpt" / "generated",
                    comparison_output_path=run_dir / "radgpt" / "comparison.json",
                    base_url=str(args.radgpt_base_url),
                    fast=bool(args.radgpt_fast),
                    force_reference=bool(args.force_radgpt_reference),
                    force_generated=bool(args.force_radgpt_generated),
                    radgpt_root=str(args.radgpt_root),
                    local_model_name=str(args.radgpt_local_model) if args.radgpt_local_model else None,
                    local_model_dtype=str(args.radgpt_local_model_dtype),
                    local_model_max_new_tokens=int(args.radgpt_local_max_new_tokens),
                )
                evaluation_payload["radgpt_oncology"] = radgpt_result
                evaluation_payload["requested_radgpt"] = requested_radgpt
                evaluation_path.write_text(json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8")
        else:
            evaluation_payload.pop("radgpt_oncology", None)
        print(f"[benchmark] evaluation done {run_spec.label}")

        row = _build_comparison_row(
            run_spec=run_spec,
            generation_payload=generation_payload,
            evaluation_payload=evaluation_payload,
            manifest=manifest,
        )
        comparison_rows.append(row)
        run_summaries.append(
            {
                "label": run_spec.label,
                "config": str(run_spec.config_path),
                "checkpoint": str(run_spec.checkpoint_path),
                "generation_path": str(generation_path),
                "evaluation_path": str(evaluation_path),
                "row": row,
                "warnings": evaluation_payload.get("warnings", []),
                "unavailable_metrics": evaluation_payload.get("unavailable_metrics", {}),
            }
        )

    summary = {
        "output_dir": str(output_dir),
        "sample_manifest": manifest,
        "runs": run_summaries,
        "comparison_rows": comparison_rows,
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "comparison_summary.csv", comparison_rows)
    _write_markdown(output_dir / "comparison_summary.md", comparison_rows)
    _print_ascii_summary(comparison_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if radgpt_server is not None:
        radgpt_server.close()


def _parse_run_specs(values: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    seen_labels: set[str] = set()
    for index, value in enumerate(values, start=1):
        parts = [part.strip() for part in str(value).split("::")]
        if len(parts) == 2:
            label = ""
            config_part, checkpoint_part = parts
        elif len(parts) == 3:
            label, config_part, checkpoint_part = parts
        else:
            raise ValueError(f"Unsupported --run spec {value!r}. Use 'label::config::checkpoint' or 'config::checkpoint'.")
        config_path = Path(config_part).expanduser().resolve()
        checkpoint_path = Path(checkpoint_part).expanduser().resolve()
        inferred_label = label or checkpoint_path.parent.name or checkpoint_path.stem or f"run_{index}"
        final_label = _dedupe_label(inferred_label, seen_labels)
        specs.append(
            RunSpec(
                label=final_label,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
        )
    return specs


def _dedupe_label(label: str, seen_labels: set[str]) -> str:
    candidate = str(label).strip() or "run"
    if candidate not in seen_labels:
        seen_labels.add(candidate)
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in seen_labels:
        suffix += 1
    deduped = f"{candidate}_{suffix}"
    seen_labels.add(deduped)
    return deduped


def _load_or_create_manifest(
    *,
    manifest_path: Path,
    base_config_path: Path,
    split: str,
    seed: int,
    study_fraction: float,
    study_limit: int | None,
    force: bool,
) -> dict[str, Any]:
    if manifest_path.is_file() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    config = load_decoder_config(str(base_config_path))
    samples, _ = load_decoder_samples(config, split=split, sample_seed=None)
    unique_study_ids = []
    seen_study_ids: set[str] = set()
    for sample in samples:
        study_id = str(sample.study_id)
        if study_id in seen_study_ids:
            continue
        seen_study_ids.add(study_id)
        unique_study_ids.append(study_id)
    rng = random.Random(int(seed))
    shuffled = list(unique_study_ids)
    rng.shuffle(shuffled)
    selected_count = len(shuffled)
    if study_fraction > 0.0:
        selected_count = max(1 if shuffled else 0, int(math.ceil(len(shuffled) * float(study_fraction))))
    if study_limit is not None:
        selected_count = min(selected_count, int(study_limit))
    selected = shuffled[:selected_count]
    digest = hashlib.sha1(
        json.dumps(
            {
                "split": split,
                "seed": int(seed),
                "study_fraction": float(study_fraction),
                "study_limit": None if study_limit is None else int(study_limit),
                "study_ids": selected,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "base_config": str(base_config_path),
        "split": split,
        "seed": int(seed),
        "study_fraction": float(study_fraction),
        "study_limit": None if study_limit is None else int(study_limit),
        "source_study_count": len(unique_study_ids),
        "selected_study_count": len(selected),
        "study_ids": selected,
        "sample_manifest_digest": digest,
        "generation_sample_seed": int(seed) + 1000,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _generation_matches(path: Path, *, run_spec: RunSpec, manifest: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        str(payload.get("checkpoint", "")) == str(run_spec.checkpoint_path)
        and str(payload.get("config", "")) == str(run_spec.config_path)
        and str(payload.get("sample_manifest_digest", "")) == str(manifest["sample_manifest_digest"])
        and isinstance(payload.get("generations"), list)
        and len(payload.get("generations", [])) > 0
    )


def _ensure_organ_abnormal_labels(
    *,
    generation_path: Path,
    generation_payload: dict[str, Any],
    config: Any,
    split: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    rows = generation_payload.get("generations", [])
    if not isinstance(rows, list):
        return generation_payload
    if not any(isinstance(row, dict) and row.get("organ_abnormal_label") is None for row in rows):
        return generation_payload

    samples, _ = load_decoder_samples(config, split=split, sample_seed=None)
    wanted = {str(value) for value in manifest.get("study_ids", [])}
    label_lookup: dict[tuple[str, str], int] = {}
    for sample in samples:
        study_id = str(sample.study_id)
        if wanted and study_id not in wanted:
            continue
        for organ_name, label in sample.organ_label_lookup.items():
            if isinstance(label, int) and label in (0, 1):
                label_lookup[(study_id, str(organ_name))] = int(label)

    changed = False
    matched = 0
    missing = 0
    for row in rows:
        if not isinstance(row, dict) or row.get("organ_abnormal_label") is not None:
            continue
        key = (str(row.get("study_id", "")), str(row.get("organ", "")))
        if key in label_lookup:
            row["organ_abnormal_label"] = int(label_lookup[key])
            matched += 1
            changed = True
        else:
            missing += 1
    if changed:
        generation_payload["organ_abnormal_label_backfill"] = {
            "source": "dataset combined.json labels[organ]",
            "matched_rows": int(matched),
            "missing_rows": int(missing),
        }
        generation_path.write_text(json.dumps(generation_payload, indent=2, sort_keys=True), encoding="utf-8")
    return generation_payload


def _evaluation_matches(path: Path, *, generation_path: Path, manifest: dict[str, Any]) -> bool:
    if not path.is_file():
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


def _radgpt_matches(
    evaluation_payload: dict[str, Any],
    *,
    generation_path: Path,
    requested: dict[str, Any],
) -> bool:
    current = evaluation_payload.get("requested_radgpt")
    result = evaluation_payload.get("radgpt_oncology")
    if not isinstance(current, dict) or not isinstance(result, dict):
        return False
    if current != requested:
        return False
    if str(result.get("generation_path", "")) != str(generation_path):
        return False
    return _radgpt_result_is_usable(result)


def _radgpt_result_is_usable(result: dict[str, Any]) -> bool:
    per_organ = result.get("per_organ")
    if not isinstance(per_organ, dict) or not per_organ:
        return False
    for organ_block in per_organ.values():
        if not isinstance(organ_block, dict):
            continue
        for task_name in ("tumor", "malignancy"):
            task_block = organ_block.get(task_name)
            if isinstance(task_block, dict) and int(task_block.get("count_valid", 0)) > 0:
                return True
    return False


def _ensure_radgpt_server(args: argparse.Namespace, output_dir: Path) -> RadGPTServerHandle | None:
    base_url = str(args.radgpt_base_url)
    if _radgpt_server_ready(base_url):
        return RadGPTServerHandle(process=None, log_path=None, launched=False, base_url=base_url)
    if not args.radgpt_auto_launch:
        raise RuntimeError(
            "RadGPT server is not reachable at "
            f"{base_url}. Start the OpenAI-compatible vLLM server first, or rerun with "
            "`--radgpt-auto-launch` to let the benchmark start it for you."
        )

    launch_cmd, launch_mode = _resolve_vllm_launch_command(args)

    parsed = urlsplit(base_url)
    port = parsed.port or 8000
    host = parsed.hostname or "127.0.0.1"
    if host not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        raise RuntimeError(
            "RadGPT auto-launch only supports a local endpoint. "
            f"Got host {host!r} from {base_url}."
        )

    log_path = Path(args.radgpt_vllm_log).expanduser().resolve() if args.radgpt_vllm_log else output_dir / "radgpt_vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    hf_cache = (
        Path(args.radgpt_vllm_cache_dir).expanduser().resolve()
        if args.radgpt_vllm_cache_dir
        else Path(args.radgpt_root).expanduser().resolve() / "evaluate_reports" / "HFCache"
    )
    hf_cache.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env["TRANSFORMERS_CACHE"] = str(hf_cache)
    env["HF_HOME"] = str(hf_cache)
    if args.radgpt_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.radgpt_cuda_visible_devices)

    if launch_mode == "binary":
        cmd = [
            launch_cmd,
            "serve",
            str(args.radgpt_vllm_model),
            "--dtype",
            str(args.radgpt_vllm_dtype),
            "--tensor-parallel-size",
            str(int(args.radgpt_vllm_tensor_parallel_size)),
            "--gpu_memory_utilization",
            str(float(args.radgpt_vllm_gpu_memory_utilization)),
            "--port",
            str(int(port)),
            "--max_model_len",
            str(int(args.radgpt_vllm_max_model_len)),
            "--enforce-eager",
        ]
    else:
        cmd = [
            launch_cmd,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(args.radgpt_vllm_model),
            "--dtype",
            str(args.radgpt_vllm_dtype),
            "--tensor-parallel-size",
            str(int(args.radgpt_vllm_tensor_parallel_size)),
            "--gpu-memory-utilization",
            str(float(args.radgpt_vllm_gpu_memory_utilization)),
            "--port",
            str(int(port)),
            "--max-model-len",
            str(int(args.radgpt_vllm_max_model_len)),
            "--enforce-eager",
        ]
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(Path(args.radgpt_root).expanduser().resolve()),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    handle = RadGPTServerHandle(process=process, log_path=log_path, launched=True, base_url=base_url)
    atexit.register(handle.close)
    _wait_for_radgpt_server(handle, timeout_seconds=int(args.radgpt_server_timeout))
    return handle


def _resolve_vllm_launch_command(args: argparse.Namespace) -> tuple[str, str]:
    attempted: list[str] = []

    explicit_binary = str(args.radgpt_vllm_command).strip() if args.radgpt_vllm_command else ""
    if explicit_binary:
        candidate = Path(explicit_binary).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve()), "binary"
        attempted.append(f"explicit binary: {candidate}")

    on_path = shutil.which("vllm")
    if on_path:
        return str(Path(on_path).resolve()), "binary"
    attempted.append("PATH lookup: vllm")

    for raw_python in [args.radgpt_vllm_python, sys.executable]:
        if not raw_python:
            continue
        python_path = Path(str(raw_python)).expanduser()
        if not python_path.is_file() or not os.access(python_path, os.X_OK):
            attempted.append(f"python missing: {python_path}")
            continue
        probe = subprocess.run(
            [str(python_path), "-c", "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('vllm') else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            # Preserve the venv executable path. Resolving follows the symlink
            # to the base interpreter and drops the venv site-packages.
            return os.path.abspath(os.path.expanduser(str(python_path))), "python-module"
        attempted.append(f"python without vllm: {python_path}")

    attempted_text = "\n".join(f"- {item}" for item in attempted)
    raise RuntimeError(
        "RadGPT auto-launch requested, but no usable vLLM launcher was found.\n"
        "Tried:\n"
        f"{attempted_text}\n"
        "Provide either `--radgpt-vllm-command /path/to/vllm` or "
        "`--radgpt-vllm-python /path/to/python` where that interpreter has the `vllm` module installed, "
        "or start the RadGPT server manually."
    )


def _wait_for_radgpt_server(handle: RadGPTServerHandle, *, timeout_seconds: int) -> None:
    deadline = time.time() + max(1, int(timeout_seconds))
    while time.time() < deadline:
        if _radgpt_server_ready(handle.base_url):
            return
        if handle.process is not None and handle.process.poll() is not None:
            raise RuntimeError(
                "Auto-launched RadGPT server exited before becoming ready. "
                f"See log: {handle.log_path}\n\n{_tail_text(handle.log_path)}"
            )
        time.sleep(5)
    raise RuntimeError(
        "Timed out waiting for the auto-launched RadGPT server to become ready. "
        f"See log: {handle.log_path}\n\n{_tail_text(handle.log_path)}"
    )


def _radgpt_server_ready(base_url: str) -> bool:
    models_url = str(base_url).rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False
    return isinstance(payload, dict) and isinstance(payload.get("data"), list) and len(payload.get("data", [])) > 0


def _tail_text(path: Path | None, *, line_count: int = 40) -> str:
    if path is None or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _load_or_update_evaluation(
    *,
    evaluation_path: Path,
    generation_path: Path,
    run_spec: RunSpec,
    manifest: dict[str, Any],
    metrics: list[str],
    tokenize_mode: str,
    green_scope: str,
    include_study_level: bool,
    force_eval: bool,
    green_batch_size: int,
    green_max_new_tokens: int,
    green_prompt_max_length: int,
) -> dict[str, Any]:
    requested_metrics = list(dict.fromkeys(metrics))
    base_matches = _evaluation_matches(
        evaluation_path, generation_path=generation_path, manifest=manifest
    )
    if not base_matches or force_eval:
        evaluation_payload = evaluate_file(
            generation_path,
            metrics=requested_metrics,
            tokenize_mode=tokenize_mode,
            green_scope=green_scope,
            limit=None,
            include_study_level=include_study_level,
            green_batch_size=green_batch_size,
            green_max_new_tokens=green_max_new_tokens,
            green_prompt_max_length=green_prompt_max_length,
        )
        evaluation_payload = _decorate_evaluation_payload(
            evaluation_payload,
            run_spec=run_spec,
            generation_path=generation_path,
            manifest=manifest,
            requested_metrics=requested_metrics,
            tokenize_mode=tokenize_mode,
            green_scope=green_scope,
            include_study_level=include_study_level,
            green_batch_size=green_batch_size,
            green_max_new_tokens=green_max_new_tokens,
            green_prompt_max_length=green_prompt_max_length,
        )
        evaluation_path.write_text(
            json.dumps(evaluation_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return evaluation_payload

    existing_payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    missing = _collect_missing_eval_components(
        existing_payload,
        metrics=requested_metrics,
        green_scope=green_scope,
        include_study_level=include_study_level,
    )
    if not _has_missing_eval_components(missing):
        return existing_payload

    partial_green_scope = _missing_green_scope(missing)
    partial_metrics = sorted(set(missing["organ_metrics"] + missing["study_metrics"]))
    partial_include_study_level = include_study_level and (
        bool(missing["study_metrics"]) or missing["study_green"]
    )
    partial_payload = evaluate_file(
        generation_path,
        metrics=partial_metrics,
        tokenize_mode=tokenize_mode,
        green_scope=partial_green_scope,
        limit=None,
        include_study_level=partial_include_study_level,
        green_batch_size=green_batch_size,
        green_max_new_tokens=green_max_new_tokens,
        green_prompt_max_length=green_prompt_max_length,
    )
    merged_payload = _merge_evaluation_payload(
        existing_payload=existing_payload,
        partial_payload=partial_payload,
        missing=missing,
    )
    merged_payload = _decorate_evaluation_payload(
        merged_payload,
        run_spec=run_spec,
        generation_path=generation_path,
        manifest=manifest,
        requested_metrics=requested_metrics,
        tokenize_mode=tokenize_mode,
        green_scope=green_scope,
        include_study_level=include_study_level,
        green_batch_size=green_batch_size,
        green_max_new_tokens=green_max_new_tokens,
        green_prompt_max_length=green_prompt_max_length,
    )
    evaluation_path.write_text(
        json.dumps(merged_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return merged_payload


def _decorate_evaluation_payload(
    payload: dict[str, Any],
    *,
    run_spec: RunSpec,
    generation_path: Path,
    manifest: dict[str, Any],
    requested_metrics: list[str],
    tokenize_mode: str,
    green_scope: str,
    include_study_level: bool,
    green_batch_size: int,
    green_max_new_tokens: int,
    green_prompt_max_length: int,
) -> dict[str, Any]:
    payload["label"] = run_spec.label
    payload["config"] = str(run_spec.config_path)
    payload["checkpoint"] = str(run_spec.checkpoint_path)
    payload["generation_path"] = str(generation_path)
    payload["sample_manifest_digest"] = manifest["sample_manifest_digest"]
    payload["requested_metrics"] = list(requested_metrics)
    payload["requested_tokenize_mode"] = tokenize_mode
    payload["requested_green_scope"] = green_scope
    payload["requested_include_study_level"] = bool(include_study_level)
    payload["requested_green_batch_size"] = int(green_batch_size)
    payload["requested_green_max_new_tokens"] = int(green_max_new_tokens)
    payload["requested_green_prompt_max_length"] = int(green_prompt_max_length)
    return payload


def _collect_missing_eval_components(
    payload: dict[str, Any],
    *,
    metrics: list[str],
    green_scope: str,
    include_study_level: bool,
) -> dict[str, Any]:
    organ_overall = payload.get("organ_level", {}).get("overall", {})
    study_overall = payload.get("study_level_support", {}).get("overall", {})
    missing = {
        "organ_metrics": [metric for metric in metrics if metric not in organ_overall],
        "study_metrics": [],
        "organ_green": False,
        "study_green": False,
        "keyword": not bool(payload.get("keyword_diagnostics")),
    }
    if include_study_level:
        missing["study_metrics"] = [
            metric for metric in metrics if metric not in study_overall
        ]
    if green_scope in {"organ", "both"} and "GREEN" not in organ_overall:
        missing["organ_green"] = True
    if include_study_level and green_scope in {"study", "both"} and "GREEN" not in study_overall:
        missing["study_green"] = True
    return missing


def _has_missing_eval_components(missing: dict[str, Any]) -> bool:
    return any(
        (
            missing["organ_metrics"],
            missing["study_metrics"],
            missing["organ_green"],
            missing["study_green"],
            missing["keyword"],
        )
    )


def _missing_green_scope(missing: dict[str, Any]) -> str:
    if missing["organ_green"] and missing["study_green"]:
        return "both"
    if missing["organ_green"]:
        return "organ"
    if missing["study_green"]:
        return "study"
    return "none"


def _merge_evaluation_payload(
    *,
    existing_payload: dict[str, Any],
    partial_payload: dict[str, Any],
    missing: dict[str, Any],
) -> dict[str, Any]:
    merged = json.loads(json.dumps(existing_payload))
    merged["input"] = partial_payload.get("input", merged.get("input"))
    merged["input_summary"] = partial_payload.get(
        "input_summary", merged.get("input_summary", {})
    )
    merged["metric_backend"] = partial_payload.get(
        "metric_backend", merged.get("metric_backend")
    )
    merged["coco_tokenize"] = partial_payload.get(
        "coco_tokenize", merged.get("coco_tokenize")
    )
    merged["green_scope"] = partial_payload.get("green_scope", merged.get("green_scope"))
    merged["warnings"] = _merge_unique_strings(
        merged.get("warnings", []), partial_payload.get("warnings", [])
    )
    merged["unavailable_metrics"] = _merge_unavailable_metrics(
        existing_payload.get("unavailable_metrics", {}),
        partial_payload.get("unavailable_metrics", {}),
        missing=missing,
    )
    merged["organ_level"] = _merge_metric_block(
        merged.get("organ_level", {}),
        partial_payload.get("organ_level", {}),
    )
    if partial_payload.get("study_level_support"):
        merged["study_level_support"] = _merge_metric_block(
            merged.get("study_level_support", {}),
            partial_payload.get("study_level_support", {}),
        )
    if missing["keyword"] and partial_payload.get("keyword_diagnostics"):
        merged["keyword_diagnostics"] = partial_payload["keyword_diagnostics"]
    merged["per_organ_summary"] = _build_per_organ_summary(merged)
    merged["score_summary"] = _build_score_summary(merged["per_organ_summary"])
    return merged


def _merge_unique_strings(existing: list[Any], new_items: list[Any]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in list(existing) + list(new_items):
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _merge_unavailable_metrics(
    existing: dict[str, Any],
    new_items: dict[str, Any],
    *,
    missing: dict[str, Any],
) -> dict[str, Any]:
    merged = {str(key): str(value) for key, value in dict(existing).items()}
    attempted = set(missing["organ_metrics"]) | set(missing["study_metrics"])
    if missing["organ_green"] or missing["study_green"]:
        attempted.add("GREEN")
    for key in attempted:
        merged.pop(str(key), None)
    for key, value in dict(new_items).items():
        merged[str(key)] = str(value)
    return merged


def _merge_metric_block(existing: dict[str, Any], new_items: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(existing))
    for key, value in new_items.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_metric_block(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_comparison_row(
    *,
    run_spec: RunSpec,
    generation_payload: dict[str, Any],
    evaluation_payload: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": run_spec.label,
        "config": str(run_spec.config_path),
        "checkpoint": str(run_spec.checkpoint_path),
        "split": str(generation_payload.get("split", evaluation_payload.get("split", manifest.get("split", "test")))),
        "selected_study_count": int(manifest["selected_study_count"]),
        "generation_row_count": len(generation_payload.get("generations", [])),
        "warning_count": len(evaluation_payload.get("warnings", [])),
        "unavailable_metric_count": len(evaluation_payload.get("unavailable_metrics", {})),
    }
    _flatten_metric_block(row, evaluation_payload.get("organ_level", {}).get("overall", {}), prefix="organ_")
    _flatten_metric_block(row, evaluation_payload.get("study_level_support", {}).get("overall", {}), prefix="study_")
    sampled_green = evaluation_payload.get("sampled_green", {})
    if isinstance(sampled_green, dict):
        _flatten_metric_block(row, sampled_green.get("overall", {}), prefix="sampled_green_")
        sampled_groups = sampled_green.get("by_organ_abnormal_label", {})
        if isinstance(sampled_groups, dict):
            _flatten_metric_block(row, sampled_groups.get("positive", {}), prefix="sampled_green_abnormal_positive_")
            _flatten_metric_block(row, sampled_groups.get("negative", {}), prefix="sampled_green_abnormal_negative_")
    abnormal_groups = evaluation_payload.get("organ_level", {}).get("by_organ_abnormal_label", {})
    _flatten_metric_block(row, abnormal_groups.get("positive", {}), prefix="organ_abnormal_positive_")
    _flatten_metric_block(row, abnormal_groups.get("negative", {}), prefix="organ_abnormal_negative_")
    lesion_groups = evaluation_payload.get("organ_level", {}).get("by_lesion_label", {})
    _flatten_metric_block(row, lesion_groups.get("positive", {}), prefix="lesion_positive_")
    _flatten_metric_block(row, lesion_groups.get("negative", {}), prefix="lesion_negative_")
    _flatten_metric_block(row, evaluation_payload.get("keyword_diagnostics", {}).get("overall", {}), prefix="keyword_")
    _flatten_dataset_abnormal_keyword_block(row, generation_payload.get("generations", []))
    _flatten_negation_aware_keyword_block(row, generation_payload.get("generations", []))
    _flatten_generation_quality_block(row, generation_payload.get("generations", []))
    _flatten_radgpt_block(row, evaluation_payload.get("radgpt_oncology", {}), prefix="radgpt_")
    _flatten_radgpt_uncertain_as_negative_block(row, evaluation_payload.get("radgpt_oncology", {}), prefix="radgpt_uncertain_as_negative_")
    if any("Skipped METEOR" in str(warning) for warning in evaluation_payload.get("warnings", [])):
        for key in list(row):
            if key.endswith("_METEOR") or key == "organ_METEOR" or key == "study_METEOR":
                row.pop(key, None)
    return row


def _flatten_generation_quality_block(row: dict[str, Any], rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        return
    total = 0
    counts = {
        "very_short_count": 0,
        "numeric_or_punct_only_count": 0,
        "lowercase_or_digit_start_count": 0,
        "joined_normal_count": 0,
        "weird_preface_count": 0,
        "unclosed_sentence_count": 0,
    }
    for item in rows:
        if not isinstance(item, dict):
            continue
        text = str(item.get("generated", "")).strip()
        total += 1
        if len(text) <= 5:
            counts["very_short_count"] += 1
        if re.fullmatch(r"[\W\d_]+", text or ""):
            counts["numeric_or_punct_only_count"] += 1
        if text and (text[0].islower() or text[0].isdigit()):
            counts["lowercase_or_digit_start_count"] += 1
        if re.search(r"\b(is|are|has|have|was|were)normal\b|\banormal\b|\bhavenormal\b|\bisnormal\b", text, re.IGNORECASE):
            counts["joined_normal_count"] += 1
        if re.match(r"^(certainly|sure|assistant|user)\b", text, re.IGNORECASE):
            counts["weird_preface_count"] += 1
        if text and text[-1] not in ".;:!?)]":
            counts["unclosed_sentence_count"] += 1
    if total <= 0:
        return
    row["generation_quality_count"] = float(total)
    for key, value in counts.items():
        row[f"generation_quality_{key}"] = float(value)
        row[f"generation_quality_{key.removesuffix('_count')}_rate"] = float(value) / float(total)


def _flatten_dataset_abnormal_keyword_block(row: dict[str, Any], rows: Any) -> None:
    if not isinstance(rows, list):
        return
    positive_rows = [
        item
        for item in rows
        if isinstance(item, dict) and _safe_binary_label(item.get("organ_abnormal_label")) == 1
    ]
    negative_rows = [
        item
        for item in rows
        if isinstance(item, dict) and _safe_binary_label(item.get("organ_abnormal_label")) == 0
    ]
    if positive_rows:
        _flatten_metric_block(row, summarize_keyword_metrics(positive_rows).get("overall", {}), prefix="keyword_abnormal_positive_")
    if negative_rows:
        _flatten_metric_block(row, summarize_keyword_metrics(negative_rows).get("overall", {}), prefix="keyword_abnormal_negative_")


def _flatten_negation_aware_keyword_block(row: dict[str, Any], rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        return
    all_rows = [item for item in rows if isinstance(item, dict)]
    positive_rows = [item for item in all_rows if _safe_binary_label(item.get("organ_abnormal_label")) == 1]
    negative_rows = [item for item in all_rows if _safe_binary_label(item.get("organ_abnormal_label")) == 0]
    for prefix, subset in (
        ("keyword_negation_aware_", all_rows),
        ("keyword_negation_aware_abnormal_positive_", positive_rows),
        ("keyword_negation_aware_abnormal_negative_", negative_rows),
    ):
        if not subset:
            continue
        generated_count = sum(1 for item in subset if _has_asserted_pathology(str(item.get("generated", ""))))
        target_count = sum(1 for item in subset if _has_asserted_pathology(str(item.get("target", ""))))
        row[f"{prefix}count"] = float(len(subset))
        row[f"{prefix}generated_asserted_pathology_rate"] = float(generated_count) / float(len(subset))
        row[f"{prefix}target_asserted_pathology_rate"] = float(target_count) / float(len(subset))


def _has_asserted_pathology(text: str) -> bool:
    pathology = {
        "lesion",
        "lesions",
        "cyst",
        "cysts",
        "mass",
        "masses",
        "nodule",
        "nodules",
        "metastasis",
        "metastases",
        "tumor",
        "tumour",
    }
    negators = {"no", "without", "absent", "negative"}
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for index, token in enumerate(tokens):
        if token not in pathology:
            continue
        window = tokens[max(0, index - 5):index]
        if any(value in negators for value in window):
            continue
        return True
    return False


def _safe_binary_label(value: Any) -> int | None:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if as_float > 0.5:
        return 1
    if as_float <= 0.5:
        return 0
    return None


def _flatten_metric_block(row: dict[str, Any], block: dict[str, Any], *, prefix: str) -> None:
    for key, value in sorted(block.items()):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[f"{prefix}{key}"] = float(value)


def _flatten_radgpt_block(row: dict[str, Any], block: dict[str, Any], *, prefix: str) -> None:
    if not isinstance(block, dict):
        return
    _flatten_metric_block(row, block.get("overall", {}), prefix=prefix)
    for organ_name, organ_metrics in sorted(block.get("per_organ", {}).items()):
        organ_slug = organ_name.lower().replace(" ", "_")
        if not isinstance(organ_metrics, dict):
            continue
        for task_name in ("tumor", "malignancy"):
            task_metrics = organ_metrics.get(task_name, {})
            if isinstance(task_metrics, dict):
                _flatten_metric_block(
                    row,
                    task_metrics,
                    prefix=f"{prefix}{organ_slug}_{task_name}_",
                )
                count_total = _safe_float(task_metrics.get("count_total"))
                count_valid = _safe_float(task_metrics.get("count_valid"))
                generated_uncertain = _safe_float(task_metrics.get("generated_uncertain_count"))
                reference_uncertain = _safe_float(task_metrics.get("reference_uncertain_count"))
                if count_total and count_total > 0:
                    row[f"{prefix}{organ_slug}_{task_name}_valid_rate"] = float(count_valid or 0.0) / count_total
                    row[f"{prefix}{organ_slug}_{task_name}_generated_uncertain_rate"] = float(generated_uncertain or 0.0) / count_total
                    row[f"{prefix}{organ_slug}_{task_name}_reference_uncertain_rate"] = float(reference_uncertain or 0.0) / count_total


def _flatten_radgpt_uncertain_as_negative_block(row: dict[str, Any], block: dict[str, Any], *, prefix: str) -> None:
    if not isinstance(block, dict):
        return
    reference_path = Path(str(block.get("reference_label_path", "")))
    generated_path = Path(str(block.get("generated_label_path", "")))
    if not reference_path.is_file() or not generated_path.is_file():
        return
    try:
        reference_rows = _read_radgpt_label_rows(reference_path)
        generated_rows = _read_radgpt_label_rows(generated_path)
    except Exception:
        return
    per_task_f1: dict[str, list[float]] = {"tumor": [], "malignancy": []}
    for organ in ("liver", "kidney", "pancreas"):
        for task_name, column in (("tumor", "tumor_label"), ("malignancy", "malignancy_label")):
            refs: list[int] = []
            preds: list[int] = []
            generated_uncertain = 0
            for sample_id, ref_row in reference_rows.items():
                if str(ref_row.get("radgpt_organ", "")).strip().lower() != organ:
                    continue
                ref_label = _parse_binary_label(ref_row.get(column))
                if ref_label is None:
                    continue
                gen_label = _parse_binary_label(generated_rows.get(sample_id, {}).get(column))
                if gen_label is None:
                    generated_uncertain += 1
                    gen_label = 0
                refs.append(ref_label)
                preds.append(gen_label)
            metrics = _binary_metrics_from_lists(refs, preds)
            organ_slug = "kidneys" if organ == "kidney" else organ
            for metric_name, value in metrics.items():
                if _is_number(value):
                    row[f"{prefix}{organ_slug}_{task_name}_{metric_name}"] = float(value)
            row[f"{prefix}{organ_slug}_{task_name}_generated_uncertain_count"] = float(generated_uncertain)
            if _is_number(metrics.get("f1")):
                per_task_f1[task_name].append(float(metrics["f1"]))
    for task_name, values in per_task_f1.items():
        if values:
            row[f"{prefix}{task_name}_macro_f1"] = sum(values) / len(values)


def _read_radgpt_label_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row.get("sample_id", "")): row for row in csv.DictReader(handle)}


def _parse_binary_label(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if as_float > 0.5:
        return 1
    if as_float <= 0.5:
        return 0
    return None


def _binary_metrics_from_lists(refs: list[int], preds: list[int]) -> dict[str, float]:
    if not refs or len(refs) != len(preds):
        return {}
    tp = sum(1 for ref, pred in zip(refs, preds) if ref == 1 and pred == 1)
    tn = sum(1 for ref, pred in zip(refs, preds) if ref == 0 and pred == 0)
    fp = sum(1 for ref, pred in zip(refs, preds) if ref == 0 and pred == 1)
    fn = sum(1 for ref, pred in zip(refs, preds) if ref == 1 and pred == 0)
    precision = tp / (tp + fp) if (tp + fp) else (0.0 if (tp + fn) else math.nan)
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    elif precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "count": float(len(refs)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else math.nan,
        "f1": float(f1),
        "generated_positive_rate": float(sum(preds) / len(preds)),
        "reference_positive_rate": float(sum(refs) / len(refs)),
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False



def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    overall_columns = [
        "label",
        "generation_row_count",
        "organ_Bleu_1",
        "organ_Bleu_2",
        "organ_Bleu_3",
        "organ_Bleu_4",
        "organ_METEOR",
        "organ_ROUGE_L",
        "organ_CIDEr",
        "sampled_green_GREEN",
        "organ_GREEN",
        "organ_abnormal_positive_GREEN",
        "organ_abnormal_negative_GREEN",
        "study_Bleu_1",
        "study_Bleu_2",
        "study_Bleu_3",
        "study_Bleu_4",
        "study_METEOR",
        "study_ROUGE_L",
        "study_CIDEr",
        "study_GREEN",
        "keyword_csv_positive_pathology_recall",
        "keyword_csv_negative_pathology_rate",
        "keyword_csv_positive_normal_rate",
        "keyword_generated_normal_word_rate",
        "radgpt_tumor_macro_f1",
        "radgpt_uncertain_as_negative_tumor_macro_f1",
        "radgpt_malignancy_macro_f1",
        "radgpt_uncertain_as_negative_malignancy_macro_f1",
        "radgpt_liver_tumor_f1",
        "radgpt_kidneys_tumor_f1",
        "radgpt_pancreas_tumor_f1",
        "radgpt_liver_malignancy_f1",
        "radgpt_kidneys_malignancy_f1",
        "radgpt_pancreas_malignancy_f1",
    ]
    positive_columns = _stratified_metric_columns("organ_abnormal_positive")
    negative_columns = _stratified_metric_columns("organ_abnormal_negative")
    behavior_columns = [
        "label",
        "generation_row_count",
        "keyword_abnormal_positive_generated_pathology_word_rate",
        "keyword_abnormal_positive_generated_normal_word_rate",
        "keyword_abnormal_negative_generated_pathology_word_rate",
        "keyword_abnormal_negative_generated_normal_word_rate",
        "keyword_negation_aware_abnormal_positive_generated_asserted_pathology_rate",
        "keyword_negation_aware_abnormal_negative_generated_asserted_pathology_rate",
        "keyword_generated_pathology_word_rate",
        "keyword_generated_normal_word_rate",
    ]
    generation_quality_columns = [
        "label",
        "generation_row_count",
        "generation_quality_very_short_rate",
        "generation_quality_numeric_or_punct_only_rate",
        "generation_quality_lowercase_or_digit_start_rate",
        "generation_quality_joined_normal_rate",
        "generation_quality_weird_preface_rate",
        "generation_quality_unclosed_sentence_rate",
    ]
    radgpt_validity_columns = [
        "label",
        "radgpt_tumor_macro_f1",
        "radgpt_uncertain_as_negative_tumor_macro_f1",
        "radgpt_liver_tumor_generated_uncertain_rate",
        "radgpt_kidneys_tumor_generated_uncertain_rate",
        "radgpt_pancreas_tumor_generated_uncertain_rate",
        "radgpt_liver_tumor_valid_rate",
        "radgpt_kidneys_tumor_valid_rate",
        "radgpt_pancreas_tumor_valid_rate",
    ]
    display_names = _comparison_display_names()
    metric_directions = _comparison_metric_directions()
    lines = [
        "# Decoder Checkpoint Comparison",
        "",
        "This report separates aggregate quality from dataset-label-stratified quality. `Abnormal` means `combined.json` `labels[organ] = 1`; `Normal` means `labels[organ] = 0`.",
    ]
    used_columns: list[str] = []
    used_columns.extend(_append_markdown_table(
        lines,
        title="Overall",
        rows=rows,
        preferred_columns=overall_columns,
        display_names=display_names,
        metric_directions=metric_directions,
    ))
    used_columns.extend(_append_markdown_table(
        lines,
        title="Abnormal Organ Rows",
        rows=rows,
        preferred_columns=positive_columns,
        display_names=display_names,
        metric_directions=metric_directions,
    ))
    used_columns.extend(_append_markdown_table(
        lines,
        title="Normal Organ Rows",
        rows=rows,
        preferred_columns=negative_columns,
        display_names=display_names,
        metric_directions=metric_directions,
    ))
    used_columns.extend(_append_markdown_table(
        lines,
        title="Generation Bias Checks",
        rows=rows,
        preferred_columns=behavior_columns,
        display_names=display_names,
        metric_directions=metric_directions,
    ))
    used_columns.extend(_append_markdown_table(
        lines,
        title="Generation Quality Audit",
        rows=rows,
        preferred_columns=generation_quality_columns,
        display_names=display_names,
        metric_directions=metric_directions,
    ))
    used_columns.extend(_append_markdown_table(
        lines,
        title="RadGPT Validity Audit",
        rows=rows,
        preferred_columns=radgpt_validity_columns,
        display_names=display_names,
        metric_directions=metric_directions,
    ))
    if not used_columns:
        used_columns = sorted({key for row in rows for key in row.keys()})
        _append_markdown_table(
            lines,
            title="All Available Fields",
            rows=rows,
            preferred_columns=used_columns,
            display_names=display_names,
            metric_directions=metric_directions,
        )
    legend_lines = _comparison_legend(list(dict.fromkeys(used_columns)), display_names, metric_directions)
    if legend_lines:
        lines.extend(["", "## Metric Legend", "", *legend_lines])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stratified_metric_columns(prefix: str) -> list[str]:
    return [
        "label",
        f"{prefix}_count",
        f"{prefix}_Bleu_1",
        f"{prefix}_Bleu_2",
        f"{prefix}_Bleu_3",
        f"{prefix}_Bleu_4",
        f"{prefix}_METEOR",
        f"{prefix}_ROUGE_L",
        f"{prefix}_CIDEr",
        f"sampled_green_{prefix.removeprefix('organ_')}_GREEN",
        f"{prefix}_GREEN",
    ]


def _append_markdown_table(
    lines: list[str],
    *,
    title: str,
    rows: list[dict[str, Any]],
    preferred_columns: list[str],
    display_names: dict[str, str],
    metric_directions: dict[str, str],
) -> list[str]:
    columns = [column for column in preferred_columns if any(column in row for row in rows)]
    if not columns:
        return []
    best_by_column = _best_values_by_column(rows, columns, metric_directions)
    header = [display_names.get(column, column) for column in columns]
    lines.extend([
        "",
        f"## {title}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ])
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if not math.isfinite(value):
                    rendered = "n/a"
                else:
                    rendered = f"{value:.6g}"
                if math.isfinite(value) and _is_best_value(value, best_by_column.get(column)):
                    rendered = f"**{rendered}**"
                values.append(rendered)
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return columns


def _comparison_display_names() -> dict[str, str]:
    names = {
        "label": "Run",
        "generation_row_count": "Rows",
        "organ_Bleu_1": "Organ BLEU-1 ↑",
        "organ_Bleu_2": "Organ BLEU-2 ↑",
        "organ_Bleu_3": "Organ BLEU-3 ↑",
        "organ_Bleu_4": "Organ BLEU-4 ↑",
        "organ_METEOR": "Organ METEOR ↑",
        "organ_ROUGE_L": "Organ ROUGE-L ↑",
        "organ_CIDEr": "Organ CIDEr ↑",
        "sampled_green_GREEN": "Sampled GREEN ↑",
        "organ_GREEN": "Organ GREEN ↑",
        "organ_abnormal_positive_GREEN": "Abnormal GREEN ↑",
        "organ_abnormal_negative_GREEN": "Normal GREEN ↑",
        "study_Bleu_1": "Study BLEU-1 ↑",
        "study_Bleu_2": "Study BLEU-2 ↑",
        "study_Bleu_3": "Study BLEU-3 ↑",
        "study_Bleu_4": "Study BLEU-4 ↑",
        "study_METEOR": "Study METEOR ↑",
        "study_ROUGE_L": "Study ROUGE-L ↑",
        "study_CIDEr": "Study CIDEr ↑",
        "study_GREEN": "Study GREEN ↑",
        "keyword_csv_positive_pathology_recall": "Positive Pathology Recall ↑",
        "keyword_csv_negative_pathology_rate": "Negative Pathology Rate ↓",
        "keyword_csv_positive_normal_rate": "Positive Normal Rate ↓",
        "keyword_generated_normal_word_rate": "Generated Normal Rate ↔",
        "radgpt_tumor_macro_f1": "RadGPT Tumor Macro-F1 ↑",
        "radgpt_uncertain_as_negative_tumor_macro_f1": "RadGPT Tumor F1, U→Neg ↑",
        "radgpt_malignancy_macro_f1": "RadGPT Malignancy Macro-F1 ↑",
        "radgpt_uncertain_as_negative_malignancy_macro_f1": "RadGPT Malignancy F1, U→Neg ↑",
        "radgpt_liver_tumor_f1": "RadGPT Liver Tumor F1 ↑",
        "radgpt_kidneys_tumor_f1": "RadGPT Kidneys Tumor F1 ↑",
        "radgpt_pancreas_tumor_f1": "RadGPT Pancreas Tumor F1 ↑",
        "radgpt_liver_malignancy_f1": "RadGPT Liver Malignancy F1 ↑",
        "radgpt_kidneys_malignancy_f1": "RadGPT Kidneys Malignancy F1 ↑",
        "radgpt_pancreas_malignancy_f1": "RadGPT Pancreas Malignancy F1 ↑",
    }
    for prefix, label in (
        ("organ_abnormal_positive", "Abnormal"),
        ("organ_abnormal_negative", "Normal"),
    ):
        names.update(
            {
                f"{prefix}_count": f"{label} Rows",
                f"{prefix}_Bleu_1": f"{label} BLEU-1 ↑",
                f"{prefix}_Bleu_2": f"{label} BLEU-2 ↑",
                f"{prefix}_Bleu_3": f"{label} BLEU-3 ↑",
                f"{prefix}_Bleu_4": f"{label} BLEU-4 ↑",
                f"{prefix}_METEOR": f"{label} METEOR ↑",
                f"{prefix}_ROUGE_L": f"{label} ROUGE-L ↑",
                f"{prefix}_CIDEr": f"{label} CIDEr ↑",
                f"sampled_green_{prefix.removeprefix('organ_')}_GREEN": f"{label} Sampled GREEN ↑",
                f"{prefix}_GREEN": f"{label} GREEN ↑",
            }
        )
    names.update(
        {
            "keyword_abnormal_positive_generated_pathology_word_rate": "Abnormal Pathology Words ↑",
            "keyword_abnormal_positive_generated_normal_word_rate": "Abnormal Normal Words ↓",
            "keyword_abnormal_negative_generated_pathology_word_rate": "Normal Pathology Words ↓",
            "keyword_abnormal_negative_generated_normal_word_rate": "Normal Normal Words ↑",
            "keyword_negation_aware_abnormal_positive_generated_asserted_pathology_rate": "Abnormal Asserted Pathology ↑",
            "keyword_negation_aware_abnormal_negative_generated_asserted_pathology_rate": "Normal Asserted Pathology ↓",
            "keyword_generated_pathology_word_rate": "Overall Pathology Words ↔",
            "generation_quality_very_short_rate": "Very Short Rate ↓",
            "generation_quality_numeric_or_punct_only_rate": "Numeric/Punct Only ↓",
            "generation_quality_lowercase_or_digit_start_rate": "Frag Start Rate ↓",
            "generation_quality_joined_normal_rate": "Joined Normal Rate ↓",
            "generation_quality_weird_preface_rate": "Weird Preface Rate ↓",
            "generation_quality_unclosed_sentence_rate": "Unclosed Sentence Rate ↓",
            "radgpt_liver_tumor_generated_uncertain_rate": "Liver Tumor U Rate ↓",
            "radgpt_kidneys_tumor_generated_uncertain_rate": "Kidneys Tumor U Rate ↓",
            "radgpt_pancreas_tumor_generated_uncertain_rate": "Pancreas Tumor U Rate ↓",
            "radgpt_liver_tumor_valid_rate": "Liver Tumor Valid Rate ↑",
            "radgpt_kidneys_tumor_valid_rate": "Kidneys Tumor Valid Rate ↑",
            "radgpt_pancreas_tumor_valid_rate": "Pancreas Tumor Valid Rate ↑",
        }
    )
    return names


def _comparison_metric_directions() -> dict[str, str]:
    higher = {
        "organ_Bleu_1",
        "organ_Bleu_2",
        "organ_Bleu_3",
        "organ_Bleu_4",
        "organ_METEOR",
        "organ_ROUGE_L",
        "organ_CIDEr",
        "sampled_green_GREEN",
        "organ_GREEN",
        "organ_abnormal_positive_GREEN",
        "organ_abnormal_negative_GREEN",
        "study_Bleu_1",
        "study_Bleu_2",
        "study_Bleu_3",
        "study_Bleu_4",
        "study_METEOR",
        "study_ROUGE_L",
        "study_CIDEr",
        "study_GREEN",
        "keyword_csv_positive_pathology_recall",
        "radgpt_tumor_macro_f1",
        "radgpt_uncertain_as_negative_tumor_macro_f1",
        "radgpt_malignancy_macro_f1",
        "radgpt_uncertain_as_negative_malignancy_macro_f1",
        "radgpt_liver_tumor_f1",
        "radgpt_kidneys_tumor_f1",
        "radgpt_pancreas_tumor_f1",
        "radgpt_liver_malignancy_f1",
        "radgpt_kidneys_malignancy_f1",
        "radgpt_pancreas_malignancy_f1",
    }
    lower = {
        "keyword_csv_negative_pathology_rate",
        "keyword_csv_positive_normal_rate",
    }
    neutral = {
        "generation_row_count",
        "keyword_generated_normal_word_rate",
    }
    out = {column: "higher" for column in higher}
    out.update({column: "lower" for column in lower})
    out.update({column: "neutral" for column in neutral})
    for prefix in ("organ_abnormal_positive", "organ_abnormal_negative"):
        for metric in ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "GREEN"):
            out[f"{prefix}_{metric}"] = "higher"
        out[f"sampled_green_{prefix.removeprefix('organ_')}_GREEN"] = "higher"
        out[f"{prefix}_count"] = "neutral"
    out.update(
        {
            "keyword_abnormal_positive_generated_pathology_word_rate": "higher",
            "keyword_abnormal_positive_generated_normal_word_rate": "lower",
            "keyword_abnormal_negative_generated_pathology_word_rate": "lower",
            "keyword_abnormal_negative_generated_normal_word_rate": "higher",
            "keyword_negation_aware_abnormal_positive_generated_asserted_pathology_rate": "higher",
            "keyword_negation_aware_abnormal_negative_generated_asserted_pathology_rate": "lower",
            "keyword_generated_pathology_word_rate": "neutral",
            "generation_quality_very_short_rate": "lower",
            "generation_quality_numeric_or_punct_only_rate": "lower",
            "generation_quality_lowercase_or_digit_start_rate": "lower",
            "generation_quality_joined_normal_rate": "lower",
            "generation_quality_weird_preface_rate": "lower",
            "generation_quality_unclosed_sentence_rate": "lower",
            "radgpt_liver_tumor_generated_uncertain_rate": "lower",
            "radgpt_kidneys_tumor_generated_uncertain_rate": "lower",
            "radgpt_pancreas_tumor_generated_uncertain_rate": "lower",
            "radgpt_liver_tumor_valid_rate": "higher",
            "radgpt_kidneys_tumor_valid_rate": "higher",
            "radgpt_pancreas_tumor_valid_rate": "higher",
        }
    )
    return out


def _best_values_by_column(
    rows: list[dict[str, Any]],
    columns: list[str],
    directions: dict[str, str],
) -> dict[str, float]:
    best: dict[str, float] = {}
    for column in columns:
        direction = directions.get(column)
        if direction not in {"higher", "lower"}:
            continue
        values = [float(row[column]) for row in rows if isinstance(row.get(column), (int, float)) and not isinstance(row.get(column), bool)]
        if not values:
            continue
        best[column] = max(values) if direction == "higher" else min(values)
    return best


def _is_best_value(value: float, best_value: float | None) -> bool:
    if best_value is None:
        return False
    return abs(float(value) - float(best_value)) <= 1.0e-12


def _comparison_legend(
    columns: list[str],
    display_names: dict[str, str],
    directions: dict[str, str],
) -> list[str]:
    descriptions = {
        "organ_Bleu_1": "Organ-level unigram overlap with the reference report; higher is better.",
        "organ_Bleu_2": "Organ-level 1-2 gram overlap with the reference report; higher is better.",
        "organ_Bleu_3": "Organ-level 1-3 gram overlap with the reference report; higher is better.",
        "organ_Bleu_4": "Organ-level BLEU-4 text overlap with the reference report; higher is better.",
        "organ_METEOR": "Organ-level METEOR text similarity, including unigram alignment and stemming/synonym-style matching when supported; higher is better.",
        "organ_ROUGE_L": "Organ-level longest-common-subsequence overlap with the reference report; higher is better.",
        "organ_CIDEr": "Organ-level consensus-style caption similarity; higher is better.",
        "sampled_green_GREEN": "Organ-level GREEN on a shared fixed 10% study-level test sample, including all organs from selected studies; higher is better.",
        "organ_GREEN": "Organ-level GREEN clinical quality score when enabled; higher is better.",
        "organ_abnormal_positive_GREEN": "Organ-level GREEN on dataset-label-positive abnormal organ rows (`combined.json` labels[organ] = 1); higher is better.",
        "organ_abnormal_negative_GREEN": "Organ-level GREEN on dataset-label-negative/normal organ rows (`combined.json` labels[organ] = 0); higher is better.",
        "study_Bleu_1": "Study-level unigram overlap after reconstructing full reports; higher is better.",
        "study_Bleu_2": "Study-level 1-2 gram overlap after reconstructing full reports; higher is better.",
        "study_Bleu_3": "Study-level 1-3 gram overlap after reconstructing full reports; higher is better.",
        "study_Bleu_4": "Study-level BLEU-4 after reconstructing full reports; higher is better.",
        "study_METEOR": "Study-level METEOR after reconstructing full reports; higher is better.",
        "study_ROUGE_L": "Study-level ROUGE-L after reconstructing full reports; higher is better.",
        "study_CIDEr": "Study-level CIDEr after reconstructing full reports; higher is better.",
        "study_GREEN": "Study-level GREEN clinical quality score when enabled; higher is better.",
        "keyword_csv_positive_pathology_recall": "Among CSV-positive organ examples, fraction of generated findings containing pathology keywords; higher suggests better sensitivity.",
        "keyword_csv_negative_pathology_rate": "Among CSV-negative organ examples, fraction of generated findings containing pathology keywords; lower suggests fewer false-positive abnormal mentions.",
        "keyword_csv_positive_normal_rate": "Among CSV-positive organ examples, fraction of generated findings containing normal wording; lower suggests fewer false-normal reports.",
        "keyword_generated_normal_word_rate": "Overall generated normal-word rate. This is descriptive, not intrinsically good or bad.",
        "radgpt_tumor_macro_f1": "RadGPT tumor label macro-F1 on oncology organs when enabled; higher is better.",
        "radgpt_malignancy_macro_f1": "RadGPT malignancy label macro-F1 on oncology organs when enabled; higher is better.",
    }
    lines = [
        "- `↑` means higher is better; `↓` means lower is better; `↔` means descriptive/context-dependent.",
        "- Bold values mark the best run for that metric among the rows in this table.",
        "- Text metrics are computed between each generated organ finding and its reference organ finding, then averaged.",
        "- Keyword diagnostics use a simple fixed keyword list, not an LLM judge: pathology words are `lesion`, `lesions`, `cyst`, `cysts`, `mass`, `masses`, `nodule`, `nodules`, `metastasis`, `metastases`, `tumor`, `tumour`; normal words are `unremarkable`, `normal`, `within normal limits`, `no abnormality`, `no focal abnormality`.",
        "- Sampled GREEN uses one fixed study-level test subset, then reports overall/abnormal/normal groups from the same sampled organ rows.",
        "- Abnormal/normal GREEN stratification uses the main dataset `combined.json` binary `labels[organ]`, not lesion CSV labels.",
        "- CSV-positive/negative groups come from the lesion label attached to each generated organ row. Rows without a lesion label are excluded from CSV-stratified keyword rates.",
    ]
    for column in columns:
        if column in {"label", "generation_row_count"}:
            continue
        description = descriptions.get(column)
        if description is None and column.startswith("radgpt_"):
            description = "RadGPT task F1 score when enabled; higher is better."
        if description is None and column.startswith("organ_abnormal_positive_"):
            description = "Metric computed only on dataset-label-positive abnormal organ rows (`combined.json` labels[organ] = 1)."
        if description is None and column.startswith("organ_abnormal_negative_"):
            description = "Metric computed only on dataset-label-negative/normal organ rows (`combined.json` labels[organ] = 0)."
        if description is None and column.startswith("sampled_green_abnormal_positive_"):
            description = "GREEN computed on dataset-label-positive abnormal organ rows inside the shared fixed study-level sample."
        if description is None and column.startswith("sampled_green_abnormal_negative_"):
            description = "GREEN computed on dataset-label-negative/normal organ rows inside the shared fixed study-level sample."
        if description is None and column.startswith("keyword_abnormal_positive_"):
            description = "Keyword behavior computed only on dataset-label-positive abnormal organ rows."
        if description is None and column.startswith("keyword_abnormal_negative_"):
            description = "Keyword behavior computed only on dataset-label-negative/normal organ rows."
        if description:
            lines.append(f"- `{display_names.get(column, column)}`: {description}")
    return lines


def _print_ascii_summary(rows: list[dict[str, Any]]) -> None:
    preferred_columns = [
        "label",
        "organ_Bleu_4",
        "organ_ROUGE_L",
        "organ_CIDEr",
        "organ_GREEN",
        "organ_abnormal_positive_GREEN",
        "organ_abnormal_negative_GREEN",
        "study_Bleu_4",
        "study_ROUGE_L",
        "study_CIDEr",
        "study_GREEN",
        "keyword_csv_positive_pathology_recall",
        "keyword_csv_negative_pathology_rate",
        "keyword_csv_positive_normal_rate",
        "keyword_generated_normal_word_rate",
    ]
    columns = [column for column in preferred_columns if any(column in row for row in rows)]
    if not columns:
        return
    display_names = {
        "label": "label",
        "organ_Bleu_4": "org_B4",
        "organ_ROUGE_L": "org_RL",
        "organ_CIDEr": "org_CIDEr",
        "organ_GREEN": "org_GREEN",
        "organ_abnormal_positive_GREEN": "abn_GREEN",
        "organ_abnormal_negative_GREEN": "norm_GREEN",
        "study_Bleu_4": "study_B4",
        "study_ROUGE_L": "study_RL",
        "study_CIDEr": "study_CIDEr",
        "study_GREEN": "study_GREEN",
        "keyword_csv_positive_pathology_recall": "pos_path_rec",
        "keyword_csv_negative_pathology_rate": "neg_path_rate",
        "keyword_csv_positive_normal_rate": "pos_norm_rate",
        "keyword_generated_normal_word_rate": "gen_norm_rate",
        "radgpt_tumor_macro_f1": "rg_tum_f1",
        "radgpt_malignancy_macro_f1": "rg_malig_f1",
        "radgpt_liver_tumor_f1": "rg_liv_t_f1",
        "radgpt_kidneys_tumor_f1": "rg_kid_t_f1",
        "radgpt_pancreas_tumor_f1": "rg_pan_t_f1",
        "radgpt_liver_malignancy_f1": "rg_liv_m_f1",
        "radgpt_kidneys_malignancy_f1": "rg_kid_m_f1",
        "radgpt_pancreas_malignancy_f1": "rg_pan_m_f1",
    }
    table_rows: list[list[str]] = []
    header = [display_names.get(column, column) for column in columns]
    table_rows.append(header)
    for row in rows:
        rendered = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        table_rows.append(rendered)
    widths = [max(len(table_rows[row_index][col_index]) for row_index in range(len(table_rows))) for col_index in range(len(columns))]

    def _format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"

    rule = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(rule)
    print(_format_row(header))
    print(rule)
    for values in table_rows[1:]:
        print(_format_row(values))
    print(rule)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _slugify(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in str(value).strip())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "run"


if __name__ == "__main__":
    main()
