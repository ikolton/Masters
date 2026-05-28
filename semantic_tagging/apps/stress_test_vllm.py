#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

bootstrap()

from semantic_tagging.backend import build_backend
from semantic_tagging.config import load_config
from semantic_tagging.dataset_adapters import MerlinDatasetAdapter
from semantic_tagging.ontology import OntologyRegistry
from semantic_tagging.paths import ensure_dir
from semantic_tagging.prompting import PromptCompiler
from semantic_tagging.schemas import load_json_schema
from semantic_tagging.types import PromptRequest, UniqueTextRecord
from semantic_tagging.validation import ValidationError, build_tag_decision, parse_llm_json


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test the semantic tagging vLLM backend.")
    parser.add_argument("--config", required=True, help="Path to semantic tagging YAML config.")
    parser.add_argument("--organ", default="Pancreas", help="Organ to sample prompts from.")
    parser.add_argument("--sample-count", type=int, default=32, help="Number of unique texts to benchmark.")
    parser.add_argument("--repeats", type=int, default=2, help="Number of timing repeats per setting.")
    parser.add_argument("--concurrency", default="1,2,4,8,16", help="Comma-separated request concurrency values.")
    parser.add_argument("--max-tokens", default="128,192,256", help="Comma-separated max_tokens values.")
    parser.add_argument("--reference-max-tokens", type=int, default=None, help="Reference max_tokens used for agreement checks.")
    parser.add_argument("--reference-concurrency", type=int, default=4, help="Concurrency for reference generation.")
    parser.add_argument("--max-fewshot-examples", type=int, default=None, help="Optional override for few-shot count.")
    parser.add_argument("--gpu-poll-seconds", type=float, default=0.5, help="GPU telemetry poll interval in seconds.")
    parser.add_argument("--examples-per-class", type=int, default=5, help="How many examples per failure class to save.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    adapter = MerlinDatasetAdapter(paths=config.paths, dataset=config.dataset)
    rows = adapter.iter_source_rows()
    unique_records = adapter.build_unique_text_inventory(rows)
    organ_records = [record for record in unique_records if record.organ == args.organ]
    organ_records = sorted(organ_records, key=lambda item: (-item.count, item.raw_text))[: args.sample_count]
    if not organ_records:
        raise SystemExit(f"No unique texts found for organ={args.organ}")

    prompt_config = config.prompt
    if args.max_fewshot_examples is not None:
        prompt_config = replace(prompt_config, max_fewshot_examples=args.max_fewshot_examples)

    ontology = OntologyRegistry(
        ontology_root=Path(config.paths.ontology_root).expanduser().resolve(),
        config=config.ontology,
    )
    compiler = PromptCompiler(
        prompt_root=Path(config.paths.prompt_root).expanduser().resolve(),
        config=prompt_config,
        ontology=ontology,
    )
    output_schema = load_json_schema(Path(config.paths.prompt_root).expanduser().resolve() / config.prompt.output_schema)
    report_dir = ensure_dir(Path(config.paths.output_root).expanduser().resolve() / "stress_test_reports")

    requests = [
        compiler.compile_request(record, request_id=f"bench:{index}")
        for index, record in enumerate(organ_records, start=1)
    ]

    print(
        f"[stress] organ={args.organ} prompts={len(requests)} "
        f"fewshot={prompt_config.max_fewshot_examples} "
        f"base_model={config.backend.model_name}"
    )

    concurrency_values = parse_int_list(args.concurrency)
    max_tokens_values = parse_int_list(args.max_tokens)
    reference_max_tokens = args.reference_max_tokens or max(max_tokens_values)

    reference_backend = build_backend(
        replace(
            config.backend,
            request_concurrency=max(1, args.reference_concurrency),
            max_tokens=reference_max_tokens,
        )
    )
    print(
        f"[stress] building reference decisions with max_tokens={reference_max_tokens} "
        f"concurrency={max(1, args.reference_concurrency)}"
    )
    reference_responses = reference_backend.generate_batch(requests)
    reference_signatures = build_signatures(
        responses=reference_responses,
        requests=requests,
        records=organ_records,
        ontology=ontology,
        output_schema=output_schema,
    )

    print(
        "concurrency|max_tokens|requests|avg_s|p50_s|p95_s|req_per_s|"
        "valid|invalid_json|invalid_semantic|length_finish|"
        "avg_completion_toks|max_completion_toks|ref_match|"
        "gpu_util_avg|gpu_util_peak|mem_used_peak_gb"
    )
    for concurrency in concurrency_values:
        for max_tokens in max_tokens_values:
            backend_config = replace(
                config.backend,
                request_concurrency=concurrency,
                max_tokens=max_tokens,
            )
            backend = build_backend(backend_config)
            durations: list[float] = []
            last_responses = None
            gpu_samples: list[list[dict[str, float]]] = []
            for _ in range(args.repeats):
                monitor = GPUMonitor(poll_seconds=args.gpu_poll_seconds)
                monitor.start()
                started = time.time()
                responses = backend.generate_batch(requests)
                durations.append(time.time() - started)
                last_responses = responses
                gpu_samples.extend(monitor.stop())
            assert last_responses is not None
            eval_stats = evaluate_responses(
                responses=last_responses,
                requests=requests,
                records=organ_records,
                ontology=ontology,
                output_schema=output_schema,
                reference_signatures=reference_signatures,
                examples_per_class=args.examples_per_class,
            )
            gpu_stats = summarize_gpu_samples(gpu_samples)
            sorted_durations = sorted(durations)
            avg_s = sum(durations) / len(durations)
            p50_s = sorted_durations[len(sorted_durations) // 2]
            p95_s = sorted_durations[min(len(sorted_durations) - 1, int(0.95 * (len(sorted_durations) - 1)))]
            req_per_s = len(requests) / avg_s if avg_s > 0 else 0.0
            report_path = report_dir / (
                f"{config.project.dataset_id}_{args.organ.lower().replace(' ', '_')}"
                f"_c{concurrency}_t{max_tokens}.json"
            )
            report_payload = {
                "dataset_id": config.project.dataset_id,
                "organ": args.organ,
                "sample_count": len(requests),
                "fewshot_examples": prompt_config.max_fewshot_examples,
                "concurrency": concurrency,
                "max_tokens": max_tokens,
                "reference_max_tokens": reference_max_tokens,
                "avg_seconds": avg_s,
                "p50_seconds": p50_s,
                "p95_seconds": p95_s,
                "requests_per_second": req_per_s,
                "evaluation": eval_stats,
                "gpu": gpu_stats,
            }
            report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"{concurrency}|{max_tokens}|{len(requests)}|"
                f"{avg_s:.2f}|{p50_s:.2f}|{p95_s:.2f}|{req_per_s:.2f}|"
                f"{eval_stats['valid']}|{eval_stats['invalid_json']}|{eval_stats['invalid_semantic']}|{eval_stats['length_finish']}|"
                f"{eval_stats['avg_completion_tokens']:.1f}|{eval_stats['max_completion_tokens']}|{eval_stats['reference_match_rate']:.3f}|"
                f"{gpu_stats['util_avg']:.1f}|{gpu_stats['util_peak']:.1f}|{gpu_stats['mem_used_peak_gb']:.1f}"
            )
            print(f"[stress] report saved: {report_path}")


def build_signatures(
    *,
    responses,
    requests: list[PromptRequest],
    records: list[UniqueTextRecord],
    ontology: OntologyRegistry,
    output_schema: dict[str, Any],
) -> dict[str, tuple[Any, ...] | None]:
    signatures: dict[str, tuple[Any, ...] | None] = {}
    for request, response, record in zip(requests, responses, records):
        signatures[request.request_id] = validated_signature(
            response=response,
            record=record,
            ontology=ontology,
            output_schema=output_schema,
        )
    return signatures


def evaluate_responses(
    *,
    responses,
    requests: list[PromptRequest],
    records: list[UniqueTextRecord],
    ontology: OntologyRegistry,
    output_schema: dict[str, Any],
    reference_signatures: dict[str, tuple[Any, ...] | None],
    examples_per_class: int,
) -> dict[str, Any]:
    valid = 0
    invalid_json = 0
    invalid_semantic = 0
    length_finish = 0
    completion_tokens: list[int] = []
    ref_matches = 0
    ref_comparable = 0
    invalid_json_examples: list[dict[str, Any]] = []
    invalid_semantic_examples: list[dict[str, Any]] = []
    valid_mismatch_examples: list[dict[str, Any]] = []
    for request, response, record in zip(requests, responses, records):
        if response.finish_reason == "length":
            length_finish += 1
        if response.completion_tokens is not None:
            completion_tokens.append(int(response.completion_tokens))
        analysis = analyze_response(
            response=response,
            record=record,
            ontology=ontology,
            output_schema=output_schema,
        )
        signature = analysis["signature"]
        if signature is None:
            if analysis["error_kind"] == "invalid_json":
                invalid_json += 1
                if len(invalid_json_examples) < examples_per_class:
                    invalid_json_examples.append(example_payload(request, record, response, analysis, reference_signatures))
            else:
                invalid_semantic += 1
                if len(invalid_semantic_examples) < examples_per_class:
                    invalid_semantic_examples.append(example_payload(request, record, response, analysis, reference_signatures))
            continue
        valid += 1
        ref_signature = reference_signatures.get(request.request_id)
        if ref_signature is not None:
            ref_comparable += 1
            if signature == ref_signature:
                ref_matches += 1
            elif len(valid_mismatch_examples) < examples_per_class:
                valid_mismatch_examples.append(example_payload(request, record, response, analysis, reference_signatures))
    avg_completion = (sum(completion_tokens) / len(completion_tokens)) if completion_tokens else 0.0
    return {
        "valid": valid,
        "invalid_json": invalid_json,
        "invalid_semantic": invalid_semantic,
        "length_finish": length_finish,
        "avg_completion_tokens": avg_completion,
        "max_completion_tokens": max(completion_tokens) if completion_tokens else 0,
        "reference_match_rate": (ref_matches / ref_comparable) if ref_comparable else 0.0,
        "invalid_json_examples": invalid_json_examples,
        "invalid_semantic_examples": invalid_semantic_examples,
        "valid_mismatch_examples": valid_mismatch_examples,
    }


def analyze_response(
    *,
    response,
    record: UniqueTextRecord,
    ontology: OntologyRegistry,
    output_schema: dict[str, Any],
) -> dict[str, Any]:
    try:
        payload = parse_llm_json(response.raw_output)
    except Exception as exc:
        return {
            "signature": None,
            "error_kind": "invalid_json",
            "error": f"{type(exc).__name__}: {exc}",
            "payload": None,
        }
    try:
        decision, proposal = build_tag_decision(
            payload,
            organ=record.organ,
            raw_text=record.raw_text,
            normalized_text=record.normalized_text,
            ontology=ontology,
            output_schema=output_schema,
            source_model=response.model_name,
            source_backend=response.backend_name,
        )
    except Exception as exc:
        return {
            "signature": None,
            "error_kind": "invalid_semantic",
            "error": f"{type(exc).__name__}: {exc}",
            "payload": payload,
        }
    return {
        "signature": (
            decision.normality,
            decision.polarity,
            decision.certainty,
            decision.primary_subtype,
            decision.secondary_subtypes,
            decision.modifiers,
        ),
        "error_kind": None,
        "error": None,
        "payload": payload,
        "decision": decision.to_dict(),
        "proposal": proposal.to_dict() if proposal is not None else None,
    }


def validated_signature(
    *,
    response,
    record: UniqueTextRecord,
    ontology: OntologyRegistry,
    output_schema: dict[str, Any],
) -> tuple[Any, ...] | None:
    return analyze_response(
        response=response,
        record=record,
        ontology=ontology,
        output_schema=output_schema,
    )["signature"]


def example_payload(
    request: PromptRequest,
    record: UniqueTextRecord,
    response,
    analysis: dict[str, Any],
    reference_signatures: dict[str, tuple[Any, ...] | None],
) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "organ": record.organ,
        "raw_text": record.raw_text,
        "normalized_text": record.normalized_text,
        "count": record.count,
        "lesion_positive_rate": record.lesion_positive_rate,
        "abnormal_positive_rate": record.abnormal_positive_rate,
        "finish_reason": response.finish_reason,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "validation_error_kind": analysis.get("error_kind"),
        "validation_error": analysis.get("error"),
        "raw_output": response.raw_output,
        "parsed_payload": analysis.get("payload"),
        "current_signature": analysis.get("signature"),
        "reference_signature": reference_signatures.get(request.request_id),
        "decision": analysis.get("decision"),
        "proposal": analysis.get("proposal"),
    }


class GPUMonitor:
    def __init__(self, *, poll_seconds: float = 0.5) -> None:
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[list[dict[str, float]]] = []

    def start(self) -> None:
        if not shutil_which("nvidia-smi"):
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[list[dict[str, float]]]:
        if self._thread is None:
            return []
        self._stop.set()
        self._thread.join(timeout=self.poll_seconds * 4 + 1.0)
        return self.samples

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = query_gpu_snapshot()
            if sample:
                self.samples.append(sample)
            self._stop.wait(self.poll_seconds)


def query_gpu_snapshot() -> list[dict[str, float]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    rows: list[dict[str, float]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            rows.append(
                {
                    "index": float(parts[0]),
                    "util_gpu": float(parts[1]),
                    "util_mem": float(parts[2]),
                    "mem_used_mb": float(parts[3]),
                    "mem_total_mb": float(parts[4]),
                    "temp_c": float(parts[5]),
                    "power_w": float(parts[6]),
                }
            )
        except ValueError:
            continue
    return rows


def summarize_gpu_samples(samples: list[list[dict[str, float]]]) -> dict[str, float]:
    flattened = [gpu for sample in samples for gpu in sample]
    if not flattened:
        return {"util_avg": 0.0, "util_peak": 0.0, "mem_used_peak_gb": 0.0}
    util_values = [row["util_gpu"] for row in flattened]
    mem_used_values = [row["mem_used_mb"] for row in flattened]
    return {
        "util_avg": sum(util_values) / len(util_values),
        "util_peak": max(util_values),
        "mem_used_peak_gb": max(mem_used_values) / 1024.0,
    }


def shutil_which(command: str) -> str | None:
    try:
        result = subprocess.run(["bash", "-lc", f"command -v {command} || true"], capture_output=True, text=True, check=False)
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


if __name__ == "__main__":
    main()
