#!/usr/bin/env python3
"""Run model-agnostic RadGPT metrics over existing benchmark generations.

This script is intentionally separate from the main decoder benchmark. It can
evaluate a deterministic subset on an interactive 1-GPU node, and the same
entrypoint can later run the full benchmark with a larger vLLM model/TP setup.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.evaluation.radgpt_oncology import evaluate_generation_file_with_radgpt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, help="Existing benchmark dir with runs/*/generations.json.")
    parser.add_argument("--output-dir", default="", help="Output dir. Defaults to <benchmark-dir>/radgpt_benchmark/<run-id>.")
    parser.add_argument("--run-id", default="", help="Output run id. Defaults to timestamp.")
    parser.add_argument("--run-labels", default="", help="Comma-separated run directory names to evaluate. Defaults to all.")
    parser.add_argument("--study-limit", type=int, default=50, help="Number of studies to sample. Use 0 for full benchmark.")
    parser.add_argument("--sample-manifest", default="", help="Optional sampled_green-style manifest with selected (study_id, organ) keys.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--radgpt-root", default="/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadGPT")
    parser.add_argument("--fast", dest="fast", action="store_true")
    parser.add_argument("--slow", dest="fast", action="store_false")
    parser.set_defaults(fast=True)
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--force-generated", action="store_true")
    parser.add_argument("--attach-full-to-evaluations", action="store_true", help="Attach full-run results under radgpt_oncology in each source run evaluation.json.")
    parser.add_argument("--attach-sampled-to-evaluations", action="store_true", help="Attach results under sampled_radgpt_oncology in each source run evaluation.json.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--api-concurrency", type=int, default=1, help="Concurrent RadGPT API requests. This does not change prompts or scoring.")
    parser.add_argument("--launch-vllm", action="store_true", help="Launch and stop a local vLLM server around the benchmark.")
    parser.add_argument("--vllm-python", default="/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-radgpt-vllm/bin/python")
    parser.add_argument("--model", default="iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--server-timeout", type=int, default=1800)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    run_id = str(args.run_id or time.strftime("%Y%m%d_%H%M%S"))
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else benchmark_dir / "radgpt_benchmark" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / "latest_run.txt").write_text(str(output_dir) + "\n", encoding="utf-8")

    run_paths = _discover_runs(benchmark_dir, _parse_csv(args.run_labels))
    sample_manifest_path = Path(args.sample_manifest).expanduser().resolve() if args.sample_manifest else None
    if sample_manifest_path:
        selected_keys, selected_studies, sample_manifest = _load_sample_manifest_keys(sample_manifest_path)
        subset_paths = _write_subset_generations_by_keys(
            run_paths,
            output_dir=output_dir,
            selected_keys=selected_keys,
            selected_studies=selected_studies,
            source_benchmark_dir=benchmark_dir,
            sample_manifest_path=sample_manifest_path,
            sample_manifest=sample_manifest,
            seed=int(args.seed),
        )
    else:
        selected_studies = _select_studies(run_paths, study_limit=int(args.study_limit), seed=int(args.seed))
        selected_keys = []
        subset_paths = _write_subset_generations(
            run_paths,
            output_dir=output_dir,
            selected_studies=selected_studies,
            source_benchmark_dir=benchmark_dir,
            seed=int(args.seed),
        )

    server: subprocess.Popen[str] | None = None
    if args.launch_vllm:
        server = _launch_vllm(args, output_dir=output_dir)
        _wait_for_server(str(args.base_url), server=server, timeout_seconds=int(args.server_timeout), log_path=output_dir / "vllm_server.log")

    try:
        summary = _run_radgpt(
            subset_paths,
            output_dir=output_dir,
            base_url=str(args.base_url),
            fast=bool(args.fast),
            force_reference=bool(args.force_reference),
            force_generated=bool(args.force_generated),
            progress_every=int(args.progress_every),
            api_concurrency=int(args.api_concurrency),
            radgpt_root=str(args.radgpt_root),
            attach_full_to_evaluations=bool(args.attach_full_to_evaluations),
            attach_sampled_to_evaluations=bool(args.attach_sampled_to_evaluations),
            source_benchmark_dir=benchmark_dir,
            sample_manifest_path=sample_manifest_path,
            metadata={
                "source_benchmark_dir": str(benchmark_dir),
                "output_dir": str(output_dir),
                "run_id": run_id,
                "study_limit": int(args.study_limit),
                "selected_study_count": len(selected_studies),
                "selected_key_count": len(selected_keys),
                "sample_manifest_path": str(sample_manifest_path or ""),
                "seed": int(args.seed),
                "base_url": str(args.base_url),
                "model": str(args.model),
                "cuda_visible_devices": str(args.cuda_visible_devices),
                "tensor_parallel_size": int(args.tensor_parallel_size),
                "dtype": str(args.dtype),
                "gpu_memory_utilization": float(args.gpu_memory_utilization),
                "max_model_len": int(args.max_model_len),
                "api_concurrency": int(args.api_concurrency),
                "fast": bool(args.fast),
            },
        )
    finally:
        if server is not None:
            _stop_server(server)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _discover_runs(benchmark_dir: Path, requested: list[str]) -> list[tuple[str, Path]]:
    runs_dir = benchmark_dir / "runs"
    paths = sorted(runs_dir.glob("*/generations.json"))
    if requested:
        wanted = set(requested)
        paths = [path for path in paths if path.parent.name in wanted]
    if not paths:
        raise FileNotFoundError(f"No generation files found under {runs_dir}")
    return [(path.parent.name, path) for path in paths]


def _select_studies(run_paths: list[tuple[str, Path]], *, study_limit: int, seed: int) -> list[str]:
    payload = json.loads(run_paths[0][1].read_text(encoding="utf-8"))
    study_ids = sorted({str(row.get("study_id", "")) for row in payload.get("generations", []) if row.get("study_id")})
    if study_limit <= 0 or study_limit >= len(study_ids):
        return study_ids
    rng = random.Random(seed)
    rng.shuffle(study_ids)
    return sorted(study_ids[:study_limit])


def _write_subset_generations(
    run_paths: list[tuple[str, Path]],
    *,
    output_dir: Path,
    selected_studies: list[str],
    source_benchmark_dir: Path,
    seed: int,
) -> list[tuple[str, Path]]:
    selected = set(selected_studies)
    subset_paths: list[tuple[str, Path]] = []
    manifest = {
        "source_benchmark_dir": str(source_benchmark_dir),
        "seed": int(seed),
        "selected_study_count": len(selected_studies),
        "selected_studies": selected_studies,
    }
    (output_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for label, path in run_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [row for row in payload.get("generations", []) if str(row.get("study_id", "")) in selected]
        subset_payload = {
            **payload,
            "source_generation_path": str(path),
            "source_sample_manifest_digest": str(payload.get("sample_manifest_digest", "")),
            "sample_manifest_digest": f"radgpt-subset-{seed}-{len(selected_studies)}-{payload.get('sample_manifest_digest', '')}",
            "selected_study_count": len(selected_studies),
            "generations": rows,
        }
        target = output_dir / "runs" / label / "generations.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(subset_payload, indent=2, sort_keys=True), encoding="utf-8")
        subset_paths.append((label, target))
    return subset_paths


def _load_sample_manifest_keys(path: Path) -> tuple[list[tuple[str, str]], list[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_keys = payload.get("selected_keys", [])
    selected_keys: list[tuple[str, str]] = []
    for item in raw_keys:
        if not isinstance(item, dict):
            continue
        study_id = str(item.get("study_id", "")).strip()
        organ = str(item.get("organ", "")).strip()
        if study_id and organ:
            selected_keys.append((study_id, organ))
    if not selected_keys:
        raise ValueError(f"No selected (study_id, organ) keys found in {path}")
    selected_studies = sorted({study_id for study_id, _ in selected_keys})
    return selected_keys, selected_studies, payload


def _write_subset_generations_by_keys(
    run_paths: list[tuple[str, Path]],
    *,
    output_dir: Path,
    selected_keys: list[tuple[str, str]],
    selected_studies: list[str],
    source_benchmark_dir: Path,
    sample_manifest_path: Path,
    sample_manifest: dict[str, Any],
    seed: int,
) -> list[tuple[str, Path]]:
    key_order = {key: index for index, key in enumerate(selected_keys)}
    subset_paths: list[tuple[str, Path]] = []
    manifest = {
        "source_benchmark_dir": str(source_benchmark_dir),
        "source_sample_manifest_path": str(sample_manifest_path),
        "source_sample_manifest": sample_manifest,
        "seed": int(seed),
        "selected_key_count": len(selected_keys),
        "selected_study_count": len(selected_studies),
        "selected_studies": selected_studies,
        "sample_key_type": ["study_id", "organ"],
    }
    (output_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for label, path in run_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            row
            for row in payload.get("generations", [])
            if _row_key(row) in key_order
        ]
        rows.sort(key=lambda row: key_order[_row_key(row)])
        subset_payload = {
            **payload,
            "source_generation_path": str(path),
            "source_sample_manifest_path": str(sample_manifest_path),
            "source_sample_manifest_digest": str(payload.get("sample_manifest_digest", "")),
            "sample_manifest_digest": f"radgpt-key-subset-{seed}-{len(selected_keys)}-{payload.get('sample_manifest_digest', '')}",
            "selected_key_count": len(selected_keys),
            "selected_study_count": len(selected_studies),
            "generations": rows,
        }
        target = output_dir / "runs" / label / "generations.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(subset_payload, indent=2, sort_keys=True), encoding="utf-8")
        subset_paths.append((label, target))
    return subset_paths


def _run_radgpt(
    subset_paths: list[tuple[str, Path]],
    *,
    output_dir: Path,
    base_url: str,
    fast: bool,
    force_reference: bool,
    force_generated: bool,
    progress_every: int,
    api_concurrency: int,
    radgpt_root: str,
    attach_full_to_evaluations: bool,
    attach_sampled_to_evaluations: bool,
    source_benchmark_dir: Path,
    sample_manifest_path: Path | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label, generation_path in subset_paths:
        print(f"[radgpt-benchmark] evaluating {label}", flush=True)
        run_dir = output_dir / "runs" / label / "radgpt"
        result = evaluate_generation_file_with_radgpt(
            generation_path,
            benchmark_cache_dir=run_dir,
            reference_cache_dir=output_dir / "radgpt_reference",
            generated_cache_dir=run_dir / "generated",
            comparison_output_path=run_dir / "comparison.json",
            base_url=base_url,
            fast=fast,
            force_reference=force_reference,
            force_generated=force_generated,
            quiet=True,
            progress_every=progress_every,
            api_concurrency=api_concurrency,
            radgpt_root=radgpt_root,
        )
        row = {"label": label, **_flatten_result(result)}
        rows.append(row)
        if attach_full_to_evaluations or attach_sampled_to_evaluations:
            evaluation_path = source_benchmark_dir / "runs" / label / "evaluation.json"
            target_key = "sampled_radgpt_oncology" if attach_sampled_to_evaluations else "radgpt_oncology"
            _locked_update_json(
                evaluation_path,
                {
                    target_key: {
                        **result,
                        "sample_manifest_path": str(sample_manifest_path or ""),
                        "sampled_generation_path": str(generation_path),
                        "api_concurrency": int(api_concurrency),
                        "model": str(metadata.get("model", "")),
                        "fast": bool(metadata.get("fast", True)),
                    }
                },
            )
        print(f"[radgpt-benchmark] done {label} tumor_f1={row.get('tumor_macro_f1')} malignancy_f1={row.get('malignancy_macro_f1')}", flush=True)
    return {"metadata": metadata, "runs": rows}


def _row_key(row: Any) -> tuple[str, str] | None:
    if not isinstance(row, dict):
        return None
    study_id = str(row.get("study_id", "")).strip()
    organ = str(row.get("organ", "")).strip()
    if not study_id or not organ:
        return None
    return (study_id, organ)


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


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in result.get("overall", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[key] = value
    for organ, block in result.get("per_organ", {}).items():
        slug = organ.lower().replace(" ", "_")
        for task in ("tumor", "malignancy"):
            task_block = block.get(task, {})
            if not isinstance(task_block, dict):
                continue
            for key, value in task_block.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[f"{slug}_{task}_{key}"] = value
    row["comparison_path"] = result.get("comparison_path", "")
    row["reference_label_path"] = result.get("reference_label_path", "")
    row["generated_label_path"] = result.get("generated_label_path", "")
    return row


def _launch_vllm(args: argparse.Namespace, *, output_dir: Path) -> subprocess.Popen[str]:
    log_path = output_dir / "vllm_server.log"
    cache_dir = Path(str(args.radgpt_root)).expanduser().resolve() / "evaluate_reports" / "HFCache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env["HF_HOME"] = str(cache_dir)
    env["TRANSFORMERS_CACHE"] = str(cache_dir)
    port = _port_from_base_url(str(args.base_url))
    # Do not use Path.resolve() here. venv Python executables are often
    # symlinks to the base interpreter; resolving the symlink launches the
    # system Python and loses the venv's site-packages.
    vllm_python = Path(os.path.abspath(os.path.expanduser(str(args.vllm_python))))
    if not vllm_python.is_file():
        raise FileNotFoundError(f"RadGPT vLLM Python does not exist: {vllm_python}")
    probe = subprocess.run(
        [str(vllm_python), "-c", "import sys, vllm; print(sys.executable); print(vllm.__version__)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "Configured RadGPT vLLM Python cannot import vllm.\n"
            f"python: {vllm_python}\n"
            f"stdout:\n{probe.stdout}\n"
            f"stderr:\n{probe.stderr}"
        )
    (output_dir / "vllm_python_probe.txt").write_text(probe.stdout + probe.stderr, encoding="utf-8")
    cmd = [
        str(vllm_python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model),
        "--dtype",
        str(args.dtype),
        "--tensor-parallel-size",
        str(int(args.tensor_parallel_size)),
        "--gpu-memory-utilization",
        str(float(args.gpu_memory_utilization)),
        "--max-model-len",
        str(int(args.max_model_len)),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--enforce-eager",
    ]
    (output_dir / "vllm_command.json").write_text(
        json.dumps(
            {
                "cmd": cmd,
                "env": {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"]},
                "vllm_python_arg": str(args.vllm_python),
                "vllm_python_preserved": str(vllm_python),
                "vllm_python_probe": probe.stdout + probe.stderr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
    (output_dir / "vllm_server.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    return process


def _wait_for_server(base_url: str, *, server: subprocess.Popen[str], timeout_seconds: int, log_path: Path) -> None:
    models_url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited before API became ready. See {log_path}\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        print("[radgpt-benchmark] waiting for vLLM API...", flush=True)
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for vLLM API. See {log_path}\n{_tail(log_path)}")


def _stop_server(server: subprocess.Popen[str]) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        server.kill()


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("runs", [])
    columns = [
        "label",
        "tumor_macro_f1",
        "tumor_macro_recall",
        "tumor_macro_precision",
        "malignancy_macro_f1",
        "liver_tumor_f1",
        "kidneys_tumor_f1",
        "pancreas_tumor_f1",
        "comparison_path",
    ]
    lines = ["# RadGPT Benchmark", "", "## Metadata", ""]
    for key, value in summary.get("metadata", {}).items():
        if key == "selected_studies":
            continue
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Results", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"])
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _port_from_base_url(base_url: str) -> int:
    from urllib.parse import urlsplit

    return int(urlsplit(base_url).port or 8000)


def _tail(path: Path, n: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


if __name__ == "__main__":
    main()
