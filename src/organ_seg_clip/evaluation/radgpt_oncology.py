"""RadGPT-backed oncology label evaluation for decoder generations."""

from __future__ import annotations

import contextlib
import concurrent.futures
import hashlib
import io
import importlib
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

DEFAULT_LOCAL_RADGPT_ROOT = "/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadGPT"
DEFAULT_LOCAL_JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SUPPORTED_ONCOLOGY_ORGANS = {
    "Liver": {
        "radgpt_organ": "liver",
    },
    "Kidneys": {
        "radgpt_organ": "kidney",
    },
    "Pancreas": {
        "radgpt_organ": "pancreas",
    },
}


def evaluate_generation_file_with_radgpt(
    generation_path: str | Path,
    *,
    benchmark_cache_dir: str | Path,
    reference_cache_dir: str | Path | None = None,
    generated_cache_dir: str | Path | None = None,
    comparison_output_path: str | Path | None = None,
    base_url: str = "http://0.0.0.0:8000/v1",
    fast: bool = True,
    force_reference: bool = False,
    force_generated: bool = False,
    quiet: bool = True,
    progress_every: int = 50,
    api_concurrency: int = 1,
    radgpt_root: str | Path = DEFAULT_LOCAL_RADGPT_ROOT,
    local_model_name: str | None = None,
    local_model_dtype: str = "bfloat16",
    local_model_max_new_tokens: int = 256,
) -> dict[str, Any]:
    generation_path = Path(generation_path).expanduser().resolve()
    benchmark_cache_dir = Path(benchmark_cache_dir).expanduser().resolve()
    benchmark_cache_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(generation_path.read_text(encoding="utf-8"))
    raw_rows = payload.get("generations")
    if not isinstance(raw_rows, list):
        raise ValueError(f"{generation_path} does not contain a list-valued 'generations' field.")

    reference_rows = build_radgpt_report_rows(raw_rows, text_field="target")
    generated_rows = build_radgpt_report_rows(raw_rows, text_field="generated")

    resolved_base_url = _resolve_client_base_url(base_url)

    metadata = {
        "sample_manifest_digest": str(payload.get("sample_manifest_digest", "")),
        "requested_base_url": str(base_url),
        "resolved_base_url": str(resolved_base_url),
        "fast": bool(fast),
        "radgpt_root": str(Path(radgpt_root).expanduser().resolve()),
        "local_model_name": str(local_model_name or ""),
        "local_model_dtype": str(local_model_dtype),
        "local_model_max_new_tokens": int(local_model_max_new_tokens),
        "schema_version": 2,
    }

    reference_cache_dir = Path(reference_cache_dir).expanduser().resolve() if reference_cache_dir else benchmark_cache_dir / "reference"
    generated_cache_dir = Path(generated_cache_dir).expanduser().resolve() if generated_cache_dir else benchmark_cache_dir / "generated"
    reference_df = load_or_label_radgpt_reports(
        reference_rows,
        cache_dir=reference_cache_dir,
        metadata={**metadata, "kind": "reference", "input_digest": _rows_digest(reference_rows)},
        base_url=resolved_base_url,
        fast=fast,
        force=force_reference,
        quiet=quiet,
        progress_prefix="reference",
        progress_every=progress_every,
        api_concurrency=api_concurrency,
        radgpt_root=radgpt_root,
        local_model_name=local_model_name,
        local_model_dtype=local_model_dtype,
        local_model_max_new_tokens=local_model_max_new_tokens,
    )
    generated_df = load_or_label_radgpt_reports(
        generated_rows,
        cache_dir=generated_cache_dir,
        metadata={**metadata, "kind": "generated", "input_digest": _rows_digest(generated_rows)},
        base_url=resolved_base_url,
        fast=fast,
        force=force_generated,
        quiet=quiet,
        progress_prefix="generated",
        progress_every=progress_every,
        api_concurrency=api_concurrency,
        radgpt_root=radgpt_root,
        local_model_name=local_model_name,
        local_model_dtype=local_model_dtype,
        local_model_max_new_tokens=local_model_max_new_tokens,
    )

    result = compare_radgpt_labels(reference_df, generated_df)
    result["generation_path"] = str(generation_path)
    result["reference_label_path"] = str((reference_cache_dir / "labels.csv").resolve())
    result["generated_label_path"] = str((generated_cache_dir / "labels.csv").resolve())
    result["requested_base_url"] = str(base_url)
    result["base_url"] = str(resolved_base_url)
    result["fast"] = bool(fast)
    result["local_model_name"] = str(local_model_name or "")
    result["supported_organs"] = list(SUPPORTED_ONCOLOGY_ORGANS.keys())
    result["row_count_reference"] = int(len(reference_df))
    result["row_count_generated"] = int(len(generated_df))

    output_path = Path(comparison_output_path).expanduser().resolve() if comparison_output_path else benchmark_cache_dir / "comparison.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["comparison_path"] = str(output_path.resolve())
    return result


def build_radgpt_report_rows(raw_rows: list[Any], *, text_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            continue
        organ = str(raw.get("organ", "")).strip()
        if organ not in SUPPORTED_ONCOLOGY_ORGANS:
            continue
        text = str(raw.get(text_field, "")).strip()
        study_id = str(raw.get("study_id", "")).strip()
        if not text or not study_id:
            continue
        rows.append(
            {
                "sample_id": f"{study_id}::{organ}::{index}",
                "study_id": study_id,
                "organ": organ,
                "radgpt_organ": SUPPORTED_ONCOLOGY_ORGANS[organ]["radgpt_organ"],
                "report_text": text,
            }
        )
    return rows


def load_or_label_radgpt_reports(
    report_rows: list[dict[str, Any]],
    *,
    cache_dir: str | Path,
    metadata: dict[str, Any],
    base_url: str,
    fast: bool,
    force: bool,
    quiet: bool,
    progress_prefix: str,
    progress_every: int,
    api_concurrency: int = 1,
    radgpt_root: str | Path,
    local_model_name: str | None = None,
    local_model_dtype: str = "bfloat16",
    local_model_max_new_tokens: int = 256,
    labeler: Callable[..., dict[str, Any]] | None = None,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    labels_path = cache_dir / "labels.csv"
    inputs_path = cache_dir / "inputs.csv"

    expected_metadata = {
        **metadata,
        "row_count": len(report_rows),
    }

    existing_df = pd.DataFrame()
    if not force and metadata_matches(metadata_path, expected_metadata) and labels_path.is_file():
        existing_df = pd.read_csv(labels_path)
    if force or not metadata_matches(metadata_path, expected_metadata):
        existing_df = pd.DataFrame()

    done_ids = set(existing_df.get("sample_id", pd.Series(dtype=str)).astype(str).tolist()) if not existing_df.empty else set()
    pending_rows = [row for row in report_rows if row["sample_id"] not in done_ids]

    if pending_rows:
        labeler_fn = labeler or run_radgpt_labeler
        pending_df = labeler_fn(
            pending_rows,
            base_url=base_url,
            fast=fast,
            quiet=quiet,
            progress_prefix=progress_prefix,
            progress_every=progress_every,
            api_concurrency=api_concurrency,
            radgpt_root=radgpt_root,
            local_model_name=local_model_name,
            local_model_dtype=local_model_dtype,
            local_model_max_new_tokens=local_model_max_new_tokens,
        )
        combined_df = pd.concat([existing_df, pending_df], ignore_index=True) if not existing_df.empty else pending_df
    else:
        combined_df = existing_df

    ordered_df = _order_label_rows(combined_df)
    ordered_df.to_csv(labels_path, index=False)
    pd.DataFrame(report_rows).to_csv(inputs_path, index=False)
    metadata_path.write_text(json.dumps(expected_metadata, indent=2, sort_keys=True), encoding="utf-8")
    return ordered_df


def run_radgpt_labeler(
    report_rows: list[dict[str, Any]],
    *,
    base_url: str,
    fast: bool,
    quiet: bool,
    progress_prefix: str,
    progress_every: int,
    api_concurrency: int = 1,
    radgpt_root: str | Path,
    local_model_name: str | None,
    local_model_dtype: str,
    local_model_max_new_tokens: int,
) -> pd.DataFrame:
    radgpt = _load_radgpt_module(radgpt_root)
    if not local_model_name and int(api_concurrency) > 1:
        return _run_radgpt_labeler_concurrent_api(
            radgpt,
            report_rows,
            base_url=base_url,
            fast=fast,
            quiet=quiet,
            progress_prefix=progress_prefix,
            progress_every=progress_every,
            api_concurrency=int(api_concurrency),
        )
    if local_model_name:
        _install_local_radgpt_backend(
            radgpt,
            model_name=str(local_model_name),
            dtype=str(local_model_dtype),
            max_new_tokens=int(local_model_max_new_tokens),
        )
    else:
        _preflight_radgpt_api(radgpt, base_url=base_url, quiet=quiet)
    records: list[dict[str, Any]] = []
    total = len(report_rows)
    for index, row in enumerate(report_rows, start=1):
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
            print(f"[radgpt] {progress_prefix} {index}/{total}")
        sample_df = pd.DataFrame(
            [
                {
                    "Anon Acc #": row["sample_id"],
                    "Anon Report Text": row["report_text"],
                }
            ]
        )
        tumor_answer = ""
        malignancy_answer = ""
        tumor_label: int | None = None
        malignancy_label: int | None = None
        warning = ""
        try:
            tumor_answer = _call_radgpt(
                radgpt.run,
                quiet=quiet,
                target=0,
                examples=[],
                data=sample_df,
                base_url=base_url,
                print_message=False,
                step="tumor detection",
                organ=row["radgpt_organ"],
                fast=fast,
                row_name="Anon Report Text",
                id_column="Anon Acc #",
            )
            tumor_output = _call_radgpt(
                radgpt.interpret_output,
                quiet=quiet,
                string=tumor_answer,
                step="tumor detection",
                organ=row["radgpt_organ"],
            )
            tumor_label = _normalize_binary_label(
                tumor_output.get(_tumor_column(row["radgpt_organ"]))
            )
            if tumor_label == 0:
                malignancy_label = 0
            elif tumor_label == 1:
                malignancy_answer = _call_radgpt(
                    radgpt.run,
                    quiet=quiet,
                    target=0,
                    examples=[],
                    data=sample_df,
                    base_url=base_url,
                    print_message=False,
                    step="malignancy detection",
                    organ=row["radgpt_organ"],
                    fast=fast,
                    row_name="Anon Report Text",
                    id_column="Anon Acc #",
                )
                malignancy_output = _call_radgpt(
                    radgpt.interpret_output,
                    quiet=quiet,
                    string=malignancy_answer,
                    step="malignancy detection",
                    organ=row["radgpt_organ"],
                )
                malignancy_label = _normalize_binary_label(
                    malignancy_output.get(_malignancy_column(row["radgpt_organ"]))
                )
        except Exception as exc:  # pragma: no cover - API/runtime dependent
            warning = f"{exc.__class__.__name__}: {exc}"

        records.append(
            {
                "sample_id": row["sample_id"],
                "study_id": row["study_id"],
                "organ": row["organ"],
                "radgpt_organ": row["radgpt_organ"],
                "report_text": row["report_text"],
                "tumor_label": tumor_label,
                "malignancy_label": malignancy_label,
                "tumor_answer": tumor_answer,
                "malignancy_answer": malignancy_answer,
                "warning": warning,
            }
        )
    if not local_model_name:
        _raise_if_transport_failed(records, base_url=base_url, progress_prefix=progress_prefix)
    return pd.DataFrame.from_records(records)


_RADGPT_PROMPT_LOCK = threading.Lock()


def _run_radgpt_labeler_concurrent_api(
    radgpt: Any,
    report_rows: list[dict[str, Any]],
    *,
    base_url: str,
    fast: bool,
    quiet: bool,
    progress_prefix: str,
    progress_every: int,
    api_concurrency: int,
) -> pd.DataFrame:
    from openai import OpenAI

    _preflight_radgpt_api(radgpt, base_url=base_url, quiet=quiet)
    client = OpenAI(api_key="YOUR_API_KEY", base_url=base_url)
    model_name = client.models.list().data[0].id
    total = len(report_rows)
    records_by_sample_id: dict[str, dict[str, Any]] = {}
    completed = 0

    def _one(row: dict[str, Any]) -> dict[str, Any]:
        sample_df = pd.DataFrame(
            [
                {
                    "Anon Acc #": row["sample_id"],
                    "Anon Report Text": row["report_text"],
                }
            ]
        )
        tumor_answer = ""
        malignancy_answer = ""
        tumor_label: int | None = None
        malignancy_label: int | None = None
        warning = ""
        try:
            tumor_answer = _radgpt_api_completion(
                radgpt,
                client=client,
                model_name=model_name,
                data=sample_df,
                step="tumor detection",
                organ=row["radgpt_organ"],
                base_url=base_url,
                fast=fast,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                tumor_output = radgpt.interpret_output(
                    tumor_answer,
                    step="tumor detection",
                    organ=row["radgpt_organ"],
                )
            tumor_label = _normalize_binary_label(tumor_output.get(_tumor_column(row["radgpt_organ"])))
            if tumor_label == 0:
                malignancy_label = 0
            elif tumor_label == 1:
                malignancy_answer = _radgpt_api_completion(
                    radgpt,
                    client=client,
                    model_name=model_name,
                    data=sample_df,
                    step="malignancy detection",
                    organ=row["radgpt_organ"],
                    base_url=base_url,
                    fast=fast,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    malignancy_output = radgpt.interpret_output(
                        malignancy_answer,
                        step="malignancy detection",
                        organ=row["radgpt_organ"],
                    )
                malignancy_label = _normalize_binary_label(
                    malignancy_output.get(_malignancy_column(row["radgpt_organ"]))
                )
        except Exception as exc:  # pragma: no cover - API/runtime dependent
            warning = f"{exc.__class__.__name__}: {exc}"
        return {
            "sample_id": row["sample_id"],
            "study_id": row["study_id"],
            "organ": row["organ"],
            "radgpt_organ": row["radgpt_organ"],
            "report_text": row["report_text"],
            "tumor_label": tumor_label,
            "malignancy_label": malignancy_label,
            "tumor_answer": tumor_answer,
            "malignancy_answer": malignancy_answer,
            "warning": warning,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(api_concurrency))) as executor:
        future_to_id = {executor.submit(_one, row): row["sample_id"] for row in report_rows}
        for future in concurrent.futures.as_completed(future_to_id):
            records_by_sample_id[future_to_id[future]] = future.result()
            completed += 1
            if progress_every > 0 and (completed == 1 or completed % progress_every == 0 or completed == total):
                print(f"[radgpt] {progress_prefix} {completed}/{total}", flush=True)

    records = [records_by_sample_id[row["sample_id"]] for row in report_rows if row["sample_id"] in records_by_sample_id]
    _raise_if_transport_failed(records, base_url=base_url, progress_prefix=progress_prefix)
    return pd.DataFrame.from_records(records)


def _radgpt_api_completion(
    radgpt: Any,
    *,
    client: Any,
    model_name: str,
    data: pd.DataFrame,
    step: str,
    organ: str,
    base_url: str,
    fast: bool,
    max_attempts: int = 4,
) -> str:
    del base_url
    with _RADGPT_PROMPT_LOCK:
        with contextlib.redirect_stdout(io.StringIO()):
            message = radgpt.create_conversation(
                data=data,
                target=0,
                examples=[],
                step=step,
                organ=organ,
                fast=fast,
                row_name="Anon Report Text",
            )
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=0,
                top_p=1,
                timeout=6000,
            )
            return str(response.choices[0].message.content or "")
        except Exception as exc:  # pragma: no cover - API/runtime dependent
            last_exc = exc
            if attempt == max_attempts:
                break
            time.sleep(min(30.0, 2.0 ** attempt))
    raise RuntimeError(f"RadGPT API completion failed after {max_attempts} attempts: {last_exc}") from last_exc


def compare_radgpt_labels(reference_df: pd.DataFrame, generated_df: pd.DataFrame) -> dict[str, Any]:
    merged = reference_df.merge(
        generated_df,
        on="sample_id",
        suffixes=("_reference", "_generated"),
        how="inner",
    )
    per_organ: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    if len(reference_df) != len(generated_df):
        warnings.append(
            f"Reference rows={len(reference_df)} and generated rows={len(generated_df)} differ; compared intersection only."
        )
    for organ_name in SUPPORTED_ONCOLOGY_ORGANS:
        organ_rows = merged[merged["organ_reference"] == organ_name].copy()
        per_organ[organ_name] = {
            "count": int(len(organ_rows)),
            "tumor": _binary_metrics(
                organ_rows,
                reference_column="tumor_label_reference",
                generated_column="tumor_label_generated",
            ),
            "malignancy": _binary_metrics(
                organ_rows,
                reference_column="malignancy_label_reference",
                generated_column="malignancy_label_generated",
            ),
        }

    tumor_macro = _macro_metrics(per_organ, task_name="tumor")
    malignancy_macro = _macro_metrics(per_organ, task_name="malignancy")
    overall = {
        "tumor_macro_accuracy": tumor_macro.get("accuracy"),
        "tumor_macro_precision": tumor_macro.get("precision"),
        "tumor_macro_recall": tumor_macro.get("recall"),
        "tumor_macro_f1": tumor_macro.get("f1"),
        "tumor_macro_specificity": tumor_macro.get("specificity"),
        "malignancy_macro_accuracy": malignancy_macro.get("accuracy"),
        "malignancy_macro_precision": malignancy_macro.get("precision"),
        "malignancy_macro_recall": malignancy_macro.get("recall"),
        "malignancy_macro_f1": malignancy_macro.get("f1"),
        "malignancy_macro_specificity": malignancy_macro.get("specificity"),
        "all_macro_f1": _mean_ignore_nan([tumor_macro.get("f1"), malignancy_macro.get("f1")]),
    }
    return {
        "overall": overall,
        "per_organ": per_organ,
        "warnings": warnings,
    }


def metadata_matches(path: str | Path, expected_metadata: dict[str, Any]) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload == expected_metadata


def _binary_metrics(df: pd.DataFrame, *, reference_column: str, generated_column: str) -> dict[str, Any]:
    if df.empty:
        return _empty_metric_block()
    ref_series = df[reference_column]
    gen_series = df[generated_column]
    valid_mask = ref_series.notna() & gen_series.notna()
    valid_df = df[valid_mask]
    ref_uncertain_count = int(ref_series.isna().sum())
    gen_uncertain_count = int(gen_series.isna().sum())
    if valid_df.empty:
        block = _empty_metric_block()
        block.update(
            {
                "count_valid": 0,
                "count_total": int(len(df)),
                "reference_uncertain_count": ref_uncertain_count,
                "generated_uncertain_count": gen_uncertain_count,
            }
        )
        return block

    ref_values = valid_df[reference_column].astype(int).tolist()
    gen_values = valid_df[generated_column].astype(int).tolist()
    tp = sum(1 for ref, pred in zip(ref_values, gen_values) if ref == 1 and pred == 1)
    tn = sum(1 for ref, pred in zip(ref_values, gen_values) if ref == 0 and pred == 0)
    fp = sum(1 for ref, pred in zip(ref_values, gen_values) if ref == 0 and pred == 1)
    fn = sum(1 for ref, pred in zip(ref_values, gen_values) if ref == 1 and pred == 0)
    accuracy = (tp + tn) / len(valid_df) if len(valid_df) else math.nan
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    elif (tp + fn) > 0:
        precision = 0.0
    else:
        precision = math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    if not math.isnan(precision) and not math.isnan(recall):
        if (precision + recall) > 0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0
    else:
        f1 = math.nan
    return {
        "count_total": int(len(df)),
        "count_valid": int(len(valid_df)),
        "reference_uncertain_count": ref_uncertain_count,
        "generated_uncertain_count": gen_uncertain_count,
        "reference_positive_rate": sum(ref_values) / len(valid_df) if len(valid_df) else math.nan,
        "generated_positive_rate": sum(gen_values) / len(valid_df) if len(valid_df) else math.nan,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def _macro_metrics(per_organ: dict[str, dict[str, Any]], *, task_name: str) -> dict[str, float]:
    metrics = {}
    for key in ("accuracy", "precision", "recall", "specificity", "f1"):
        values = [float(block[task_name][key]) for block in per_organ.values() if _is_number(block[task_name].get(key))]
        metrics[key] = _mean_ignore_nan(values)
    return metrics


def _empty_metric_block() -> dict[str, Any]:
    return {
        "count_total": 0,
        "count_valid": 0,
        "reference_uncertain_count": 0,
        "generated_uncertain_count": 0,
        "reference_positive_rate": math.nan,
        "generated_positive_rate": math.nan,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "accuracy": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "specificity": math.nan,
        "f1": math.nan,
    }


def _call_radgpt(function: Callable[..., Any], *, quiet: bool, **kwargs: Any) -> Any:
    if not quiet:
        return function(**kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return function(**kwargs)


def _load_radgpt_module(radgpt_root: str | Path) -> Any:
    radgpt_root = Path(radgpt_root).expanduser().resolve()
    module_root = radgpt_root / "evaluate_reports"
    if not module_root.is_dir():
        raise FileNotFoundError(f"RadGPT evaluate_reports directory not found at {module_root}")
    module_root_str = str(module_root)
    if module_root_str not in sys.path:
        sys.path.insert(0, module_root_str)
    return importlib.import_module("RadGPT")


_LOCAL_RADGPT_BACKENDS: dict[tuple[str, str, int], Any] = {}


def _install_local_radgpt_backend(
    radgpt: Any,
    *,
    model_name: str,
    dtype: str,
    max_new_tokens: int,
) -> None:
    backend = _get_local_radgpt_backend(
        model_name=model_name,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )

    def _send_message_local(
        text,
        conver,
        base_url="local://radgpt",
        prt=True,
        max_tokens=None,
        batch=1,
        labels=None,
        id=None,
    ):
        del base_url, labels
        if text is not None:
            if batch > 1:
                for i in range(batch):
                    conver[i] = radgpt.CreateConversation(text=text[i], conver=conver[i])
            else:
                conver = radgpt.CreateConversation(text=text, conver=conver)

        if batch == 1:
            answer = backend.generate(conver, max_tokens=max_tokens)
            if prt:
                print("Conversation:")
                for item in conver:
                    print(item["content"])
                if id is not None:
                    print("ID:", id)
                print("Answer:", answer)
            conver.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
            return conver, answer

        answers: list[str] = []
        for i in range(batch):
            answer = backend.generate(conver[i], max_tokens=max_tokens)
            answers.append(answer)
            if prt:
                print("Conversation:")
                for item in conver[i]:
                    print(item["content"])
                if id is not None:
                    print("ID:", id[i] if isinstance(id, list) else id)
                print("Answer:", answer)
            conver[i].append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return conver, answers

    radgpt.clt = object()
    radgpt.mdl = model_name
    radgpt.InitializeOpenAIClient = lambda base_url="local://radgpt": (radgpt.clt, radgpt.mdl)
    radgpt.SendMessageAPI = _send_message_local


class _LocalRadGPTBackend:
    def __init__(self, *, model_name: str, dtype: str, max_new_tokens: int) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": "auto",
        }
        resolved_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if resolved_dtype != "auto":
            kwargs["torch_dtype"] = resolved_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            **kwargs,
        )
        self.max_new_tokens = int(max_new_tokens)

    def generate(self, conversation: list[dict[str, Any]], *, max_tokens: int | None) -> str:
        import torch

        messages = []
        for item in conversation:
            raw_content = item.get("content", [])
            if isinstance(raw_content, list):
                text = "".join(
                    str(part.get("text", ""))
                    for part in raw_content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                text = str(raw_content)
            messages.append({"role": str(item.get("role", "user")), "content": text})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": int(max_tokens or self.max_new_tokens),
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _get_local_radgpt_backend(*, model_name: str, dtype: str, max_new_tokens: int) -> _LocalRadGPTBackend:
    key = (str(model_name), str(dtype), int(max_new_tokens))
    backend = _LOCAL_RADGPT_BACKENDS.get(key)
    if backend is None:
        backend = _LocalRadGPTBackend(
            model_name=model_name,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
        _LOCAL_RADGPT_BACKENDS[key] = backend
    return backend


def _resolve_client_base_url(base_url: str) -> str:
    parts = urlsplit(str(base_url))
    if parts.hostname != "0.0.0.0":
        return str(base_url)
    netloc = parts.netloc.replace("0.0.0.0", "127.0.0.1", 1)
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _preflight_radgpt_api(radgpt: Any, *, base_url: str, quiet: bool) -> None:
    try:
        _call_radgpt(radgpt.InitializeOpenAIClient, quiet=quiet, base_url=base_url)
    except Exception as exc:  # pragma: no cover - API/runtime dependent
        raise RuntimeError(
            "RadGPT API preflight failed before labeling started. "
            f"Resolved base URL: {base_url}. "
            "The original RadGPT implementation initializes an OpenAI-compatible client "
            "in `evaluate_reports/RadGPT.py` via `InitializeOpenAIClient(...)`; "
            "make sure the vLLM/OpenAI server is running and reachable from this node."
        ) from exc


def _raise_if_transport_failed(records: list[dict[str, Any]], *, base_url: str, progress_prefix: str) -> None:
    warnings = [str(record.get("warning", "")).strip() for record in records]
    if not warnings:
        return
    transport_failures = [warning for warning in warnings if "APIConnectionError" in warning]
    answered = [
        record
        for record in records
        if str(record.get("tumor_answer", "")).strip() or str(record.get("malignancy_answer", "")).strip()
    ]
    if transport_failures and len(transport_failures) == len(records) and not answered:
        raise RuntimeError(
            "RadGPT labeling produced only API connection failures "
            f"for the '{progress_prefix}' pass at {base_url}. "
            "No usable labels were extracted, so the cache was not trusted. "
            "Check the OpenAI-compatible server endpoint and prefer `127.0.0.1` over `0.0.0.0` for client calls."
        )


def _tumor_column(radgpt_organ: str) -> str:
    mapping = {
        "liver": "Liver Tumor",
        "kidney": "Kidney Tumor",
        "pancreas": "Pancreas Tumor",
    }
    return mapping[radgpt_organ]


def _malignancy_column(radgpt_organ: str) -> str:
    return f"Malignant Tumor in {radgpt_organ}"


def _normalize_binary_label(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric):
        return None
    if numeric >= 0.5:
        return 1
    return 0


def _order_label_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    columns = [
        "sample_id",
        "study_id",
        "organ",
        "radgpt_organ",
        "report_text",
        "tumor_label",
        "malignancy_label",
        "tumor_answer",
        "malignancy_answer",
        "warning",
    ]
    available_columns = [column for column in columns if column in df.columns]
    return df.sort_values(["study_id", "organ", "sample_id"]).reset_index(drop=True)[available_columns]


def _rows_digest(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "sample_id": row["sample_id"],
            "study_id": row["study_id"],
            "organ": row["organ"],
            "report_text": row["report_text"],
        }
        for row in rows
    ]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _mean_ignore_nan(values: list[float | None]) -> float:
    usable = [float(value) for value in values if _is_number(value)]
    if not usable:
        return math.nan
    return sum(usable) / len(usable)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value))
