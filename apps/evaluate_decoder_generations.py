#!/usr/bin/env python3
"""Evaluate saved per-organ decoder generations with text metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from analyze_decoder_generations import _summarize as summarize_keyword_metrics


DEFAULT_COCO_METRICS = (
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "meteor",
    "ROUGE_L",
    "CIDEr",
)
METRIC_REFERENCE = {
    "Bleu_1": {"direction": "higher_better", "max": 1.0},
    "Bleu_2": {"direction": "higher_better", "max": 1.0},
    "Bleu_3": {"direction": "higher_better", "max": 1.0},
    "Bleu_4": {"direction": "higher_better", "max": 1.0},
    "METEOR": {"direction": "higher_better", "max": 1.0},
    "ROUGE_L": {"direction": "higher_better", "max": 1.0},
    "CIDEr": {"direction": "higher_better", "max": None},
    "GREEN": {"direction": "higher_better", "max": 1.0},
}
DEFAULT_ORGAN_ORDER = (
    "Spleen",
    "Kidneys",
    "Gallbladder",
    "Liver",
    "Stomach",
    "Pancreas",
    "Adrenal glands",
    "Small bowel",
    "Colon",
    "Urinary bladder",
    "Prostate",
)
DEFAULT_LOCAL_PYCOCOEVALCAP_ROOT = "/net/scratch/hscra/plgrid/plgikolton/Magisterka/pycocoevalcap"
DEFAULT_LOCAL_RADEVAL_ROOT = "/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadEval"
_GREEN_EVALUATOR = None
_GREEN_EVALUATOR_SETTINGS = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON file from apps/generate_decoder.py, or a raw row list.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_COCO_METRICS),
        help="Comma-separated pycocoevalcap metric names. Defaults to Bleu_1..4, METEOR, ROUGE_L, CIDEr.",
    )
    parser.add_argument(
        "--tokenize",
        choices=("auto", "java", "none"),
        default="auto",
        help="Use pycocoevalcap PTB Java tokenization, no tokenization, or auto-detect Java.",
    )
    parser.add_argument("--green", action="store_true", help="Also compute RadEval GREEN. This is model-heavy and intended for GPU runs.")
    parser.add_argument(
        "--green-scope",
        choices=("organ", "study", "both"),
        default="organ",
        help="Where to compute GREEN when --green is set.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on usable rows, useful for heavy metric smoke tests.")
    parser.add_argument("--no-study-level", action="store_true", help="Disable reconstructed study-level support metrics.")
    parser.add_argument("--green-batch-size", type=int, default=32, help="GREEN judge batch size.")
    parser.add_argument("--green-max-new-tokens", type=int, default=192, help="GREEN judge max_new_tokens.")
    parser.add_argument("--green-prompt-max-length", type=int, default=2048, help="GREEN prompt truncation length.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation for output.")
    args = parser.parse_args()

    result = evaluate_file(
        Path(args.input).expanduser(),
        metrics=_parse_csv(args.metrics),
        tokenize_mode=args.tokenize,
        green_scope=args.green_scope if args.green else "none",
        limit=args.limit,
        include_study_level=not args.no_study_level,
        green_batch_size=args.green_batch_size,
        green_max_new_tokens=args.green_max_new_tokens,
        green_prompt_max_length=args.green_prompt_max_length,
    )
    text = json.dumps(result, indent=args.indent)
    if args.output:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)


def evaluate_file(
    path: Path,
    *,
    metrics: list[str],
    tokenize_mode: str,
    green_scope: str,
    limit: int | None,
    include_study_level: bool,
    green_batch_size: int = 32,
    green_max_new_tokens: int = 192,
    green_prompt_max_length: int = 2048,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows, payload_warning = _extract_rows(payload)
    rows, input_summary, warnings = _validate_rows(raw_rows)
    if payload_warning:
        warnings.append(payload_warning)
    if limit is not None and limit >= 0 and len(rows) > limit:
        rows = rows[:limit]
        input_summary["rows_used_after_limit"] = len(rows)
        warnings.append(f"Limited evaluation to first {limit} usable rows.")

    unavailable: dict[str, str] = {}
    tokenize, tokenize_warning = _resolve_tokenize(tokenize_mode)
    if tokenize_warning:
        warnings.append(tokenize_warning)
    metrics = _normalize_coco_metrics(metrics)
    java_path = shutil.which("java")
    java_available = bool(java_path and _java_runs(java_path))
    metrics = _filter_coco_metrics(metrics, java_available=java_available, warnings=warnings)

    result: dict[str, Any] = {
        "input": str(path.expanduser().resolve()),
        "input_summary": input_summary,
        "warnings": warnings,
        "unavailable_metrics": unavailable,
        "metric_backend": "pycocoevalcap",
        "coco_tokenize": "java" if tokenize else "none",
        "green_scope": green_scope,
        "organ_level": {},
    }
    if rows:
        result["organ_level"] = _evaluate_rows(
            rows,
            metrics=metrics,
            tokenize=tokenize,
            unavailable=unavailable,
            include_green=green_scope in {"organ", "both"},
            green_batch_size=green_batch_size,
            green_max_new_tokens=green_max_new_tokens,
            green_prompt_max_length=green_prompt_max_length,
        )
        result["keyword_diagnostics"] = summarize_keyword_metrics(rows)
        if include_study_level:
            study_rows, study_warning = _build_study_rows(rows)
            if study_warning:
                warnings.append(study_warning)
            if study_rows:
                result["study_level_support"] = _evaluate_rows(
                    study_rows,
                    metrics=metrics,
                    tokenize=tokenize,
                    unavailable=unavailable,
                    include_green=green_scope in {"study", "both"},
                    include_per_organ=False,
                    include_lesion=False,
                    green_batch_size=green_batch_size,
                    green_max_new_tokens=green_max_new_tokens,
                    green_prompt_max_length=green_prompt_max_length,
                )
                result["study_level_support"]["count"] = len(study_rows)
    else:
        warnings.append("No usable rows found; text metrics were not computed.")
    result["per_organ_summary"] = _build_per_organ_summary(result)
    result["score_summary"] = _build_score_summary(result["per_organ_summary"])
    return result


def _extract_rows(payload: Any) -> tuple[list[Any], str]:
    if isinstance(payload, dict):
        rows = payload.get("generations")
        if isinstance(rows, list):
            return rows, ""
        return [], "Input JSON is an object but does not contain a list-valued 'generations' field."
    if isinstance(payload, list):
        return payload, "Input JSON is a raw list; accepted, but expected a top-level 'generations' field."
    return [], "Input JSON must be either an object with 'generations' or a raw list of rows."


def _validate_rows(raw_rows: list[Any]) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    summary = {
        "rows_seen": len(raw_rows),
        "rows_used": 0,
        "rows_skipped": 0,
        "non_object_rows": 0,
        "missing_generated": 0,
        "missing_target": 0,
        "missing_organ": 0,
        "missing_study_id": 0,
        "missing_organ_abnormal_label": 0,
        "missing_lesion_label": 0,
    }
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    skipped_examples: list[str] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            summary["non_object_rows"] += 1
            summary["rows_skipped"] += 1
            _append_example(skipped_examples, f"row {index}: not an object")
            continue
        missing_required = []
        generated = _clean_text(raw.get("generated"))
        target = _clean_text(raw.get("target"))
        if not generated:
            summary["missing_generated"] += 1
            missing_required.append("generated")
        if not target:
            summary["missing_target"] += 1
            missing_required.append("target")
        if missing_required:
            summary["rows_skipped"] += 1
            _append_example(skipped_examples, f"row {index}: missing/empty {', '.join(missing_required)}")
            continue
        if not _clean_text(raw.get("organ")):
            summary["missing_organ"] += 1
        if not _clean_text(raw.get("study_id")):
            summary["missing_study_id"] += 1
        if raw.get("organ_abnormal_label") is None:
            summary["missing_organ_abnormal_label"] += 1
        if raw.get("lesion_label") is None:
            summary["missing_lesion_label"] += 1
        row = dict(raw)
        row["generated"] = generated
        row["target"] = target
        rows.append(row)
    summary["rows_used"] = len(rows)
    if skipped_examples:
        warnings.append(f"Skipped {summary['rows_skipped']} rows with invalid required fields. Examples: {'; '.join(skipped_examples)}")
    if summary["missing_organ"]:
        warnings.append(f"{summary['missing_organ']} usable rows are missing 'organ'; excluded from per-organ metrics.")
    if summary["missing_study_id"]:
        warnings.append(f"{summary['missing_study_id']} usable rows are missing 'study_id'; study-level support metrics may be partial or disabled.")
    if summary["missing_organ_abnormal_label"]:
        warnings.append(f"{summary['missing_organ_abnormal_label']} usable rows are missing 'organ_abnormal_label'; excluded from dataset-label-stratified metrics.")
    if summary["missing_lesion_label"]:
        warnings.append(f"{summary['missing_lesion_label']} usable rows are missing 'lesion_label'; excluded from lesion-stratified metrics.")
    return rows, summary, warnings


def _evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    metrics: list[str],
    tokenize: bool,
    unavailable: dict[str, str],
    include_green: bool = False,
    include_per_organ: bool = True,
    include_lesion: bool = True,
    green_batch_size: int = 32,
    green_max_new_tokens: int = 192,
    green_prompt_max_length: int = 2048,
) -> dict[str, Any]:
    corpus_scores, sentence_scores = _run_coco_metrics(
        rows,
        metrics=metrics,
        tokenize=tokenize,
        unavailable=unavailable,
    )
    if include_green:
        green_score, green_sentence_scores = _run_green_metric(
            rows,
            unavailable=unavailable,
            green_batch_size=green_batch_size,
            green_max_new_tokens=green_max_new_tokens,
            green_prompt_max_length=green_prompt_max_length,
        )
        corpus_scores.update(green_score)
        sentence_scores.update(green_sentence_scores)
    output: dict[str, Any] = {
        "count": len(rows),
        "overall": corpus_scores,
    }
    if include_per_organ:
        per_organ = {}
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            organ = _clean_text(row.get("organ"))
            if organ:
                buckets[organ].append(index)
        for organ, indices in sorted(buckets.items()):
            per_organ[organ] = _mean_sentence_scores(sentence_scores, indices) | {"count": len(indices)}
        output["per_organ"] = per_organ
        output["per_organ_aggregation"] = "mean_sentence_scores"
    if include_lesion:
        abnormal_buckets: dict[str, list[int]] = {"positive": [], "negative": []}
        for index, row in enumerate(rows):
            label = row.get("organ_abnormal_label")
            if label is None:
                continue
            try:
                bucket = "positive" if float(label) > 0.5 else "negative"
            except (TypeError, ValueError):
                continue
            abnormal_buckets[bucket].append(index)
        output["by_organ_abnormal_label"] = {
            name: _mean_sentence_scores(sentence_scores, indices) | {"count": len(indices)}
            for name, indices in abnormal_buckets.items()
            if indices
        }
        output["by_organ_abnormal_label_aggregation"] = "mean_sentence_scores"

        lesion_buckets: dict[str, list[int]] = {"positive": [], "negative": []}
        for index, row in enumerate(rows):
            label = row.get("lesion_label")
            if label is None:
                continue
            try:
                bucket = "positive" if float(label) > 0.5 else "negative"
            except (TypeError, ValueError):
                continue
            lesion_buckets[bucket].append(index)
        output["by_lesion_label"] = {
            name: _mean_sentence_scores(sentence_scores, indices) | {"count": len(indices)}
            for name, indices in lesion_buckets.items()
            if indices
        }
        output["by_lesion_label_aggregation"] = "mean_sentence_scores"
    return output


def _run_coco_metrics(
    rows: list[dict[str, Any]],
    *,
    metrics: list[str],
    tokenize: bool,
    unavailable: dict[str, str],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    if not rows or not metrics:
        return {}, {}
    _ensure_metric_backend_paths()
    gts, res = _build_coco_inputs(rows)
    scores: dict[str, float] = {}
    sentence_scores: dict[str, list[float]] = {}
    try:
        if tokenize:
            from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

            tokenizer = PTBTokenizer(verbose=False)
            gts = tokenizer.tokenize(gts)
            res = tokenizer.tokenize(res)
        else:
            gts, res = _caption_dicts_to_strings(gts, res)
        scorers = _build_coco_scorers(metrics)
    except Exception as exc:  # pragma: no cover - environment dependent
        unavailable["pycocoevalcap"] = _exception_summary(exc)
        return scores, sentence_scores
    for scorer, names in scorers:
        try:
            try:
                corpus_score, sent_score = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                corpus_score, sent_score = scorer.compute_score(gts, res)
        except Exception as exc:
            for name in names:
                unavailable[name] = _exception_summary(exc)
            continue
        if len(names) == 1:
            scores[names[0]] = _to_float(corpus_score)
            sentence_scores[names[0]] = _to_float_list(sent_score)
        else:
            for name, score, per_item in zip(names, corpus_score, sent_score):
                if name not in metrics:
                    continue
                scores[name] = _to_float(score)
                sentence_scores[name] = _to_float_list(per_item)
    return scores, sentence_scores


def _run_green_metric(
    rows: list[dict[str, Any]],
    *,
    unavailable: dict[str, str],
    green_batch_size: int,
    green_max_new_tokens: int,
    green_prompt_max_length: int,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    if not rows:
        return {}, {}
    refs = [str(row["target"]).replace("\n", " ") for row in rows]
    hyps = [str(row["generated"]).replace("\n", " ") for row in rows]
    try:
        evaluator = _get_green_evaluator(
            batch_size=green_batch_size,
            max_new_tokens=green_max_new_tokens,
            prompt_max_length=green_prompt_max_length,
        )
        result = evaluator(refs=refs, hyps=hyps)
        sample_scores = _to_float_list(result["green"])
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        unavailable["GREEN"] = _exception_summary(exc)
        return {}, {}
    if not sample_scores:
        return {}, {}
    return {"GREEN": sum(sample_scores) / len(sample_scores)}, {"GREEN": sample_scores}


def _get_green_evaluator(*, batch_size: int, max_new_tokens: int, prompt_max_length: int) -> Any:
    global _GREEN_EVALUATOR
    global _GREEN_EVALUATOR_SETTINGS
    settings = (int(batch_size), int(max_new_tokens), int(prompt_max_length))
    if _GREEN_EVALUATOR is not None and _GREEN_EVALUATOR_SETTINGS == settings:
        return _GREEN_EVALUATOR
    _ensure_metric_backend_paths()
    try:
        from radeval import RadEval
    except Exception:
        from RadEval import RadEval
    _GREEN_EVALUATOR = RadEval(metrics=["green"], per_sample=True, show_progress=False)
    scorer = getattr(_GREEN_EVALUATOR, "_scorer", None)
    if scorer is not None:
        scorer.batch_size = int(batch_size)
        scorer.max_new_tokens = int(max_new_tokens)
        scorer.prompt_max_length = int(prompt_max_length)
    _GREEN_EVALUATOR_SETTINGS = settings
    return _GREEN_EVALUATOR


def _build_coco_inputs(rows: list[dict[str, Any]]) -> tuple[dict[int, list[dict[str, str]]], dict[int, list[dict[str, str]]]]:
    gts = {}
    res = {}
    for index, row in enumerate(rows):
        gts[index] = [{"caption": str(row["target"]).replace("\n", " ")}]
        res[index] = [{"caption": str(row["generated"]).replace("\n", " ")}]
    return gts, res


def _caption_dicts_to_strings(
    gts: dict[int, list[dict[str, str]]],
    res: dict[int, list[dict[str, str]]],
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    return (
        {key: [item["caption"] for item in value] for key, value in gts.items()},
        {key: [item["caption"] for item in value] for key, value in res.items()},
    )


def _build_coco_scorers(metrics: list[str]) -> list[tuple[Any, list[str]]]:
    scorers: list[tuple[Any, list[str]]] = []
    bleu_names = [name for name in ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4") if name in metrics]
    if bleu_names:
        from pycocoevalcap.bleu.bleu import Bleu

        scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
    if "METEOR" in metrics:
        from pycocoevalcap.meteor.meteor import Meteor

        scorers.append((Meteor(), ["METEOR"]))
    if "ROUGE_L" in metrics:
        from pycocoevalcap.rouge.rouge import Rouge

        scorers.append((Rouge(), ["ROUGE_L"]))
    if "CIDEr" in metrics:
        from pycocoevalcap.cider.cider import Cider

        scorers.append((Cider(), ["CIDEr"]))
    return scorers


def _mean_sentence_scores(sentence_scores: dict[str, list[float]], indices: list[int]) -> dict[str, float]:
    means = {}
    for metric, values in sentence_scores.items():
        selected = [values[index] for index in indices if index < len(values)]
        if selected:
            means[metric] = sum(selected) / len(selected)
    return means


def _build_per_organ_summary(result: dict[str, Any]) -> dict[str, Any]:
    organ_level = result.get("organ_level", {})
    per_organ = organ_level.get("per_organ", {})
    if not per_organ:
        return {"note": "No per-organ metrics available."}
    macro = _macro_mean(per_organ)
    best = _rank_per_organ(per_organ, best=True)
    worst = _rank_per_organ(per_organ, best=False)
    rows = {}
    for metric, mean_score in macro.items():
        reference = METRIC_REFERENCE.get(metric, {"direction": "higher_better", "max": None})
        row: dict[str, Any] = {
            "mean": mean_score,
            "ref": _short_metric_reference(reference),
        }
        if reference["max"]:
            row["bar"] = _ascii_bar(100.0 * mean_score / float(reference["max"]))
        row["best"] = best.get(metric)
        row["worst"] = worst.get(metric)
        rows[metric] = row
    return rows


def _build_score_summary(per_organ_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary = {}
    for metric, row in per_organ_summary.items():
        if not isinstance(row, dict) or "mean" not in row:
            continue
        summary[metric] = {
            "score": row["mean"],
            "ref": row.get("ref", "higher better"),
        }
    return summary


def _short_metric_reference(reference: dict[str, Any]) -> str:
    max_value = reference.get("max")
    if max_value is None:
        return "higher better"
    return f"max {max_value:g}, higher better"


def _ascii_bar(percent: float, *, width: int = 10) -> str:
    percent = max(0.0, min(100.0, percent))
    filled = int(round(width * percent / 100.0))
    filled = max(0, min(width, filled))
    return f"[{'#' * filled}{'-' * (width - filled)}] {percent:.1f}%"


def _macro_mean(per_organ: dict[str, dict[str, Any]]) -> dict[str, float]:
    metric_values: dict[str, list[float]] = defaultdict(list)
    for scores in per_organ.values():
        for metric, value in scores.items():
            if metric != "count" and _is_number(value):
                metric_values[metric].append(float(value))
    return {
        metric: sum(values) / len(values)
        for metric, values in sorted(metric_values.items())
        if values
    }


def _rank_per_organ(per_organ: dict[str, dict[str, Any]], *, best: bool) -> dict[str, dict[str, Any]]:
    metric_values: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for organ, scores in per_organ.items():
        for metric, value in scores.items():
            if metric != "count" and _is_number(value):
                metric_values[metric].append((organ, float(value)))
    ranked = {}
    for metric, values in sorted(metric_values.items()):
        organ, score = sorted(values, key=lambda item: item[1], reverse=best)[0]
        ranked[metric] = {"organ": organ, "score": score}
    return ranked


def _build_study_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str]:
    if any(not _clean_text(row.get("study_id")) for row in rows):
        return [], "Study-level support metrics disabled because at least one usable row is missing 'study_id'."
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["study_id"])].append(row)
    study_rows = []
    for study_id, study_items in sorted(grouped.items()):
        ordered = sorted(study_items, key=_organ_sort_key)
        generated = " ".join(_format_organ_finding(row, "generated") for row in ordered)
        target = " ".join(_format_organ_finding(row, "target") for row in ordered)
        study_rows.append({"study_id": study_id, "generated": generated, "target": target})
    return study_rows, ""


def _format_organ_finding(row: dict[str, Any], field: str) -> str:
    organ = _clean_text(row.get("organ"))
    text = _clean_text(row.get(field))
    return f"{organ}: {text}" if organ else text


def _organ_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    organ = _clean_text(row.get("organ"))
    try:
        index = DEFAULT_ORGAN_ORDER.index(organ)
    except ValueError:
        index = len(DEFAULT_ORGAN_ORDER)
    return index, organ


def _resolve_tokenize(mode: str) -> tuple[bool, str]:
    if mode == "none":
        return False, "pycocoevalcap PTB Java tokenization disabled by --tokenize none."
    java_path = shutil.which("java")
    if java_path and _java_runs(java_path):
        return True, ""
    if mode == "java":
        return False, "Requested Java tokenization, but no runnable 'java' executable was found; falling back to no tokenization."
    return False, "No runnable 'java' executable found; using pycocoevalcap without PTB tokenization and skipping METEOR."


def _ensure_metric_backend_paths() -> None:
    candidates = [
        os.environ.get("ORGAN_SEG_CLIP_PYCOCOEVALCAP_ROOT", "").strip(),
        DEFAULT_LOCAL_PYCOCOEVALCAP_ROOT,
        os.environ.get("ORGAN_SEG_CLIP_RADEVAL_ROOT", "").strip(),
        DEFAULT_LOCAL_RADEVAL_ROOT,
    ]
    for value in candidates:
        if not value:
            continue
        target = Path(value).expanduser().resolve()
        if not target.exists():
            continue
        target_str = str(target)
        if target_str not in sys.path:
            sys.path.insert(0, target_str)


def _normalize_coco_metrics(metrics: list[str]) -> list[str]:
    aliases = {
        "bleu_1": "Bleu_1",
        "bleu1": "Bleu_1",
        "Bleu1": "Bleu_1",
        "bleu_2": "Bleu_2",
        "bleu2": "Bleu_2",
        "Bleu2": "Bleu_2",
        "bleu_3": "Bleu_3",
        "bleu3": "Bleu_3",
        "Bleu3": "Bleu_3",
        "bleu_4": "Bleu_4",
        "bleu4": "Bleu_4",
        "Bleu4": "Bleu_4",
        "meteor": "METEOR",
        "rouge_l": "ROUGE_L",
        "rougeL": "ROUGE_L",
        "cider": "CIDEr",
        "cider_d": "CIDEr",
    }
    normalized = []
    for metric in metrics:
        normalized.append(aliases.get(metric, metric))
    return list(dict.fromkeys(normalized))


def _filter_coco_metrics(metrics: list[str], *, java_available: bool, warnings: list[str]) -> list[str]:
    supported = {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"}
    filtered = []
    skipped_meteor = False
    for metric in metrics:
        if metric not in supported:
            warnings.append(f"Unsupported pycocoevalcap metric {metric!r}; skipping.")
            continue
        if metric == "METEOR" and not java_available:
            skipped_meteor = True
            continue
        filtered.append(metric)
    if skipped_meteor:
        warnings.append("Skipped METEOR because it requires a runnable Java executable.")
    return filtered


def _java_runs(java_path: str) -> bool:
    try:
        subprocess.run([java_path, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:
        return False
    return True


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _append_example(items: list[str], value: str, *, limit: int = 5) -> None:
    if len(items) < limit:
        items.append(value)


def _to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _exception_summary(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


if __name__ == "__main__":
    main()
