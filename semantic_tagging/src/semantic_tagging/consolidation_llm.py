from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from .consolidation_artifacts import ConsolidationConfig
from .table_store import read_jsonl, write_json, write_jsonl


def run_llm_consolidation(config: ConsolidationConfig, *, limit: int | None = None) -> dict[str, Any]:
    llm_config = dict(config.raw.get("consolidation", {}).get("llm", {}))
    allowed_families = _allowed_families(config)
    items_path = config.output_dir / "llm_consolidation_items.jsonl"
    items = read_jsonl(items_path)
    if limit is not None:
        items = items[:limit]

    client = ConsolidationLlmClient(
        base_url=str(llm_config.get("base_url", "http://127.0.0.1:8000/v1")),
        api_key=os.environ.get(str(llm_config.get("api_key_env", "VLLM_API_KEY")), "EMPTY"),
        model_name=str(llm_config.get("model_name", "meta-llama/Llama-3.3-70B-Instruct")),
        temperature=float(llm_config.get("temperature", 0.0)),
        top_p=float(llm_config.get("top_p", 1.0)),
        max_tokens=int(llm_config.get("max_tokens", 384)),
        request_retries=int(llm_config.get("request_retries", 4)),
        retry_backoff_seconds=float(llm_config.get("retry_backoff_seconds", 2.0)),
        allowed_families=allowed_families,
    )
    concurrency = int(llm_config.get("request_concurrency", 16))

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    started = time.time()
    done = 0
    total = len(items)

    print(f"[consolidation_llm] start items={total} concurrency={concurrency} model={client.model_name}", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(client.consolidate_one, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            done += 1
            raw_row = future.result()
            raw_rows.append(raw_row)
            parsed_rows.append(_parse_decision(raw_row, item, allowed_families=allowed_families))
            if done == 1 or done % 25 == 0 or done == total:
                elapsed = max(time.time() - started, 1e-6)
                rate = done / elapsed
                remaining = (total - done) / rate if rate else 0.0
                print(
                    f"[consolidation_llm] done={done}/{total} rate={rate:.2f}/s eta={_format_seconds(remaining)}",
                    flush=True,
                )

    raw_rows.sort(key=lambda row: str(row["request_id"]))
    parsed_rows.sort(key=lambda row: str(row["request_id"]))
    write_jsonl(config.output_dir / "llm_consolidation_raw.jsonl", raw_rows)
    write_jsonl(config.output_dir / "llm_consolidation_decisions.jsonl", parsed_rows)
    training_vocab = _build_training_vocab(parsed_rows)
    write_json(config.output_dir / "training_vocab_draft.json", training_vocab)
    _write_report(config.output_dir / "reports" / "llm_consolidation_report.md", parsed_rows, training_vocab)

    status_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    for row in parsed_rows:
        status_counts[str(row["parse_status"])] = status_counts.get(str(row["parse_status"]), 0) + 1
        mode_counts[str(row.get("decision_mode", "unknown"))] = mode_counts.get(str(row.get("decision_mode", "unknown")), 0) + 1

    summary = {
        "items": total,
        "status_counts": status_counts,
        "mode_counts": mode_counts,
        "artifacts": {
            "raw": "llm_consolidation_raw.jsonl",
            "decisions": "llm_consolidation_decisions.jsonl",
            "training_vocab_draft": "training_vocab_draft.json",
            "report": "reports/llm_consolidation_report.md",
        },
    }
    write_json(config.output_dir / "llm_consolidation_summary.json", summary)
    return summary


class ConsolidationLlmClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        request_retries: int,
        retry_backoff_seconds: float,
        allowed_families: list[str],
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.request_retries = request_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.allowed_families = allowed_families
        self.session = requests.Session()

    def consolidate_one(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(item, allowed_families=self.allowed_families)},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        last_error = None
        for attempt in range(self.request_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                return {
                    "request_id": item["request_id"],
                    "organ": item["organ"],
                    "observed_subtype": item["observed_subtype"],
                    "raw_output": choice["message"]["content"],
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "model_name": self.model_name,
                }
            except Exception as exc:  # noqa: BLE001 - journal transient backend failures.
                last_error = str(exc)
                if attempt < self.request_retries:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
        return {
            "request_id": item["request_id"],
            "organ": item["organ"],
            "observed_subtype": item["observed_subtype"],
            "raw_output": "",
            "finish_reason": "request_error",
            "validation_error": last_error,
            "model_name": self.model_name,
        }


def _parse_decision(raw_row: dict[str, Any], item: dict[str, Any], *, allowed_families: list[str]) -> dict[str, Any]:
    base = {
        "request_id": raw_row["request_id"],
        "organ": raw_row["organ"],
        "observed_subtype": raw_row["observed_subtype"],
        "unique_text_count": item["tag_stats"]["unique_text_count"],
        "frequency_tier": item["tag_stats"]["frequency_tier"],
        "parse_status": "invalid",
        "decision_mode": "exclude",
        "use_for_subtype_loss": False,
        "subtype_mode": "no_subtype",
        "subtype_label": None,
        "subtype_loss_weight": 0.0,
        "use_for_family_loss": False,
        "family_label": None,
        "family_loss_weight": 0.0,
        "exclude_from_loss": True,
        "merge_relation": "not_applicable",
        "rationale": "",
        "needs_human_review": True,
        "validation_error": None,
    }
    try:
        payload = _loads_json_object(str(raw_row.get("raw_output", "")))
        _validate_payload(payload, organ=str(raw_row["organ"]), allowed_families=allowed_families)
    except Exception as exc:  # noqa: BLE001 - keep bad outputs as artifacts.
        base["validation_error"] = str(exc)
        return base

    base.update(
        {
            "parse_status": "valid",
            "decision_mode": _decision_mode(payload),
            "use_for_subtype_loss": bool(payload["use_for_subtype_loss"]),
            "subtype_mode": payload["subtype_mode"],
            "subtype_label": payload.get("subtype_label"),
            "subtype_loss_weight": float(payload["subtype_loss_weight"]),
            "use_for_family_loss": bool(payload["use_for_family_loss"]),
            "family_label": payload.get("family_label"),
            "family_loss_weight": float(payload["family_loss_weight"]),
            "exclude_from_loss": bool(payload["exclude_from_loss"]),
            "merge_relation": payload.get("merge_relation", "not_applicable"),
            "rationale": payload["rationale"],
            "needs_human_review": bool(payload["needs_human_review"]),
        }
    )
    return base


def _validate_payload(payload: dict[str, Any], *, organ: str, allowed_families: list[str]) -> None:
    allowed_subtype_modes = {"direct", "merge_to_subtype", "no_subtype"}
    allowed_relations = {"direct", "synonym", "parent_child", "clinically_related", "unsafe", "not_applicable"}
    required = {
        "use_for_subtype_loss",
        "subtype_mode",
        "subtype_label",
        "subtype_loss_weight",
        "use_for_family_loss",
        "family_label",
        "family_loss_weight",
        "exclude_from_loss",
        "merge_relation",
        "needs_human_review",
        "rationale",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(payload["use_for_subtype_loss"], bool):
        raise ValueError("use_for_subtype_loss must be boolean")
    if not isinstance(payload["use_for_family_loss"], bool):
        raise ValueError("use_for_family_loss must be boolean")
    if not isinstance(payload["exclude_from_loss"], bool):
        raise ValueError("exclude_from_loss must be boolean")
    if payload.get("subtype_mode") not in allowed_subtype_modes:
        raise ValueError(f"subtype_mode must be one of {sorted(allowed_subtype_modes)}")
    if payload.get("merge_relation", "not_applicable") not in allowed_relations:
        raise ValueError(f"merge_relation must be one of {sorted(allowed_relations)}")
    if not isinstance(payload.get("rationale"), str) or not payload["rationale"].strip():
        raise ValueError("rationale must be a non-empty string")
    if not isinstance(payload.get("needs_human_review"), bool):
        raise ValueError("needs_human_review must be boolean")
    subtype_weight = float(payload["subtype_loss_weight"])
    family_weight = float(payload["family_loss_weight"])
    if not (0.0 <= subtype_weight <= 1.0):
        raise ValueError("subtype_loss_weight must be in [0, 1]")
    if not (0.0 <= family_weight <= 1.0):
        raise ValueError("family_loss_weight must be in [0, 1]")

    subtype_label = payload.get("subtype_label")
    family = payload.get("family_label")
    if family is not None and family not in allowed_families:
        raise ValueError(f"family_label must be one of the controlled families, got {family!r}")
    if family == organ:
        raise ValueError("family_label must not be the organ name")
    if subtype_label is not None and subtype_label in allowed_families:
        raise ValueError("subtype_label must not be a controlled family; put coarse labels in family_label")
    if subtype_label == organ:
        raise ValueError("subtype_label must not be the organ name")

    if payload["exclude_from_loss"]:
        if payload["use_for_subtype_loss"] or payload["use_for_family_loss"]:
            raise ValueError("exclude_from_loss cannot be true while using subtype/family loss")
        if subtype_weight != 0.0 or family_weight != 0.0:
            raise ValueError("excluded rows must use zero subtype/family weights")
        return

    if not payload["use_for_subtype_loss"] and not payload["use_for_family_loss"]:
        raise ValueError("non-excluded rows must use subtype loss, family loss, or both")
    if payload["use_for_subtype_loss"]:
        if payload["subtype_mode"] not in {"direct", "merge_to_subtype"}:
            raise ValueError("use_for_subtype_loss requires subtype_mode direct or merge_to_subtype")
        if not subtype_label:
            raise ValueError("use_for_subtype_loss requires subtype_label")
        if subtype_weight <= 0.0:
            raise ValueError("use_for_subtype_loss requires positive subtype_loss_weight")
    else:
        if payload["subtype_mode"] != "no_subtype":
            raise ValueError("without subtype loss, subtype_mode must be no_subtype")
        if subtype_label is not None:
            raise ValueError("without subtype loss, subtype_label must be null")
        if subtype_weight != 0.0:
            raise ValueError("without subtype loss, subtype_loss_weight must be 0")
    if payload["use_for_family_loss"]:
        if family is None:
            raise ValueError("use_for_family_loss requires family_label")
        if not (0.1 <= family_weight <= 1.0):
            raise ValueError("use_for_family_loss requires family_loss_weight in [0.1, 1.0]")
    else:
        if family is not None:
            raise ValueError("without family loss, family_label must be null")
        if family_weight != 0.0:
            raise ValueError("without family loss, family_loss_weight must be 0")
    if payload["subtype_mode"] == "direct" and payload.get("merge_relation") != "direct":
        raise ValueError("subtype_mode=direct requires merge_relation=direct")
    if payload["subtype_mode"] == "merge_to_subtype" and payload.get("merge_relation") not in {"synonym", "parent_child", "clinically_related"}:
        raise ValueError("merge_to_subtype requires synonym, parent_child, or clinically_related merge_relation")
    if payload["subtype_mode"] == "no_subtype" and payload.get("merge_relation") not in {"not_applicable", "unsafe"}:
        raise ValueError("no_subtype requires merge_relation not_applicable or unsafe")
    if payload.get("merge_relation") == "clinically_related" and not payload["needs_human_review"]:
        raise ValueError("clinically_related rows must set needs_human_review=true")


def _loads_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty model output")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    return payload


def _build_training_vocab(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subtype_labels_by_organ: dict[str, dict[str, dict[str, Any]]] = {}
    family_labels_by_organ: dict[str, dict[str, dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for row in rows:
        if row["parse_status"] != "valid":
            review.append(row)
            continue
        if row["exclude_from_loss"]:
            excluded.append(row)
        organ = str(row["organ"])
        if row["use_for_subtype_loss"]:
            _add_vocab_entry(
                subtype_labels_by_organ,
                organ=organ,
                label=str(row["subtype_label"]),
                source=str(row["subtype_mode"]),
                observed_subtype=str(row["observed_subtype"]),
                count=int(row["unique_text_count"]),
                loss_weight=float(row["subtype_loss_weight"]),
                needs_review=bool(row["needs_human_review"]),
            )
        if row["use_for_family_loss"]:
            _add_vocab_entry(
                family_labels_by_organ,
                organ=organ,
                label=str(row["family_label"]),
                source="family",
                observed_subtype=str(row["observed_subtype"]),
                count=int(row["unique_text_count"]),
                loss_weight=float(row["family_loss_weight"]),
                needs_review=bool(row["needs_human_review"]),
            )
        if row["needs_human_review"]:
            review.append(row)

    subtype_organs = _finalize_vocab_entries(subtype_labels_by_organ)
    family_organs = _finalize_vocab_entries(family_labels_by_organ)
    return {
        "subtype_labels_by_organ": subtype_organs,
        "family_labels_by_organ": family_organs,
        "excluded": excluded,
        "needs_review": review,
    }


def _add_vocab_entry(
    labels_by_organ: dict[str, dict[str, dict[str, Any]]],
    *,
    organ: str,
    label: str,
    source: str,
    observed_subtype: str,
    count: int,
    loss_weight: float,
    needs_review: bool,
) -> None:
    labels = labels_by_organ.setdefault(organ, {})
    entry = labels.setdefault(
        label,
        {
            "label": label,
            "organ": organ,
            "mode_sources": CounterLike(),
            "observed_subtypes": [],
            "total_unique_text_count": 0,
            "min_loss_weight": 1.0,
            "needs_human_review": False,
        },
    )
    entry["mode_sources"].add(source)
    entry["observed_subtypes"].append(observed_subtype)
    entry["total_unique_text_count"] += count
    entry["min_loss_weight"] = min(float(entry["min_loss_weight"]), loss_weight)
    entry["needs_human_review"] = bool(entry["needs_human_review"] or needs_review)


def _finalize_vocab_entries(labels_by_organ: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    organs: dict[str, list[dict[str, Any]]] = {}
    for organ, labels in labels_by_organ.items():
        organs[organ] = []
        for entry in labels.values():
            entry["mode_sources"] = entry["mode_sources"].to_dict()
            entry["observed_subtypes"] = sorted(set(entry["observed_subtypes"]))
            organs[organ].append(entry)
        organs[organ].sort(key=lambda item: (-int(item["total_unique_text_count"]), str(item["label"])))
    return organs


class CounterLike:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}

    def add(self, value: str) -> None:
        self.values[value] = self.values.get(value, 0) + 1

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self.values.items()))


def _write_report(path: Path, rows: list[dict[str, Any]], training_vocab: dict[str, Any]) -> None:
    mode_counts: dict[str, int] = {}
    invalid = 0
    for row in rows:
        if row["parse_status"] != "valid":
            invalid += 1
        mode_counts[str(row.get("decision_mode", "unknown"))] = mode_counts.get(str(row.get("decision_mode", "unknown")), 0) + 1
    lines = [
        "# LLM Tag Consolidation Report",
        "",
        f"- decisions: `{len(rows)}`",
        f"- invalid outputs: `{invalid}`",
        "",
        "## Modes",
        "",
    ]
    for mode, count in sorted(mode_counts.items()):
        lines.append(f"- `{mode}`: {count}")
    lines.extend(["", "## Subtype Vocabulary Draft", ""])
    for organ, labels in sorted(training_vocab["subtype_labels_by_organ"].items()):
        lines.append(f"### {organ}")
        lines.append("")
        lines.append(f"- labels: `{len(labels)}`")
        for label in labels[:15]:
            lines.append(
                f"- `{label['label']}`: {label['total_unique_text_count']} unique texts, "
                f"{len(label['observed_subtypes'])} observed subtype(s), min_weight={label['min_loss_weight']:.2f}"
            )
        lines.append("")
    lines.extend(["", "## Family Vocabulary Draft", ""])
    for organ, labels in sorted(training_vocab["family_labels_by_organ"].items()):
        lines.append(f"### {organ}")
        lines.append("")
        lines.append(f"- families: `{len(labels)}`")
        for label in labels[:15]:
            lines.append(
                f"- `{label['label']}`: {label['total_unique_text_count']} unique texts, "
                f"{len(label['observed_subtypes'])} observed subtype(s), min_weight={label['min_loss_weight']:.2f}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _system_prompt() -> str:
    return """You consolidate radiology semantic subtype labels into a training vocabulary.
Return strict JSON only. No prose outside JSON.
You are not re-tagging the report text. You are deciding how an observed subtype label should be used for diagnostic-loss supervision.
Prefer medically useful, reproducible labels. Avoid over-fragmenting rare wording variants.
Use the controlled family enum exactly. Never invent a family. Never use the organ name as a family.
Subtype and family are separate decisions:
- subtype supervision is organ-specific and uses subtype_label.
- family supervision is coarse and uses family_label from the controlled enum.
- A label may use subtype loss only, family loss only, both, or neither.
- Never put a controlled family such as inflammation or obstruction into subtype_label.
- If a finding is real but too broad/rare for subtype loss, set use_for_subtype_loss=false and use_for_family_loss=true.
- If a finding is artifact, adjacent-organ leakage, or not useful for loss, set exclude_from_loss=true.
Merge relation:
- direct: same label retained.
- synonym: wording variant with same clinical meaning.
- parent_child: source is a subtype of target, or target is a safe parent label.
- clinically_related: related but not identical; mark needs_human_review=true.
- unsafe: not safe for subtype merge; use family-only or exclude.
- not_applicable: use when no subtype loss is used.
Output schema:
{
  "use_for_subtype_loss": true,
  "subtype_mode": "direct|merge_to_subtype|no_subtype",
  "subtype_label": "organ-specific subtype label or null",
  "subtype_loss_weight": 0.0,
  "use_for_family_loss": true,
  "family_label": "one controlled family string or null",
  "family_loss_weight": 0.0,
  "exclude_from_loss": false,
  "merge_relation": "direct|synonym|parent_child|clinically_related|unsafe|not_applicable",
  "needs_human_review": false,
  "rationale": "short grounded reason"
}"""


def _user_prompt(item: dict[str, Any], *, allowed_families: list[str]) -> str:
    return json.dumps(
        {
            "task": "consolidate_one_observed_subtype",
            "organ": item["organ"],
            "observed_subtype": item["observed_subtype"],
            "controlled_training_families": allowed_families,
            "stats": item["tag_stats"],
            "candidate_training_labels_by_family": item.get("candidate_training_labels_by_family", item.get("candidate_training_labels_for_organ", [])),
            "decision_guidance": {
                "frequent_clean_labels": "usually direct unless a more canonical synonym already exists among the candidates or examples show leakage or wording duplication",
                "review_labels": "prefer merge_to_subtype over direct when a synonym or wording variant exists among the frequent candidates; only use direct if no equivalent candidate exists",
                "rare_labels": "usually family only unless clinically important and clearly grounded as a subtype with no better candidate to merge into",
                "very_rare_labels": "usually family only or exclude unless unmistakably important",
                "normal_or_absent_postop": "use subtype and family when organ-specific and clean",
                "synonym_merge_priority": "ALWAYS prefer merge_to_subtype over direct when the observed_subtype is a wording variant of a more frequent candidate. Common synonym patterns to merge: singular/plural (metastasis->metastases), organ-prefix variants (liver_X vs hepatic_X), equivalent medical terms (steatosis/fatty_infiltration/fatty_change/fatty_deposition), specificity variants (diffuse_steatosis->steatosis, focal_steatosis->steatosis), spelling variants (dilatation/dilation). Set merge_relation=synonym and subtype_label to the most frequent equivalent candidate.",
                "unsafe_merges": "do not merge opposite or merely associated concepts into subtype labels; use family loss with needs_human_review=true when uncertain",
                "forbidden_family_outputs": [
                    item["organ"],
                    "organ",
                    "finding",
                    "abnormality without a controlled family",
                ],
                "family_loss_weight": "use 0.2-0.4 for weak/rare coarse families and 0.8-1.0 for strong frequent family evidence",
                "subtype_loss_weight": "use 1.0 for direct clean labels, 0.8-1.0 for safe synonym merges, 0.5-0.8 for parent_child subtype merges",
                "clinically_related_merge": "if you use merge_relation=clinically_related, set needs_human_review=true",
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def _allowed_families(config: ConsolidationConfig) -> list[str]:
    configured = config.raw.get("consolidation", {}).get("controlled_families")
    if configured:
        return [str(item) for item in configured]
    return [
        "normal",
        "absent_postop",
        "focal_lesion",
        "mass_or_malignancy",
        "inflammation",
        "wall_thickening",
        "ductal_or_luminal_dilatation",
        "obstruction",
        "stone_or_calcification",
        "cystic_or_fluid_lesion",
        "fluid_or_collection",
        "vascular",
        "gas_or_air",
        "postoperative_or_device",
        "anatomic_variant",
        "size_or_morphology",
        "limited_assessment",
        "ambiguous_or_indeterminate",
        "trauma_or_injury",
        "other_abnormal",
    ]


def _decision_mode(payload: dict[str, Any]) -> str:
    if payload["exclude_from_loss"]:
        return "exclude"
    if payload["use_for_subtype_loss"] and payload["use_for_family_loss"]:
        return "subtype_and_family"
    if payload["use_for_subtype_loss"]:
        return "subtype_only"
    if payload["use_for_family_loss"]:
        return "family_only"
    return "invalid"


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
