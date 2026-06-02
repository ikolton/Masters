import json
from collections import defaultdict
from pathlib import Path
import time
from typing import Any

from .backend import InferenceBackend, build_backend
from .config import SemanticTaggingConfig
from .consolidation import consolidate_proposals
from .dataset_adapters import MerlinDatasetAdapter
from .materialize import materialize_loss_targets
from .ontology import OntologyRegistry
from .paths import ensure_dir
from .prompting import PromptCompiler
from .reporting import write_summary_markdown
from .schemas import load_json_schema
from .table_store import ParquetTableStore, TableStore, append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from .types import ProposedFamily, ProposedSubtype, RowLevelTag, RunSummary, SourceRow, UniqueTextRecord
from .validation import ValidationError, build_tag_decision_with_family, parse_llm_json


def _make_label_derived_normal_decision(record: UniqueTextRecord) -> dict[str, Any]:
    """Deterministic normal decision for organ_abnormal_label=0 records — no LLM call."""
    organ_prefix = record.organ.lower().replace(" ", "_")
    return {
        "organ": record.organ,
        "raw_text": record.raw_text,
        "normalized_text": record.normalized_text,
        "normality": "normal",
        "polarity": "negative",
        "certainty": "certain",
        "primary_subtype": f"{organ_prefix}_normal",
        "secondary_subtypes": [],
        "modifiers": [],
        "evidence_spans": [],
        "confidence": 1.0,
        "decision_status": "accepted",
        "decision_source": "label_derived",
        "ontology_version": None,
        "proposed_new_subtype": None,
        "proposed_new_family": None,
        "validation_flags": ["label_derived_normal"],
        "source_model": "label_derived",
        "source_backend": "label_derived",
    }


class SemanticTaggingPipeline:
    def __init__(
        self,
        config: SemanticTaggingConfig,
        *,
        table_store: TableStore | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        self.config = config
        self.output_dir = ensure_dir(config.output_dir)
        self.table_store = table_store or ParquetTableStore()
        self.backend = backend or build_backend(config.backend)
        self.adapter = MerlinDatasetAdapter(paths=config.paths, dataset=config.dataset)
        self.ontology = OntologyRegistry(
            ontology_root=Path(config.paths.ontology_root).expanduser().resolve(),
            config=config.ontology,
        )
        self.prompt_compiler = PromptCompiler(
            prompt_root=Path(config.paths.prompt_root).expanduser().resolve(),
            config=config.prompt,
            ontology=self.ontology,
        )
        self.output_schema = load_json_schema(Path(config.paths.prompt_root).expanduser().resolve() / config.prompt.output_schema)
        self.summary = RunSummary(dataset_id=config.project.dataset_id, run_id=config.project.run_id)

    def build_source_rows(self, *, force: bool = False) -> list[SourceRow]:
        path = self.output_dir / "source_rows.parquet"
        if self.table_store.exists(path) and self.config.execution.resume and not force:
            records = self.table_store.read_records(path)
            rows = [SourceRow(**record) for record in records]
            self.summary.source_row_count = len(rows)
            self._log(f"reusing source rows: {len(rows)}")
            return rows
        self._log("building source rows...")
        rows = self.adapter.iter_source_rows()
        self.table_store.write_records(path, [row.to_dict() for row in rows])
        self.summary.source_row_count = len(rows)
        self._log(f"built source rows: {len(rows)}")
        return rows

    def build_unique_text_inventory(self, rows: list[SourceRow], *, force: bool = False) -> list[UniqueTextRecord]:
        unique_path = self.output_dir / "unique_texts.parquet"
        stats_path = self.output_dir / "unique_text_stats.parquet"
        if self.table_store.exists(unique_path) and self.config.execution.resume and not force:
            records = self.table_store.read_records(unique_path)
            unique_records = [self._unique_from_record(record) for record in records]
            self.summary.unique_text_count = len(unique_records)
            self._log(f"reusing unique-text inventory: {len(unique_records)}")
            return unique_records
        self._log("building unique-text inventory...")
        unique_records = self.adapter.build_unique_text_inventory(rows)
        unique_payload = [record.to_dict() for record in unique_records]
        stats_payload = [
            {
                "organ": record.organ,
                "raw_text": record.raw_text,
                "count": record.count,
                "split_counts": record.split_counts,
                "abnormal_positive_count": record.abnormal_positive_count,
                "abnormal_negative_count": record.abnormal_negative_count,
                "lesion_labeled_count": record.lesion_labeled_count,
                "lesion_positive_count": record.lesion_positive_count,
                "lesion_positive_rate": record.lesion_positive_rate,
                "abnormal_positive_rate": record.abnormal_positive_rate,
            }
            for record in unique_records
        ]
        self.table_store.write_records(unique_path, unique_payload)
        self.table_store.write_records(stats_path, stats_payload)
        self.summary.unique_text_count = len(unique_records)
        self._log(f"built unique-text inventory: {len(unique_records)}")
        return unique_records

    def run_tagging(self, unique_records: list[UniqueTextRecord], *, force: bool = False) -> list[dict[str, Any]]:
        validated_path = self.output_dir / "validated_decisions.parquet"
        raw_path = self.output_dir / "raw_llm_decisions.jsonl"
        partial_raw_path = self.output_dir / "raw_llm_decisions.partial.jsonl"
        partial_validated_path = self.output_dir / "validated_decisions.partial.jsonl"
        provisional_path = self.output_dir / "provisional_subtypes.json"
        proposed_families_path = self.output_dir / "proposed_families.json"
        crash_report_path = self.output_dir / "crash_report.json"
        ontology_snapshot = self.output_dir / "ontology_snapshot"
        final_ontology_snapshot = self.output_dir / "final_ontology_snapshot"
        if self.table_store.exists(validated_path) and self.config.execution.resume and not force:
            records = self.table_store.read_records(validated_path)
            self.summary.validated_decision_count = len(records)
            if provisional_path.exists():
                proposals = read_json(provisional_path)
                self.summary.provisional_subtype_count = len(proposals.get("provisional_subtypes", []))
            self._log(f"reusing validated decisions: {len(records)}")
            return records

        self.ontology.snapshot(ontology_snapshot)
        grouped: dict[str, list[UniqueTextRecord]] = defaultdict(list)
        for record in unique_records:
            grouped[record.organ].append(record)

        raw_rows: list[dict[str, Any]] = []
        validated_rows: list[dict[str, Any]] = []
        proposals: list[ProposedSubtype] = []
        family_proposals: list[ProposedFamily] = []
        completed_keys: set[tuple[str, str]] = set()
        if self.config.execution.resume and not force and partial_validated_path.exists():
            validated_rows = read_jsonl(partial_validated_path)
            completed_keys = {(str(row["organ"]), str(row["raw_text"])) for row in validated_rows}
            if partial_raw_path.exists():
                raw_rows = read_jsonl(partial_raw_path)
            self._replay_partial_proposals(validated_rows, proposals, family_proposals)
            self._log(
                f"resuming partial tagging progress: validated={len(validated_rows)}, "
                f"raw={len(raw_rows)}, completed_unique_texts={len(completed_keys)}"
            )

        total_records = 0
        for records in grouped.values():
            if completed_keys:
                records = [record for record in records if (record.organ, record.raw_text) not in completed_keys]
            if self.config.execution.max_records_per_organ is not None:
                total_records += min(len(records), int(self.config.execution.max_records_per_organ))
            else:
                total_records += len(records)
        total_organs = len(grouped)
        self._log(
            f"starting tagging: organs={total_organs}, unique_texts={total_records}, "
            f"batch_size={self.config.execution.batch_size}, backend={self.backend.backend_name}, model={self.backend.model_name}"
        )

        processed_records = len(completed_keys)
        run_start = time.time()
        for organ_index, (organ, records) in enumerate(grouped.items(), start=1):
            records = sorted(records, key=lambda item: (-item.count, item.raw_text))
            if completed_keys:
                records = [record for record in records if (record.organ, record.raw_text) not in completed_keys]
            if self.config.execution.max_records_per_organ is not None:
                records = records[: int(self.config.execution.max_records_per_organ)]
            organ_total = len(records)
            if organ_total == 0:
                self._log(f"[{organ_index}/{total_organs}] organ={organ} already complete, skipping")
                continue
            organ_start = time.time()
            self._log(f"[{organ_index}/{total_organs}] organ={organ} start records={organ_total}")
            for batch_index in range(0, len(records), self.config.execution.batch_size):
                batch = records[batch_index : batch_index + self.config.execution.batch_size]
                batch_start = time.time()
                requests_batch = []
                llm_batch: list[UniqueTextRecord] = []
                batch_raw_rows: list[dict[str, Any]] = []
                batch_validated_rows: list[dict[str, Any]] = []

                # Records with organ_abnormal_label=0 are healthy by label — skip LLM,
                # write a deterministic normal decision directly.
                for item_index, record in enumerate(batch):
                    if record.abnormal_positive_count == 0 and record.abnormal_negative_count > 0:
                        decision = _make_label_derived_normal_decision(record)
                        batch_validated_rows.append(decision)
                        validated_rows.append(decision)
                        completed_keys.add((record.organ, record.raw_text))
                    else:
                        request_id = f"{organ}:{batch_index + item_index}"
                        requests_batch.append(self.prompt_compiler.compile_request(record, request_id=request_id))
                        llm_batch.append(record)

                if not requests_batch:
                    append_jsonl(partial_validated_path, batch_validated_rows)
                    processed_records += len(batch)
                    continue

                try:
                    responses = self.backend.generate_batch(requests_batch)
                except Exception as exc:
                    self._write_crash_report(
                        crash_report_path,
                        organ=organ,
                        organ_index=organ_index,
                        total_organs=total_organs,
                        batch_index=(batch_index // self.config.execution.batch_size) + 1,
                        organ_batch_count=((organ_total - 1) // self.config.execution.batch_size) + 1,
                        processed_records=processed_records,
                        total_records=total_records,
                        batch=batch,
                        requests_batch=requests_batch,
                        error=exc,
                    )
                    raise
                for request, response, record in zip(requests_batch, responses, llm_batch):
                    raw_row = {
                        "request_id": response.request_id,
                        "organ": record.organ,
                        "raw_text": record.raw_text,
                        "raw_output": response.raw_output,
                        "model_name": response.model_name,
                        "backend_name": response.backend_name,
                    }
                    raw_rows.append(raw_row)
                    batch_raw_rows.append(raw_row)
                    decision = self._validate_response(request, response, record)
                    batch_validated_rows.append(decision["decision"])
                    validated_rows.append(decision["decision"])
                    proposal = decision["proposal"]
                    family_proposal = decision["family_proposal"]
                    if proposal is not None:
                        accepted, reason = self.ontology.maybe_register_provisional(proposal)
                        if accepted:
                            proposals.append(proposal)
                        validated_rows[-1]["validation_flags"] = list(validated_rows[-1]["validation_flags"]) + [reason]
                        batch_validated_rows[-1]["validation_flags"] = list(batch_validated_rows[-1]["validation_flags"]) + [reason]
                    if family_proposal is not None:
                        accepted, reason = self.ontology.maybe_record_family_proposal(family_proposal)
                        if accepted:
                            family_proposals.append(family_proposal)
                        validated_rows[-1]["validation_flags"] = list(validated_rows[-1]["validation_flags"]) + [f"family_proposal:{reason}"]
                        batch_validated_rows[-1]["validation_flags"] = list(batch_validated_rows[-1]["validation_flags"]) + [f"family_proposal:{reason}"]
                    completed_keys.add((record.organ, record.raw_text))
                append_jsonl(partial_raw_path, batch_raw_rows)
                append_jsonl(partial_validated_path, batch_validated_rows)
                write_json(
                    provisional_path,
                    {
                        "provisional_subtypes": [proposal.to_dict() for proposal in proposals],
                        "merged_into": {},
                    }
                )
                write_json(
                    proposed_families_path,
                    {
                        "proposed_families": [proposal.to_dict() for proposal in family_proposals],
                    },
                )
                processed_records += len(batch)
                batch_elapsed = time.time() - batch_start
                overall_elapsed = time.time() - run_start
                rate = processed_records / overall_elapsed if overall_elapsed > 0 else 0.0
                remaining = max(total_records - processed_records, 0)
                eta_seconds = remaining / rate if rate > 0 else 0.0
                self._log(
                    f"[{organ_index}/{total_organs}] organ={organ} batch={(batch_index // self.config.execution.batch_size) + 1}/"
                    f"{((organ_total - 1) // self.config.execution.batch_size) + 1} "
                    f"done {processed_records}/{total_records} "
                    f"batch_time={batch_elapsed:.1f}s rate={rate:.2f}/s eta={self._format_duration(eta_seconds)}"
                )
            organ_elapsed = time.time() - organ_start
            self._log(f"[{organ_index}/{total_organs}] organ={organ} done records={organ_total} elapsed={self._format_duration(organ_elapsed)}")
        write_jsonl(raw_path, raw_rows)
        self.table_store.write_records(validated_path, validated_rows)
        kept, merged = consolidate_proposals(proposals, config=self.config.ontology)
        write_json(
            provisional_path,
            {
                "provisional_subtypes": [proposal.to_dict() for proposal in kept],
                "merged_into": merged,
            },
        )
        write_json(
            proposed_families_path,
            {
                "proposed_families": [proposal.to_dict() for proposal in family_proposals],
            },
        )
        self.ontology.consolidated_copy().snapshot(final_ontology_snapshot)
        self.summary.validated_decision_count = len(validated_rows)
        self.summary.provisional_subtype_count = len(kept)
        self.summary.organ_counts = dict(sorted(grouped_counts(validated_rows, key="organ").items()))
        self.summary.status_counts = dict(sorted(grouped_counts(validated_rows, key="decision_status").items()))
        self._log(
            f"tagging complete: validated={len(validated_rows)}, provisional={len(kept)}, "
            f"elapsed={self._format_duration(time.time() - run_start)}"
        )
        return validated_rows

    def _write_crash_report(
        self,
        path: Path,
        *,
        organ: str,
        organ_index: int,
        total_organs: int,
        batch_index: int,
        organ_batch_count: int,
        processed_records: int,
        total_records: int,
        batch: list[UniqueTextRecord],
        requests_batch: list[Any],
        error: Exception,
    ) -> None:
        payload = {
            "dataset_id": self.config.project.dataset_id,
            "run_id": self.config.project.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "organ": organ,
            "organ_index": organ_index,
            "total_organs": total_organs,
            "batch_index": batch_index,
            "organ_batch_count": organ_batch_count,
            "processed_records": processed_records,
            "total_records": total_records,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "backend_status": self.backend.debug_status(),
            "batch_records": [
                {
                    "organ": record.organ,
                    "raw_text": record.raw_text,
                    "normalized_text": record.normalized_text,
                    "count": record.count,
                    "lesion_positive_rate": record.lesion_positive_rate,
                    "abnormal_positive_rate": record.abnormal_positive_rate,
                }
                for record in batch
            ],
            "request_ids": [getattr(request, "request_id", None) for request in requests_batch],
            "partial_files": {
                "raw_partial": str(self.output_dir / "raw_llm_decisions.partial.jsonl"),
                "validated_partial": str(self.output_dir / "validated_decisions.partial.jsonl"),
                "provisional_subtypes": str(self.output_dir / "provisional_subtypes.json"),
                "proposed_families": str(self.output_dir / "proposed_families.json"),
            },
        }
        write_json(path, payload)
        self._log(f"crash report written: {path}")

    def _replay_partial_proposals(
        self,
        validated_rows: list[dict[str, Any]],
        proposals: list[ProposedSubtype],
        family_proposals: list[ProposedFamily],
    ) -> None:
        seen_subtypes: set[tuple[str, str]] = set()
        seen_families: set[tuple[str, str]] = set()
        for row in validated_rows:
            proposed_subtype = row.get("proposed_new_subtype")
            if isinstance(proposed_subtype, dict):
                proposal = ProposedSubtype(
                    organ=str(row["organ"]),
                    subtype_name=str(proposed_subtype["name"]),
                    family=str(proposed_subtype["family"]),
                    canonical_label=str(proposed_subtype.get("canonical_label") or proposed_subtype["name"]),
                    rationale=str(proposed_subtype.get("reason") or proposed_subtype.get("rationale") or "resumed_from_partial"),
                    confidence=float(row.get("confidence", 0.0)),
                    first_seen_text=str(row["raw_text"]),
                    source_model=str(row.get("source_model", self.backend.model_name)),
                    source_backend=str(row.get("source_backend", self.backend.backend_name)),
                    support_examples=(str(row["raw_text"]),),
                )
                key = (proposal.organ, proposal.subtype_name)
                accepted, _ = self.ontology.maybe_register_provisional(proposal)
                if accepted and key not in seen_subtypes:
                    proposals.append(proposal)
                    seen_subtypes.add(key)
            proposed_family = row.get("proposed_new_family")
            if isinstance(proposed_family, dict):
                family = ProposedFamily(
                    organ=str(row["organ"]),
                    family_name=str(proposed_family["name"]),
                    rationale=str(proposed_family.get("reason") or proposed_family.get("rationale") or "resumed_from_partial"),
                    confidence=float(row.get("confidence", 0.0)),
                    first_seen_text=str(row["raw_text"]),
                    source_model=str(row.get("source_model", self.backend.model_name)),
                    source_backend=str(row.get("source_backend", self.backend.backend_name)),
                    suggested_parent_family=str(proposed_family.get("suggested_parent_family", "other_abnormal")),
                    canonical_label=str(proposed_family["canonical_label"]) if proposed_family.get("canonical_label") is not None else None,
                    support_examples=(str(row["raw_text"]),),
                )
                key = (family.organ, family.family_name)
                accepted, _ = self.ontology.maybe_record_family_proposal(family)
                if accepted and key not in seen_families:
                    family_proposals.append(family)
                    seen_families.add(key)

    def propagate_row_level_tags(self, rows: list[SourceRow], validated_rows: list[dict[str, Any]], *, force: bool = False) -> list[RowLevelTag]:
        path = self.output_dir / "row_level_tags.parquet"
        if self.table_store.exists(path) and self.config.execution.resume and not force:
            records = self.table_store.read_records(path)
            tags = [self._row_tag_from_record(record) for record in records]
            self.summary.row_level_tag_count = len(tags)
            self._log(f"reusing row-level tags: {len(tags)}")
            return tags
        self._log("propagating validated decisions to row-level tags...")
        mapping = {
            (record["organ"], record["raw_text"]): record
            for record in validated_rows
        }
        tags: list[RowLevelTag] = []
        for row in rows:
            decision = mapping.get((row.organ, row.raw_text))
            if decision is None:
                continue
            tags.append(
                RowLevelTag(
                    study_id=row.study_id,
                    split=row.split,
                    organ=row.organ,
                    raw_text=row.raw_text,
                    normalized_text=row.normalized_text,
                    normality=str(decision["normality"]),
                    polarity=str(decision["polarity"]),
                    certainty=str(decision["certainty"]),
                    primary_subtype=decision.get("primary_subtype"),
                    secondary_subtypes=tuple(str(v) for v in decision.get("secondary_subtypes", [])),
                    modifiers=tuple(str(v) for v in decision.get("modifiers", [])),
                    evidence_spans=tuple(str(v) for v in decision.get("evidence_spans", [])),
                    confidence=float(decision["confidence"]),
                    decision_status=str(decision["decision_status"]),
                    decision_source=str(decision["decision_source"]),
                    ontology_version=str(decision["ontology_version"]),
                    proposed_new_subtype=decision.get("proposed_new_subtype"),
                    proposed_new_family=decision.get("proposed_new_family"),
                    validation_flags=tuple(str(v) for v in decision.get("validation_flags", [])),
                    organ_abnormal_label=row.organ_abnormal_label,
                    lesion_label=row.lesion_label,
                    lesion_mask=row.lesion_mask,
                )
            )
        self.table_store.write_records(path, [tag.to_dict() for tag in tags])
        self.summary.row_level_tag_count = len(tags)
        self._log(f"row-level tags materialized: {len(tags)}")
        return tags

    def materialize_loss_targets(self, row_tags: list[RowLevelTag], *, force: bool = False) -> list[dict[str, Any]]:
        path = self.output_dir / "loss_ready_targets.parquet"
        if self.table_store.exists(path) and self.config.execution.resume and not force:
            records = self.table_store.read_records(path)
            self.summary.loss_target_count = len(records)
            self._log(f"reusing loss-ready targets: {len(records)}")
            return records
        self._log("materializing loss-ready targets...")
        targets = materialize_loss_targets(row_tags)
        payload = [target.to_dict() for target in targets]
        self.table_store.write_records(path, payload)
        self.summary.loss_target_count = len(payload)
        self._log(f"loss-ready targets materialized: {len(payload)}")
        return payload

    def finalize_report(self) -> None:
        if not self.config.reporting.write_summary_markdown:
            return
        write_summary_markdown(self.output_dir / "reports" / "summary.md", self.summary)
        self._log(f"summary written: {self.output_dir / 'reports' / 'summary.md'}")

    def run_all(self, *, force: bool = False) -> RunSummary:
        total_start = time.time()
        self._log(
            f"run start dataset={self.config.project.dataset_id} run_id={self.config.project.run_id} "
            f"output_dir={self.output_dir}"
        )
        rows = self.build_source_rows(force=force)
        unique_records = self.build_unique_text_inventory(rows, force=force)
        validated = self.run_tagging(unique_records, force=force)
        row_tags = self.propagate_row_level_tags(rows, validated, force=force)
        self.materialize_loss_targets(row_tags, force=force)
        self.finalize_report()
        self._log(f"run complete elapsed={self._format_duration(time.time() - total_start)}")
        return self.summary

    def _validate_response(self, request, response, record) -> dict[str, Any]:
        last_error: str | None = None
        current_request = request
        current_response = response
        for attempt in range(self.config.execution.retry_attempts + 1):
            try:
                payload = parse_llm_json(current_response.raw_output)
                decision, proposal, family_proposal = build_tag_decision_with_family(
                    payload,
                    organ=record.organ,
                    raw_text=record.raw_text,
                    normalized_text=record.normalized_text,
                    ontology=self.ontology,
                    output_schema=self.output_schema,
                    source_model=current_response.model_name,
                    source_backend=current_response.backend_name,
                )
                return {"decision": decision.to_dict(), "proposal": proposal, "family_proposal": family_proposal}
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                if attempt >= self.config.execution.retry_attempts:
                    break
                current_request = self.prompt_compiler.compile_repair_prompt(
                    current_request,
                    raw_output=current_response.raw_output,
                    validation_error=str(exc),
                )
                current_response = self.backend.generate_batch([current_request])[0]
        fallback = {
            "organ": record.organ,
            "raw_text": record.raw_text,
            "normalized_text": record.normalized_text,
            "normality": "mixed",
            "polarity": "mixed",
            "certainty": "indeterminate",
            "primary_subtype": None,
            "secondary_subtypes": [],
            "modifiers": [],
            "evidence_spans": [],
            "confidence": 0.0,
            "decision_status": "unresolved",
            "decision_source": "validation_fallback",
            "ontology_version": self.ontology.version,
            "proposed_new_subtype": None,
            "proposed_new_family": None,
            "validation_flags": [f"unresolved:{last_error or 'unknown'}"],
            "source_model": self.backend.model_name,
            "source_backend": self.backend.backend_name,
        }
        return {"decision": fallback, "proposal": None, "family_proposal": None}

    @staticmethod
    def _unique_from_record(record: dict[str, Any]) -> UniqueTextRecord:
        return UniqueTextRecord(
            organ=str(record["organ"]),
            raw_text=str(record["raw_text"]),
            normalized_text=str(record["normalized_text"]),
            count=int(record["count"]),
            split_counts=dict(record.get("split_counts_json") or record.get("split_counts") or {}),
            abnormal_positive_count=int(record["abnormal_positive_count"]),
            abnormal_negative_count=int(record["abnormal_negative_count"]),
            lesion_labeled_count=int(record["lesion_labeled_count"]),
            lesion_positive_count=int(record["lesion_positive_count"]),
            lesion_positive_rate=float(record["lesion_positive_rate"]),
            abnormal_positive_rate=float(record["abnormal_positive_rate"]),
        )

    @staticmethod
    def _row_tag_from_record(record: dict[str, Any]) -> RowLevelTag:
        return RowLevelTag(
            study_id=str(record["study_id"]),
            split=str(record["split"]),
            organ=str(record["organ"]),
            raw_text=str(record["raw_text"]),
            normalized_text=str(record["normalized_text"]),
            normality=str(record["normality"]),
            polarity=str(record["polarity"]),
            certainty=str(record["certainty"]),
            primary_subtype=record.get("primary_subtype"),
            secondary_subtypes=tuple(str(v) for v in record.get("secondary_subtypes", [])),
            modifiers=tuple(str(v) for v in record.get("modifiers", [])),
            evidence_spans=tuple(str(v) for v in record.get("evidence_spans", [])),
            confidence=float(record["confidence"]),
            decision_status=str(record["decision_status"]),
            decision_source=str(record["decision_source"]),
            ontology_version=str(record["ontology_version"]),
            proposed_new_subtype=record.get("proposed_new_subtype"),
            proposed_new_family=record.get("proposed_new_family"),
            validation_flags=tuple(str(v) for v in record.get("validation_flags", [])),
            organ_abnormal_label=record.get("organ_abnormal_label"),
            lesion_label=float(record["lesion_label"]),
            lesion_mask=bool(record["lesion_mask"]),
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(int(seconds), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    @staticmethod
    def _log(message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[semantic_tagging {timestamp}] {message}", flush=True)


def grouped_counts(rows: list[dict[str, Any]], *, key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key))] += 1
    return dict(counts)
