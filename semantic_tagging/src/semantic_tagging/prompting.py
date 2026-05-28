import json
from pathlib import Path
from typing import Any

from .config import PromptConfig
from .ontology import OntologyRegistry
from .paths import subproject_root
from .types import PromptExample, PromptRequest, UniqueTextRecord


class PromptCompiler:
    def __init__(self, *, prompt_root: Path, config: PromptConfig, ontology: OntologyRegistry) -> None:
        self.prompt_root = prompt_root
        self.config = config
        self.ontology = ontology
        self.system_template = (prompt_root / config.system_template).read_text(encoding="utf-8")
        self.user_template = (prompt_root / config.user_template).read_text(encoding="utf-8")
        self.output_schema_text = (prompt_root / config.output_schema).read_text(encoding="utf-8")

    def compile_request(self, record: UniqueTextRecord, *, request_id: str) -> PromptRequest:
        organ_spec = self.ontology.get_organ(record.organ)
        examples = self._load_examples(record.organ)
        subtype_lines = []
        if self.config.include_existing_subtypes:
            for subtype_name in self.ontology.list_allowed_subtypes(record.organ):
                meta = self.ontology.get_subtype_meta(record.organ, subtype_name)
                if meta is None:
                    continue
                subtype_lines.append(
                    f"- {subtype_name}: family={meta['family']}, canonical_label={meta['canonical_label']}, maturity={meta['maturity_tier']}"
                )
        prompt_text = self.user_template.format(
            system_instructions=self.system_template.strip(),
            organ=record.organ,
            raw_text=record.raw_text,
            normalized_text=record.normalized_text,
            count=record.count,
            lesion_positive_rate=f"{record.lesion_positive_rate:.4f}",
            abnormal_positive_rate=f"{record.abnormal_positive_rate:.4f}",
            organ_maturity=organ_spec.maturity_tier,
            allow_online_expansion=str(organ_spec.allow_online_expansion).lower(),
            output_schema=self.output_schema_text.strip(),
            existing_subtypes="\n".join(subtype_lines) if subtype_lines else "- none",
            fewshot_examples=self._render_examples(examples),
        )
        return PromptRequest(
            request_id=request_id,
            organ=record.organ,
            raw_text=record.raw_text,
            normalized_text=record.normalized_text,
            prompt_text=prompt_text,
        )

    def compile_repair_prompt(self, request: PromptRequest, *, raw_output: str, validation_error: str) -> PromptRequest:
        repair_text = (
            request.prompt_text
            + "\n\nThe previous answer was invalid JSON or violated the schema.\n"
            + f"Validation error: {validation_error}\n"
            + "Return exactly one corrected JSON object that matches the schema and allowed enum values.\n"
            + "Do not include markdown fences, commentary, or any text outside the JSON object.\n"
            + "Prefer existing listed subtypes over inventing synonyms.\n"
            + "If you propose a new subtype, keep proposed_new_subtype.family within the allowed family list and put any nicer unsupported family label into proposed_new_family.\n"
            + f"Previous invalid output:\n{raw_output}\n"
        )
        return PromptRequest(
            request_id=request.request_id + "::repair",
            organ=request.organ,
            raw_text=request.raw_text,
            normalized_text=request.normalized_text,
            prompt_text=repair_text,
        )

    def _load_examples(self, organ: str) -> list[PromptExample]:
        filename = organ.lower().replace(" ", "_") + ".jsonl"
        path = self.prompt_root / self.config.fewshot_dir / filename
        if not path.is_file():
            path = self.prompt_root / self.config.fewshot_dir / "_generic.jsonl"
        examples: list[PromptExample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                output_json = dict(payload["output_json"])
                output_json.setdefault("proposed_new_family", None)
                examples.append(
                    PromptExample(
                        organ=str(payload["organ"]),
                        input_text=str(payload["input_text"]),
                        output_json=output_json,
                        notes=str(payload.get("notes", "")),
                    )
                )
        return examples[: self.config.max_fewshot_examples]

    @staticmethod
    def _render_examples(examples: list[PromptExample]) -> str:
        chunks: list[str] = []
        for index, example in enumerate(examples, start=1):
            chunks.append(
                f"Example {index}\n"
                f"Input text: {example.input_text}\n"
                f"Output JSON: {json.dumps(example.output_json, ensure_ascii=False)}\n"
                f"Notes: {example.notes or 'n/a'}"
            )
        return "\n\n".join(chunks)
