from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ORGANS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class ProjectConfig:
    dataset_id: str
    run_id: str


@dataclass(frozen=True)
class PathsConfig:
    dataset_root: str
    lesion_csv: str
    output_root: str
    ontology_root: str
    prompt_root: str
    schema_root: str


@dataclass(frozen=True)
class DatasetConfig:
    splits: tuple[str, ...] = ("train", "val")
    verify_files: bool = True
    organ_names: tuple[str, ...] = DEFAULT_ORGANS


@dataclass(frozen=True)
class PromptConfig:
    system_template: str
    user_template: str
    output_schema: str
    fewshot_dir: str
    max_fewshot_examples: int = 5
    include_existing_subtypes: bool = True


@dataclass(frozen=True)
class OntologyConfig:
    allow_online_expansion: bool = True
    proposal_confidence_threshold: float = 0.7
    duplicate_similarity_threshold: float = 0.88
    max_provisional_examples_per_subtype: int = 8


@dataclass(frozen=True)
class ExecutionConfig:
    batch_size: int = 8
    retry_attempts: int = 1
    worker_index: int = 0
    num_workers: int = 1
    resume: bool = True
    max_records_per_organ: int | None = None


@dataclass(frozen=True)
class BackendConfig:
    kind: str = "mock"
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "VLLM_API_KEY"
    timeout_seconds: int = 120
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 900
    use_response_format_json: bool = False
    use_guided_json: bool = False
    request_concurrency: int = 1
    request_retries: int = 4
    retry_backoff_seconds: float = 2.0


@dataclass(frozen=True)
class ReportingConfig:
    write_summary_markdown: bool = True


@dataclass(frozen=True)
class SemanticTaggingConfig:
    project: ProjectConfig
    paths: PathsConfig
    dataset: DatasetConfig
    prompt: PromptConfig
    ontology: OntologyConfig
    execution: ExecutionConfig
    backend: BackendConfig
    reporting: ReportingConfig

    @property
    def output_dir(self) -> Path:
        return Path(self.paths.output_root).expanduser().resolve() / self.project.dataset_id / self.project.run_id


def load_config(path: str | Path) -> SemanticTaggingConfig:
    payload = yaml.safe_load(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return SemanticTaggingConfig(
        project=ProjectConfig(**payload["project"]),
        paths=PathsConfig(**payload["paths"]),
        dataset=_dataset_config(payload.get("dataset", {})),
        prompt=PromptConfig(**payload["prompt"]),
        ontology=OntologyConfig(**payload.get("ontology", {})),
        execution=ExecutionConfig(**payload.get("execution", {})),
        backend=BackendConfig(**payload.get("backend", {})),
        reporting=ReportingConfig(**payload.get("reporting", {})),
    )


def _dataset_config(payload: dict[str, Any]) -> DatasetConfig:
    splits = tuple(str(value) for value in payload.get("splits", ("train", "val")))
    organ_names = tuple(str(value) for value in payload.get("organ_names", DEFAULT_ORGANS))
    return DatasetConfig(
        splits=splits,
        verify_files=bool(payload.get("verify_files", True)),
        organ_names=organ_names,
    )
