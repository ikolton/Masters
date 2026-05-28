from pathlib import Path

from semantic_tagging.config import (
    BackendConfig,
    DatasetConfig,
    ExecutionConfig,
    OntologyConfig,
    PathsConfig,
    ProjectConfig,
    PromptConfig,
    ReportingConfig,
    SemanticTaggingConfig,
)
from semantic_tagging.backend import MockBackend
from semantic_tagging.pipeline import SemanticTaggingPipeline
from semantic_tagging.table_store import MemoryTableStore
from semantic_tagging.types import SourceRow


class DummyPipeline(SemanticTaggingPipeline):
    def __init__(self, config, canned):
        super().__init__(config, table_store=MemoryTableStore(), backend=MockBackend(canned_by_text=canned))

    def build_source_rows(self, *, force: bool = False):
        rows = [
            SourceRow("s1", "train", "Pancreas", "The pancreas demonstrates normal attenuation. No pancreatic duct dilatation or focal masses.", "the pancreas demonstrates normal attenuation. no pancreatic duct dilatation or focal masses.", 0, 0.0, True),
            SourceRow("s2", "val", "Pancreas", "A 1.7 x 1.6 cm mass at the pancreatic tail.", "a 1.7 x 1.6 cm mass at the pancreatic tail.", 1, 1.0, True),
            SourceRow("s3", "val", "Pancreas", "New infiltrative duct-centered lesion.", "new infiltrative duct-centered lesion.", 1, 1.0, True),
        ]
        self.summary.source_row_count = len(rows)
        self.table_store.write_records(self.output_dir / "source_rows.parquet", [row.to_dict() for row in rows])
        return rows


def test_pipeline_runs_with_mock_backend_and_online_proposal(tmp_path: Path) -> None:
    subproject_root = Path(__file__).resolve().parents[1]
    config = SemanticTaggingConfig(
        project=ProjectConfig(dataset_id="synthetic", run_id="integration"),
        paths=PathsConfig(
            dataset_root=str(tmp_path / "dataset"),
            lesion_csv=str(tmp_path / "lesions.csv"),
            output_root=str(tmp_path / "outputs"),
            ontology_root=str(subproject_root / "ontology"),
            prompt_root=str(subproject_root / "prompts"),
            schema_root=str(subproject_root / "schemas"),
        ),
        dataset=DatasetConfig(),
        prompt=PromptConfig(
            system_template="system_v1.md",
            user_template="user_v1.md",
            output_schema="output_schema_v1.json",
            fewshot_dir="fewshot",
            max_fewshot_examples=2,
            include_existing_subtypes=True,
        ),
        ontology=OntologyConfig(),
        execution=ExecutionConfig(batch_size=2, retry_attempts=0, resume=False),
        backend=BackendConfig(kind="mock", model_name="mock"),
        reporting=ReportingConfig(write_summary_markdown=False),
    )
    canned = {
        "The pancreas demonstrates normal attenuation. No pancreatic duct dilatation or focal masses.": {
            "organ": "Pancreas",
            "normality": "normal",
            "polarity": "negative",
            "certainty": "definite",
            "primary_subtype": "pancreas_normal",
            "secondary_subtypes": [],
            "modifiers": [],
            "evidence_spans": ["normal attenuation"],
            "confidence": 0.95,
            "decision_status": "accepted",
            "decision_source": "mock",
            "ontology_version": "v1",
            "proposed_new_subtype": None,
            "proposed_new_family": None,
            "validation_flags": []
        },
        "A 1.7 x 1.6 cm mass at the pancreatic tail.": {
            "organ": "Pancreas",
            "normality": "abnormal",
            "polarity": "positive",
            "certainty": "definite",
            "primary_subtype": "pancreas_mass",
            "secondary_subtypes": [],
            "modifiers": ["size_present"],
            "evidence_spans": ["mass"],
            "confidence": 0.96,
            "decision_status": "accepted",
            "decision_source": "mock",
            "ontology_version": "v1",
            "proposed_new_subtype": None,
            "proposed_new_family": None,
            "validation_flags": []
        },
        "New infiltrative duct-centered lesion.": {
            "organ": "Pancreas",
            "normality": "abnormal",
            "polarity": "positive",
            "certainty": "probable",
            "primary_subtype": None,
            "secondary_subtypes": [],
            "modifiers": [],
            "evidence_spans": ["duct-centered lesion"],
            "confidence": 0.82,
            "decision_status": "accepted_provisional",
            "decision_source": "mock",
            "ontology_version": "v1",
            "proposed_new_subtype": {
                "name": "pancreas_duct_centered_lesion",
                "family": "focal_lesion",
                "canonical_label": "duct-centered pancreatic lesion",
                "reason": "No existing subtype captures this lesion wording.",
                "confidence": 0.84
            },
            "proposed_new_family": None,
            "validation_flags": []
        },
    }
    pipeline = DummyPipeline(config, canned)
    summary = pipeline.run_all(force=True)
    assert summary.source_row_count == 3
    assert summary.validated_decision_count == 3
    assert summary.provisional_subtype_count >= 1
    loss_rows = pipeline.table_store.read_records(pipeline.output_dir / "loss_ready_targets.parquet")
    assert len(loss_rows) == 3
