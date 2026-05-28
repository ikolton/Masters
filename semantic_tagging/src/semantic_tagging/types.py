from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceRow:
    study_id: str
    split: str
    organ: str
    raw_text: str
    normalized_text: str
    organ_abnormal_label: int | None
    lesion_label: float
    lesion_mask: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UniqueTextRecord:
    organ: str
    raw_text: str
    normalized_text: str
    count: int
    split_counts: dict[str, int]
    abnormal_positive_count: int
    abnormal_negative_count: int
    lesion_labeled_count: int
    lesion_positive_count: int
    lesion_positive_rate: float
    abnormal_positive_rate: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split_counts_json"] = self.split_counts
        return payload


@dataclass(frozen=True)
class PromptExample:
    organ: str
    input_text: str
    output_json: dict[str, Any]
    notes: str = ""


@dataclass(frozen=True)
class PromptRequest:
    request_id: str
    organ: str
    raw_text: str
    normalized_text: str
    prompt_text: str


@dataclass(frozen=True)
class BackendResponse:
    request_id: str
    raw_output: str
    model_name: str
    backend_name: str
    prompt_text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class ProposedSubtype:
    organ: str
    subtype_name: str
    family: str
    canonical_label: str
    rationale: str
    confidence: float
    first_seen_text: str
    source_model: str
    source_backend: str
    support_examples: tuple[str, ...] = ()
    status: str = "provisional"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProposedFamily:
    organ: str
    family_name: str
    rationale: str
    confidence: float
    first_seen_text: str
    source_model: str
    source_backend: str
    suggested_parent_family: str = "other_abnormal"
    canonical_label: str | None = None
    support_examples: tuple[str, ...] = ()
    status: str = "proposed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TagDecision:
    organ: str
    raw_text: str
    normalized_text: str
    normality: str
    polarity: str
    certainty: str
    primary_subtype: str | None
    secondary_subtypes: tuple[str, ...]
    modifiers: tuple[str, ...]
    evidence_spans: tuple[str, ...]
    confidence: float
    decision_status: str
    decision_source: str
    ontology_version: str
    proposed_new_subtype: dict[str, Any] | None
    proposed_new_family: dict[str, Any] | None
    validation_flags: tuple[str, ...]
    source_model: str
    source_backend: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["secondary_subtypes"] = list(self.secondary_subtypes)
        payload["modifiers"] = list(self.modifiers)
        payload["evidence_spans"] = list(self.evidence_spans)
        payload["validation_flags"] = list(self.validation_flags)
        return payload


@dataclass(frozen=True)
class RowLevelTag:
    study_id: str
    split: str
    organ: str
    raw_text: str
    normalized_text: str
    normality: str
    polarity: str
    certainty: str
    primary_subtype: str | None
    secondary_subtypes: tuple[str, ...]
    modifiers: tuple[str, ...]
    evidence_spans: tuple[str, ...]
    confidence: float
    decision_status: str
    decision_source: str
    ontology_version: str
    proposed_new_subtype: dict[str, Any] | None
    proposed_new_family: dict[str, Any] | None
    validation_flags: tuple[str, ...]
    organ_abnormal_label: int | None
    lesion_label: float
    lesion_mask: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["secondary_subtypes"] = list(self.secondary_subtypes)
        payload["modifiers"] = list(self.modifiers)
        payload["evidence_spans"] = list(self.evidence_spans)
        payload["validation_flags"] = list(self.validation_flags)
        return payload


@dataclass(frozen=True)
class LossReadyTarget:
    study_id: str
    split: str
    organ: str
    raw_text: str
    normality: str
    polarity: str
    certainty: str
    primary_subtype: str | None
    secondary_subtypes: tuple[str, ...]
    confidence_weight: float
    contradiction_flags: tuple[str, ...]
    provenance: str
    lesion_label: float
    lesion_mask: bool
    organ_abnormal_label: int | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["secondary_subtypes"] = list(self.secondary_subtypes)
        payload["contradiction_flags"] = list(self.contradiction_flags)
        return payload


@dataclass
class RunSummary:
    dataset_id: str
    run_id: str
    source_row_count: int = 0
    unique_text_count: int = 0
    validated_decision_count: int = 0
    provisional_subtype_count: int = 0
    row_level_tag_count: int = 0
    loss_target_count: int = 0
    organ_counts: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
