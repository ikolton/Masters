from semantic_tagging.config import OntologyConfig
from semantic_tagging.ontology import OntologyRegistry
from semantic_tagging.paths import subproject_root
from semantic_tagging.schemas import load_json_schema
from semantic_tagging.validation import ValidationError, build_tag_decision


def test_validation_accepts_known_subtype() -> None:
    root = subproject_root()
    registry = OntologyRegistry(ontology_root=root / "ontology", config=OntologyConfig())
    schema = load_json_schema(root / "prompts" / "output_schema_v1.json")
    payload = {
        "organ": "Pancreas",
        "normality": "abnormal",
        "polarity": "positive",
        "certainty": "definite",
        "primary_subtype": "pancreas_mass",
        "secondary_subtypes": [],
        "modifiers": [],
        "evidence_spans": ["mass"],
        "confidence": 0.95,
        "decision_status": "accepted",
        "decision_source": "test",
        "ontology_version": "v1",
        "proposed_new_subtype": None,
        "proposed_new_family": None,
        "validation_flags": [],
    }
    decision, proposal = build_tag_decision(
        payload,
        organ="Pancreas",
        raw_text="mass text",
        normalized_text="mass text",
        ontology=registry,
        output_schema=schema,
        source_model="mock",
        source_backend="mock",
    )
    assert decision.primary_subtype == "pancreas_mass"
    assert proposal is None


def test_validation_rejects_unknown_subtype() -> None:
    root = subproject_root()
    registry = OntologyRegistry(ontology_root=root / "ontology", config=OntologyConfig())
    schema = load_json_schema(root / "prompts" / "output_schema_v1.json")
    payload = {
        "organ": "Pancreas",
        "normality": "abnormal",
        "polarity": "positive",
        "certainty": "definite",
        "primary_subtype": "pancreas_unknown",
        "secondary_subtypes": [],
        "modifiers": [],
        "evidence_spans": [],
        "confidence": 0.5,
        "decision_status": "accepted",
        "decision_source": "test",
        "ontology_version": "v1",
        "proposed_new_subtype": None,
        "proposed_new_family": None,
        "validation_flags": [],
    }
    try:
        build_tag_decision(
            payload,
            organ="Pancreas",
            raw_text="x",
            normalized_text="x",
            ontology=registry,
            output_schema=schema,
            source_model="mock",
            source_backend="mock",
        )
    except ValidationError:
        return
    raise AssertionError("Expected ValidationError for unknown subtype")
