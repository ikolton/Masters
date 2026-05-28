from pathlib import Path

from semantic_tagging.config import OntologyConfig
from semantic_tagging.ontology import OntologyRegistry
from semantic_tagging.paths import subproject_root
from semantic_tagging.types import ProposedSubtype


def test_ontology_loads_and_lists_allowed_subtypes() -> None:
    registry = OntologyRegistry(
        ontology_root=subproject_root() / "ontology",
        config=OntologyConfig(),
    )
    allowed = registry.list_allowed_subtypes("Pancreas")
    assert "pancreas_normal" in allowed
    assert "pancreas_mass" in allowed


def test_provisional_subtype_registration_rejects_duplicates() -> None:
    registry = OntologyRegistry(
        ontology_root=subproject_root() / "ontology",
        config=OntologyConfig(),
    )
    proposal = ProposedSubtype(
        organ="Pancreas",
        subtype_name="pancreas_mass_variant",
        family="focal_lesion",
        canonical_label="pancreatic mass variant",
        rationale="test",
        confidence=0.9,
        first_seen_text="mass text",
        source_model="mock",
        source_backend="mock",
    )
    accepted, _ = registry.maybe_register_provisional(proposal)
    assert accepted is True
    duplicate = ProposedSubtype(
        organ="Pancreas",
        subtype_name="pancreas-mass-variant",
        family="focal_lesion",
        canonical_label="pancreatic mass variant",
        rationale="test",
        confidence=0.95,
        first_seen_text="mass text 2",
        source_model="mock",
        source_backend="mock",
    )
    accepted2, reason2 = registry.maybe_register_provisional(duplicate)
    assert accepted2 is False
    assert reason2.startswith("near_duplicate:")
