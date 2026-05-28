from semantic_tagging.config import OntologyConfig
from semantic_tagging.consolidation import consolidate_proposals
from semantic_tagging.types import ProposedSubtype


def test_consolidation_merges_similar_names_within_same_family() -> None:
    proposals = [
        ProposedSubtype(
            organ="Liver",
            subtype_name="liver_ring_enhancing_lesions",
            family="focal_lesion",
            canonical_label="ring enhancing lesions",
            rationale="a",
            confidence=0.9,
            first_seen_text="a",
            source_model="mock",
            source_backend="mock",
        ),
        ProposedSubtype(
            organ="Liver",
            subtype_name="liver-ring-enhancing-lesions",
            family="focal_lesion",
            canonical_label="ring enhancing lesions",
            rationale="b",
            confidence=0.92,
            first_seen_text="b",
            source_model="mock",
            source_backend="mock",
        ),
    ]
    kept, merged = consolidate_proposals(proposals, config=OntologyConfig())
    assert len(kept) == 1
    assert "liver-ring-enhancing-lesions" in merged
