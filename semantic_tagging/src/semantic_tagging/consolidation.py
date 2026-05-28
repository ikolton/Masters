import difflib
from collections import defaultdict

from .config import OntologyConfig
from .types import ProposedSubtype


def consolidate_proposals(
    proposals: list[ProposedSubtype],
    *,
    config: OntologyConfig,
) -> tuple[list[ProposedSubtype], dict[str, str]]:
    kept: list[ProposedSubtype] = []
    merged_into: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[ProposedSubtype]] = defaultdict(list)
    for proposal in proposals:
        grouped[(proposal.organ, proposal.family)].append(proposal)
    for _, bucket in grouped.items():
        for proposal in bucket:
            duplicate = _find_duplicate(kept, proposal, threshold=config.duplicate_similarity_threshold)
            if duplicate is None:
                kept.append(proposal)
            else:
                merged_into[proposal.subtype_name] = duplicate.subtype_name
    return kept, merged_into


def _find_duplicate(
    existing: list[ProposedSubtype],
    candidate: ProposedSubtype,
    *,
    threshold: float,
) -> ProposedSubtype | None:
    for item in existing:
        if item.organ != candidate.organ or item.family != candidate.family:
            continue
        similarity = difflib.SequenceMatcher(
            a=_normalize_name(item.subtype_name),
            b=_normalize_name(candidate.subtype_name),
        ).ratio()
        if similarity >= threshold:
            return item
    return None


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
