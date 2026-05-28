import json
import re
from typing import Any

from .ontology import OntologyRegistry
from .schemas import validate_payload
from .types import ProposedFamily, ProposedSubtype, TagDecision


class ValidationError(ValueError):
    pass


def parse_llm_json(raw_output: str) -> dict[str, Any]:
    text = str(raw_output).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def build_tag_decision(
    payload: dict[str, Any],
    *,
    organ: str,
    raw_text: str,
    normalized_text: str,
    ontology: OntologyRegistry,
    output_schema: dict[str, Any],
    source_model: str,
    source_backend: str,
) -> tuple[TagDecision, ProposedSubtype | None]:
    return build_tag_decision_with_family(
        payload,
        organ=organ,
        raw_text=raw_text,
        normalized_text=normalized_text,
        ontology=ontology,
        output_schema=output_schema,
        source_model=source_model,
        source_backend=source_backend,
    )[:2]


def build_tag_decision_with_family(
    payload: dict[str, Any],
    *,
    organ: str,
    raw_text: str,
    normalized_text: str,
    ontology: OntologyRegistry,
    output_schema: dict[str, Any],
    source_model: str,
    source_backend: str,
) -> tuple[TagDecision, ProposedSubtype | None, ProposedFamily | None]:
    validate_payload(payload, output_schema)
    _validate_organ(payload, organ)
    flags = list(_contradiction_flags(payload))
    primary_subtype = payload.get("primary_subtype")
    secondary_subtypes_raw = [str(value) for value in payload.get("secondary_subtypes", [])]
    proposed_payload = payload.get("proposed_new_subtype")
    proposed_family_payload = payload.get("proposed_new_family")
    proposal = None
    family_proposal = None
    if proposed_family_payload is not None:
        family_proposal = _parse_family_proposal(
            proposed_family_payload,
            organ=organ,
            raw_text=raw_text,
            source_model=source_model,
            source_backend=source_backend,
        )
    if proposed_payload is not None:
        proposal = _parse_proposal(proposed_payload, organ=organ, raw_text=raw_text, source_model=source_model, source_backend=source_backend)
        if not ontology.is_allowed_family(proposal.family):
            if family_proposal is None:
                raise ValidationError(f"Proposed family is not allowed: {proposal.family}")
            flags.append(f"family_remapped:{proposal.family}->other_abnormal")
            proposal = ProposedSubtype(
                organ=proposal.organ,
                subtype_name=proposal.subtype_name,
                family="other_abnormal",
                canonical_label=proposal.canonical_label,
                rationale=proposal.rationale,
                confidence=proposal.confidence,
                first_seen_text=proposal.first_seen_text,
                source_model=proposal.source_model,
                source_backend=proposal.source_backend,
                support_examples=proposal.support_examples,
                status=proposal.status,
            )
    proposed_subtype_name = proposal.subtype_name if proposal is not None else None
    if primary_subtype is not None and not ontology.validate_subtype_for_organ(organ, str(primary_subtype)):
        if proposed_subtype_name is not None and str(primary_subtype) == proposed_subtype_name:
            flags.append("primary_uses_proposed_subtype")
        else:
            raise ValidationError(f"Unknown subtype for organ {organ}: {primary_subtype}")
    normalized_secondary_subtypes: list[str] = []
    for subtype in secondary_subtypes_raw:
        if ontology.validate_subtype_for_organ(organ, subtype):
            normalized_secondary_subtypes.append(subtype)
            continue
        if proposed_subtype_name is not None and subtype == proposed_subtype_name:
            flags.append("secondary_uses_proposed_subtype")
            normalized_secondary_subtypes.append(subtype)
            continue
        if subtype.endswith("_negated"):
            base_subtype = subtype[: -len("_negated")]
            if ontology.validate_subtype_for_organ(organ, base_subtype):
                flags.append(f"dropped_negated_secondary:{subtype}")
                continue
        raise ValidationError(f"Unknown secondary subtype for organ {organ}: {subtype}")
    secondary_subtypes = tuple(normalized_secondary_subtypes)
    decision = TagDecision(
        organ=organ,
        raw_text=raw_text,
        normalized_text=normalized_text,
        normality=str(payload["normality"]),
        polarity=str(payload["polarity"]),
        certainty=str(payload["certainty"]),
        primary_subtype=str(primary_subtype) if primary_subtype is not None else None,
        secondary_subtypes=secondary_subtypes,
        modifiers=tuple(str(value) for value in payload.get("modifiers", [])),
        evidence_spans=tuple(str(value) for value in payload.get("evidence_spans", [])),
        confidence=float(payload["confidence"]),
        decision_status=str(payload["decision_status"]),
        decision_source=str(payload["decision_source"]),
        ontology_version=str(payload["ontology_version"]),
        proposed_new_subtype=proposed_payload,
        proposed_new_family=proposed_family_payload,
        validation_flags=tuple(flags + [str(value) for value in payload.get("validation_flags", [])]),
        source_model=source_model,
        source_backend=source_backend,
    )
    return decision, proposal, family_proposal


def _validate_organ(payload: dict[str, Any], expected_organ: str) -> None:
    organ = str(payload["organ"])
    if organ != expected_organ:
        raise ValidationError(f"Payload organ {organ} does not match expected organ {expected_organ}.")


def _parse_proposal(
    payload: dict[str, Any],
    *,
    organ: str,
    raw_text: str,
    source_model: str,
    source_backend: str,
) -> ProposedSubtype:
    required = ("name", "family", "canonical_label", "reason", "confidence")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValidationError(f"Missing proposed_new_subtype fields: {missing}")
    return ProposedSubtype(
        organ=organ,
        subtype_name=str(payload["name"]),
        family=str(payload["family"]),
        canonical_label=str(payload["canonical_label"]),
        rationale=str(payload["reason"]),
        confidence=float(payload["confidence"]),
        first_seen_text=raw_text,
        source_model=source_model,
        source_backend=source_backend,
        support_examples=(raw_text,),
    )


def _parse_family_proposal(
    payload: dict[str, Any],
    *,
    organ: str,
    raw_text: str,
    source_model: str,
    source_backend: str,
) -> ProposedFamily:
    required = ("name", "reason", "confidence")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValidationError(f"Missing proposed_new_family fields: {missing}")
    return ProposedFamily(
        organ=organ,
        family_name=str(payload["name"]),
        rationale=str(payload["reason"]),
        confidence=float(payload["confidence"]),
        first_seen_text=raw_text,
        source_model=source_model,
        source_backend=source_backend,
        suggested_parent_family=str(payload.get("suggested_parent_family") or "other_abnormal"),
        canonical_label=str(payload["canonical_label"]) if payload.get("canonical_label") is not None else None,
        support_examples=(raw_text,),
    )


def _contradiction_flags(payload: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    normality = str(payload["normality"])
    polarity = str(payload["polarity"])
    primary_subtype = payload.get("primary_subtype")
    secondary_subtypes = list(payload.get("secondary_subtypes", []))
    if normality == "normal" and (primary_subtype is not None or secondary_subtypes):
        flags.append("normal_with_subtypes")
    if normality == "absent_postop" and (primary_subtype is not None or secondary_subtypes) and polarity != "mixed":
        flags.append("absent_postop_with_positive_subtypes")
    if polarity == "negative" and primary_subtype is not None and normality != "mixed":
        flags.append("negative_with_primary_subtype")
    return flags
