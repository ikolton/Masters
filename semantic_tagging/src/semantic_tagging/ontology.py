import copy
import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import OntologyConfig
from .paths import ensure_dir
from .types import ProposedFamily, ProposedSubtype


@dataclass(frozen=True)
class OrganSubtype:
    name: str
    family: str
    canonical_label: str
    positive_examples: tuple[str, ...]
    contrast_examples: tuple[str, ...]
    maturity_tier: str


@dataclass
class OrganOntology:
    organ: str
    maturity_tier: str
    allow_online_expansion: bool
    subtypes: dict[str, OrganSubtype]
    provisional_subtypes: dict[str, ProposedSubtype]
    proposed_families: dict[str, ProposedFamily]


class OntologyRegistry:
    def __init__(self, *, ontology_root: Path, config: OntologyConfig) -> None:
        self.ontology_root = ontology_root
        self.config = config
        self.global_axes = _load_yaml(ontology_root / "global_axes.yaml")
        self.shared_families = _load_yaml(ontology_root / "shared_families.yaml")
        self.version = str(self.global_axes.get("version", "v1"))
        self.organs = self._load_organs()

    def get_organ(self, organ: str) -> OrganOntology:
        if organ not in self.organs:
            raise KeyError(f"Unknown organ ontology: {organ}")
        return self.organs[organ]

    def list_allowed_subtypes(self, organ: str) -> list[str]:
        spec = self.get_organ(organ)
        return sorted(list(spec.subtypes.keys()) + list(spec.provisional_subtypes.keys()))

    def get_subtype_meta(self, organ: str, subtype: str) -> dict[str, Any] | None:
        spec = self.get_organ(organ)
        if subtype in spec.subtypes:
            item = spec.subtypes[subtype]
            return {
                "name": item.name,
                "family": item.family,
                "canonical_label": item.canonical_label,
                "maturity_tier": item.maturity_tier,
            }
        if subtype in spec.provisional_subtypes:
            item = spec.provisional_subtypes[subtype]
            return {
                "name": item.subtype_name,
                "family": item.family,
                "canonical_label": item.canonical_label,
                "maturity_tier": "provisional",
            }
        return None

    def maybe_record_family_proposal(self, proposal: ProposedFamily) -> tuple[bool, str]:
        spec = self.get_organ(proposal.organ)
        if proposal.confidence < self.config.proposal_confidence_threshold:
            return False, "below_confidence_threshold"
        duplicate = self.find_near_duplicate_family(proposal.organ, proposal.family_name)
        if duplicate is not None:
            return False, f"near_duplicate:{duplicate}"
        spec.proposed_families[proposal.family_name] = proposal
        return True, "recorded"

    def is_allowed_family(self, family: str) -> bool:
        return family in set(self.shared_families.get("families", {}).keys())

    def validate_subtype_for_organ(self, organ: str, subtype: str) -> bool:
        spec = self.get_organ(organ)
        return subtype in spec.subtypes or subtype in spec.provisional_subtypes

    def maybe_register_provisional(self, proposal: ProposedSubtype) -> tuple[bool, str]:
        spec = self.get_organ(proposal.organ)
        if not self.config.allow_online_expansion or not spec.allow_online_expansion:
            return False, "online_expansion_disabled"
        if not self.is_allowed_family(proposal.family):
            return False, "invalid_family"
        if proposal.confidence < self.config.proposal_confidence_threshold:
            return False, "below_confidence_threshold"
        duplicate = self.find_near_duplicate_subtype(proposal.organ, proposal.subtype_name)
        if duplicate is not None:
            return False, f"near_duplicate:{duplicate}"
        spec.provisional_subtypes[proposal.subtype_name] = proposal
        return True, "registered"

    def find_near_duplicate_subtype(self, organ: str, subtype_name: str) -> str | None:
        spec = self.get_organ(organ)
        for existing in list(spec.subtypes.keys()) + list(spec.provisional_subtypes.keys()):
            score = difflib.SequenceMatcher(a=_normalize_slug(existing), b=_normalize_slug(subtype_name)).ratio()
            if score >= self.config.duplicate_similarity_threshold:
                return existing
        return None

    def find_near_duplicate_family(self, organ: str, family_name: str) -> str | None:
        spec = self.get_organ(organ)
        existing_names = list(self.shared_families.get("families", {}).keys()) + list(spec.proposed_families.keys())
        for existing in existing_names:
            score = difflib.SequenceMatcher(a=_normalize_slug(existing), b=_normalize_slug(family_name)).ratio()
            if score >= self.config.duplicate_similarity_threshold:
                return existing
        return None

    def snapshot(self, target_dir: Path) -> None:
        ensure_dir(target_dir)
        _write_yaml(target_dir / "global_axes.yaml", self.global_axes)
        _write_yaml(target_dir / "shared_families.yaml", self.shared_families)
        organs_dir = ensure_dir(target_dir / "organs")
        for organ, spec in self.organs.items():
            payload = {
                "organ": organ,
                "maturity_tier": spec.maturity_tier,
                "allow_online_expansion": spec.allow_online_expansion,
                "subtypes": {
                    name: {
                        "family": subtype.family,
                        "canonical_label": subtype.canonical_label,
                        "positive_examples": list(subtype.positive_examples),
                        "contrast_examples": list(subtype.contrast_examples),
                        "maturity_tier": subtype.maturity_tier,
                    }
                    for name, subtype in spec.subtypes.items()
                },
                "provisional_subtypes": {
                    name: proposal.to_dict()
                    for name, proposal in spec.provisional_subtypes.items()
                },
                "proposed_families": {
                    name: proposal.to_dict()
                    for name, proposal in spec.proposed_families.items()
                },
            }
            filename = organ.lower().replace(" ", "_") + ".yaml"
            _write_yaml(organs_dir / filename, payload)
        manifest = {
            "version": self.version,
            "organ_count": len(self.organs),
            "provisional_count": sum(len(spec.provisional_subtypes) for spec in self.organs.values()),
            "proposed_family_count": sum(len(spec.proposed_families) for spec in self.organs.values()),
        }
        (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def consolidated_copy(self) -> "OntologyRegistry":
        copied = copy.deepcopy(self)
        for spec in copied.organs.values():
            merged: dict[str, ProposedSubtype] = {}
            for name, proposal in spec.provisional_subtypes.items():
                duplicate = None
                for existing in merged.keys():
                    score = difflib.SequenceMatcher(a=_normalize_slug(existing), b=_normalize_slug(name)).ratio()
                    if score >= copied.config.duplicate_similarity_threshold:
                        duplicate = existing
                        break
                if duplicate is None:
                    merged[name] = proposal
            spec.provisional_subtypes = merged
            merged_families: dict[str, ProposedFamily] = {}
            for name, proposal in spec.proposed_families.items():
                duplicate = None
                for existing in merged_families.keys():
                    score = difflib.SequenceMatcher(a=_normalize_slug(existing), b=_normalize_slug(name)).ratio()
                    if score >= copied.config.duplicate_similarity_threshold:
                        duplicate = existing
                        break
                if duplicate is None:
                    merged_families[name] = proposal
            spec.proposed_families = merged_families
        return copied

    def _load_organs(self) -> dict[str, OrganOntology]:
        organs_dir = self.ontology_root / "organs"
        payloads: dict[str, OrganOntology] = {}
        for path in sorted(organs_dir.glob("*.yaml")):
            raw = _load_yaml(path)
            organ = str(raw["organ"])
            subtypes = {
                name: OrganSubtype(
                    name=name,
                    family=str(spec["family"]),
                    canonical_label=str(spec["canonical_label"]),
                    positive_examples=tuple(str(v) for v in spec.get("positive_examples", [])),
                    contrast_examples=tuple(str(v) for v in spec.get("contrast_examples", [])),
                    maturity_tier=str(spec.get("maturity_tier", raw.get("maturity_tier", "tier_1"))),
                )
                for name, spec in raw.get("subtypes", {}).items()
            }
            provisional_subtypes = {
                name: ProposedSubtype(
                    organ=organ,
                    subtype_name=str(spec["subtype_name"]),
                    family=str(spec["family"]),
                    canonical_label=str(spec["canonical_label"]),
                    rationale=str(spec["rationale"]),
                    confidence=float(spec["confidence"]),
                    first_seen_text=str(spec["first_seen_text"]),
                    source_model=str(spec["source_model"]),
                    source_backend=str(spec["source_backend"]),
                    support_examples=tuple(str(v) for v in spec.get("support_examples", [])),
                    status=str(spec.get("status", "provisional")),
                )
                for name, spec in raw.get("provisional_subtypes", {}).items()
            }
            proposed_families = {
                name: ProposedFamily(
                    organ=organ,
                    family_name=str(spec["family_name"]),
                    rationale=str(spec["rationale"]),
                    confidence=float(spec["confidence"]),
                    first_seen_text=str(spec["first_seen_text"]),
                    source_model=str(spec["source_model"]),
                    source_backend=str(spec["source_backend"]),
                    suggested_parent_family=str(spec.get("suggested_parent_family", "other_abnormal")),
                    canonical_label=str(spec["canonical_label"]) if spec.get("canonical_label") is not None else None,
                    support_examples=tuple(str(v) for v in spec.get("support_examples", [])),
                    status=str(spec.get("status", "proposed")),
                )
                for name, spec in raw.get("proposed_families", {}).items()
            }
            payloads[organ] = OrganOntology(
                organ=organ,
                maturity_tier=str(raw.get("maturity_tier", "tier_1")),
                allow_online_expansion=bool(raw.get("allow_online_expansion", True)),
                subtypes=subtypes,
                provisional_subtypes=provisional_subtypes,
                proposed_families=proposed_families,
            )
        return payloads


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _normalize_slug(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())
