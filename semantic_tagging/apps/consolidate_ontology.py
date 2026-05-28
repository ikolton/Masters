#!/usr/bin/env python3
import json

from _bootstrap import bootstrap

project_root = bootstrap()

from semantic_tagging.cli_helpers import build_base_parser, load_pipeline_from_args
from semantic_tagging.consolidation import consolidate_proposals
from semantic_tagging.table_store import read_json, write_json
from semantic_tagging.types import ProposedSubtype


def main() -> None:
    parser = build_base_parser(__doc__ or "Consolidate provisional ontology proposals.")
    args = parser.parse_args()
    pipeline = load_pipeline_from_args(args)
    path = pipeline.output_dir / "provisional_subtypes.json"
    payload = read_json(path)
    proposals = [
        ProposedSubtype(
            organ=str(item["organ"]),
            subtype_name=str(item["subtype_name"]),
            family=str(item["family"]),
            canonical_label=str(item["canonical_label"]),
            rationale=str(item["rationale"]),
            confidence=float(item["confidence"]),
            first_seen_text=str(item["first_seen_text"]),
            source_model=str(item["source_model"]),
            source_backend=str(item["source_backend"]),
            support_examples=tuple(str(v) for v in item.get("support_examples", [])),
            status=str(item.get("status", "provisional")),
        )
        for item in payload.get("provisional_subtypes", [])
    ]
    kept, merged = consolidate_proposals(proposals, config=pipeline.config.ontology)
    write_json(
        path,
        {
            "provisional_subtypes": [proposal.to_dict() for proposal in kept],
            "merged_into": merged,
        },
    )


if __name__ == "__main__":
    main()
