#!/usr/bin/env python3
from _bootstrap import bootstrap

bootstrap()

from semantic_tagging.cli_helpers import build_base_parser, load_pipeline_from_args


def main() -> None:
    parser = build_base_parser(__doc__ or "Materialize loss-ready target artifact.")
    args = parser.parse_args()
    pipeline = load_pipeline_from_args(args)
    rows = pipeline.build_source_rows(force=args.force)
    unique_records = pipeline.build_unique_text_inventory(rows, force=args.force)
    validated = pipeline.run_tagging(unique_records, force=args.force)
    row_tags = pipeline.propagate_row_level_tags(rows, validated, force=args.force)
    pipeline.materialize_loss_targets(row_tags, force=args.force)
    pipeline.finalize_report()


if __name__ == "__main__":
    main()
