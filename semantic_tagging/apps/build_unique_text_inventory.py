#!/usr/bin/env python3
from _bootstrap import bootstrap

bootstrap()

from semantic_tagging.cli_helpers import build_base_parser, load_pipeline_from_args


def main() -> None:
    parser = build_base_parser(__doc__ or "Build unique text inventory artifact.")
    args = parser.parse_args()
    pipeline = load_pipeline_from_args(args)
    rows = pipeline.build_source_rows(force=args.force)
    pipeline.build_unique_text_inventory(rows, force=args.force)


if __name__ == "__main__":
    main()
