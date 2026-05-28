#!/usr/bin/env python3
from _bootstrap import bootstrap

bootstrap()

from semantic_tagging.cli_helpers import build_base_parser, load_pipeline_from_args


def main() -> None:
    parser = build_base_parser(__doc__ or "Build source row artifact.")
    args = parser.parse_args()
    pipeline = load_pipeline_from_args(args)
    pipeline.build_source_rows(force=args.force)


if __name__ == "__main__":
    main()
