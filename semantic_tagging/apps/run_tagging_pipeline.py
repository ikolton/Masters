#!/usr/bin/env python3
from _bootstrap import bootstrap

bootstrap()

from semantic_tagging.cli_helpers import build_base_parser, load_pipeline_from_args


def main() -> None:
    parser = build_base_parser(__doc__ or "Run semantic tagging end-to-end pipeline.")
    args = parser.parse_args()
    pipeline = load_pipeline_from_args(args)
    pipeline.run_all(force=args.force)


if __name__ == "__main__":
    main()
