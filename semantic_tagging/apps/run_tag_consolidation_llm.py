#!/usr/bin/env python3
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from semantic_tagging.consolidation_artifacts import build_consolidation_artifacts, load_consolidation_config
from semantic_tagging.consolidation_llm import run_llm_consolidation


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-assisted consolidation over observed semantic subtype labels.")
    parser.add_argument("--config", required=True, help="Path to consolidation config YAML.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test item limit.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse existing deterministic consolidation input artifacts.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_consolidation_config(config_path)
    if not args.skip_build:
        build_consolidation_artifacts(config, config_path=config_path)
    summary = run_llm_consolidation(config, limit=args.limit)
    print(f"[consolidation_llm] output_dir={config.output_dir}")
    print(f"[consolidation_llm] status_counts={summary['status_counts']}")
    print(f"[consolidation_llm] mode_counts={summary['mode_counts']}")


if __name__ == "__main__":
    main()
