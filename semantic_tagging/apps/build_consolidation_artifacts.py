#!/usr/bin/env python3
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from semantic_tagging.consolidation_artifacts import build_consolidation_artifacts, load_consolidation_config


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build versioned tag-consolidation inputs from semantic tagging runs.")
    parser.add_argument("--config", required=True, help="Path to consolidation config YAML.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_consolidation_config(config_path)
    manifest = build_consolidation_artifacts(config, config_path=config_path)
    print(f"[consolidation] output_dir={config.output_dir}")
    print(f"[consolidation] decisions={manifest['decision_count']}")
    print(f"[consolidation] observed_tags={manifest['observed_tag_count']}")
    print(f"[consolidation] llm_items={manifest['llm_item_count']}")


if __name__ == "__main__":
    main()
