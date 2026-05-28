#!/usr/bin/env python3
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from semantic_tagging.consolidation_postprocess import load_postprocess_config, run_postprocess


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Deterministically postprocess a consolidation draft into training artifacts.")
    parser.add_argument("--config", required=True, help="Path to postprocess config YAML.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_postprocess_config(config_path)
    manifest = run_postprocess(config, config_path=config_path)
    summary = manifest["summary"]
    print(f"[postprocess] output_dir={config.output_dir}")
    print(f"[postprocess] map_rows={summary['map_rows']}")
    print(f"[postprocess] target_rows={summary['target_rows']}")
    print(f"[postprocess] subtype_labels={summary['subtype_label_count']}")
    print(f"[postprocess] family_labels={summary['family_label_count']}")
    print(f"[postprocess] review_rows={summary['review_rows']}")


if __name__ == "__main__":
    main()
