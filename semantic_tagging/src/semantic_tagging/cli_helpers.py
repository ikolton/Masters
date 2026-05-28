import argparse
from pathlib import Path

from .config import SemanticTaggingConfig, load_config
from .pipeline import SemanticTaggingPipeline


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to semantic tagging YAML config.")
    parser.add_argument("--force", action="store_true", help="Recompute outputs even if they already exist.")
    return parser


def load_pipeline_from_args(args: argparse.Namespace) -> SemanticTaggingPipeline:
    config = load_config(Path(args.config))
    return SemanticTaggingPipeline(config)
