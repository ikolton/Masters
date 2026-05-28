#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from semantic_tagging.table_store import ParquetTableStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "Inspect semantic tagging run outputs.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    store = ParquetTableStore()
    for name in [
        "source_rows.parquet",
        "unique_texts.parquet",
        "validated_decisions.parquet",
        "row_level_tags.parquet",
        "loss_ready_targets.parquet",
    ]:
        path = output_dir / name
        if path.exists():
            try:
                count = len(store.read_records(path))
            except Exception as exc:
                count = f"error: {exc}"
            print(f"{name}: {count}")
        else:
            print(f"{name}: missing")


if __name__ == "__main__":
    main()
