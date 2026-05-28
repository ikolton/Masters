#!/usr/bin/env python3
"""Build a native test manifest dataset without symlinks or fake val splits.

The output layout is intentionally explicit:

    <output_root>/test/combined.json

Each record in combined.json contains absolute `scan_path` and
`segmentation_path` fields pointing at the immutable held-out test files under
`/net/storage/pr3/plgrid/plggjmiag/Merlin_test`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-root",
        default="/net/storage/pr3/plgrid/plggjmiag/Merlin_test",
        help="Root containing held-out test study directories.",
    )
    parser.add_argument(
        "--source-combined-json",
        default="/net/storage/pr3/plgrid/plggjmiag/Merlin_converted/train/combined.json",
        help="Combined metadata containing report references and organ labels.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/datasets/merlin_test_native",
        help="Destination dataset root containing test/combined.json.",
    )
    args = parser.parse_args()

    test_root = Path(args.test_root).expanduser().resolve()
    source_combined_json = Path(args.source_combined_json).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing test root: {test_root}")
    if not source_combined_json.is_file():
        raise FileNotFoundError(f"Missing source combined.json: {source_combined_json}")

    test_ids = sorted(path.name for path in test_root.iterdir() if path.is_dir())
    test_id_set = set(test_ids)
    with source_combined_json.open("r", encoding="utf-8") as handle:
        source_records = json.load(handle)
    if not isinstance(source_records, list):
        raise ValueError(f"Expected list payload in {source_combined_json}")

    source_by_id = {
        str(record.get("study_id")): record
        for record in source_records
        if isinstance(record, dict) and isinstance(record.get("study_id"), str)
    }
    records: list[dict[str, Any]] = []
    missing_metadata: list[str] = []
    missing_files: list[str] = []
    for study_id in test_ids:
        source_record = source_by_id.get(study_id)
        if source_record is None:
            missing_metadata.append(study_id)
            continue
        scan_path = test_root / study_id / f"{study_id}.nii.gz"
        segmentation_path = test_root / study_id / f"{study_id}_seg.nii.gz"
        if not scan_path.is_file() or not segmentation_path.is_file():
            missing_files.append(study_id)
            continue
        record = dict(source_record)
        record["scan_path"] = str(scan_path)
        record["segmentation_path"] = str(segmentation_path)
        records.append(record)

    test_dir = output_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / "combined.json").write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "dataset_kind": "merlin_test_native_absolute_paths",
        "test_root": str(test_root),
        "source_combined_json": str(source_combined_json),
        "output_root": str(output_root),
        "split": "test",
        "test_study_dirs": len(test_ids),
        "metadata_records": len(records),
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_examples": missing_metadata[:20],
        "missing_file_count": len(missing_files),
        "missing_file_examples": missing_files[:20],
        "notes": [
            "No scan or segmentation symlinks are created.",
            "Evaluation should use --split test.",
            "combined.json records carry absolute scan_path and segmentation_path fields.",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
