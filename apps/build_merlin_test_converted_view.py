#!/usr/bin/env python3
"""Build a non-destructive Merlin test view compatible with MerlinConvertedDataset.

The held-out test images are stored as:

    <test_root>/<study_id>/<study_id>.nii.gz
    <test_root>/<study_id>/<study_id>_seg.nii.gz

The decoder/evaluation pipeline expects a converted layout:

    <output_root>/train/combined.json
    <output_root>/val/combined.json
    <output_root>/val/<study_id>/<study_id>_resampled.nii.gz
    <output_root>/val/<study_id>/<study_id>_seg_resampled.nii.gz

This script creates only metadata files and symlinks. It never modifies the
source test dataset.
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
        help="Root containing direct test study directories.",
    )
    parser.add_argument(
        "--source-combined-json",
        default="/net/storage/pr3/plgrid/plggjmiag/Merlin_converted/train/combined.json",
        help="Converted Merlin combined.json containing reference findings.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/datasets/merlin_test_converted_view",
        help="Destination wrapper dataset root.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing symlinks/files in the wrapper.")
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

    filtered_records: list[dict[str, Any]] = [
        record for record in source_records if isinstance(record, dict) and record.get("study_id") in test_id_set
    ]
    found_ids = {str(record.get("study_id")) for record in filtered_records}
    missing_metadata = sorted(test_id_set - found_ids)

    train_dir = output_root / "train"
    val_dir = output_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    metadata_payload = json.dumps(filtered_records, indent=2, sort_keys=True)
    (train_dir / "combined.json").write_text(metadata_payload, encoding="utf-8")
    (val_dir / "combined.json").write_text(metadata_payload, encoding="utf-8")

    linked = 0
    missing_files: list[str] = []
    for study_id in test_ids:
        source_dir = test_root / study_id
        scan_source = source_dir / f"{study_id}.nii.gz"
        seg_source = source_dir / f"{study_id}_seg.nii.gz"
        if not scan_source.is_file() or not seg_source.is_file():
            missing_files.append(study_id)
            continue

        target_dir = val_dir / study_id
        target_dir.mkdir(parents=True, exist_ok=True)
        # The MerlinConvertedDataset contract expects *_resampled.nii.gz filenames.
        # These symlinks satisfy that contract but the files are NOT pre-resampled —
        # they are raw scanner output at native spacing (typically 0.6–0.9mm in-plane).
        # The preprocessing pipeline (resample_spacing in PreprocessingConfig) performs
        # the actual resampling at feature-cache build time, just as it does for the
        # training data (which Merlin pre-resampled to 1.5mm isotropic).
        _symlink(scan_source, target_dir / f"{study_id}_resampled.nii.gz", force=bool(args.force))
        _symlink(seg_source, target_dir / f"{study_id}_seg_resampled.nii.gz", force=bool(args.force))
        linked += 1

    manifest = {
        "test_root": str(test_root),
        "source_combined_json": str(source_combined_json),
        "output_root": str(output_root),
        "test_study_dirs": len(test_ids),
        "metadata_records": len(filtered_records),
        "linked_val_studies": linked,
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata_examples": missing_metadata[:20],
        "missing_file_count": len(missing_files),
        "missing_file_examples": missing_files[:20],
        "notes": [
            "train/ exists only to satisfy the converted-dataset contract.",
            "test studies are exposed through val/ so existing benchmark commands can use --split val.",
            "all scan and segmentation files are symlinks back to the immutable test root.",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _symlink(source: Path, target: Path, *, force: bool) -> None:
    if target.exists() or target.is_symlink():
        if not force:
            if target.is_symlink() and target.resolve() == source.resolve():
                return
            raise FileExistsError(f"Target exists, pass --force to replace: {target}")
        target.unlink()
    target.symlink_to(source)


if __name__ == "__main__":
    main()
