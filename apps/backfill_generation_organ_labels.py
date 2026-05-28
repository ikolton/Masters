#!/usr/bin/env python3
"""Backfill main dataset organ abnormal labels into generation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.config import load_decoder_config
from organ_seg_clip.decoder.data import load_decoder_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, help="Benchmark directory containing sample_manifest.json and runs/*/generations.json.")
    parser.add_argument("--config", required=True, help="Decoder config pointing at the same dataset root/split used by the benchmark.")
    parser.add_argument("--split", default="val", help="Dataset split used by the benchmark.")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    manifest_path = benchmark_dir / "sample_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing benchmark manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = load_decoder_config(str(Path(args.config).expanduser().resolve()))
    label_lookup = _build_label_lookup(config, split=str(args.split), manifest=manifest)

    generation_paths = sorted((benchmark_dir / "runs").glob("*/generations.json"))
    if not generation_paths:
        raise FileNotFoundError(f"No generation files found under {benchmark_dir / 'runs'}")
    summary: dict[str, Any] = {"benchmark_dir": str(benchmark_dir), "files": {}}
    for path in generation_paths:
        summary["files"][str(path)] = _backfill_file(path, label_lookup)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_label_lookup(config: Any, *, split: str, manifest: dict[str, Any]) -> dict[tuple[str, str], int]:
    samples, _ = load_decoder_samples(config, split=split, sample_seed=None)
    wanted = {str(value) for value in manifest.get("study_ids", [])}
    label_lookup: dict[tuple[str, str], int] = {}
    for sample in samples:
        study_id = str(sample.study_id)
        if wanted and study_id not in wanted:
            continue
        for organ_name, label in sample.organ_label_lookup.items():
            if isinstance(label, int) and label in (0, 1):
                label_lookup[(study_id, str(organ_name))] = int(label)
    return label_lookup


def _backfill_file(path: Path, label_lookup: dict[tuple[str, str], int]) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("generations", [])
    if not isinstance(rows, list):
        return {"rows": 0, "matched": 0, "missing": 0, "already_present": 0, "changed": 0}

    matched = 0
    missing = 0
    already_present = 0
    changed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("organ_abnormal_label") is not None:
            already_present += 1
            continue
        key = (str(row.get("study_id", "")), str(row.get("organ", "")))
        label = label_lookup.get(key)
        if label is None:
            missing += 1
            continue
        row["organ_abnormal_label"] = int(label)
        matched += 1
        changed += 1

    if changed:
        payload["organ_abnormal_label_backfill"] = {
            "source": "dataset combined.json labels[organ]",
            "matched_rows": int(matched),
            "missing_rows": int(missing),
            "already_present_rows": int(already_present),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "rows": len(rows),
        "matched": int(matched),
        "missing": int(missing),
        "already_present": int(already_present),
        "changed": int(changed),
    }


if __name__ == "__main__":
    main()
