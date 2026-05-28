#!/usr/bin/env python3
"""Analyze organ finding text distribution restricted to lesion-positive studies."""

import argparse
from collections import Counter
import csv
import json
import os
import re
from pathlib import Path


DEFAULT_ORGANS = (
    "Spleen",
    "Kidneys",
    "Gallbladder",
    "Liver",
    "Stomach",
    "Pancreas",
    "Adrenal glands",
    "Small bowel",
    "Colon",
    "Urinary bladder",
    "Prostate",
)


CSV_TO_ORGAN_NAME = {
    "liver": "Liver",
    "pancreatic": "Pancreas",
    "kidney": "Kidneys",
    "colon": "Colon",
    "spleen": "Spleen",
    "adrenal gland": "Adrenal glands",
    "bladder": "Urinary bladder",
    "gallbladder": "Gallbladder",
    "stomach": "Stomach",
    "prostate": "Prostate",
    "duodenum": "Small bowel",
}


NEGATIVE_PATTERN = re.compile(
    r"\bno\b|without evidence of|without ct evidence of|absence of|"
    r"normal|unremarkable|within normal limits|surgically absent",
    re.IGNORECASE,
)

ABNORMAL_CUE_PATTERN = re.compile(
    r"lesion|lesions|mass\b|masses|metast|cyst|cysts|hypodens|nodule|nodules|"
    r"calcification|calcifications|stone|stones|steatosis|diverticul|"
    r"hydronephrosis|nephrolith|obstruction|dilat|dilation|thickening|"
    r"cholecystitis|hyperplasia|adenoma|enlarged|prostatomegaly",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/net/storage/pr3/plgrid/plggjmiag/Merlin_converted")
    parser.add_argument("--lesion-csv", default="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv")
    parser.add_argument("--output", default="")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--organ", action="append", default=[])
    args = parser.parse_args()

    organ_names = tuple(args.organ) if args.organ else DEFAULT_ORGANS
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    layout_root = resolve_layout_root(dataset_root)
    metadata_by_id = load_metadata_lookup(layout_root)
    positive_lookup = load_lesion_positive_lookup(Path(args.lesion_csv).expanduser().resolve(), organ_names=organ_names)

    result = {
        "dataset_root": str(dataset_root),
        "layout_root": str(layout_root),
        "lesion_csv": str(Path(args.lesion_csv).expanduser().resolve()),
        "organ_positive_text_distribution": {},
    }

    for organ_name in organ_names:
        rows = []
        split_counts = Counter()
        for study in iter_usable_studies(layout_root, metadata_by_id):
            if float(positive_lookup.get(study["study_id"], {}).get(organ_name, 0.0)) <= 0.0:
                continue
            finding_value = study["findings"].get(organ_name)
            if organ_name == "Kidneys" and isinstance(finding_value, dict):
                continue
            if not isinstance(finding_value, str):
                continue
            text = normalize_text(finding_value)
            if not text:
                continue
            rows.append({"study_id": study["study_id"], "split": study["split"], "text": text, "raw_text": finding_value})
            split_counts[study["split"]] += 1

        counter = Counter(row["text"] for row in rows)
        raw_examples = {}
        for row in rows:
            raw_examples.setdefault(row["text"], row["raw_text"])

        positive_candidates = []
        negative_like = []
        for text, count in counter.most_common(max(int(args.top_k) * 10, 100)):
            item = {
                "text": text,
                "count": int(count),
                "fraction": 0.0 if not rows else float(count) / float(len(rows)),
                "raw_example": raw_examples.get(text, text),
            }
            if looks_positive_candidate(text):
                positive_candidates.append(item)
            if NEGATIVE_PATTERN.search(text):
                negative_like.append(item)

        result["organ_positive_text_distribution"][organ_name] = {
            "positive_study_count": len(rows),
            "positive_studies_per_split": dict(split_counts),
            "unique_text_count": len(counter),
            "top_texts": [
                {
                    "text": text,
                    "count": int(count),
                    "fraction": 0.0 if not rows else float(count) / float(len(rows)),
                    "raw_example": raw_examples.get(text, text),
                }
                for text, count in counter.most_common(max(1, int(args.top_k)))
            ],
            "positive_candidate_texts": positive_candidates[: max(1, int(args.top_k))],
            "negative_like_texts": negative_like[: max(1, int(args.top_k))],
        }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def looks_positive_candidate(text: str) -> bool:
    if NEGATIVE_PATTERN.search(text):
        return False
    return bool(ABNORMAL_CUE_PATTERN.search(text))


def resolve_layout_root(dataset_root):
    if (dataset_root / "train").is_dir() and (dataset_root / "val").is_dir():
        return dataset_root
    if (dataset_root / "dataset_split" / "train").is_dir() and (dataset_root / "dataset_split" / "val").is_dir():
        return dataset_root / "dataset_split"
    raise IOError("Could not locate train/val split directories under {}".format(dataset_root))


def load_metadata_lookup(layout_root):
    manifests = [layout_root / "train" / "combined.json", layout_root / "val" / "combined.json"]
    metadata_by_id = {}
    for manifest_path in manifests:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for raw_record in payload:
            if not isinstance(raw_record, dict):
                continue
            study_id = raw_record.get("study_id")
            findings = raw_record.get("findings")
            if not isinstance(study_id, str) or not isinstance(findings, dict):
                continue
            metadata_by_id[study_id] = raw_record
    return metadata_by_id


def iter_usable_studies(layout_root, metadata_by_id):
    seen_studies = set()
    rows = []
    for split in ("train", "val"):
        split_dir = layout_root / split
        for case_entry in os.scandir(str(split_dir)):
            if not case_entry.is_dir():
                continue
            study_id = case_entry.name
            if study_id in seen_studies:
                continue
            seen_studies.add(study_id)
            case_path = Path(case_entry.path)
            scan_name = "{}_resampled.nii.gz".format(study_id)
            seg_name = "{}_seg_resampled.nii.gz".format(study_id)
            file_names = set(entry.name for entry in os.scandir(case_entry.path) if entry.is_file())
            if scan_name not in file_names or seg_name not in file_names:
                continue
            record = metadata_by_id.get(study_id)
            if record is None:
                continue
            rows.append({"study_id": study_id, "split": split, "findings": record.get("findings", {})})
    return rows


def load_lesion_positive_lookup(path, organ_names):
    wanted_csv_names = dict((csv_name, organ_name) for csv_name, organ_name in CSV_TO_ORGAN_NAME.items() if organ_name in organ_names)
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            study_id = str(row.get("Encrypted Accession Number") or "").strip()
            if not study_id:
                continue
            organ_labels = {}
            for csv_name, organ_name in wanted_csv_names.items():
                key = "number of {} lesion instances".format(csv_name)
                value = parse_float(row.get(key))
                if value is not None:
                    organ_labels[organ_name] = float(value > 0.0)
            rows[study_id] = organ_labels
    return rows


def parse_float(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
