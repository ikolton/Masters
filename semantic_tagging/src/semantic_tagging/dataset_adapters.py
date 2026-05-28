import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import DatasetConfig, PathsConfig
from .types import SourceRow, UniqueTextRecord


CSV_TO_ORGAN_NAME: dict[str, str] = {
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
}


DEFAULT_METADATA_FILES: tuple[str, ...] = ("train/combined.json", "val/combined.json")


@dataclass(frozen=True)
class _StudyMetadata:
    study_id: str
    cleaned_report: str
    findings: dict[str, Any]
    labels: dict[str, Any]


@dataclass(frozen=True)
class _LesionRecord:
    organ_labels: dict[str, float]


class MerlinDatasetAdapter:
    def __init__(self, *, paths: PathsConfig, dataset: DatasetConfig) -> None:
        self.dataset_root = Path(paths.dataset_root).expanduser().resolve()
        self.lesion_csv = Path(paths.lesion_csv).expanduser().resolve()
        self.dataset = dataset
        self.layout_root = self._resolve_layout_root(self.dataset_root)
        self.metadata_by_id = self._load_metadata_lookup()
        self.lesion_by_id = self._load_lesion_lookup()

    def iter_source_rows(self) -> list[SourceRow]:
        rows: list[SourceRow] = []
        for split in self.dataset.splits:
            for study_id in self._iter_usable_study_ids(split):
                metadata = self.metadata_by_id.get(study_id)
                if metadata is None:
                    continue
                lesion_record = self.lesion_by_id.get(study_id)
                for organ in self.dataset.organ_names:
                    finding_value = metadata.findings.get(organ)
                    label_value = metadata.labels.get(organ)
                    if organ == "Kidneys" and (isinstance(finding_value, dict) or isinstance(label_value, dict)):
                        continue
                    if not isinstance(finding_value, str):
                        continue
                    raw_text = finding_value.strip()
                    if not raw_text:
                        continue
                    organ_abnormal_label = label_value if isinstance(label_value, int) and label_value in (0, 1) else None
                    lesion_label = 0.0
                    lesion_mask = False
                    if lesion_record is not None and organ in lesion_record.organ_labels:
                        lesion_label = float(lesion_record.organ_labels[organ])
                        lesion_mask = True
                    rows.append(
                        SourceRow(
                            study_id=study_id,
                            split=split,
                            organ=organ,
                            raw_text=raw_text,
                            normalized_text=normalize_text(raw_text),
                            organ_abnormal_label=organ_abnormal_label,
                            lesion_label=lesion_label,
                            lesion_mask=lesion_mask,
                        )
                    )
        return rows

    def build_unique_text_inventory(self, rows: Iterable[SourceRow]) -> list[UniqueTextRecord]:
        buckets: dict[tuple[str, str], list[SourceRow]] = defaultdict(list)
        for row in rows:
            buckets[(row.organ, row.raw_text)].append(row)
        unique_records: list[UniqueTextRecord] = []
        for (organ, raw_text), bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
            split_counts: dict[str, int] = defaultdict(int)
            abnormal_positive_count = 0
            abnormal_negative_count = 0
            lesion_labeled_count = 0
            lesion_positive_count = 0
            for row in bucket:
                split_counts[row.split] += 1
                if row.organ_abnormal_label == 1:
                    abnormal_positive_count += 1
                elif row.organ_abnormal_label == 0:
                    abnormal_negative_count += 1
                if row.lesion_mask:
                    lesion_labeled_count += 1
                    if row.lesion_label > 0.5:
                        lesion_positive_count += 1
            count = len(bucket)
            lesion_positive_rate = float(lesion_positive_count / lesion_labeled_count) if lesion_labeled_count else 0.0
            abnormal_den = abnormal_positive_count + abnormal_negative_count
            abnormal_positive_rate = float(abnormal_positive_count / abnormal_den) if abnormal_den else 0.0
            unique_records.append(
                UniqueTextRecord(
                    organ=organ,
                    raw_text=raw_text,
                    normalized_text=bucket[0].normalized_text,
                    count=count,
                    split_counts=dict(split_counts),
                    abnormal_positive_count=abnormal_positive_count,
                    abnormal_negative_count=abnormal_negative_count,
                    lesion_labeled_count=lesion_labeled_count,
                    lesion_positive_count=lesion_positive_count,
                    lesion_positive_rate=lesion_positive_rate,
                    abnormal_positive_rate=abnormal_positive_rate,
                )
            )
        return unique_records

    @staticmethod
    def _resolve_layout_root(dataset_root: Path) -> Path:
        direct_train = dataset_root / "train"
        direct_val = dataset_root / "val"
        if direct_train.is_dir() and direct_val.is_dir():
            return dataset_root
        nested_root = dataset_root / "dataset_split"
        if (nested_root / "train").is_dir() and (nested_root / "val").is_dir():
            return nested_root
        raise FileNotFoundError(
            f"Could not locate train/val split directories under {dataset_root} or {dataset_root / 'dataset_split'}."
        )

    def _load_metadata_lookup(self) -> dict[str, _StudyMetadata]:
        manifests = [self.layout_root / rel_path for rel_path in DEFAULT_METADATA_FILES]
        metadata_by_id: dict[str, _StudyMetadata] = {}
        for manifest in manifests:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"Expected list payload in {manifest}")
            for raw_record in payload:
                if not isinstance(raw_record, dict):
                    continue
                study_id = str(raw_record.get("study_id") or "").strip()
                if not study_id:
                    continue
                cleaned_report = raw_record.get("cleaned_report")
                findings = raw_record.get("findings")
                labels = raw_record.get("labels")
                if not isinstance(cleaned_report, str):
                    cleaned_report = ""
                if not isinstance(findings, dict):
                    findings = {}
                if not isinstance(labels, dict):
                    labels = {}
                metadata_by_id[study_id] = _StudyMetadata(
                    study_id=study_id,
                    cleaned_report=cleaned_report,
                    findings=findings,
                    labels=labels,
                )
        return metadata_by_id

    def _load_lesion_lookup(self) -> dict[str, _LesionRecord]:
        if not self.lesion_csv.is_file():
            return {}
        organ_names = set(self.dataset.organ_names)
        csv_to_target = {
            csv_name: organ_name
            for csv_name, organ_name in CSV_TO_ORGAN_NAME.items()
            if organ_name in organ_names
        }
        records: dict[str, _LesionRecord] = {}
        with self.lesion_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                study_id = str(row.get("Encrypted Accession Number") or "").strip()
                if not study_id:
                    continue
                organ_labels: dict[str, float] = {}
                for csv_name, organ_name in csv_to_target.items():
                    count = _parse_float(row.get(f"number of {csv_name} lesion instances"))
                    if count is not None:
                        organ_labels[organ_name] = float(count > 0.0)
                records[study_id] = _LesionRecord(organ_labels=organ_labels)
        return records

    def _iter_usable_study_ids(self, split: str) -> list[str]:
        split_dir = self.layout_root / split
        scan_suffix = "_resampled.nii.gz"
        seg_suffix = "_seg_resampled.nii.gz"
        ids: list[str] = []
        with os.scandir(split_dir) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                study_id = entry.name
                if study_id not in self.metadata_by_id:
                    continue
                if self.dataset.verify_files:
                    try:
                        with os.scandir(entry.path) as fit:
                            file_names = {f.name for f in fit if f.is_file()}
                    except FileNotFoundError:
                        continue
                    if f"{study_id}{scan_suffix}" not in file_names:
                        continue
                    if f"{study_id}{seg_suffix}" not in file_names:
                        continue
                ids.append(study_id)
        return ids


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
