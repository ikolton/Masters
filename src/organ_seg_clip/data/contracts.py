"""Dataset contracts for the Merlin-converted dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import json
from pathlib import Path
from typing import Any, Iterable

from ..config.schemas import DEFAULT_ORGANS

DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class StudyMetadata:
    study_id: str
    cleaned_report: str
    findings: dict[str, Any]
    labels: dict[str, Any]
    scan_path: Path | None = None
    segmentation_path: Path | None = None


@dataclass(frozen=True)
class WholeStudySample:
    study_id: str
    split: str
    scan_path: Path
    segmentation_path: Path
    report_text: str
    organ_text_lookup: dict[str, str]
    organ_label_lookup: dict[str, int]


@dataclass
class ExclusionLog:
    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def add(self, reason: str, identifier: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        bucket = self.examples.setdefault(reason, [])
        if len(bucket) < 5:
            bucket.append(identifier)

    def extend(self, entries: Iterable[tuple[str, str]]) -> None:
        for reason, identifier in entries:
            self.add(reason, identifier)


class MerlinDatasetError(ValueError):
    pass


class MerlinConvertedDataset:
    def __init__(self, dataset_root: str | Path, *, verify_metadata: bool = True) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.layout_root = self._resolve_layout_root(self.dataset_root)
        self.verify_metadata = bool(verify_metadata)
        self.exclusions = ExclusionLog()
        self._metadata_ids_by_split: dict[str, set[str]] = {}
        self.metadata_by_study_id = self._load_metadata_lookup()
        self._usable_studies_by_split = self._build_usable_studies()

    def iter_samples(self, split: str, organ_names: tuple[str, ...] = DEFAULT_ORGANS) -> list[WholeStudySample]:
        normalized_split = split.lower().strip()
        if normalized_split not in self._usable_studies_by_split:
            raise ValueError(f"Unsupported split: {split}")
        samples: list[WholeStudySample] = []
        for study in self._usable_studies_by_split[normalized_split]:
            organ_text_lookup: dict[str, str] = {}
            organ_label_lookup: dict[str, int] = {}
            for organ_name in organ_names:
                finding_value = study.metadata.findings.get(organ_name)
                label_value = study.metadata.labels.get(organ_name)
                if organ_name == "Kidneys" and isinstance(finding_value, dict):
                    self.exclusions.add("kidneys_side_aware_finding", f"{study.study_id}:{organ_name}")
                    continue
                if organ_name == "Kidneys" and isinstance(label_value, dict):
                    self.exclusions.add("kidneys_side_aware_label", f"{study.study_id}:{organ_name}")
                    continue
                if isinstance(finding_value, str):
                    organ_text_lookup[organ_name] = finding_value
                if isinstance(label_value, int) and label_value in (0, 1):
                    organ_label_lookup[organ_name] = label_value
            samples.append(
                WholeStudySample(
                    study_id=study.study_id,
                    split=study.split,
                    scan_path=study.scan_path,
                    segmentation_path=study.segmentation_path,
                    report_text=study.metadata.cleaned_report,
                    organ_text_lookup=organ_text_lookup,
                    organ_label_lookup=organ_label_lookup,
                )
            )
        return samples

    def inspection_summary(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "layout_root": str(self.layout_root),
            "metadata_verified": self.verify_metadata,
            "usable_studies_per_split": {
                split: len(studies) for split, studies in self._usable_studies_by_split.items()
            },
            "excluded_records": dict(sorted(self.exclusions.counts.items())),
            "excluded_examples": self.exclusions.examples,
        }

    @staticmethod
    def _resolve_layout_root(dataset_root: Path) -> Path:
        if any((dataset_root / split / "combined.json").is_file() for split in DEFAULT_SPLITS):
            return dataset_root

        nested_root = dataset_root / "dataset_split"
        if any((nested_root / split / "combined.json").is_file() for split in DEFAULT_SPLITS):
            return nested_root

        raise MerlinDatasetError(
            "Could not locate dataset split metadata. Expected at least one combined.json under "
            "<dataset_root>/{train,val,test}/ or <dataset_root>/dataset_split/{train,val,test}/. "
            f"Got dataset_root={dataset_root}."
        )

    def _load_metadata_lookup(self) -> dict[str, StudyMetadata]:
        manifests = self._metadata_manifest_paths()
        if not manifests:
            raise MerlinDatasetError(f"No split combined.json files found under {self.layout_root}")
        payloads_by_split = {split: self._read_json_list(path) for split, path in manifests.items()}
        if self.verify_metadata and {"train", "val"}.issubset(payloads_by_split) and payloads_by_split["train"] != payloads_by_split["val"]:
            raise MerlinDatasetError("Expected train/val combined.json files to contain identical records.")

        metadata_by_id: dict[str, StudyMetadata] = {}
        for split, payload in payloads_by_split.items():
            split_ids: set[str] = set()
            for raw_record in payload:
                metadata = self._validate_metadata_record(raw_record, strict=self.verify_metadata)
                split_ids.add(metadata.study_id)
                existing = metadata_by_id.get(metadata.study_id)
                if existing is not None and self.verify_metadata and existing != metadata:
                    raise MerlinDatasetError(f"Conflicting metadata record for study {metadata.study_id}.")
                metadata_by_id[metadata.study_id] = metadata
            self._metadata_ids_by_split[split] = split_ids
        return metadata_by_id

    def _metadata_manifest_paths(self) -> dict[str, Path]:
        return {
            split: self.layout_root / split / "combined.json"
            for split in DEFAULT_SPLITS
            if (self.layout_root / split / "combined.json").is_file()
        }

    def _build_usable_studies(self) -> dict[str, list[_UsableStudy]]:
        split_names = tuple(split for split in DEFAULT_SPLITS if (self.layout_root / split).is_dir())
        usable: dict[str, list[_UsableStudy]] = {split: [] for split in split_names}
        seen_studies: set[str] = set()
        for split in split_names:
            split_dir = self.layout_root / split
            if not split_dir.is_dir():
                raise MerlinDatasetError(f"Missing split directory: {split_dir}")
            split_seen: set[str] = set()
            with os.scandir(split_dir) as case_iter:
                for case_entry in case_iter:
                    if not case_entry.is_dir():
                        continue
                    study_id = case_entry.name
                    if study_id in seen_studies:
                        self.exclusions.add("duplicate_split_membership", study_id)
                        continue
                    seen_studies.add(study_id)

                    case_path = Path(case_entry.path)
                    scan_name = f"{study_id}_resampled.nii.gz"
                    segmentation_name = f"{study_id}_seg_resampled.nii.gz"
                    try:
                        with os.scandir(case_entry.path) as file_iter:
                            file_names = {entry.name for entry in file_iter if entry.is_file()}
                    except FileNotFoundError:
                        self.exclusions.add("missing_case_directory", f"{split}:{study_id}")
                        continue

                    if scan_name not in file_names:
                        self.exclusions.add("missing_scan_file", f"{split}:{study_id}")
                        continue
                    if segmentation_name not in file_names:
                        self.exclusions.add("missing_segmentation_file", f"{split}:{study_id}")
                        continue

                    metadata = self.metadata_by_study_id.get(study_id)
                    if metadata is None:
                        self.exclusions.add("directory_study_missing_metadata", f"{split}:{study_id}")
                        continue
                    usable[split].append(
                        _UsableStudy(
                            study_id=study_id,
                            split=split,
                            scan_path=case_path / scan_name,
                            segmentation_path=case_path / segmentation_name,
                            metadata=metadata,
                        )
                    )
                    split_seen.add(study_id)
            for study_id in sorted(self._metadata_ids_by_split.get(split, set())):
                if study_id in split_seen:
                    continue
                metadata = self.metadata_by_study_id.get(study_id)
                if metadata is None:
                    self.exclusions.add("manifest_study_missing_metadata", f"{split}:{study_id}")
                    continue
                if metadata.scan_path is None or metadata.segmentation_path is None:
                    continue
                if study_id in seen_studies:
                    self.exclusions.add("duplicate_split_membership", study_id)
                    continue
                seen_studies.add(study_id)
                if not metadata.scan_path.is_file():
                    self.exclusions.add("missing_scan_file", f"{split}:{study_id}")
                    continue
                if not metadata.segmentation_path.is_file():
                    self.exclusions.add("missing_segmentation_file", f"{split}:{study_id}")
                    continue
                usable[split].append(
                    _UsableStudy(
                        study_id=study_id,
                        split=split,
                        scan_path=metadata.scan_path,
                        segmentation_path=metadata.segmentation_path,
                        metadata=metadata,
                    )
                )
        if self.exclusions.counts:
            summary = ", ".join(f"{reason}={count}" for reason, count in sorted(self.exclusions.counts.items()))
            print(f"[dataset] excluded studies: {summary}", flush=True)
            for reason, examples in self.exclusions.examples.items():
                print(f"[dataset]   {reason}: {examples}", flush=True)
        return usable

    @staticmethod
    def _read_json_list(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise MerlinDatasetError(f"Expected list payload in {path}")
        return payload

    @staticmethod
    def _validate_metadata_record(raw_record: Any, *, strict: bool) -> StudyMetadata:
        if not isinstance(raw_record, dict):
            raise MerlinDatasetError("Metadata record must be a dictionary.")
        study_id = raw_record.get("study_id")
        cleaned_report = raw_record.get("cleaned_report")
        findings = raw_record.get("findings")
        labels = raw_record.get("labels")
        scan_path = raw_record.get("scan_path")
        segmentation_path = raw_record.get("segmentation_path")
        if not isinstance(study_id, str) or not study_id:
            raise MerlinDatasetError("Invalid or missing study_id.")
        if not strict:
            cleaned_report = cleaned_report if isinstance(cleaned_report, str) else ""
            findings = findings if isinstance(findings, dict) else {}
            labels = labels if isinstance(labels, dict) else {}
        if not isinstance(cleaned_report, str):
            raise MerlinDatasetError(f"Invalid cleaned_report for study {study_id}")
        if not isinstance(findings, dict):
            raise MerlinDatasetError(f"Invalid findings table for study {study_id}")
        if not isinstance(labels, dict):
            raise MerlinDatasetError(f"Invalid labels table for study {study_id}")
        resolved_scan_path = Path(scan_path).expanduser().resolve() if isinstance(scan_path, str) and scan_path.strip() else None
        resolved_segmentation_path = (
            Path(segmentation_path).expanduser().resolve()
            if isinstance(segmentation_path, str) and segmentation_path.strip()
            else None
        )
        return StudyMetadata(
            study_id=study_id,
            cleaned_report=cleaned_report,
            findings=findings,
            labels=labels,
            scan_path=resolved_scan_path,
            segmentation_path=resolved_segmentation_path,
        )


@dataclass(frozen=True)
class _UsableStudy:
    study_id: str
    split: str
    scan_path: Path
    segmentation_path: Path
    metadata: StudyMetadata
