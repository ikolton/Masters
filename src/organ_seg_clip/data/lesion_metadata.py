"""CSV-backed lesion metadata targets for auxiliary supervision."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class LesionTargetRecord:
    global_label: float
    organ_labels: dict[str, float]


class LesionMetadataLookup:
    def __init__(self, path: str | Path | None, *, organ_names: tuple[str, ...]) -> None:
        self.organ_names = tuple(organ_names)
        self.records: dict[str, LesionTargetRecord] = {}
        if path is None or str(path).strip() == "":
            return
        resolved = Path(path).expanduser().resolve()
        self.records = load_lesion_metadata_csv(resolved, organ_names=self.organ_names)

    def get(self, study_id: str) -> LesionTargetRecord | None:
        return self.records.get(str(study_id))

    @property
    def has_records(self) -> bool:
        return bool(self.records)


def load_lesion_metadata_csv(path: str | Path, *, organ_names: tuple[str, ...]) -> dict[str, LesionTargetRecord]:
    target_organs = set(organ_names)
    csv_to_target = {csv_name: organ_name for csv_name, organ_name in CSV_TO_ORGAN_NAME.items() if organ_name in target_organs}
    records: dict[str, LesionTargetRecord] = {}
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
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
            no_lesion = _parse_float(row.get("no lesion"))
            if no_lesion is None:
                global_label = float(any(value > 0.0 for value in organ_labels.values()))
            else:
                global_label = float(no_lesion <= 0.0)
            records[study_id] = LesionTargetRecord(global_label=global_label, organ_labels=organ_labels)
    return records


def _parse_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
