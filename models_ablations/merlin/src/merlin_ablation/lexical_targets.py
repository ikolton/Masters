"""Lesion-derived lexical targets for Merlin ablations."""

from __future__ import annotations

import csv
from pathlib import Path


ORGAN_TO_COUNT_COLUMN = {
    "Adrenal glands": "number of adrenal gland lesion instances",
    "Colon": "number of colon lesion instances",
    "Gallbladder": "number of gallbladder lesion instances",
    "Kidneys": "number of kidney lesion instances",
    "Liver": "number of liver lesion instances",
    "Pancreas": "number of pancreatic lesion instances",
    "Prostate": "number of prostate lesion instances",
    "Small bowel": "number of duodenum lesion instances",
    "Spleen": "number of spleen lesion instances",
    "Stomach": "number of stomach lesion instances",
    "Urinary bladder": "number of bladder lesion instances",
}


class LexicalTargetLookup:
    def __init__(self, path: Path, organ_names: tuple[str, ...]) -> None:
        self.path = path
        self.organ_names = organ_names
        self._values = self._load(path)

    def get(self, study_id: str, organ: str) -> tuple[float, bool]:
        organ_values = self._values.get(str(study_id), {})
        value = organ_values.get(str(organ))
        if value is None:
            return 0.0, False
        return float(value), True

    def _load(self, path: Path) -> dict[str, dict[str, float]]:
        if not path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {path}")
        values: dict[str, dict[str, float]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                study_id = str(row.get("Encrypted Accession Number", "")).strip()
                if not study_id:
                    continue
                organ_values: dict[str, float] = {}
                for organ in self.organ_names:
                    column = ORGAN_TO_COUNT_COLUMN.get(organ)
                    if column is None or column not in row:
                        continue
                    organ_values[organ] = 1.0 if _safe_float(row.get(column)) > 0.0 else 0.0
                values[study_id] = organ_values
        return values


def _safe_float(value: object) -> float:
    try:
        text = str(value or "").strip()
        return 0.0 if not text else float(text)
    except ValueError:
        return 0.0

