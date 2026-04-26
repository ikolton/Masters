"""Organ-to-segmentation-label mappings for Merlin masks."""

from __future__ import annotations

from pathlib import Path


DEFAULT_MERLIN_MASK_MAP: dict[str, tuple[int, ...]] = {
    "Spleen": (1,),
    "Kidneys": (2,),
    "Gallbladder": (4,),
    "Liver": (5,),
    "Stomach": (6,),
    "Pancreas": (7,),
    "Adrenal glands": (8,),
    "Small bowel": (18,),
    "Colon": (20,),
    "Urinary bladder": (21,),
    "Prostate": (22,),
}


def load_organ_mask_map(path: str | Path | None = None) -> dict[str, tuple[int, ...]]:
    if path is None or str(path).strip() == "":
        return dict(DEFAULT_MERLIN_MASK_MAP)
    import json

    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): tuple(int(v) for v in value) for key, value in payload.items()}
