"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML at {path}.")
    return payload


def dump_json(path: str | Path, payload: Any) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
