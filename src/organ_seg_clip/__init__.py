"""OrganSegCLIP package."""

from __future__ import annotations

from typing import Any

__all__ = ["load_encoder_config", "run_encoder_training"]


def load_encoder_config(*args: Any, **kwargs: Any) -> Any:
    from .config import load_encoder_config as _load_encoder_config

    return _load_encoder_config(*args, **kwargs)


def run_encoder_training(*args: Any, **kwargs: Any) -> Any:
    from .training import run_encoder_training as _run_encoder_training

    return _run_encoder_training(*args, **kwargs)
