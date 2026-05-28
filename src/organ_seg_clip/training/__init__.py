"""Training runtime."""

from __future__ import annotations

from typing import Any

__all__ = ["run_decoder_evaluation", "run_decoder_training", "run_encoder_training"]


def run_decoder_evaluation(*args: Any, **kwargs: Any) -> Any:
    from .decoder_engine import run_decoder_evaluation as _run_decoder_evaluation

    return _run_decoder_evaluation(*args, **kwargs)


def run_decoder_training(*args: Any, **kwargs: Any) -> Any:
    from .decoder_engine import run_decoder_training as _run_decoder_training

    return _run_decoder_training(*args, **kwargs)


def run_encoder_training(*args: Any, **kwargs: Any) -> Any:
    from .engine import run_encoder_training as _run_encoder_training

    return _run_encoder_training(*args, **kwargs)
