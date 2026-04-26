"""Model factory."""

from __future__ import annotations

from ..config.schemas import EncoderConfig
from .aggregation.model import OrganSegCLIPModel


def build_model(config: EncoderConfig) -> OrganSegCLIPModel:
    return OrganSegCLIPModel(config)
