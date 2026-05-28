"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from .schemas import (
    AggregatorConfig,
    AlignmentProjectionConfig,
    DataConfig,
    DecoderConfig,
    DecoderDataConfig,
    DecoderDiagnosticLossConfig,
    DecoderGenerationConfig,
    DecoderLoraConfig,
    DecoderModelConfig,
    DecoderPathsConfig,
    DecoderSemanticLossConfig,
    DecoderTrainingConfig,
    EncoderConfig,
    GridCombinerConfig,
    LoggingConfig,
    LossConfig,
    ModelConfig,
    OrgansConfig,
    PatchingConfig,
    PathsConfig,
    PreprocessingConfig,
    RuntimeConfig,
    SegMambaConfig,
    TextEncoderConfig,
    TokenizerConfig,
    TrainingConfig,
)
from ..utils.io import load_yaml

T = TypeVar("T")


def _construct_dataclass(cls: type[T], payload: dict[str, Any] | None = None) -> T:
    payload = dict(payload or {})
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass type.")
    valid_names = {field.name for field in fields(cls)}
    unknown = sorted(set(payload) - valid_names)
    if unknown:
        raise ValueError(f"Unknown config fields for {cls.__name__}: {', '.join(unknown)}")
    return cls(**payload)


def encoder_config_from_dict(payload: dict[str, Any], *, config_path: str) -> EncoderConfig:
    path_obj = Path(config_path).expanduser()
    config_dir = str(path_obj.resolve().parent)
    model_payload = dict(payload.get("model", {}))
    model_payload["segmamba"] = _construct_dataclass(SegMambaConfig, model_payload.get("segmamba"))
    model_payload["patching"] = _construct_dataclass(PatchingConfig, model_payload.get("patching"))
    model_payload["tokenizer"] = _construct_dataclass(TokenizerConfig, model_payload.get("tokenizer"))
    model_payload["aggregator"] = _construct_dataclass(AggregatorConfig, model_payload.get("aggregator"))
    model_payload["grid_combiner"] = _construct_dataclass(GridCombinerConfig, model_payload.get("grid_combiner"))
    model_payload["alignment_projection"] = _construct_dataclass(AlignmentProjectionConfig, model_payload.get("alignment_projection"))
    model_payload["organs"] = _construct_dataclass(OrgansConfig, model_payload.get("organs"))
    config = EncoderConfig(
        config_path=str(path_obj),
        config_dir=config_dir,
        paths=_construct_dataclass(PathsConfig, payload.get("paths")),
        data=_construct_dataclass(DataConfig, payload.get("data")),
        preprocessing=_construct_dataclass(PreprocessingConfig, payload.get("preprocessing")),
        text_encoder=_construct_dataclass(TextEncoderConfig, payload.get("text_encoder")),
        model=_construct_dataclass(ModelConfig, model_payload),
        loss=_construct_dataclass(LossConfig, payload.get("loss")),
        training=_construct_dataclass(TrainingConfig, payload.get("training")),
        runtime=_construct_dataclass(RuntimeConfig, payload.get("runtime")),
        logging=_construct_dataclass(LoggingConfig, payload.get("logging")),
    )
    if config.model.organ_query_count != len(config.data.organ_names):
        raise ValueError(
            "model.organ_query_count must match data.organ_names. "
            f"Got {config.model.organ_query_count} and {len(config.data.organ_names)}."
        )
    if config.text_encoder.projection_dim != config.model.tokenizer.model_dim:
        raise ValueError(
            "text_encoder.projection_dim must match model.tokenizer.model_dim for shared contrastive space. "
            f"Got {config.text_encoder.projection_dim} and {config.model.tokenizer.model_dim}."
        )
    return config


def load_encoder_config(path: str | Path) -> EncoderConfig:
    resolved_path = Path(path).expanduser().resolve()
    return encoder_config_from_dict(load_yaml(resolved_path), config_path=str(resolved_path))


def decoder_config_from_dict(payload: dict[str, Any], *, config_path: str) -> DecoderConfig:
    path_obj = Path(config_path).expanduser()
    config_dir = str(path_obj.resolve().parent)
    return DecoderConfig(
        config_path=str(path_obj),
        config_dir=config_dir,
        paths=_construct_dataclass(DecoderPathsConfig, payload.get("paths")),
        data=_construct_dataclass(DecoderDataConfig, payload.get("data")),
        model=_construct_dataclass(DecoderModelConfig, payload.get("model")),
        lora=_construct_dataclass(DecoderLoraConfig, payload.get("lora")),
        diagnostic_loss=_construct_dataclass(DecoderDiagnosticLossConfig, payload.get("diagnostic_loss")),
        semantic_loss=_construct_dataclass(DecoderSemanticLossConfig, payload.get("semantic_loss")),
        training=_construct_dataclass(DecoderTrainingConfig, payload.get("training")),
        generation=_construct_dataclass(DecoderGenerationConfig, payload.get("generation")),
        logging=_construct_dataclass(LoggingConfig, payload.get("logging")),
    )


def load_decoder_config(path: str | Path) -> DecoderConfig:
    resolved_path = Path(path).expanduser().resolve()
    return decoder_config_from_dict(load_yaml(resolved_path), config_path=str(resolved_path))
