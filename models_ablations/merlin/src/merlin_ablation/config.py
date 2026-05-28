"""Config loading for Merlin ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ORGANS = (
    "Adrenal glands",
    "Colon",
    "Gallbladder",
    "Kidneys",
    "Liver",
    "Pancreas",
    "Prostate",
    "Small bowel",
    "Spleen",
    "Stomach",
    "Urinary bladder",
)


@dataclass(frozen=True)
class PathConfig:
    merlin_repo: Path
    dataset_root: Path
    metadata_csv: Path
    semantic_targets_jsonl: Path | None
    semantic_vocab_json: Path | None
    output_root: Path
    cache_dir: Path
    image_embedding_cache_dir: Path


@dataclass(frozen=True)
class DataConfig:
    train_split: str = "train_subset"
    val_split: str = "val_subset"
    organ_names: tuple[str, ...] = DEFAULT_ORGANS
    train_limit: int | None = 8
    val_limit: int | None = 4
    sample_seed: int = 13
    include_review_required_semantic_targets: bool = False


@dataclass(frozen=True)
class ModelConfig:
    freeze_image_encoder: bool = True
    train_adapter: bool = True
    train_decoder_lora: bool = True
    max_length: int = 1024
    merlin_load_cwd: Path | None = None
    image_embedding_mode: str = "online"
    append_eos_to_target: bool = True


@dataclass(frozen=True)
class LossConfig:
    ce_weight: float = 1.0
    lexical_weight: float = 0.0
    lexical_mode: str = "auxiliary"
    lexical_target_cache: Path | None = None
    negative_temperature: float = 8.0
    positive_pathology_weight: float = 1.0
    negative_pathology_weight: float = 0.5
    epsilon: float = 1.0e-6
    semantic_weight: float = 0.0
    semantic_variant: str = "family"
    normality_weight: float = 1.0
    polarity_weight: float = 0.25
    family_weight: float = 1.0
    subtype_weight: float = 1.0
    confidence_scaling: bool = True
    review_required_weight: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    run_id: str
    epochs: int = 1
    max_steps: int | None = 2
    batch_size: int = 1
    num_workers: int = 0
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    grad_accum_steps: int = 1
    log_every: int = 1
    save_checkpoint: bool = False
    save_trainable_checkpoint: bool = True
    eval_every_epochs: int = 1
    mixed_precision: str = "bf16"
    device: str = "cuda"
    resume_from_checkpoint: Path | None = None


@dataclass(frozen=True)
class AblationConfig:
    paths: PathConfig
    data: DataConfig
    model: ModelConfig
    losses: LossConfig
    train: TrainConfig
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return self.paths.output_root / self.train.run_id


def load_config(path: str | Path) -> AblationConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths_raw = raw.get("paths", {})
    train_raw = raw.get("train", {})
    if "run_id" not in train_raw:
        raise ValueError("Config requires train.run_id")

    merlin_repo = _path(paths_raw.get("merlin_repo", "/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin"))
    paths = PathConfig(
        merlin_repo=merlin_repo,
        dataset_root=_path(paths_raw.get("dataset_root", "/net/storage/pr3/plgrid/plggjmiag/Merlin_converted")),
        metadata_csv=_path(paths_raw.get("metadata_csv", "/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv")),
        semantic_targets_jsonl=_optional_path(paths_raw.get("semantic_targets_jsonl")),
        semantic_vocab_json=_optional_path(paths_raw.get("semantic_vocab_json")),
        output_root=_path(paths_raw.get("output_root", "/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/models_ablations/merlin")),
        cache_dir=_path(paths_raw.get("cache_dir", "/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/models_ablations/merlin/cache")),
        image_embedding_cache_dir=_path(
            paths_raw.get(
                "image_embedding_cache_dir",
                "/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/models_ablations/merlin/image_embedding_cache",
            )
        ),
    )

    data_raw = raw.get("data", {})
    data = DataConfig(
        train_split=str(data_raw.get("train_split", "train_subset")),
        val_split=str(data_raw.get("val_split", "val_subset")),
        organ_names=tuple(data_raw.get("organ_names", DEFAULT_ORGANS)),
        train_limit=_optional_int(data_raw.get("train_limit", 8)),
        val_limit=_optional_int(data_raw.get("val_limit", 4)),
        sample_seed=int(data_raw.get("sample_seed", 13)),
        include_review_required_semantic_targets=bool(data_raw.get("include_review_required_semantic_targets", False)),
    )

    model_raw = raw.get("model", {})
    model = ModelConfig(
        freeze_image_encoder=bool(model_raw.get("freeze_image_encoder", True)),
        train_adapter=bool(model_raw.get("train_adapter", True)),
        train_decoder_lora=bool(model_raw.get("train_decoder_lora", True)),
        max_length=int(model_raw.get("max_length", 1024)),
        merlin_load_cwd=_optional_path(model_raw.get("merlin_load_cwd")) or merlin_repo,
        image_embedding_mode=str(model_raw.get("image_embedding_mode", "online")),
        append_eos_to_target=bool(model_raw.get("append_eos_to_target", True)),
    )
    if model.image_embedding_mode not in {"online", "cached"}:
        raise ValueError(f"Unsupported model.image_embedding_mode: {model.image_embedding_mode}")

    losses_raw = raw.get("losses", {})
    losses = LossConfig(
        ce_weight=float(losses_raw.get("ce_weight", 1.0)),
        lexical_weight=float(losses_raw.get("lexical_weight", 0.0)),
        lexical_mode=str(losses_raw.get("lexical_mode", "auxiliary")),
        lexical_target_cache=_optional_path(losses_raw.get("lexical_target_cache")),
        negative_temperature=float(losses_raw.get("negative_temperature", 8.0)),
        positive_pathology_weight=float(losses_raw.get("positive_pathology_weight", 1.0)),
        negative_pathology_weight=float(losses_raw.get("negative_pathology_weight", 0.5)),
        epsilon=float(losses_raw.get("epsilon", 1.0e-6)),
        semantic_weight=float(losses_raw.get("semantic_weight", 0.0)),
        semantic_variant=str(losses_raw.get("semantic_variant", "family")),
        normality_weight=float(losses_raw.get("normality_weight", 1.0)),
        polarity_weight=float(losses_raw.get("polarity_weight", 0.25)),
        family_weight=float(losses_raw.get("family_weight", 1.0)),
        subtype_weight=float(losses_raw.get("subtype_weight", 1.0)),
        confidence_scaling=bool(losses_raw.get("confidence_scaling", True)),
        review_required_weight=float(losses_raw.get("review_required_weight", 0.0)),
    )
    if losses.lexical_mode not in {"auxiliary", "concept_specific"}:
        raise ValueError(f"Unsupported lexical_mode: {losses.lexical_mode}")
    if losses.semantic_variant not in {"minimal", "normality", "family", "family_subtype"}:
        raise ValueError(f"Unsupported semantic_variant: {losses.semantic_variant}")

    train = TrainConfig(
        run_id=str(train_raw["run_id"]),
        epochs=int(train_raw.get("epochs", 1)),
        max_steps=_optional_int(train_raw.get("max_steps", 2)),
        batch_size=int(train_raw.get("batch_size", 1)),
        num_workers=int(train_raw.get("num_workers", 0)),
        learning_rate=float(train_raw.get("learning_rate", 1.0e-5)),
        weight_decay=float(train_raw.get("weight_decay", 0.0)),
        grad_accum_steps=int(train_raw.get("grad_accum_steps", 1)),
        log_every=int(train_raw.get("log_every", 1)),
        save_checkpoint=bool(train_raw.get("save_checkpoint", False)),
        save_trainable_checkpoint=bool(train_raw.get("save_trainable_checkpoint", True)),
        eval_every_epochs=int(train_raw.get("eval_every_epochs", 1)),
        mixed_precision=str(train_raw.get("mixed_precision", "bf16")),
        device=str(train_raw.get("device", "cuda")),
        resume_from_checkpoint=_optional_path(train_raw.get("resume_from_checkpoint")),
    )
    return AblationConfig(paths=paths, data=data, model=model, losses=losses, train=train, raw=raw)


def _path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _optional_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _path(value)


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return int(value)
