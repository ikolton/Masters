"""Typed configuration schemas for OrganSegCLIP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path


DEFAULT_ORGANS: tuple[str, ...] = (
    "Spleen",
    "Kidneys",
    "Gallbladder",
    "Liver",
    "Stomach",
    "Pancreas",
    "Adrenal glands",
    "Small bowel",
    "Colon",
    "Urinary bladder",
    "Prostate",
)


@dataclass(frozen=True)
class PathsConfig:
    dataset_root_env: str = "ORGAN_SEG_CLIP_DATASET_ROOT"
    dataset_root: str = ""
    output_dir: str = "outputs/encoder/default"

    def resolve_dataset_root(self, config_dir: Path) -> Path:
        root = self.dataset_root or os.environ.get(self.dataset_root_env, "")
        if not root:
            raise ValueError(
                "Dataset root is not configured. Set it in the config or export "
                f"{self.dataset_root_env}."
            )
        target = Path(root)
        if not target.is_absolute():
            target = config_dir / target
        return target.expanduser().resolve()

    def resolve_output_dir(self, config_dir: Path) -> Path:
        target = Path(self.output_dir)
        if not target.is_absolute():
            target = config_dir / target
        return target.expanduser().resolve()


@dataclass(frozen=True)
class DataConfig:
    train_split: str = "train"
    val_split: str = "val"
    train_limit: int | None = None
    val_limit: int | None = None
    organ_names: tuple[str, ...] = DEFAULT_ORGANS
    verify_metadata: bool = True
    lesion_metadata_csv: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "organ_names", tuple(self.organ_names))


@dataclass(frozen=True)
class PreprocessingConfig:
    intensity_clip_min: float = -1000.0
    intensity_clip_max: float = 1000.0
    intensity_mode: str = "scale_to_unit"
    verify_orientation_spacing: bool = True
    canonicalize_orientation: bool = False
    resample_spacing: tuple[float, float, float] | None = (2.0, 2.0, 2.0)
    foreground_crop: bool = True
    foreground_crop_backend: str = "monai"
    foreground_threshold: float | None = -950.0
    foreground_crop_margin: tuple[int, int, int] | int = (8, 8, 8)
    foreground_crop_k_divisible: tuple[int, int, int] | int = (1, 1, 1)
    canonical_size: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.intensity_mode not in {"scale_to_unit", "zscore"}:
            raise ValueError("preprocessing.intensity_mode must be 'scale_to_unit' or 'zscore'.")
        if self.foreground_crop_backend not in {"monai", "manual"}:
            raise ValueError("preprocessing.foreground_crop_backend must be 'monai' or 'manual'.")
        if self.resample_spacing is not None:
            object.__setattr__(self, "resample_spacing", tuple(float(v) for v in self.resample_spacing))
        if self.foreground_threshold is not None:
            object.__setattr__(self, "foreground_threshold", float(self.foreground_threshold))
        object.__setattr__(self, "foreground_crop_margin", _spatial_tuple(self.foreground_crop_margin, "foreground_crop_margin"))
        object.__setattr__(self, "foreground_crop_k_divisible", _spatial_tuple(self.foreground_crop_k_divisible, "foreground_crop_k_divisible"))
        if self.canonical_size is not None:
            object.__setattr__(self, "canonical_size", tuple(int(v) for v in self.canonical_size))


def _spatial_tuple(value: tuple[int, int, int] | int, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    converted = tuple(int(v) for v in value)
    if len(converted) != 3:
        raise ValueError(f"preprocessing.{field_name} must be an int or a length-3 sequence.")
    return converted


@dataclass(frozen=True)
class TextEncoderConfig:
    backend_family: str = "pubmedbert"
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    freeze_encoder: bool = True
    unfreeze_last_n_layers: int = 0
    max_tokens: int = 48
    report_max_tokens: int | None = None
    pooling: str = "cls"
    projection_dim: int = 256
    hash_vocab_size: int = 4096
    organ_text_template: str = "{organ}: {finding}"
    cache_frozen_outputs: bool = True
    cache_max_entries: int = 100000

    def __post_init__(self) -> None:
        if self.backend_family not in {"pubmedbert", "hash"}:
            raise ValueError("text_encoder.backend_family must be 'pubmedbert' or 'hash'.")
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("text_encoder.pooling must be 'cls' or 'mean'.")
        object.__setattr__(self, "unfreeze_last_n_layers", int(self.unfreeze_last_n_layers))
        object.__setattr__(self, "cache_max_entries", int(self.cache_max_entries))
        if self.cache_max_entries < 0:
            raise ValueError("text_encoder.cache_max_entries must be non-negative.")
        if self.report_max_tokens is not None:
            object.__setattr__(self, "report_max_tokens", int(self.report_max_tokens))


@dataclass(frozen=True)
class SegMambaConfig:
    in_channels: int = 1
    depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    feat_size: tuple[int, int, int, int] = (48, 96, 192, 384)
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    out_indices: tuple[int, int, int, int] = (0, 1, 2, 3)
    activation_checkpointing: bool = True
    pretrained_checkpoint_path: str = ""
    segmentation_class_count: int = 23
    segmentation_full_resolution: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(int(v) for v in self.depths))
        object.__setattr__(self, "feat_size", tuple(int(v) for v in self.feat_size))
        object.__setattr__(self, "out_indices", tuple(int(v) for v in self.out_indices))


@dataclass(frozen=True)
class PatchingConfig:
    patch_size: tuple[int, int, int] = (128, 128, 128)
    patch_stride: tuple[int, int, int] = (96, 96, 96)
    patch_batch_size: int = 1
    segmentation_supervision_max_patches_per_study: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "patch_size", tuple(int(v) for v in self.patch_size))
        object.__setattr__(self, "patch_stride", tuple(int(v) for v in self.patch_stride))
        object.__setattr__(
            self,
            "segmentation_supervision_max_patches_per_study",
            int(self.segmentation_supervision_max_patches_per_study),
        )


@dataclass(frozen=True)
class TokenizerConfig:
    model_dim: int = 256
    summary_grid: tuple[int, int, int] = (2, 2, 2)

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary_grid", tuple(int(v) for v in self.summary_grid))


@dataclass(frozen=True)
class AggregatorConfig:
    num_latents: int = 32
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.0


@dataclass(frozen=True)
class GridCombinerConfig:
    enabled: bool = False
    depth: int = 2
    num_heads: int = 8
    dropout: float = 0.0
    position_features: str = "grid_box"
    use_global_token: bool = False
    patch_summary_mode: str = "attention"

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", int(self.depth))
        object.__setattr__(self, "num_heads", int(self.num_heads))
        object.__setattr__(self, "dropout", float(self.dropout))
        if self.position_features != "grid_box":
            raise ValueError("model.grid_combiner.position_features must be 'grid_box'.")
        if self.patch_summary_mode not in {"attention", "attention_mean"}:
            raise ValueError("model.grid_combiner.patch_summary_mode must be 'attention' or 'attention_mean'.")


@dataclass(frozen=True)
class AlignmentProjectionConfig:
    enabled: bool = False
    hidden_dim: int = 512
    bottleneck_dim: int = 256
    dropout: float = 0.0
    layer_norm: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dim", int(self.hidden_dim))
        object.__setattr__(self, "bottleneck_dim", int(self.bottleneck_dim))
        object.__setattr__(self, "dropout", float(self.dropout))


@dataclass(frozen=True)
class OrgansConfig:
    diagnostic_dropout: float = 0.0
    patch_organ_min_voxels: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "patch_organ_min_voxels", int(self.patch_organ_min_voxels))


@dataclass(frozen=True)
class ModelConfig:
    segmamba: SegMambaConfig = field(default_factory=SegMambaConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    grid_combiner: GridCombinerConfig = field(default_factory=GridCombinerConfig)
    alignment_projection: AlignmentProjectionConfig = field(default_factory=AlignmentProjectionConfig)
    organs: OrgansConfig = field(default_factory=OrgansConfig)
    organ_query_count: int = len(DEFAULT_ORGANS)


@dataclass(frozen=True)
class LossConfig:
    organ_clip_weight: float = 1.0
    report_clip_weight: float = 0.0
    alignment_type: str = "clip"
    organ_alignment_weight: float | None = None
    report_alignment_weight: float | None = None
    segmentation_weight: float = 1.0
    diagnostic_weight: float = 0.25
    patch_organ_presence_weight: float = 0.0
    lesion_global_weight: float = 0.0
    lesion_organ_weight: float = 0.0
    organ_attention_weight: float = 0.0
    organ_pair_balance: bool = False
    organ_positive_weight: float = 1.0
    organ_same_organ_weight: float = 1.0
    organ_cross_organ_weight: float = 1.0
    organ_frequency_balance: bool = False
    organ_frequency_balance_power: float = 0.5
    organ_frequency_balance_min: float = 0.25
    organ_frequency_balance_max: float = 4.0
    segmentation_loss_type: str = "dice_ce"

    def __post_init__(self) -> None:
        if self.segmentation_loss_type not in {"ce", "dice_ce"}:
            raise ValueError("loss.segmentation_loss_type must be 'ce' or 'dice_ce'.")
        if self.alignment_type not in {"clip", "siglip"}:
            raise ValueError("loss.alignment_type must be 'clip' or 'siglip'.")
        for field_name in (
            "organ_positive_weight",
            "organ_same_organ_weight",
            "organ_cross_organ_weight",
            "organ_frequency_balance_power",
            "organ_frequency_balance_min",
            "organ_frequency_balance_max",
        ):
            value = float(getattr(self, field_name))
            if value < 0.0:
                raise ValueError(f"loss.{field_name} must be non-negative.")
            object.__setattr__(self, field_name, value)
        if self.organ_frequency_balance_min <= 0.0:
            raise ValueError("loss.organ_frequency_balance_min must be positive.")
        if self.organ_frequency_balance_max < self.organ_frequency_balance_min:
            raise ValueError("loss.organ_frequency_balance_max must be >= loss.organ_frequency_balance_min.")
        if self.organ_alignment_weight is None:
            object.__setattr__(self, "organ_alignment_weight", float(self.organ_clip_weight))
        else:
            object.__setattr__(self, "organ_alignment_weight", float(self.organ_alignment_weight))
            object.__setattr__(self, "organ_clip_weight", float(self.organ_alignment_weight))
        if self.report_alignment_weight is None:
            object.__setattr__(self, "report_alignment_weight", float(self.report_clip_weight))
        else:
            object.__setattr__(self, "report_alignment_weight", float(self.report_alignment_weight))
            object.__setattr__(self, "report_clip_weight", float(self.report_alignment_weight))


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 13
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    batch_size: int = 1
    epochs: int = 1
    learning_rate: float = 1e-4
    text_learning_rate: float | None = None
    alignment_parameter_learning_rate: float | None = None
    alignment_parameter_names: tuple[str, ...] = ("organ_logit_scale", "organ_logit_bias")
    weight_decay: float = 1e-4
    scheduler_type: str = "none"
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    scheduler_interval: str = "step"
    amp: bool = True
    amp_dtype: str = "float16"
    max_grad_norm: float | None = None
    log_every_steps: int = 10
    save_every_steps: int = 0
    validation_every_epochs: int = 1
    fast_val_limit: int | None = None
    fast_val_sampling: str = "fixed"
    fast_val_skip_segmentation: bool = False
    profile_timing: bool = False
    max_train_steps: int = 0
    max_val_steps: int = 0
    save_every_epochs: int = 1
    save_last_checkpoint: bool = True
    save_best_checkpoint: bool = True
    best_checkpoint_metric: str = "val_total_loss"
    ddp_find_unused_parameters: bool = False
    resume_from: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "alignment_parameter_names", tuple(str(value) for value in self.alignment_parameter_names))
        scheduler_type = str(self.scheduler_type).strip().lower()
        scheduler_interval = str(self.scheduler_interval).strip().lower()
        amp_dtype = str(self.amp_dtype).strip().lower()
        fast_val_sampling = str(self.fast_val_sampling).strip().lower()
        if scheduler_type not in {"none", "cosine"}:
            raise ValueError("training.scheduler_type must be 'none' or 'cosine'.")
        if scheduler_interval != "step":
            raise ValueError("training.scheduler_interval currently only supports 'step'.")
        if amp_dtype not in {"float16", "fp16", "bfloat16", "bf16"}:
            raise ValueError("training.amp_dtype must be 'float16' or 'bfloat16'.")
        if fast_val_sampling not in {"fixed", "epoch_random"}:
            raise ValueError("training.fast_val_sampling must be 'fixed' or 'epoch_random'.")
        if int(self.warmup_steps) < 0:
            raise ValueError("training.warmup_steps must be non-negative.")
        if float(self.min_learning_rate) < 0.0:
            raise ValueError("training.min_learning_rate must be non-negative.")
        if self.alignment_parameter_learning_rate is not None and float(self.alignment_parameter_learning_rate) <= 0.0:
            raise ValueError("training.alignment_parameter_learning_rate must be positive when set.")
        object.__setattr__(self, "scheduler_type", scheduler_type)
        object.__setattr__(self, "scheduler_interval", scheduler_interval)
        object.__setattr__(self, "amp_dtype", "bfloat16" if amp_dtype in {"bfloat16", "bf16"} else "float16")
        object.__setattr__(self, "fast_val_sampling", fast_val_sampling)
        object.__setattr__(self, "warmup_steps", int(self.warmup_steps))
        object.__setattr__(self, "min_learning_rate", float(self.min_learning_rate))


@dataclass(frozen=True)
class RuntimeConfig:
    compile_model: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    experiment_name: str = "organ_seg_clip"
    wandb_enabled: bool = False
    wandb_project: str = "organ_seg_clip"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "online"
    wandb_tags: tuple[str, ...] = ()
    wandb_dir: str = ""
    wandb_step_log_start: int = 0
    wandb_resume_run_id: str = ""

    def __post_init__(self) -> None:
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("logging.wandb_mode must be 'online', 'offline', or 'disabled'.")
        object.__setattr__(self, "wandb_tags", tuple(str(value) for value in self.wandb_tags))
        object.__setattr__(self, "wandb_step_log_start", int(self.wandb_step_log_start))


@dataclass(frozen=True)
class DecoderPathsConfig:
    dataset_root_env: str = "ORGAN_SEG_CLIP_DATASET_ROOT"
    dataset_root: str = ""
    output_dir: str = "outputs/decoder/default"
    visual_encoder_checkpoint: str = ""
    feature_cache_dir: str = ""

    def resolve_dataset_root(self, config_dir: Path) -> Path:
        root = self.dataset_root or os.environ.get(self.dataset_root_env, "")
        if not root:
            raise ValueError(
                "Dataset root is not configured. Set it in the decoder config or export "
                f"{self.dataset_root_env}."
            )
        target = Path(root)
        if not target.is_absolute():
            target = config_dir / target
        return target.expanduser().resolve()

    def resolve_output_dir(self, config_dir: Path) -> Path:
        target = Path(self.output_dir)
        if not target.is_absolute():
            target = config_dir / target
        return target.expanduser().resolve()

    def resolve_visual_encoder_checkpoint(self, config_dir: Path) -> Path:
        if not str(self.visual_encoder_checkpoint).strip():
            raise ValueError("paths.visual_encoder_checkpoint must be configured for decoder training.")
        target = Path(self.visual_encoder_checkpoint)
        if not target.is_absolute():
            target = config_dir / target
        return target.expanduser().resolve()

    def resolve_feature_cache_dir(self, config_dir: Path) -> Path | None:
        if not str(self.feature_cache_dir).strip():
            return None
        target = Path(self.feature_cache_dir)
        if not target.is_absolute():
            target = config_dir / target
        return target.expanduser().resolve()


@dataclass(frozen=True)
class DecoderDataConfig:
    train_split: str = "train"
    val_split: str = "val"
    train_limit: int | None = None
    val_limit: int | None = None
    organ_names: tuple[str, ...] = DEFAULT_ORGANS
    verify_metadata: bool = True
    lesion_metadata_csv: str = ""
    train_abnormal_only: bool = False
    val_abnormal_only: bool = False
    lesion_positive_repeat_factor: int = 1
    abnormal_label_repeat_factor: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "organ_names", tuple(self.organ_names))
        object.__setattr__(self, "lesion_positive_repeat_factor", int(self.lesion_positive_repeat_factor))
        object.__setattr__(self, "abnormal_label_repeat_factor", int(self.abnormal_label_repeat_factor))
        if self.lesion_positive_repeat_factor < 1:
            raise ValueError("data.lesion_positive_repeat_factor must be >= 1.")
        if self.abnormal_label_repeat_factor < 1:
            raise ValueError("data.abnormal_label_repeat_factor must be >= 1.")


@dataclass(frozen=True)
class DecoderModelConfig:
    llm_model_name_or_path: str = "Qwen/Qwen2.5-0.5B"
    visual_prefix_mode: str = "report_plus_organ"
    prompt_template: str = "Generate the CT finding for {organ}###\n"
    max_length: int = 256
    max_new_tokens: int = 128
    freeze_llm: bool = True
    gradient_checkpointing: bool = True
    torch_dtype: str = "auto"

    def __post_init__(self) -> None:
        allowed_modes = {
            "organ_only",
            "report_plus_organ",
            "report_plus_organ_plus_study_latents",
            "report_plus_organ_plus_visual_tokens",
        }
        if self.visual_prefix_mode not in allowed_modes:
            raise ValueError(f"decoder.model.visual_prefix_mode must be one of {sorted(allowed_modes)}.")
        object.__setattr__(self, "max_length", int(self.max_length))
        object.__setattr__(self, "max_new_tokens", int(self.max_new_tokens))


@dataclass(frozen=True)
class DecoderLoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", int(self.r))
        object.__setattr__(self, "alpha", int(self.alpha))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "target_modules", tuple(str(value) for value in self.target_modules))


@dataclass(frozen=True)
class DecoderDiagnosticLossConfig:
    enabled: bool = True
    weight: float = 0.5
    positive_pathology_weight: float = 1.0
    negative_pathology_weight: float = 0.5
    small_bowel_duodenum_negative_weight: float = 0.25
    positive_normal_penalty_weight: float = 0.25
    epsilon: float = 1.0e-6
    pathology_words: tuple[str, ...] = (
        "lesion",
        "lesions",
        "cyst",
        "cysts",
        "mass",
        "masses",
        "nodule",
        "nodules",
        "metastasis",
        "metastases",
        "tumor",
        "tumour",
    )
    normal_words: tuple[str, ...] = (
        "unremarkable",
        "normal",
        "within normal limits",
        "no abnormality",
        "no focal abnormality",
    )

    def __post_init__(self) -> None:
        for field_name in (
            "weight",
            "positive_pathology_weight",
            "negative_pathology_weight",
            "small_bowel_duodenum_negative_weight",
            "positive_normal_penalty_weight",
            "epsilon",
        ):
            object.__setattr__(self, field_name, float(getattr(self, field_name)))
        object.__setattr__(self, "pathology_words", tuple(str(value) for value in self.pathology_words))
        object.__setattr__(self, "normal_words", tuple(str(value) for value in self.normal_words))


@dataclass(frozen=True)
class DecoderTrainingConfig:
    seed: int = 13
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    batch_size: int = 1
    epochs: int = 1
    learning_rate: float = 2.0e-4
    projector_learning_rate: float | None = None
    weight_decay: float = 0.0
    amp: bool = True
    max_grad_norm: float | None = 1.0
    log_every_steps: int = 10
    save_every_steps: int = 0
    save_every_epochs: int = 1
    save_last_checkpoint: bool = True
    save_best_checkpoint: bool = True
    best_checkpoint_metric: str = "val_total_loss"
    ddp_find_unused_parameters: bool = False
    resume_from: str | None = None
    precompute_features_if_missing: bool = True


@dataclass(frozen=True)
class DecoderGenerationConfig:
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1.2
    max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_beams", int(self.num_beams))
        object.__setattr__(self, "repetition_penalty", float(self.repetition_penalty))
        if self.max_new_tokens is not None:
            object.__setattr__(self, "max_new_tokens", int(self.max_new_tokens))


@dataclass(frozen=True)
class DecoderConfig:
    config_path: str
    config_dir: str
    paths: DecoderPathsConfig
    data: DecoderDataConfig
    model: DecoderModelConfig
    lora: DecoderLoraConfig
    diagnostic_loss: DecoderDiagnosticLossConfig
    training: DecoderTrainingConfig
    generation: DecoderGenerationConfig
    logging: LoggingConfig

    @property
    def resolved_dataset_root(self) -> Path:
        return self.paths.resolve_dataset_root(Path(self.config_dir))

    @property
    def resolved_output_dir(self) -> Path:
        return self.paths.resolve_output_dir(Path(self.config_dir))

    @property
    def resolved_visual_encoder_checkpoint(self) -> Path:
        return self.paths.resolve_visual_encoder_checkpoint(Path(self.config_dir))

    @property
    def resolved_feature_cache_dir(self) -> Path | None:
        return self.paths.resolve_feature_cache_dir(Path(self.config_dir))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EncoderConfig:
    config_path: str
    config_dir: str
    paths: PathsConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    text_encoder: TextEncoderConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    logging: LoggingConfig

    @property
    def resolved_dataset_root(self) -> Path:
        return self.paths.resolve_dataset_root(Path(self.config_dir))

    @property
    def resolved_output_dir(self) -> Path:
        return self.paths.resolve_output_dir(Path(self.config_dir))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
