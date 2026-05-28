"""Dataset and collation utilities for per-organ decoder training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from ..config.schemas import DecoderConfig
from ..data.contracts import MerlinConvertedDataset, WholeStudySample
from ..data.lesion_metadata import LesionMetadataLookup
from .semantic_targets import SemanticTargetLookup


FEATURE_CACHE_FORMAT = "organsegclip_decoder_feature_cache_v1"


@dataclass(frozen=True)
class DecoderFeatureRecord:
    study_id: str
    report_embedding: torch.Tensor
    organ_embeddings: torch.Tensor
    study_latents: torch.Tensor | None = None
    visual_tokens: torch.Tensor | None = None
    visual_token_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class DecoderFeatureStore:
    organ_names: tuple[str, ...]
    visual_dim: int
    records: dict[str, DecoderFeatureRecord]
    metadata: dict[str, Any]

    def get(self, study_id: str) -> DecoderFeatureRecord | None:
        return self.records.get(str(study_id))


@dataclass(frozen=True)
class DecoderExample:
    study_id: str
    organ_name: str
    organ_index: int
    target_text: str
    organ_abnormal_label: int | None
    lesion_label: float
    lesion_mask: bool
    is_small_bowel: bool
    features: DecoderFeatureRecord
    semantic_available: bool
    semantic_weight: float
    semantic_status: str
    semantic_normality_target: int
    semantic_polarity_target: int
    semantic_primary_subtype_target: int
    semantic_active_subtype_indices: tuple[int, ...]
    semantic_active_subtype_weights: tuple[float, ...]
    semantic_secondary_subtype_indices: tuple[int, ...]
    semantic_allowed_subtype_indices: tuple[int, ...]
    semantic_family_indices: tuple[int, ...]
    semantic_family_weights: tuple[float, ...]
    semantic_allowed_family_indices: tuple[int, ...]
    semantic_subtype_vocab_size: int
    semantic_family_vocab_size: int


@dataclass(frozen=True)
class DecoderBatch:
    study_ids: list[str]
    organ_names: list[str]
    organ_indices: torch.Tensor
    visual_features: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    organ_abnormal_labels: torch.Tensor
    organ_abnormal_mask: torch.Tensor
    lesion_labels: torch.Tensor
    lesion_mask: torch.Tensor
    small_bowel_mask: torch.Tensor
    target_texts: list[str]
    semantic_statuses: list[str]
    semantic_available: torch.Tensor
    semantic_weights: torch.Tensor
    semantic_normality_targets: torch.Tensor
    semantic_polarity_targets: torch.Tensor
    semantic_primary_subtype_targets: torch.Tensor
    semantic_subtype_targets: torch.Tensor
    semantic_secondary_subtype_targets: torch.Tensor
    semantic_allowed_subtype_mask: torch.Tensor
    semantic_family_targets: torch.Tensor
    semantic_allowed_family_mask: torch.Tensor


def load_decoder_samples(config: DecoderConfig, *, split: str, sample_seed: int | None = None) -> tuple[list[WholeStudySample], dict[str, Any]]:
    import random

    dataset = MerlinConvertedDataset(config.resolved_dataset_root, verify_metadata=config.data.verify_metadata)
    samples = dataset.iter_samples(split, organ_names=config.data.organ_names)
    if sample_seed is not None:
        random.Random(int(sample_seed)).shuffle(samples)
    limit = config.data.train_limit if split == config.data.train_split else config.data.val_limit
    if limit is not None:
        samples = samples[: int(limit)]
    return samples, dataset.inspection_summary()


class PerOrganDecoderDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[WholeStudySample],
        *,
        feature_store: DecoderFeatureStore,
        config: DecoderConfig,
        split: str,
        repeat_positives: bool = False,
    ) -> None:
        self.config = config
        self.split = str(split)
        self.organ_names = tuple(config.data.organ_names)
        self.repeat_positives = bool(repeat_positives)
        lesion_csv = config.resolved_lesion_metadata_csv
        self.lesion_lookup = LesionMetadataLookup(lesion_csv, organ_names=self.organ_names)
        self.semantic_lookup = None
        if config.semantic_loss.enabled:
            self.semantic_lookup = _load_semantic_lookup(config, self.organ_names)
        self.examples = self._build_examples(samples, feature_store)
        self.summary = self._summarize_examples(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DecoderExample:
        return self.examples[index]

    def _build_examples(self, samples: Sequence[WholeStudySample], feature_store: DecoderFeatureStore) -> list[DecoderExample]:
        examples: list[DecoderExample] = []
        abnormal_only = bool(self.config.data.train_abnormal_only) if self.split == self.config.data.train_split else bool(self.config.data.val_abnormal_only)
        for sample in samples:
            features = feature_store.get(sample.study_id)
            if features is None:
                continue
            lesion_record = self.lesion_lookup.get(sample.study_id)
            for organ_index, organ_name in enumerate(self.organ_names):
                target_text = str(sample.organ_text_lookup.get(organ_name, "")).strip()
                if not target_text:
                    continue
                organ_abnormal_label = sample.organ_label_lookup.get(organ_name)
                lesion_label = 0.0
                lesion_mask = False
                if lesion_record is not None and organ_name in lesion_record.organ_labels:
                    lesion_label = float(lesion_record.organ_labels[organ_name])
                    lesion_mask = True
                if abnormal_only and not _is_abnormal_example(organ_abnormal_label, lesion_label=lesion_label, lesion_mask=lesion_mask):
                    continue
                semantic_target = self.semantic_lookup.get(organ_name, target_text) if self.semantic_lookup is not None else None
                allowed_subtype_indices = ()
                allowed_family_indices = ()
                subtype_vocab_size = 0
                family_vocab_size = 0
                if self.semantic_lookup is not None:
                    allowed_subtype_indices = self.semantic_lookup.spec.organ_to_subtype_indices.get(organ_name, ())
                    allowed_family_indices = self.semantic_lookup.spec.organ_to_family_indices.get(organ_name, ())
                    subtype_vocab_size = len(self.semantic_lookup.spec.subtype_vocab)
                    family_vocab_size = len(self.semantic_lookup.spec.family_vocab)
                active_subtype_indices = () if semantic_target is None else tuple(int(value) for value in semantic_target.subtype_indices)
                active_subtype_weights = ()
                if semantic_target is not None:
                    active_subtype_weights = tuple(float(semantic_target.subtype_weights.get(index, 1.0)) for index in active_subtype_indices)
                family_indices = () if semantic_target is None else tuple(int(value) for value in semantic_target.family_indices)
                family_weights = ()
                if semantic_target is not None:
                    family_weights = tuple(float(semantic_target.family_weights.get(index, 1.0)) for index in family_indices)
                examples.append(
                    DecoderExample(
                        study_id=sample.study_id,
                        organ_name=organ_name,
                        organ_index=organ_index,
                        target_text=target_text,
                        organ_abnormal_label=organ_abnormal_label,
                        lesion_label=lesion_label,
                        lesion_mask=lesion_mask,
                        is_small_bowel=(organ_name == "Small bowel"),
                        features=features,
                        semantic_available=semantic_target is not None and float(semantic_target.sample_weight) > 0.0,
                        semantic_weight=0.0 if semantic_target is None else float(semantic_target.sample_weight),
                        semantic_status="" if semantic_target is None else str(semantic_target.decision_status),
                        semantic_normality_target=-100 if semantic_target is None else int(semantic_target.normality_index),
                        semantic_polarity_target=-100 if semantic_target is None else int(semantic_target.polarity_index),
                        semantic_primary_subtype_target=-100 if semantic_target is None else int(semantic_target.primary_subtype_index),
                        semantic_active_subtype_indices=active_subtype_indices,
                        semantic_active_subtype_weights=active_subtype_weights,
                        semantic_secondary_subtype_indices=() if semantic_target is None else tuple(int(value) for value in semantic_target.secondary_subtype_indices),
                        semantic_allowed_subtype_indices=tuple(int(value) for value in allowed_subtype_indices),
                        semantic_family_indices=family_indices,
                        semantic_family_weights=family_weights,
                        semantic_allowed_family_indices=tuple(int(value) for value in allowed_family_indices),
                        semantic_subtype_vocab_size=int(subtype_vocab_size),
                        semantic_family_vocab_size=int(family_vocab_size),
                    )
                )
        if not self.repeat_positives:
            return examples
        return _repeat_supervised_positives(
            examples,
            lesion_positive_repeat_factor=self.config.data.lesion_positive_repeat_factor,
            abnormal_label_repeat_factor=self.config.data.abnormal_label_repeat_factor,
        )

    def _summarize_examples(self, examples: Sequence[DecoderExample]) -> dict[str, int]:
        abnormal = 0
        normal = 0
        unknown = 0
        lesion_positive = 0
        lesion_labeled = 0
        semantic_available = 0
        semantic_provisional = 0
        for example in examples:
            if example.organ_abnormal_label == 1:
                abnormal += 1
            elif example.organ_abnormal_label == 0:
                normal += 1
            else:
                unknown += 1
            if example.lesion_mask:
                lesion_labeled += 1
                if example.lesion_label > 0.5:
                    lesion_positive += 1
            if example.semantic_available:
                semantic_available += 1
            if example.semantic_status == "accepted_provisional":
                semantic_provisional += 1
        return {
            "example_count": int(len(examples)),
            "abnormal_label_positive_count": int(abnormal),
            "abnormal_label_negative_count": int(normal),
            "abnormal_label_unknown_count": int(unknown),
            "lesion_labeled_count": int(lesion_labeled),
            "lesion_positive_count": int(lesion_positive),
            "semantic_available_count": int(semantic_available),
            "semantic_provisional_count": int(semantic_provisional),
        }


def select_visual_prefix(record: DecoderFeatureRecord, *, organ_index: int, mode: str) -> torch.Tensor:
    organ = record.organ_embeddings[int(organ_index)].unsqueeze(0)
    if mode == "organ_only":
        chunks = [organ]
    elif mode == "report_plus_organ":
        chunks = [record.report_embedding.unsqueeze(0), organ]
    elif mode == "report_plus_organ_plus_study_latents":
        if record.study_latents is None:
            raise ValueError("Feature cache does not contain study_latents required by visual prefix mode.")
        chunks = [record.report_embedding.unsqueeze(0), organ, record.study_latents]
    elif mode == "report_plus_organ_plus_visual_tokens":
        if record.visual_tokens is None:
            raise ValueError("Feature cache does not contain visual_tokens required by visual prefix mode.")
        chunks = [record.report_embedding.unsqueeze(0), organ, record.visual_tokens]
    else:
        raise ValueError(f"Unsupported visual prefix mode: {mode}")
    return torch.cat(chunks, dim=0).float()


def collate_decoder_batch(
    batch: Sequence[DecoderExample],
    *,
    tokenizer: Any,
    prompt_template: str,
    visual_prefix_mode: str,
    max_length: int,
    include_target: bool = True,
) -> DecoderBatch:
    input_id_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    visual_rows: list[torch.Tensor] = []
    study_ids: list[str] = []
    organ_names: list[str] = []
    organ_indices: list[int] = []
    target_texts: list[str] = []
    organ_abnormal_labels: list[float] = []
    organ_abnormal_mask: list[bool] = []
    lesion_labels: list[float] = []
    lesion_mask: list[bool] = []
    small_bowel_mask: list[bool] = []
    semantic_available: list[bool] = []
    semantic_weights: list[float] = []
    semantic_statuses: list[str] = []
    semantic_normality_targets: list[int] = []
    semantic_polarity_targets: list[int] = []
    semantic_primary_targets: list[int] = []

    eos = getattr(tokenizer, "eos_token", None) or ""
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    subtype_dim = 0
    family_dim = 0
    for example in batch:
        subtype_dim = max(subtype_dim, int(example.semantic_subtype_vocab_size))
        family_dim = max(family_dim, int(example.semantic_family_vocab_size))
        if example.semantic_allowed_subtype_indices:
            subtype_dim = max(subtype_dim, max(example.semantic_allowed_subtype_indices) + 1)
        if example.semantic_active_subtype_indices:
            subtype_dim = max(subtype_dim, max(example.semantic_active_subtype_indices) + 1)
        if example.semantic_secondary_subtype_indices:
            subtype_dim = max(subtype_dim, max(example.semantic_secondary_subtype_indices) + 1)
        if example.semantic_allowed_family_indices:
            family_dim = max(family_dim, max(example.semantic_allowed_family_indices) + 1)
        if example.semantic_family_indices:
            family_dim = max(family_dim, max(example.semantic_family_indices) + 1)
    for example in batch:
        prompt = prompt_template.format(organ=example.organ_name)
        target = f"{example.target_text}{eos}"
        prompt_ids = _encode_text(tokenizer, prompt)
        target_ids = _encode_text(tokenizer, target) if include_target else []
        ids = (prompt_ids + target_ids)[: int(max_length)]
        label_values = ([-100] * len(prompt_ids) + target_ids)[: int(max_length)]
        input_id_rows.append(torch.tensor(ids, dtype=torch.long))
        label_rows.append(torch.tensor(label_values, dtype=torch.long))
        visual_rows.append(select_visual_prefix(example.features, organ_index=example.organ_index, mode=visual_prefix_mode))
        study_ids.append(example.study_id)
        organ_names.append(example.organ_name)
        organ_indices.append(int(example.organ_index))
        target_texts.append(example.target_text)
        organ_abnormal_labels.append(0.0 if example.organ_abnormal_label is None else float(example.organ_abnormal_label))
        organ_abnormal_mask.append(example.organ_abnormal_label is not None)
        lesion_labels.append(float(example.lesion_label))
        lesion_mask.append(bool(example.lesion_mask))
        small_bowel_mask.append(bool(example.is_small_bowel))
        semantic_available.append(bool(example.semantic_available))
        semantic_weights.append(float(example.semantic_weight))
        semantic_statuses.append(str(example.semantic_status))
        semantic_normality_targets.append(int(example.semantic_normality_target))
        semantic_polarity_targets.append(int(example.semantic_polarity_target))
        semantic_primary_targets.append(int(example.semantic_primary_subtype_target))

    text_length = max(row.numel() for row in input_id_rows)
    visual_length = max(row.shape[0] for row in visual_rows)
    visual_dim = int(visual_rows[0].shape[-1])
    input_ids = torch.full((len(batch), text_length), int(pad_token_id), dtype=torch.long)
    labels = torch.full((len(batch), text_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), text_length), dtype=torch.long)
    visual_features = torch.zeros((len(batch), visual_length, visual_dim), dtype=torch.float32)
    semantic_subtype_targets = torch.zeros((len(batch), subtype_dim), dtype=torch.float32)
    semantic_secondary_subtype_targets = torch.zeros((len(batch), subtype_dim), dtype=torch.float32)
    semantic_allowed_subtype_mask = torch.zeros((len(batch), subtype_dim), dtype=torch.bool)
    semantic_family_targets = torch.zeros((len(batch), family_dim), dtype=torch.float32)
    semantic_allowed_family_mask = torch.zeros((len(batch), family_dim), dtype=torch.bool)
    for row_index, (ids, row_labels, visual) in enumerate(zip(input_id_rows, label_rows, visual_rows)):
        input_ids[row_index, : ids.numel()] = ids
        labels[row_index, : row_labels.numel()] = row_labels
        attention_mask[row_index, : ids.numel()] = 1
        visual_features[row_index, : visual.shape[0]] = visual
        for subtype_index, weight in zip(batch[row_index].semantic_active_subtype_indices, batch[row_index].semantic_active_subtype_weights):
            semantic_subtype_targets[row_index, int(subtype_index)] = float(weight)
        for subtype_index in batch[row_index].semantic_secondary_subtype_indices:
            semantic_secondary_subtype_targets[row_index, int(subtype_index)] = 1.0
        for subtype_index in batch[row_index].semantic_allowed_subtype_indices:
            semantic_allowed_subtype_mask[row_index, int(subtype_index)] = True
        for family_index, weight in zip(batch[row_index].semantic_family_indices, batch[row_index].semantic_family_weights):
            semantic_family_targets[row_index, int(family_index)] = float(weight)
        for family_index in batch[row_index].semantic_allowed_family_indices:
            semantic_allowed_family_mask[row_index, int(family_index)] = True
    return DecoderBatch(
        study_ids=study_ids,
        organ_names=organ_names,
        organ_indices=torch.tensor(organ_indices, dtype=torch.long),
        visual_features=visual_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        organ_abnormal_labels=torch.tensor(organ_abnormal_labels, dtype=torch.float32),
        organ_abnormal_mask=torch.tensor(organ_abnormal_mask, dtype=torch.bool),
        lesion_labels=torch.tensor(lesion_labels, dtype=torch.float32),
        lesion_mask=torch.tensor(lesion_mask, dtype=torch.bool),
        small_bowel_mask=torch.tensor(small_bowel_mask, dtype=torch.bool),
        target_texts=target_texts,
        semantic_statuses=semantic_statuses,
        semantic_available=torch.tensor(semantic_available, dtype=torch.bool),
        semantic_weights=torch.tensor(semantic_weights, dtype=torch.float32),
        semantic_normality_targets=torch.tensor(semantic_normality_targets, dtype=torch.long),
        semantic_polarity_targets=torch.tensor(semantic_polarity_targets, dtype=torch.long),
        semantic_primary_subtype_targets=torch.tensor(semantic_primary_targets, dtype=torch.long),
        semantic_subtype_targets=semantic_subtype_targets,
        semantic_secondary_subtype_targets=semantic_secondary_subtype_targets,
        semantic_allowed_subtype_mask=semantic_allowed_subtype_mask,
        semantic_family_targets=semantic_family_targets,
        semantic_allowed_family_mask=semantic_allowed_family_mask,
    )


def decoder_collate_fn(
    *,
    tokenizer: Any,
    prompt_template: str,
    visual_prefix_mode: str,
    max_length: int,
    include_target: bool = True,
) -> Callable[[Sequence[DecoderExample]], DecoderBatch]:
    def _collate(batch: Sequence[DecoderExample]) -> DecoderBatch:
        return collate_decoder_batch(
            batch,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            visual_prefix_mode=visual_prefix_mode,
            max_length=max_length,
            include_target=include_target,
        )

    return _collate


def _load_semantic_lookup(config: DecoderConfig, organ_names: Sequence[str]) -> SemanticTargetLookup | None:
    training_targets = config.resolved_semantic_training_targets_jsonl
    training_vocab = config.resolved_semantic_training_vocab_json
    if training_targets is not None or training_vocab is not None:
        if training_targets is None or training_vocab is None:
            raise ValueError("semantic_loss.training_targets_jsonl and semantic_loss.training_vocab_json must be configured together.")
        return SemanticTargetLookup.from_training_targets(
            targets_path=training_targets,
            vocab_path=training_vocab,
            organ_names=organ_names,
            accepted_sample_weight=float(config.semantic_loss.accepted_sample_weight),
            provisional_sample_weight=float(config.semantic_loss.provisional_sample_weight),
            unresolved_sample_weight=float(config.semantic_loss.unresolved_sample_weight),
            use_confidence_scaling=bool(config.semantic_loss.use_confidence_scaling),
            include_review_required=bool(config.semantic_loss.include_review_required),
            review_required_sample_weight=float(config.semantic_loss.review_required_sample_weight),
        )
    return SemanticTargetLookup.from_jsonl_paths(
        config.resolved_semantic_target_jsonl_paths,
        organ_names=organ_names,
        accepted_sample_weight=float(config.semantic_loss.accepted_sample_weight),
        provisional_sample_weight=float(config.semantic_loss.provisional_sample_weight),
        unresolved_sample_weight=float(config.semantic_loss.unresolved_sample_weight),
        use_confidence_scaling=bool(config.semantic_loss.use_confidence_scaling),
    )


def save_feature_store(path: str | Path, store: DecoderFeatureStore) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FEATURE_CACHE_FORMAT,
        "organ_names": list(store.organ_names),
        "visual_dim": int(store.visual_dim),
        "metadata": dict(store.metadata),
        "features": {
            study_id: {
                "report_embedding": record.report_embedding.cpu(),
                "organ_embeddings": record.organ_embeddings.cpu(),
                "study_latents": None if record.study_latents is None else record.study_latents.cpu(),
                "visual_tokens": None if record.visual_tokens is None else record.visual_tokens.cpu(),
                "visual_token_mask": None if record.visual_token_mask is None else record.visual_token_mask.cpu(),
            }
            for study_id, record in store.records.items()
        },
    }
    torch.save(payload, target)
    return target


def load_feature_store(path: str | Path) -> DecoderFeatureStore:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu")
    if payload.get("format") != FEATURE_CACHE_FORMAT:
        raise ValueError("Unsupported decoder feature cache format.")
    records: dict[str, DecoderFeatureRecord] = {}
    for study_id, raw in payload["features"].items():
        records[str(study_id)] = DecoderFeatureRecord(
            study_id=str(study_id),
            report_embedding=raw["report_embedding"].float(),
            organ_embeddings=raw["organ_embeddings"].float(),
            study_latents=None if raw.get("study_latents") is None else raw["study_latents"].float(),
            visual_tokens=None if raw.get("visual_tokens") is None else raw["visual_tokens"].float(),
            visual_token_mask=None if raw.get("visual_token_mask") is None else raw["visual_token_mask"].bool(),
        )
    return DecoderFeatureStore(
        organ_names=tuple(payload["organ_names"]),
        visual_dim=int(payload["visual_dim"]),
        records=records,
        metadata=dict(payload.get("metadata", {})),
    )


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _repeat_supervised_positives(
    examples: list[DecoderExample],
    *,
    lesion_positive_repeat_factor: int,
    abnormal_label_repeat_factor: int,
) -> list[DecoderExample]:
    repeated: list[DecoderExample] = []
    for example in examples:
        repeat = 1
        if example.lesion_mask and example.lesion_label > 0.5:
            repeat = max(repeat, int(lesion_positive_repeat_factor))
        if example.organ_abnormal_label == 1:
            repeat = max(repeat, int(abnormal_label_repeat_factor))
        repeated.extend([example] * repeat)
    return repeated


def _is_abnormal_example(organ_abnormal_label: int | None, *, lesion_label: float, lesion_mask: bool) -> bool:
    if organ_abnormal_label == 1:
        return True
    if lesion_mask and lesion_label > 0.5:
        return True
    return False
