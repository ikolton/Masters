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


@dataclass(frozen=True)
class DecoderBatch:
    study_ids: list[str]
    organ_names: list[str]
    organ_indices: torch.Tensor
    visual_features: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    lesion_labels: torch.Tensor
    lesion_mask: torch.Tensor
    small_bowel_mask: torch.Tensor
    target_texts: list[str]


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
        lesion_csv = Path(config.data.lesion_metadata_csv).expanduser() if config.data.lesion_metadata_csv else None
        if lesion_csv is not None and not lesion_csv.is_absolute():
            lesion_csv = Path(config.config_dir) / lesion_csv
        self.lesion_lookup = LesionMetadataLookup(lesion_csv, organ_names=self.organ_names)
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
        return {
            "example_count": int(len(examples)),
            "abnormal_label_positive_count": int(abnormal),
            "abnormal_label_negative_count": int(normal),
            "abnormal_label_unknown_count": int(unknown),
            "lesion_labeled_count": int(lesion_labeled),
            "lesion_positive_count": int(lesion_positive),
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
    lesion_labels: list[float] = []
    lesion_mask: list[bool] = []
    small_bowel_mask: list[bool] = []

    eos = getattr(tokenizer, "eos_token", None) or ""
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
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
        lesion_labels.append(float(example.lesion_label))
        lesion_mask.append(bool(example.lesion_mask))
        small_bowel_mask.append(bool(example.is_small_bowel))

    text_length = max(row.numel() for row in input_id_rows)
    visual_length = max(row.shape[0] for row in visual_rows)
    visual_dim = int(visual_rows[0].shape[-1])
    input_ids = torch.full((len(batch), text_length), int(pad_token_id), dtype=torch.long)
    labels = torch.full((len(batch), text_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), text_length), dtype=torch.long)
    visual_features = torch.zeros((len(batch), visual_length, visual_dim), dtype=torch.float32)
    for row_index, (ids, row_labels, visual) in enumerate(zip(input_id_rows, label_rows, visual_rows)):
        input_ids[row_index, : ids.numel()] = ids
        labels[row_index, : row_labels.numel()] = row_labels
        attention_mask[row_index, : ids.numel()] = 1
        visual_features[row_index, : visual.shape[0]] = visual
    return DecoderBatch(
        study_ids=study_ids,
        organ_names=organ_names,
        organ_indices=torch.tensor(organ_indices, dtype=torch.long),
        visual_features=visual_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        lesion_labels=torch.tensor(lesion_labels, dtype=torch.float32),
        lesion_mask=torch.tensor(lesion_mask, dtype=torch.bool),
        small_bowel_mask=torch.tensor(small_bowel_mask, dtype=torch.bool),
        target_texts=target_texts,
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
