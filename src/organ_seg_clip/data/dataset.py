"""Whole-study dataset and padded collator."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from ..config.schemas import EncoderConfig
from ..models.interfaces.types import EncoderBatch
from .contracts import MerlinConvertedDataset, WholeStudySample
from .lesion_metadata import LesionMetadataLookup
from .organ_masks import DEFAULT_MERLIN_MASK_MAP, load_organ_mask_map
from .preprocessing import load_and_preprocess_study


def load_samples_from_config(
    config: EncoderConfig,
    *,
    split: str,
    sample_seed: int | None = None,
) -> tuple[list[WholeStudySample], dict[str, Any]]:
    dataset = MerlinConvertedDataset(config.resolved_dataset_root, verify_metadata=config.data.verify_metadata)
    samples = dataset.iter_samples(split, organ_names=config.data.organ_names)
    if sample_seed is not None:
        random.Random(int(sample_seed)).shuffle(samples)
    limit = config.data.train_limit if split == config.data.train_split else config.data.val_limit
    if limit is not None:
        samples = samples[: int(limit)]
    return samples, dataset.inspection_summary()


class MerlinWholeStudyDataset(Dataset):
    def __init__(self, samples: list[WholeStudySample], *, config: EncoderConfig) -> None:
        self.samples = list(samples)
        self.config = config
        self.organ_names = tuple(config.data.organ_names)
        lesion_csv = Path(config.data.lesion_metadata_csv).expanduser() if config.data.lesion_metadata_csv else None
        if lesion_csv is not None and not lesion_csv.is_absolute():
            lesion_csv = Path(config.config_dir) / lesion_csv
        self.lesion_lookup = LesionMetadataLookup(lesion_csv, organ_names=self.organ_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        preprocessed = load_and_preprocess_study(
            scan_path=sample.scan_path,
            segmentation_path=sample.segmentation_path,
            config=self.config.preprocessing,
        )
        organ_texts: list[str] = []
        organ_raw_texts: list[str] = []
        organ_text_mask = torch.zeros((len(self.organ_names),), dtype=torch.bool)
        organ_labels = torch.zeros((len(self.organ_names),), dtype=torch.float32)
        organ_label_mask = torch.zeros((len(self.organ_names),), dtype=torch.bool)
        lesion_organ_labels = torch.zeros((len(self.organ_names),), dtype=torch.float32)
        lesion_organ_mask = torch.zeros((len(self.organ_names),), dtype=torch.bool)
        lesion_global_label = torch.zeros((), dtype=torch.float32)
        lesion_global_mask = torch.tensor(False, dtype=torch.bool)
        lesion_record = self.lesion_lookup.get(sample.study_id)
        if lesion_record is not None:
            lesion_global_label = torch.tensor(float(lesion_record.global_label), dtype=torch.float32)
            lesion_global_mask = torch.tensor(True, dtype=torch.bool)
        for organ_index, organ_name in enumerate(self.organ_names):
            raw_text = sample.organ_text_lookup.get(organ_name, "")
            text = _format_organ_text(self.config.text_encoder.organ_text_template, organ=organ_name, finding=raw_text) if raw_text else ""
            label = sample.organ_label_lookup.get(organ_name)
            organ_texts.append(text)
            organ_raw_texts.append(raw_text)
            if raw_text:
                organ_text_mask[organ_index] = True
            if label is not None:
                organ_labels[organ_index] = float(label)
                organ_label_mask[organ_index] = True
            if lesion_record is not None and organ_name in lesion_record.organ_labels:
                lesion_organ_labels[organ_index] = float(lesion_record.organ_labels[organ_name])
                lesion_organ_mask[organ_index] = True
        return {
            "study_id": sample.study_id,
            "image": preprocessed.image,
            "segmentation": preprocessed.segmentation,
            "report_text": sample.report_text,
            "organ_texts": organ_texts,
            "organ_raw_texts": organ_raw_texts,
            "organ_text_mask": organ_text_mask,
            "organ_labels": organ_labels,
            "organ_label_mask": organ_label_mask,
            "lesion_global_label": lesion_global_label,
            "lesion_global_mask": lesion_global_mask,
            "lesion_organ_labels": lesion_organ_labels,
            "lesion_organ_mask": lesion_organ_mask,
            "metadata": {
                "split": sample.split,
                "image_metadata": preprocessed.image_metadata,
                "segmentation_metadata": preprocessed.segmentation_metadata,
                "preprocessing_crop": dict(preprocessed.crop_info),
                "original_image_shape": tuple(int(v) for v in preprocessed.image.shape[-3:]),
            },
        }


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def collate_whole_study_batch(batch: list[dict[str, Any]]) -> EncoderBatch:
    # The encoder/decoder downsamples four times, so padding each batch to a
    # stride-compatible size avoids odd-shape skip-connection mismatches while
    # keeping natural per-study extents when canonical_size is disabled.
    spatial_multiple = 16
    max_d = _ceil_to_multiple(max(int(sample["image"].shape[-3]) for sample in batch), spatial_multiple)
    max_h = _ceil_to_multiple(max(int(sample["image"].shape[-2]) for sample in batch), spatial_multiple)
    max_w = _ceil_to_multiple(max(int(sample["image"].shape[-1]) for sample in batch), spatial_multiple)

    images: list[torch.Tensor] = []
    image_masks: list[torch.Tensor] = []
    segmentations: list[torch.Tensor] = []
    segmentation_masks: list[torch.Tensor] = []
    has_any_segmentation = any(sample["segmentation"] is not None for sample in batch)

    for sample in batch:
        image = sample["image"]
        d, h, w = (int(v) for v in image.shape[-3:])
        padded_image = torch.zeros((1, max_d, max_h, max_w), dtype=image.dtype)
        padded_image[:, :d, :h, :w] = image
        image_mask = torch.zeros((1, max_d, max_h, max_w), dtype=torch.bool)
        image_mask[:, :d, :h, :w] = True
        images.append(padded_image)
        image_masks.append(image_mask)

        if has_any_segmentation:
            padded_segmentation = torch.full((max_d, max_h, max_w), fill_value=-1, dtype=torch.long)
            segmentation_mask = torch.zeros((max_d, max_h, max_w), dtype=torch.bool)
            if sample["segmentation"] is not None:
                segmentation = sample["segmentation"]
                padded_segmentation[:d, :h, :w] = segmentation.long()
                segmentation_mask[:d, :h, :w] = True
            segmentations.append(padded_segmentation)
            segmentation_masks.append(segmentation_mask)

    return EncoderBatch(
        study_ids=[sample["study_id"] for sample in batch],
        images=torch.stack(images, dim=0),
        image_mask=torch.stack(image_masks, dim=0),
        segmentations=None if not has_any_segmentation else torch.stack(segmentations, dim=0),
        segmentation_mask=None if not has_any_segmentation else torch.stack(segmentation_masks, dim=0),
        report_texts=[sample["report_text"] for sample in batch],
        organ_texts=[list(sample["organ_texts"]) for sample in batch],
        organ_raw_texts=[list(sample.get("organ_raw_texts", sample["organ_texts"])) for sample in batch],
        organ_text_mask=torch.stack([sample["organ_text_mask"] for sample in batch], dim=0),
        organ_labels=torch.stack([sample["organ_labels"] for sample in batch], dim=0),
        organ_label_mask=torch.stack([sample["organ_label_mask"] for sample in batch], dim=0),
        lesion_global_labels=torch.stack([sample["lesion_global_label"] for sample in batch], dim=0),
        lesion_global_mask=torch.stack([sample["lesion_global_mask"] for sample in batch], dim=0),
        lesion_organ_labels=torch.stack([sample["lesion_organ_labels"] for sample in batch], dim=0),
        lesion_organ_mask=torch.stack([sample["lesion_organ_mask"] for sample in batch], dim=0),
        metadata=[dict(sample["metadata"]) for sample in batch],
    )


def _format_organ_text(template: str, *, organ: str, finding: str) -> str:
    try:
        return str(template).format(organ=organ, finding=finding).strip()
    except Exception as exc:
        raise ValueError("text_encoder.organ_text_template must support {organ} and {finding} fields.") from exc
