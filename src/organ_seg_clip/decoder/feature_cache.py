"""Visual feature cache creation for decoder training."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config.schemas import DecoderConfig
from ..data.dataset import MerlinWholeStudyDataset, collate_whole_study_batch
from ..models.visual_encoder import load_distilled_visual_encoder
from .data import DecoderFeatureRecord, DecoderFeatureStore, load_decoder_samples, load_feature_store, save_feature_store


def feature_cache_path(config: DecoderConfig, split: str) -> Path | None:
    cache_dir = config.resolved_feature_cache_dir
    if cache_dir is None:
        return None
    return cache_dir / f"{split}_features.pt"


def load_or_build_feature_store(config: DecoderConfig, *, split: str, device: torch.device) -> tuple[DecoderFeatureStore, dict[str, Any]]:
    path = feature_cache_path(config, split)
    if path is not None and path.is_file():
        return load_feature_store(path), {"feature_cache": str(path), "built": False}
    if not config.training.precompute_features_if_missing:
        raise FileNotFoundError(f"Missing feature cache for split={split}: {path}")
    store, summary = build_feature_store(config, split=split, device=device)
    if path is not None:
        save_feature_store(path, store)
        summary["feature_cache"] = str(path)
    return store, summary


def build_feature_store(config: DecoderConfig, *, split: str, device: torch.device) -> tuple[DecoderFeatureStore, dict[str, Any]]:
    visual_encoder, payload = load_distilled_visual_encoder(config.resolved_visual_encoder_checkpoint, map_location=device)
    visual_encoder = visual_encoder.to(device)
    visual_encoder.eval()
    encoder_config = visual_encoder.config
    encoder_config = replace(
        encoder_config,
        paths=replace(encoder_config.paths, dataset_root=str(config.resolved_dataset_root)),
        data=replace(
            encoder_config.data,
            train_split=config.data.train_split,
            val_split=config.data.val_split,
            train_limit=config.data.train_limit,
            val_limit=config.data.val_limit,
            organ_names=config.data.organ_names,
            verify_metadata=config.data.verify_metadata,
            lesion_metadata_csv=config.data.lesion_metadata_csv,
        ),
    )
    sample_seed = config.training.seed if split == config.data.train_split else config.training.seed + 1
    samples, dataset_summary = load_decoder_samples(config, split=split, sample_seed=sample_seed)
    dataset = MerlinWholeStudyDataset(samples, config=encoder_config)
    cache_batch_size = int(config.training.cache_build_batch_size or config.training.batch_size)
    cache_num_workers = int(
        config.training.num_workers if config.training.cache_build_num_workers is None else config.training.cache_build_num_workers
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, cache_batch_size),
        shuffle=False,
        num_workers=max(0, cache_num_workers),
        pin_memory=bool(config.training.pin_memory),
        persistent_workers=bool(config.training.persistent_workers and cache_num_workers > 0),
        collate_fn=collate_whole_study_batch,
    )
    records: dict[str, DecoderFeatureRecord] = {}
    use_amp = bool(device.type == "cuda" and getattr(config.training, "amp", False))
    with torch.no_grad():
        for batch in loader:
            moved = _move_encoder_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = visual_encoder(moved)
            for row_index, study_id in enumerate(output.study_ids):
                records[str(study_id)] = DecoderFeatureRecord(
                    study_id=str(study_id),
                    report_embedding=output.report_embedding[row_index].detach().cpu(),
                    organ_embeddings=output.organ_embeddings[row_index].detach().cpu(),
                    study_latents=output.study_latents[row_index].detach().cpu(),
                    visual_tokens=output.visual_tokens[row_index].detach().cpu(),
                    visual_token_mask=output.visual_token_mask[row_index].detach().cpu(),
                )
    resample_spacing = encoder_config.preprocessing.resample_spacing
    store = DecoderFeatureStore(
        organ_names=tuple(getattr(visual_encoder, "organ_names", config.data.organ_names)),
        visual_dim=int(getattr(visual_encoder, "visual_dim", 256)),
        records=records,
        metadata={
            "source_visual_encoder_checkpoint": str(config.resolved_visual_encoder_checkpoint),
            "source_visual_encoder_epoch": payload.get("source_epoch"),
            "source_visual_encoder_step": payload.get("source_step"),
            "split": split,
            # Stored so downstream consumers can verify the cache was built at the expected spacing.
            "resample_spacing_mm": list(resample_spacing) if resample_spacing is not None else None,
        },
    )
    return store, {
        "dataset": dataset_summary,
        "feature_count": len(records),
        "built": True,
        "cache_build_batch_size": float(cache_batch_size),
        "cache_build_num_workers": float(cache_num_workers),
    }


def _move_encoder_batch_to_device(batch: Any, device: torch.device) -> Any:
    return type(batch)(
        study_ids=batch.study_ids,
        images=batch.images.to(device, non_blocking=True),
        image_mask=batch.image_mask.to(device, non_blocking=True),
        segmentations=None if batch.segmentations is None else batch.segmentations.to(device, non_blocking=True),
        segmentation_mask=None if batch.segmentation_mask is None else batch.segmentation_mask.to(device, non_blocking=True),
        report_texts=batch.report_texts,
        organ_texts=batch.organ_texts,
        organ_raw_texts=batch.organ_raw_texts,
        organ_text_mask=batch.organ_text_mask.to(device, non_blocking=True),
        organ_labels=batch.organ_labels.to(device, non_blocking=True),
        organ_label_mask=batch.organ_label_mask.to(device, non_blocking=True),
        lesion_global_labels=batch.lesion_global_labels.to(device, non_blocking=True),
        lesion_global_mask=batch.lesion_global_mask.to(device, non_blocking=True),
        lesion_organ_labels=batch.lesion_organ_labels.to(device, non_blocking=True),
        lesion_organ_mask=batch.lesion_organ_mask.to(device, non_blocking=True),
        metadata=batch.metadata,
    )
