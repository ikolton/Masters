"""Image-only visual encoder distilled from OrganSegCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.schemas import EncoderConfig
from ..models.interfaces.types import EncoderBatch
from ..models.segmamba import SegMambaEncoder
from ..training.checkpointing import resolve_checkpoint_state_dict
from .aggregation.modules import (
    GridFeatureCombiner,
    LatentStudyAggregator,
    OrganPatchAttentionHead,
    OrganQueryHead,
    PatchPositionEmbedding,
    PatchSummaryHead,
    PatchTokenizer,
    StudyReportHead,
)
from .aggregation.tiling import crop_to_bounds, extract_tile, generate_tile_boxes, mask_bounds, normalized_box_features
from .aggregation.model import _grid_position_features, _pad_token_sequences


@dataclass(frozen=True)
class VisualEncoderOutput:
    """Visual representations used by the report decoder stage."""

    study_ids: list[str]
    report_embedding: torch.Tensor
    organ_embeddings: torch.Tensor
    study_latents: torch.Tensor
    visual_tokens: torch.Tensor
    visual_token_mask: torch.Tensor
    organ_names: tuple[str, ...]


class VisualOrganEncoder(nn.Module):
    """Image-only OrganSegCLIP path for downstream report generation.

    This module intentionally excludes text encoders, alignment losses,
    segmentation heads, and auxiliary classifiers. It preserves the trained
    volumetric visual path and the learned report/organ query heads.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        seg_config = config.model.segmamba
        model_dim = int(config.model.tokenizer.model_dim)
        organ_count = int(config.model.organ_query_count)
        self.organ_names = tuple(config.data.organ_names)
        self.patch_encoder = SegMambaEncoder(
            in_chans=seg_config.in_channels,
            depths=seg_config.depths,
            dims=seg_config.feat_size,
            d_state=seg_config.d_state,
            d_conv=seg_config.d_conv,
            expand=seg_config.expand,
            out_indices=seg_config.out_indices,
            activation_checkpointing=seg_config.activation_checkpointing,
        )
        self.patch_tokenizer = PatchTokenizer(
            input_dim=seg_config.feat_size[-1],
            model_dim=model_dim,
            summary_grid=config.model.tokenizer.summary_grid,
        )
        self.patch_position_embedding = PatchPositionEmbedding(model_dim)
        self.use_grid_combiner = bool(config.model.grid_combiner.enabled)
        self.patch_summary_head = PatchSummaryHead(
            model_dim=model_dim,
            num_heads=config.model.grid_combiner.num_heads,
            dropout=config.model.grid_combiner.dropout,
            summary_mode=config.model.grid_combiner.patch_summary_mode,
        )
        self.grid_combiner = GridFeatureCombiner(
            model_dim=model_dim,
            depth=config.model.grid_combiner.depth,
            num_heads=config.model.grid_combiner.num_heads,
            dropout=config.model.grid_combiner.dropout,
            use_global_token=config.model.grid_combiner.use_global_token,
        )
        self.study_aggregator = LatentStudyAggregator(
            model_dim=model_dim,
            num_latents=config.model.aggregator.num_latents,
            num_layers=config.model.aggregator.num_layers,
            num_heads=config.model.aggregator.num_heads,
            dropout=config.model.aggregator.dropout,
        )
        self.organ_head = OrganQueryHead(
            query_count=organ_count,
            model_dim=model_dim,
            num_heads=config.model.aggregator.num_heads,
            dropout=config.model.aggregator.dropout,
        )
        self.organ_patch_attention_head = OrganPatchAttentionHead(
            query_count=organ_count,
            model_dim=model_dim,
            dropout=config.model.aggregator.dropout,
        )
        self.organ_patch_fusion = nn.Sequential(nn.LayerNorm(model_dim * 2), nn.Linear(model_dim * 2, model_dim))
        self.report_head = StudyReportHead(
            model_dim=model_dim,
            num_heads=config.model.aggregator.num_heads,
            dropout=config.model.aggregator.dropout,
        )
        self.patch_size = tuple(int(v) for v in config.model.patching.patch_size)
        self.patch_stride = tuple(int(v) for v in config.model.patching.patch_stride)
        self.patch_batch_size = int(config.model.patching.patch_batch_size)
        self.visual_dim = model_dim

    def forward(self, batch: EncoderBatch) -> VisualEncoderOutput:
        token_sequences = [self._encode_single_study(batch, sample_index) for sample_index in range(batch.images.shape[0])]
        visual_tokens, visual_token_mask = _pad_token_sequences(token_sequences)
        study_latents = self.study_aggregator(visual_tokens, visual_token_mask)
        organ_features = self.organ_head(study_latents)
        organ_patch_features, _ = self.organ_patch_attention_head(visual_tokens, visual_token_mask)
        organ_embeddings = F.normalize(
            self.organ_patch_fusion(torch.cat([organ_features, organ_patch_features], dim=-1)),
            dim=-1,
        )
        report_embedding = self.report_head(study_latents)
        return VisualEncoderOutput(
            study_ids=list(batch.study_ids),
            report_embedding=report_embedding,
            organ_embeddings=organ_embeddings,
            study_latents=study_latents,
            visual_tokens=visual_tokens,
            visual_token_mask=visual_token_mask,
            organ_names=self.organ_names,
        )

    def _encode_single_study(self, batch: EncoderBatch, sample_index: int) -> torch.Tensor:
        image = batch.images[sample_index]
        image_mask = batch.image_mask[sample_index, 0]
        bounds = mask_bounds(image_mask)
        cropped_image = crop_to_bounds(image, bounds)
        spatial_shape = tuple(int(v) for v in cropped_image.shape[-3:])
        boxes = generate_tile_boxes(spatial_shape, self.patch_size, self.patch_stride)
        token_chunks: list[torch.Tensor] = []
        patch_summary_chunks: list[torch.Tensor] = []
        position_chunks: list[torch.Tensor] = []
        for start in range(0, len(boxes), self.patch_batch_size):
            chunk_boxes = boxes[start:start + self.patch_batch_size]
            image_tiles = torch.stack([extract_tile(cropped_image, box, self.patch_size) for box in chunk_boxes], dim=0)
            feature_pyramid = self.patch_encoder(image_tiles)
            patch_tokens = self.patch_tokenizer(feature_pyramid[-1])
            if self.use_grid_combiner:
                patch_summary_chunks.append(self.patch_summary_head(patch_tokens))
                position_chunks.append(_grid_position_features(chunk_boxes, spatial_shape, reference_boxes=boxes, device=image_tiles.device))
            else:
                positions = torch.stack(
                    [normalized_box_features(box, spatial_shape, device=image_tiles.device) for box in chunk_boxes],
                    dim=0,
                )
                chunk_tokens = patch_tokens + self.patch_position_embedding(positions).unsqueeze(1)
                token_chunks.append(chunk_tokens.reshape(-1, chunk_tokens.shape[-1]))
        if self.use_grid_combiner:
            patch_summaries = torch.cat(patch_summary_chunks, dim=0).unsqueeze(0)
            grid_positions = torch.cat(position_chunks, dim=0).unsqueeze(0)
            grid_mask = torch.ones((1, patch_summaries.shape[1]), device=patch_summaries.device, dtype=torch.bool)
            return self.grid_combiner(patch_summaries, grid_positions, grid_mask).squeeze(0)
        return torch.cat(token_chunks, dim=0)


def build_visual_encoder(config: EncoderConfig) -> VisualOrganEncoder:
    return VisualOrganEncoder(config)


def load_visual_weights_from_full_checkpoint(
    visual_encoder: VisualOrganEncoder,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)
    full_state = resolve_checkpoint_state_dict(payload)
    visual_state = visual_encoder.state_dict()
    matched_state = {
        key: value
        for key, value in full_state.items()
        if key in visual_state and getattr(visual_state[key], "shape", None) == getattr(value, "shape", None)
    }
    missing, unexpected = visual_encoder.load_state_dict(matched_state, strict=False)
    return {
        "payload": payload,
        "matched_keys": int(len(matched_state)),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "skipped_full_keys": int(len(full_state) - len(matched_state)),
    }


def load_distilled_visual_encoder(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[VisualOrganEncoder, dict[str, Any]]:
    """Load a visual encoder exported by apps/export_visual_encoder.py."""
    payload = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=map_location)
    if payload.get("format") != "organsegclip_visual_encoder_v1":
        raise ValueError("Unsupported visual encoder checkpoint format.")
    from ..config.loader import encoder_config_from_dict

    config_dict = dict(payload["config"])
    config = encoder_config_from_dict(config_dict, config_path=str(Path(checkpoint_path).expanduser().resolve()))
    encoder = build_visual_encoder(config)
    encoder.load_state_dict(payload["model_state"], strict=True)
    return encoder, payload
