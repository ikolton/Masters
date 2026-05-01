"""OrganSegCLIP model."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ...config.schemas import EncoderConfig
from ...data.organ_masks import DEFAULT_MERLIN_MASK_MAP
from ...models.interfaces.types import EncoderBatch, OrganSegOutput
from ...models.losses.segmentation import multiclass_dice_score, segmentation_supervision_loss
from ...models.segmamba import SegMambaEncoder, SegMambaSegmentationHead
from ...models.text import build_text_encoder
from .modules import AlignmentProjectionHead, GridFeatureCombiner, LatentStudyAggregator, OrganPatchAttentionHead, OrganQueryHead, PatchPositionEmbedding, PatchSummaryHead, PatchTokenizer, StudyReportHead
from .tiling import crop_to_bounds, extract_tile, generate_tile_boxes, mask_bounds, normalized_box_features


@dataclass(frozen=True)
class _PreparedStudy:
    cropped_image: torch.Tensor
    cropped_segmentation: torch.Tensor | None
    cropped_segmentation_mask: torch.Tensor | None
    spatial_shape: tuple[int, int, int]
    boxes: list[tuple[int, int, int, int, int, int]]
    supervised_patch_indices: torch.Tensor | None


@dataclass(frozen=True)
class _DeferredSegmentationWork:
    study_index: int
    supervised_image_tiles: torch.Tensor
    supervised_segmentation_tiles: torch.Tensor
    supervised_segmentation_mask_tiles: torch.Tensor | None


class OrganSegCLIPModel(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        seg_config = config.model.segmamba
        model_dim = int(config.model.tokenizer.model_dim)
        organ_count = int(config.model.organ_query_count)
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
        self.patch_segmentation_head = SegMambaSegmentationHead(
            in_chans=seg_config.in_channels,
            out_chans=seg_config.segmentation_class_count,
            feat_size=seg_config.feat_size,
            full_resolution=seg_config.segmentation_full_resolution,
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
        self.diagnostic_head = nn.Sequential(
            nn.Dropout(config.model.organs.diagnostic_dropout),
            nn.Linear(model_dim, 1),
        )
        self.patch_organ_presence_head = nn.Linear(model_dim, organ_count)
        self.lesion_global_head = nn.Linear(model_dim, 1)
        self.lesion_organ_head = nn.Linear(model_dim, 1)
        self.text_encoder = build_text_encoder(config.text_encoder)
        projection_config = config.model.alignment_projection
        self.enable_alignment_projection = bool(projection_config.enabled)
        if self.enable_alignment_projection:
            projection_kwargs = {
                "model_dim": model_dim,
                "hidden_dim": projection_config.hidden_dim,
                "bottleneck_dim": projection_config.bottleneck_dim,
                "dropout": projection_config.dropout,
                "layer_norm": projection_config.layer_norm,
            }
            self.organ_image_projection = AlignmentProjectionHead(**projection_kwargs)
            self.organ_text_projection = AlignmentProjectionHead(**projection_kwargs)
            self.report_image_projection = AlignmentProjectionHead(**projection_kwargs)
            self.report_text_projection = AlignmentProjectionHead(**projection_kwargs)
        else:
            self.organ_image_projection = nn.Identity()
            self.organ_text_projection = nn.Identity()
            self.report_image_projection = nn.Identity()
            self.report_text_projection = nn.Identity()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))
        self.organ_logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        self.organ_logit_bias = nn.Parameter(torch.tensor(-10.0, dtype=torch.float32))
        self.report_logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        self.report_logit_bias = nn.Parameter(torch.tensor(-10.0, dtype=torch.float32))
        self.segmentation_loss_type = config.loss.segmentation_loss_type
        self.patch_size = tuple(int(v) for v in config.model.patching.patch_size)
        self.patch_stride = tuple(int(v) for v in config.model.patching.patch_stride)
        self.patch_batch_size = int(config.model.patching.patch_batch_size)
        self.segmentation_supervision_max_patches_per_study = int(
            config.model.patching.segmentation_supervision_max_patches_per_study
        )
        self.patch_encoder_min_free_bytes = _patch_encoder_min_free_bytes_from_env()
        self.segmentation_recompute_min_free_bytes = _segmentation_recompute_min_free_bytes_from_env()
        self.skip_segmentation_supervision_in_eval = False
        self.patch_organ_min_voxels = int(config.model.organs.patch_organ_min_voxels)
        self.organ_mask_label_values = tuple(DEFAULT_MERLIN_MASK_MAP.get(name, ()) for name in config.data.organ_names)
        self.organ_patch_target_mask = tuple(bool(values) for values in self.organ_mask_label_values)
        self.enable_organ_alignment = float(config.loss.organ_alignment_weight or 0.0) != 0.0
        self.enable_report_clip = float(config.loss.report_alignment_weight or 0.0) != 0.0
        self.enable_siglip_alignment = config.loss.alignment_type == "siglip"
        self.logit_scale.requires_grad_(not self.enable_siglip_alignment)
        self.organ_logit_scale.requires_grad_(self.enable_siglip_alignment and float(config.loss.organ_alignment_weight or 0.0) != 0.0)
        self.organ_logit_bias.requires_grad_(self.enable_siglip_alignment and float(config.loss.organ_alignment_weight or 0.0) != 0.0)
        self.report_logit_scale.requires_grad_(self.enable_siglip_alignment and self.enable_report_clip)
        self.report_logit_bias.requires_grad_(self.enable_siglip_alignment and self.enable_report_clip)
        self.enable_patch_organ_presence = float(config.loss.patch_organ_presence_weight) != 0.0
        self.enable_organ_attention = float(config.loss.organ_attention_weight) != 0.0
        self.enable_lesion_global = float(config.loss.lesion_global_weight) != 0.0
        self.enable_lesion_organ = float(config.loss.lesion_organ_weight) != 0.0
        _set_module_requires_grad(self.report_head, self.enable_report_clip or self.enable_lesion_global)
        _set_module_requires_grad(self.organ_patch_attention_head, self.enable_organ_attention)
        _set_module_requires_grad(self.organ_patch_fusion, self.enable_organ_attention)
        _set_module_requires_grad(self.organ_image_projection, self.enable_alignment_projection and self.enable_organ_alignment)
        _set_module_requires_grad(self.organ_text_projection, self.enable_alignment_projection and self.enable_organ_alignment)
        _set_module_requires_grad(self.report_image_projection, self.enable_alignment_projection and self.enable_report_clip)
        _set_module_requires_grad(self.report_text_projection, self.enable_alignment_projection and self.enable_report_clip)
        _set_module_requires_grad(self.patch_position_embedding, not self.use_grid_combiner)
        _set_module_requires_grad(self.patch_summary_head, self.use_grid_combiner)
        _set_module_requires_grad(self.grid_combiner, self.use_grid_combiner)
        _set_module_requires_grad(self.patch_organ_presence_head, self.enable_patch_organ_presence)
        _set_module_requires_grad(self.lesion_global_head, self.enable_lesion_global)
        _set_module_requires_grad(self.lesion_organ_head, self.enable_lesion_organ)

    def forward(self, batch: EncoderBatch) -> OrganSegOutput:
        prepared_studies = [self._prepare_study(batch, sample_index) for sample_index in range(batch.images.shape[0])]
        patches_per_study = [len(study.boxes) for study in prepared_studies]
        patches_per_batch_total = int(sum(patches_per_study))
        patches_per_study_mean = (
            float(patches_per_batch_total) / float(len(patches_per_study))
            if patches_per_study
            else 0.0
        )
        patches_per_study_max = int(max(patches_per_study)) if patches_per_study else 0
        encoded_studies = self._encode_studies_batched(prepared_studies)
        token_sequences: list[torch.Tensor] = []
        organ_attention_target_sequences: list[torch.Tensor] = []
        organ_attention_mask_sequences: list[torch.Tensor] = []
        segmentation_loss_sum = self.logit_scale.new_zeros(())
        segmentation_dice_sum = 0.0
        segmentation_patch_count = 0
        patch_organ_loss_sum = self.logit_scale.new_zeros(())
        patch_organ_correct_sum = 0.0
        patch_organ_count = 0
        segmentation_oom_fallback_count = 0
        for (
            sample_tokens,
            sample_seg_loss,
            sample_seg_dice,
            sample_patch_count,
            sample_patch_organ_loss,
            sample_patch_organ_accuracy,
            sample_patch_organ_count,
            sample_attention_targets,
            sample_attention_mask,
            sample_segmentation_oom_fallback_count,
        ) in encoded_studies:
            token_sequences.append(sample_tokens)
            organ_attention_target_sequences.append(sample_attention_targets)
            organ_attention_mask_sequences.append(sample_attention_mask)
            segmentation_loss_sum = segmentation_loss_sum + sample_seg_loss * max(sample_patch_count, 1)
            segmentation_dice_sum += sample_seg_dice * max(sample_patch_count, 1)
            segmentation_patch_count += sample_patch_count
            segmentation_oom_fallback_count += int(sample_segmentation_oom_fallback_count)
            if sample_patch_organ_count > 0:
                patch_organ_loss_sum = patch_organ_loss_sum + sample_patch_organ_loss * sample_patch_organ_count
                patch_organ_correct_sum += sample_patch_organ_accuracy * sample_patch_organ_count
                patch_organ_count += sample_patch_organ_count
        token_tensor, token_mask = _pad_token_sequences(token_sequences)
        attention_targets, attention_target_mask = _pad_organ_attention_targets(
            organ_attention_target_sequences,
            organ_attention_mask_sequences,
            max_tokens=token_tensor.shape[1],
            device=token_tensor.device,
        )
        study_latents = self.study_aggregator(token_tensor, token_mask)
        organ_image_features = self.organ_head(study_latents)
        organ_attention_loss = token_tensor.sum() * 0.0
        organ_attention_accuracy = 0.0
        organ_attention_positive_accuracy = 0.0
        organ_attention_negative_accuracy = 0.0
        organ_attention_count = 0
        organ_attention_positive_count = 0
        organ_attention_negative_count = 0
        if self.enable_organ_attention:
            organ_patch_features, organ_attention_logits = self.organ_patch_attention_head(token_tensor, token_mask)
            fused_organ_features = self.organ_patch_fusion(torch.cat([organ_image_features, organ_patch_features], dim=-1))
            organ_image_features = F.normalize(fused_organ_features.float(), dim=-1, eps=1e-6).to(fused_organ_features.dtype)
            if attention_target_mask.any():
                target_logits = organ_attention_logits.transpose(1, 2)[attention_target_mask]
                target_values = attention_targets[attention_target_mask]
                organ_attention_loss = _balanced_binary_cross_entropy_with_logits(target_logits, target_values)
                organ_attention_count = int(attention_target_mask.sum().item())
                positive_mask = target_values >= 0.5
                negative_mask = ~positive_mask
                organ_attention_positive_count = int(positive_mask.sum().item())
                organ_attention_negative_count = int(negative_mask.sum().item())
                predictions = (target_logits.detach().sigmoid() >= 0.5).float()
                correct = predictions == target_values
                organ_attention_accuracy = float(correct.float().mean().item())
                if positive_mask.any():
                    organ_attention_positive_accuracy = float(correct[positive_mask].float().mean().item())
                if negative_mask.any():
                    organ_attention_negative_accuracy = float(correct[negative_mask].float().mean().item())
        # Encode only the finding text here; organ identity is already provided
        # by the dedicated query slots and auxiliary organ supervision.
        organ_text_features = self.text_encoder.encode_nested_texts(batch.organ_raw_texts, batch.organ_text_mask)
        report_image_features = (
            self.report_head(study_latents)
            if self.enable_report_clip or self.enable_lesion_global
            else organ_image_features.new_zeros((organ_image_features.shape[0], organ_image_features.shape[-1]))
        )
        organ_image_embeddings = self.organ_image_projection(organ_image_features)
        organ_text_embeddings = self.organ_text_projection(organ_text_features)
        report_image_embeddings = self.report_image_projection(report_image_features) if self.enable_report_clip else report_image_features
        report_text_mask = torch.tensor([bool(text) for text in batch.report_texts], device=organ_image_embeddings.device, dtype=torch.bool)
        report_text_features = (
            self.text_encoder.encode_texts(batch.report_texts, report_text_mask, max_tokens=self.config.text_encoder.report_max_tokens)
            if self.enable_report_clip
            else organ_image_embeddings.new_zeros((organ_image_embeddings.shape[0], organ_image_embeddings.shape[-1]))
        )
        report_text_embeddings = self.report_text_projection(report_text_features) if self.enable_report_clip else report_text_features
        diagnostic_logits = self.diagnostic_head(organ_image_features).squeeze(-1)
        lesion_global_logits = (
            self.lesion_global_head(report_image_features).squeeze(-1)
            if self.enable_lesion_global
            else organ_image_features.new_zeros((organ_image_features.shape[0],))
        )
        lesion_organ_logits = (
            self.lesion_organ_head(organ_image_features).squeeze(-1)
            if self.enable_lesion_organ
            else organ_image_features.new_zeros(organ_image_features.shape[:2])
        )
        patch_count = max(segmentation_patch_count, 1)
        if patch_organ_count > 0:
            patch_organ_presence_loss = patch_organ_loss_sum / patch_organ_count
            patch_organ_presence_accuracy = patch_organ_correct_sum / patch_organ_count
        else:
            patch_organ_presence_loss = self.patch_organ_presence_head.weight.sum() * 0.0
            patch_organ_presence_accuracy = 0.0
        return OrganSegOutput(
            organ_image_embeddings=organ_image_embeddings,
            organ_text_embeddings=organ_text_embeddings,
            report_image_embeddings=report_image_embeddings,
            report_text_embeddings=report_text_embeddings,
            diagnostic_logits=diagnostic_logits,
            lesion_global_logits=lesion_global_logits,
            lesion_organ_logits=lesion_organ_logits,
            logit_scale=self.logit_scale.exp().clamp(max=100.0),
            organ_logit_scale=self.organ_logit_scale.clamp(min=0.0, max=math.log(100.0)).exp(),
            organ_logit_bias=self.organ_logit_bias.clamp(min=-20.0, max=20.0),
            report_logit_scale=self.report_logit_scale.clamp(min=0.0, max=math.log(100.0)).exp(),
            report_logit_bias=self.report_logit_bias.clamp(min=-20.0, max=20.0),
            segmentation_loss=segmentation_loss_sum / patch_count,
            segmentation_dice=segmentation_dice_sum / patch_count,
            segmentation_patch_count=segmentation_patch_count,
            patch_organ_presence_loss=patch_organ_presence_loss,
            patch_organ_presence_accuracy=float(patch_organ_presence_accuracy),
            patch_organ_presence_count=patch_organ_count,
            organ_attention_loss=organ_attention_loss,
            organ_attention_accuracy=float(organ_attention_accuracy),
            organ_attention_positive_accuracy=float(organ_attention_positive_accuracy),
            organ_attention_negative_accuracy=float(organ_attention_negative_accuracy),
            organ_attention_count=organ_attention_count,
            organ_attention_positive_count=organ_attention_positive_count,
            organ_attention_negative_count=organ_attention_negative_count,
            patches_per_batch_total=patches_per_batch_total,
            patches_per_study_mean=float(patches_per_study_mean),
            patches_per_study_max=patches_per_study_max,
            segmentation_oom_fallback_count=segmentation_oom_fallback_count,
        )

    @torch.no_grad()
    def clamp_alignment_parameters(self) -> None:
        self.organ_logit_scale.clamp_(min=0.0, max=math.log(100.0))
        self.report_logit_scale.clamp_(min=0.0, max=math.log(100.0))
        self.organ_logit_bias.clamp_(min=-20.0, max=20.0)
        self.report_logit_bias.clamp_(min=-20.0, max=20.0)

    def _prepare_study(self, batch: EncoderBatch, sample_index: int) -> _PreparedStudy:
        image = batch.images[sample_index]
        image_mask = batch.image_mask[sample_index, 0]
        bounds = mask_bounds(image_mask)
        cropped_image = crop_to_bounds(image, bounds)
        cropped_segmentation = None if batch.segmentations is None else crop_to_bounds(batch.segmentations[sample_index], bounds)
        cropped_segmentation_mask = None if batch.segmentation_mask is None else crop_to_bounds(batch.segmentation_mask[sample_index], bounds)
        spatial_shape = tuple(int(v) for v in cropped_image.shape[-3:])
        boxes = generate_tile_boxes(spatial_shape, self.patch_size, self.patch_stride)
        supervised_patch_indices = _sample_segmentation_supervision_indices(
            box_count=len(boxes),
            max_patches=self.segmentation_supervision_max_patches_per_study,
            training=self.training,
            device=image.device,
        )
        return _PreparedStudy(
            cropped_image=cropped_image,
            cropped_segmentation=cropped_segmentation,
            cropped_segmentation_mask=cropped_segmentation_mask,
            spatial_shape=spatial_shape,
            boxes=boxes,
            supervised_patch_indices=supervised_patch_indices,
        )

    def _encode_studies_batched(
        self,
        studies: list[_PreparedStudy],
    ) -> list[tuple[torch.Tensor, torch.Tensor, float, int, torch.Tensor, float, int, torch.Tensor, torch.Tensor, int]]:
        if not studies:
            return []
        study_count = len(studies)
        token_chunks: list[list[torch.Tensor]] = [[] for _ in range(study_count)]
        patch_summary_chunks: list[list[torch.Tensor]] = [[] for _ in range(study_count)]
        position_chunks: list[list[torch.Tensor]] = [[] for _ in range(study_count)]
        attention_target_chunks: list[list[torch.Tensor]] = [[] for _ in range(study_count)]
        attention_mask_chunks: list[list[torch.Tensor]] = [[] for _ in range(study_count)]
        loss_sums = [self.logit_scale.new_zeros(()) for _ in range(study_count)]
        dice_sums = [0.0 for _ in range(study_count)]
        patch_counts = [0 for _ in range(study_count)]
        patch_organ_loss_sums = [self.logit_scale.new_zeros(()) for _ in range(study_count)]
        patch_organ_correct_sums = [0.0 for _ in range(study_count)]
        patch_organ_counts = [0 for _ in range(study_count)]
        segmentation_oom_fallback_counts = [0 for _ in range(study_count)]

        patch_refs: list[tuple[int, int]] = []
        for study_index, study in enumerate(studies):
            patch_refs.extend(
                (study_index, box_index)
                for box_index in _ordered_patch_indices_for_encoding(
                    box_count=len(study.boxes),
                    supervised_patch_indices=study.supervised_patch_indices,
                    training=self.training,
                )
            )

        default_patch_batch_size = max(int(self.patch_batch_size), 1)
        total_chunks = (len(patch_refs) + default_patch_batch_size - 1) // default_patch_batch_size
        active_debug_step = os.environ.get("ORGAN_SEG_CLIP_ACTIVE_STEP", "").strip()
        chunk_index = 1
        start = 0
        while start < len(patch_refs):
            remaining = len(patch_refs) - start
            requested_chunk_size = min(default_patch_batch_size, remaining)
            chunk_size = _adaptive_patch_encoder_chunk_size(
                device=self.logit_scale.device,
                requested_chunk_size=requested_chunk_size,
                default_chunk_size=default_patch_batch_size,
                min_free_bytes=self.patch_encoder_min_free_bytes,
            )
            chunk_refs = patch_refs[start:start + chunk_size]
            deferred_segmentation_work: list[_DeferredSegmentationWork] = []
            debug_counts_by_study: dict[int, int] = {}
            if active_debug_step:
                for study_index, _ in chunk_refs:
                    debug_counts_by_study[study_index] = debug_counts_by_study.get(study_index, 0) + 1
                if chunk_size < requested_chunk_size:
                    free_gb = _cuda_free_memory_gb(self.logit_scale.device)
                    print(
                        f"[debug step {active_debug_step}] chunk={chunk_index}/{total_chunks}"
                        f" shrinking_patch_encoder_tiles={requested_chunk_size}->{chunk_size}"
                        f" free_gb={free_gb:.2f}",
                        flush=True,
                    )
            chunk_device = self.logit_scale.device
            while True:
                chunk_tile_count = int(len(chunk_refs))
                try:
                    image_tiles = torch.stack(
                        [
                            extract_tile(studies[study_index].cropped_image, studies[study_index].boxes[box_index], self.patch_size)
                            for study_index, box_index in chunk_refs
                        ],
                        dim=0,
                    )
                    chunk_device = image_tiles.device
                    chunk_tile_count = int(image_tiles.shape[0])
                    if active_debug_step:
                        _debug_chunk_memory(
                            step=active_debug_step,
                            phase="before_patch_encoder",
                            chunk_index=chunk_index,
                            total_chunks=total_chunks,
                            device=chunk_device,
                            tile_count=chunk_tile_count,
                            study_counts=debug_counts_by_study,
                        )
                    feature_pyramid = self.patch_encoder(image_tiles)
                    break
                except torch.OutOfMemoryError:
                    if active_debug_step:
                        _debug_chunk_memory(
                            step=active_debug_step,
                            phase="patch_encoder_oom",
                            chunk_index=chunk_index,
                            total_chunks=total_chunks,
                            device=chunk_device,
                            tile_count=chunk_tile_count,
                            study_counts=debug_counts_by_study,
                        )
                    if "image_tiles" in locals():
                        del image_tiles
                    if chunk_device.type == "cuda":
                        torch.cuda.empty_cache()
                    if chunk_tile_count <= 1:
                        raise
                    chunk_size = max(1, chunk_tile_count // 2)
                    chunk_refs = patch_refs[start:start + chunk_size]
                    debug_counts_by_study = {}
                    if active_debug_step:
                        for study_index, _ in chunk_refs:
                            debug_counts_by_study[study_index] = debug_counts_by_study.get(study_index, 0) + 1
                        print(
                            f"[debug step {active_debug_step}] chunk={chunk_index}/{total_chunks}"
                            f" retrying_patch_encoder_with_tiles={chunk_size}",
                            flush=True,
                        )
                    continue
            if active_debug_step:
                _debug_chunk_memory(
                    step=active_debug_step,
                    phase="after_patch_encoder",
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    device=chunk_device,
                    tile_count=chunk_tile_count,
                    study_counts=debug_counts_by_study,
                )
            patch_tokens = self.patch_tokenizer(feature_pyramid[-1])
            patch_summaries = self.patch_summary_head(patch_tokens) if self.use_grid_combiner else None

            local_indices_by_study: dict[int, list[int]] = {}
            box_indices_by_study: dict[int, list[int]] = {}
            for local_index, (study_index, box_index) in enumerate(chunk_refs):
                local_indices_by_study.setdefault(study_index, []).append(local_index)
                box_indices_by_study.setdefault(study_index, []).append(box_index)

            for study_index, local_indices_list in local_indices_by_study.items():
                study = studies[study_index]
                local_indices = torch.tensor(local_indices_list, device=image_tiles.device, dtype=torch.long)
                box_indices = box_indices_by_study[study_index]
                study_boxes = [study.boxes[box_index] for box_index in box_indices]
                positions = torch.stack(
                    [normalized_box_features(box, study.spatial_shape, device=image_tiles.device) for box in study_boxes],
                    dim=0,
                )
                study_patch_tokens = patch_tokens.index_select(0, local_indices)

                segmentation_tiles = None
                segmentation_mask_tiles = None
                if study.cropped_segmentation is not None:
                    segmentation_tiles = torch.stack(
                        [extract_tile(study.cropped_segmentation, box, self.patch_size) for box in study_boxes],
                        dim=0,
                    )
                    if study.cropped_segmentation_mask is not None:
                        segmentation_mask_tiles = torch.stack(
                            [extract_tile(study.cropped_segmentation_mask, box, self.patch_size).bool() for box in study_boxes],
                            dim=0,
                        )
                    supervised_mask = _chunk_supervision_mask(
                        box_indices=box_indices,
                        supervised_patch_indices=study.supervised_patch_indices,
                        device=image_tiles.device,
                    )
                    should_run_segmentation_supervision = not (
                        not self.training and self.skip_segmentation_supervision_in_eval
                    )
                    if should_run_segmentation_supervision and supervised_mask.any():
                        if active_debug_step:
                            print(
                                f"[debug step {active_debug_step}] chunk={chunk_index}/{total_chunks}"
                                f" study={study_index} supervised_tiles={int(supervised_mask.sum().item())}",
                                flush=True,
                            )
                        supervised_indices = local_indices[supervised_mask]
                        supervised_image_tiles = image_tiles.index_select(0, supervised_indices)
                        supervised_segmentation_tiles = segmentation_tiles[supervised_mask]
                        supervised_segmentation_mask_tiles = (
                            None if segmentation_mask_tiles is None else segmentation_mask_tiles[supervised_mask]
                        )
                        if _should_defer_segmentation_recompute(
                            device=image_tiles.device,
                            min_free_bytes=self.segmentation_recompute_min_free_bytes,
                        ):
                            if active_debug_step:
                                free_gb = _cuda_free_memory_gb(image_tiles.device)
                                threshold_gb = self.segmentation_recompute_min_free_bytes / float(1024 ** 3)
                                print(
                                    f"[debug step {active_debug_step}] chunk={chunk_index}/{total_chunks}"
                                    f" study={study_index} deferring_segmentation_recompute"
                                    f" free_gb={free_gb:.2f} threshold_gb={threshold_gb:.2f}",
                                    flush=True,
                                )
                            deferred_segmentation_work.append(
                                _DeferredSegmentationWork(
                                    study_index=study_index,
                                    supervised_image_tiles=supervised_image_tiles,
                                    supervised_segmentation_tiles=supervised_segmentation_tiles,
                                    supervised_segmentation_mask_tiles=supervised_segmentation_mask_tiles,
                                )
                            )
                        else:
                            supervised_feature_pyramid = tuple(
                                features.index_select(0, supervised_indices) for features in feature_pyramid
                            )
                            try:
                                chunk_loss_sum, chunk_dice_sum, supervised_count, fallback_count = _segmentation_supervision_with_oom_fallback(
                                    patch_segmentation_head=self.patch_segmentation_head,
                                    supervised_image_tiles=supervised_image_tiles,
                                    supervised_feature_pyramid=supervised_feature_pyramid,
                                    supervised_segmentation_tiles=supervised_segmentation_tiles,
                                    supervised_segmentation_mask_tiles=supervised_segmentation_mask_tiles,
                                    loss_type=self.segmentation_loss_type,
                                    debug_step=active_debug_step,
                                    debug_phase=f"direct_segmentation chunk={chunk_index}/{total_chunks} study={study_index}",
                                )
                            except torch.OutOfMemoryError:
                                if active_debug_step:
                                    print(
                                        f"[debug step {active_debug_step}] chunk={chunk_index}/{total_chunks}"
                                        f" study={study_index} direct_segmentation_oom_deferring_recompute",
                                        flush=True,
                                    )
                                del supervised_feature_pyramid
                                if image_tiles.device.type == "cuda":
                                    torch.cuda.empty_cache()
                                deferred_segmentation_work.append(
                                    _DeferredSegmentationWork(
                                        study_index=study_index,
                                        supervised_image_tiles=supervised_image_tiles,
                                        supervised_segmentation_tiles=supervised_segmentation_tiles,
                                        supervised_segmentation_mask_tiles=supervised_segmentation_mask_tiles,
                                    )
                                )
                            else:
                                loss_sums[study_index] = loss_sums[study_index] + chunk_loss_sum
                                dice_sums[study_index] += chunk_dice_sum
                                patch_counts[study_index] += supervised_count
                                segmentation_oom_fallback_counts[study_index] += fallback_count

                patch_targets = None
                patch_target_mask = None
                if (self.enable_patch_organ_presence or self.enable_organ_attention) and segmentation_tiles is not None:
                    patch_targets, patch_target_mask = _patch_organ_presence_targets(
                        segmentation_tiles,
                        segmentation_mask_tiles,
                        organ_label_values=self.organ_mask_label_values,
                        enabled_organs=self.organ_patch_target_mask,
                        min_voxels=self.patch_organ_min_voxels,
                    )
                if self.enable_organ_attention and patch_targets is not None and patch_target_mask is not None:
                    attention_target_chunks[study_index].append(patch_targets)
                    attention_mask_chunks[study_index].append(patch_target_mask)
                if self.enable_patch_organ_presence and patch_targets is not None and patch_target_mask is not None:
                    patch_logits = self.patch_organ_presence_head(study_patch_tokens.mean(dim=1))
                    if patch_target_mask.any():
                        masked_logits = patch_logits[patch_target_mask]
                        masked_targets = patch_targets[patch_target_mask]
                        chunk_patch_loss = F.binary_cross_entropy_with_logits(masked_logits, masked_targets)
                        chunk_count = int(patch_target_mask.sum().item())
                        predictions = (masked_logits.detach().sigmoid() >= 0.5).float()
                        patch_organ_correct_sums[study_index] += float((predictions == masked_targets).float().sum().item())
                        patch_organ_loss_sums[study_index] = patch_organ_loss_sums[study_index] + chunk_patch_loss * chunk_count
                        patch_organ_counts[study_index] += chunk_count
                if self.use_grid_combiner:
                    assert patch_summaries is not None
                    patch_summary_chunks[study_index].append(patch_summaries.index_select(0, local_indices))
                    position_chunks[study_index].append(
                        _grid_position_features(
                            study_boxes,
                            study.spatial_shape,
                            reference_boxes=study.boxes,
                            device=image_tiles.device,
                        )
                    )
                else:
                    chunk_tokens = study_patch_tokens + self.patch_position_embedding(positions).unsqueeze(1)
                    token_chunks[study_index].append(chunk_tokens.reshape(-1, chunk_tokens.shape[-1]))
            if deferred_segmentation_work:
                del feature_pyramid
                del patch_tokens
                if patch_summaries is not None:
                    del patch_summaries
                del image_tiles
                if chunk_device.type == "cuda":
                    torch.cuda.empty_cache()
                if active_debug_step:
                    _debug_chunk_memory(
                        step=active_debug_step,
                        phase="before_deferred_segmentation_recompute",
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        device=chunk_device,
                        tile_count=sum(int(work.supervised_image_tiles.shape[0]) for work in deferred_segmentation_work),
                        study_counts=debug_counts_by_study,
                    )
                for work in deferred_segmentation_work:
                    recomputed_feature_pyramid = self.patch_encoder(work.supervised_image_tiles)
                    chunk_loss_sum, chunk_dice_sum, supervised_count, fallback_count = _segmentation_supervision_with_oom_fallback(
                        patch_segmentation_head=self.patch_segmentation_head,
                        supervised_image_tiles=work.supervised_image_tiles,
                        supervised_feature_pyramid=recomputed_feature_pyramid,
                        supervised_segmentation_tiles=work.supervised_segmentation_tiles,
                        supervised_segmentation_mask_tiles=work.supervised_segmentation_mask_tiles,
                        loss_type=self.segmentation_loss_type,
                        debug_step=active_debug_step,
                        debug_phase=f"deferred_segmentation chunk={chunk_index}/{total_chunks} study={work.study_index}",
                    )
                    loss_sums[work.study_index] = loss_sums[work.study_index] + chunk_loss_sum
                    dice_sums[work.study_index] += chunk_dice_sum
                    patch_counts[work.study_index] += supervised_count
                    segmentation_oom_fallback_counts[work.study_index] += fallback_count + 1
                    del recomputed_feature_pyramid
                if active_debug_step:
                    _debug_chunk_memory(
                        step=active_debug_step,
                        phase="after_deferred_segmentation_recompute",
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        device=chunk_device,
                        tile_count=sum(int(work.supervised_image_tiles.shape[0]) for work in deferred_segmentation_work),
                        study_counts=debug_counts_by_study,
                    )
            if active_debug_step:
                _debug_chunk_memory(
                    step=active_debug_step,
                    phase="end_chunk",
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    device=chunk_device,
                    tile_count=chunk_tile_count,
                    study_counts=debug_counts_by_study,
                )
            start += chunk_tile_count
            chunk_index += 1

        encoded_studies: list[tuple[torch.Tensor, torch.Tensor, float, int, torch.Tensor, float, int, torch.Tensor, torch.Tensor, int]] = []
        for study_index, study in enumerate(studies):
            if attention_target_chunks[study_index]:
                attention_targets = torch.cat(attention_target_chunks[study_index], dim=0)
                attention_mask = torch.cat(attention_mask_chunks[study_index], dim=0)
            else:
                attention_targets = self.logit_scale.new_zeros((0, len(self.organ_mask_label_values)))
                attention_mask = torch.zeros(
                    (0, len(self.organ_mask_label_values)),
                    device=self.logit_scale.device,
                    dtype=torch.bool,
                )
            if self.use_grid_combiner:
                patch_summaries = torch.cat(patch_summary_chunks[study_index], dim=0).unsqueeze(0)
                grid_positions = torch.cat(position_chunks[study_index], dim=0).unsqueeze(0)
                grid_mask = torch.ones((1, patch_summaries.shape[1]), device=patch_summaries.device, dtype=torch.bool)
                study_tokens = self.grid_combiner(patch_summaries, grid_positions, grid_mask).squeeze(0)
                if self.grid_combiner.use_global_token:
                    attention_targets, attention_mask = _prepend_empty_attention_target(attention_targets, attention_mask)
            else:
                study_tokens = torch.cat(token_chunks[study_index], dim=0)
                attention_targets, attention_mask = _repeat_attention_targets_for_flat_tokens(
                    attention_targets,
                    attention_mask,
                    repeats=self.patch_tokenizer.token_count,
                )
            patch_organ_loss = patch_organ_loss_sums[study_index] / max(patch_organ_counts[study_index], 1)
            patch_organ_accuracy = (
                0.0 if patch_organ_counts[study_index] == 0 else patch_organ_correct_sums[study_index] / patch_organ_counts[study_index]
            )
            encoded_studies.append(
                (
                    study_tokens,
                    loss_sums[study_index] / max(patch_counts[study_index], 1),
                    0.0 if patch_counts[study_index] == 0 else dice_sums[study_index] / patch_counts[study_index],
                    patch_counts[study_index],
                    patch_organ_loss,
                    patch_organ_accuracy,
                    patch_organ_counts[study_index],
                    attention_targets,
                    attention_mask,
                    segmentation_oom_fallback_counts[study_index],
                )
            )
        return encoded_studies

    def _encode_single_study(
        self,
        batch: EncoderBatch,
        sample_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float, int, torch.Tensor, float, int, torch.Tensor, torch.Tensor, int]:
        prepared_study = self._prepare_study(batch, sample_index)
        return self._encode_studies_batched([prepared_study])[0]

    def set_eval_segmentation_supervision(self, enabled: bool) -> None:
        self.skip_segmentation_supervision_in_eval = not bool(enabled)



def _balanced_binary_cross_entropy_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive_mask = targets >= 0.5
    negative_mask = ~positive_mask
    losses: list[torch.Tensor] = []
    if positive_mask.any():
        losses.append(F.binary_cross_entropy_with_logits(logits[positive_mask], targets[positive_mask]))
    if negative_mask.any():
        losses.append(F.binary_cross_entropy_with_logits(logits[negative_mask], targets[negative_mask]))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _segmentation_supervision_with_oom_fallback(
    *,
    patch_segmentation_head: nn.Module,
    supervised_image_tiles: torch.Tensor,
    supervised_feature_pyramid: tuple[torch.Tensor, ...],
    supervised_segmentation_tiles: torch.Tensor,
    supervised_segmentation_mask_tiles: torch.Tensor | None,
    loss_type: str,
    debug_step: str = "",
    debug_phase: str = "",
) -> tuple[torch.Tensor, float, int, int]:
    supervised_count = int(supervised_image_tiles.shape[0])
    if supervised_count == 0:
        zero = supervised_image_tiles.sum() * 0.0
        return zero, 0.0, 0, 0

    ce_numerator_sum = supervised_image_tiles.sum() * 0.0
    ce_denominator_sum = supervised_image_tiles.sum() * 0.0
    dice_entry_sum = supervised_image_tiles.sum() * 0.0
    dice_entry_count = 0
    dice_intersection_sums: torch.Tensor | None = None
    dice_prediction_sums: torch.Tensor | None = None
    dice_target_sums: torch.Tensor | None = None
    fallback_count = 0

    for sample_index in range(supervised_count):
        sample_slice = slice(sample_index, sample_index + 1)
        sample_image_tiles = supervised_image_tiles[sample_slice]
        sample_feature_pyramid = tuple(features[sample_slice] for features in supervised_feature_pyramid)
        sample_segmentation_tiles = supervised_segmentation_tiles[sample_slice]
        sample_segmentation_mask_tiles = (
            None
            if supervised_segmentation_mask_tiles is None
            else supervised_segmentation_mask_tiles[sample_slice]
        )
        if debug_step:
            _debug_segmentation_memory(
                step=debug_step,
                phase=f"{debug_phase} before_seg_head sample={sample_index + 1}/{supervised_count}",
                device=sample_image_tiles.device,
                image_tiles=sample_image_tiles,
                feature_pyramid=sample_feature_pyramid,
            )
        try:
            logits = patch_segmentation_head(sample_image_tiles, sample_feature_pyramid)
        except torch.OutOfMemoryError:
            fallback_count += 1
            if sample_image_tiles.device.type == "cuda":
                torch.cuda.empty_cache()
            if debug_step:
                print(
                    f"[debug step {debug_step}] phase={debug_phase}"
                    f" sample={sample_index + 1}/{supervised_count}"
                    " seg_head_oom_retry_checkpoint",
                    flush=True,
                )
            logits = _checkpointed_segmentation_head(
                patch_segmentation_head,
                sample_image_tiles,
                sample_feature_pyramid,
            )
        if debug_step:
            _debug_segmentation_memory(
                step=debug_step,
                phase=f"{debug_phase} after_seg_head sample={sample_index + 1}/{supervised_count}",
                device=sample_image_tiles.device,
                image_tiles=sample_image_tiles,
                feature_pyramid=sample_feature_pyramid,
                logits=logits,
            )
        _, sample_ce_numerator, sample_ce_denominator, sample_dice_sum, sample_dice_count = (
            segmentation_supervision_loss(
                logits,
                sample_segmentation_tiles,
                sample_segmentation_mask_tiles,
                loss_type=loss_type,
                return_components=True,
            )
        )
        ce_numerator_sum = ce_numerator_sum + sample_ce_numerator
        ce_denominator_sum = ce_denominator_sum + sample_ce_denominator
        dice_entry_sum = dice_entry_sum + sample_dice_sum
        dice_entry_count += sample_dice_count
        sample_intersection_sums, sample_prediction_sums, sample_target_sums = multiclass_dice_score(
            logits.detach(),
            sample_segmentation_tiles,
            sample_segmentation_mask_tiles,
            return_components=True,
        )
        if dice_intersection_sums is None:
            dice_intersection_sums = torch.zeros_like(sample_intersection_sums)
            dice_prediction_sums = torch.zeros_like(sample_prediction_sums)
            dice_target_sums = torch.zeros_like(sample_target_sums)
        dice_intersection_sums += sample_intersection_sums
        dice_prediction_sums += sample_prediction_sums
        dice_target_sums += sample_target_sums
        if debug_step:
            _debug_segmentation_memory(
                step=debug_step,
                phase=f"{debug_phase} after_seg_loss sample={sample_index + 1}/{supervised_count}",
                device=sample_image_tiles.device,
                image_tiles=sample_image_tiles,
                feature_pyramid=sample_feature_pyramid,
                logits=logits,
            )

    if loss_type == "ce":
        chunk_loss = ce_numerator_sum / ce_denominator_sum.clamp_min(1.0)
    else:
        ce = ce_numerator_sum / ce_denominator_sum.clamp_min(1.0)
        dice_loss = 1.0 - (dice_entry_sum / max(dice_entry_count, 1))
        chunk_loss = ce + dice_loss
    if (
        dice_intersection_sums is not None
        and dice_prediction_sums is not None
        and dice_target_sums is not None
        and ((dice_prediction_sums[1:] > 0) | (dice_target_sums[1:] > 0)).any()
    ):
        valid_class_mask = (dice_prediction_sums[1:] > 0) | (dice_target_sums[1:] > 0)
        class_scores = (
            (2.0 * dice_intersection_sums[1:] + 1.0)
            / (dice_prediction_sums[1:] + dice_target_sums[1:] + 1.0)
        )[valid_class_mask]
        chunk_dice = float(class_scores.mean().item())
    else:
        chunk_dice = 1.0
    return chunk_loss * supervised_count, chunk_dice * supervised_count, supervised_count, fallback_count


def _checkpointed_segmentation_head(
    patch_segmentation_head: nn.Module,
    image_tiles: torch.Tensor,
    feature_pyramid: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    def _forward(image: torch.Tensor, *features: torch.Tensor) -> torch.Tensor:
        return patch_segmentation_head(image, tuple(features))

    return activation_checkpoint(
        _forward,
        image_tiles,
        *feature_pyramid,
        use_reentrant=False,
        preserve_rng_state=False,
    )


def _segmentation_recompute_min_free_bytes_from_env() -> int:
    raw = os.environ.get("ORGAN_SEG_CLIP_SEGMENTATION_RECOMPUTE_MIN_FREE_GB", "").strip()
    if not raw:
        return int(2.0 * (1024 ** 3))
    return max(int(float(raw) * (1024 ** 3)), 0)


def _patch_encoder_min_free_bytes_from_env() -> int:
    raw = os.environ.get("ORGAN_SEG_CLIP_PATCH_ENCODER_MIN_FREE_GB", "").strip()
    if not raw:
        return int(8.0 * (1024 ** 3))
    return max(int(float(raw) * (1024 ** 3)), 0)


def _adaptive_patch_encoder_chunk_size(
    *,
    device: torch.device,
    requested_chunk_size: int,
    default_chunk_size: int,
    min_free_bytes: int,
) -> int:
    chunk_size = max(int(requested_chunk_size), 1)
    if device.type != "cuda" or min_free_bytes <= 0 or chunk_size <= 1:
        return chunk_size
    free_bytes = _cuda_free_memory_bytes(device)
    if free_bytes <= 0:
        return chunk_size
    while chunk_size > 1:
        scaled_min_free_bytes = int(min_free_bytes * (chunk_size / max(float(default_chunk_size), 1.0)))
        if free_bytes >= scaled_min_free_bytes:
            break
        chunk_size = max(1, chunk_size // 2)
    return chunk_size


def _should_defer_segmentation_recompute(*, device: torch.device, min_free_bytes: int) -> bool:
    if device.type != "cuda" or min_free_bytes <= 0:
        return False
    free_bytes = _cuda_free_memory_bytes(device)
    return 0 < free_bytes < int(min_free_bytes)


def _cuda_free_memory_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return int(free_bytes)


def _cuda_free_memory_gb(device: torch.device) -> float:
    return float(_cuda_free_memory_bytes(device) / float(1024 ** 3))


def _sample_segmentation_supervision_indices(
    *,
    box_count: int,
    max_patches: int,
    training: bool,
    device: torch.device,
) -> torch.Tensor | None:
    if not training or max_patches <= 0 or box_count <= max_patches:
        return None
    return torch.randperm(int(box_count), device=device)[: int(max_patches)]


def _ordered_patch_indices_for_encoding(
    *,
    box_count: int,
    supervised_patch_indices: torch.Tensor | None,
    training: bool,
) -> list[int]:
    if not training or supervised_patch_indices is None or int(supervised_patch_indices.numel()) == 0:
        return list(range(int(box_count)))
    supervised_order: list[int] = []
    supervised_set: set[int] = set()
    for raw_index in supervised_patch_indices.detach().cpu().tolist():
        box_index = int(raw_index)
        if 0 <= box_index < int(box_count) and box_index not in supervised_set:
            supervised_order.append(box_index)
            supervised_set.add(box_index)
    if not supervised_order:
        return list(range(int(box_count)))
    return supervised_order + [box_index for box_index in range(int(box_count)) if box_index not in supervised_set]


def _chunk_supervision_mask(
    *,
    box_indices: list[int],
    supervised_patch_indices: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if supervised_patch_indices is None:
        return torch.ones((len(box_indices),), device=device, dtype=torch.bool)
    chunk_indices = torch.tensor([int(index) for index in box_indices], device=device, dtype=torch.long)
    return (chunk_indices.unsqueeze(1) == supervised_patch_indices.unsqueeze(0)).any(dim=1)

def _prepend_empty_attention_target(targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    empty_target = targets.new_zeros((1, targets.shape[-1]))
    empty_mask = torch.zeros((1, mask.shape[-1]), device=mask.device, dtype=torch.bool)
    return torch.cat([empty_target, targets], dim=0), torch.cat([empty_mask, mask], dim=0)


def _repeat_attention_targets_for_flat_tokens(targets: torch.Tensor, mask: torch.Tensor, *, repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
    if targets.shape[0] == 0:
        return targets, mask
    repeat_count = max(int(repeats), 1)
    return targets.repeat_interleave(repeat_count, dim=0), mask.repeat_interleave(repeat_count, dim=0)


def _debug_chunk_memory(
    *,
    step: str,
    phase: str,
    chunk_index: int,
    total_chunks: int,
    device: torch.device,
    tile_count: int,
    study_counts: dict[int, int],
) -> None:
    if device.type == "cuda":
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
        max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    else:
        allocated_gb = 0.0
        reserved_gb = 0.0
        max_allocated_gb = 0.0
    counts_text = ",".join(f"{study}:{count}" for study, count in sorted(study_counts.items()))
    print(
        f"[debug step {step}] phase={phase} chunk={chunk_index}/{total_chunks}"
        f" tiles={tile_count} study_patch_counts={counts_text}"
        f" alloc_gb={allocated_gb:.2f} reserved_gb={reserved_gb:.2f} max_alloc_gb={max_allocated_gb:.2f}",
        flush=True,
    )


def _debug_segmentation_memory(
    *,
    step: str,
    phase: str,
    device: torch.device,
    image_tiles: torch.Tensor,
    feature_pyramid: tuple[torch.Tensor, ...],
    logits: torch.Tensor | None = None,
) -> None:
    if device.type == "cuda":
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
        max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        free_gb = _cuda_free_memory_gb(device)
    else:
        allocated_gb = 0.0
        reserved_gb = 0.0
        max_allocated_gb = 0.0
        free_gb = 0.0
    feature_shapes = ",".join("x".join(str(int(dim)) for dim in feature.shape) for feature in feature_pyramid)
    logits_shape = "none" if logits is None else "x".join(str(int(dim)) for dim in logits.shape)
    image_shape = "x".join(str(int(dim)) for dim in image_tiles.shape)
    print(
        f"[debug step {step}] phase={phase}"
        f" image_shape={image_shape} feature_shapes={feature_shapes} logits_shape={logits_shape}"
        f" alloc_gb={allocated_gb:.2f} reserved_gb={reserved_gb:.2f}"
        f" max_alloc_gb={max_allocated_gb:.2f} free_gb={free_gb:.2f}",
        flush=True,
    )


def _pad_organ_attention_targets(
    target_sequences: list[torch.Tensor],
    mask_sequences: list[torch.Tensor],
    *,
    max_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not target_sequences:
        return torch.zeros((0, 0, 0), device=device), torch.zeros((0, 0, 0), device=device, dtype=torch.bool)
    organ_count = target_sequences[0].shape[-1]
    targets = torch.zeros((len(target_sequences), int(max_tokens), organ_count), device=device, dtype=target_sequences[0].dtype)
    masks = torch.zeros((len(mask_sequences), int(max_tokens), organ_count), device=device, dtype=torch.bool)
    for index, (sample_targets, sample_mask) in enumerate(zip(target_sequences, mask_sequences)):
        token_count = min(int(sample_targets.shape[0]), int(max_tokens))
        if token_count <= 0:
            continue
        targets[index, :token_count] = sample_targets[:token_count].to(device=device)
        masks[index, :token_count] = sample_mask[:token_count].to(device=device)
    return targets, masks


def _grid_position_features(
    boxes: list[tuple[int, int, int, int, int, int]],
    spatial_shape: tuple[int, int, int],
    *,
    reference_boxes: list[tuple[int, int, int, int, int, int]] | None = None,
    device: torch.device,
) -> torch.Tensor:
    grid_boxes = boxes if reference_boxes is None else reference_boxes
    starts_d = sorted({int(box[0]) for box in grid_boxes})
    starts_h = sorted({int(box[2]) for box in grid_boxes})
    starts_w = sorted({int(box[4]) for box in grid_boxes})
    d_lookup = {value: index for index, value in enumerate(starts_d)}
    h_lookup = {value: index for index, value in enumerate(starts_h)}
    w_lookup = {value: index for index, value in enumerate(starts_w)}
    coords: list[torch.Tensor] = []
    for box in boxes:
        d0, _, h0, _, w0, _ = box
        grid_d = _normalized_index(d_lookup[int(d0)], len(starts_d))
        grid_h = _normalized_index(h_lookup[int(h0)], len(starts_h))
        grid_w = _normalized_index(w_lookup[int(w0)], len(starts_w))
        box_features = normalized_box_features(box, spatial_shape, device=device)
        grid_features = torch.tensor([grid_d, grid_h, grid_w], device=device, dtype=torch.float32)
        coords.append(torch.cat([grid_features, box_features], dim=0))
    return torch.stack(coords, dim=0)


def _normalized_index(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    return float(index) / float(count - 1)


def _patch_organ_presence_targets(
    segmentation_tiles: torch.Tensor,
    segmentation_mask_tiles: torch.Tensor | None,
    *,
    organ_label_values: tuple[tuple[int, ...], ...],
    enabled_organs: tuple[bool, ...],
    min_voxels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = segmentation_tiles.shape[0]
    organ_count = len(organ_label_values)
    targets = segmentation_tiles.new_zeros((batch_size, organ_count), dtype=torch.float32)
    target_mask = torch.zeros((batch_size, organ_count), device=segmentation_tiles.device, dtype=torch.bool)
    valid_mask = segmentation_mask_tiles.bool() if segmentation_mask_tiles is not None else torch.ones_like(segmentation_tiles, dtype=torch.bool)
    threshold = max(int(min_voxels), 1)
    for organ_index, label_values in enumerate(organ_label_values):
        if not enabled_organs[organ_index] or not label_values:
            continue
        organ_voxels = torch.zeros_like(valid_mask, dtype=torch.bool)
        for label_value in label_values:
            organ_voxels |= segmentation_tiles == int(label_value)
        counts = (organ_voxels & valid_mask).flatten(start_dim=1).sum(dim=1)
        targets[:, organ_index] = (counts >= threshold).float()
        target_mask[:, organ_index] = True
    return targets, target_mask


def _pad_token_sequences(token_sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_tokens = max(sequence.shape[0] for sequence in token_sequences)
    dim = token_sequences[0].shape[-1]
    padded = token_sequences[0].new_zeros((len(token_sequences), max_tokens, dim))
    mask = torch.zeros((len(token_sequences), max_tokens), device=token_sequences[0].device, dtype=torch.bool)
    for index, sequence in enumerate(token_sequences):
        padded[index, : sequence.shape[0]] = sequence
        mask[index, : sequence.shape[0]] = True
    return padded, mask


def _set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad
