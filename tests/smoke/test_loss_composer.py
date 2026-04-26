from __future__ import annotations

import torch

from organ_seg_clip.models.interfaces.types import EncoderBatch, OrganSegOutput
from organ_seg_clip.models.losses.composer import OrganSegLossComposer
from organ_seg_clip.config.schemas import LossConfig


def test_loss_composer_returns_finite_values() -> None:
    composer = OrganSegLossComposer(LossConfig())
    batch = EncoderBatch(
        study_ids=["study-1"],
        images=torch.zeros((1, 1, 8, 8, 8)),
        image_mask=torch.ones((1, 1, 8, 8, 8), dtype=torch.bool),
        segmentations=torch.zeros((1, 8, 8, 8), dtype=torch.long),
        segmentation_mask=torch.ones((1, 8, 8, 8), dtype=torch.bool),
        report_texts=["unused"],
        organ_texts=[["A: a", "B: b"]],
        organ_raw_texts=[["a", "b"]],
        organ_text_mask=torch.tensor([[True, True]]),
        organ_labels=torch.tensor([[0.0, 1.0]]),
        organ_label_mask=torch.tensor([[True, True]]),
        lesion_global_labels=torch.tensor([1.0]),
        lesion_global_mask=torch.tensor([True]),
        lesion_organ_labels=torch.tensor([[0.0, 1.0]]),
        lesion_organ_mask=torch.tensor([[True, True]]),
        metadata=[{}],
    )
    outputs = OrganSegOutput(
        organ_image_embeddings=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        organ_text_embeddings=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        report_image_embeddings=torch.tensor([[1.0, 0.0]]),
        report_text_embeddings=torch.tensor([[1.0, 0.0]]),
        diagnostic_logits=torch.tensor([[0.0, 2.0]]),
        lesion_global_logits=torch.tensor([2.0]),
        lesion_organ_logits=torch.tensor([[0.0, 2.0]]),
        logit_scale=torch.tensor(1.0),
        organ_logit_scale=torch.tensor(1.0),
        organ_logit_bias=torch.tensor(-10.0),
        report_logit_scale=torch.tensor(1.0),
        report_logit_bias=torch.tensor(-10.0),
        segmentation_loss=torch.tensor(1.5),
        segmentation_dice=0.25,
        segmentation_patch_count=2,
        patch_organ_presence_loss=torch.tensor(0.5),
        patch_organ_presence_accuracy=1.0,
        patch_organ_presence_count=2,
        organ_attention_loss=torch.tensor(0.0),
        organ_attention_accuracy=0.0,
        organ_attention_positive_accuracy=0.0,
        organ_attention_negative_accuracy=0.0,
        organ_attention_count=0,
        organ_attention_positive_count=0,
        organ_attention_negative_count=0,
    )
    loss_output, metrics = composer(outputs, batch)
    assert torch.isfinite(loss_output.total_loss)
    assert torch.isfinite(loss_output.organ_clip_loss)
    assert torch.isfinite(loss_output.segmentation_loss)
    assert torch.isfinite(loss_output.diagnostic_loss)
    assert torch.isfinite(loss_output.report_clip_loss)
    assert torch.isfinite(loss_output.patch_organ_presence_loss)
    assert torch.isfinite(loss_output.lesion_global_loss)
    assert torch.isfinite(loss_output.lesion_organ_loss)
    assert metrics["organ_image_to_text_top1"] >= 0.0
    assert metrics["diagnostic_accuracy"] >= 0.0
