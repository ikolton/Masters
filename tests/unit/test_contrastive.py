from __future__ import annotations

import torch

from organ_seg_clip.models.losses.contrastive import _multi_positive_logits_loss, masked_organ_clip_loss


def test_multi_positive_logits_loss_supports_rectangular_distributed_shapes() -> None:
    logits_image_to_text = torch.tensor([[2.0, 0.0, -1.0, -2.0], [0.1, 1.5, -0.5, -1.0]])
    logits_text_to_image = torch.tensor([[1.8, -0.2, -1.0, -2.0], [0.0, 1.3, -0.3, -1.1]])
    positive_mask = torch.tensor([[True, False, True, False], [False, True, False, True]])

    loss, metrics = _multi_positive_logits_loss(
        logits_image_to_text,
        logits_text_to_image,
        positive_mask,
        positive_mask,
    )

    assert torch.isfinite(loss)
    assert metrics["image_to_text_top1"] >= 0.0
    assert metrics["text_to_image_top1"] >= 0.0

def test_masked_organ_clip_loss_excludes_missing_text_rows_from_metrics() -> None:
    organ_embeddings = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    organ_text_embeddings = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    organ_mask = torch.tensor([[True, False, True]])

    loss, metrics = masked_organ_clip_loss(
        organ_embeddings,
        organ_text_embeddings,
        organ_mask,
        [["normal spleen", "", "normal liver"]],
        logit_scale=torch.tensor(10.0),
    )

    assert torch.isfinite(loss)
    assert metrics["image_to_text_top1"] == 1.0
    assert metrics["text_to_image_top1"] == 1.0

