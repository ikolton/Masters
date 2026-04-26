from __future__ import annotations

import torch

from organ_seg_clip.config.schemas import LossConfig
from organ_seg_clip.models.aggregation.model import _grid_position_features
from organ_seg_clip.models.aggregation.modules import AlignmentProjectionHead, GridFeatureCombiner, PatchSummaryHead
from organ_seg_clip.models.interfaces.types import EncoderBatch, OrganSegOutput
from organ_seg_clip.models.losses.contrastive import _build_positive_mask
from organ_seg_clip.models.losses.composer import OrganSegLossComposer
from organ_seg_clip.models.losses.siglip import masked_organ_siglip_loss, masked_report_siglip_loss


def test_same_text_different_organs_are_not_positive_pairs() -> None:
    mask = _build_positive_mask(["unremarkable", "unremarkable"], [0, 1], device=torch.device("cpu"))

    assert mask.tolist() == [[True, False], [False, True]]


def test_report_siglip_has_positive_loss_with_single_report() -> None:
    loss, metrics = masked_report_siglip_loss(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([True]),
        ["study-1"],
        logit_scale=torch.tensor(5.0),
        logit_bias=torch.tensor(-10.0),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert metrics["valid_count"] == 1.0
    assert metrics["image_to_text_top1"] == 1.0


def test_grid_position_features_use_full_reference_grid() -> None:
    all_boxes = [
        (0, 8, 0, 8, 0, 8),
        (0, 8, 0, 8, 8, 16),
        (8, 16, 0, 8, 0, 8),
        (8, 16, 0, 8, 8, 16),
    ]
    chunk = [all_boxes[-1]]

    features = _grid_position_features(chunk, (16, 8, 16), reference_boxes=all_boxes, device=torch.device("cpu"))

    assert tuple(features.shape) == (1, 9)
    assert features[0, :3].tolist() == [1.0, 0.0, 1.0]


def test_siglip_loss_composer_keeps_same_text_organs_separate() -> None:
    composer = OrganSegLossComposer(
        LossConfig(
            alignment_type="siglip",
            organ_alignment_weight=1.0,
            report_alignment_weight=1.0,
            segmentation_weight=0.0,
            diagnostic_weight=0.0,
        )
    )
    batch = EncoderBatch(
        study_ids=["study-1"],
        images=torch.zeros((1, 1, 4, 4, 4)),
        image_mask=torch.ones((1, 1, 4, 4, 4), dtype=torch.bool),
        segmentations=None,
        segmentation_mask=None,
        report_texts=["report"],
        organ_texts=[["Liver: unremarkable", "Kidneys: unremarkable"]],
        organ_raw_texts=[["unremarkable", "unremarkable"]],
        organ_text_mask=torch.tensor([[True, True]]),
        organ_labels=torch.zeros((1, 2)),
        organ_label_mask=torch.zeros((1, 2), dtype=torch.bool),
        lesion_global_labels=torch.zeros((1,)),
        lesion_global_mask=torch.zeros((1,), dtype=torch.bool),
        lesion_organ_labels=torch.zeros((1, 2)),
        lesion_organ_mask=torch.zeros((1, 2), dtype=torch.bool),
        metadata=[{}],
    )
    outputs = OrganSegOutput(
        organ_image_embeddings=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        organ_text_embeddings=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        report_image_embeddings=torch.tensor([[1.0, 0.0]]),
        report_text_embeddings=torch.tensor([[1.0, 0.0]]),
        diagnostic_logits=torch.zeros((1, 2)),
        lesion_global_logits=torch.zeros((1,)),
        lesion_organ_logits=torch.zeros((1, 2)),
        logit_scale=torch.tensor(1.0),
        organ_logit_scale=torch.tensor(5.0),
        organ_logit_bias=torch.tensor(-10.0),
        report_logit_scale=torch.tensor(5.0),
        report_logit_bias=torch.tensor(-10.0),
        segmentation_loss=torch.tensor(0.0),
        segmentation_dice=0.0,
        segmentation_patch_count=0,
        patch_organ_presence_loss=torch.tensor(0.0),
        patch_organ_presence_accuracy=0.0,
        patch_organ_presence_count=0,
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
    assert loss_output.organ_alignment_loss.item() > 0.0
    assert loss_output.report_alignment_loss.item() > 0.0
    assert metrics["organ_image_to_text_top1"] == 1.0
    assert metrics["organ_positive_logit_mean"] > metrics["organ_negative_logit_mean"]
    assert metrics["organ_logit_scale"] == 5.0
    assert metrics["report_logit_bias"] == -10.0


def test_siglip_balances_positive_and_negative_terms_per_row() -> None:
    loss, metrics = masked_report_siglip_loss(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([True, True, True, True]),
        ["study-1", "study-2", "study-3", "study-4"],
        logit_scale=torch.tensor(0.0),
        logit_bias=torch.tensor(-10.0),
    )

    assert torch.isfinite(loss)
    assert loss.item() > 4.0
    assert metrics["positive_logit_mean"] == metrics["negative_logit_mean"]
    assert metrics["logit_gap"] == 0.0


def test_pair_balanced_organ_siglip_emphasizes_same_organ_negatives() -> None:
    image_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    text_embeddings = image_embeddings.clone()
    organ_mask = torch.ones((2, 2), dtype=torch.bool)
    organ_texts = [
        ["cyst", "unremarkable"],
        ["mass", "unremarkable"],
    ]

    unbalanced_loss, _ = masked_organ_siglip_loss(
        image_embeddings,
        text_embeddings,
        organ_mask,
        organ_texts,
        logit_scale=torch.tensor(10.0),
        logit_bias=torch.tensor(0.0),
    )
    balanced_loss, metrics = masked_organ_siglip_loss(
        image_embeddings,
        text_embeddings,
        organ_mask,
        organ_texts,
        logit_scale=torch.tensor(10.0),
        logit_bias=torch.tensor(0.0),
        pair_balance=True,
        positive_weight=1.0,
        same_organ_weight=1.0,
        cross_organ_weight=1.0,
    )

    assert torch.isfinite(balanced_loss)
    assert balanced_loss.item() > unbalanced_loss.item()
    assert metrics["same_organ_negative_logit_mean"] > metrics["cross_organ_negative_logit_mean"]
    assert metrics["same_organ_logit_gap"] < metrics["cross_organ_logit_gap"]


def test_pair_balanced_organ_siglip_disables_cross_organ_pairs_when_weight_is_zero() -> None:
    image_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    text_embeddings = image_embeddings.clone()
    organ_mask = torch.ones((2, 2), dtype=torch.bool)
    organ_texts = [
        ["cyst", "unremarkable"],
        ["mass", "unremarkable"],
    ]

    loss, metrics = masked_organ_siglip_loss(
        image_embeddings,
        text_embeddings,
        organ_mask,
        organ_texts,
        logit_scale=torch.tensor(10.0),
        logit_bias=torch.tensor(0.0),
        pair_balance=True,
        positive_weight=1.0,
        same_organ_weight=1.0,
        cross_organ_weight=0.0,
    )

    assert torch.isfinite(loss)
    assert metrics["cross_organ_negative_logit_mean"] == 0.0
    assert metrics["negative_logit_mean"] == metrics["same_organ_negative_logit_mean"]
    assert metrics["same_organ_negative_logit_mean"] > 0.0


def test_frequency_balanced_organ_siglip_reports_row_weights() -> None:
    image_embeddings = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    text_embeddings = image_embeddings.clone()
    organ_mask = torch.ones((3, 1), dtype=torch.bool)
    organ_texts = [["common"], ["common"], ["rare"]]

    loss, metrics = masked_organ_siglip_loss(
        image_embeddings,
        text_embeddings,
        organ_mask,
        organ_texts,
        logit_scale=torch.tensor(1.0),
        logit_bias=torch.tensor(0.0),
        finding_counts={(0, "common"): 100, (0, "rare"): 1},
        frequency_balance=True,
        frequency_balance_power=0.5,
        frequency_balance_min=0.25,
        frequency_balance_max=4.0,
    )

    assert torch.isfinite(loss)
    assert metrics["row_weight_mean"] > 0.0


def test_spectre_style_patch_summary_and_grid_global_token_shapes() -> None:
    patch_tokens = torch.randn(3, 8, 16)
    summary_head = PatchSummaryHead(model_dim=16, num_heads=4, dropout=0.0, summary_mode="attention_mean")
    patch_summaries = summary_head(patch_tokens).unsqueeze(0)
    positions = torch.randn(1, 3, 9)
    mask = torch.ones((1, 3), dtype=torch.bool)
    combiner = GridFeatureCombiner(model_dim=16, depth=1, num_heads=4, dropout=0.0, use_global_token=True)

    combined = combiner(patch_summaries, positions, mask)

    assert tuple(patch_summaries.shape) == (1, 3, 16)
    assert tuple(combined.shape) == (1, 4, 16)


def test_alignment_projection_head_preserves_embedding_shape() -> None:
    head = AlignmentProjectionHead(model_dim=16, hidden_dim=32, bottleneck_dim=8, dropout=0.0, layer_norm=True)
    embeddings = torch.randn(2, 3, 16)

    projected = head(embeddings)

    assert tuple(projected.shape) == tuple(embeddings.shape)
    assert torch.isfinite(projected).all()
