from __future__ import annotations

import torch

from organ_seg_clip.config.loader import encoder_config_from_dict
from organ_seg_clip.data.lesion_metadata import load_lesion_metadata_csv
from organ_seg_clip.models.aggregation.model import _patch_organ_presence_targets
from organ_seg_clip.models.interfaces.types import EncoderBatch, OrganSegOutput
from organ_seg_clip.models.losses.composer import OrganSegLossComposer
from organ_seg_clip.config.schemas import LossConfig


def test_lesion_metadata_csv_maps_counts_and_global_label(tmp_path) -> None:
    csv_path = tmp_path / "lesions.csv"
    csv_path.write_text(
        "Encrypted Accession Number,number of liver lesion instances,number of kidney lesion instances,no lesion\n"
        "study-a,2,0,0\n"
        "study-b,0,1,1\n",
        encoding="utf-8",
    )
    records = load_lesion_metadata_csv(csv_path, organ_names=("Liver", "Kidneys", "Colon"))

    assert records["study-a"].global_label == 1.0
    assert records["study-a"].organ_labels == {"Liver": 1.0, "Kidneys": 0.0}
    assert records["study-b"].global_label == 0.0
    assert records["study-b"].organ_labels["Kidneys"] == 1.0
    assert "Colon" not in records["study-a"].organ_labels


def test_patch_organ_presence_targets_respect_min_voxels() -> None:
    segmentation = torch.zeros((2, 4, 4, 4), dtype=torch.long)
    segmentation[0, :2, :2, :2] = 5
    segmentation[1, 0, 0, 0] = 5

    targets, mask = _patch_organ_presence_targets(
        segmentation,
        None,
        organ_label_values=((5,), (2,)),
        enabled_organs=(True, True),
        min_voxels=4,
    )

    assert targets.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert mask.tolist() == [[True, True], [True, True]]


def test_auxiliary_losses_zero_when_masks_empty() -> None:
    composer = OrganSegLossComposer(
        LossConfig(report_clip_weight=0.2, patch_organ_presence_weight=0.1, lesion_global_weight=0.05, lesion_organ_weight=0.05)
    )
    batch = EncoderBatch(
        study_ids=["study-1"],
        images=torch.zeros((1, 1, 4, 4, 4)),
        image_mask=torch.ones((1, 1, 4, 4, 4), dtype=torch.bool),
        segmentations=None,
        segmentation_mask=None,
        report_texts=[""],
        organ_texts=[[""]],
        organ_raw_texts=[[""]],
        organ_text_mask=torch.tensor([[False]]),
        organ_labels=torch.zeros((1, 1)),
        organ_label_mask=torch.tensor([[False]]),
        lesion_global_labels=torch.zeros((1,)),
        lesion_global_mask=torch.tensor([False]),
        lesion_organ_labels=torch.zeros((1, 1)),
        lesion_organ_mask=torch.tensor([[False]]),
        metadata=[{}],
    )
    outputs = OrganSegOutput(
        organ_image_embeddings=torch.zeros((1, 1, 2)),
        organ_text_embeddings=torch.zeros((1, 1, 2)),
        report_image_embeddings=torch.zeros((1, 2)),
        report_text_embeddings=torch.zeros((1, 2)),
        diagnostic_logits=torch.zeros((1, 1)),
        lesion_global_logits=torch.zeros((1,)),
        lesion_organ_logits=torch.zeros((1, 1)),
        logit_scale=torch.tensor(1.0),
        organ_logit_scale=torch.tensor(1.0),
        organ_logit_bias=torch.tensor(-10.0),
        report_logit_scale=torch.tensor(1.0),
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
    assert loss_output.report_clip_loss.item() == 0.0
    assert loss_output.lesion_global_loss.item() == 0.0
    assert loss_output.lesion_organ_loss.item() == 0.0
    assert metrics["lesion_global_accuracy"] == 0.0


def test_report_clip_loss_is_zero_with_single_report() -> None:
    composer = OrganSegLossComposer(LossConfig(report_clip_weight=1.0, organ_clip_weight=0.0, segmentation_weight=0.0, diagnostic_weight=0.0))
    batch = EncoderBatch(
        study_ids=["study-1"],
        images=torch.zeros((1, 1, 4, 4, 4)),
        image_mask=torch.ones((1, 1, 4, 4, 4), dtype=torch.bool),
        segmentations=None,
        segmentation_mask=None,
        report_texts=["only report"],
        organ_texts=[[""]],
        organ_raw_texts=[[""]],
        organ_text_mask=torch.zeros((1, 1), dtype=torch.bool),
        organ_labels=torch.zeros((1, 1)),
        organ_label_mask=torch.zeros((1, 1), dtype=torch.bool),
        lesion_global_labels=torch.zeros((1,)),
        lesion_global_mask=torch.zeros((1,), dtype=torch.bool),
        lesion_organ_labels=torch.zeros((1, 1)),
        lesion_organ_mask=torch.zeros((1, 1), dtype=torch.bool),
        metadata=[{}],
    )
    outputs = OrganSegOutput(
        organ_image_embeddings=torch.zeros((1, 1, 2)),
        organ_text_embeddings=torch.zeros((1, 1, 2)),
        report_image_embeddings=torch.tensor([[1.0, 0.0]]),
        report_text_embeddings=torch.tensor([[1.0, 0.0]]),
        diagnostic_logits=torch.zeros((1, 1)),
        lesion_global_logits=torch.zeros((1,)),
        lesion_organ_logits=torch.zeros((1, 1)),
        logit_scale=torch.tensor(5.0),
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

    assert loss_output.report_clip_loss.item() == 0.0
    assert metrics["report_valid_count"] == 1.0


def test_auxiliary_losses_contribute_when_enabled() -> None:
    composer = OrganSegLossComposer(
        LossConfig(
            organ_clip_weight=0.0,
            segmentation_weight=0.0,
            diagnostic_weight=0.0,
            report_clip_weight=1.0,
            patch_organ_presence_weight=1.0,
            lesion_global_weight=1.0,
            lesion_organ_weight=1.0,
        )
    )
    batch = EncoderBatch(
        study_ids=["study-1", "study-2"],
        images=torch.zeros((2, 1, 4, 4, 4)),
        image_mask=torch.ones((2, 1, 4, 4, 4), dtype=torch.bool),
        segmentations=None,
        segmentation_mask=None,
        report_texts=["report one", "report two"],
        organ_texts=[[""], [""]],
        organ_raw_texts=[[""], [""]],
        organ_text_mask=torch.zeros((2, 1), dtype=torch.bool),
        organ_labels=torch.zeros((2, 1)),
        organ_label_mask=torch.zeros((2, 1), dtype=torch.bool),
        lesion_global_labels=torch.tensor([1.0, 0.0]),
        lesion_global_mask=torch.tensor([True, True]),
        lesion_organ_labels=torch.tensor([[1.0], [0.0]]),
        lesion_organ_mask=torch.tensor([[True], [True]]),
        metadata=[{}, {}],
    )
    outputs = OrganSegOutput(
        organ_image_embeddings=torch.zeros((2, 1, 2)),
        organ_text_embeddings=torch.zeros((2, 1, 2)),
        report_image_embeddings=torch.eye(2),
        report_text_embeddings=torch.eye(2),
        diagnostic_logits=torch.zeros((2, 1)),
        lesion_global_logits=torch.tensor([2.0, -2.0]),
        lesion_organ_logits=torch.tensor([[2.0], [-2.0]]),
        logit_scale=torch.tensor(5.0),
        organ_logit_scale=torch.tensor(5.0),
        organ_logit_bias=torch.tensor(-10.0),
        report_logit_scale=torch.tensor(5.0),
        report_logit_bias=torch.tensor(-10.0),
        segmentation_loss=torch.tensor(0.0),
        segmentation_dice=0.0,
        segmentation_patch_count=0,
        patch_organ_presence_loss=torch.tensor(0.25),
        patch_organ_presence_accuracy=0.5,
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

    assert loss_output.report_clip_loss.item() > 0.0
    assert loss_output.patch_organ_presence_loss.item() == 0.25
    assert loss_output.lesion_global_loss.item() < 0.2
    assert loss_output.lesion_organ_loss.item() < 0.2
    assert torch.isfinite(loss_output.total_loss)
    assert metrics["lesion_global_accuracy"] == 1.0


def test_old_style_config_loads_without_auxiliary_fields() -> None:
    config = encoder_config_from_dict(
        {
            "paths": {"dataset_root": "/tmp", "output_dir": "/tmp/out"},
            "text_encoder": {"backend_family": "hash", "projection_dim": 16},
            "model": {
                "organ_query_count": 1,
                "tokenizer": {"model_dim": 16},
            },
            "data": {"organ_names": ["Liver"]},
        },
        config_path="/tmp/config.yaml",
    )

    assert config.loss.report_clip_weight == 0.0
    assert config.loss.patch_organ_presence_weight == 0.0
    assert config.data.lesion_metadata_csv == ""


def test_loss_config_accepts_pair_balanced_organ_siglip_fields() -> None:
    config = LossConfig(
        alignment_type="siglip",
        organ_pair_balance=True,
        organ_positive_weight=1.0,
        organ_same_organ_weight=2.0,
        organ_cross_organ_weight=0.5,
        organ_frequency_balance=True,
        organ_frequency_balance_power=0.5,
        organ_frequency_balance_min=0.25,
        organ_frequency_balance_max=4.0,
    )

    assert config.organ_pair_balance is True
    assert config.organ_same_organ_weight == 2.0
    assert config.organ_cross_organ_weight == 0.5
    assert config.organ_frequency_balance is True
