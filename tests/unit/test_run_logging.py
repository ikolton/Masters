from __future__ import annotations

from organ_seg_clip.training.run_logging import (
    _decoder_epoch_summary_payload,
    _decoder_step_metric_payload,
    _epoch_summary_payload,
    _step_metric_payload,
)


def test_step_metric_payload_only_logs_minimal_train_step_metrics() -> None:
    payload = _step_metric_payload(
        run_label="train",
        metrics={
            "total_loss": 1.0,
            "organ_alignment_loss": 2.0,
            "segmentation_loss": 3.0,
            "segmentation_dice": 0.4,
            "diagnostic_accuracy": 0.5,
            "organ_logit_gap": -0.2,
            "lr_main": 1e-4,
            "lr_alignment_parameters": 1e-5,
            "step_seconds": 0.7,
            "data_wait_seconds": 0.03,
            "cuda_memory_allocated_gb": 12.5,
            "segmentation_oom_fallback_count": 0.0,
            "organ_text_to_image_top1": 0.9,
            "patch_organ_presence_loss": 1.7,
        },
    )

    assert payload == {
        "train_step/total_loss": 1.0,
        "train_step/organ_alignment_loss": 2.0,
        "train_step/segmentation_loss": 3.0,
        "train_step/segmentation_dice": 0.4,
        "train_step/diagnostic_accuracy": 0.5,
        "train_step/organ_logit_gap": -0.2,
        "train_step/lr_main": 1e-4,
        "train_step/lr_alignment_parameters": 1e-5,
        "train_step/step_seconds": 0.7,
        "train_step/data_wait_seconds": 0.03,
        "train_step/cuda_memory_allocated_gb": 12.5,
        "train_step/segmentation_oom_fallback_count": 0.0,
    }


def test_step_metric_payload_drops_validation_batches() -> None:
    assert _step_metric_payload(run_label="val", metrics={"total_loss": 1.0}) == {}
    assert _step_metric_payload(run_label="smoke-val", metrics={"total_loss": 1.0}) == {}


def test_epoch_summary_payload_only_logs_allowlisted_groups() -> None:
    payload = _epoch_summary_payload(
        {
            "epoch": 2.0,
            "validation_kind_full": 1.0,
            "train_total_loss": 1.0,
            "train_organ_alignment_loss": 2.0,
            "train_segmentation_loss": 3.0,
            "train_segmentation_dice": 0.4,
            "train_diagnostic_accuracy": 0.5,
            "train_organ_image_to_text_top1": 0.6,
            "train_organ_text_to_image_top1": 0.7,
            "train_organ_logit_gap": -0.3,
            "train_lr_main": 1e-4,
            "train_lr_alignment_parameters": 1e-5,
            "train_step_seconds": 0.8,
            "train_data_wait_seconds": 0.05,
            "train_cuda_memory_allocated_gb": 10.0,
            "train_segmentation_oom_fallback_count": 0.0,
            "full_val_total_loss": 4.0,
            "full_val_organ_alignment_loss": 5.0,
            "full_val_segmentation_loss": 6.0,
            "full_val_segmentation_dice": 0.8,
            "full_val_diagnostic_accuracy": 0.9,
            "full_val_organ_image_to_text_top1": 0.91,
            "full_val_organ_text_to_image_top1": 0.92,
            "full_val_organ_logit_gap": -0.1,
            "full_val_cuda_memory_allocated_gb": 8.0,
            "full_val_segmentation_oom_fallback_count": 1.0,
            "smoke_val_total_loss": 7.0,
            "smoke_val_organ_alignment_loss": 8.0,
            "smoke_val_organ_logit_gap": -0.05,
            "smoke_val_segmentation_dice": 0.3,
            "smoke_val_segmentation_oom_fallback_count": 0.0,
            "val_total_loss": 999.0,
            "train_patch_organ_presence_loss": 123.0,
        }
    )

    assert payload == {
        "meta/epoch": 2.0,
        "train_epoch/total_loss": 1.0,
        "train_epoch/organ_alignment_loss": 2.0,
        "train_epoch/segmentation_loss": 3.0,
        "train_epoch/segmentation_dice": 0.4,
        "train_epoch/diagnostic_accuracy": 0.5,
        "train_epoch/organ_image_to_text_top1": 0.6,
        "train_epoch/organ_text_to_image_top1": 0.7,
        "train_epoch/organ_logit_gap": -0.3,
        "train_epoch/lr_main": 1e-4,
        "train_epoch/lr_alignment_parameters": 1e-5,
        "train_epoch/step_seconds": 0.8,
        "train_epoch/data_wait_seconds": 0.05,
        "train_epoch/cuda_memory_allocated_gb": 10.0,
        "train_epoch/segmentation_oom_fallback_count": 0.0,
        "full_val/total_loss": 4.0,
        "full_val/organ_alignment_loss": 5.0,
        "full_val/segmentation_loss": 6.0,
        "full_val/segmentation_dice": 0.8,
        "full_val/diagnostic_accuracy": 0.9,
        "full_val/organ_image_to_text_top1": 0.91,
        "full_val/organ_text_to_image_top1": 0.92,
        "full_val/organ_logit_gap": -0.1,
        "full_val/cuda_memory_allocated_gb": 8.0,
        "full_val/segmentation_oom_fallback_count": 1.0,
        "smoke_val/total_loss": 7.0,
        "smoke_val/organ_alignment_loss": 8.0,
        "smoke_val/organ_logit_gap": -0.05,
        "smoke_val/segmentation_dice": 0.3,
        "smoke_val/segmentation_oom_fallback_count": 0.0,
    }

    assert not any(key.startswith("epoch/") for key in payload)
    assert not any("val_" in key for key in payload)


def test_decoder_step_metric_payload_only_logs_minimal_train_metrics() -> None:
    payload = _decoder_step_metric_payload(
        "train",
        {
            "total_loss": 1.0,
            "ce_loss": 0.7,
            "diagnostic_loss": 0.3,
            "lr_main": 2e-4,
            "diagnostic_sample_count": 8.0,
        },
    )

    assert payload == {
        "train_step/total_loss": 1.0,
        "train_step/ce_loss": 0.7,
        "train_step/diagnostic_loss": 0.3,
        "train_step/lr_main": 2e-4,
    }
    assert _decoder_step_metric_payload("val", {"total_loss": 1.0}) == {}


def test_decoder_epoch_summary_payload_only_logs_minimal_groups() -> None:
    payload = _decoder_epoch_summary_payload(
        {
            "epoch": 3.0,
            "train_total_loss": 1.1,
            "train_ce_loss": 0.8,
            "train_diagnostic_loss": 0.3,
            "train_lr_main": 2e-4,
            "val_total_loss": 1.5,
            "val_ce_loss": 1.1,
            "val_diagnostic_loss": 0.4,
            "val_diagnostic_accuracy": 0.9,
        }
    )

    assert payload == {
        "meta/epoch": 3.0,
        "train_epoch/total_loss": 1.1,
        "train_epoch/ce_loss": 0.8,
        "train_epoch/diagnostic_loss": 0.3,
        "train_epoch/lr_main": 2e-4,
        "full_val/total_loss": 1.5,
        "full_val/ce_loss": 1.1,
        "full_val/diagnostic_loss": 0.4,
    }
