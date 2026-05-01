"""Optional experiment logging backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config.schemas import EncoderConfig
from ..runtime.distributed import is_main_process


class ExperimentLogger:
    def __init__(self) -> None:
        self._wandb = None
        self._run = None
        self._enabled = False
        self._init_kwargs: dict[str, Any] | None = None
        self._step_log_start = 0
        self._train_steps_seen = 0
        self._run_id: str = ""

    @classmethod
    def for_encoder_training(
        cls,
        config: EncoderConfig,
        *,
        output_dir: Path,
        resume_run_id: str = "",
    ) -> "ExperimentLogger":
        logger = cls()
        if not is_main_process():
            return logger
        if not bool(config.logging.wandb_enabled) or str(config.logging.wandb_mode).strip().lower() == "disabled":
            return logger
        try:
            import wandb  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "logging.wandb_enabled is true, but wandb could not be imported in this environment."
            ) from exc
        run_name = str(config.logging.wandb_run_name).strip() or output_dir.name
        logger._init_kwargs = {
            "project": config.logging.wandb_project,
            "name": run_name,
            "mode": config.logging.wandb_mode,
            "config": config.to_dict(),
            "tags": list(config.logging.wandb_tags),
            "dir": str(output_dir),
        }
        explicit_resume_run_id = str(config.logging.wandb_resume_run_id).strip()
        resume_run_id = explicit_resume_run_id or str(resume_run_id).strip()
        if resume_run_id:
            logger._init_kwargs["id"] = resume_run_id
            logger._init_kwargs["resume"] = "must"
            logger._run_id = resume_run_id
        entity = str(config.logging.wandb_entity).strip()
        if entity:
            logger._init_kwargs["entity"] = entity
        logger._wandb = wandb
        logger._enabled = True
        logger._step_log_start = int(config.logging.wandb_step_log_start)
        logger._ensure_initialized()
        return logger

    def set_train_steps_seen(self, value: int) -> None:
        self._train_steps_seen = max(int(value), 0)

    def observe_train_step(self) -> None:
        self._train_steps_seen += 1

    def train_steps_seen(self) -> int:
        return int(self._train_steps_seen)

    def run_id(self) -> str:
        return str(self._run_id)

    def log_step(self, *, run_label: str, metrics: dict[str, float], epoch: int, step_index: int) -> None:
        if not self._enabled or self._wandb is None:
            return
        if int(self._train_steps_seen) < int(self._step_log_start):
            return
        self._ensure_initialized()
        if self._run is None:
            return
        payload = {
            "meta/epoch": float(epoch),
            "meta/step_in_epoch": float(step_index),
            "meta/train_steps_seen": float(self._train_steps_seen),
        }
        payload.update(_step_metric_payload(run_label=run_label, metrics=metrics))
        try:
            self._wandb.log(payload, step=self._current_wandb_step())
        except Exception as exc:
            self._handle_failure("wandb.log", exc, drop_run=True)

    def log_epoch_summary(self, *, epoch_metrics: dict[str, float]) -> None:
        if not self._enabled or self._wandb is None:
            return
        self._ensure_initialized()
        if self._run is None:
            return
        payload = _epoch_summary_payload(epoch_metrics)
        if not payload:
            return
        try:
            self._wandb.log(payload, step=self._current_wandb_step())
        except Exception as exc:
            self._handle_failure("wandb.log", exc, drop_run=True)

    def finish(self) -> None:
        if not self._enabled or self._wandb is None or self._run is None:
            return
        try:
            self._wandb.finish()
        except Exception as exc:
            self._handle_failure("wandb.finish", exc, drop_run=False)
        finally:
            self._run = None

    def _ensure_initialized(self) -> None:
        if not self._enabled or self._wandb is None or self._run is not None or self._init_kwargs is None:
            return
        try:
            self._run = self._wandb.init(**self._init_kwargs)
            if self._run is not None:
                run_id = str(getattr(self._run, "id", "")).strip()
                if run_id:
                    self._run_id = run_id
                    self._init_kwargs["id"] = run_id
                    self._init_kwargs["resume"] = "allow"
        except Exception as exc:
            self._handle_failure("wandb.init", exc, drop_run=True)

    def _handle_failure(self, action: str, exc: Exception, *, drop_run: bool) -> None:
        if is_main_process():
            print(f"[wandb] {action} failed; training will continue. reason={exc}", flush=True)
        if drop_run:
            self._run = None

    def _current_wandb_step(self) -> int:
        return max(int(self._train_steps_seen), 1)


def _step_metric_payload(*, run_label: str, metrics: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    normalized = "smoke_val" if run_label in {"fast-val", "smoke-val"} else ("full_val" if run_label == "val" else run_label)
    if normalized not in {"train", "full_val", "smoke_val"}:
        normalized = run_label
    if normalized == "train":
        primary_keys = (
            ("train/total_loss", "total_loss"),
            ("train/organ_alignment_loss", "organ_alignment_loss"),
            ("train/segmentation_loss", "segmentation_loss"),
            ("train/segmentation_dice", "segmentation_dice"),
            ("train/diagnostic_loss", "diagnostic_loss"),
            ("train/diagnostic_accuracy", "diagnostic_accuracy"),
        )
        secondary_keys = (
            ("train/organ_image_to_text_top1", "organ_image_to_text_top1"),
            ("train/organ_text_to_image_top1", "organ_text_to_image_top1"),
            ("train/patch_organ_presence_loss", "patch_organ_presence_loss"),
            ("train/organ_attention_loss", "organ_attention_loss"),
            ("train/lesion_organ_loss", "lesion_organ_loss"),
            ("train/segmentation_oom_fallback_count", "segmentation_oom_fallback_count"),
            ("train/step_seconds", "step_seconds"),
            ("train/data_wait_seconds", "data_wait_seconds"),
            ("train/lr_main", "lr_main"),
            ("train/lr_text", "lr_text"),
            ("train/lr_alignment_parameters", "lr_alignment_parameters"),
            ("train/organ_logit_scale", "organ_logit_scale"),
            ("train/organ_logit_bias", "organ_logit_bias"),
        )
    else:
        prefix = f"{normalized}_batch"
        primary_keys = ()
        secondary_keys = (
            (f"{prefix}/total_loss", "total_loss"),
            (f"{prefix}/organ_alignment_loss", "organ_alignment_loss"),
            (f"{prefix}/segmentation_loss", "segmentation_loss"),
            (f"{prefix}/segmentation_dice", "segmentation_dice"),
            (f"{prefix}/diagnostic_loss", "diagnostic_loss"),
            (f"{prefix}/diagnostic_accuracy", "diagnostic_accuracy"),
            (f"{prefix}/organ_image_to_text_top1", "organ_image_to_text_top1"),
            (f"{prefix}/organ_text_to_image_top1", "organ_text_to_image_top1"),
            (f"{prefix}/patch_organ_presence_loss", "patch_organ_presence_loss"),
            (f"{prefix}/organ_attention_loss", "organ_attention_loss"),
            (f"{prefix}/lesion_organ_loss", "lesion_organ_loss"),
            (f"{prefix}/segmentation_oom_fallback_count", "segmentation_oom_fallback_count"),
            (f"{prefix}/step_seconds", "step_seconds"),
            (f"{prefix}/organ_logit_scale", "organ_logit_scale"),
            (f"{prefix}/organ_logit_bias", "organ_logit_bias"),
        )
    for alias_key, source_key in primary_keys + secondary_keys:
        if source_key in metrics and isinstance(metrics[source_key], (int, float)):
            payload[alias_key] = float(metrics[source_key])
    return payload


def _epoch_summary_payload(epoch_metrics: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    if "epoch" in epoch_metrics and isinstance(epoch_metrics["epoch"], (int, float)):
        payload["meta/epoch"] = float(epoch_metrics["epoch"])
    if "validation_kind_full" in epoch_metrics and isinstance(epoch_metrics["validation_kind_full"], (int, float)):
        payload["meta/validation_kind_full"] = float(epoch_metrics["validation_kind_full"])
    preferred_keys = (
        ("epoch/train_total_loss", "train_total_loss"),
        ("epoch/train_organ_alignment_loss", "train_organ_alignment_loss"),
        ("epoch/train_segmentation_loss", "train_segmentation_loss"),
        ("epoch/train_segmentation_dice", "train_segmentation_dice"),
        ("epoch/train_diagnostic_loss", "train_diagnostic_loss"),
        ("epoch/train_diagnostic_accuracy", "train_diagnostic_accuracy"),
        ("epoch/train_organ_image_to_text_top1", "train_organ_image_to_text_top1"),
        ("epoch/train_organ_text_to_image_top1", "train_organ_text_to_image_top1"),
        ("epoch/train_organ_logit_gap", "train_organ_logit_gap"),
        ("epoch/train_organ_logit_scale", "train_organ_logit_scale"),
        ("epoch/train_organ_logit_bias", "train_organ_logit_bias"),
        ("epoch/train_patch_organ_presence_loss", "train_patch_organ_presence_loss"),
        ("epoch/train_organ_attention_loss", "train_organ_attention_loss"),
        ("epoch/train_organ_attention_positive_accuracy", "train_organ_attention_positive_accuracy"),
        ("epoch/train_organ_attention_negative_accuracy", "train_organ_attention_negative_accuracy"),
        ("epoch/train_lr_main", "train_lr_main"),
        ("epoch/train_lr_text", "train_lr_text"),
        ("epoch/train_lr_alignment_parameters", "train_lr_alignment_parameters"),
        ("epoch/full_val_total_loss", "full_val_total_loss"),
        ("epoch/full_val_organ_alignment_loss", "full_val_organ_alignment_loss"),
        ("epoch/full_val_segmentation_loss", "full_val_segmentation_loss"),
        ("epoch/full_val_segmentation_dice", "full_val_segmentation_dice"),
        ("epoch/full_val_diagnostic_loss", "full_val_diagnostic_loss"),
        ("epoch/full_val_diagnostic_accuracy", "full_val_diagnostic_accuracy"),
        ("epoch/full_val_organ_image_to_text_top1", "full_val_organ_image_to_text_top1"),
        ("epoch/full_val_organ_text_to_image_top1", "full_val_organ_text_to_image_top1"),
        ("epoch/full_val_organ_logit_gap", "full_val_organ_logit_gap"),
        ("epoch/full_val_organ_positive_logit_mean", "full_val_organ_positive_logit_mean"),
        ("epoch/full_val_organ_negative_logit_mean", "full_val_organ_negative_logit_mean"),
        ("epoch/full_val_organ_logit_scale", "full_val_organ_logit_scale"),
        ("epoch/full_val_organ_logit_bias", "full_val_organ_logit_bias"),
        ("epoch/full_val_patch_organ_presence_loss", "full_val_patch_organ_presence_loss"),
        ("epoch/full_val_organ_attention_loss", "full_val_organ_attention_loss"),
        ("epoch/full_val_organ_attention_positive_accuracy", "full_val_organ_attention_positive_accuracy"),
        ("epoch/full_val_organ_attention_negative_accuracy", "full_val_organ_attention_negative_accuracy"),
        ("epoch/smoke_val_total_loss", "smoke_val_total_loss"),
        ("epoch/smoke_val_organ_alignment_loss", "smoke_val_organ_alignment_loss"),
        ("epoch/smoke_val_segmentation_loss", "smoke_val_segmentation_loss"),
        ("epoch/smoke_val_segmentation_dice", "smoke_val_segmentation_dice"),
        ("epoch/smoke_val_diagnostic_loss", "smoke_val_diagnostic_loss"),
        ("epoch/smoke_val_diagnostic_accuracy", "smoke_val_diagnostic_accuracy"),
        ("epoch/smoke_val_organ_image_to_text_top1", "smoke_val_organ_image_to_text_top1"),
        ("epoch/smoke_val_organ_text_to_image_top1", "smoke_val_organ_text_to_image_top1"),
        ("epoch/smoke_val_organ_logit_gap", "smoke_val_organ_logit_gap"),
        ("epoch/smoke_val_organ_positive_logit_mean", "smoke_val_organ_positive_logit_mean"),
        ("epoch/smoke_val_organ_negative_logit_mean", "smoke_val_organ_negative_logit_mean"),
        ("epoch/smoke_val_organ_logit_scale", "smoke_val_organ_logit_scale"),
        ("epoch/smoke_val_organ_logit_bias", "smoke_val_organ_logit_bias"),
        ("epoch/smoke_val_patch_organ_presence_loss", "smoke_val_patch_organ_presence_loss"),
        ("epoch/smoke_val_organ_attention_loss", "smoke_val_organ_attention_loss"),
        ("epoch/smoke_val_organ_attention_positive_accuracy", "smoke_val_organ_attention_positive_accuracy"),
        ("epoch/smoke_val_organ_attention_negative_accuracy", "smoke_val_organ_attention_negative_accuracy"),
        ("epoch/train_segmentation_oom_fallback_count", "train_segmentation_oom_fallback_count"),
        ("epoch/full_val_segmentation_oom_fallback_count", "full_val_segmentation_oom_fallback_count"),
        ("epoch/smoke_val_segmentation_oom_fallback_count", "smoke_val_segmentation_oom_fallback_count"),
    )
    for alias_key, source_key in preferred_keys:
        if source_key in epoch_metrics and isinstance(epoch_metrics[source_key], (int, float)):
            payload[alias_key] = float(epoch_metrics[source_key])
    for source_key, value in sorted(epoch_metrics.items()):
        if not isinstance(value, (int, float)):
            continue
        if source_key in {"epoch", "validation_kind_full"}:
            continue
        payload.setdefault(f"epoch/{source_key}", float(value))
    return payload
