"""Optional experiment logging backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config.schemas import DecoderConfig, EncoderConfig
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
        self._step_payload_builder = lambda run_label=None, metrics=None: {}
        self._epoch_payload_builder = lambda epoch_metrics: {}

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
        logger._step_payload_builder = _step_metric_payload
        logger._epoch_payload_builder = _epoch_summary_payload
        logger._ensure_initialized()
        return logger

    @classmethod
    def for_decoder_training(
        cls,
        config: DecoderConfig,
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
        logger._step_payload_builder = _decoder_step_metric_payload
        logger._epoch_payload_builder = _decoder_epoch_summary_payload
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
        payload.update(self._step_payload_builder(run_label=run_label, metrics=metrics))
        if len(payload) <= 3:
            return
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
        payload = self._epoch_payload_builder(epoch_metrics)
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
    if normalized != "train":
        return payload
    for alias_key, source_key in (
        ("train_step/total_loss", "total_loss"),
        ("train_step/organ_alignment_loss", "organ_alignment_loss"),
        ("train_step/segmentation_loss", "segmentation_loss"),
        ("train_step/segmentation_dice", "segmentation_dice"),
        ("train_step/diagnostic_accuracy", "diagnostic_accuracy"),
        ("train_step/organ_logit_gap", "organ_logit_gap"),
        ("train_step/lr_main", "lr_main"),
        ("train_step/lr_alignment_parameters", "lr_alignment_parameters"),
        ("train_step/step_seconds", "step_seconds"),
        ("train_step/data_wait_seconds", "data_wait_seconds"),
        ("train_step/cuda_memory_allocated_gb", "cuda_memory_allocated_gb"),
        ("train_step/segmentation_oom_fallback_count", "segmentation_oom_fallback_count"),
    ):
        if source_key in metrics and isinstance(metrics[source_key], (int, float)):
            payload[alias_key] = float(metrics[source_key])
    return payload


def _epoch_summary_payload(epoch_metrics: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    if "epoch" in epoch_metrics and isinstance(epoch_metrics["epoch"], (int, float)):
        payload["meta/epoch"] = float(epoch_metrics["epoch"])
    for alias_key, source_key in (
        ("train_epoch/total_loss", "train_total_loss"),
        ("train_epoch/organ_alignment_loss", "train_organ_alignment_loss"),
        ("train_epoch/segmentation_loss", "train_segmentation_loss"),
        ("train_epoch/segmentation_dice", "train_segmentation_dice"),
        ("train_epoch/diagnostic_accuracy", "train_diagnostic_accuracy"),
        ("train_epoch/organ_image_to_text_top1", "train_organ_image_to_text_top1"),
        ("train_epoch/organ_text_to_image_top1", "train_organ_text_to_image_top1"),
        ("train_epoch/organ_logit_gap", "train_organ_logit_gap"),
        ("train_epoch/lr_main", "train_lr_main"),
        ("train_epoch/lr_alignment_parameters", "train_lr_alignment_parameters"),
        ("train_epoch/step_seconds", "train_step_seconds"),
        ("train_epoch/data_wait_seconds", "train_data_wait_seconds"),
        ("train_epoch/cuda_memory_allocated_gb", "train_cuda_memory_allocated_gb"),
        ("train_epoch/segmentation_oom_fallback_count", "train_segmentation_oom_fallback_count"),
        ("full_val/total_loss", "full_val_total_loss"),
        ("full_val/organ_alignment_loss", "full_val_organ_alignment_loss"),
        ("full_val/segmentation_loss", "full_val_segmentation_loss"),
        ("full_val/segmentation_dice", "full_val_segmentation_dice"),
        ("full_val/diagnostic_accuracy", "full_val_diagnostic_accuracy"),
        ("full_val/organ_image_to_text_top1", "full_val_organ_image_to_text_top1"),
        ("full_val/organ_text_to_image_top1", "full_val_organ_text_to_image_top1"),
        ("full_val/organ_logit_gap", "full_val_organ_logit_gap"),
        ("full_val/cuda_memory_allocated_gb", "full_val_cuda_memory_allocated_gb"),
        ("full_val/segmentation_oom_fallback_count", "full_val_segmentation_oom_fallback_count"),
        ("smoke_val/total_loss", "smoke_val_total_loss"),
        ("smoke_val/organ_alignment_loss", "smoke_val_organ_alignment_loss"),
        ("smoke_val/organ_logit_gap", "smoke_val_organ_logit_gap"),
        ("smoke_val/segmentation_dice", "smoke_val_segmentation_dice"),
        ("smoke_val/segmentation_oom_fallback_count", "smoke_val_segmentation_oom_fallback_count"),
    ):
        if source_key in epoch_metrics and isinstance(epoch_metrics[source_key], (int, float)):
            payload[alias_key] = float(epoch_metrics[source_key])
    return payload


def _decoder_step_metric_payload(run_label: str, metrics: dict[str, float]) -> dict[str, float]:
    if run_label != "train":
        return {}
    payload: dict[str, float] = {}
    for alias_key, source_key in (
        ("train_step/total_loss", "total_loss"),
        ("train_step/ce_loss", "ce_loss"),
        ("train_step/diagnostic_loss", "diagnostic_loss"),
        ("train_step/binary_diagnostic_loss", "binary_diagnostic_loss"),
        ("train_step/semantic_diagnostic_loss", "semantic_diagnostic_loss"),
        ("train_step/semantic_family_loss", "semantic_family_loss"),
        ("train_step/semantic_subtype_loss", "semantic_subtype_loss"),
        ("train_step/semantic_sample_count", "semantic_sample_count"),
        ("train_step/lr_main", "lr_main"),
    ):
        if source_key in metrics and isinstance(metrics[source_key], (int, float)):
            payload[alias_key] = float(metrics[source_key])
    return payload


def _decoder_epoch_summary_payload(epoch_metrics: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    if "epoch" in epoch_metrics and isinstance(epoch_metrics["epoch"], (int, float)):
        payload["meta/epoch"] = float(epoch_metrics["epoch"])
    for alias_key, source_key in (
        ("train_epoch/total_loss", "train_total_loss"),
        ("train_epoch/ce_loss", "train_ce_loss"),
        ("train_epoch/diagnostic_loss", "train_diagnostic_loss"),
        ("train_epoch/binary_diagnostic_loss", "train_binary_diagnostic_loss"),
        ("train_epoch/semantic_diagnostic_loss", "train_semantic_diagnostic_loss"),
        ("train_epoch/semantic_family_loss", "train_semantic_family_loss"),
        ("train_epoch/semantic_subtype_loss", "train_semantic_subtype_loss"),
        ("train_epoch/lr_main", "train_lr_main"),
        ("full_val/total_loss", "val_total_loss"),
        ("full_val/ce_loss", "val_ce_loss"),
        ("full_val/diagnostic_loss", "val_diagnostic_loss"),
        ("full_val/binary_diagnostic_loss", "val_binary_diagnostic_loss"),
        ("full_val/semantic_diagnostic_loss", "val_semantic_diagnostic_loss"),
        ("full_val/semantic_family_loss", "val_semantic_family_loss"),
        ("full_val/semantic_subtype_loss", "val_semantic_subtype_loss"),
    ):
        if source_key in epoch_metrics and isinstance(epoch_metrics[source_key], (int, float)):
            payload[alias_key] = float(epoch_metrics[source_key])
    return payload
