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
        normalized = "val" if run_label == "fast-val" else run_label
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
    normalized = "val" if run_label == "fast-val" else run_label
    if normalized not in {"train", "val"}:
        normalized = run_label
    if normalized == "train":
        primary_keys = (
            ("00_train/total_loss", "total_loss"),
            ("00_train/organ_alignment_loss", "organ_alignment_loss"),
            ("00_train/segmentation_loss", "segmentation_loss"),
            ("00_train/segmentation_dice", "segmentation_dice"),
            ("00_train/diagnostic_loss", "diagnostic_loss"),
            ("00_train/diagnostic_accuracy", "diagnostic_accuracy"),
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
        )
    else:
        primary_keys = ()
        secondary_keys = (
            ("val/total_loss", "total_loss"),
            ("val/organ_alignment_loss", "organ_alignment_loss"),
            ("val/segmentation_loss", "segmentation_loss"),
            ("val/segmentation_dice", "segmentation_dice"),
            ("val/diagnostic_loss", "diagnostic_loss"),
            ("val/diagnostic_accuracy", "diagnostic_accuracy"),
            ("val/organ_image_to_text_top1", "organ_image_to_text_top1"),
            ("val/organ_text_to_image_top1", "organ_text_to_image_top1"),
            ("val/patch_organ_presence_loss", "patch_organ_presence_loss"),
            ("val/organ_attention_loss", "organ_attention_loss"),
            ("val/lesion_organ_loss", "lesion_organ_loss"),
            ("val/segmentation_oom_fallback_count", "segmentation_oom_fallback_count"),
            ("val/step_seconds", "step_seconds"),
        )
    for alias_key, source_key in primary_keys + secondary_keys:
        if source_key in metrics and isinstance(metrics[source_key], (int, float)):
            payload[alias_key] = float(metrics[source_key])
    return payload


def _epoch_summary_payload(epoch_metrics: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    if "epoch" in epoch_metrics and isinstance(epoch_metrics["epoch"], (int, float)):
        payload["meta/epoch"] = float(epoch_metrics["epoch"])
    ordered_keys = (
        ("01_epoch/train_total_loss", "train_total_loss"),
        ("01_epoch/train_organ_alignment_loss", "train_organ_alignment_loss"),
        ("01_epoch/train_segmentation_loss", "train_segmentation_loss"),
        ("01_epoch/train_segmentation_dice", "train_segmentation_dice"),
        ("01_epoch/train_diagnostic_loss", "train_diagnostic_loss"),
        ("01_epoch/train_diagnostic_accuracy", "train_diagnostic_accuracy"),
        ("01_epoch/val_total_loss", "val_total_loss"),
        ("01_epoch/val_organ_alignment_loss", "val_organ_alignment_loss"),
        ("01_epoch/val_segmentation_loss", "val_segmentation_loss"),
        ("01_epoch/val_segmentation_dice", "val_segmentation_dice"),
        ("01_epoch/val_diagnostic_loss", "val_diagnostic_loss"),
        ("01_epoch/val_diagnostic_accuracy", "val_diagnostic_accuracy"),
        ("01_epoch/train_segmentation_oom_fallback_count", "train_segmentation_oom_fallback_count"),
        ("01_epoch/val_segmentation_oom_fallback_count", "val_segmentation_oom_fallback_count"),
    )
    for alias_key, source_key in ordered_keys:
        if source_key in epoch_metrics and isinstance(epoch_metrics[source_key], (int, float)):
            payload[alias_key] = float(epoch_metrics[source_key])
    return payload
