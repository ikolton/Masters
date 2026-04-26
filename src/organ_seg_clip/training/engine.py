"""Encoder training runtime."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any
from collections import Counter

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler

from ..config.schemas import EncoderConfig
from ..data.dataset import MerlinWholeStudyDataset, collate_whole_study_batch, load_samples_from_config
from ..evaluation.metrics_encoder import MetricTracker
from ..models import build_model
from ..models.interfaces.types import EncoderBatch, OrganSegOutput
from ..models.losses import OrganSegLossComposer
from ..runtime.distributed import barrier, destroy_distributed, get_world_size, is_distributed, is_main_process, maybe_init_distributed, reduce_weighted_metrics, wrap_ddp
from ..utils.io import dump_json, ensure_dir
from ..utils.seeding import set_seed
from .checkpointing import load_checkpoint, load_pretrained_submodule, save_checkpoint
from .run_logging import ExperimentLogger


def run_encoder_training(config: EncoderConfig) -> dict[str, Any]:
    distributed, _, _, _ = maybe_init_distributed()
    set_seed(config.training.seed)
    output_dir = ensure_dir(config.resolved_output_dir)
    if is_main_process():
        dump_json(output_dir / "config_snapshot.json", config.to_dict())
    train_samples_for_counts, _ = load_samples_from_config(
        config,
        split=config.data.train_split,
        sample_seed=config.training.seed,
    )
    train_loader, val_loader, fast_val_loader, dataset_summary = _build_dataloaders(config)
    model = build_model(config)
    device = _resolve_device(config.training.device)
    model = model.to(device)
    optimizer = _build_optimizer(model, config)
    use_amp = bool(config.training.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    resume_step = 0
    payload: dict[str, Any] = {}
    if config.training.resume_from:
        payload = load_checkpoint(config.training.resume_from, model=model, optimizer=optimizer, scaler=scaler, map_location=device, strict=False)
        resume_epoch = int(payload.get("epoch", 0))
        resume_step = int(payload.get("step") or 0)
        start_epoch = resume_epoch if resume_step > 0 else resume_epoch + 1
    elif str(config.model.segmamba.pretrained_checkpoint_path).strip():
        load_pretrained_submodule(
            config.model.segmamba.pretrained_checkpoint_path,
            model=model.patch_encoder,
            map_location=device,
            candidate_prefixes=("patch_encoder", "visual_encoder.patch_encoder", "tile_encoder", "segmamba_encoder"),
        )
    if config.runtime.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    model = wrap_ddp(model, find_unused_parameters=config.training.ddp_find_unused_parameters)
    resume_wandb_run_id = str(payload.get("wandb_run_id", "") if config.training.resume_from else "").strip()
    explicit_wandb_resume_run_id = str(config.logging.wandb_resume_run_id).strip()
    experiment_logger = ExperimentLogger.for_encoder_training(
        config,
        output_dir=output_dir,
        resume_run_id=explicit_wandb_resume_run_id or resume_wandb_run_id,
    )
    organ_finding_counts = _build_organ_finding_counts(
        train_samples_for_counts,
        organ_names=tuple(config.data.organ_names),
    )
    loss_composer = OrganSegLossComposer(config.loss, organ_finding_counts=organ_finding_counts)
    history_payload = payload.get("history", []) if config.training.resume_from else []
    history = list(history_payload) if isinstance(history_payload, list) else []
    best_metric_payload = payload.get("best_metric") if config.training.resume_from else None
    best_metric = None if best_metric_payload is None else float(best_metric_payload)
    train_steps_seen_payload = payload.get("train_steps_seen", 0) if config.training.resume_from else 0
    experiment_logger.set_train_steps_seen(int(train_steps_seen_payload))
    last_val_metrics: dict[str, float] | None = None
    for epoch in range(start_epoch, config.training.epochs + 1):
        if distributed:
            _set_sampler_epoch(train_loader, epoch)
            _set_sampler_epoch(val_loader, epoch)
            if fast_val_loader is not None:
                _set_sampler_epoch(fast_val_loader, epoch)
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            loss_composer=loss_composer,
            training=True,
            config=config,
            epoch=epoch,
            output_dir=output_dir,
            initial_step_offset=resume_step if epoch == start_epoch else 0,
            checkpoint_history=history,
            checkpoint_best_metric=best_metric,
            experiment_logger=experiment_logger,
        )
        run_full_validation = (
            int(config.training.validation_every_epochs) <= 1
            or epoch == config.training.epochs
            or epoch % int(config.training.validation_every_epochs) == 0
        )
        if run_full_validation:
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                scaler=None,
                device=device,
                loss_composer=loss_composer,
                training=False,
                config=config,
                epoch=epoch,
                output_dir=output_dir,
                run_label="val",
                skip_eval_segmentation_supervision=False,
                experiment_logger=experiment_logger,
            )
            val_metrics["validation_is_fast"] = 0.0
            last_val_metrics = dict(val_metrics)
        elif fast_val_loader is not None:
            val_metrics = _run_epoch(
                model=model,
                loader=fast_val_loader,
                optimizer=None,
                scaler=None,
                device=device,
                loss_composer=loss_composer,
                training=False,
                config=config,
                epoch=epoch,
                output_dir=output_dir,
                run_label="fast-val",
                skip_eval_segmentation_supervision=bool(config.training.fast_val_skip_segmentation),
                experiment_logger=experiment_logger,
            )
            val_metrics["validation_is_fast"] = 1.0
        else:
            val_metrics = {"validation_ran": 0.0, "validation_is_fast": 0.0}
            if last_val_metrics is not None:
                val_metrics["last_full_val_total_loss"] = float(last_val_metrics.get("total_loss", 0.0))
        epoch_metrics = {f"train_{key}": value for key, value in train_metrics.items()} | {f"val_{key}": value for key, value in val_metrics.items()}
        epoch_metrics["epoch"] = float(epoch)
        history.append(epoch_metrics)
        if is_main_process():
            dump_json(output_dir / "metrics.json", history)
            experiment_logger.log_epoch_summary(epoch_metrics=epoch_metrics)
            if config.training.save_last_checkpoint:
                save_checkpoint(
                    output_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    config=config.to_dict(),
                    metrics=val_metrics,
                    extra_state={
                        "history": history,
                        "best_metric": best_metric,
                        "train_steps_seen": experiment_logger.train_steps_seen(),
                        "wandb_run_id": experiment_logger.run_id(),
                    },
                )
            metric_name = config.training.best_checkpoint_metric.replace("val_", "")
            candidate = float(val_metrics.get(metric_name, val_metrics.get("total_loss", 0.0)))
            has_validation_metric = metric_name in val_metrics or "total_loss" in val_metrics
            should_consider_for_best = bool(run_full_validation and has_validation_metric)
            if config.training.save_best_checkpoint and should_consider_for_best and (best_metric is None or _is_better_checkpoint_metric(metric_name, candidate, best_metric)):
                best_metric = candidate
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    config=config.to_dict(),
                    metrics=val_metrics,
                    extra_state={
                        "history": history,
                        "best_metric": best_metric,
                        "train_steps_seen": experiment_logger.train_steps_seen(),
                        "wandb_run_id": experiment_logger.run_id(),
                    },
                )
        barrier()
        if epoch == start_epoch:
            resume_step = 0
    summary = {
        "output_dir": str(output_dir),
        "epochs_completed": len(history),
        "history": history,
        "dataset_summary": dataset_summary,
    }
    experiment_logger.finish()
    destroy_distributed()
    return summary


def _build_optimizer(model: torch.nn.Module, config: EncoderConfig) -> torch.optim.Optimizer:
    base_lr = float(config.training.learning_rate)
    text_lr = base_lr if config.training.text_learning_rate is None else float(config.training.text_learning_rate)
    weight_decay = float(config.training.weight_decay)
    text_module = getattr(model, "text_encoder", None)
    text_backbone = None if text_module is None else getattr(text_module, "encoder", None)
    text_backbone_params = [] if text_backbone is None else [parameter for parameter in text_backbone.parameters() if parameter.requires_grad]
    text_param_ids = {id(parameter) for parameter in text_backbone_params}
    other_params = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in text_param_ids]
    param_groups: list[dict[str, Any]] = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    if text_backbone_params:
        param_groups.append({"params": text_backbone_params, "lr": text_lr, "weight_decay": weight_decay})
    if not param_groups:
        raise RuntimeError("No trainable parameters were found when constructing the optimizer.")
    return torch.optim.AdamW(param_groups)


def _build_dataloaders(config: EncoderConfig) -> tuple[DataLoader, DataLoader, DataLoader | None, dict[str, Any]]:
    train_samples, train_summary = load_samples_from_config(config, split=config.data.train_split, sample_seed=config.training.seed)
    val_seed = config.training.seed if config.data.val_split == config.data.train_split else config.training.seed + 1
    val_samples, val_summary = load_samples_from_config(config, split=config.data.val_split, sample_seed=val_seed)
    train_dataset = MerlinWholeStudyDataset(train_samples, config=config)
    val_dataset = MerlinWholeStudyDataset(val_samples, config=config)
    fast_val_dataset = None
    fast_val_limit = config.training.fast_val_limit
    if fast_val_limit is not None and 0 < int(fast_val_limit) < len(val_samples):
        fast_val_dataset = MerlinWholeStudyDataset(val_samples[: int(fast_val_limit)], config=config)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None
    fast_val_sampler = DistributedSampler(fast_val_dataset, shuffle=False) if is_distributed() and fast_val_dataset is not None else None
    common_loader_kwargs = {
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "persistent_workers": config.training.persistent_workers if config.training.num_workers > 0 else False,
        "collate_fn": collate_whole_study_batch,
    }
    if config.training.num_workers > 0 and config.training.prefetch_factor is not None:
        common_loader_kwargs["prefetch_factor"] = int(config.training.prefetch_factor)
    train_index_sampler: Sampler[int]
    if train_sampler is not None:
        train_index_sampler = train_sampler
    else:
        train_index_sampler = RandomSampler(train_dataset)
    train_batch_sampler = _ResumableBatchSampler(
        BatchSampler(
            train_index_sampler,
            batch_size=int(config.training.batch_size),
            drop_last=False,
        )
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **common_loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        **common_loader_kwargs,
    )
    fast_val_loader = None if fast_val_dataset is None else DataLoader(
        fast_val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=fast_val_sampler,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, fast_val_loader, {"train": train_summary, "val": val_summary}


def _build_organ_finding_counts(samples: list[Any], *, organ_names: tuple[str, ...]) -> dict[tuple[int, str], int]:
    counts: Counter[tuple[int, str]] = Counter()
    for sample in samples:
        organ_text_lookup = getattr(sample, "organ_text_lookup", {})
        for organ_index, organ_name in enumerate(organ_names):
            raw_text = organ_text_lookup.get(organ_name)
            if not isinstance(raw_text, str):
                continue
            normalized = " ".join(raw_text.strip().lower().split())
            if not normalized:
                continue
            counts[(int(organ_index), normalized)] += 1
    return dict(counts)


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    loss_composer: OrganSegLossComposer,
    training: bool,
    config: EncoderConfig,
    epoch: int,
    output_dir: Path,
    run_label: str = "train",
    skip_eval_segmentation_supervision: bool = False,
    initial_step_offset: int = 0,
    checkpoint_history: list[dict[str, float]] | None = None,
    checkpoint_best_metric: float | None = None,
    experiment_logger: ExperimentLogger | None = None,
) -> dict[str, float]:
    tracker = MetricTracker()
    model.train(training)
    use_amp = bool(config.training.amp and device.type == "cuda")
    debug_steps = _parse_debug_steps_env() if training and run_label == "train" else set()
    max_steps = int(config.training.max_train_steps if training else config.training.max_val_steps)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    profile_timing = bool(config.training.profile_timing)
    if profile_timing and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    grad_context = torch.enable_grad if training else torch.no_grad
    previous_step_end = time.perf_counter()
    _set_loader_resume_batches(loader, initial_step_offset if training else 0)
    _set_eval_segmentation_supervision(model, enabled=not skip_eval_segmentation_supervision)
    segmentation_oom_fallback_total = 0
    try:
        for step_index, batch in enumerate(loader, start=1 + (initial_step_offset if training else 0)):
            if max_steps > 0 and step_index > max_steps:
                break
            data_wait_seconds = time.perf_counter() - previous_step_end
            if profile_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            step_start = time.perf_counter()
            batch = _move_batch_to_device(batch, device)
            if step_index in debug_steps:
                os.environ["ORGAN_SEG_CLIP_ACTIVE_STEP"] = str(step_index)
                if is_main_process():
                    print(f"[debug] enabling chunk memory diagnostics for train step={step_index}", flush=True)
            else:
                os.environ.pop("ORGAN_SEG_CLIP_ACTIVE_STEP", None)
            with grad_context():
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(batch)
                    loss_output, metric_output = loss_composer(outputs, batch)
            os.environ.pop("ORGAN_SEG_CLIP_ACTIVE_STEP", None)
            _raise_if_nonfinite_loss(loss_output, batch=batch, epoch=epoch, step_index=step_index, training=training)
            if training and optimizer is not None and scaler is not None:
                scaler.scale(loss_output.total_loss).backward()
                if config.training.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                _clamp_alignment_parameters(model)
                optimizer.zero_grad(set_to_none=True)
            if profile_timing and device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_start
            previous_step_end = time.perf_counter()
            scalar_metrics = loss_output.to_dict() | metric_output
            scalar_metrics["patches_per_batch_total"] = float(outputs.patches_per_batch_total)
            scalar_metrics["patches_per_study_mean"] = float(outputs.patches_per_study_mean)
            scalar_metrics["patches_per_study_max"] = float(outputs.patches_per_study_max)
            scalar_metrics["segmentation_oom_fallback_count"] = float(outputs.segmentation_oom_fallback_count)
            scalar_metrics["seconds_per_study_global"] = float(
                step_seconds / max(float(batch.images.shape[0] * get_world_size()), 1.0)
            )
            segmentation_oom_fallback_total += int(outputs.segmentation_oom_fallback_count)
            if profile_timing:
                scalar_metrics["step_seconds"] = float(step_seconds)
                scalar_metrics["data_wait_seconds"] = float(data_wait_seconds)
                if device.type == "cuda":
                    scalar_metrics["cuda_memory_allocated_gb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
                    scalar_metrics["cuda_memory_reserved_gb"] = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))
            tracker_metrics = dict(scalar_metrics)
            tracker_metrics.pop("segmentation_oom_fallback_count", None)
            tracker.update(tracker_metrics, n=int(batch.images.shape[0]), metric_weights=_build_metric_weights(batch, outputs))
            if training and experiment_logger is not None and is_main_process():
                experiment_logger.observe_train_step()
            if is_main_process() and int(outputs.segmentation_oom_fallback_count) > 0:
                print(
                    f"[{run_label} epoch {epoch}] step={step_index} segmentation_oom_fallback_count="
                    f"{int(outputs.segmentation_oom_fallback_count)}",
                    flush=True,
                )
            log_every = max(1, int(config.training.log_every_steps))
            should_log_step = step_index == 1 or (step_index % log_every == 0)
            save_every_steps = max(0, int(config.training.save_every_steps))
            should_save_step = bool(training and optimizer is not None and save_every_steps > 0 and step_index % save_every_steps == 0)
            if should_save_step and is_main_process():
                step_metrics = dict(scalar_metrics)
                step_metrics["step"] = float(step_index)
                step_metrics["partial_epoch"] = 1.0
                save_checkpoint(
                    output_dir / "last_step.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=step_index,
                    config=config.to_dict(),
                    metrics=step_metrics,
                    extra_state={
                        "history": [] if checkpoint_history is None else list(checkpoint_history),
                        "best_metric": checkpoint_best_metric,
                        "train_steps_seen": 0 if experiment_logger is None else experiment_logger.train_steps_seen(),
                        "wandb_run_id": "" if experiment_logger is None else experiment_logger.run_id(),
                    },
                )
                print(f"[epoch {epoch}] step={step_index} saved_step_checkpoint=last_step.pt", flush=True)
            if is_main_process() and should_log_step:
                if experiment_logger is not None:
                    experiment_logger.log_step(
                        run_label=run_label,
                        metrics=scalar_metrics,
                        epoch=epoch,
                        step_index=step_index,
                    )
                print(
                    f"[{run_label} epoch {epoch}] step={step_index} total_loss={scalar_metrics['total_loss']:.4f} "
                f"organ_align={scalar_metrics['organ_alignment_loss']:.4f} "
                f"seg_loss={scalar_metrics['segmentation_loss']:.4f} "
                f"seg_dice={scalar_metrics['segmentation_dice']:.4f} "
                f"diag_loss={scalar_metrics['diagnostic_loss']:.4f} "
                f"diag_acc={scalar_metrics['diagnostic_accuracy']:.4f} "
                f"report_align={scalar_metrics['report_alignment_loss']:.4f} "
                f"report_n={scalar_metrics.get('report_valid_count', 0.0):.0f} "
                f"org_gap={scalar_metrics.get('organ_logit_gap', 0.0):.3f} "
                f"org_same_gap={scalar_metrics.get('organ_same_organ_logit_gap', 0.0):.3f} "
                f"org_cross_gap={scalar_metrics.get('organ_cross_organ_logit_gap', 0.0):.3f} "
                f"rep_gap={scalar_metrics.get('report_logit_gap', 0.0):.3f} "
                f"org_s={scalar_metrics.get('organ_logit_scale', 0.0):.2f} "
                f"org_b={scalar_metrics.get('organ_logit_bias', 0.0):.2f} "
                f"rep_s={scalar_metrics.get('report_logit_scale', 0.0):.2f} "
                f"rep_b={scalar_metrics.get('report_logit_bias', 0.0):.2f} "
                f"patch_org={scalar_metrics['patch_organ_presence_loss']:.4f} "
                f"org_attn={scalar_metrics['organ_attention_loss']:.4f} "
                f"org_attn_pos_acc={scalar_metrics.get('organ_attention_positive_accuracy', 0.0):.3f} "
                f"org_attn_neg_acc={scalar_metrics.get('organ_attention_negative_accuracy', 0.0):.3f} "
                f"lesion_global={scalar_metrics['lesion_global_loss']:.4f} "
                f"lesion_organ={scalar_metrics['lesion_organ_loss']:.4f}"
                f" patches_batch={scalar_metrics.get('patches_per_batch_total', 0.0):.0f}"
                f" patches_mean={scalar_metrics.get('patches_per_study_mean', 0.0):.1f}"
                f" patches_max={scalar_metrics.get('patches_per_study_max', 0.0):.0f}"
                f" seg_oom_fallbacks={int(scalar_metrics.get('segmentation_oom_fallback_count', 0.0))}"
                f" s_per_study={scalar_metrics.get('seconds_per_study_global', 0.0):.3f}"
                f" step_s={scalar_metrics.get('step_seconds', 0.0):.2f}"
                f" data_s={scalar_metrics.get('data_wait_seconds', 0.0):.2f}"
                f" mem_alloc_gb={scalar_metrics.get('cuda_memory_allocated_gb', 0.0):.2f}"
                f" mem_reserved_gb={scalar_metrics.get('cuda_memory_reserved_gb', 0.0):.2f}",
                flush=True,
            )
    finally:
        os.environ.pop("ORGAN_SEG_CLIP_ACTIVE_STEP", None)
        _set_eval_segmentation_supervision(model, enabled=True)
    reduced_metrics = reduce_weighted_metrics(tracker.totals, tracker.weights, device=device)
    oom_tensor = torch.tensor(float(segmentation_oom_fallback_total), device=device, dtype=torch.float32)
    if is_distributed():
        torch.distributed.all_reduce(oom_tensor, op=torch.distributed.ReduceOp.SUM)
    reduced_metrics["segmentation_oom_fallback_count"] = float(oom_tensor.item())
    return reduced_metrics


def _raise_if_nonfinite_loss(
    loss_output: Any,
    *,
    batch: EncoderBatch,
    epoch: int,
    step_index: int,
    training: bool,
) -> None:
    components = {
        "total_loss": loss_output.total_loss,
        "organ_clip_loss": loss_output.organ_clip_loss,
        "organ_alignment_loss": loss_output.organ_alignment_loss,
        "segmentation_loss": loss_output.segmentation_loss,
        "diagnostic_loss": loss_output.diagnostic_loss,
        "report_clip_loss": loss_output.report_clip_loss,
        "report_alignment_loss": loss_output.report_alignment_loss,
        "patch_organ_presence_loss": loss_output.patch_organ_presence_loss,
        "organ_attention_loss": loss_output.organ_attention_loss,
        "lesion_global_loss": loss_output.lesion_global_loss,
        "lesion_organ_loss": loss_output.lesion_organ_loss,
    }
    nonfinite = [name for name, value in components.items() if not torch.isfinite(value.detach()).all()]
    if not nonfinite:
        return
    phase = "train" if training else "val"
    values = ", ".join(f"{name}={_format_loss_value(value)}" for name, value in components.items())
    study_ids = ", ".join(batch.study_ids[:5])
    raise RuntimeError(
        f"Non-finite loss during {phase} epoch={epoch} step={step_index}; "
        f"study_ids=[{study_ids}]; nonfinite={nonfinite}; {values}"
    )


def _format_loss_value(value: torch.Tensor) -> str:
    detached = value.detach().float().reshape(-1)
    if detached.numel() != 1:
        return f"shape={tuple(value.shape)}"
    return f"{float(detached.cpu().item()):.6g}"


def _build_metric_weights(batch: EncoderBatch, outputs: OrganSegOutput) -> dict[str, float]:
    batch_weight = float(batch.images.shape[0])
    global_batch_weight = float(batch.images.shape[0] * get_world_size())
    organ_text_count = float(batch.organ_text_mask.sum().item())
    organ_label_count = float(batch.organ_label_mask.sum().item())
    local_report_count = sum(1 for text in batch.report_texts if text)
    report_text_count = float(local_report_count)
    patch_organ_count = float(max(outputs.patch_organ_presence_count, 1))
    organ_attention_count = float(max(outputs.organ_attention_count, 1))
    organ_attention_positive_count = float(max(outputs.organ_attention_positive_count, 1))
    organ_attention_negative_count = float(max(outputs.organ_attention_negative_count, 1))
    lesion_global_count = float(batch.lesion_global_mask.sum().item())
    lesion_organ_count = float(batch.lesion_organ_mask.sum().item())
    segmentation_count = float(max(outputs.segmentation_patch_count, 1))
    return {
        "total_loss": batch_weight,
        "organ_clip_loss": organ_text_count,
        "organ_alignment_loss": organ_text_count,
        "segmentation_loss": segmentation_count,
        "diagnostic_loss": organ_label_count,
        "report_clip_loss": report_text_count,
        "report_alignment_loss": report_text_count,
        "patch_organ_presence_loss": patch_organ_count,
        "organ_attention_loss": organ_attention_count,
        "lesion_global_loss": lesion_global_count,
        "lesion_organ_loss": lesion_organ_count,
        "organ_image_to_text_top1": organ_text_count,
        "organ_text_to_image_top1": organ_text_count,
        "report_image_to_text_top1": report_text_count,
        "report_text_to_image_top1": report_text_count,
        "organ_positive_logit_mean": organ_text_count,
        "organ_negative_logit_mean": organ_text_count,
        "organ_logit_gap": organ_text_count,
        "organ_same_organ_negative_logit_mean": organ_text_count,
        "organ_cross_organ_negative_logit_mean": organ_text_count,
        "organ_same_organ_logit_gap": organ_text_count,
        "organ_cross_organ_logit_gap": organ_text_count,
        "organ_logit_scale": organ_text_count,
        "organ_logit_bias": organ_text_count,
        "report_positive_logit_mean": report_text_count,
        "report_negative_logit_mean": report_text_count,
        "report_logit_gap": report_text_count,
        "report_logit_scale": report_text_count,
        "report_logit_bias": report_text_count,
        "diagnostic_accuracy": organ_label_count,
        "patch_organ_presence_accuracy": patch_organ_count,
        "organ_attention_accuracy": organ_attention_count,
        "organ_attention_positive_accuracy": organ_attention_positive_count,
        "organ_attention_negative_accuracy": organ_attention_negative_count,
        "lesion_global_accuracy": lesion_global_count,
        "lesion_organ_accuracy": lesion_organ_count,
        "segmentation_dice": segmentation_count,
        "patches_per_batch_total": batch_weight,
        "patches_per_study_mean": batch_weight,
        "patches_per_study_max": batch_weight,
        "seconds_per_study_global": global_batch_weight,
        "step_seconds": batch_weight,
        "data_wait_seconds": batch_weight,
        "cuda_memory_allocated_gb": batch_weight,
        "cuda_memory_reserved_gb": batch_weight,
    }


def _clamp_alignment_parameters(model: torch.nn.Module) -> None:
    target = model.module if hasattr(model, "module") else model
    clamp = getattr(target, "clamp_alignment_parameters", None)
    if clamp is not None:
        clamp()


def _move_batch_to_device(batch: EncoderBatch, device: torch.device) -> EncoderBatch:
    return EncoderBatch(
        study_ids=batch.study_ids,
        images=batch.images.to(device, non_blocking=True),
        image_mask=batch.image_mask.to(device, non_blocking=True),
        segmentations=None if batch.segmentations is None else batch.segmentations.to(device, non_blocking=True),
        segmentation_mask=None if batch.segmentation_mask is None else batch.segmentation_mask.to(device, non_blocking=True),
        report_texts=batch.report_texts,
        organ_texts=batch.organ_texts,
        organ_raw_texts=batch.organ_raw_texts,
        organ_text_mask=batch.organ_text_mask.to(device, non_blocking=True),
        organ_labels=batch.organ_labels.to(device, non_blocking=True),
        organ_label_mask=batch.organ_label_mask.to(device, non_blocking=True),
        lesion_global_labels=batch.lesion_global_labels.to(device, non_blocking=True),
        lesion_global_mask=batch.lesion_global_mask.to(device, non_blocking=True),
        lesion_organ_labels=batch.lesion_organ_labels.to(device, non_blocking=True),
        lesion_organ_mask=batch.lesion_organ_mask.to(device, non_blocking=True),
        metadata=batch.metadata,
    )


def _set_eval_segmentation_supervision(model: torch.nn.Module, *, enabled: bool) -> None:
    target = model.module if hasattr(model, "module") else model
    setter = getattr(target, "set_eval_segmentation_supervision", None)
    if setter is not None:
        setter(enabled)


def _set_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = _resolve_loader_sampler(loader)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _set_loader_resume_batches(loader: DataLoader, skip_batches: int) -> None:
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_skip_batches"):
        batch_sampler.set_skip_batches(skip_batches)


def _resolve_loader_sampler(loader: DataLoader) -> Sampler[Any] | None:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        return sampler
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is None:
        return sampler
    nested_sampler = getattr(batch_sampler, "sampler", None)
    if nested_sampler is not None and hasattr(nested_sampler, "set_epoch"):
        return nested_sampler
    nested_batch_sampler = getattr(batch_sampler, "batch_sampler", None)
    if nested_batch_sampler is None:
        return nested_sampler if isinstance(nested_sampler, Sampler) else sampler
    nested_sampler = getattr(nested_batch_sampler, "sampler", None)
    if nested_sampler is not None and hasattr(nested_sampler, "set_epoch"):
        return nested_sampler
    return nested_sampler if isinstance(nested_sampler, Sampler) else sampler


class _ResumableBatchSampler:
    def __init__(self, batch_sampler: BatchSampler[list[int]] | BatchSampler) -> None:
        self.batch_sampler = batch_sampler
        self.sampler = getattr(batch_sampler, "sampler", None)
        self._skip_batches = 0

    def set_skip_batches(self, skip_batches: int) -> None:
        self._skip_batches = max(int(skip_batches), 0)

    def __iter__(self):
        skip_batches = self._skip_batches
        self._skip_batches = 0
        iterator = iter(self.batch_sampler)
        for _ in range(skip_batches):
            try:
                next(iterator)
            except StopIteration:
                return
        for batch in iterator:
            yield batch

    def __len__(self) -> int:
        return max(len(self.batch_sampler) - self._skip_batches, 0)


def _is_better_checkpoint_metric(metric_name: str, candidate: float, current_best: float) -> bool:
    normalized = metric_name.strip().lower()
    higher_is_better_tokens = ("top1", "dice", "accuracy", "acc", "auc", "f1", "precision", "recall")
    if any(token in normalized for token in higher_is_better_tokens):
        return candidate > current_best
    return candidate < current_best


def _resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device_name!r}, but CUDA is not available. Refusing to fall back to CPU."
        )
    return torch.device(device_name)


def _parse_debug_steps_env() -> set[int]:
    raw = os.environ.get("ORGAN_SEG_CLIP_DEBUG_STEPS", "").strip()
    if not raw:
        return set()
    steps: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, end_text = token.split(":", 1)
            start = int(start_text)
            end = int(end_text)
            lo, hi = sorted((start, end))
            steps.update(range(lo, hi + 1))
        else:
            steps.add(int(token))
    return steps
