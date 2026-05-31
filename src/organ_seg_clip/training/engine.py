"""Encoder training runtime."""

from __future__ import annotations

import os
from pathlib import Path
import math
import random
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
from ..models.losses.contrastive import _normalize_text_label as _normalize_finding_label


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
    train_loader, val_loader, fast_val_loader, val_samples, dataset_summary = _build_dataloaders(config)
    model = build_model(config)
    device = _resolve_device(config.training.device)
    model = model.to(device)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config, train_loader=train_loader)
    use_amp = bool(config.training.amp and device.type == "cuda")
    amp_dtype = _resolve_amp_dtype(config)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    start_epoch = 1
    resume_step = 0
    payload: dict[str, Any] = {}
    if config.training.resume_from:
        payload = load_checkpoint(
            config.training.resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            map_location=device,
            strict=False,
        )
        resume_epoch = int(payload.get("epoch", 0))
        resume_step = int(payload.get("step") or 0)
        start_epoch = resume_epoch if resume_step > 0 else resume_epoch + 1
    elif str(config.training.warm_start_from or "").strip():
        if float(config.loss.organ_alignment_weight or 0.0) == 0.0 and is_main_process():
            print(
                "[warning] warm_start_from is set but organ_alignment_weight=0 — "
                "warm-starting into a stage1 run is unusual.",
                flush=True,
            )
        load_checkpoint(
            config.training.warm_start_from,
            model=model,
            map_location=device,
            strict=False,
            restore_rng=False,
        )
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
    epochs_without_improvement = 0
    early_stopping_patience = int(config.training.early_stopping_patience)
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
            scheduler=scheduler,
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
        elif config.training.fast_val_limit is not None:
            if config.training.fast_val_sampling == "epoch_random":
                fast_val_loader = _build_fast_val_loader(config, val_samples, epoch=epoch)
                if distributed and fast_val_loader is not None:
                    _set_sampler_epoch(fast_val_loader, epoch)
            if fast_val_loader is not None:
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
                    run_label="smoke-val",
                    skip_eval_segmentation_supervision=bool(config.training.fast_val_skip_segmentation),
                    experiment_logger=experiment_logger,
                )
                val_metrics["validation_is_fast"] = 1.0
            else:
                val_metrics = {"validation_ran": 0.0, "validation_is_fast": 1.0}
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
                run_label="smoke-val",
                skip_eval_segmentation_supervision=bool(config.training.fast_val_skip_segmentation),
                experiment_logger=experiment_logger,
            )
            val_metrics["validation_is_fast"] = 1.0
        else:
            val_metrics = {"validation_ran": 0.0, "validation_is_fast": 0.0}
            if last_val_metrics is not None:
                val_metrics["last_full_val_total_loss"] = float(last_val_metrics.get("total_loss", 0.0))
        validation_prefix = "full_val" if run_full_validation else "smoke_val"
        epoch_metrics = {f"train_{key}": value for key, value in train_metrics.items()} | {f"{validation_prefix}_{key}": value for key, value in val_metrics.items()}
        epoch_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
        epoch_metrics["epoch"] = float(epoch)
        epoch_metrics["validation_kind_full"] = 1.0 if run_full_validation else 0.0
        history.append(epoch_metrics)
        if is_main_process():
            dump_json(output_dir / "metrics.json", history)
            experiment_logger.log_epoch_summary(epoch_metrics=epoch_metrics)
            metric_name = _normalize_best_metric_name(config.training.best_checkpoint_metric)
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
                    scheduler=scheduler,
                    epoch=epoch,
                    config=config.to_dict(),
                    metrics=val_metrics,
                    extra_state={
                        "history": history,
                        "best_metric": best_metric,
                        "best_metric_source": "full_val",
                        "train_steps_seen": experiment_logger.train_steps_seen(),
                        "wandb_run_id": experiment_logger.run_id(),
                    },
                )
            if run_full_validation:
                save_checkpoint(
                    output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    epoch=epoch,
                    config=config.to_dict(),
                    metrics=val_metrics,
                    extra_state={
                        "history": history,
                        "best_metric": best_metric,
                        "best_metric_source": "full_val",
                        "train_steps_seen": experiment_logger.train_steps_seen(),
                        "wandb_run_id": experiment_logger.run_id(),
                    },
                )
            if config.training.save_last_checkpoint:
                save_checkpoint(
                    output_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    epoch=epoch,
                    config=config.to_dict(),
                    metrics=val_metrics,
                    extra_state={
                        "history": history,
                        "best_metric": best_metric,
                        "best_metric_source": "full_val",
                        "train_steps_seen": experiment_logger.train_steps_seen(),
                        "wandb_run_id": experiment_logger.run_id(),
                    },
                )
            _flush_text_embedding_cache(model)
        barrier()
        if early_stopping_patience > 0:
            if run_full_validation:
                metric_name = _normalize_best_metric_name(config.training.best_checkpoint_metric)
                candidate = float(val_metrics.get(metric_name, val_metrics.get("total_loss", 0.0)))
                improved = best_metric is None or _is_better_checkpoint_metric(metric_name, candidate, best_metric)
                if improved:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if is_main_process() and epochs_without_improvement > 0:
                    print(
                        f"[early stopping] epochs_without_improvement={epochs_without_improvement}/{early_stopping_patience}",
                        flush=True,
                    )
            stop_tensor = torch.tensor(
                1.0 if (epochs_without_improvement >= early_stopping_patience and run_full_validation) else 0.0,
                device=device,
                dtype=torch.float32,
            )
            if is_distributed():
                torch.distributed.all_reduce(stop_tensor, op=torch.distributed.ReduceOp.MAX)
            if stop_tensor.item() >= 1.0:
                if is_main_process():
                    print(
                        f"[early stopping] stopping at epoch {epoch} after {epochs_without_improvement} epochs without improvement.",
                        flush=True,
                    )
                break
        freeze_after = config.training.freeze_text_projection_after_epoch
        if freeze_after is not None and epoch == int(freeze_after):
            _freeze_text_projection(model)
            if is_main_process():
                print(f"[freeze_text_projection] freezing text projection after epoch {epoch}", flush=True)
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
    alignment_lr = config.training.alignment_parameter_learning_rate
    weight_decay = float(config.training.weight_decay)
    patch_encoder_scale = float(config.training.patch_encoder_learning_rate_scale)
    text_module = getattr(model, "text_encoder", None)
    text_backbone = None if text_module is None else getattr(text_module, "encoder", None)
    text_backbone_params = [] if text_backbone is None else [parameter for parameter in text_backbone.parameters() if parameter.requires_grad]
    text_param_ids = {id(parameter) for parameter in text_backbone_params}
    alignment_names = tuple(str(name) for name in config.training.alignment_parameter_names)
    alignment_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and alignment_lr is not None and any(name == target or name.endswith(f".{target}") for target in alignment_names)
    ]
    alignment_param_ids = {id(parameter) for parameter in alignment_params}
    _backbone_prefixes = ("patch_encoder.", "patch_segmentation_head.")
    backbone_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(name.startswith(prefix) for prefix in _backbone_prefixes)
        and id(parameter) not in alignment_param_ids
    ]
    backbone_param_ids = {id(parameter) for parameter in backbone_params}
    other_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in text_param_ids
        and id(parameter) not in alignment_param_ids
        and id(parameter) not in backbone_param_ids
    ]
    param_groups: list[dict[str, Any]] = []
    if backbone_params and patch_encoder_scale != 1.0:
        backbone_lr = base_lr * patch_encoder_scale
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "patch_encoder"})
    elif backbone_params:
        other_params = backbone_params + other_params
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay, "name": "main"})
    if text_backbone_params:
        param_groups.append({"params": text_backbone_params, "lr": text_lr, "weight_decay": weight_decay, "name": "text"})
    if alignment_params:
        param_groups.append({"params": alignment_params, "lr": float(alignment_lr), "weight_decay": weight_decay, "name": "alignment_parameters"})
    if not param_groups:
        raise RuntimeError("No trainable parameters were found when constructing the optimizer.")
    return torch.optim.AdamW(param_groups)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: EncoderConfig,
    *,
    train_loader: DataLoader,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.training.scheduler_type == "none":
        return None
    if config.training.scheduler_type != "cosine":
        raise ValueError(f"Unsupported scheduler_type: {config.training.scheduler_type}")
    steps_per_epoch = len(train_loader)
    if int(config.training.max_train_steps) > 0:
        steps_per_epoch = min(steps_per_epoch, int(config.training.max_train_steps))
    total_steps = max(1, int(config.training.epochs) * max(1, steps_per_epoch))
    warmup_steps = min(int(config.training.warmup_steps), total_steps)
    min_lr = float(config.training.min_learning_rate)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def make_lr_lambda(base_lr: float):
        def lr_lambda(step: int) -> float:
            step = max(int(step), 0)
            min_factor = 0.0 if base_lr <= 0.0 else min(min_lr / base_lr, 1.0)
            if warmup_steps > 0 and step < warmup_steps:
                factor = float(step + 1) / float(warmup_steps)
            else:
                decay_steps = max(total_steps - warmup_steps, 1)
                progress = min(max(float(step - warmup_steps) / float(decay_steps), 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
                factor = min_factor + (1.0 - min_factor) * cosine
            return float(factor)
        return lr_lambda

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[make_lr_lambda(base_lr) for base_lr in base_lrs])


def _build_dataloaders(config: EncoderConfig) -> tuple[DataLoader, DataLoader, DataLoader | None, list[Any], dict[str, Any]]:
    train_samples, train_summary = load_samples_from_config(config, split=config.data.train_split, sample_seed=config.training.seed)
    val_seed = config.training.seed if config.data.val_split == config.data.train_split else config.training.seed + 1
    val_samples, val_summary = load_samples_from_config(config, split=config.data.val_split, sample_seed=val_seed)
    train_dataset = MerlinWholeStudyDataset(train_samples, config=config)
    val_dataset = MerlinWholeStudyDataset(val_samples, config=config)
    fast_val_loader = _build_fast_val_loader(config, val_samples, epoch=None)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None
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
            drop_last=is_distributed(),
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
    return train_loader, val_loader, fast_val_loader, val_samples, {"train": train_summary, "val": val_summary}


def _build_fast_val_loader(config: EncoderConfig, val_samples: list[Any], *, epoch: int | None) -> DataLoader | None:
    fast_val_limit = config.training.fast_val_limit
    if fast_val_limit is None or int(fast_val_limit) <= 0 or not val_samples:
        return None
    limit = min(int(fast_val_limit), len(val_samples))
    if config.training.fast_val_sampling == "epoch_random" and epoch is not None:
        rng = random.Random(int(config.training.seed) + int(epoch))
        indices = rng.sample(range(len(val_samples)), k=limit)
        fast_samples = [val_samples[index] for index in indices]
    else:
        fast_samples = val_samples[:limit]
    fast_val_dataset = MerlinWholeStudyDataset(fast_samples, config=config)
    fast_val_sampler = DistributedSampler(fast_val_dataset, shuffle=False) if is_distributed() else None
    common_loader_kwargs = {
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "persistent_workers": config.training.persistent_workers if config.training.num_workers > 0 else False,
        "collate_fn": collate_whole_study_batch,
    }
    if config.training.num_workers > 0 and config.training.prefetch_factor is not None:
        common_loader_kwargs["prefetch_factor"] = int(config.training.prefetch_factor)
    return DataLoader(
        fast_val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=fast_val_sampler,
        **common_loader_kwargs,
    )


def _build_organ_finding_counts(samples: list[Any], *, organ_names: tuple[str, ...]) -> dict[tuple[int, str], int]:
    counts: Counter[tuple[int, str]] = Counter()
    for sample in samples:
        organ_text_lookup = getattr(sample, "organ_text_lookup", {})
        for organ_index, organ_name in enumerate(organ_names):
            raw_text = organ_text_lookup.get(organ_name)
            if not isinstance(raw_text, str):
                continue
            normalized = _normalize_finding_label(raw_text)
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    tracker = MetricTracker()
    model.train(training)
    use_amp = bool(config.training.amp and device.type == "cuda")
    amp_dtype = _resolve_amp_dtype(config)
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
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(batch)
                    loss_output, metric_output = loss_composer(outputs, batch)
            os.environ.pop("ORGAN_SEG_CLIP_ACTIVE_STEP", None)
            _raise_if_nonfinite_loss(
                loss_output,
                outputs=outputs,
                batch=batch,
                epoch=epoch,
                step_index=step_index,
                training=training,
            )
            if training and optimizer is not None and scaler is not None:
                scaler.scale(loss_output.total_loss).backward()
                if config.training.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                _clamp_alignment_parameters(model)
                if scheduler is not None:
                    scheduler.step()
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
            scalar_metrics["segmentation_foreground_patch_count"] = float(outputs.segmentation_foreground_patch_count)
            for group_index, group in enumerate(optimizer.param_groups if optimizer is not None else ()):
                group_name = str(group.get("name", f"group_{group_index}"))
                scalar_metrics[f"lr_{group_name}"] = float(group.get("lr", 0.0))
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
                    scheduler=scheduler,
                    epoch=epoch,
                    step=step_index,
                    config=config.to_dict(),
                    metrics=step_metrics,
                    extra_state={
                        "history": [] if checkpoint_history is None else list(checkpoint_history),
                        "best_metric": checkpoint_best_metric,
                        "best_metric_source": "full_val",
                        "train_steps_seen": 0 if experiment_logger is None else experiment_logger.train_steps_seen(),
                        "wandb_run_id": "" if experiment_logger is None else experiment_logger.run_id(),
                    },
                )
                print(f"[epoch {epoch}] step={step_index} saved_step_checkpoint=last_step.pt", flush=True)
            if is_main_process() and should_log_step:
                if training and experiment_logger is not None:
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
                f"seg_fg_dice={scalar_metrics.get('segmentation_foreground_dice', 0.0):.4f} "
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
    outputs: OrganSegOutput,
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
    output_diagnostics = _format_nonfinite_output_diagnostics(outputs)
    study_ids = ", ".join(batch.study_ids[:5])
    raise RuntimeError(
        f"Non-finite loss during {phase} epoch={epoch} step={step_index}; "
        f"study_ids=[{study_ids}]; nonfinite={nonfinite}; {values}; {output_diagnostics}"
    )


def _format_loss_value(value: torch.Tensor) -> str:
    detached = value.detach().float().reshape(-1)
    if detached.numel() != 1:
        return f"shape={tuple(value.shape)}"
    return f"{float(detached.cpu().item()):.6g}"


def _format_nonfinite_output_diagnostics(outputs: OrganSegOutput) -> str:
    summaries: list[str] = []
    for name, value in vars(outputs).items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value.detach()).all():
            summaries.append(f"{name}={_format_tensor_finiteness(value)}")
    if summaries:
        return "nonfinite_outputs=[" + "; ".join(summaries) + "]"
    return "nonfinite_outputs=[]"


def _format_tensor_finiteness(value: torch.Tensor) -> str:
    detached = value.detach()
    flat = detached.reshape(-1)
    finite = torch.isfinite(flat)
    nan_count = int(torch.isnan(flat).sum().cpu().item())
    posinf_count = int(torch.isposinf(flat).sum().cpu().item())
    neginf_count = int(torch.isneginf(flat).sum().cpu().item())
    finite_count = int(finite.sum().cpu().item())
    total_count = int(flat.numel())
    if finite_count > 0:
        finite_values = flat[finite].float()
        finite_min = float(finite_values.min().cpu().item())
        finite_max = float(finite_values.max().cpu().item())
        finite_range = f"finite_min={finite_min:.6g},finite_max={finite_max:.6g}"
    else:
        finite_range = "finite_min=none,finite_max=none"
    return (
        f"shape={tuple(detached.shape)},dtype={detached.dtype},"
        f"finite={finite_count}/{total_count},nan={nan_count},"
        f"+inf={posinf_count},-inf={neginf_count},{finite_range}"
    )


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
        "segmentation_foreground_dice": float(max(outputs.segmentation_foreground_patch_count, 1)),
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


def _freeze_text_projection(model: torch.nn.Module) -> None:
    target = model.module if hasattr(model, "module") else model
    text_encoder = getattr(target, "text_encoder", None)
    if text_encoder is not None and hasattr(text_encoder, "freeze_projection"):
        text_encoder.freeze_projection()


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


def _flush_text_embedding_cache(model: torch.nn.Module) -> None:
    underlying = model.module if hasattr(model, "module") else model
    text_encoder = getattr(underlying, "text_encoder", None)
    if text_encoder is not None and hasattr(text_encoder, "save_disk_cache"):
        text_encoder.save_disk_cache()


def _is_better_checkpoint_metric(metric_name: str, candidate: float, current_best: float) -> bool:
    normalized = metric_name.strip().lower()
    higher_is_better_tokens = ("top1", "dice", "accuracy", "acc", "auc", "f1", "precision", "recall")
    if any(token in normalized for token in higher_is_better_tokens):
        return candidate > current_best
    return candidate < current_best


def _normalize_best_metric_name(metric_name: str) -> str:
    normalized = str(metric_name).strip()
    for prefix in ("full_val_", "smoke_val_", "val_"):
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return normalized


def _resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device_name!r}, but CUDA is not available. Refusing to fall back to CPU."
        )
    return torch.device(device_name)


def _resolve_amp_dtype(config: EncoderConfig) -> torch.dtype:
    return torch.bfloat16 if config.training.amp_dtype == "bfloat16" else torch.float16


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
