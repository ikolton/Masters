"""Decoder training runtime."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..config.schemas import DecoderConfig
from ..decoder.data import PerOrganDecoderDataset, decoder_collate_fn, load_decoder_samples, load_feature_store
from ..decoder.feature_cache import build_feature_store, feature_cache_path, load_or_build_feature_store
from ..decoder.model import PerOrganReportDecoder
from ..evaluation.metrics_encoder import MetricTracker
from ..runtime.distributed import barrier, destroy_distributed, is_distributed, is_main_process, maybe_init_distributed, reduce_weighted_metrics, wrap_ddp
from ..utils.io import dump_json, ensure_dir
from ..utils.seeding import set_seed
from .checkpointing import load_checkpoint, save_checkpoint, unwrap_model
from .run_logging import ExperimentLogger


def run_decoder_training(config: DecoderConfig) -> dict[str, Any]:
    distributed, _, _, _ = maybe_init_distributed()
    set_seed(config.training.seed)
    device = _resolve_device(config.training.device)
    output_dir = ensure_dir(config.resolved_output_dir)
    if is_main_process():
        dump_json(output_dir / "config_snapshot.json", config.to_dict())
    experiment_logger = ExperimentLogger.for_decoder_training(config, output_dir=output_dir)
    train_store, train_feature_summary = _load_split_store(config, split=config.data.train_split, device=device)
    val_store, val_feature_summary = _load_split_store(config, split=config.data.val_split, device=device)
    visual_dim = int(train_store.visual_dim)
    model = PerOrganReportDecoder.from_config(config, visual_dim=visual_dim).to(device)
    optimizer = _build_optimizer(model, config)
    train_loader, val_loader, dataset_summary = _build_dataloaders(config, train_store=train_store, val_store=val_store, tokenizer=unwrap_model(model).tokenizer)
    scheduler = _build_scheduler(optimizer, config, train_loader=train_loader)
    amp_dtype = _resolve_amp_dtype(config)
    use_amp = bool(config.training.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    start_epoch = 1
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
        start_epoch = int(payload.get("epoch", 0)) + 1
    model = wrap_ddp(model, find_unused_parameters=config.training.ddp_find_unused_parameters)
    history: list[dict[str, float]] = []
    best_metric: float | None = None
    for epoch in range(start_epoch, config.training.epochs + 1):
        if distributed:
            _set_sampler_epoch(train_loader, epoch)
            _set_sampler_epoch(val_loader, epoch)
        train_metrics = _run_decoder_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch,
            output_dir=output_dir,
            training=True,
            experiment_logger=experiment_logger,
            scheduler=scheduler,
        )
        val_metrics = _run_decoder_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            config=config,
            epoch=epoch,
            output_dir=output_dir,
            training=False,
            experiment_logger=experiment_logger,
            scheduler=None,
        )
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
                    scheduler=scheduler,
                    epoch=epoch,
                    config=config.to_dict(),
                    metrics=val_metrics,
                )
            metric_name = config.training.best_checkpoint_metric.replace("val_", "")
            candidate = float(val_metrics.get(metric_name, val_metrics.get("total_loss", 0.0)))
            if config.training.save_best_checkpoint and (best_metric is None or _is_better_checkpoint_metric(metric_name, candidate, best_metric)):
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
                )
        barrier()
    experiment_logger.finish()
    destroy_distributed()
    return {
        "output_dir": str(output_dir),
        "epochs_completed": len(history),
        "history": history,
        "dataset_summary": dataset_summary,
        "feature_summary": {"train": train_feature_summary, "val": val_feature_summary},
    }


def run_decoder_evaluation(config: DecoderConfig, *, checkpoint_path: str | Path, split: str) -> dict[str, Any]:
    maybe_init_distributed()
    set_seed(config.training.seed)
    device = _resolve_device(config.training.device)
    store, feature_summary = _load_split_store(config, split=split, device=device)
    model = PerOrganReportDecoder.from_config(config, visual_dim=int(store.visual_dim)).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device, strict=False)
    _, loader, dataset_summary = _build_eval_dataloader(config, store=store, tokenizer=model.tokenizer, split=split)
    metrics = _run_decoder_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        scaler=None,
        device=device,
        config=config,
        epoch=0,
        output_dir=config.resolved_output_dir,
        training=False,
        experiment_logger=ExperimentLogger(),
        scheduler=None,
    )
    destroy_distributed()
    return {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "split": split,
        "metrics": metrics,
        "dataset_summary": dataset_summary,
        "feature_summary": feature_summary,
    }


def _load_split_store(config: DecoderConfig, *, split: str, device: torch.device) -> tuple[Any, dict[str, Any]]:
    path = feature_cache_path(config, split)
    if path is None or path.is_file() or not is_distributed():
        return load_or_build_feature_store(config, split=split, device=device)
    if is_main_process():
        store, summary = build_feature_store(config, split=split, device=device)
        from ..decoder.data import save_feature_store

        save_feature_store(path, store)
        summary["feature_cache"] = str(path)
    barrier()
    store = load_feature_store(path)
    return store, {"feature_cache": str(path), "built": False}


def _build_dataloaders(config: DecoderConfig, *, train_store: Any, val_store: Any, tokenizer: Any) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_samples, train_summary = load_decoder_samples(config, split=config.data.train_split, sample_seed=config.training.seed)
    val_seed = config.training.seed if config.data.val_split == config.data.train_split else config.training.seed + 1
    val_samples, val_summary = load_decoder_samples(config, split=config.data.val_split, sample_seed=val_seed)
    train_dataset = PerOrganDecoderDataset(
        train_samples,
        feature_store=train_store,
        config=config,
        split=config.data.train_split,
        repeat_positives=True,
    )
    val_dataset = PerOrganDecoderDataset(
        val_samples,
        feature_store=val_store,
        config=config,
        split=config.data.val_split,
        repeat_positives=False,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None
    collate = decoder_collate_fn(
        tokenizer=tokenizer,
        prompt_template=config.model.prompt_template,
        visual_prefix_mode=config.model.visual_prefix_mode,
        max_length=config.model.max_length,
    )
    common = {
        "batch_size": int(config.training.batch_size),
        "num_workers": int(config.training.num_workers),
        "pin_memory": bool(config.training.pin_memory),
        "persistent_workers": bool(config.training.persistent_workers and config.training.num_workers > 0),
        "collate_fn": collate,
    }
    return (
        DataLoader(train_dataset, shuffle=train_sampler is None, sampler=train_sampler, **common),
        DataLoader(val_dataset, shuffle=False, sampler=val_sampler, **common),
        {
            "train": train_summary | {"decoder_examples": len(train_dataset)} | dict(train_dataset.summary),
            "val": val_summary | {"decoder_examples": len(val_dataset)} | dict(val_dataset.summary),
        },
    )


def _build_eval_dataloader(config: DecoderConfig, *, store: Any, tokenizer: Any, split: str) -> tuple[Any, DataLoader, dict[str, Any]]:
    sample_seed = config.training.seed if split == config.data.train_split else config.training.seed + 1
    samples, summary = load_decoder_samples(config, split=split, sample_seed=sample_seed)
    dataset = PerOrganDecoderDataset(
        samples,
        feature_store=store,
        config=config,
        split=split,
        repeat_positives=False,
    )
    sampler = DistributedSampler(dataset, shuffle=False) if is_distributed() else None
    collate = decoder_collate_fn(
        tokenizer=tokenizer,
        prompt_template=config.model.prompt_template,
        visual_prefix_mode=config.model.visual_prefix_mode,
        max_length=config.model.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(config.training.num_workers),
        pin_memory=bool(config.training.pin_memory),
        persistent_workers=bool(config.training.persistent_workers and config.training.num_workers > 0),
        collate_fn=collate,
    )
    return dataset, loader, summary | {"decoder_examples": len(dataset)} | dict(dataset.summary)


def _run_decoder_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    config: DecoderConfig,
    epoch: int,
    output_dir: Path,
    training: bool,
    experiment_logger: ExperimentLogger,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> dict[str, float]:
    tracker = MetricTracker()
    model.train(training)
    amp_dtype = _resolve_amp_dtype(config)
    use_amp = bool(config.training.amp and device.type == "cuda")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    grad_context = torch.enable_grad if training else torch.no_grad
    for step_index, batch in enumerate(loader, start=1):
        batch = _move_decoder_batch_to_device(batch, device)
        with grad_context():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(batch)
        _raise_if_nonfinite(output.total_loss, epoch=epoch, step=step_index, training=training, study_ids=batch.study_ids)
        if training and optimizer is not None and scaler is not None:
            scaler.scale(output.total_loss).backward()
            if config.training.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.training.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        metrics = dict(output.metrics)
        if training and optimizer is not None:
            metrics["lr_main"] = float(optimizer.param_groups[0]["lr"])
        tracker.update(metrics, n=len(batch.study_ids), metric_weights=_decoder_metric_weights(batch))
        save_every_steps = max(0, int(config.training.save_every_steps))
        if training and save_every_steps > 0 and step_index % save_every_steps == 0 and is_main_process():
            save_checkpoint(
                output_dir / "last_step.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                epoch=epoch,
                step=step_index,
                config=config.to_dict(),
                metrics=metrics,
            )
            print(f"[epoch {epoch}] step={step_index} saved_step_checkpoint=last_step.pt", flush=True)
        log_every = max(1, int(config.training.log_every_steps))
        if training and is_main_process() and (step_index == 1 or step_index % log_every == 0):
            print(
                f"[epoch {epoch}] step={step_index} total_loss={metrics['total_loss']:.4f} "
                f"ce={metrics['ce_loss']:.4f} diag={metrics['diagnostic_loss']:.4f} "
                f"bin_diag={metrics.get('binary_diagnostic_loss', 0.0):.4f} "
                f"sem_diag={metrics.get('semantic_diagnostic_loss', 0.0):.4f} "
                f"diag_pos={metrics['diagnostic_pathology_positive_loss']:.4f} "
                f"diag_neg={metrics['diagnostic_pathology_negative_loss']:.4f} "
                f"normal_neg={metrics['diagnostic_normal_negative_loss']:.4f} "
                f"diag_n={metrics['diagnostic_sample_count']:.0f} "
                f"sem_n={metrics.get('semantic_sample_count', 0.0):.0f}",
                flush=True,
            )
        if training:
            experiment_logger.observe_train_step()
            experiment_logger.log_step(run_label="train", metrics=metrics, epoch=epoch, step_index=step_index)
    return reduce_weighted_metrics(tracker.totals, tracker.weights, device=device)


def _build_optimizer(model: PerOrganReportDecoder, config: DecoderConfig) -> torch.optim.Optimizer:
    base_lr = float(config.training.learning_rate)
    projector_lr = base_lr if config.training.projector_learning_rate is None else float(config.training.projector_learning_rate)
    projector_params = [parameter for parameter in model.visual_projector.parameters() if parameter.requires_grad]
    projector_ids = {id(parameter) for parameter in projector_params}
    other_params = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in projector_ids]
    groups: list[dict[str, Any]] = []
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": float(config.training.weight_decay)})
    if projector_params:
        groups.append({"params": projector_params, "lr": projector_lr, "weight_decay": float(config.training.weight_decay)})
    if not groups:
        raise RuntimeError("No trainable decoder parameters were found.")
    return torch.optim.AdamW(groups)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: DecoderConfig,
    *,
    train_loader: DataLoader,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.training.scheduler_type == "none":
        return None
    if config.training.scheduler_type != "cosine":
        raise ValueError(f"Unsupported decoder scheduler_type: {config.training.scheduler_type}")
    total_steps = max(1, int(config.training.epochs) * max(1, len(train_loader)))
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


def _move_decoder_batch_to_device(batch: Any, device: torch.device) -> Any:
    return type(batch)(
        study_ids=batch.study_ids,
        organ_names=batch.organ_names,
        organ_indices=batch.organ_indices.to(device, non_blocking=True),
        visual_features=batch.visual_features.to(device, non_blocking=True),
        input_ids=batch.input_ids.to(device, non_blocking=True),
        attention_mask=batch.attention_mask.to(device, non_blocking=True),
        labels=batch.labels.to(device, non_blocking=True),
        organ_abnormal_labels=batch.organ_abnormal_labels.to(device, non_blocking=True),
        organ_abnormal_mask=batch.organ_abnormal_mask.to(device, non_blocking=True),
        lesion_labels=batch.lesion_labels.to(device, non_blocking=True),
        lesion_mask=batch.lesion_mask.to(device, non_blocking=True),
        small_bowel_mask=batch.small_bowel_mask.to(device, non_blocking=True),
        target_texts=batch.target_texts,
        semantic_statuses=batch.semantic_statuses,
        semantic_available=batch.semantic_available.to(device, non_blocking=True),
        semantic_weights=batch.semantic_weights.to(device, non_blocking=True),
        semantic_normality_targets=batch.semantic_normality_targets.to(device, non_blocking=True),
        semantic_polarity_targets=batch.semantic_polarity_targets.to(device, non_blocking=True),
        semantic_primary_subtype_targets=batch.semantic_primary_subtype_targets.to(device, non_blocking=True),
        semantic_subtype_targets=batch.semantic_subtype_targets.to(device, non_blocking=True),
        semantic_secondary_subtype_targets=batch.semantic_secondary_subtype_targets.to(device, non_blocking=True),
        semantic_allowed_subtype_mask=batch.semantic_allowed_subtype_mask.to(device, non_blocking=True),
        semantic_family_targets=batch.semantic_family_targets.to(device, non_blocking=True),
        semantic_allowed_family_mask=batch.semantic_allowed_family_mask.to(device, non_blocking=True),
    )


def _decoder_metric_weights(batch: Any) -> dict[str, float]:
    batch_count = float(len(batch.study_ids))
    diag_count = float(max(int(batch.lesion_mask.sum().item()), 1))
    semantic_count = float(max(int(batch.semantic_available.sum().item()), 1))
    combined_diag_count = float(max(int(batch.lesion_mask.sum().item()), int(batch.semantic_available.sum().item()), 1))
    return {
        "total_loss": batch_count,
        "ce_loss": batch_count,
        "diagnostic_loss": combined_diag_count,
        "binary_diagnostic_loss": diag_count,
        "diagnostic_pathology_positive_loss": diag_count,
        "diagnostic_pathology_negative_loss": diag_count,
        "diagnostic_normal_negative_loss": diag_count,
        "diagnostic_sample_count": batch_count,
        "diagnostic_positive_sample_count": batch_count,
        "diagnostic_negative_sample_count": batch_count,
        "semantic_diagnostic_loss": semantic_count,
        "semantic_diagnostic_loss_weighted": semantic_count,
        "semantic_diagnostic_loss_weighted_total": semantic_count,
        "semantic_normality_loss": semantic_count,
        "semantic_polarity_loss": semantic_count,
        "semantic_family_loss": semantic_count,
        "semantic_subtype_loss": semantic_count,
        "semantic_primary_loss": semantic_count,
        "semantic_secondary_loss": semantic_count,
        "semantic_sample_count": batch_count,
        "semantic_provisional_sample_count": batch_count,
    }


def _set_sampler_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _raise_if_nonfinite(loss: torch.Tensor, *, epoch: int, step: int, training: bool, study_ids: list[str]) -> None:
    if torch.isfinite(loss.detach()).all():
        return
    phase = "train" if training else "val"
    raise RuntimeError(f"Non-finite decoder loss during {phase} epoch={epoch} step={step}; study_ids={study_ids[:5]}")


def _is_better_checkpoint_metric(metric_name: str, candidate: float, current_best: float) -> bool:
    normalized = metric_name.strip().lower()
    if any(token in normalized for token in ("accuracy", "acc", "recall", "precision", "f1")):
        return candidate > current_best
    return candidate < current_best


def _resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
    if normalized == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device_name)


def _resolve_amp_dtype(config: DecoderConfig) -> torch.dtype:
    normalized = str(config.training.amp_dtype).strip().lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unsupported decoder.training.amp_dtype: {config.training.amp_dtype!r}")
