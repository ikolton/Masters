"""Training loop for Merlin ablations."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader as TorchDataLoader

MASTERS_SRC = Path(__file__).resolve().parents[4] / "src"
if str(MASTERS_SRC) not in sys.path:
    sys.path.insert(0, str(MASTERS_SRC))

from organ_seg_clip.config.schemas import DecoderDiagnosticLossConfig
from organ_seg_clip.decoder.losses import BinaryDiagnosticLoss

from .config import AblationConfig
from .cache import load_cached_image_features
from .data import build_datasets
from .losses import AuxiliaryDiagnosticLosses
from .modeling import MerlinReportTrainingWrapper, trainable_parameter_summary


def run_training(config: AblationConfig) -> dict[str, Any]:
    _prepare_imports(config)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.resolved.json", _jsonable(config.raw))

    datasets = build_datasets(config)
    train_loader = _make_loader(config, datasets.train_records, split_name="train", shuffle=True)
    val_loader = _make_loader(config, datasets.val_records, split_name="val", shuffle=False)

    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    model = MerlinReportTrainingWrapper(config).to(device)
    family_count = 0 if datasets.semantic_lookup is None else len(datasets.semantic_lookup.spec.family_vocab)
    subtype_count = 0 if datasets.semantic_lookup is None else len(datasets.semantic_lookup.spec.subtype_vocab)
    aux_losses = AuxiliaryDiagnosticLosses(
        config.losses,
        hidden_size=model.hidden_size,
        family_count=family_count,
        subtype_count=subtype_count,
    ).to(device)
    concept_loss = _build_concept_diagnostic_loss(config, model.tokenizer, device)
    parameters = [p for p in list(model.parameters()) + list(aux_losses.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    autocast_dtype = torch.bfloat16 if config.train.mixed_precision == "bf16" else torch.float16

    manifest = {
        "run_id": config.train.run_id,
        "output_dir": str(output_dir),
        "paths": _jsonable(config.raw.get("paths", {})),
        "model": _jsonable(config.raw.get("model", {})),
        "train": _jsonable(config.raw.get("train", {})),
        "losses": _jsonable(config.raw.get("losses", {})),
        "dataset_summary": datasets.summary,
        "model_parameters": trainable_parameter_summary(model),
        "aux_parameters": trainable_parameter_summary(aux_losses),
        "resume_from_checkpoint": None if config.train.resume_from_checkpoint is None else str(config.train.resume_from_checkpoint),
    }
    _write_json(output_dir / "manifest.json", manifest)

    metrics_rows: list[dict[str, Any]] = []
    global_step = 0
    best_val_loss = math.inf
    best_epoch: int | None = None
    val_metrics: dict[str, Any] | None = None
    start_epoch = 0
    if config.train.resume_from_checkpoint is not None:
        loaded = _load_training_checkpoint(
            config.train.resume_from_checkpoint,
            model=model,
            aux_losses=aux_losses,
            optimizer=optimizer,
            device=device,
        )
        global_step = int(loaded["step"])
        start_epoch = int(loaded["epoch"]) + 1
        val_metrics = loaded.get("val_metrics")
        if isinstance(val_metrics, dict):
            current_val_loss = float(val_metrics.get("loss", math.inf))
            if math.isfinite(current_val_loss):
                best_val_loss = current_val_loss
                best_epoch = int(loaded["epoch"])
                if config.train.save_checkpoint:
                    _save_checkpoint(
                        output_dir / "checkpoint_best.pt",
                        model=model,
                        aux_losses=aux_losses,
                        optimizer=optimizer,
                        config=config,
                        epoch=best_epoch,
                        step=global_step,
                        val_metrics=val_metrics,
                    )
        print(
            "[merlin-ablation] "
            f"resumed checkpoint={config.train.resume_from_checkpoint} "
            f"start_epoch={start_epoch} step={global_step} best_val_loss={best_val_loss:.6f}",
            flush=True,
        )
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    aux_losses.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.train.epochs):
        for batch in train_loader:
            global_step += 1
            prompts = _string_list(_batch_value(batch, "prompt"))
            full_texts = _string_list(_batch_value(batch, "full_text"))
            images = None
            image_features = None
            if config.model.image_embedding_mode == "cached":
                image_features = load_cached_image_features(batch, device)
            else:
                images = _batch_value(batch, "image").to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                output = model(images=images, image_features=image_features, prompts=prompts, full_texts=full_texts)
                aux_output = aux_losses(output.pooled_hidden, batch)
                concept_output = _concept_loss_output(concept_loss, output, batch, device)
                loss = float(config.losses.ce_weight) * output.ce_loss + aux_output.total + concept_output.loss
                loss = loss / max(int(config.train.grad_accum_steps), 1)
            scaler.scale(loss).backward()
            if global_step % config.train.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if global_step % config.train.log_every == 0:
                elapsed = max(time.time() - start, 1.0e-6)
                row = {
                    "phase": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float((loss * max(int(config.train.grad_accum_steps), 1)).detach().cpu().item()),
                    "ce_loss": float(output.ce_loss.detach().cpu().item()),
                    "examples_per_second": float(global_step * config.train.batch_size / elapsed),
                }
                row.update(_cuda_memory_metrics(device))
                row.update(aux_output.metrics())
                row.update(_concept_metrics(concept_output))
                metrics_rows.append(row)
                print(
                    "[merlin-ablation] "
                    f"step={global_step} loss={row['loss']:.4f} ce={row['ce_loss']:.4f} "
                    f"lex={row['lexical_loss']:.4f} concept_lex={row['concept_diagnostic_loss_weighted']:.4f} "
                    f"sem={row['semantic_loss']:.4f} "
                    f"rate={row['examples_per_second']:.3f}/s",
                    flush=True,
                )
            if config.train.max_steps is not None and global_step >= config.train.max_steps:
                break
        if global_step % max(int(config.train.grad_accum_steps), 1) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        should_eval = (epoch + 1) % max(int(config.train.eval_every_epochs), 1) == 0
        if should_eval:
            val_metrics = _evaluate(model, aux_losses, concept_loss, val_loader, config, device)
            val_metrics = dict(val_metrics) | {"epoch": epoch, "step": global_step}
            metrics_rows.append(val_metrics)
            _write_jsonl(output_dir / "metrics.jsonl", metrics_rows)
            print(
                "[merlin-ablation] "
                f"epoch={epoch} val_loss={val_metrics['loss']:.4f} val_ce={val_metrics['ce_loss']:.4f}",
                flush=True,
            )
            if config.train.save_checkpoint:
                _save_checkpoint(
                    output_dir / "checkpoint_last.pt",
                    model=model,
                    aux_losses=aux_losses,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    step=global_step,
                    val_metrics=val_metrics,
                )
                current_val_loss = float(val_metrics.get("loss", math.inf))
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    _save_checkpoint(
                        output_dir / "checkpoint_best.pt",
                        model=model,
                        aux_losses=aux_losses,
                        optimizer=optimizer,
                        config=config,
                        epoch=epoch,
                        step=global_step,
                        val_metrics=val_metrics,
                    )
        if config.train.max_steps is not None and global_step >= config.train.max_steps:
            break

    if val_metrics is None:
        val_metrics = _evaluate(model, aux_losses, concept_loss, val_loader, config, device)
        val_metrics = dict(val_metrics) | {"epoch": -1, "step": global_step}
        metrics_rows.append(val_metrics)
    _write_jsonl(output_dir / "metrics.jsonl", metrics_rows)
    if config.train.save_checkpoint:
        last_checkpoint = output_dir / "checkpoint_last.pt"
        if last_checkpoint.is_file():
            _replace_with_hardlink_or_save(
                source=last_checkpoint,
                target=output_dir / "checkpoint.pt",
                model=model,
                aux_losses=aux_losses,
                optimizer=optimizer,
                config=config,
                epoch=int(val_metrics.get("epoch", -1)),
                step=global_step,
                val_metrics=val_metrics,
            )
        else:
            _save_checkpoint(
                output_dir / "checkpoint.pt",
                model=model,
                aux_losses=aux_losses,
                optimizer=optimizer,
                config=config,
                epoch=int(val_metrics.get("epoch", -1)),
                step=global_step,
                val_metrics=val_metrics,
            )
    summary = {
        "final_step": global_step,
        "val": val_metrics,
        "best_val_loss": None if not math.isfinite(best_val_loss) else best_val_loss,
        "best_epoch": best_epoch,
        "manifest": manifest,
        "cuda_memory": _cuda_memory_metrics(device),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


@torch.no_grad()
def _evaluate(model, aux_losses, concept_loss, loader, config: AblationConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    aux_losses.eval()
    totals: dict[str, float] = {}
    count = 0
    autocast_dtype = torch.bfloat16 if config.train.mixed_precision == "bf16" else torch.float16
    for batch in loader:
        prompts = _string_list(_batch_value(batch, "prompt"))
        full_texts = _string_list(_batch_value(batch, "full_text"))
        images = None
        image_features = None
        if config.model.image_embedding_mode == "cached":
            image_features = load_cached_image_features(batch, device)
            batch_size = int(image_features.shape[0])
        else:
            images = _batch_value(batch, "image").to(device, non_blocking=True)
            batch_size = int(images.shape[0])
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            output = model(images=images, image_features=image_features, prompts=prompts, full_texts=full_texts)
            aux_output = aux_losses(output.pooled_hidden, batch)
            concept_output = _concept_loss_output(concept_loss, output, batch, device)
            loss = float(config.losses.ce_weight) * output.ce_loss + aux_output.total + concept_output.loss
        row = {"loss": float(loss.detach().cpu().item()), "ce_loss": float(output.ce_loss.detach().cpu().item())}
        row.update(aux_output.metrics())
        row.update(_concept_metrics(concept_output))
        count += batch_size
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
    model.train()
    aux_losses.train()
    return {"phase": "val", "count": count, **{key: value / max(count, 1) for key, value in totals.items()}}


def _build_concept_diagnostic_loss(config: AblationConfig, tokenizer, device: torch.device) -> BinaryDiagnosticLoss | None:
    if config.losses.lexical_mode != "concept_specific" or config.losses.lexical_weight <= 0.0:
        return None
    if config.losses.lexical_target_cache is None:
        raise ValueError("losses.lexical_target_cache is required when lexical_mode=concept_specific.")
    diagnostic_config = DecoderDiagnosticLossConfig(
        enabled=True,
        variant="concept_specific_lexical",
        weight=float(config.losses.lexical_weight),
        lexical_target_cache=str(config.losses.lexical_target_cache),
        positive_pathology_weight=float(config.losses.positive_pathology_weight),
        negative_pathology_weight=float(config.losses.negative_pathology_weight),
        negative_temperature=float(config.losses.negative_temperature),
        epsilon=float(config.losses.epsilon),
    )
    return BinaryDiagnosticLoss(diagnostic_config, tokenizer).to(device)


def _concept_loss_output(concept_loss, output, batch: dict[str, Any], device: torch.device):
    if concept_loss is None:
        zero = output.ce_loss * 0.0
        from organ_seg_clip.decoder.losses import DiagnosticLossOutput

        return DiagnosticLossOutput(zero, zero, zero, zero, zero, 0, 0, 0)
    batch_size = int(output.logits.shape[0])
    zeros = torch.zeros(batch_size, device=device)
    return concept_loss(
        logits=output.logits,
        labels=output.labels,
        lesion_labels=zeros,
        lesion_mask=torch.ones(batch_size, dtype=torch.bool, device=device),
        small_bowel_mask=torch.zeros(batch_size, dtype=torch.bool, device=device),
        target_texts=_string_list(_batch_value(batch, "target_text")),
        organ_names=_string_list(_batch_value(batch, "organ")),
    )


def _concept_metrics(output) -> dict[str, float]:
    metrics = output.to_metrics()
    return {f"concept_{key}": value for key, value in metrics.items()}


def _make_loader(config: AblationConfig, records: list[dict[str, Any]], *, split_name: str, shuffle: bool):
    if config.model.image_embedding_mode == "cached":
        return TorchDataLoader(
            records,
            batch_size=config.train.batch_size,
            shuffle=shuffle,
            num_workers=config.train.num_workers,
            collate_fn=_simple_collate,
            pin_memory=torch.cuda.is_available(),
        )

    from merlin.data import DataLoader

    return DataLoader(
        datalist=records,
        cache_dir=str(config.paths.cache_dir / config.train.run_id / split_name),
        batchsize=config.train.batch_size,
        shuffle=shuffle,
        num_workers=config.train.num_workers,
    )


def _simple_collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    string_keys = {"image", "image_embedding", "study_id", "organ", "prompt", "target_text", "full_text"}
    vector_keys = {
        "semantic_family_targets",
        "semantic_family_allowed",
        "semantic_subtype_targets",
        "semantic_subtype_allowed",
    }
    batch: dict[str, Any] = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if key in string_keys:
            batch[key] = [str(value) for value in values]
        elif key in vector_keys:
            batch[key] = torch.as_tensor(values)
        elif all(isinstance(value, bool) for value in values):
            batch[key] = torch.as_tensor(values, dtype=torch.bool)
        elif all(isinstance(value, int) for value in values):
            batch[key] = torch.as_tensor(values, dtype=torch.long)
        elif all(isinstance(value, float) for value in values):
            batch[key] = torch.as_tensor(values, dtype=torch.float32)
        else:
            batch[key] = values
    return batch


def _cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    index = torch.cuda.current_device() if device.index is None else device.index
    return {
        "cuda_memory_allocated_gb": float(torch.cuda.memory_allocated(index) / 1024**3),
        "cuda_memory_reserved_gb": float(torch.cuda.memory_reserved(index) / 1024**3),
        "cuda_memory_peak_allocated_gb": float(torch.cuda.max_memory_allocated(index) / 1024**3),
        "cuda_memory_peak_reserved_gb": float(torch.cuda.max_memory_reserved(index) / 1024**3),
    }


def _prepare_imports(config: AblationConfig) -> None:
    paths = [str(config.paths.merlin_repo)]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def _batch_value(batch: dict[str, Any], key: str) -> Any:
    value = batch[key]
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in list(value)]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _save_checkpoint(
    path: Path,
    *,
    model,
    aux_losses,
    optimizer,
    config: AblationConfig,
    epoch: int,
    step: int,
    val_metrics: dict[str, Any],
) -> None:
    full_payload = {
        "checkpoint_format": "full_v1",
        "model": model.state_dict(),
        "aux_losses": aux_losses.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.raw,
        "epoch": int(epoch),
        "step": int(step),
        "val_metrics": val_metrics,
    }
    torch.save(
        full_payload,
        path,
    )
    if config.train.save_trainable_checkpoint:
        _save_trainable_checkpoint(
            _trainable_checkpoint_path(path),
            model=model,
            aux_losses=aux_losses,
            config=config,
            epoch=epoch,
            step=step,
            val_metrics=val_metrics,
        )


def _save_trainable_checkpoint(
    path: Path,
    *,
    model,
    aux_losses,
    config: AblationConfig,
    epoch: int,
    step: int,
    val_metrics: dict[str, Any],
) -> None:
    model_trainable = _trainable_state_dict(model)
    torch.save(
        {
            "checkpoint_format": "trainable_only_v1",
            "model_trainable": model_trainable,
            "model_trainable_keys": sorted(model_trainable),
            "aux_losses": {key: value.detach().cpu() for key, value in aux_losses.state_dict().items()},
            "config": config.raw,
            "epoch": int(epoch),
            "step": int(step),
            "val_metrics": val_metrics,
        },
        path,
    )


def _trainable_checkpoint_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_trainable{path.suffix}")


def _trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _replace_with_hardlink_or_save(
    *,
    source: Path,
    target: Path,
    model,
    aux_losses,
    optimizer,
    config: AblationConfig,
    epoch: int,
    step: int,
    val_metrics: dict[str, Any],
) -> None:
    try:
        if target.exists() or target.is_symlink():
            target.unlink()
        os.link(source, target)
    except OSError:
        _save_checkpoint(
            target,
            model=model,
            aux_losses=aux_losses,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            step=step,
            val_metrics=val_metrics,
        )


def _load_training_checkpoint(
    path: Path,
    *,
    model,
    aux_losses,
    optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise KeyError(f"Resume checkpoint must contain full model state, got keys={sorted(payload)}")
    model.load_state_dict(payload["model"], strict=True)
    if "aux_losses" in payload:
        aux_losses.load_state_dict(payload["aux_losses"], strict=True)
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)
    return {
        "epoch": int(payload.get("epoch", -1)),
        "step": int(payload.get("step", 0)),
        "val_metrics": payload.get("val_metrics"),
    }


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
