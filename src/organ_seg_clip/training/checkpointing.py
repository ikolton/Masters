"""Checkpoint save/load helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_WRAPPER_PREFIXES: tuple[str, ...] = ("module", "_orig_mod")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    while True:
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "_orig_mod"):
            current = current._orig_mod
            continue
        return current


def resolve_checkpoint_state_dict(
    payload: Mapping[str, Any],
    *,
    preferred_state_key: str | None = None,
    submodule_prefix: str | None = None,
) -> dict[str, Any]:
    candidate_keys: list[str] = []
    if preferred_state_key is not None:
        candidate_keys.append(str(preferred_state_key))
    candidate_keys.extend(["model_state", "visual_encoder_state", "state_dict"])
    state_dict: Mapping[str, Any] | None = None
    for key in candidate_keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            state_dict = value
            break
    if state_dict is None:
        raise KeyError(f"Could not find a compatible state dict in checkpoint payload. Tried keys: {candidate_keys}")
    normalized = dict(state_dict)
    while True:
        updated = normalized
        for prefix in _WRAPPER_PREFIXES:
            stripped = _strip_prefix_if_present(updated, prefix)
            if stripped != updated:
                updated = stripped
                break
        if updated == normalized:
            break
        normalized = updated
    if submodule_prefix is not None:
        normalized = _strip_prefix_if_present(normalized, submodule_prefix)
    return normalized


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    step: int | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "step": None if step is None else int(step),
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    if extra_state:
        payload.update(dict(extra_state))
    tmp_target = target.with_name(f"{target.name}.tmp")
    torch.save(payload, tmp_target)
    tmp_target.replace(target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    state_dict = resolve_checkpoint_state_dict(payload)
    unwrap_model(model).load_state_dict(state_dict, strict=strict)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    return payload


def load_pretrained_submodule(
    path: str | Path,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
    candidate_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    state_dict = resolve_checkpoint_state_dict(payload)
    module_state = unwrap_model(model).state_dict()
    candidates = [state_dict]
    for prefix in candidate_prefixes:
        stripped = _strip_prefix_if_present(state_dict, prefix)
        if stripped is not state_dict:
            candidates.append(stripped)
    best_state = None
    best_score = -1
    for candidate in candidates:
        score = sum(
            1 for key, value in candidate.items()
            if key in module_state and getattr(module_state[key], 'shape', None) == getattr(value, 'shape', None)
        )
        if score > best_score:
            best_score = score
            best_state = candidate
    if best_state is None or best_score <= 0:
        raise RuntimeError("Could not find overlapping weights for the requested pretrained submodule load.")
    missing, unexpected = unwrap_model(model).load_state_dict(best_state, strict=False)
    return {
        "payload": payload,
        "matched_keys": int(best_score),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _strip_prefix_if_present(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    normalized_prefix = prefix if prefix.endswith('.') else f"{prefix}."
    if not any(str(key).startswith(normalized_prefix) for key in state_dict):
        return state_dict
    return {
        str(key)[len(normalized_prefix):]: value
        for key, value in state_dict.items()
        if str(key).startswith(normalized_prefix)
    }
