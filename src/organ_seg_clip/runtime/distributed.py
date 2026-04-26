"""Distributed runtime helpers."""

from __future__ import annotations

import os
from typing import Mapping

import torch
from torch.nn.parallel import DistributedDataParallel


def maybe_init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or not torch.distributed.is_available():
        return False, 0, 0, 1
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def destroy_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    return torch.distributed.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        torch.distributed.barrier()


def wrap_ddp(module: torch.nn.Module, *, find_unused_parameters: bool) -> torch.nn.Module:
    if not is_distributed():
        return module
    device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
    return DistributedDataParallel(module, device_ids=device_ids, find_unused_parameters=find_unused_parameters)


def reduce_mean_metrics(metrics: Mapping[str, float], *, device: torch.device) -> dict[str, float]:
    if not is_distributed():
        return {key: float(value) for key, value in metrics.items()}
    keys = sorted(metrics)
    tensor = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= get_world_size()
    return {key: float(value.item()) for key, value in zip(keys, tensor)}


def reduce_weighted_metrics(
    totals: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    device: torch.device,
) -> dict[str, float]:
    keys = sorted(set(totals) | set(weights))
    if is_distributed():
        gathered_keys: list[list[str] | None] = [None for _ in range(get_world_size())]
        torch.distributed.all_gather_object(gathered_keys, keys)
        keys = sorted({key for chunk in gathered_keys if chunk is not None for key in chunk})
    if not keys:
        return {}
    total_tensor = torch.tensor([float(totals.get(key, 0.0)) for key in keys], device=device, dtype=torch.float64)
    weight_tensor = torch.tensor([float(weights.get(key, 0.0)) for key in keys], device=device, dtype=torch.float64)
    if is_distributed():
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(weight_tensor, op=torch.distributed.ReduceOp.SUM)
    return {
        key: 0.0 if weight <= 0.0 else float(total / weight)
        for key, total, weight in zip(keys, total_tensor.tolist(), weight_tensor.tolist())
    }
