"""Tiling helpers for irregular foreground-cropped studies."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


def mask_bounds(mask: torch.Tensor) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    coords = mask.nonzero(as_tuple=False)
    if coords.numel() == 0:
        d, h, w = mask.shape[-3:]
        return (0, d), (0, h), (0, w)
    mins = coords.min(dim=0).values.tolist()
    maxs = coords.max(dim=0).values.tolist()
    return tuple((int(lo), int(hi) + 1) for lo, hi in zip(mins, maxs))


def crop_to_bounds(tensor: torch.Tensor, bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> torch.Tensor:
    (d0, d1), (h0, h1), (w0, w1) = bounds
    if tensor.ndim == 4:
        return tensor[:, d0:d1, h0:h1, w0:w1]
    return tensor[d0:d1, h0:h1, w0:w1]


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    final_start = length - tile
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def generate_tile_boxes(shape: tuple[int, int, int], tile_size: tuple[int, int, int], tile_stride: tuple[int, int, int]) -> list[tuple[int, int, int, int, int, int]]:
    d, h, w = shape
    starts_d = tile_starts(d, tile_size[0], tile_stride[0])
    starts_h = tile_starts(h, tile_size[1], tile_stride[1])
    starts_w = tile_starts(w, tile_size[2], tile_stride[2])
    boxes: list[tuple[int, int, int, int, int, int]] = []
    for d0, h0, w0 in itertools.product(starts_d, starts_h, starts_w):
        d1 = min(d0 + tile_size[0], d)
        h1 = min(h0 + tile_size[1], h)
        w1 = min(w0 + tile_size[2], w)
        boxes.append((d0, d1, h0, h1, w0, w1))
    return boxes


def extract_tile(volume: torch.Tensor, box: tuple[int, int, int, int, int, int], tile_size: tuple[int, int, int]) -> torch.Tensor:
    d0, d1, h0, h1, w0, w1 = box
    if volume.ndim == 4:
        tile = volume[:, d0:d1, h0:h1, w0:w1]
    else:
        tile = volume[d0:d1, h0:h1, w0:w1]
    pad_d = tile_size[0] - tile.shape[-3]
    pad_h = tile_size[1] - tile.shape[-2]
    pad_w = tile_size[2] - tile.shape[-1]
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        tile = F.pad(tile, (0, pad_w, 0, pad_h, 0, pad_d), value=0)
    return tile


def normalized_box_features(box: tuple[int, int, int, int, int, int], shape: tuple[int, int, int], *, device: torch.device) -> torch.Tensor:
    d0, d1, h0, h1, w0, w1 = box
    depth, height, width = [max(int(v), 1) for v in shape]
    center_d = ((d0 + d1) * 0.5) / depth
    center_h = ((h0 + h1) * 0.5) / height
    center_w = ((w0 + w1) * 0.5) / width
    size_d = (d1 - d0) / depth
    size_h = (h1 - h0) / height
    size_w = (w1 - w0) / width
    return torch.tensor([center_d, center_h, center_w, size_d, size_h, size_w], device=device, dtype=torch.float32)
