"""Minimal and configurable whole-volume preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import warnings

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import CropForegroundd

from ..config.schemas import PreprocessingConfig


@dataclass(frozen=True)
class VolumeMetadata:
    path: str
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    orientation: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessedVolume:
    image: torch.Tensor
    segmentation: torch.Tensor | None
    image_metadata: VolumeMetadata
    segmentation_metadata: VolumeMetadata | None
    crop_info: dict[str, object]


def load_and_preprocess_study(
    *,
    scan_path: str | Path,
    segmentation_path: str | Path | None,
    config: PreprocessingConfig,
) -> PreprocessedVolume:
    image_nii = nib.load(str(scan_path))
    segmentation_nii = nib.load(str(segmentation_path)) if segmentation_path is not None else None

    if config.canonicalize_orientation:
        image_nii = nib.as_closest_canonical(image_nii)
        if segmentation_nii is not None:
            segmentation_nii = nib.as_closest_canonical(segmentation_nii)

    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    segmentation = None
    if segmentation_nii is not None:
        segmentation = np.asarray(segmentation_nii.dataobj, dtype=np.int64)
        if segmentation.shape != image.shape:
            raise ValueError(
                f"Segmentation shape {segmentation.shape} does not match image shape {image.shape} for {scan_path}."
            )

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    segmentation_tensor = None if segmentation is None else torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0)

    if config.resample_spacing is not None:
        image_spacing = tuple(float(v) for v in image_nii.header.get_zooms()[:3])
        image_tensor = _resample_to_spacing(image_tensor, image_spacing=image_spacing, target_spacing=config.resample_spacing)
        if segmentation_tensor is not None:
            segmentation_tensor = _resample_to_spacing(
                segmentation_tensor.float(),
                image_spacing=image_spacing,
                target_spacing=config.resample_spacing,
                mode="nearest",
            ).long()

    crop_info: dict[str, object] = {
        "enabled": bool(config.foreground_crop),
        "backend": "none",
        "shape_before_crop": tuple(int(v) for v in image_tensor.shape[-3:]),
        "shape_after_crop": tuple(int(v) for v in image_tensor.shape[-3:]),
    }
    if config.foreground_crop:
        image_tensor, segmentation_tensor, crop_info = _apply_foreground_crop(
            image_tensor,
            segmentation_tensor,
            config=config,
        )

    image_tensor = _normalize_intensity(
        image_tensor,
        clip_min=config.intensity_clip_min,
        clip_max=config.intensity_clip_max,
        mode=config.intensity_mode,
    )

    if config.canonical_size is not None:
        image_tensor = _crop_or_pad_to_size(image_tensor, tuple(int(v) for v in config.canonical_size), mode="trilinear")
        if segmentation_tensor is not None:
            segmentation_tensor = _crop_or_pad_to_size(
                segmentation_tensor.float(),
                tuple(int(v) for v in config.canonical_size),
                mode="nearest",
            ).long()

    image_metadata = _build_metadata(image_nii, scan_path)
    segmentation_metadata = None if segmentation_nii is None else _build_metadata(segmentation_nii, segmentation_path)
    if config.verify_orientation_spacing and segmentation_metadata is not None:
        if image_metadata.orientation != segmentation_metadata.orientation:
            raise ValueError(
                f"Orientation mismatch for {scan_path}: {image_metadata.orientation} vs {segmentation_metadata.orientation}."
            )
        if image_metadata.spacing != segmentation_metadata.spacing and config.resample_spacing is None:
            raise ValueError(
                f"Spacing mismatch for {scan_path}: {image_metadata.spacing} vs {segmentation_metadata.spacing}."
            )

    crop_info = dict(crop_info)
    crop_info["shape_after_preprocessing"] = tuple(int(v) for v in image_tensor.shape[-3:])
    return PreprocessedVolume(
        image=image_tensor.squeeze(0),
        segmentation=None if segmentation_tensor is None else segmentation_tensor.squeeze(0).squeeze(0),
        image_metadata=image_metadata,
        segmentation_metadata=segmentation_metadata,
        crop_info=crop_info,
    )


def _apply_foreground_crop(
    image_tensor: torch.Tensor,
    segmentation_tensor: torch.Tensor | None,
    *,
    config: PreprocessingConfig,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, object]]:
    threshold = (
        float(config.intensity_clip_min)
        if config.foreground_threshold is None
        else float(config.foreground_threshold)
    )
    shape_before = tuple(int(v) for v in image_tensor.shape[-3:])
    crop_info: dict[str, object] = {
        "enabled": True,
        "backend": config.foreground_crop_backend,
        "threshold": threshold,
        "shape_before_crop": shape_before,
    }
    if not image_tensor.gt(threshold).any():
        crop_info.update(
            {
                "backend": "none",
                "reason": "empty_foreground",
                "shape_after_crop": shape_before,
            }
        )
        return image_tensor, segmentation_tensor, crop_info

    if config.foreground_crop_backend == "monai":
        return _monai_foreground_crop(image_tensor, segmentation_tensor, config=config, threshold=threshold, crop_info=crop_info)

    crop_bounds = _foreground_bounds(
        image_tensor,
        threshold=threshold,
        margin=config.foreground_crop_margin,
    )
    if crop_bounds is None:
        crop_info.update({"backend": "none", "reason": "empty_foreground", "shape_after_crop": shape_before})
        return image_tensor, segmentation_tensor, crop_info
    image_tensor = _apply_bounds(image_tensor, crop_bounds)
    if segmentation_tensor is not None:
        segmentation_tensor = _apply_bounds(segmentation_tensor, crop_bounds)
    crop_info.update(_crop_bounds_to_info(crop_bounds))
    crop_info["shape_after_crop"] = tuple(int(v) for v in image_tensor.shape[-3:])
    return image_tensor, segmentation_tensor, crop_info


def _monai_foreground_crop(
    image_tensor: torch.Tensor,
    segmentation_tensor: torch.Tensor | None,
    *,
    config: PreprocessingConfig,
    threshold: float,
    crop_info: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, object]]:
    keys = ["image"]
    payload: dict[str, torch.Tensor] = {"image": image_tensor.squeeze(0)}
    if segmentation_tensor is not None:
        keys.append("segmentation")
        payload["segmentation"] = segmentation_tensor.squeeze(0)
    cropper = CropForegroundd(
        keys=keys,
        source_key="image",
        select_fn=lambda tensor: tensor > threshold,
        margin=config.foreground_crop_margin,
        allow_smaller=True,
        k_divisible=config.foreground_crop_k_divisible,
        mode=tuple("constant" for _ in keys),
    )
    cropped = cropper(payload)
    image_tensor = torch.as_tensor(cropped["image"], dtype=image_tensor.dtype).unsqueeze(0)
    if segmentation_tensor is not None:
        segmentation_tensor = torch.as_tensor(cropped["segmentation"], dtype=segmentation_tensor.dtype).unsqueeze(0)
    start = tuple(int(v) for v in cropped["foreground_start_coord"])
    end = tuple(int(v) for v in cropped["foreground_end_coord"])
    crop_info.update(
        {
            "start": start,
            "end": end,
            "margin": tuple(int(v) for v in config.foreground_crop_margin),
            "k_divisible": tuple(int(v) for v in config.foreground_crop_k_divisible),
            "shape_after_crop": tuple(int(v) for v in image_tensor.shape[-3:]),
        }
    )
    return image_tensor, segmentation_tensor, crop_info


def _build_metadata(nii: nib.spatialimages.SpatialImage, path: str | Path) -> VolumeMetadata:
    axis_codes = nib.aff2axcodes(nii.affine)
    return VolumeMetadata(
        path=str(Path(path).expanduser().resolve()),
        shape=tuple(int(v) for v in nii.shape),
        spacing=tuple(float(v) for v in nii.header.get_zooms()[: len(nii.shape)]),
        orientation=tuple(str(v) for v in axis_codes),
    )


def _normalize_intensity(image: torch.Tensor, *, clip_min: float, clip_max: float, mode: str) -> torch.Tensor:
    clipped = image.clamp(min=float(clip_min), max=float(clip_max))
    if mode == "scale_to_unit":
        denom = max(float(clip_max) - float(clip_min), 1e-6)
        return (clipped - float(clip_min)) / denom
    dims = tuple(range(2, clipped.ndim))
    mean = clipped.mean(dim=dims, keepdim=True)
    std = clipped.std(dim=dims, keepdim=True).clamp(min=1e-6)
    return (clipped - mean) / std


def _resample_to_spacing(
    tensor: torch.Tensor,
    *,
    image_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    mode: str = "trilinear",
) -> torch.Tensor:
    if all(abs(float(src) - float(dst)) <= 1e-3 for src, dst in zip(image_spacing, target_spacing)):
        return tensor
    scale = [float(src) / float(dst) for src, dst in zip(image_spacing, target_spacing)]

    # Upsampling CT to a finer spacing almost always indicates a misconfiguration —
    # it hallucinates detail that isn't in the source data.
    if any(s > 1.0 + 1e-3 for s in scale):
        warnings.warn(
            f"Resampling is upsampling on at least one axis: "
            f"source={image_spacing}, target={target_spacing}, scale={[round(s, 3) for s in scale]}. "
            f"Verify that resample_spacing is coarser than the source data.",
            stacklevel=3,
        )

    target_size = tuple(max(1, int(round(size * factor))) for size, factor in zip(tensor.shape[-3:], scale))
    if tuple(int(v) for v in tensor.shape[-3:]) == target_size:
        return tensor
    align_corners = False if mode != "nearest" else None
    return F.interpolate(tensor, size=target_size, mode=mode, align_corners=align_corners)


def _foreground_bounds(
    tensor: torch.Tensor,
    *,
    threshold: float,
    margin: tuple[int, int, int] = (0, 0, 0),
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    spatial_mask = tensor.gt(threshold).any(dim=0).any(dim=0)
    coords = spatial_mask.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    mins = coords.min(dim=0).values.tolist()
    maxs = coords.max(dim=0).values.tolist()
    shape = tuple(int(v) for v in spatial_mask.shape)
    return tuple(
        (max(0, int(low) - int(pad)), min(size, int(high) + 1 + int(pad)))
        for low, high, pad, size in zip(mins, maxs, margin, shape)
    )


def _crop_bounds_to_info(bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> dict[str, object]:
    return {
        "start": tuple(int(axis_bounds[0]) for axis_bounds in bounds),
        "end": tuple(int(axis_bounds[1]) for axis_bounds in bounds),
    }


def _apply_bounds(
    tensor: torch.Tensor,
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> torch.Tensor:
    (d0, d1), (h0, h1), (w0, w1) = bounds
    return tensor[:, :, d0:d1, h0:h1, w0:w1]


def _crop_or_pad_to_size(
    tensor: torch.Tensor,
    target_size: tuple[int, int, int],
    *,
    mode: str,
) -> torch.Tensor:
    if tensor.shape[-3:] == target_size:
        return tensor
    padding: list[int] = []
    for current, target in zip(reversed(tensor.shape[-3:]), reversed(target_size)):
        diff = max(0, target - current)
        left = diff // 2
        right = diff - left
        padding.extend([left, right])
    if any(padding):
        tensor = F.pad(tensor, tuple(padding), mode="constant", value=0)
    starts = [max((current - target) // 2, 0) for current, target in zip(tensor.shape[-3:], target_size)]
    d0, h0, w0 = starts
    td, th, tw = target_size
    tensor = tensor[:, :, d0 : d0 + td, h0 : h0 + th, w0 : w0 + tw]
    if tensor.shape[-3:] != target_size:
        align_corners = False if mode != "nearest" else None
        tensor = F.interpolate(tensor, size=target_size, mode=mode, align_corners=align_corners)
    return tensor
