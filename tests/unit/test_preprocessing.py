from __future__ import annotations

import torch

from organ_seg_clip.config.schemas import PreprocessingConfig
from organ_seg_clip.data.preprocessing import _apply_foreground_crop


def test_monai_foreground_crop_uses_threshold_and_margin() -> None:
    image = torch.full((1, 1, 10, 12, 14), -1000.0)
    image[:, :, 3:7, 4:8, 5:9] = -800.0
    segmentation = torch.zeros((1, 1, 10, 12, 14), dtype=torch.long)
    segmentation[:, :, 4:6, 5:7, 6:8] = 2
    config = PreprocessingConfig(
        resample_spacing=None,
        foreground_crop_backend="monai",
        foreground_threshold=-950.0,
        foreground_crop_margin=(1, 2, 3),
    )

    cropped_image, cropped_segmentation, crop_info = _apply_foreground_crop(
        image,
        segmentation,
        config=config,
    )

    assert tuple(cropped_image.shape) == (1, 1, 6, 8, 10)
    assert cropped_segmentation is not None
    assert tuple(cropped_segmentation.shape) == (1, 1, 6, 8, 10)
    assert crop_info["backend"] == "monai"
    assert crop_info["start"] == (2, 2, 2)
    assert crop_info["end"] == (8, 10, 12)
    assert int(cropped_segmentation.sum().item()) == 16


def test_foreground_crop_keeps_empty_foreground_unchanged() -> None:
    image = torch.full((1, 1, 5, 6, 7), -1000.0)
    config = PreprocessingConfig(
        resample_spacing=None,
        foreground_crop_backend="monai",
        foreground_threshold=-950.0,
    )

    cropped_image, cropped_segmentation, crop_info = _apply_foreground_crop(
        image,
        None,
        config=config,
    )

    assert cropped_segmentation is None
    assert cropped_image.data_ptr() == image.data_ptr()
    assert tuple(cropped_image.shape) == (1, 1, 5, 6, 7)
    assert crop_info["reason"] == "empty_foreground"
