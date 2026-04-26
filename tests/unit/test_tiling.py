from __future__ import annotations

import torch

from organ_seg_clip.models.aggregation.tiling import generate_tile_boxes, normalized_box_features


def test_generate_tile_boxes_is_deterministic() -> None:
    boxes = generate_tile_boxes((10, 12, 14), (6, 6, 6), (4, 4, 4))
    assert boxes[0] == (0, 6, 0, 6, 0, 6)
    assert boxes[1] == (0, 6, 0, 6, 4, 10)
    assert boxes[-1] == (4, 10, 6, 12, 8, 14)


def test_normalized_box_features_shape_and_range() -> None:
    features = normalized_box_features((2, 8, 3, 9, 4, 10), (10, 12, 14), device=torch.device("cpu"))
    assert tuple(features.shape) == (6,)
    assert torch.all(features >= 0.0)
    assert torch.all(features <= 1.0)
