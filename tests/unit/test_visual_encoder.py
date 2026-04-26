from __future__ import annotations

import torch

from organ_seg_clip.config.schemas import (
    AggregatorConfig,
    DataConfig,
    EncoderConfig,
    LoggingConfig,
    LossConfig,
    ModelConfig,
    OrgansConfig,
    PatchingConfig,
    PathsConfig,
    PreprocessingConfig,
    RuntimeConfig,
    SegMambaConfig,
    TextEncoderConfig,
    TokenizerConfig,
    TrainingConfig,
)
from organ_seg_clip.models.factory import build_model
from organ_seg_clip.models.interfaces.types import EncoderBatch
from organ_seg_clip.models.visual_encoder import build_visual_encoder, load_distilled_visual_encoder


def _build_config(*, patching: PatchingConfig | None = None) -> EncoderConfig:
    return EncoderConfig(
        config_path="tests",
        config_dir="/tmp",
        paths=PathsConfig(dataset_root="/tmp", output_dir="/tmp"),
        data=DataConfig(organ_names=("A", "B", "C")),
        preprocessing=PreprocessingConfig(resample_spacing=None, foreground_crop=False),
        text_encoder=TextEncoderConfig(backend_family="hash", projection_dim=16),
        model=ModelConfig(
            segmamba=SegMambaConfig(feat_size=(8, 16, 24, 32), activation_checkpointing=False, segmentation_class_count=4),
            patching=patching or PatchingConfig(patch_size=(32, 32, 32), patch_stride=(32, 32, 32), patch_batch_size=1),
            tokenizer=TokenizerConfig(model_dim=16, summary_grid=(2, 2, 2)),
            aggregator=AggregatorConfig(num_latents=4, num_layers=1, num_heads=4, dropout=0.0),
            organs=OrgansConfig(diagnostic_dropout=0.0),
            organ_query_count=3,
        ),
        loss=LossConfig(organ_attention_weight=0.1, report_clip_weight=0.2),
        training=TrainingConfig(device="cpu", amp=False, batch_size=1),
        runtime=RuntimeConfig(compile_model=False),
        logging=LoggingConfig(),
    )


def _batch() -> EncoderBatch:
    return EncoderBatch(
        study_ids=["study-1"],
        images=torch.rand((1, 1, 32, 32, 32)),
        image_mask=torch.ones((1, 1, 32, 32, 32), dtype=torch.bool),
        segmentations=torch.randint(0, 4, (1, 32, 32, 32), dtype=torch.long),
        segmentation_mask=torch.ones((1, 32, 32, 32), dtype=torch.bool),
        report_texts=["full report text"],
        organ_texts=[["A: organ a", "B: organ b", "C: organ c"]],
        organ_raw_texts=[["organ a", "organ b", "organ c"]],
        organ_text_mask=torch.tensor([[True, True, True]]),
        organ_labels=torch.tensor([[0.0, 1.0, 0.0]]),
        organ_label_mask=torch.tensor([[True, True, True]]),
        lesion_global_labels=torch.tensor([1.0]),
        lesion_global_mask=torch.tensor([True]),
        lesion_organ_labels=torch.tensor([[0.0, 1.0, 0.0]]),
        lesion_organ_mask=torch.tensor([[True, True, True]]),
        metadata=[{}],
    )


def test_visual_encoder_forward_shapes_cpu() -> None:
    config = _build_config()
    encoder = build_visual_encoder(config)
    outputs = encoder(_batch())
    assert outputs.study_ids == ["study-1"]
    assert outputs.organ_names == ("A", "B", "C")
    assert tuple(outputs.report_embedding.shape) == (1, 16)
    assert tuple(outputs.organ_embeddings.shape) == (1, 3, 16)
    assert tuple(outputs.study_latents.shape) == (1, 4, 16)
    assert outputs.visual_tokens.shape[0] == 1
    assert outputs.visual_tokens.shape[-1] == 16
    assert outputs.visual_token_mask.dtype == torch.bool


def test_visual_encoder_state_keys_overlap_full_model() -> None:
    config = _build_config()
    full = build_model(config)
    visual = build_visual_encoder(config)
    full_state = full.state_dict()
    visual_state = visual.state_dict()
    matched = {
        key: value
        for key, value in full_state.items()
        if key in visual_state and getattr(visual_state[key], "shape", None) == getattr(value, "shape", None)
    }
    missing, unexpected = visual.load_state_dict(matched, strict=False)
    assert unexpected == []
    assert "patch_encoder.stem.0.weight" not in missing
    assert "study_aggregator.latents" not in missing
    assert "organ_head.queries" not in missing
    assert "report_head.query" not in missing
    assert len(matched) > 0


def test_training_mode_caps_segmentation_supervision_patches_cpu() -> None:
    config = _build_config(
        patching=PatchingConfig(
            patch_size=(32, 32, 32),
            patch_stride=(32, 32, 32),
            patch_batch_size=2,
            segmentation_supervision_max_patches_per_study=1,
        )
    )
    model = build_model(config)
    batch = EncoderBatch(
        study_ids=["study-1"],
        images=torch.rand((1, 1, 64, 64, 32)),
        image_mask=torch.ones((1, 1, 64, 64, 32), dtype=torch.bool),
        segmentations=torch.randint(0, 4, (1, 64, 64, 32), dtype=torch.long),
        segmentation_mask=torch.ones((1, 64, 64, 32), dtype=torch.bool),
        report_texts=["full report text"],
        organ_texts=[["A: organ a", "B: organ b", "C: organ c"]],
        organ_raw_texts=[["organ a", "organ b", "organ c"]],
        organ_text_mask=torch.tensor([[True, True, True]]),
        organ_labels=torch.tensor([[0.0, 1.0, 0.0]]),
        organ_label_mask=torch.tensor([[True, True, True]]),
        lesion_global_labels=torch.tensor([1.0]),
        lesion_global_mask=torch.tensor([True]),
        lesion_organ_labels=torch.tensor([[0.0, 1.0, 0.0]]),
        lesion_organ_mask=torch.tensor([[True, True, True]]),
        metadata=[{}],
    )

    model.train()
    train_outputs = model(batch)
    assert train_outputs.segmentation_patch_count == 1

    model.eval()
    with torch.no_grad():
        eval_outputs = model(batch)
    assert eval_outputs.segmentation_patch_count == 4



def test_batched_study_encoding_matches_per_study_cpu() -> None:
    config = _build_config(
        patching=PatchingConfig(
            patch_size=(32, 32, 32),
            patch_stride=(32, 32, 32),
            patch_batch_size=2,
        )
    )
    model = build_model(config)
    batch = EncoderBatch(
        study_ids=["study-1", "study-2"],
        images=torch.rand((2, 1, 32, 32, 32)),
        image_mask=torch.ones((2, 1, 32, 32, 32), dtype=torch.bool),
        segmentations=torch.randint(0, 4, (2, 32, 32, 32), dtype=torch.long),
        segmentation_mask=torch.ones((2, 32, 32, 32), dtype=torch.bool),
        report_texts=["full report text 1", "full report text 2"],
        organ_texts=[
            ["A: organ a", "B: organ b", "C: organ c"],
            ["A: organ a2", "B: organ b2", "C: organ c2"],
        ],
        organ_raw_texts=[
            ["organ a", "organ b", "organ c"],
            ["organ a2", "organ b2", "organ c2"],
        ],
        organ_text_mask=torch.tensor([[True, True, True], [True, True, True]]),
        organ_labels=torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        organ_label_mask=torch.tensor([[True, True, True], [True, True, True]]),
        lesion_global_labels=torch.tensor([1.0, 0.0]),
        lesion_global_mask=torch.tensor([True, True]),
        lesion_organ_labels=torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        lesion_organ_mask=torch.tensor([[True, True, True], [True, True, True]]),
        metadata=[{}, {}],
    )

    model.eval()
    with torch.no_grad():
        prepared = [model._prepare_study(batch, index) for index in range(2)]
        batched_outputs = model._encode_studies_batched(prepared)
        single_outputs = [model._encode_studies_batched([study])[0] for study in prepared]

    for batched, single in zip(batched_outputs, single_outputs, strict=True):
        assert torch.allclose(batched[0], single[0], atol=1e-6, rtol=1e-6)
        assert torch.allclose(batched[1], single[1], atol=1e-6, rtol=1e-6)
        assert batched[2] == single[2]
        assert batched[3] == single[3]
        assert torch.allclose(batched[4], single[4], atol=1e-6, rtol=1e-6)
        assert batched[5] == single[5]
        assert batched[6] == single[6]
        assert torch.equal(batched[7], single[7])
        assert torch.equal(batched[8], single[8])


def test_distilled_visual_encoder_loader_roundtrip(tmp_path) -> None:
    config = _build_config()
    visual = build_visual_encoder(config)
    path = tmp_path / "visual_encoder.pt"
    torch.save(
        {
            "format": "organsegclip_visual_encoder_v1",
            "model_state": visual.state_dict(),
            "config": config.to_dict(),
        },
        path,
    )
    loaded, payload = load_distilled_visual_encoder(path, map_location="cpu")
    assert payload["format"] == "organsegclip_visual_encoder_v1"
    outputs = loaded(_batch())
    assert tuple(outputs.organ_embeddings.shape) == (1, 3, 16)
