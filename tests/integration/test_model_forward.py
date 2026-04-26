from __future__ import annotations

import pytest
import torch

pytest.importorskip("monai")
pytest.importorskip("mamba_ssm")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for the full OrganSegCLIP integration forward path in this environment.", allow_module_level=True)

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
from organ_seg_clip.models.losses.composer import OrganSegLossComposer


def _build_config() -> EncoderConfig:
    return EncoderConfig(
        config_path="tests",
        config_dir="/tmp",
        paths=PathsConfig(dataset_root="/tmp", output_dir="/tmp"),
        data=DataConfig(organ_names=("A", "B", "C")),
        preprocessing=PreprocessingConfig(resample_spacing=None, foreground_crop=False),
        text_encoder=TextEncoderConfig(backend_family="hash", projection_dim=16),
        model=ModelConfig(
            segmamba=SegMambaConfig(feat_size=(8, 16, 24, 32), activation_checkpointing=False, segmentation_class_count=4),
            patching=PatchingConfig(patch_size=(32, 32, 32), patch_stride=(32, 32, 32), patch_batch_size=1),
            tokenizer=TokenizerConfig(model_dim=16, summary_grid=(2, 2, 2)),
            aggregator=AggregatorConfig(num_latents=4, num_layers=1, num_heads=4, dropout=0.0),
            organs=OrgansConfig(diagnostic_dropout=0.0),
            organ_query_count=3,
        ),
        loss=LossConfig(),
        training=TrainingConfig(device="cpu", amp=False, batch_size=1),
        runtime=RuntimeConfig(compile_model=False),
        logging=LoggingConfig(),
    )


def test_model_forward_and_backward_cpu() -> None:
    config = _build_config()
    model = build_model(config)
    composer = OrganSegLossComposer(config.loss)
    batch = EncoderBatch(
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
    outputs = model(batch)
    assert tuple(outputs.organ_image_embeddings.shape) == (1, 3, 16)
    assert tuple(outputs.organ_text_embeddings.shape) == (1, 3, 16)
    assert tuple(outputs.diagnostic_logits.shape) == (1, 3)
    loss_output, metrics = composer(outputs, batch)
    loss_output.total_loss.backward()
    grads = {name: param.grad for name, param in model.named_parameters() if param.requires_grad}
    assert any(name.startswith("patch_encoder") and grad is not None for name, grad in grads.items())
    assert any(name.startswith("patch_segmentation_head") and grad is not None for name, grad in grads.items())
    assert any(name.startswith("study_aggregator") and grad is not None for name, grad in grads.items())
    assert any(name.startswith("diagnostic_head") and grad is not None for name, grad in grads.items())
    assert metrics["segmentation_dice"] >= 0.0
