from __future__ import annotations

import csv

import torch

from organ_seg_clip.config.loader import load_decoder_config
from organ_seg_clip.data.contracts import WholeStudySample
from organ_seg_clip.decoder.data import (
    DecoderFeatureRecord,
    DecoderFeatureStore,
    PerOrganDecoderDataset,
    collate_decoder_batch,
)
from organ_seg_clip.decoder.losses import BinaryDiagnosticLoss
from organ_seg_clip.training.decoder_engine import _build_eval_dataloader


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token = "<pad>"
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<eos>": 1}

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        ids = []
        for token in text.replace("\n", " ").split():
            ids.append(self.vocab.setdefault(token.lower(), len(self.vocab)))
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse.get(int(value), "?") for value in ids)


def test_decoder_config_paths_are_configurable(tmp_path):
    config_path = tmp_path / "decoder.yaml"
    config_path.write_text(
        """
paths:
  dataset_root: /tmp/dataset_a
  visual_encoder_checkpoint: /tmp/visual_a.pt
  output_dir: /tmp/out_a
model:
  llm_model_name_or_path: /tmp/qwen_a
""",
        encoding="utf-8",
    )
    config = load_decoder_config(config_path)
    assert str(config.resolved_dataset_root) == "/tmp/dataset_a"
    assert str(config.resolved_visual_encoder_checkpoint) == "/tmp/visual_a.pt"
    assert config.model.llm_model_name_or_path == "/tmp/qwen_a"


def test_duodenum_maps_to_small_bowel_decoder_label(tmp_path):
    csv_path = tmp_path / "lesions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Encrypted Accession Number", "number of duodenum lesion instances"])
        writer.writeheader()
        writer.writerow({"Encrypted Accession Number": "AC1", "number of duodenum lesion instances": "1"})
    config_path = tmp_path / "decoder.yaml"
    config_path.write_text(
        f"""
paths:
  dataset_root: /tmp/dataset
  visual_encoder_checkpoint: /tmp/visual.pt
data:
  lesion_metadata_csv: {csv_path}
  organ_names: [Small bowel]
model:
  llm_model_name_or_path: /tmp/qwen
""",
        encoding="utf-8",
    )
    config = load_decoder_config(config_path)
    sample = WholeStudySample(
        study_id="AC1",
        split="train",
        scan_path=tmp_path / "scan.nii.gz",
        segmentation_path=tmp_path / "seg.nii.gz",
        report_text="",
        organ_text_lookup={"Small bowel": "Duodenal lesion."},
        organ_label_lookup={"Small bowel": 1},
    )
    store = _feature_store("AC1", organ_count=1)
    dataset = PerOrganDecoderDataset([sample], feature_store=store, config=config, split="train")
    assert len(dataset) == 1
    assert dataset[0].organ_name == "Small bowel"
    assert dataset[0].lesion_mask is True
    assert dataset[0].lesion_label == 1.0
    assert dataset[0].is_small_bowel is True


def test_collate_masks_prompt_and_visual_prefix():
    tokenizer = TinyTokenizer()
    example = _example(target="Small cyst.", lesion_label=1.0)
    batch = collate_decoder_batch(
        [example],
        tokenizer=tokenizer,
        prompt_template="Generate {organ}###\n",
        visual_prefix_mode="report_plus_organ",
        max_length=32,
    )
    prompt_len = len(tokenizer("Generate Liver###\n", add_special_tokens=False)["input_ids"])
    assert batch.visual_features.shape[1] == 2
    assert batch.labels[0, :prompt_len].eq(-100).all()
    assert batch.labels[0, prompt_len:].ne(-100).any()


def test_diagnostic_loss_skips_negative_word_present_in_target():
    tokenizer = TinyTokenizer()
    config = load_decoder_config(_minimal_config_path())
    loss_fn = BinaryDiagnosticLoss(config.diagnostic_loss, tokenizer)
    lesion_token = tokenizer("lesion", add_special_tokens=False)["input_ids"][0]
    vocab_size = max(tokenizer.vocab.values()) + 5
    logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
    logits[..., lesion_token] = 8.0
    labels = torch.tensor([[-100, 3, 4, 1]], dtype=torch.long)
    output = loss_fn(
        logits=logits,
        labels=labels,
        lesion_labels=torch.tensor([0.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["No lesion."],
    )
    assert torch.isclose(output.pathology_negative_loss, torch.tensor(0.0))


def test_positive_lesion_penalizes_generic_normal_words():
    tokenizer = TinyTokenizer()
    config = load_decoder_config(_minimal_config_path())
    loss_fn = BinaryDiagnosticLoss(config.diagnostic_loss, tokenizer)
    normal_token = tokenizer("normal", add_special_tokens=False)["input_ids"][0]
    vocab_size = max(tokenizer.vocab.values()) + 5
    logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
    logits[..., normal_token] = 8.0
    labels = torch.tensor([[-100, 3, 4, 1]], dtype=torch.long)
    output = loss_fn(
        logits=logits,
        labels=labels,
        lesion_labels=torch.tensor([1.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["Mass."],
    )
    assert output.normal_negative_loss.item() > 0.0
    assert output.loss.item() >= 0.0
    assert output.sample_count == 1


def test_negative_lesion_penalty_is_nonnegative():
    tokenizer = TinyTokenizer()
    config = load_decoder_config(_minimal_config_path())
    loss_fn = BinaryDiagnosticLoss(config.diagnostic_loss, tokenizer)
    lesion_token = tokenizer("lesion", add_special_tokens=False)["input_ids"][0]
    vocab_size = max(tokenizer.vocab.values()) + 5
    logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
    logits[..., lesion_token] = 8.0
    labels = torch.tensor([[-100, 3, 4, 1]], dtype=torch.long)
    output = loss_fn(
        logits=logits,
        labels=labels,
        lesion_labels=torch.tensor([0.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["No focal mass."],
    )
    assert output.pathology_negative_loss.item() >= 0.0
    assert output.loss.item() >= 0.0


def test_eval_dataloader_uses_seeded_val_subset(tmp_path):
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset_root / "dataset_split" / split).mkdir(parents=True)
    records = []
    for index in range(6):
        study_id = f"AC{index}"
        records.append(
            {
                "study_id": study_id,
                "cleaned_report": "",
                "findings": {"Liver": f"Finding {index}"},
                "labels": {"Liver": 0},
            }
        )
        case_dir = dataset_root / "dataset_split" / "val" / study_id
        case_dir.mkdir()
        (case_dir / f"{study_id}_resampled.nii.gz").write_text("", encoding="utf-8")
        (case_dir / f"{study_id}_seg_resampled.nii.gz").write_text("", encoding="utf-8")
    for split in ("train", "val"):
        (dataset_root / "dataset_split" / split / "combined.json").write_text(__import__("json").dumps(records), encoding="utf-8")
    config_path = tmp_path / "decoder.yaml"
    config_path.write_text(
        f"""
paths:
  dataset_root: {dataset_root}
  visual_encoder_checkpoint: /tmp/visual.pt
data:
  organ_names: [Liver]
  val_limit: 2
model:
  llm_model_name_or_path: /tmp/qwen
training:
  seed: 5
""",
        encoding="utf-8",
    )
    config = load_decoder_config(config_path)
    seeded_samples = __import__("organ_seg_clip.decoder.data", fromlist=["load_decoder_samples"]).load_decoder_samples(
        config,
        split="val",
        sample_seed=config.training.seed + 1,
    )[0]
    store = DecoderFeatureStore(
        organ_names=("Liver",),
        visual_dim=4,
        records={sample.study_id: _feature_store(sample.study_id, organ_count=1).records[sample.study_id] for sample in seeded_samples},
        metadata={},
    )
    tokenizer = TinyTokenizer()
    dataset, _, _ = _build_eval_dataloader(config, store=store, tokenizer=tokenizer, split="val")
    assert len(dataset) == 2


def test_positive_repetition_is_opt_in(tmp_path):
    csv_path = tmp_path / "lesions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Encrypted Accession Number", "number of liver lesion instances"])
        writer.writeheader()
        writer.writerow({"Encrypted Accession Number": "AC3", "number of liver lesion instances": "1"})
    config_path = tmp_path / "decoder.yaml"
    config_path.write_text(
        f"""
paths:
  dataset_root: /tmp/dataset
  visual_encoder_checkpoint: /tmp/visual.pt
data:
  lesion_metadata_csv: {csv_path}
  organ_names: [Liver]
  lesion_positive_repeat_factor: 4
model:
  llm_model_name_or_path: /tmp/qwen
""",
        encoding="utf-8",
    )
    config = load_decoder_config(config_path)
    sample = WholeStudySample(
        study_id="AC3",
        split="train",
        scan_path=tmp_path / "scan.nii.gz",
        segmentation_path=tmp_path / "seg.nii.gz",
        report_text="",
        organ_text_lookup={"Liver": "Hepatic lesion."},
        organ_label_lookup={"Liver": 1},
    )
    store = _feature_store("AC3")
    plain = PerOrganDecoderDataset([sample], feature_store=store, config=config, split="train", repeat_positives=False)
    repeated = PerOrganDecoderDataset([sample], feature_store=store, config=config, split="train", repeat_positives=True)
    assert len(plain) == 1
    assert len(repeated) == 4


def test_train_abnormal_only_filters_decoder_examples(tmp_path):
    config_path = tmp_path / "decoder.yaml"
    config_path.write_text(
        """
paths:
  dataset_root: /tmp/dataset
  visual_encoder_checkpoint: /tmp/visual.pt
data:
  organ_names: [Liver]
  train_abnormal_only: true
model:
  llm_model_name_or_path: /tmp/qwen
""",
        encoding="utf-8",
    )
    config = load_decoder_config(config_path)
    positive = WholeStudySample(
        study_id="AC_POS",
        split="train",
        scan_path=tmp_path / "pos_scan.nii.gz",
        segmentation_path=tmp_path / "pos_seg.nii.gz",
        report_text="",
        organ_text_lookup={"Liver": "Hepatic lesion."},
        organ_label_lookup={"Liver": 1},
    )
    negative = WholeStudySample(
        study_id="AC_NEG",
        split="train",
        scan_path=tmp_path / "neg_scan.nii.gz",
        segmentation_path=tmp_path / "neg_seg.nii.gz",
        report_text="",
        organ_text_lookup={"Liver": "Unremarkable."},
        organ_label_lookup={"Liver": 0},
    )
    store = DecoderFeatureStore(
        organ_names=("Liver",),
        visual_dim=4,
        records={
            "AC_POS": _feature_store("AC_POS", organ_count=1).records["AC_POS"],
            "AC_NEG": _feature_store("AC_NEG", organ_count=1).records["AC_NEG"],
        },
        metadata={},
    )
    train_dataset = PerOrganDecoderDataset([positive, negative], feature_store=store, config=config, split="train")
    val_dataset = PerOrganDecoderDataset([positive, negative], feature_store=store, config=config, split="val")
    assert len(train_dataset) == 1
    assert train_dataset[0].study_id == "AC_POS"
    assert len(val_dataset) == 2


def _feature_store(study_id: str, organ_count: int = 11) -> DecoderFeatureStore:
    record = DecoderFeatureRecord(
        study_id=study_id,
        report_embedding=torch.zeros(4),
        organ_embeddings=torch.zeros(organ_count, 4),
        study_latents=torch.zeros(2, 4),
        visual_tokens=torch.zeros(3, 4),
        visual_token_mask=torch.ones(3, dtype=torch.bool),
    )
    return DecoderFeatureStore(
        organ_names=tuple(["Small bowel"] if organ_count == 1 else ["Liver"] * organ_count),
        visual_dim=4,
        records={study_id: record},
        metadata={},
    )


def _example(target: str, lesion_label: float):
    sample = WholeStudySample(
        study_id="AC2",
        split="train",
        scan_path=None,
        segmentation_path=None,
        report_text="",
        organ_text_lookup={"Liver": target},
        organ_label_lookup={"Liver": int(lesion_label > 0)},
    )
    config = load_decoder_config(_minimal_config_path())
    store = _feature_store("AC2")
    return PerOrganDecoderDataset([sample], feature_store=store, config=config, split="train")[0]


def _minimal_config_path():
    from pathlib import Path
    import tempfile

    path = Path(tempfile.mkdtemp()) / "decoder.yaml"
    path.write_text(
        """
paths:
  dataset_root: /tmp/dataset
  visual_encoder_checkpoint: /tmp/visual.pt
data:
  organ_names: [Liver]
model:
  llm_model_name_or_path: /tmp/qwen
diagnostic_loss:
  pathology_words: [lesion]
  normal_words: [normal, unremarkable]
""",
        encoding="utf-8",
    )
    return path
