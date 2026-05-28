from __future__ import annotations

import csv
from pathlib import Path
import tempfile

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
from organ_seg_clip.decoder.semantic_losses import SemanticDiagnosticLoss
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


def test_duodenum_not_mapped_to_small_bowel_decoder_label(tmp_path):
    # Duodenum was removed from CSV_TO_ORGAN_NAME to avoid noisy supervision; CSV
    # entries for duodenum must not produce a lesion label for Small bowel.
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
    assert dataset[0].lesion_mask is False
    assert dataset[0].lesion_label == 0.0
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


def test_concept_specific_positive_loss_decreases_with_concept_mass(tmp_path):
    tokenizer = TinyTokenizer()
    concept_token = tokenizer("adenoma", add_special_tokens=False)["input_ids"][0]
    cache_path = _concept_cache(
        tmp_path,
        positive_concepts=[{"source_label": "adenoma", "label_type": "subtype", "weight": 1.0, "token_ids": [concept_token]}],
        negative_concepts=[],
    )
    config = load_decoder_config(_minimal_config_path(tmp_path, variant="concept_specific_lexical", lexical_target_cache=cache_path))
    loss_fn = BinaryDiagnosticLoss(config.diagnostic_loss, tokenizer)
    vocab_size = max(tokenizer.vocab.values()) + 5
    labels = torch.tensor([[-100, 3, 4, 1]], dtype=torch.long)
    low_logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
    high_logits = low_logits.clone()
    high_logits[..., concept_token] = 8.0
    low = loss_fn(
        logits=low_logits,
        labels=labels,
        lesion_labels=torch.tensor([1.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["Adenoma."],
        organ_names=["Liver"],
    )
    high = loss_fn(
        logits=high_logits,
        labels=labels,
        lesion_labels=torch.tensor([1.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["Adenoma."],
        organ_names=["Liver"],
    )
    assert high.pathology_positive_loss < low.pathology_positive_loss
    assert high.positive_concept_count == 1


def test_concept_specific_negative_loss_increases_with_spike(tmp_path):
    tokenizer = TinyTokenizer()
    concept_token = tokenizer("mass", add_special_tokens=False)["input_ids"][0]
    cache_path = _concept_cache(
        tmp_path,
        positive_concepts=[],
        negative_concepts=[{"source_label": "mass", "label_type": "subtype", "weight": 1.0, "token_ids": [concept_token]}],
    )
    config = load_decoder_config(_minimal_config_path(tmp_path, variant="concept_specific_lexical", lexical_target_cache=cache_path))
    loss_fn = BinaryDiagnosticLoss(config.diagnostic_loss, tokenizer)
    vocab_size = max(tokenizer.vocab.values()) + 5
    labels = torch.tensor([[-100, 3, 4, 1]], dtype=torch.long)
    low_logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
    high_logits = low_logits.clone()
    high_logits[:, 2, concept_token] = 8.0
    low = loss_fn(
        logits=low_logits,
        labels=labels,
        lesion_labels=torch.tensor([0.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["Adenoma."],
        organ_names=["Liver"],
    )
    high = loss_fn(
        logits=high_logits,
        labels=labels,
        lesion_labels=torch.tensor([0.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["Adenoma."],
        organ_names=["Liver"],
    )
    assert high.pathology_negative_loss > low.pathology_negative_loss
    assert high.negative_concept_count == 1


def test_concept_specific_averages_multiple_positive_concepts(tmp_path):
    tokenizer = TinyTokenizer()
    adenoma = tokenizer("adenoma", add_special_tokens=False)["input_ids"][0]
    thickening = tokenizer("thickening", add_special_tokens=False)["input_ids"][0]
    cache_path = _concept_cache(
        tmp_path,
        positive_concepts=[
            {"source_label": "adenoma", "label_type": "subtype", "weight": 1.0, "token_ids": [adenoma]},
            {"source_label": "thickening", "label_type": "subtype", "weight": 1.0, "token_ids": [thickening]},
        ],
        negative_concepts=[],
    )
    config = load_decoder_config(_minimal_config_path(tmp_path, variant="concept_specific_lexical", lexical_target_cache=cache_path))
    loss_fn = BinaryDiagnosticLoss(config.diagnostic_loss, tokenizer)
    vocab_size = max(tokenizer.vocab.values()) + 5
    logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
    logits[..., adenoma] = 8.0
    output = loss_fn(
        logits=logits,
        labels=torch.tensor([[-100, 3, 4, 1]], dtype=torch.long),
        lesion_labels=torch.tensor([1.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["Adenoma."],
        organ_names=["Liver"],
    )
    assert output.positive_concept_count == 2
    assert output.pathology_positive_loss.item() > 0.0


def test_concept_specific_concept_weight_scales_loss(tmp_path):
    # A concept with weight=2.0 should produce exactly 2× the weighted loss of weight=1.0,
    # because the weighted mean divides by sum(weights) = the single concept weight.
    tokenizer = TinyTokenizer()
    concept_token = tokenizer("adenoma", add_special_tokens=False)["input_ids"][0]
    (tmp_path / "low").mkdir()
    (tmp_path / "high").mkdir()
    (tmp_path / "cfg_low").mkdir()
    (tmp_path / "cfg_high").mkdir()
    cache_low = _concept_cache(
        tmp_path / "low",
        positive_concepts=[{"source_label": "adenoma", "label_type": "subtype", "weight": 1.0, "token_ids": [concept_token]}],
        negative_concepts=[],
    )
    cache_high = _concept_cache(
        tmp_path / "high",
        positive_concepts=[{"source_label": "adenoma", "label_type": "subtype", "weight": 2.0, "token_ids": [concept_token]}],
        negative_concepts=[],
    )
    vocab_size = max(tokenizer.vocab.values()) + 5
    logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
    labels = torch.tensor([[-100, 3, 4, 1]], dtype=torch.long)
    kwargs = dict(
        logits=logits,
        labels=labels,
        lesion_labels=torch.tensor([1.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["Adenoma."],
        organ_names=["Liver"],
    )
    config_low = load_decoder_config(_minimal_config_path(tmp_path / "cfg_low", variant="concept_specific_lexical", lexical_target_cache=cache_low))
    config_high = load_decoder_config(_minimal_config_path(tmp_path / "cfg_high", variant="concept_specific_lexical", lexical_target_cache=cache_high))
    out_low = BinaryDiagnosticLoss(config_low.diagnostic_loss, tokenizer)(**kwargs)
    out_high = BinaryDiagnosticLoss(config_high.diagnostic_loss, tokenizer)(**kwargs)
    # weighted_mean({loss * w}) / sum(w) is invariant to scaling of a single concept weight,
    # so both should produce the same positive loss value.
    assert abs(out_low.pathology_positive_loss.item() - out_high.pathology_positive_loss.item()) < 1e-5


def test_concept_specific_sample_weight_scales_loss(tmp_path):
    tokenizer = TinyTokenizer()
    concept_token = tokenizer("adenoma", add_special_tokens=False)["input_ids"][0]
    (tmp_path / "low").mkdir()
    (tmp_path / "high").mkdir()
    (tmp_path / "cfg_low").mkdir()
    (tmp_path / "cfg_high").mkdir()
    cache_low = _concept_cache(
        tmp_path / "low",
        positive_concepts=[{"source_label": "adenoma", "label_type": "subtype", "weight": 1.0, "token_ids": [concept_token]}],
        negative_concepts=[],
        sample_weight=0.5,
    )
    cache_high = _concept_cache(
        tmp_path / "high",
        positive_concepts=[{"source_label": "adenoma", "label_type": "subtype", "weight": 1.0, "token_ids": [concept_token]}],
        negative_concepts=[],
        sample_weight=1.0,
    )
    vocab_size = max(tokenizer.vocab.values()) + 5
    logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
    labels = torch.tensor([[-100, 3, 4, 1]], dtype=torch.long)
    kwargs = dict(
        logits=logits,
        labels=labels,
        lesion_labels=torch.tensor([1.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["Adenoma."],
        organ_names=["Liver"],
    )
    config_low = load_decoder_config(_minimal_config_path(tmp_path / "cfg_low", variant="concept_specific_lexical", lexical_target_cache=cache_low))
    config_high = load_decoder_config(_minimal_config_path(tmp_path / "cfg_high", variant="concept_specific_lexical", lexical_target_cache=cache_high))
    out_low = BinaryDiagnosticLoss(config_low.diagnostic_loss, tokenizer)(**kwargs)
    out_high = BinaryDiagnosticLoss(config_high.diagnostic_loss, tokenizer)(**kwargs)
    # raw_loss = mean(sample_weight * per_concept_loss); sample_weight=0.5 vs 1.0 → 2× ratio
    assert abs(out_high.raw_loss.item() - 2.0 * out_low.raw_loss.item()) < 1e-5


def test_concept_specific_empty_token_set_skipped_no_nan(tmp_path):
    tokenizer = TinyTokenizer()
    real_token = tokenizer("adenoma", add_special_tokens=False)["input_ids"][0]
    cache_path = _concept_cache(
        tmp_path,
        positive_concepts=[
            {"source_label": "empty", "label_type": "subtype", "weight": 1.0, "token_ids": []},
            {"source_label": "adenoma", "label_type": "subtype", "weight": 1.0, "token_ids": [real_token]},
        ],
        negative_concepts=[
            {"source_label": "empty_neg", "label_type": "subtype", "weight": 1.0, "token_ids": []},
        ],
    )
    config = load_decoder_config(_minimal_config_path(tmp_path, variant="concept_specific_lexical", lexical_target_cache=cache_path))
    loss_fn = BinaryDiagnosticLoss(config.diagnostic_loss, tokenizer)
    vocab_size = max(tokenizer.vocab.values()) + 5
    logits = torch.zeros((1, 4, vocab_size), dtype=torch.float32)
    output = loss_fn(
        logits=logits,
        labels=torch.tensor([[-100, 3, 4, 1]], dtype=torch.long),
        lesion_labels=torch.tensor([1.0]),
        lesion_mask=torch.tensor([True]),
        small_bowel_mask=torch.tensor([False]),
        target_texts=["Adenoma."],
        organ_names=["Liver"],
    )
    assert not output.loss.isnan()
    assert output.positive_concept_count == 1  # empty skipped, only real token counted
    assert output.negative_concept_count == 0  # empty neg skipped


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


def test_semantic_targets_overlay_into_decoder_examples(tmp_path):
    semantic_path = tmp_path / "semantic.jsonl"
    semantic_path.write_text(
        "\n".join(
            [
                __import__("json").dumps(
                    {
                        "organ": "Liver",
                        "raw_text": "Mass in liver.",
                        "normality": "abnormal",
                        "polarity": "positive",
                        "certainty": "definite",
                        "primary_subtype": "liver_mass",
                        "secondary_subtypes": ["liver_steatosis"],
                        "confidence": 0.8,
                        "decision_status": "accepted",
                    }
                ),
                __import__("json").dumps(
                    {
                        "organ": "Liver",
                        "raw_text": "Normal.",
                        "normality": "normal",
                        "polarity": "negative",
                        "certainty": "definite",
                        "primary_subtype": "liver_normal",
                        "secondary_subtypes": [],
                        "confidence": 0.9,
                        "decision_status": "accepted_provisional",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "decoder.yaml"
    config_path.write_text(
        f"""
paths:
  dataset_root: /tmp/dataset
  visual_encoder_checkpoint: /tmp/visual.pt
data:
  organ_names: [Liver]
model:
  llm_model_name_or_path: /tmp/qwen
semantic_loss:
  enabled: true
  target_jsonl_paths: [{semantic_path}]
  accepted_sample_weight: 1.0
  provisional_sample_weight: 0.5
""",
        encoding="utf-8",
    )
    config = load_decoder_config(config_path)
    samples = [
        WholeStudySample(
            study_id="AC_MASS",
            split="train",
            scan_path=tmp_path / "mass_scan.nii.gz",
            segmentation_path=tmp_path / "mass_seg.nii.gz",
            report_text="",
            organ_text_lookup={"Liver": "Mass in liver."},
            organ_label_lookup={"Liver": 1},
        ),
        WholeStudySample(
            study_id="AC_NORMAL",
            split="train",
            scan_path=tmp_path / "normal_scan.nii.gz",
            segmentation_path=tmp_path / "normal_seg.nii.gz",
            report_text="",
            organ_text_lookup={"Liver": "Normal."},
            organ_label_lookup={"Liver": 0},
        ),
    ]
    store = DecoderFeatureStore(
        organ_names=("Liver",),
        visual_dim=4,
        records={
            "AC_MASS": _feature_store("AC_MASS", organ_count=1).records["AC_MASS"],
            "AC_NORMAL": _feature_store("AC_NORMAL", organ_count=1).records["AC_NORMAL"],
        },
        metadata={},
    )
    dataset = PerOrganDecoderDataset(samples, feature_store=store, config=config, split="train")
    assert len(dataset) == 2
    examples = {example.study_id: example for example in dataset}
    assert examples["AC_MASS"].semantic_available is True
    assert examples["AC_MASS"].semantic_weight == 0.8
    assert len(examples["AC_MASS"].semantic_active_subtype_indices) == 2
    assert examples["AC_NORMAL"].semantic_available is True
    assert examples["AC_NORMAL"].semantic_weight == 0.45


def test_semantic_loss_variants_return_nonnegative_losses():
    pooled_hidden = torch.randn(2, 4)
    available = torch.tensor([True, True])
    weights = torch.tensor([1.0, 0.5], dtype=torch.float32)
    statuses = ["accepted", "accepted_provisional"]
    normality_targets = torch.tensor([1, 0], dtype=torch.long)
    polarity_targets = torch.tensor([0, 1], dtype=torch.long)
    primary_targets = torch.tensor([0, 1], dtype=torch.long)
    subtype_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    secondary_targets = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    allowed_mask = torch.tensor([[True, True], [True, True]])
    family_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    allowed_family_mask = torch.tensor([[True, True], [True, True]])
    config = load_decoder_config(_minimal_config_path(semantic_variant="minimal", semantic_enabled=True))
    minimal = SemanticDiagnosticLoss(config.semantic_loss, hidden_size=4, subtype_count=2, family_count=2)
    minimal_output = minimal(
        pooled_hidden=pooled_hidden,
        semantic_available=available,
        semantic_weights=weights,
        semantic_statuses=statuses,
        semantic_normality_targets=normality_targets,
        semantic_polarity_targets=polarity_targets,
        semantic_primary_subtype_targets=primary_targets,
        semantic_subtype_targets=subtype_targets,
        semantic_secondary_subtype_targets=secondary_targets,
        semantic_allowed_subtype_mask=allowed_mask,
        semantic_family_targets=family_targets,
        semantic_allowed_family_mask=allowed_family_mask,
    )
    assert minimal_output.loss.item() >= 0.0
    config_ps = load_decoder_config(_minimal_config_path(semantic_variant="primary_secondary", semantic_enabled=True))
    primary_secondary = SemanticDiagnosticLoss(config_ps.semantic_loss, hidden_size=4, subtype_count=2, family_count=2)
    ps_output = primary_secondary(
        pooled_hidden=pooled_hidden,
        semantic_available=available,
        semantic_weights=weights,
        semantic_statuses=statuses,
        semantic_normality_targets=normality_targets,
        semantic_polarity_targets=polarity_targets,
        semantic_primary_subtype_targets=primary_targets,
        semantic_subtype_targets=subtype_targets,
        semantic_secondary_subtype_targets=secondary_targets,
        semantic_allowed_subtype_mask=allowed_mask,
        semantic_family_targets=family_targets,
        semantic_allowed_family_mask=allowed_family_mask,
    )
    assert ps_output.loss.item() >= 0.0


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


def _concept_cache(
    tmp_path: Path,
    *,
    positive_concepts: list[dict[str, object]],
    negative_concepts: list[dict[str, object]],
    sample_weight: float = 1.0,
) -> Path:
    path = tmp_path / "concept_targets.pt"
    torch.save(
        {
            "tokenizer_name": "tiny",
            "target_format": "concept_specific_lexical_v1",
            "rows": [
                {
                    "key": ("Liver", "adenoma."),
                    "positive_concepts": positive_concepts,
                    "negative_concepts": negative_concepts,
                    "sample_weight": sample_weight,
                    "review_required": False,
                }
            ],
        },
        path,
    )
    return path


def _minimal_config_path(
    tmp_path: Path | None = None,
    *,
    variant: str = "binary",
    lexical_target_cache: Path | None = None,
    semantic_variant: str = "minimal",
    semantic_enabled: bool = False,
):
    path = (tmp_path or Path(tempfile.mkdtemp())) / "decoder.yaml"
    lexical_cache_line = "" if lexical_target_cache is None else f"  lexical_target_cache: {lexical_target_cache}\n"
    path.write_text(
        f"""
paths:
  dataset_root: /tmp/dataset
  visual_encoder_checkpoint: /tmp/visual.pt
data:
  organ_names: [Liver]
model:
  llm_model_name_or_path: /tmp/qwen
diagnostic_loss:
  variant: {variant}
{lexical_cache_line}  negative_temperature: 8.0
  pathology_words: [lesion]
  normal_words: [normal, unremarkable]
semantic_loss:
  enabled: {str(semantic_enabled).lower()}
  variant: {semantic_variant}
""",
        encoding="utf-8",
    )
    return path
