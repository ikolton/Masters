#!/usr/bin/env python3
"""Generate per-organ findings with a trained decoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from organ_seg_clip.config import load_decoder_config
from organ_seg_clip.decoder.data import PerOrganDecoderDataset, decoder_collate_fn, load_decoder_samples
from organ_seg_clip.decoder.feature_cache import load_or_build_feature_store
from organ_seg_clip.decoder.model import PerOrganReportDecoder
from organ_seg_clip.training.checkpointing import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the decoder YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Decoder checkpoint path.")
    parser.add_argument("--split", default="val", help="Dataset split to generate.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of organ examples to generate.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()
    config = load_decoder_config(args.config)
    result = generate_generations(
        config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        limit=args.limit,
    )
    if args.output:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def generate_generations(
    config,
    *,
    checkpoint_path: str | Path,
    split: str,
    study_id_filter: Iterable[str] | None = None,
    sample_seed: int | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_new_tokens: int | None = None,
    do_sample: bool | None = None,
    num_beams: int | None = None,
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    device = _resolve_device(config.training.device)
    store, _ = load_or_build_feature_store(config, split=split, device=device)
    model = PerOrganReportDecoder.from_config(config, visual_dim=store.visual_dim).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device, strict=False)
    model.eval()
    resolved_sample_seed = sample_seed
    if resolved_sample_seed is None:
        resolved_sample_seed = config.training.seed if split == config.data.train_split else config.training.seed + 1
    samples, _ = load_decoder_samples(config, split=split, sample_seed=resolved_sample_seed)
    if study_id_filter is not None:
        wanted = {str(value) for value in study_id_filter}
        samples = [sample for sample in samples if str(sample.study_id) in wanted]
    dataset = PerOrganDecoderDataset(
        samples,
        feature_store=store,
        config=config,
        split=split,
        repeat_positives=False,
    )
    if limit is not None:
        dataset.examples = dataset.examples[: int(limit)]
    collate = decoder_collate_fn(
        tokenizer=model.tokenizer,
        prompt_template=config.model.prompt_template,
        visual_prefix_mode=config.model.visual_prefix_mode,
        max_length=config.model.max_length,
        include_target=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or config.training.batch_size),
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else config.training.num_workers),
        collate_fn=collate,
    )
    generation_max_new_tokens = max_new_tokens or config.generation.max_new_tokens or config.model.max_new_tokens
    generation_do_sample = bool(config.generation.do_sample if do_sample is None else do_sample)
    generation_num_beams = int(config.generation.num_beams if num_beams is None else num_beams)
    generation_repetition_penalty = float(
        config.generation.repetition_penalty if repetition_penalty is None else repetition_penalty
    )
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            generations = model.generate(
                batch,
                max_new_tokens=generation_max_new_tokens,
                do_sample=generation_do_sample,
                num_beams=generation_num_beams,
                repetition_penalty=generation_repetition_penalty,
            )
            for index, text in enumerate(generations):
                rows.append(
                    {
                        "study_id": batch.study_ids[index],
                        "organ": batch.organ_names[index],
                        "target": batch.target_texts[index],
                        "generated": text,
                        "organ_abnormal_label": None
                        if not bool(batch.organ_abnormal_mask[index].item())
                        else int(batch.organ_abnormal_labels[index].item()),
                        "lesion_label": None if not bool(batch.lesion_mask[index].item()) else float(batch.lesion_labels[index].item()),
                    }
                )
    return {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "split": split,
        "generations": rows,
    }


def _move_batch(batch, device: torch.device):
    return type(batch)(
        study_ids=batch.study_ids,
        organ_names=batch.organ_names,
        organ_indices=batch.organ_indices.to(device),
        visual_features=batch.visual_features.to(device),
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
        organ_abnormal_labels=batch.organ_abnormal_labels.to(device),
        organ_abnormal_mask=batch.organ_abnormal_mask.to(device),
        lesion_labels=batch.lesion_labels.to(device),
        lesion_mask=batch.lesion_mask.to(device),
        small_bowel_mask=batch.small_bowel_mask.to(device),
        target_texts=batch.target_texts,
        semantic_statuses=batch.semantic_statuses,
        semantic_available=batch.semantic_available.to(device),
        semantic_weights=batch.semantic_weights.to(device),
        semantic_normality_targets=batch.semantic_normality_targets.to(device),
        semantic_polarity_targets=batch.semantic_polarity_targets.to(device),
        semantic_primary_subtype_targets=batch.semantic_primary_subtype_targets.to(device),
        semantic_subtype_targets=batch.semantic_subtype_targets.to(device),
        semantic_secondary_subtype_targets=batch.semantic_secondary_subtype_targets.to(device),
        semantic_allowed_subtype_mask=batch.semantic_allowed_subtype_mask.to(device),
        semantic_family_targets=batch.semantic_family_targets.to(device),
        semantic_allowed_family_mask=batch.semantic_allowed_family_mask.to(device),
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
