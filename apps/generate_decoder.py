#!/usr/bin/env python3
"""Generate per-organ findings with a trained decoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    device = _resolve_device(config.training.device)
    store, _ = load_or_build_feature_store(config, split=args.split, device=device)
    model = PerOrganReportDecoder.from_config(config, visual_dim=store.visual_dim).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device, strict=False)
    model.eval()
    sample_seed = config.training.seed if args.split == config.data.train_split else config.training.seed + 1
    samples, _ = load_decoder_samples(config, split=args.split, sample_seed=sample_seed)
    dataset = PerOrganDecoderDataset(
        samples,
        feature_store=store,
        config=config,
        split=args.split,
        repeat_positives=False,
    )
    if args.limit is not None:
        dataset.examples = dataset.examples[: int(args.limit)]
    collate = decoder_collate_fn(
        tokenizer=model.tokenizer,
        prompt_template=config.model.prompt_template,
        visual_prefix_mode=config.model.visual_prefix_mode,
        max_length=config.model.max_length,
        include_target=False,
    )
    loader = DataLoader(dataset, batch_size=int(config.training.batch_size), shuffle=False, collate_fn=collate)
    max_new_tokens = config.generation.max_new_tokens or config.model.max_new_tokens
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            generations = model.generate(
                batch,
                max_new_tokens=max_new_tokens,
                do_sample=config.generation.do_sample,
                num_beams=config.generation.num_beams,
                repetition_penalty=config.generation.repetition_penalty,
            )
            for index, text in enumerate(generations):
                rows.append(
                    {
                        "study_id": batch.study_ids[index],
                        "organ": batch.organ_names[index],
                        "target": batch.target_texts[index],
                        "generated": text,
                        "lesion_label": None if not bool(batch.lesion_mask[index].item()) else float(batch.lesion_labels[index].item()),
                    }
                )
    result = {"checkpoint": str(Path(args.checkpoint).expanduser().resolve()), "split": args.split, "generations": rows}
    if args.output:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def _move_batch(batch, device: torch.device):
    return type(batch)(
        study_ids=batch.study_ids,
        organ_names=batch.organ_names,
        organ_indices=batch.organ_indices.to(device),
        visual_features=batch.visual_features.to(device),
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
        lesion_labels=batch.lesion_labels.to(device),
        lesion_mask=batch.lesion_mask.to(device),
        small_bowel_mask=batch.small_bowel_mask.to(device),
        target_texts=batch.target_texts,
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_name!r}, but CUDA is not available.")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
