"""Thin training wrapper around Merlin report-generation components."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import AblationConfig


@dataclass(frozen=True)
class MerlinForwardOutput:
    ce_loss: torch.Tensor
    pooled_hidden: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor


class MerlinReportTrainingWrapper(nn.Module):
    """Calls original Merlin submodules while exposing hidden states for losses."""

    def __init__(self, config: AblationConfig) -> None:
        super().__init__()
        self.config = config
        self.merlin = _load_merlin_model(config)
        self.core = self.merlin.model
        self.max_length = int(config.model.max_length)
        self.tokenizer = self.core.decode_text.tokenizer
        self.text_decoder = self.core.decode_text.text_decoder
        self.hidden_size = int(self.text_decoder.config.hidden_size)
        self._configure_trainable_parameters()

    def forward(
        self,
        *,
        prompts: list[str],
        full_texts: list[str],
        images: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
    ) -> MerlinForwardOutput:
        if image_embeds is None:
            if image_features is not None:
                adapter_dtype = next(self.core.adapter.parameters()).dtype
                image_embeds = self.core.adapter(
                    image_features.to(self.text_decoder.device, dtype=adapter_dtype, non_blocking=True)
                )
            elif images is not None:
                image_embeds = self._encode_images(images)
            else:
                raise ValueError("Either images, image_features, or image_embeds must be provided.")
        else:
            image_embeds = image_embeds.to(self.text_decoder.device, non_blocking=True)
        training_texts = self._training_texts(full_texts)
        tokenized = self.tokenizer(
            training_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.text_decoder.device)
        input_ids = tokenized.input_ids[:, 1:]
        attention_mask = tokenized.attention_mask[:, 1:]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        self._mask_prompt_labels(labels, prompts)

        input_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        image_len = int(image_embeds.shape[1])
        image_labels = torch.full(
            (image_embeds.shape[0], image_len),
            fill_value=-100,
            dtype=torch.long,
            device=self.text_decoder.device,
        )
        input_embeds = torch.cat((image_embeds, input_embeds), dim=1)
        labels = torch.cat((image_labels, labels), dim=1)
        attention_mask = torch.cat((torch.ones_like(image_labels), attention_mask), dim=1)
        if input_embeds.shape[1] > self.max_length:
            input_embeds = input_embeds[:, : self.max_length, :]
            labels = labels[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]

        outputs = self.text_decoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        report_mask = labels.ne(-100)
        pooled_hidden = _masked_mean(hidden, report_mask)
        return MerlinForwardOutput(
            ce_loss=outputs.loss,
            pooled_hidden=pooled_hidden,
            logits=outputs.logits,
            labels=labels,
        )

    def _training_texts(self, full_texts: list[str]) -> list[str]:
        if not self.config.model.append_eos_to_target:
            return full_texts
        eos_token = self.tokenizer.eos_token
        if not eos_token:
            raise ValueError("model.append_eos_to_target=true but tokenizer has no eos_token.")
        return [text if str(text).endswith(eos_token) else f"{text}{eos_token}" for text in full_texts]

    @torch.no_grad()
    def generate(
        self,
        *,
        prompts: list[str],
        images: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        **generation_kwargs: Any,
    ) -> list[str]:
        if image_embeds is None:
            if image_features is not None:
                adapter_dtype = next(self.core.adapter.parameters()).dtype
                image_embeds = self.core.adapter(
                    image_features.to(self.text_decoder.device, dtype=adapter_dtype, non_blocking=True)
                )
            elif images is not None:
                image_embeds = self._encode_images(images)
            else:
                raise ValueError("Either images, image_features, or image_embeds must be provided.")
        else:
            image_embeds = image_embeds.to(self.text_decoder.device, non_blocking=True)
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.text_decoder.device)
        input_ids = tokenized.input_ids[:, 1:]
        input_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        input_embeds = torch.cat((image_embeds, input_embeds), dim=1)
        generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        output_ids = self.text_decoder.generate(inputs_embeds=input_embeds, **generation_kwargs)
        return [clean_merlin_generation(text) for text in self._decode_generated_ids(output_ids)]

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.text_decoder.device, non_blocking=True)
        if self.config.model.freeze_image_encoder:
            with torch.no_grad():
                image_features = self.core.encode_image(images)
        else:
            image_features = self.core.encode_image(images)
        return self.core.adapter(image_features)

    @torch.no_grad()
    def encode_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return frozen pre-adapter Merlin image features for cache-backed training."""
        return self.core.encode_image(images.to(self.text_decoder.device, non_blocking=True)).detach()

    def _mask_prompt_labels(self, labels: torch.Tensor, prompts: list[str]) -> None:
        prompt_tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(labels.device)
        prompt_lengths = prompt_tokens.attention_mask.sum(dim=1).tolist()
        marker_id = self.tokenizer.convert_tokens_to_ids("###\n")
        for row, prompt_len in enumerate(prompt_lengths):
            crop_len = max(int(prompt_len) - 1, 0)
            labels[row, :crop_len] = -100
            if marker_id is not None and marker_id >= 0:
                marker_positions = (labels[row] == marker_id).nonzero(as_tuple=False)
                if marker_positions.numel() > 0:
                    labels[row, : int(marker_positions[0].item()) + 1] = -100

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.core.encode_image.parameters():
            parameter.requires_grad = not self.config.model.freeze_image_encoder
        for parameter in self.core.adapter.parameters():
            parameter.requires_grad = bool(self.config.model.train_adapter)
        for parameter in self.text_decoder.parameters():
            parameter.requires_grad = False
        if self.config.model.train_decoder_lora:
            for name, parameter in self.text_decoder.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = True

    def _decode_generated_ids(self, output_ids: torch.Tensor) -> list[str]:
        eos_token_id = self.tokenizer.eos_token_id
        decoded = []
        for row in output_ids.detach().cpu().tolist():
            if eos_token_id is not None and eos_token_id in row:
                row = row[: row.index(eos_token_id)]
            decoded.append("" if not row else self.tokenizer.decode(row, skip_special_tokens=True))
        return decoded


def trainable_parameter_summary(module: nn.Module) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in module.parameters())
    return {"trainable_parameters": int(trainable), "total_parameters": int(total)}


def _load_merlin_model(config: AblationConfig) -> nn.Module:
    merlin_repo = str(config.paths.merlin_repo)
    if merlin_repo not in sys.path:
        sys.path.insert(0, merlin_repo)
    from merlin import Merlin

    with _temporary_cwd(config.model.merlin_load_cwd or config.paths.merlin_repo):
        model = Merlin(RadiologyReport=True)
    return model


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denominator


def clean_merlin_generation(text: str) -> str:
    cleaned = " ".join(str(text).replace("\n", " ").split()).strip()
    if "###" in cleaned:
        before, after = cleaned.split("###", 1)
        cleaned = after if before.lower().startswith("generate ") else before
    prefixes = (
        "Generate a radiology report for adrenal glands",
        "Generate a radiology report for colon",
        "Generate a radiology report for gallbladder",
        "Generate a radiology report for kidneys",
        "Generate a radiology report for liver",
        "Generate a radiology report for pancreas",
        "Generate a radiology report for prostate",
        "Generate a radiology report for small bowel",
        "Generate a radiology report for spleen",
        "Generate a radiology report for stomach",
        "Generate a radiology report for urinary bladder",
    )
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip(" :#\n\t")
            break
    return cleaned.strip()
