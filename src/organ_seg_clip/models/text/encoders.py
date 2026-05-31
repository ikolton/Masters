"""Configurable text encoders for organ alignment."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config.schemas import TextEncoderConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class _BaseTextEncoder(nn.Module):
    def encode_texts(self, texts: list[str], text_mask: torch.Tensor | None = None, *, max_tokens: int | None = None) -> torch.Tensor:
        if text_mask is None:
            text_mask = torch.tensor([bool(text) for text in texts], device=self._device(), dtype=torch.bool)
        selected: list[str] = []
        indices: list[int] = []
        for index, text in enumerate(texts):
            if bool(text_mask[index].item()) and text:
                selected.append(text)
                indices.append(index)
        output = self._output_text_zeros(len(texts))
        if not selected:
            return output
        encoded = self.forward(selected, max_tokens=max_tokens)
        for row_index, original_index in enumerate(indices):
            output[original_index] = encoded[row_index]
        return output

    def _output_text_zeros(self, batch_size: int) -> torch.Tensor:
        return self._output_zeros(batch_size, 1).squeeze(1)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_nested_texts(self, texts: list[list[str]], text_mask: torch.Tensor) -> torch.Tensor:
        batch_size, organ_count = text_mask.shape
        flattened: list[str] = []
        indices: list[tuple[int, int]] = []
        for batch_index, sample_texts in enumerate(texts):
            for organ_index, text in enumerate(sample_texts):
                if bool(text_mask[batch_index, organ_index].item()) and text:
                    flattened.append(text)
                    indices.append((batch_index, organ_index))
        output = self._output_zeros(batch_size, organ_count)
        if not flattened:
            return output
        encoded = self.forward(flattened)
        for row_index, (batch_index, organ_index) in enumerate(indices):
            output[batch_index, organ_index] = encoded[row_index]
        return output

    def _output_zeros(self, batch_size: int, organ_count: int) -> torch.Tensor:
        raise NotImplementedError


class HFTextEncoder(_BaseTextEncoder):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        if AutoModel is None or AutoTokenizer is None:
            raise ModuleNotFoundError("transformers is required for the text encoder subsystem.") from _TRANSFORMERS_IMPORT_ERROR
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.cls_token
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.proj = nn.Linear(hidden_size, config.projection_dim)
        self._frozen_output_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._configure_encoder_trainability()
        if self.config.disk_cache_path:
            self._load_disk_cache(Path(self.config.disk_cache_path))

    def train(self, mode: bool = True) -> "HFTextEncoder":
        super().train(mode)
        if not self._encoder_has_trainable_params:
            self.encoder.eval()
        return self

    def forward(self, texts: list[str], *, max_tokens: int | None = None) -> torch.Tensor:
        if not texts:
            return self.proj.weight.new_zeros((0, self.config.projection_dim))
        max_length = self.config.max_tokens if max_tokens is None else int(max_tokens)
        if self.config.cache_frozen_outputs and not self._encoder_has_trainable_params:
            pooled = self._cached_frozen_pooled_outputs(texts, max_length=max_length)
            projected = self.proj(pooled)
            return F.normalize(projected.float(), dim=-1, eps=1e-6).to(projected.dtype)
        pooled = self._encode_pooled(texts, max_length=max_length)
        projected = self.proj(pooled)
        return F.normalize(projected.float(), dim=-1, eps=1e-6).to(projected.dtype)

    def _encode_pooled(self, texts: list[str], *, max_length: int) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        )
        device = self.proj.weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with contextlib.nullcontext() if self._encoder_has_trainable_params else torch.no_grad():
            outputs = self.encoder(**encoded)
        return self._pool(outputs.last_hidden_state, encoded["attention_mask"])

    def _cached_frozen_pooled_outputs(self, texts: list[str], *, max_length: int) -> torch.Tensor:
        device = self.proj.weight.device
        cache_limit = int(self.config.cache_max_entries)
        cached_rows: dict[int, torch.Tensor] = {}
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        seen_missing: dict[tuple[int, str], int] = {}
        for index, text in enumerate(texts):
            key = (int(max_length), str(text))
            cached = self._frozen_output_cache.get(key)
            if cached is not None:
                cached_rows[index] = cached.to(device=device, non_blocking=True)
                continue
            if key in seen_missing:
                missing_indices.append(index)
                missing_texts.append(str(text))
                continue
            seen_missing[key] = index
            missing_indices.append(index)
            missing_texts.append(str(text))
        if missing_texts:
            pooled_missing = self._encode_pooled(missing_texts, max_length=max_length).detach()
            for row, original_index in enumerate(missing_indices):
                value = pooled_missing[row]
                cached_rows[original_index] = value
                key = (int(max_length), str(texts[original_index]))
                if cache_limit > 0 and len(self._frozen_output_cache) < cache_limit:
                    self._frozen_output_cache[key] = value.detach().cpu()
        return torch.stack([cached_rows[index].to(device=device) for index in range(len(texts))], dim=0)

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "cls":
            return hidden_states[:, 0, :]
        weights = attention_mask.unsqueeze(-1).float()
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def _output_zeros(self, batch_size: int, organ_count: int) -> torch.Tensor:
        return self.proj.weight.new_zeros((batch_size, organ_count, self.config.projection_dim))

    def _load_disk_cache(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data: dict = torch.load(path, map_location="cpu", weights_only=True)
            self._frozen_output_cache.update(data)
        except Exception as exc:
            print(f"[text_encoder] disk cache load failed ({path}): {exc}", flush=True)

    def save_disk_cache(self) -> None:
        if not self.config.disk_cache_path or not self._frozen_output_cache:
            return
        path = Path(self.config.disk_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if path.exists():
            try:
                existing = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                existing = {}
        merged = {**existing, **{k: v.cpu() for k, v in self._frozen_output_cache.items()}}
        torch.save(merged, path)
        print(f"[text_encoder] disk cache saved: {len(merged)} entries → {path}", flush=True)

    def freeze_projection(self) -> None:
        for parameter in self.proj.parameters():
            parameter.requires_grad = False

    def _configure_encoder_trainability(self) -> None:
        self._set_module_requires_grad(self.encoder, False)
        if self.config.unfreeze_last_n_layers > 0:
            encoder_stack = getattr(self.encoder, "encoder", None)
            layers = getattr(encoder_stack, "layer", None)
            if layers is None:
                raise ValueError("text_encoder.unfreeze_last_n_layers requires an encoder.layer stack.")
            for layer in list(layers)[-int(self.config.unfreeze_last_n_layers):]:
                self._set_module_requires_grad(layer, True)
        elif not self.config.freeze_encoder:
            self._set_module_requires_grad(self.encoder, True)
        self._encoder_has_trainable_params = any(parameter.requires_grad for parameter in self.encoder.parameters())
        if not self._encoder_has_trainable_params:
            self.encoder.eval()

    @staticmethod
    def _set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad


class HashTextEncoder(_BaseTextEncoder):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(int(config.hash_vocab_size), int(config.projection_dim))
        self.proj = nn.Linear(int(config.projection_dim), int(config.projection_dim))

    def forward(self, texts: list[str], *, max_tokens: int | None = None) -> torch.Tensor:
        if not texts:
            return self.proj.weight.new_zeros((0, self.config.projection_dim))
        pooled = []
        device = self.proj.weight.device
        vocab_size = int(self.config.hash_vocab_size)
        for text in texts:
            words = str(text).lower().split()
            if not words:
                pooled.append(self.proj.weight.new_zeros((self.config.projection_dim,)))
                continue
            ids = torch.tensor([hash(word) % vocab_size for word in words], device=device, dtype=torch.long)
            pooled.append(self.embedding(ids).mean(dim=0))
        projected = self.proj(torch.stack(pooled, dim=0))
        return F.normalize(projected.float(), dim=-1, eps=1e-6).to(projected.dtype)

    def _output_zeros(self, batch_size: int, organ_count: int) -> torch.Tensor:
        return self.proj.weight.new_zeros((batch_size, organ_count, self.config.projection_dim))


def build_text_encoder(config: TextEncoderConfig) -> nn.Module:
    if config.backend_family == "hash":
        return HashTextEncoder(config)
    return HFTextEncoder(config)
