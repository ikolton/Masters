"""Qwen-style per-organ report decoder with visual soft prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..config.schemas import DecoderConfig
from .data import DecoderBatch
from .losses import BinaryDiagnosticLoss


@dataclass(frozen=True)
class DecoderForwardOutput:
    total_loss: torch.Tensor
    ce_loss: torch.Tensor
    diagnostic_loss: torch.Tensor
    logits: torch.Tensor
    metrics: dict[str, float]


class PerOrganReportDecoder(nn.Module):
    def __init__(self, config: DecoderConfig, *, tokenizer: Any, llm: nn.Module, visual_dim: int) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.llm = llm
        hidden_size = _resolve_hidden_size(llm)
        self.visual_projector = nn.Sequential(
            nn.LayerNorm(int(visual_dim)),
            nn.Linear(int(visual_dim), int(hidden_size)),
        )
        self.diagnostic_loss = BinaryDiagnosticLoss(config.diagnostic_loss, tokenizer)

    @classmethod
    def from_config(cls, config: DecoderConfig, *, visual_dim: int) -> "PerOrganReportDecoder":
        tokenizer, llm = load_llm_and_tokenizer(config)
        return cls(config, tokenizer=tokenizer, llm=llm, visual_dim=visual_dim)

    def forward(self, batch: DecoderBatch) -> DecoderForwardOutput:
        visual_features = batch.visual_features.to(next(self.parameters()).device)
        input_ids = batch.input_ids.to(visual_features.device)
        attention_mask = batch.attention_mask.to(visual_features.device)
        labels = batch.labels.to(visual_features.device)
        prefix_embeds = self.visual_projector(visual_features)
        token_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        prefix_attention = torch.ones(prefix_embeds.shape[:2], device=visual_features.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        prefix_labels = torch.full(prefix_embeds.shape[:2], -100, device=visual_features.device, dtype=labels.dtype)
        full_labels = torch.cat([prefix_labels, labels], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, labels=full_labels)
        ce_loss = outputs.loss
        diagnostic = self.diagnostic_loss(
            logits=outputs.logits,
            labels=full_labels,
            lesion_labels=batch.lesion_labels.to(visual_features.device),
            lesion_mask=batch.lesion_mask.to(visual_features.device),
            small_bowel_mask=batch.small_bowel_mask.to(visual_features.device),
            target_texts=batch.target_texts,
        )
        total_loss = ce_loss + diagnostic.loss
        metrics = {
            "total_loss": float(total_loss.detach().cpu().item()),
            "ce_loss": float(ce_loss.detach().cpu().item()),
        } | diagnostic.to_metrics()
        return DecoderForwardOutput(
            total_loss=total_loss,
            ce_loss=ce_loss,
            diagnostic_loss=diagnostic.loss,
            logits=outputs.logits,
            metrics=metrics,
        )

    @torch.no_grad()
    def generate(
        self,
        batch: DecoderBatch,
        *,
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        repetition_penalty: float,
    ) -> list[str]:
        device = next(self.parameters()).device
        visual_features = batch.visual_features.to(device)
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        prefix_embeds = self.visual_projector(visual_features)
        token_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        prefix_attention = torch.ones(prefix_embeds.shape[:2], device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        return [self.tokenizer.decode(row, skip_special_tokens=True).split("###")[0].strip() for row in generated]


def load_llm_and_tokenizer(config: DecoderConfig) -> tuple[Any, nn.Module]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Decoder training requires transformers.") from exc

    model_name = config.model.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _resolve_torch_dtype(config.model.torch_dtype)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    llm = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if config.model.gradient_checkpointing and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()
        if hasattr(llm.config, "use_cache"):
            llm.config.use_cache = False
    if config.model.freeze_llm:
        for parameter in llm.parameters():
            parameter.requires_grad = False
    if config.lora.enabled:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("LoRA decoder training requires peft. Install peft or set lora.enabled=false.") from exc
        lora_config = LoraConfig(
            r=int(config.lora.r),
            lora_alpha=int(config.lora.alpha),
            lora_dropout=float(config.lora.dropout),
            target_modules=list(config.lora.target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        llm = get_peft_model(llm, lora_config)
    return tokenizer, llm


def _resolve_hidden_size(llm: nn.Module) -> int:
    config = getattr(llm, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    embedding = llm.get_input_embeddings()
    return int(embedding.embedding_dim)


def _resolve_torch_dtype(value: str) -> torch.dtype | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return None
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported model.torch_dtype: {value!r}")
