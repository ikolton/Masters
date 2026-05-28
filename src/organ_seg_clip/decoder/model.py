"""Qwen-style per-organ report decoder with visual soft prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..config.schemas import DecoderConfig
from .data import DecoderBatch
from .losses import BinaryDiagnosticLoss
from .semantic_losses import SemanticDiagnosticLoss
from .semantic_targets import SemanticTargetLookup


@dataclass(frozen=True)
class DecoderForwardOutput:
    total_loss: torch.Tensor
    ce_loss: torch.Tensor
    diagnostic_loss: torch.Tensor
    binary_diagnostic_loss: torch.Tensor
    semantic_diagnostic_loss: torch.Tensor
    logits: torch.Tensor
    metrics: dict[str, float]


class PerOrganReportDecoder(nn.Module):
    def __init__(self, config: DecoderConfig, *, tokenizer: Any, llm: nn.Module, visual_dim: int) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.llm = llm
        hidden_size = _resolve_hidden_size(llm)
        self.visual_projector = _build_visual_projector(int(visual_dim), int(hidden_size), depth=int(config.model.visual_projector_depth))
        self.diagnostic_loss = BinaryDiagnosticLoss(config.diagnostic_loss, tokenizer)
        semantic_lookup = None
        if config.semantic_loss.enabled:
            semantic_lookup = _load_semantic_lookup(config)
        semantic_subtype_count = 0 if semantic_lookup is None else len(semantic_lookup.spec.subtype_vocab)
        semantic_family_count = 0 if semantic_lookup is None else len(semantic_lookup.spec.family_vocab)
        self.semantic_loss = SemanticDiagnosticLoss(
            config.semantic_loss,
            hidden_size=int(hidden_size),
            subtype_count=int(semantic_subtype_count),
            family_count=int(semantic_family_count),
        )

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
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            output_hidden_states=True,
            return_dict=True,
        )
        ce_loss = outputs.loss
        binary_diagnostic = self.diagnostic_loss(
            logits=outputs.logits,
            labels=full_labels,
            lesion_labels=batch.lesion_labels.to(visual_features.device),
            lesion_mask=batch.lesion_mask.to(visual_features.device),
            small_bowel_mask=batch.small_bowel_mask.to(visual_features.device),
            target_texts=batch.target_texts,
            organ_names=batch.organ_names,
        )
        pooled_hidden = _pool_report_hidden_states(
            hidden_states=outputs.hidden_states[-1],
            label_mask=full_labels.ne(-100),
        )
        semantic_diagnostic = self.semantic_loss(
            pooled_hidden=pooled_hidden,
            semantic_available=batch.semantic_available.to(visual_features.device),
            semantic_weights=batch.semantic_weights.to(visual_features.device),
            semantic_statuses=batch.semantic_statuses,
            semantic_normality_targets=batch.semantic_normality_targets.to(visual_features.device),
            semantic_polarity_targets=batch.semantic_polarity_targets.to(visual_features.device),
            semantic_primary_subtype_targets=batch.semantic_primary_subtype_targets.to(visual_features.device),
            semantic_subtype_targets=batch.semantic_subtype_targets.to(visual_features.device),
            semantic_secondary_subtype_targets=batch.semantic_secondary_subtype_targets.to(visual_features.device),
            semantic_allowed_subtype_mask=batch.semantic_allowed_subtype_mask.to(visual_features.device),
            semantic_family_targets=batch.semantic_family_targets.to(visual_features.device),
            semantic_allowed_family_mask=batch.semantic_allowed_family_mask.to(visual_features.device),
        )
        diagnostic_loss = binary_diagnostic.loss + semantic_diagnostic.loss
        total_loss = ce_loss + diagnostic_loss
        metrics = (
            binary_diagnostic.to_metrics()
            | semantic_diagnostic.to_metrics()
            | {
            "total_loss": float(total_loss.detach().cpu().item()),
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "diagnostic_loss": float((binary_diagnostic.raw_loss + semantic_diagnostic.raw_loss).detach().cpu().item()),
            "diagnostic_loss_weighted": float(diagnostic_loss.detach().cpu().item()),
            "binary_diagnostic_loss": float(binary_diagnostic.raw_loss.detach().cpu().item()),
            "semantic_diagnostic_loss": float(semantic_diagnostic.raw_loss.detach().cpu().item()),
            "semantic_diagnostic_loss_weighted_total": float(semantic_diagnostic.loss.detach().cpu().item()),
        }
        )
        return DecoderForwardOutput(
            total_loss=total_loss,
            ce_loss=ce_loss,
            diagnostic_loss=diagnostic_loss,
            binary_diagnostic_loss=binary_diagnostic.loss,
            semantic_diagnostic_loss=semantic_diagnostic.loss,
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
        llm_dtype = _resolve_llm_compute_dtype(self.llm)
        use_amp = bool(device.type == "cuda" and llm_dtype in {torch.float16, torch.bfloat16})
        autocast_dtype = llm_dtype if llm_dtype in {torch.float16, torch.bfloat16} else torch.float16
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=autocast_dtype):
            prefix_embeds = self.visual_projector(visual_features)
            token_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            inputs_embeds = inputs_embeds.to(token_embeds.dtype)
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
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated]


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


def _build_visual_projector(visual_dim: int, hidden_size: int, *, depth: int) -> nn.Module:
    if depth == 1:
        return nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_size))
    layers: list[nn.Module] = [nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden_size)]
    for _ in range(depth - 1):
        layers += [nn.GELU(), nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size)]
    return nn.Sequential(*layers)


def _resolve_llm_compute_dtype(llm: nn.Module) -> torch.dtype:
    try:
        return next(llm.parameters()).dtype
    except StopIteration:
        return torch.float32


def _pool_report_hidden_states(*, hidden_states: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
    mask = label_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _load_semantic_lookup(config: DecoderConfig) -> SemanticTargetLookup | None:
    training_targets = config.resolved_semantic_training_targets_jsonl
    training_vocab = config.resolved_semantic_training_vocab_json
    if training_targets is not None or training_vocab is not None:
        if training_targets is None or training_vocab is None:
            raise ValueError("semantic_loss.training_targets_jsonl and semantic_loss.training_vocab_json must be configured together.")
        return SemanticTargetLookup.from_training_targets(
            targets_path=training_targets,
            vocab_path=training_vocab,
            organ_names=config.data.organ_names,
            accepted_sample_weight=float(config.semantic_loss.accepted_sample_weight),
            provisional_sample_weight=float(config.semantic_loss.provisional_sample_weight),
            unresolved_sample_weight=float(config.semantic_loss.unresolved_sample_weight),
            use_confidence_scaling=bool(config.semantic_loss.use_confidence_scaling),
            include_review_required=bool(config.semantic_loss.include_review_required),
            review_required_sample_weight=float(config.semantic_loss.review_required_sample_weight),
        )
    return SemanticTargetLookup.from_jsonl_paths(
        config.resolved_semantic_target_jsonl_paths,
        organ_names=config.data.organ_names,
        accepted_sample_weight=float(config.semantic_loss.accepted_sample_weight),
        provisional_sample_weight=float(config.semantic_loss.provisional_sample_weight),
        unresolved_sample_weight=float(config.semantic_loss.unresolved_sample_weight),
        use_confidence_scaling=bool(config.semantic_loss.use_confidence_scaling),
    )
