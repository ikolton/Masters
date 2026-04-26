"""Patch tokenization and study-level aggregation modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer(nn.Module):
    def __init__(self, *, input_dim: int, model_dim: int, summary_grid: tuple[int, int, int]) -> None:
        super().__init__()
        self.summary_grid = tuple(int(v) for v in summary_grid)
        self.proj = nn.Conv3d(int(input_dim), int(model_dim), kernel_size=1)
        self.norm = nn.LayerNorm(int(model_dim))

    @property
    def token_count(self) -> int:
        d, h, w = self.summary_grid
        return int(d * h * w)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        projected = self.proj(feature_map)
        pooled = F.adaptive_avg_pool3d(projected, self.summary_grid)
        tokens = pooled.flatten(start_dim=2).transpose(1, 2)
        return self.norm(tokens)


class PatchPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.net(positions)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentAttentionBlock(nn.Module):
    def __init__(self, *, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_token_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_ff = FeedForward(dim, dropout)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.self_ff = FeedForward(dim, dropout)

    def forward(self, latents: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        cross_update, _ = self.cross_attn(
            self.cross_query_norm(latents),
            self.cross_token_norm(tokens),
            self.cross_token_norm(tokens),
            key_padding_mask=None if token_mask is None else ~token_mask,
            need_weights=False,
        )
        latents = latents + cross_update
        latents = latents + self.cross_ff(latents)
        self_update, _ = self.self_attn(
            self.self_norm(latents),
            self.self_norm(latents),
            self.self_norm(latents),
            need_weights=False,
        )
        latents = latents + self_update
        latents = latents + self.self_ff(latents)
        return latents


class LatentStudyAggregator(nn.Module):
    def __init__(self, *, model_dim: int, num_latents: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(int(num_latents), int(model_dim)) * 0.02)
        self.blocks = nn.ModuleList(
            [
                LatentAttentionBlock(dim=int(model_dim), num_heads=int(num_heads), dropout=float(dropout))
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(int(model_dim))

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size = tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            latents = block(latents, tokens, token_mask)
        return self.output_norm(latents)


class OrganQueryHead(nn.Module):
    def __init__(self, *, query_count: int, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(int(query_count), int(model_dim)) * 0.02)
        self.query_norm = nn.LayerNorm(int(model_dim))
        self.token_norm = nn.LayerNorm(int(model_dim))
        self.attn = nn.MultiheadAttention(int(model_dim), num_heads=int(num_heads), dropout=float(dropout), batch_first=True)
        self.proj = nn.Sequential(nn.LayerNorm(int(model_dim)), nn.Linear(int(model_dim), int(model_dim)))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size = latents.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(
            self.query_norm(queries),
            self.token_norm(latents),
            self.token_norm(latents),
            need_weights=False,
        )
        return F.normalize(self.proj(attended), dim=-1)


class OrganPatchAttentionHead(nn.Module):
    def __init__(self, *, query_count: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(int(query_count), int(model_dim)) * 0.02)
        self.query_norm = nn.LayerNorm(int(model_dim))
        self.token_norm = nn.LayerNorm(int(model_dim))
        self.key = nn.Linear(int(model_dim), int(model_dim), bias=False)
        self.value = nn.Linear(int(model_dim), int(model_dim), bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.proj = nn.Sequential(nn.LayerNorm(int(model_dim)), nn.Linear(int(model_dim), int(model_dim)))
        self.scale = float(model_dim) ** -0.5

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = tokens.shape[0]
        queries = self.query_norm(self.queries).unsqueeze(0).expand(batch_size, -1, -1)
        normalized_tokens = self.token_norm(tokens)
        keys = self.key(normalized_tokens)
        values = self.value(normalized_tokens)
        logits = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        if token_mask is not None:
            logits = logits.masked_fill(~token_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        features = torch.matmul(weights, values)
        return F.normalize(self.proj(features), dim=-1), logits


class PatchSummaryHead(nn.Module):
    def __init__(self, *, model_dim: int, num_heads: int, dropout: float, summary_mode: str = "attention") -> None:
        super().__init__()
        if summary_mode not in {"attention", "attention_mean"}:
            raise ValueError("summary_mode must be 'attention' or 'attention_mean'.")
        self.summary_mode = summary_mode
        self.query = nn.Parameter(torch.randn(1, int(model_dim)) * 0.02)
        self.query_norm = nn.LayerNorm(int(model_dim))
        self.token_norm = nn.LayerNorm(int(model_dim))
        self.attn = nn.MultiheadAttention(int(model_dim), num_heads=int(num_heads), dropout=float(dropout), batch_first=True)
        input_dim = int(model_dim) * (2 if summary_mode == "attention_mean" else 1)
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, int(model_dim)))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        queries = self.query.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        normalized_tokens = self.token_norm(patch_tokens)
        attended, _ = self.attn(
            self.query_norm(queries),
            normalized_tokens,
            normalized_tokens,
            need_weights=False,
        )
        summary = attended.squeeze(1)
        if self.summary_mode == "attention_mean":
            summary = torch.cat([summary, patch_tokens.mean(dim=1)], dim=-1)
        return self.proj(summary)


class GridFeatureCombiner(nn.Module):
    def __init__(self, *, model_dim: int, depth: int, num_heads: int, dropout: float, use_global_token: bool = False) -> None:
        super().__init__()
        self.use_global_token = bool(use_global_token)
        self.global_token = nn.Parameter(torch.randn(1, 1, int(model_dim)) * 0.02) if self.use_global_token else None
        self.position_embedding = nn.Sequential(
            nn.Linear(9, int(model_dim)),
            nn.GELU(),
            nn.Linear(int(model_dim), int(model_dim)),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(num_heads),
            dim_feedforward=int(model_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(int(model_dim))

    def forward(self, patch_summaries: torch.Tensor, position_features: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = patch_summaries + self.position_embedding(position_features)
        if self.global_token is not None:
            global_tokens = self.global_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([global_tokens, tokens], dim=1)
            if token_mask is not None:
                global_mask = torch.ones((token_mask.shape[0], 1), device=token_mask.device, dtype=token_mask.dtype)
                token_mask = torch.cat([global_mask, token_mask], dim=1)
        padding_mask = None if token_mask is None else ~token_mask
        tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.norm(tokens)


class AlignmentProjectionHead(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        dropout: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(int(model_dim), int(hidden_dim))]
        if layer_norm:
            layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.extend([nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), int(hidden_dim))])
        if layer_norm:
            layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.extend([nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), int(bottleneck_dim)), nn.GELU()])
        self.layers = nn.Sequential(*layers)
        self.last_layer = nn.Linear(int(bottleneck_dim), int(model_dim), bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.last_layer(self.layers(embeddings))


class StudyReportHead(nn.Module):
    def __init__(self, *, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, int(model_dim)) * 0.02)
        self.query_norm = nn.LayerNorm(int(model_dim))
        self.token_norm = nn.LayerNorm(int(model_dim))
        self.attn = nn.MultiheadAttention(int(model_dim), num_heads=int(num_heads), dropout=float(dropout), batch_first=True)
        self.proj = nn.Sequential(nn.LayerNorm(int(model_dim)), nn.Linear(int(model_dim), int(model_dim)))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        queries = self.query.unsqueeze(0).expand(latents.shape[0], -1, -1)
        attended, _ = self.attn(
            self.query_norm(queries),
            self.token_norm(latents),
            self.token_norm(latents),
            need_weights=False,
        )
        return F.normalize(self.proj(attended.squeeze(1)), dim=-1)

