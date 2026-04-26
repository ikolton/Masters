"""Adapted SegMamba encoder blocks for tiled 3D encoding."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm import Mamba
except Exception as exc:  # pragma: no cover - runtime optional
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = float(eps)
        self.data_format = data_format
        if self.data_format not in {"channels_last", "channels_first"}:
            raise NotImplementedError(f"Unsupported data_format={data_format!r}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None, None] * x + self.bias[:, None, None, None]


class MambaLayer(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, num_slices: int | None = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(dim)
        self.cpu_fallback = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.mamba = None
        if Mamba is not None:
            mamba_kwargs = {
                "d_model": dim,
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
            }
            try:
                self.mamba = Mamba(bimamba_type="v3", nslices=num_slices, **mamba_kwargs)
            except TypeError:
                self.mamba = Mamba(**mamba_kwargs)
        if self.mamba is not None and torch.cuda.is_available():
            for parameter in self.cpu_fallback.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.shape[:2]
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {channels}.")
        image_dims = x.shape[2:]
        token_count = x.shape[2:].numel()
        x_skip = x
        x = x.reshape(batch_size, channels, token_count).transpose(-1, -2)
        x = self.norm(x)
        if self.mamba is not None and x.is_cuda:
            x = self.mamba(x)
        else:
            x = self.cpu_fallback(x)
        x = x.transpose(-1, -2).reshape(batch_size, channels, *image_dims)
        return x + x_skip


class MlpChannel(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GSC(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonliner = nn.ReLU()
        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonliner2 = nn.ReLU()
        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonliner3 = nn.ReLU()
        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonliner4 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.nonliner(self.norm(self.proj(x)))
        x1 = self.nonliner2(self.norm2(self.proj2(x1)))
        x2 = self.nonliner3(self.norm3(self.proj3(x)))
        x = self.nonliner4(self.norm4(self.proj4(x1 + x2)))
        return x + residual


class SegMambaEncoder(nn.Module):
    """SegMamba local encoder adapted from the upstream implementation."""

    def __init__(
        self,
        in_chans: int = 1,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        dims: tuple[int, int, int, int] = (48, 96, 192, 384),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3))
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.InstanceNorm3d(dims[i]),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        for i in range(4):
            self.gscs.append(GSC(dims[i]))
            self.stages.append(
                nn.Sequential(
                    *[
                        MambaLayer(
                            dim=dims[i],
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                            num_slices=num_slices_list[i],
                        )
                        for _ in range(depths[i])
                    ]
                )
            )

        self.out_indices = tuple(int(v) for v in out_indices)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            setattr(self, f"norm{i_layer}", layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def _maybe_checkpoint(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and x.requires_grad:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        for index in range(4):
            x = self.downsample_layers[index](x)
            x = self._maybe_checkpoint(self.gscs[index], x)
            x = self._maybe_checkpoint(self.stages[index], x)
            if index in self.out_indices:
                norm_layer = getattr(self, f"norm{index}")
                x_out = self._maybe_checkpoint(nn.Sequential(norm_layer, self.mlps[index]), x)
                outputs.append(x_out)
        return tuple(outputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.forward_features(x)
