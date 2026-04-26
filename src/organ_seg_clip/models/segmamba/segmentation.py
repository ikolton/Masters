"""SegMamba-style tile segmentation decoder adapted from upstream SegMamba/MambaHoME."""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock


class SegMambaSegmentationHead(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int,
        out_chans: int,
        feat_size: tuple[int, int, int, int],
        hidden_size: int | None = None,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims: int = 3,
        full_resolution: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(hidden_size if hidden_size is not None else feat_size[-1] * 2)
        self.full_resolution = bool(full_resolution)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=int(in_chans),
            out_channels=int(feat_size[0]),
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=int(feat_size[0]),
            out_channels=int(feat_size[1]),
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=int(feat_size[1]),
            out_channels=int(feat_size[2]),
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=int(feat_size[2]),
            out_channels=int(feat_size[3]),
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=int(feat_size[3]),
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden,
            out_channels=int(feat_size[3]),
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=int(feat_size[3]),
            out_channels=int(feat_size[2]),
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=int(feat_size[2]),
            out_channels=int(feat_size[1]),
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        if self.full_resolution:
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=int(feat_size[1]),
                out_channels=int(feat_size[0]),
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=int(feat_size[0]),
                out_channels=int(feat_size[0]),
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            out_in_channels = int(feat_size[0])
        else:
            for parameter in self.encoder1.parameters():
                parameter.requires_grad_(False)
            self.decoder2 = None
            self.decoder1 = None
            out_in_channels = int(feat_size[1])
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=out_in_channels,
            out_channels=int(out_chans),
        )

    def forward(self, image: torch.Tensor, feature_pyramid: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x2, x3, x4, x5 = feature_pyramid
        enc2 = self.encoder2(x2)
        enc3 = self.encoder3(x3)
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(x5)
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        if self.full_resolution:
            enc1 = self.encoder1(image)
            dec0 = self.decoder2(dec1, enc1)
            out = self.decoder1(dec0)
            return self.out(out)
        return self.out(dec1)
