# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py at diffusers commit 578c9b2c6636ab2424a0e56186268b83623656b2.
"""Qwen-Image-2.1 VAE as a plain `nn.Module` for single-image encode and decode.

The building blocks keep the diffusers parameter names, so the diffusers safetensors load with strict key
matching, and the feature-cache mechanics of the diffusers single-frame path (`_encode` / `_decode`), so the
numerics match. Tiling, slicing, patchify and multi-frame chunking are not part of this module.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

CACHE_T = 2

FeatCache = list[torch.Tensor | str | None]


class QwenImage21AvgDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        factor = factor_t * factor_s * factor_s
        if in_channels * factor % out_channels != 0:
            raise ValueError(
                f"`in_channels` ({in_channels}) times the downsampling factor ({factor}) must be divisible by "
                f"`out_channels` ({out_channels})."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor
        self.group_size = in_channels * factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class QwenImage21DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        if out_channels * self.factor % in_channels != 0:
            raise ValueError(
                f"`out_channels` ({out_channels}) times the upsampling factor ({self.factor}) must be divisible by "
                f"`in_channels` ({in_channels})."
            )
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class QwenImage21CausalConv3d(nn.Conv2d):
    """Image specialization of Wan's causal 3D convolution: a 2D convolution applied to the single frame."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        self.padding = (0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | str | None = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None:
            raise ValueError(
                "This convolution is the image specialization of Wan's causal 3D one: it folds the single frame away "
                "and has no temporal context to prepend, so it cannot take a feature cache."
            )
        x = x.squeeze(2)  # Remove the temporal dimension
        x = F.pad(x, padding)
        x = super().forward(x)
        x = x.unsqueeze(2)  # Add the temporal dimension back
        return x


class QwenImage21RMS_norm(nn.Module):
    """RMS normalization over the channel dimension with a learnable per-channel scale."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias: nn.Parameter | float = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        needs_fp32_normalize = x.dtype in (torch.float16, torch.bfloat16) or any(
            t in str(x.dtype) for t in ("float4_", "float8_")
        )
        normalized = F.normalize(x.float() if needs_fp32_normalize else x, dim=(1 if self.channel_first else -1)).to(
            x.dtype
        )

        return normalized * self.scale * self.gamma + self.bias


class QwenImage21Upsample(nn.Upsample):
    """Upsampling computed in fp32 and returned in the input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class QwenImage21Resample(nn.Module):
    """Spatial resampling ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d') of [B, C, T, H, W]."""

    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        self.resample: nn.Module
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImage21Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImage21Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = QwenImage21CausalConv3d(dim, dim * 2, (1, 1), padding=(0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenImage21CausalConv3d(dim, dim, (1, 1), stride=(1, 1), padding=(0, 0))

        else:
            self.resample = nn.Identity()

    def forward(
        self, x: torch.Tensor, feat_cache: FeatCache | None = None, feat_idx: list[int] | None = None
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class QwenImage21ResidualBlock(nn.Module):
    """Two RMS-norm / SiLU / causal-conv stages with a (projected) residual connection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = nn.SiLU()

        # layers
        self.norm1 = QwenImage21RMS_norm(in_dim, images=False)
        self.conv1 = QwenImage21CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImage21RMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImage21CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut: nn.Module = (
            QwenImage21CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, feat_cache: FeatCache | None = None, feat_idx: list[int] | None = None
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        # Add residual connection
        return x + h


class QwenImage21AttentionBlock(nn.Module):
    """Single-head spatial self-attention applied per frame."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        # layers
        self.norm = QwenImage21RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class QwenImage21MidBlock(nn.Module):
    """Middle block of the encoder and decoder: residual, then (attention, residual) x num_layers."""

    def __init__(self, dim: int, dropout: float = 0.0, num_layers: int = 1) -> None:
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [QwenImage21ResidualBlock(dim, dim, dropout)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImage21AttentionBlock(dim))
            resnets.append(QwenImage21ResidualBlock(dim, dim, dropout))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, x: torch.Tensor, feat_cache: FeatCache | None = None, feat_idx: list[int] | None = None
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        # First residual block
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)

        return x


class QwenImage21ResidualDownBlock(nn.Module):
    """Encoder stage: residual blocks and an optional downsampler, plus an average-pooling shortcut."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        num_res_blocks: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = QwenImage21AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(QwenImage21ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        self.downsampler: QwenImage21Resample | None
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = QwenImage21Resample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(
        self, x: torch.Tensor, feat_cache: FeatCache | None = None, feat_idx: list[int] | None = None
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)

        return x + self.avg_shortcut(x_copy)


class QwenImage21Encoder3d(nn.Module):
    """Encoder: RGBA [B, C_in, T, H, W] -> [B, z_dim, T', H / 2^(len(dim_mult)-1), W / 2^(len(dim_mult)-1)]."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        z_dim: int,
        dim_mult: list[int],
        num_res_blocks: int,
        attn_scales: list[float],
        temperal_downsample: list[bool],
        dropout: float,
        is_residual: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = nn.SiLU()

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = QwenImage21CausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if is_residual:
                self.down_blocks.append(
                    QwenImage21ResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(QwenImage21ResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(QwenImage21AttentionBlock(out_dim))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(QwenImage21Resample(out_dim, mode=mode))
                    scale /= 2.0

        # middle blocks
        self.mid_block = QwenImage21MidBlock(out_dim, dropout, num_layers=1)

        # output blocks
        self.norm_out = QwenImage21RMS_norm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, z_dim, 3, padding=1)

    def forward(
        self, x: torch.Tensor, feat_cache: FeatCache | None = None, feat_idx: list[int] | None = None
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class QwenImage21ResidualUpBlock(nn.Module):
    """Residual decoder stage: residual blocks and an optional upsampler, plus a duplication shortcut."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.avg_shortcut: QwenImage21DupUp3D | None
        if up_flag:
            self.avg_shortcut = QwenImage21DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImage21ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsampler: QwenImage21Resample | None
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = QwenImage21Resample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        x_copy = x.clone()

        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsampler is not None:
            if feat_cache is not None:
                x = self.upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = self.upsampler(x)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x


class QwenImage21UpBlock(nn.Module):
    """Non-residual decoder stage: residual blocks and an optional upsampler."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImage21ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers: nn.ModuleList | None = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImage21Resample(out_dim, mode=upsample_mode)])

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool | None = None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class QwenImage21Decoder3d(nn.Module):
    """Decoder: [B, z_dim, T, h, w] -> [B, C_out, T', h * 2^(len(dim_mult)-1), w * 2^(len(dim_mult)-1)]."""

    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: list[int],
        num_res_blocks: int,
        attn_scales: list[float],
        temperal_upsample: list[bool],
        dropout: float,
        out_channels: int,
        is_residual: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = nn.SiLU()

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = QwenImage21CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = QwenImage21MidBlock(dims[0], dropout, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            # Create and add the upsampling block
            up_block: nn.Module
            if is_residual:
                up_block = QwenImage21ResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                )
            else:
                up_block = QwenImage21UpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = QwenImage21RMS_norm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: FeatCache | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


@dataclass
class QwenImage21VAEConfig:
    """Architecture fields of the diffusers `AutoencoderKLQwenImage21` config.json plus the latent statistics."""

    base_dim: int
    decoder_base_dim: int | None
    z_dim: int
    dim_mult: list[int]
    num_res_blocks: int
    attn_scales: list[float]
    temperal_downsample: list[bool]
    dropout: float
    is_residual: bool
    in_channels: int
    out_channels: int
    latents_mean: list[float]
    latents_std: list[float]

    @classmethod
    def from_json(cls, path: Path) -> "QwenImage21VAEConfig":
        raw = json.loads(path.read_text())
        if raw["patch_size"] is not None:
            raise ValueError(f"patch_size={raw['patch_size']} in {path}: only patch_size=null is supported")
        return cls(
            base_dim=int(raw["base_dim"]),
            decoder_base_dim=None if raw["decoder_base_dim"] is None else int(raw["decoder_base_dim"]),
            z_dim=int(raw["z_dim"]),
            dim_mult=[int(v) for v in raw["dim_mult"]],
            num_res_blocks=int(raw["num_res_blocks"]),
            attn_scales=[float(v) for v in raw["attn_scales"]],
            temperal_downsample=[bool(v) for v in raw["temperal_downsample"]],
            dropout=float(raw["dropout"]),
            is_residual=bool(raw["is_residual"]),
            in_channels=int(raw["in_channels"]),
            out_channels=int(raw["out_channels"]),
            latents_mean=[float(v) for v in raw["latents_mean"]],
            latents_std=[float(v) for v in raw["latents_std"]],
        )


class QwenImage21VAE(nn.Module):
    """Single-image Qwen-Image-2.1 VAE: RGB in [-1, 1] <-> normalized latents [B, z_dim, H/16, W/16].

    The VAE itself works on RGBA. `encode` appends an opaque alpha channel and returns the posterior mode normalized
    per channel with `latents_mean` / `latents_std`; `decode` inverts the normalization and drops the alpha channel.
    The latent statistics are non-persistent buffers, so they follow `.to(device, dtype)` and stay out of the
    state dict.
    """

    def __init__(self, config: QwenImage21VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.z_dim = config.z_dim
        temperal_upsample = config.temperal_downsample[::-1]
        decoder_base_dim = config.base_dim if config.decoder_base_dim is None else config.decoder_base_dim

        self.encoder = QwenImage21Encoder3d(
            in_channels=config.in_channels,
            dim=config.base_dim,
            z_dim=config.z_dim * 2,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temperal_downsample=config.temperal_downsample,
            dropout=config.dropout,
            is_residual=config.is_residual,
        )
        self.quant_conv = QwenImage21CausalConv3d(config.z_dim * 2, config.z_dim * 2, 1)
        self.post_quant_conv = QwenImage21CausalConv3d(config.z_dim, config.z_dim, 1)

        self.decoder = QwenImage21Decoder3d(
            dim=decoder_base_dim,
            z_dim=config.z_dim,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temperal_upsample=temperal_upsample,
            dropout=config.dropout,
            out_channels=config.out_channels,
            is_residual=config.is_residual,
        )

        self.encoder_conv_count = sum(isinstance(m, QwenImage21CausalConv3d) for m in self.encoder.modules())
        self.decoder_conv_count = sum(isinstance(m, QwenImage21CausalConv3d) for m in self.decoder.modules())

        self.latents_mean: torch.Tensor
        self.latents_std: torch.Tensor
        self.register_buffer(
            "latents_mean", torch.tensor(config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.tensor(config.latents_std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False
        )

    @classmethod
    def from_pretrained(cls, vae_dir: Path, device: torch.device, torch_dtype: torch.dtype) -> "QwenImage21VAE":
        """Build from `vae_dir/config.json` and load `vae_dir/diffusion_pytorch_model.safetensors` strictly."""
        config = QwenImage21VAEConfig.from_json(vae_dir / "config.json")
        model = cls(config)
        state_dict = load_file(str(vae_dir / "diffusion_pytorch_model.safetensors"), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device=device, dtype=torch_dtype)
        model.eval()
        return model

    @property
    def dtype(self) -> torch.dtype:
        return self.quant_conv.weight.dtype

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """RGB [B, 3, H, W] in [-1, 1] -> normalized posterior mode [B, z_dim, H/16, W/16] in the module dtype."""
        rgb = rgb.to(self.dtype)
        alpha = torch.ones_like(rgb[:, :1])
        x = torch.cat([rgb, alpha], dim=1).unsqueeze(2)

        feat_cache: FeatCache = [None] * self.encoder_conv_count
        feat_idx = [0]
        out = self.encoder(x[:, :, :1, :, :], feat_cache=feat_cache, feat_idx=feat_idx)
        enc = self.quant_conv(out)

        mode = enc[:, : self.z_dim, 0]
        return (mode - self.latents_mean) / self.latents_std

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latents [B, z_dim, h, w] -> RGB [B, 3, 16h, 16w] in [-1, 1] in the module dtype."""
        latents = latents.to(self.dtype)
        z = (latents * self.latents_std + self.latents_mean).unsqueeze(2)

        x = self.post_quant_conv(z)
        feat_cache: FeatCache = [None] * self.decoder_conv_count
        feat_idx = [0]
        out = self.decoder(x[:, :, 0:1, :, :], feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=True)
        out = torch.clamp(out, min=-1.0, max=1.0)
        return out[:, :3, 0]
