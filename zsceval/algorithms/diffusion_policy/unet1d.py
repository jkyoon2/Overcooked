from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.conv1(x)
        h = self.norm1(h)
        scale, bias = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + residual


class DownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, downsample: bool) -> None:
        super().__init__()
        self.res1 = FiLMResidualBlock1D(in_channels, out_channels, cond_dim)
        self.res2 = FiLMResidualBlock1D(out_channels, out_channels, cond_dim)
        self.downsample = (
            nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cond_dim: int,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.res1 = FiLMResidualBlock1D(in_channels + skip_channels, out_channels, cond_dim)
        self.res2 = FiLMResidualBlock1D(out_channels, out_channels, cond_dim)
        self.upsample = (
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if upsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        return self.upsample(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        global_cond_dim: int = 320,
        down_dims: Sequence[int] | None = None,
        time_emb_dim: int = 128,
    ) -> None:
        super().__init__()
        down_dims = list(down_dims or [256, 512, 1024])
        if not down_dims:
            raise ValueError("down_dims must not be empty")

        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        self.down_dims = down_dims
        self.time_emb_dim = time_emb_dim
        cond_dim = time_emb_dim + global_cond_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.input_proj = nn.Conv1d(input_dim, down_dims[0], kernel_size=3, padding=1)

        down_blocks = []
        in_channels = down_dims[0]
        for idx, out_channels in enumerate(down_dims):
            down_blocks.append(
                DownBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    cond_dim=cond_dim,
                    downsample=idx < len(down_dims) - 1,
                )
            )
            in_channels = out_channels
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_blocks = nn.ModuleList(
            [
                FiLMResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim),
                FiLMResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim),
            ]
        )

        up_blocks = []
        current_channels = down_dims[-1]
        reversed_dims = list(reversed(down_dims))
        for idx, skip_channels in enumerate(reversed_dims):
            out_channels = down_dims[-idx - 2] if idx < len(down_dims) - 1 else down_dims[0]
            up_blocks.append(
                UpBlock1D(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    cond_dim=cond_dim,
                    upsample=idx < len(down_dims) - 1,
                )
            )
            current_channels = out_channels
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Sequential(
            nn.Conv1d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(current_channels, input_dim, kernel_size=1),
        )

    def forward(self, sample: torch.Tensor, t: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        if sample.dim() != 3:
            raise ValueError(f"Expected sample shape (B,horizon,C), got {tuple(sample.shape)}")
        if global_cond.dim() != 2:
            raise ValueError(f"Expected global_cond shape (B,D), got {tuple(global_cond.shape)}")
        if t.dim() == 0:
            t = t[None].expand(sample.shape[0])

        target_horizon = sample.shape[1]
        time_emb = self.time_mlp(t)
        cond = torch.cat([time_emb, global_cond], dim=-1)

        x = sample.transpose(1, 2)
        x = self.input_proj(x)

        skips = []
        for block in self.down_blocks:
            x, skip = block(x, cond)
            skips.append(skip)

        for block in self.mid_blocks:
            x = block(x, cond)

        for block in self.up_blocks:
            x = block(x, skips.pop(), cond)

        x = self.final(x)
        if x.shape[-1] != target_horizon:
            x = F.interpolate(x, size=target_horizon, mode="nearest")
        return x.transpose(1, 2)
