from __future__ import annotations

import torch
import torch.nn as nn


def _init_layer(module: nn.Module, gain: float) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class OvercookedObsEncoder(nn.Module):
    """CNN observation encoder for `(B, n_obs, H, W, 25)` tensors."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_obs: int = 2,
        obs_emb_dim: int = 64,
    ) -> None:
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError(f"obs_shape must be (H, W, C), got {obs_shape}")
        height, width, channels = obs_shape
        if channels != 25:
            raise ValueError(f"Expected 25 observation channels, got {channels}")

        self.obs_shape = obs_shape
        self.n_obs = n_obs
        self.obs_emb_dim = obs_emb_dim

        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat_dim = 32 * height * width
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, obs_emb_dim),
            nn.ReLU(),
            nn.LayerNorm(obs_emb_dim),
            nn.Linear(obs_emb_dim, obs_emb_dim),
            nn.ReLU(),
            nn.LayerNorm(obs_emb_dim),
            nn.Linear(obs_emb_dim, obs_emb_dim),
            nn.ReLU(),
            nn.LayerNorm(obs_emb_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        gain = nn.init.calculate_gain("relu")
        for module in self.modules():
            _init_layer(module, gain)

    @property
    def output_dim(self) -> int:
        return self.n_obs * self.obs_emb_dim

    def forward(self, obs_hist: torch.Tensor) -> torch.Tensor:
        if obs_hist.dim() != 5:
            raise ValueError(f"Expected obs_hist shape (B,n_obs,H,W,C), got {tuple(obs_hist.shape)}")
        batch_size, n_obs, height, width, channels = obs_hist.shape
        if n_obs != self.n_obs:
            raise ValueError(f"Expected n_obs={self.n_obs}, got {n_obs}")
        if (height, width, channels) != self.obs_shape:
            raise ValueError(
                f"Expected obs shape {self.obs_shape}, got {(height, width, channels)}"
            )

        x = obs_hist.reshape(batch_size * n_obs, height, width, channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = self.proj(x)
        return x.reshape(batch_size, n_obs * self.obs_emb_dim)
