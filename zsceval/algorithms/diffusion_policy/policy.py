from __future__ import annotations

import torch
import torch.nn as nn

from .dataset import NULL_GOAL_ID
from .obs_encoder import OvercookedObsEncoder
from .scheduler import DDPMScheduler
from .unet1d import ConditionalUnet1D


class DiffusionOvercookedPolicy(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_obs: int = 2,
        horizon: int = 16,
        n_action_steps: int = 8,
        action_dim: int = 6,
        obs_emb_dim: int = 64,
        goal_emb_dim: int = 64,
        down_dims: list[int] | tuple[int, ...] = (256, 512, 1024),
        num_goals: int = 5,
        num_diffusion_steps: int = 100,
    ) -> None:
        super().__init__()
        if horizon < 1 + n_action_steps:
            raise ValueError("horizon must be at least 1 + n_action_steps for planned execution slice")

        self.obs_shape = obs_shape
        self.n_obs = n_obs
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.obs_emb_dim = obs_emb_dim
        self.goal_emb_dim = goal_emb_dim
        self.down_dims = tuple(int(v) for v in down_dims)
        self.num_goals = num_goals

        self.obs_encoder = OvercookedObsEncoder(
            obs_shape,
            n_obs=n_obs,
            obs_emb_dim=obs_emb_dim,
        )
        self.goal_embedding = nn.Embedding(num_goals, goal_emb_dim)
        self.global_cond_dim = self.obs_encoder.output_dim + goal_emb_dim
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=self.global_cond_dim,
            down_dims=self.down_dims,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_steps)

    def encode_condition(self, obs_hist: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        obs_hist = obs_hist.float()
        if z_goal.dim() == 0:
            z_goal = z_goal[None]
        z_goal = z_goal.long()
        obs_feat = self.obs_encoder(obs_hist)
        goal_emb = self.goal_embedding(z_goal)
        return torch.cat([obs_feat, goal_emb], dim=-1)

    def forward(
        self,
        noised_chunk: torch.Tensor,
        t: torch.Tensor,
        obs_hist: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        global_cond = self.encode_condition(obs_hist, z_goal)
        return self.unet(noised_chunk.float(), t, global_cond)

    def denoise(
        self,
        obs_hist: torch.Tensor,
        z_goal: torch.Tensor,
        scheduler: DDPMScheduler | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float = 1.0,
        scheduler_method: str = "ddim",
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        scheduler = scheduler or self.scheduler
        device = next(self.parameters()).device
        obs_hist = obs_hist.to(device=device, dtype=torch.float32)
        z_goal = z_goal.to(device=device, dtype=torch.long)
        if z_goal.dim() == 0:
            z_goal = z_goal[None]

        batch_size = obs_hist.shape[0]
        sample = torch.randn(
            (batch_size, self.horizon, self.action_dim),
            device=device,
            dtype=obs_hist.dtype,
            generator=generator,
        )

        timesteps = scheduler.inference_timesteps(num_inference_steps, device=device)
        for idx, timestep in enumerate(timesteps):
            t_int = int(timestep.item())
            t_batch = torch.full((batch_size,), t_int, device=device, dtype=torch.long)

            if guidance_scale != 1.0:
                null_goal = torch.full_like(z_goal, NULL_GOAL_ID)
                model_sample = torch.cat([sample, sample], dim=0)
                model_t = torch.cat([t_batch, t_batch], dim=0)
                model_obs = torch.cat([obs_hist, obs_hist], dim=0)
                model_goal = torch.cat([null_goal, z_goal], dim=0)
                eps_uncond, eps_cond = self(model_sample, model_t, model_obs, model_goal).chunk(2, dim=0)
                model_output = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                model_output = self(sample, t_batch, obs_hist, z_goal)

            prev_timestep = int(timesteps[idx + 1].item()) if idx + 1 < len(timesteps) else -1
            sample = scheduler.step(
                model_output,
                t_int,
                sample,
                prev_timestep=prev_timestep,
                method=scheduler_method,
                eta=eta,
                generator=generator,
            )

        return sample

    @torch.no_grad()
    def sample_action_chunk(
        self,
        obs_hist: torch.Tensor,
        z_goal: torch.Tensor,
        available_actions: torch.Tensor | None = None,
        scheduler: DDPMScheduler | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float = 1.0,
        scheduler_method: str = "ddim",
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = self.denoise(
            obs_hist,
            z_goal,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            scheduler_method=scheduler_method,
            eta=eta,
            generator=generator,
        )
        chunk_logits = logits[:, 1 : 1 + self.n_action_steps, :]
        if available_actions is not None:
            mask = self._expand_available_actions(available_actions, chunk_logits)
            chunk_logits = chunk_logits.masked_fill(~mask, -1e10)
        actions = torch.argmax(chunk_logits, dim=-1)
        if return_logits:
            return actions, chunk_logits
        return actions

    def _expand_available_actions(
        self,
        available_actions: torch.Tensor,
        chunk_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_steps, action_dim = chunk_logits.shape
        mask = torch.as_tensor(available_actions, device=chunk_logits.device).bool()
        if mask.dim() == 1:
            mask = mask.reshape(1, 1, action_dim).expand(batch_size, n_steps, action_dim)
        elif mask.dim() == 2:
            if mask.shape == (batch_size, action_dim):
                mask = mask[:, None, :].expand(batch_size, n_steps, action_dim)
            elif batch_size == 1 and mask.shape == (n_steps, action_dim):
                mask = mask[None, :, :]
            else:
                raise ValueError(f"Unsupported available_actions shape: {tuple(mask.shape)}")
        elif mask.dim() == 3:
            if mask.shape != (batch_size, n_steps, action_dim):
                raise ValueError(f"Unsupported available_actions shape: {tuple(mask.shape)}")
        else:
            raise ValueError(f"Unsupported available_actions shape: {tuple(mask.shape)}")
        return mask

    def config_dict(self) -> dict:
        return {
            "obs_shape": self.obs_shape,
            "n_obs": self.n_obs,
            "horizon": self.horizon,
            "n_action_steps": self.n_action_steps,
            "action_dim": self.action_dim,
            "obs_emb_dim": self.obs_emb_dim,
            "goal_emb_dim": self.goal_emb_dim,
            "down_dims": list(self.down_dims),
            "num_goals": self.num_goals,
            "global_cond_dim": self.global_cond_dim,
            "scheduler": self.scheduler.state_dict()["config"],
        }
