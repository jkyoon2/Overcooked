from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .dataset import NULL_GOAL_ID
from .scheduler import DDPMScheduler


def move_batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_hist, z_goal, x0_expert = batch
    return (
        obs_hist.to(device=device, dtype=torch.float32),
        z_goal.to(device=device, dtype=torch.long),
        x0_expert.to(device=device, dtype=torch.float32),
    )


def apply_classifier_free_goal_dropout(
    z_goal: torch.Tensor,
    p_uncond: float,
    null_goal_id: int = NULL_GOAL_ID,
) -> torch.Tensor:
    if p_uncond <= 0.0:
        return z_goal
    mask = torch.rand(z_goal.shape, device=z_goal.device) < p_uncond
    return torch.where(mask, torch.full_like(z_goal, null_goal_id), z_goal)


@torch.no_grad()
def denoised_action_metrics(
    pred_noise: torch.Tensor,
    noised_chunk: torch.Tensor,
    timesteps: torch.Tensor,
    x0_expert: torch.Tensor,
    scheduler: DDPMScheduler,
    n_action_steps: int,
) -> dict[str, float]:
    pred_x0 = scheduler.predict_x0(noised_chunk, pred_noise, timesteps)
    pred_actions = torch.argmax(pred_x0, dim=-1)
    target_actions = torch.argmax(x0_expert, dim=-1)

    action_acc = (pred_actions == target_actions).float().mean()
    exec_end = min(1 + n_action_steps, target_actions.shape[1])
    exec_acc = (pred_actions[:, 1:exec_end] == target_actions[:, 1:exec_end]).float().mean()
    return {
        "action_acc": float(action_acc.detach().cpu()),
        "exec_action_acc": float(exec_acc.detach().cpu()),
    }


def train_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: DDPMScheduler,
    device: torch.device,
    p_uncond: float = 0.1,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    obs_hist, z_goal, x0_expert = move_batch_to_device(batch, device)
    z_goal = apply_classifier_free_goal_dropout(z_goal, p_uncond)

    timesteps = scheduler.sample_timesteps(x0_expert.shape[0], device)
    noise = torch.randn_like(x0_expert)
    noised_chunk = scheduler.add_noise(x0_expert, noise, timesteps)

    pred_noise = model(noised_chunk, timesteps, obs_hist, z_goal)
    loss = F.mse_loss(pred_noise, noise)
    metrics = denoised_action_metrics(
        pred_noise,
        noised_chunk,
        timesteps,
        x0_expert,
        scheduler,
        n_action_steps=getattr(model, "n_action_steps", x0_expert.shape[1] - 1),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip_norm is not None and grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    metrics["loss"] = float(loss.detach().cpu())
    return metrics


@torch.no_grad()
def eval_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scheduler: DDPMScheduler,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    obs_hist, z_goal, x0_expert = move_batch_to_device(batch, device)
    timesteps = scheduler.sample_timesteps(x0_expert.shape[0], device)
    noise = torch.randn_like(x0_expert)
    noised_chunk = scheduler.add_noise(x0_expert, noise, timesteps)
    pred_noise = model(noised_chunk, timesteps, obs_hist, z_goal)
    loss = F.mse_loss(pred_noise, noise)
    metrics = denoised_action_metrics(
        pred_noise,
        noised_chunk,
        timesteps,
        x0_expert,
        scheduler,
        n_action_steps=getattr(model, "n_action_steps", x0_expert.shape[1] - 1),
    )
    metrics["loss"] = float(loss.detach().cpu())
    return metrics


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: DDPMScheduler,
    config: dict[str, Any],
    metadata: dict[str, Any],
    step: int,
    epoch: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict(),
        "model_config": model.config_dict() if hasattr(model, "config_dict") else None,
        "config": config,
        "metadata": metadata,
        "step": step,
        "epoch": epoch,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
