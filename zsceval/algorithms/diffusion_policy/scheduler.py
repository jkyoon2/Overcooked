from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class SchedulerConfig:
    num_train_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02


def _randn_like(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


class DDPMScheduler:
    """Minimal DDPM/DDIM scheduler for eps-prediction diffusion policies."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.config = SchedulerConfig(num_train_timesteps, beta_start, beta_end)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    @property
    def num_train_timesteps(self) -> int:
        return self.config.num_train_timesteps

    def state_dict(self) -> dict:
        return {"config": asdict(self.config)}

    @classmethod
    def from_config(cls, config: dict) -> "DDPMScheduler":
        return cls(**config)

    def _values(self, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return getattr(self, name).to(device=device, dtype=dtype)

    def _extract(self, name: str, timesteps: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        values = self._values(name, sample.device, sample.dtype)
        timesteps = timesteps.long()
        if timesteps.dim() == 0:
            timesteps = timesteps[None].expand(sample.shape[0])
        out = values[timesteps]
        while out.dim() < sample.dim():
            out = out.unsqueeze(-1)
        return out

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self._extract("sqrt_alphas_cumprod", timesteps, original_samples)
        sqrt_one_minus_alpha = self._extract(
            "sqrt_one_minus_alphas_cumprod",
            timesteps,
            original_samples,
        )
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

    def inference_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if num_inference_steps is None:
            num_inference_steps = self.num_train_timesteps
        num_inference_steps = min(num_inference_steps, self.num_train_timesteps)
        timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            device=device,
        ).round().long()
        return torch.unique_consecutive(timesteps)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: int | None = None,
        method: str = "ddim",
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if method == "ddim":
            return self.ddim_step(model_output, timestep, sample, prev_timestep, eta, generator)
        if method == "ddpm":
            return self.ddpm_step(model_output, timestep, sample, generator)
        raise ValueError(f"Unknown scheduler method: {method}")

    def predict_x0(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self._extract("sqrt_alphas_cumprod", timesteps, sample)
        sqrt_one_minus_alpha = self._extract("sqrt_one_minus_alphas_cumprod", timesteps, sample)
        return (sample - sqrt_one_minus_alpha * model_output) / sqrt_alpha

    def ddpm_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = sample.shape[0]
        t = torch.full((batch_size,), timestep, dtype=torch.long, device=sample.device)
        pred_x0 = self.predict_x0(sample, model_output, t)

        beta_t = self._extract("betas", t, sample)
        alpha_t = self._extract("alphas", t, sample)
        alpha_prod_t = self._extract("alphas_cumprod", t, sample)

        prev_t = torch.clamp(t - 1, min=0)
        alpha_prod_prev = self._extract("alphas_cumprod", prev_t, sample)
        alpha_prod_prev = torch.where(
            (t > 0).reshape(batch_size, *([1] * (sample.dim() - 1))),
            alpha_prod_prev,
            torch.ones_like(alpha_prod_prev),
        )

        pred_x0_coeff = torch.sqrt(alpha_prod_prev) * beta_t / (1.0 - alpha_prod_t)
        sample_coeff = torch.sqrt(alpha_t) * (1.0 - alpha_prod_prev) / (1.0 - alpha_prod_t)
        mean = pred_x0_coeff * pred_x0 + sample_coeff * sample

        variance = beta_t * (1.0 - alpha_prod_prev) / (1.0 - alpha_prod_t)
        noise = _randn_like(sample, generator)
        nonzero_mask = (t > 0).reshape(batch_size, *([1] * (sample.dim() - 1))).to(sample.dtype)
        return mean + nonzero_mask * torch.sqrt(torch.clamp(variance, min=1e-20)) * noise

    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: int | None = None,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = sample.shape[0]
        t = torch.full((batch_size,), timestep, dtype=torch.long, device=sample.device)
        if prev_timestep is None:
            prev_timestep = timestep - 1

        pred_x0 = self.predict_x0(sample, model_output, t)
        alpha_prod_t = self._extract("alphas_cumprod", t, sample)

        if prev_timestep >= 0:
            prev_t = torch.full((batch_size,), prev_timestep, dtype=torch.long, device=sample.device)
            alpha_prod_prev = self._extract("alphas_cumprod", prev_t, sample)
        else:
            alpha_prod_prev = torch.ones_like(alpha_prod_t)

        variance = (1.0 - alpha_prod_prev) / (1.0 - alpha_prod_t)
        variance = variance * (1.0 - alpha_prod_t / alpha_prod_prev)
        sigma = eta * torch.sqrt(torch.clamp(variance, min=0.0))
        pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_prod_prev - sigma**2, min=0.0)) * model_output
        prev_sample = torch.sqrt(alpha_prod_prev) * pred_x0 + pred_dir
        if eta > 0.0 and prev_timestep >= 0:
            prev_sample = prev_sample + sigma * _randn_like(sample, generator)
        return prev_sample
