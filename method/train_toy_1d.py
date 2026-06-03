"""Train and evaluate the paper's 1D synthetic REG experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.toy_1d import CONDITIONAL_DIST, UNCONDITIONAL_DIST, log_reward_ratio
from method.reg_guidance import cfg_adjustment, reg_adjustment


@dataclass
class TrainConfig:
    seed: int = 0
    train_samples: int = 8000
    diffusion_steps: int = 20
    beta_start: float = 0.001
    beta_end: float = 0.2
    embed_dim: int = 128
    hidden_dim: int = 128
    hidden_layers: int = 3
    epochs: int = 120
    batch_size: int = 512
    lr: float = 1e-3
    guidance_scale: float = 1.5
    eval_grid_points: int = 301
    eval_steps: tuple[int, ...] = (4, 8, 12, 16, 20)


class FourierFeatures(nn.Module):
    def __init__(self, dim: int, max_freq: float = 64.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.reshape(x.shape[0], -1) * self.freqs.reshape(1, -1)
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)


class EpsilonMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, hidden_layers: int, diffusion_steps: int):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.x_embed = FourierFeatures(embed_dim)
        self.t_embed = FourierFeatures(embed_dim)
        layers: list[nn.Module] = []
        in_dim = 2 * embed_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
        t_scaled = t_index.float().reshape(-1, 1) / float(self.diffusion_steps)
        return self.net(torch.cat([self.x_embed(x_t), self.t_embed(t_scaled)], dim=-1))


class DDPM1D:
    def __init__(self, cfg: TrainConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_steps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t_index: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bars[t_index - 1].reshape(-1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def x0_hat(self, x_t: torch.Tensor, t_index: torch.Tensor, eps_cond: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bars[t_index - 1].reshape(-1, 1)
        return (x_t - (1.0 - ab).sqrt() * eps_cond) / ab.sqrt()

    def deterministic_previous(self, x_t: torch.Tensor, t_index: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """DDPM mean transition from x_t to x_{t-1}, preserving gradients."""
        beta_t = self.betas[t_index - 1].reshape(-1, 1)
        alpha_t = self.alphas[t_index - 1].reshape(-1, 1)
        alpha_bar_t = self.alpha_bars[t_index - 1].reshape(-1, 1)
        return (x_t - beta_t / (1.0 - alpha_bar_t).sqrt() * eps) / alpha_t.sqrt()

    def denoise_to_zero(self, model: nn.Module, x_t: torch.Tensor, start_step: int) -> torch.Tensor:
        """Deterministic foresight chain used to evaluate the joint-scaling target."""
        x = x_t
        batch = x.shape[0]
        for step in range(start_step, 0, -1):
            t = torch.full((batch,), step, device=x.device, dtype=torch.long)
            eps = model(x, t)
            x = self.deterministic_previous(x, t, eps)
        return x


def make_loader(dist, cfg: TrainConfig, generator: torch.Generator) -> DataLoader:
    x0 = dist.sample(cfg.train_samples, generator=generator)
    return DataLoader(TensorDataset(x0), batch_size=cfg.batch_size, shuffle=True, drop_last=True)


def train_one(model: nn.Module, loader: DataLoader, diffusion: DDPM1D, cfg: TrainConfig, device: torch.device) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    losses: list[float] = []
    for _epoch in range(cfg.epochs):
        total = 0.0
        batches = 0
        for (x0_cpu,) in loader:
            x0 = x0_cpu.to(device)
            t = torch.randint(1, cfg.diffusion_steps + 1, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise)
            pred = model(x_t, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            batches += 1
        losses.append(total / max(1, batches))
    return losses


def evaluate_guidance(cond_model: nn.Module, uncond_model: nn.Module, diffusion: DDPM1D, cfg: TrainConfig) -> dict:
    device = diffusion.device
    records = []
    all_cfg_abs = []
    all_reg_abs = []
    all_wins = []
    for step in cfg.eval_steps:
        x = torch.linspace(-1.0, 2.0, cfg.eval_grid_points, device=device).reshape(-1, 1)
        t = torch.full((x.shape[0],), step, device=device, dtype=torch.long)
        x_req = x.detach().clone().requires_grad_(True)
        eps_c = cond_model(x_req, t)
        eps_u = uncond_model(x_req, t)
        ab = diffusion.alpha_bars[step - 1]
        vanilla = cfg_adjustment(eps_c, eps_u, cfg.guidance_scale).detach()
        reg = reg_adjustment(x_req, eps_c, eps_u, torch.sqrt(1.0 - ab), cfg.guidance_scale).detach()

        x_req2 = x.detach().clone().requires_grad_(True)
        x0_deterministic = diffusion.denoise_to_zero(cond_model, x_req2, step)
        reward = log_reward_ratio(x0_deterministic).sum()
        reward_grad = torch.autograd.grad(reward, x_req2, create_graph=False)[0]
        optimal = (-torch.sqrt(1.0 - ab) * reward_grad).detach()

        cfg_abs = (vanilla - optimal).abs()
        reg_abs = (reg - optimal).abs()
        win = (reg_abs < cfg_abs).float()
        records.append(
            {
                "time_step": step,
                "cfg_mae": float(cfg_abs.mean().cpu()),
                "reg_mae": float(reg_abs.mean().cpu()),
                "reg_win_rate": float(win.mean().cpu()),
                "cfg_win_rate": float((1.0 - win).mean().cpu()),
            }
        )
        all_cfg_abs.append(cfg_abs)
        all_reg_abs.append(reg_abs)
        all_wins.append(win)

    cfg_all = torch.cat(all_cfg_abs)
    reg_all = torch.cat(all_reg_abs)
    win_all = torch.cat(all_wins)
    return {
        "records": records,
        "summary": {
            "cfg_mae": float(cfg_all.mean().cpu()),
            "reg_mae": float(reg_all.mean().cpu()),
            "relative_mae_reduction": float(((cfg_all.mean() - reg_all.mean()) / cfg_all.mean()).cpu()),
            "reg_win_rate": float(win_all.mean().cpu()),
            "cfg_win_rate": float((1.0 - win_all).mean().cpu()),
            "additional_vjp_per_guided_step": 1.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs/toy_1d")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.quick:
        cfg.epochs = min(cfg.epochs, 3)
        cfg.train_samples = 1024
        cfg.eval_grid_points = 51

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    diffusion = DDPM1D(cfg, device)
    cond_model = EpsilonMLP(cfg.embed_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.diffusion_steps).to(device)
    uncond_model = EpsilonMLP(cfg.embed_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.diffusion_steps).to(device)

    cond_losses = train_one(cond_model, make_loader(CONDITIONAL_DIST, cfg, generator), diffusion, cfg, device)
    uncond_losses = train_one(uncond_model, make_loader(UNCONDITIONAL_DIST, cfg, generator), diffusion, cfg, device)
    guidance = evaluate_guidance(cond_model, uncond_model, diffusion, cfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"cond": cond_model.state_dict(), "uncond": uncond_model.state_dict(), "config": asdict(cfg)}, outdir / "checkpoint.pt")
    result = {
        "config": asdict(cfg),
        "device": str(device),
        "cond_final_loss": cond_losses[-1],
        "uncond_final_loss": uncond_losses[-1],
        "cond_losses": cond_losses,
        "uncond_losses": uncond_losses,
        "guidance": guidance,
    }
    with (outdir / "results.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["guidance"]["summary"], indent=2))


if __name__ == "__main__":
    main()
