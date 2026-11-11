"""Public problem definition for the 1D synthetic guidance benchmark.

Everything in this module is part of the *fixed problem*, not of any guidance
method: the data distributions, the DDPM noise schedule, the denoiser
architecture and its training recipe, and the estimator of the optimal
(foresight) guidance target.  The settings follow the paper's released 1D code
(`1d-toy/` in https://github.com/zhengqigao/REG): 20 DDPM steps with betas
linear in [1e-4, 2e-2], separate conditional / unconditional epsilon networks
(sinusoidal embeddings of size 128, three residual GELU blocks of width 128),
AdamW lr 1e-3, batch 64, 200 epochs, 8,000 samples per distribution, and the
reward R0(x) = q(x|y) / q(x) evaluated with the data mixtures.

Time-step convention: t = 1..T (1-indexed).  The network receives the
0-indexed step t-1, exactly as the upstream code does.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

T_STEPS = 20
BETA_START = 1e-4
BETA_END = 2e-2
GRID_LOW = -1.0
GRID_HIGH = 2.0


# --------------------------------------------------------------------------
# Data distributions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianMixture1D:
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]

    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        w = torch.tensor(self.weights, dtype=torch.float64)
        comp = torch.multinomial(w / w.sum(), n, replacement=True, generator=generator)
        means = torch.tensor(self.means, dtype=torch.float32)[comp]
        stds = torch.tensor(self.stds, dtype=torch.float32)[comp]
        return means + stds * torch.randn(n, generator=generator)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xf = x.reshape(-1, 1)
        means = torch.tensor(self.means, dtype=xf.dtype, device=xf.device).reshape(1, -1)
        stds = torch.tensor(self.stds, dtype=xf.dtype, device=xf.device).reshape(1, -1)
        w = torch.tensor(self.weights, dtype=xf.dtype, device=xf.device).reshape(1, -1)
        w = w / w.sum()
        comp = torch.log(w) - torch.log(stds) - 0.5 * math.log(2.0 * math.pi) - 0.5 * ((xf - means) / stds) ** 2
        return torch.logsumexp(comp, dim=1).reshape(shape)

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in asdict(self).items()}

    @staticmethod
    def from_dict(d: dict) -> "GaussianMixture1D":
        return GaussianMixture1D(tuple(d["means"]), tuple(d["stds"]), tuple(d["weights"]))


# The paper's 1D example (Section 5.1 / Appendix E.1).  The conditional mixture is
# as stated in the text.  The unconditional mixture follows the authors' released
# configuration (1d-toy/models/*.pkl: uncond_mean = [-2, 0], uncond_std = [0.5, 0.5]),
# which is what produced the paper's Figures 1 and 5 with the released denoisers;
# the text's "N(-1, 0.5^2) + N(1, 0.5^2)" is inconsistent with those figures.
PAPER_COND = GaussianMixture1D(means=(0.5, 1.5), stds=(0.25, 0.25), weights=(0.5, 0.5))
PAPER_UNCOND = GaussianMixture1D(means=(-2.0, 0.0), stds=(0.5, 0.5), weights=(0.5, 0.5))

# Backwards-compatible aliases.
CONDITIONAL_DIST = PAPER_COND
UNCONDITIONAL_DIST = PAPER_UNCOND


def log_reward(x0: torch.Tensor, cond: GaussianMixture1D, uncond: GaussianMixture1D) -> torch.Tensor:
    """log R0(x, y) = log q(x|y) - log q(x)  (guidance scale 1, as in the paper's code)."""
    return cond.log_prob(x0) - uncond.log_prob(x0)


def log_reward_ratio(x0: torch.Tensor) -> torch.Tensor:
    return log_reward(x0, PAPER_COND, PAPER_UNCOND)


def _spaced_means(rng: random.Random, k: int, low: float, high: float, min_gap: float) -> list[float]:
    for _ in range(10_000):
        means = sorted(rng.uniform(low, high) for _ in range(k))
        if all(b - a >= min_gap for a, b in zip(means, means[1:])):
            return means
    raise RuntimeError("could not place mixture means")


def _dirichlet(rng: random.Random, k: int, alpha: float) -> list[float]:
    g = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(g)
    return [v / s for v in g]


def sample_family_problem(rng: random.Random) -> tuple[GaussianMixture1D, GaussianMixture1D]:
    """Public family of paper-like problems used for the visible training slice.

    Conditional: 2-3 narrow components inside [0.15, 1.85].
    Unconditional: one wide component near -2 plus 1-2 wide components near 0.
    The paper's own mixtures are a member of this family.
    """
    kc = rng.choice([2, 3])
    cond = GaussianMixture1D(
        means=tuple(_spaced_means(rng, kc, 0.15, 1.85, 0.45)),
        stds=tuple(rng.uniform(0.17, 0.30) for _ in range(kc)),
        weights=tuple(_dirichlet(rng, kc, 5.0)),
    )
    ku = rng.choice([2, 3])
    right = _spaced_means(rng, ku - 1, -0.4, 0.6, 0.45)
    uncond = GaussianMixture1D(
        means=tuple([rng.uniform(-2.3, -1.7)] + right),
        stds=tuple(rng.uniform(0.35, 0.60) for _ in range(ku)),
        weights=tuple(_dirichlet(rng, ku, 5.0)),
    )
    return cond, uncond


# --------------------------------------------------------------------------
# Denoiser architecture (module names match the upstream state dicts)
# --------------------------------------------------------------------------


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale
        half = self.size // 2
        freq = torch.exp(
            -(math.log(10000.0) / (half - 1)) * torch.arange(half, device=x.device, dtype=x.dtype)
        )
        emb = x.unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

    def __len__(self) -> int:
        return self.size


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.layer = SinusoidalEmbedding(size, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    """Epsilon network of the paper's 1D code (one class, label -1 = unconditional)."""

    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128, num_class: int = 1):
        super().__init__()
        self.time_mlp = PositionalEmbedding(emb_size)
        self.input_mlp = PositionalEmbedding(emb_size, scale=25.0)
        self.class_emb = nn.Embedding(num_class, emb_size)
        layers: list[nn.Module] = [nn.Linear(2 * emb_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 1))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t_index: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x_emb = self.input_mlp(x[:, 0])
        t_emb = self.time_mlp(t_index.to(x.dtype))
        ind = label == -1
        class_embd = self.class_emb(torch.clamp(label, min=0))
        class_embd = torch.where(ind.unsqueeze(-1), torch.zeros_like(class_embd), class_embd)
        return self.joint_mlp(torch.cat((x_emb, t_emb + class_embd), dim=-1))


class Denoiser(nn.Module):
    """eps_theta(x_t, t) for one fixed label; t is 1-indexed."""

    def __init__(self, net: MLP, label: int):
        super().__init__()
        self.net = net
        self.label = label

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (N,) states, t: (N,) 1-indexed steps -> (N,) epsilon predictions."""
        lab = torch.full((x.shape[0],), self.label, dtype=torch.long, device=x.device)
        return self.net(x.reshape(-1, 1), (t - 1).long(), lab).reshape(-1)


# --------------------------------------------------------------------------
# DDPM schedule and sampler (upstream `Diffusion` class)
# --------------------------------------------------------------------------


class Schedule:
    def __init__(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        betas = torch.linspace(BETA_START, BETA_END, T_STEPS, dtype=torch.float32).to(dtype)
        self.T = T_STEPS
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        ).clamp(min=1e-20)
        self.coef1 = self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        self.coef2 = (1.0 - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)

    def as_lists(self) -> dict:
        return {
            "T": self.T,
            "betas": self.betas.double().cpu().tolist(),
            "alphas": self.alphas.double().cpu().tolist(),
            "alpha_bars": self.alpha_bars.double().cpu().tolist(),
            "posterior_variance": self.posterior_variance.double().cpu().tolist(),
        }

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bars[t - 1]
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def posterior_mean(self, x: torch.Tensor, eps: torch.Tensor, step: int) -> torch.Tensor:
        i = step - 1
        ab = self.alpha_bars[i]
        x0_hat = torch.sqrt(1.0 / ab) * x - torch.sqrt(1.0 / ab - 1.0) * eps
        return self.coef1[i] * x0_hat + self.coef2[i] * x


def train_denoiser(
    mixture: GaussianMixture1D,
    label: int,
    seed: int,
    device: torch.device,
    num_samples: int = 8000,
    batch_size: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
) -> MLP:
    """Upstream training recipe (`Diffusion.train`), seeded."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    data = mixture.sample(num_samples, gen).to(device)
    net = MLP().to(device)
    sched = Schedule(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    lab = torch.full((batch_size,), label, dtype=torch.long, device=device)
    dgen = torch.Generator(device=device).manual_seed(seed + 7919)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(num_samples, device=device, generator=dgen)
        for b in range(num_samples // batch_size):
            x0 = data[perm[b * batch_size : (b + 1) * batch_size]].reshape(-1, 1)
            noise = torch.randn(x0.shape, device=device, generator=dgen)
            t_idx = torch.randint(0, T_STEPS, (batch_size,), device=device, generator=dgen)
            ab = sched.alpha_bars[t_idx].reshape(-1, 1)
            noisy = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
            loss = F.mse_loss(net(noisy, t_idx, lab), noise)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
    net.eval()
    return net


def load_denoiser(path: str, label: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> Denoiser:
    net = MLP()
    state = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net = net.to(device=device, dtype=dtype).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return Denoiser(net, label)


# --------------------------------------------------------------------------
# Optimal (foresight) guidance target, Eq. (14)/(17) with Appendix E.1's
# Monte Carlo estimator: run the stochastic conditional chain from x_t to x_0
# K times, average the reward, differentiate the log-average w.r.t. x_t.
# --------------------------------------------------------------------------


def foresight_target(
    cond_model: Denoiser,
    cond: GaussianMixture1D,
    uncond: GaussianMixture1D,
    x_t: torch.Tensor,
    t: int,
    completions: int,
    chunk: int,
    seed: int,
    sched: Schedule,
) -> torch.Tensor:
    """Return g*(x_t, t) = -sqrt(1 - alpha_bar_t) * d/dx_t log E_t(x_t) for a batch of states at one t."""
    device = x_t.device
    n = x_t.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    chunk_logs = []
    chunk_grads = []
    for start in range(0, completions, chunk):
        k = min(chunk, completions - start)
        leaf = x_t.detach().clone().requires_grad_(True)
        x = leaf.repeat_interleave(k)
        for step in range(t, 0, -1):
            tt = torch.full((x.shape[0],), step, device=device, dtype=torch.long)
            eps = cond_model(x, tt)
            x_mean = sched.posterior_mean(x, eps, step)
            if step > 1:
                noise = torch.randn(x.shape, device=device, generator=gen, dtype=x.dtype)
                x = x_mean + sched.posterior_variance[step - 1].sqrt() * noise
            else:
                x = x_mean
        log_r = log_reward(x, cond, uncond).reshape(n, k)
        lse = torch.logsumexp(log_r, dim=1)
        (grad,) = torch.autograd.grad(lse.sum(), leaf)
        chunk_logs.append(lse.detach())
        chunk_grads.append(grad.detach())
    logs = torch.stack(chunk_logs, dim=0)
    grads = torch.stack(chunk_grads, dim=0)
    weights = torch.softmax(logs, dim=0)
    grad_log_e = (weights * grads).sum(dim=0)
    return -torch.sqrt(1.0 - sched.alpha_bars[t - 1]) * grad_log_e
