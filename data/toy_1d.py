"""Problem data for the 1D synthetic guidance benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaussianMixture1D:
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]

    def sample(self, n: int, generator: torch.Generator | None = None, device: str = "cpu") -> torch.Tensor:
        weights = torch.tensor(self.weights, device=device, dtype=torch.float32)
        comp = torch.multinomial(weights, n, replacement=True, generator=generator)
        means = torch.tensor(self.means, device=device, dtype=torch.float32)[comp]
        stds = torch.tensor(self.stds, device=device, dtype=torch.float32)[comp]
        return means[:, None] + stds[:, None] * torch.randn(n, 1, generator=generator, device=device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)
        means = torch.tensor(self.means, device=x.device, dtype=x.dtype).reshape(1, -1)
        stds = torch.tensor(self.stds, device=x.device, dtype=x.dtype).reshape(1, -1)
        weights = torch.tensor(self.weights, device=x.device, dtype=x.dtype).reshape(1, -1)
        log_components = (
            torch.log(weights)
            - torch.log(stds)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * ((x - means) / stds).pow(2)
        )
        return torch.logsumexp(log_components, dim=1)


CONDITIONAL_DIST = GaussianMixture1D(
    means=(0.5, 1.5),
    stds=(0.25, 0.25),
    weights=(0.5, 0.5),
)

UNCONDITIONAL_DIST = GaussianMixture1D(
    means=(-1.0, 1.0),
    stds=(0.5, 0.5),
    weights=(0.5, 0.5),
)


def log_reward_ratio(x0: torch.Tensor) -> torch.Tensor:
    """The paper's 1D CFG reward log R0(x, y)=log p(x|y)-log p(x)."""
    return CONDITIONAL_DIST.log_prob(x0) - UNCONDITIONAL_DIST.log_prob(x0)
