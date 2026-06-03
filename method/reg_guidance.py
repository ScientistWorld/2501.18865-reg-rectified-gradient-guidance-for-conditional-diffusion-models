"""Rectified Gradient Guidance primitives.

This module implements the paper's epsilon-prediction REG update for CFG:

    eps_reg = eps_cond + w * (eps_cond - eps_uncond)
              * (1 - sqrt(1 - alpha_bar_t) * d(sum eps_cond) / dx_t)

The same correction can be reused by future methods that emit conditional and
unconditional epsilon predictions in the output format used by this benchmark.
"""

from __future__ import annotations

import torch


def cfg_adjustment(eps_cond: torch.Tensor, eps_uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    return guidance_scale * (eps_cond - eps_uncond)


def reg_correction(
    x_t: torch.Tensor,
    eps_cond: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor | float,
) -> torch.Tensor:
    grad = torch.autograd.grad(
        eps_cond.sum(),
        x_t,
        create_graph=False,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return 1.0 - sqrt_one_minus_alpha_bar * grad


def reg_adjustment(
    x_t: torch.Tensor,
    eps_cond: torch.Tensor,
    eps_uncond: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor | float,
    guidance_scale: float,
) -> torch.Tensor:
    return cfg_adjustment(eps_cond, eps_uncond, guidance_scale) * reg_correction(
        x_t, eps_cond, sqrt_one_minus_alpha_bar
    )
