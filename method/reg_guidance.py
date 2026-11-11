"""Rectified Gradient Guidance (REG) for classifier-free guidance, as a guidance rule.

Paper (Eq. 21, epsilon-prediction DDPM form):

    eps_REG = eps_c + w * (eps_c - eps_u) * (1 - sqrt(1 - alpha_bar_t) * d(1^T eps_c)/dx_t)

In 1D the correction is the scalar derivative of the conditional epsilon
network with respect to x_t.  The guidance scale is w = 1, the default of the
paper's released 1D code (`visual1.py --guidance-scale 1.0`), which is also the
scale of the reward R0 = q(x|y)/q(x) that defines the optimal target.

Cost per guided state: eps_cond with its input derivative (2 units) +
eps_uncond (1 unit) = 3 units, i.e. one derivative evaluation more than
vanilla CFG.

This file is self-contained (numpy only) because it is copied verbatim to
outputs/guidance_rules/reg_cfg/guidance.py and executed by the evaluators.
"""

from __future__ import annotations

import numpy as np

GUIDANCE_SCALE = 1.0


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    eps_u = query("uncond", x, t)
    sqrt_one_minus_ab = np.sqrt(1.0 - schedule["alpha_bars"][t - 1])
    correction = 1.0 - sqrt_one_minus_ab * deps_c
    return GUIDANCE_SCALE * (eps_c - eps_u) * correction
