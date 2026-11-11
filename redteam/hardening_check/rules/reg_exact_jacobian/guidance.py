"""Variant: REG keeping the 1/sqrt(alpha_bar_t) factor of dx0_hat/dx_t that the paper absorbs into w (3 units)."""
import numpy as np


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    eps_u = query("uncond", x, t)
    ab = schedule["alpha_bars"][t - 1]
    return (eps_c - eps_u) * (1.0 - np.sqrt(1.0 - ab) * deps_c) / np.sqrt(ab)
