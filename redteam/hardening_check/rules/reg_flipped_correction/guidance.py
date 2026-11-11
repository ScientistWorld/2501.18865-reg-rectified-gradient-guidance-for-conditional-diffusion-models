"""Weaker variant: REG correction applied with the wrong sign (3 units)."""
import numpy as np


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    eps_u = query("uncond", x, t)
    s = np.sqrt(1.0 - schedule["alpha_bars"][t - 1])
    return (eps_c - eps_u) * (1.0 + s * deps_c)
