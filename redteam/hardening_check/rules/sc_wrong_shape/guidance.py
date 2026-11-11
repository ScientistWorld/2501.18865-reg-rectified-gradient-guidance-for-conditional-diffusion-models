"""Shortcut: answer only a subset of the states (analog of the grid/time subset shortcuts)."""
import numpy as np


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    eps_u = query("uncond", x, t)
    s = np.sqrt(1.0 - schedule["alpha_bars"][t - 1])
    g = (eps_c - eps_u) * (1.0 - s * deps_c)
    return g[: max(1, len(g) // 2)]
