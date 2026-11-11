"""Shortcut: pools information across the states of a batch (output depends on batch partners)."""
import numpy as np


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    eps_u = query("uncond", x, t)
    s = np.sqrt(1.0 - schedule["alpha_bars"][t - 1])
    g = (eps_c - eps_u) * (1.0 - s * deps_c)
    return g * (1.0 - 0.2 * np.tanh(np.mean(np.abs(g))))
