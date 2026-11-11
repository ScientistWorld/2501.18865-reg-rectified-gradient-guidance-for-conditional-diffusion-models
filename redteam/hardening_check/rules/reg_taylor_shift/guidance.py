"""Better-variant candidate: REG with a first-order Taylor shift of the CFG direction from x_t to x0_hat (4 units)."""
import numpy as np


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    eps_u, deps_u = query("uncond", x, t, deriv=True)
    ab = schedule["alpha_bars"][t - 1]
    s = np.sqrt(1.0 - ab)
    x0 = (x - s * eps_c) / np.sqrt(ab)
    cfg = (eps_c - eps_u) + (x0 - x) * (deps_c - deps_u)
    return (1.0 - s * deps_c) * cfg
