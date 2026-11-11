"""Better-variant candidate: REG Jacobian at x_t, guidance direction evaluated at the Tweedie point x0_hat (4 units).

The paper approximates grad log R0(x0_hat) by grad log R_t(x_t); this variant evaluates the CFG direction
at x0_hat instead of x_t (same noise level t)."""
import numpy as np


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    ab = schedule["alpha_bars"][t - 1]
    s = np.sqrt(1.0 - ab)
    x0 = (x - s * eps_c) / np.sqrt(ab)
    cfg_at_x0 = query("cond", x0, t) - query("uncond", x0, t)
    return (1.0 - s * deps_c) * cfg_at_x0
