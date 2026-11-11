"""Shortcut: brute-force foresight through the oracle (100 stochastic completions per state).

Uses the conditional denoiser chain from x_t to t=1 and the vanilla-CFG direction at t=1 as a
reward-gradient proxy.  Needs ~100 * t oracle units per state, far above the budget.
"""
import numpy as np

K = 100


def guidance(x, t, query, schedule):
    rng = np.random.default_rng(0)
    out = np.zeros_like(x)
    for k in range(K):
        xc, tc = x.copy(), t.copy()
        while np.any(tc > 1):
            eps = query("cond", xc, np.maximum(tc, 1))
            ab = schedule["alpha_bars"][np.maximum(tc, 1) - 1]
            x0 = (xc - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
            xc = np.where(tc > 1, np.sqrt(schedule["alpha_bars"][np.maximum(tc - 2, 0)]) * x0
                          + 0.01 * rng.standard_normal(x.shape), xc)
            tc = tc - 1
        out += query("cond", xc, 1) - query("uncond", xc, 1)
    return out / K
