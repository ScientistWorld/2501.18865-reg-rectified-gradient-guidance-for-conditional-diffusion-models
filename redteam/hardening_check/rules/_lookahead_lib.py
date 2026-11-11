"""Shared helper for the lookahead variants (copied into each rule dir by run_all.sh)."""
import numpy as np


def _s(schedule, tt):
    return np.sqrt(1.0 - schedule["alpha_bars"][tt - 1])


def posterior_mean_and_slope(x, eps, deps, tt, schedule):
    ab = schedule["alpha_bars"][tt - 1]
    ab_prev = np.where(tt > 1, schedule["alpha_bars"][np.maximum(tt - 2, 0)], 1.0)
    beta = schedule["betas"][tt - 1]
    alpha = schedule["alphas"][tt - 1]
    c1 = beta * np.sqrt(ab_prev) / (1.0 - ab)
    c2 = (1.0 - ab_prev) * np.sqrt(alpha) / (1.0 - ab)
    a, b = np.sqrt(1.0 / ab), np.sqrt(1.0 / ab - 1.0)
    mu = c1 * (a * x - b * eps) + c2 * x
    dmu = c1 * (a - b * deps) + c2
    return mu, dmu


def lookahead_reg(x, t, query, schedule, steps):
    """Deterministic k-step lookahead with chain-rule Jacobian, then REG at the end point."""
    xc, tc, jac = x.copy(), t.copy(), np.ones_like(x)
    for _ in range(steps):
        tq = np.maximum(tc, 1)
        eps, deps = query("cond", xc, tq, deriv=True)
        mu, dmu = posterior_mean_and_slope(xc, eps, deps, tq, schedule)
        live = tc >= 1
        xc = np.where(live, mu, xc)
        jac = np.where(live, jac * dmu, jac)
        tc = tc - 1
    tf = np.maximum(tc, 1)
    eps_c, deps_c = query("cond", xc, tf, deriv=True)
    eps_u = query("uncond", xc, tf)
    g_end = (eps_c - eps_u) * (1.0 - _s(schedule, tf) * deps_c)
    return _s(schedule, t) / _s(schedule, tf) * jac * g_end
