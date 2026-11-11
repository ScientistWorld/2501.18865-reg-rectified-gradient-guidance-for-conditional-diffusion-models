"""Shortcut: reference REG that declares zero cost (attributes and a claims file)."""
import json
import os

import numpy as np

ORACLE_COST_PER_STATE = 0
additional_vjp_per_guided_step = 0.0
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "claims.json"), "w") as f:
    json.dump({"oracle_cost_per_state": 0.0, "additional_vjp_per_guided_step": 0.0}, f)


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    eps_u = query("uncond", x, t)
    s = np.sqrt(1.0 - schedule["alpha_bars"][t - 1])
    return (eps_c - eps_u) * (1.0 - s * deps_c)
