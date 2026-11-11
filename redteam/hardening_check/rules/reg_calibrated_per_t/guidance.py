"""Better-variant candidate: REG with a per-time-step multiplier fitted on the public training problems (3 units)."""
import json
import os

import numpy as np

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants.json")) as f:
    SCALES = np.asarray(json.load(f)["scales"], dtype=np.float64)  # index t-1


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    eps_u = query("uncond", x, t)
    s = np.sqrt(1.0 - schedule["alpha_bars"][t - 1])
    return SCALES[t - 1] * (eps_c - eps_u) * (1.0 - s * deps_c)
