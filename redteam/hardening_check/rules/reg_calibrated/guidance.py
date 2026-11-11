"""Better-variant candidate: REG times one scalar fitted on the public training problems (3 units)."""
import json
import os

import numpy as np

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants.json")) as f:
    SCALE = float(json.load(f)["scale"])


def guidance(x, t, query, schedule):
    eps_c, deps_c = query("cond", x, t, deriv=True)
    eps_u = query("uncond", x, t)
    s = np.sqrt(1.0 - schedule["alpha_bars"][t - 1])
    return SCALE * (eps_c - eps_u) * (1.0 - s * deps_c)
