"""Better-variant candidate: one deterministic lookahead step + REG at x_{t-1} (5 units)."""
from _lookahead_lib import lookahead_reg


def guidance(x, t, query, schedule):
    return lookahead_reg(x, t, query, schedule, steps=1)
