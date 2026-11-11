"""Better-variant candidate: two deterministic lookahead steps + REG at x_{t-2} (7 units)."""
from _lookahead_lib import lookahead_reg


def guidance(x, t, query, schedule):
    return lookahead_reg(x, t, query, schedule, steps=2)
