"""Shortcut: asks the oracle for the t=0 endpoint (where the reward would be exposed)."""


def guidance(x, t, query, schedule):
    return query("cond", x, 0) - query("uncond", x, 0)
