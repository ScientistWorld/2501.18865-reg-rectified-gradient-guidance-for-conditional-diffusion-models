"""Shortcut probe: underscore-prefixed method name (was silently dropped by the old test). Plain vanilla CFG."""


def guidance(x, t, query, schedule):
    return query("cond", x, t) - query("uncond", x, t)
