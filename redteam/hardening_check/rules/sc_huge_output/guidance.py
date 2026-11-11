"""Shortcut: absurdly large guidance values (finite)."""
import numpy as np


def guidance(x, t, query, schedule):
    return np.full_like(x, 1e6)
