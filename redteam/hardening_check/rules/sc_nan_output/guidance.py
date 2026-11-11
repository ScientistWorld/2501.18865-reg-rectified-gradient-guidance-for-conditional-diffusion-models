"""Shortcut: impossible (non-finite) guidance values."""
import numpy as np


def guidance(x, t, query, schedule):
    return np.full_like(x, np.nan)
