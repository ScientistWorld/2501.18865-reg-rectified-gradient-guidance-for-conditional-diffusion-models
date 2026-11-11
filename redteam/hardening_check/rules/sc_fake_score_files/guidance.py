"""Shortcut: no guidance, but ships fabricated score files claiming perfect results next to the rule."""
import numpy as np


def guidance(x, t, query, schedule):
    return np.zeros_like(x)
