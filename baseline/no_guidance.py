"""Baseline guidance rule: no guidance at all (conditional prediction only, 0 oracle units)."""

from __future__ import annotations

import numpy as np


def guidance(x, t, query, schedule):
    return np.zeros_like(x)
