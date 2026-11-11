"""Baseline guidance rule: vanilla classifier-free guidance at scale 1 (2 oracle units per state).

    eps_guided = eps_c + w * (eps_c - eps_u),  w = 1
"""

from __future__ import annotations

GUIDANCE_SCALE = 1.0


def guidance(x, t, query, schedule):
    eps_c = query("cond", x, t)
    eps_u = query("uncond", x, t)
    return GUIDANCE_SCALE * (eps_c - eps_u)
