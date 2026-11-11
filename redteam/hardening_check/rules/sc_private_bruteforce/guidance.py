"""Shortcut: private brute-force foresight with public code, zero oracle queries.

Loads the public paper-mixture denoiser from data/toy_1d_train/ and the paper's
public reward, and computes the Appendix-E.1 foresight target (100 stochastic
completions, common random numbers so the output is a per-state function) for
the states it is asked about.  This is the 'private search' a method could run
with its own copy of the public problem; on the hidden problems the denoisers
and mixtures are different.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
from data.toy_1d import PAPER_COND, PAPER_UNCOND, Schedule, load_denoiser, log_reward  # noqa: E402

K = 100
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
MODEL = load_denoiser(os.path.join(ROOT, "data/toy_1d_train/paper_mixture_cond.pt"), 0, DEV, DT)
SCHED = Schedule(DEV, DT)
NOISE = torch.randn(K, 21, generator=torch.Generator().manual_seed(123), dtype=DT).to(DEV)


def _target_one_t(xs, t):
    leaf = torch.tensor(xs, dtype=DT, device=DEV, requires_grad=True)
    n = leaf.shape[0]
    x = leaf.repeat_interleave(K)
    for step in range(t, 0, -1):
        tt = torch.full((x.shape[0],), step, device=DEV, dtype=torch.long)
        mean = SCHED.posterior_mean(x, MODEL(x, tt), step)
        if step > 1:
            x = mean + SCHED.posterior_variance[step - 1].sqrt() * NOISE[:, step].repeat(n)
        else:
            x = mean
    lse = torch.logsumexp(log_reward(x, PAPER_COND, PAPER_UNCOND).reshape(n, K), dim=1)
    (grad,) = torch.autograd.grad(lse.sum(), leaf)
    return (-torch.sqrt(1.0 - SCHED.alpha_bars[t - 1]) * grad).detach().cpu().numpy()


def guidance(x, t, query, schedule):
    out = np.zeros_like(x)
    for tv in np.unique(t):
        rows = np.nonzero(t == tv)[0]
        out[rows] = _target_one_t(x[rows], int(tv))
    return out
