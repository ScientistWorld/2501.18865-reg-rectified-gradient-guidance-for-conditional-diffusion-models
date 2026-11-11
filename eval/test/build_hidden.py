"""Build the hidden test problems (run once; the outputs are committed in eval/test/assets/).

Hidden problems are drawn from a paper-like family of 1D Gaussian mixtures with
a seed that exists only here.  For each problem we train the conditional and
unconditional denoisers with the paper's recipe, draw hidden evaluation states
(x_t uniform in [-1, 2], every t = 1..20), and estimate the optimal foresight
guidance target with 1,000 stochastic denoising completions per state.

Resumable: finished checkpoints and problem files are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.test.toy_problem import (  # noqa: E402
    GRID_HIGH,
    GRID_LOW,
    T_STEPS,
    Schedule,
    foresight_target,
    load_denoiser,
    sample_hidden_problem,
    train_denoiser,
)

ASSETS = Path(__file__).resolve().parent / "assets"
HIDDEN_SEED = 860_517_293
NUM_PROBLEMS = 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=125)
    ap.add_argument("--states-per-step", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--num-problems", type=int, default=NUM_PROBLEMS)
    ap.add_argument("--out", default=str(ASSETS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched = Schedule(device)
    rng = random.Random(HIDDEN_SEED)
    names = []
    for pi in range(args.num_problems):
        cond_mix, uncond_mix = sample_hidden_problem(rng)
        seeds = (rng.randrange(10**9), rng.randrange(10**9))
        state_seed = rng.randrange(10**9)
        name = f"hidden_{pi:02d}"
        names.append(name)
        pfile = out / f"{name}.json"
        if pfile.is_file():
            print(f"[hidden-build] {name}: exists, skipping", flush=True)
            continue
        t0 = time.time()
        cpaths = (out / f"{name}_cond.pt", out / f"{name}_uncond.pt")
        for path, mix, label, seed in ((cpaths[0], cond_mix, 0, seeds[0]), (cpaths[1], uncond_mix, -1, seeds[1])):
            if not path.is_file():
                net = train_denoiser(mix, label, seed, device, epochs=args.epochs)
                torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, path)
                print(f"[hidden-build] trained {path.name} in {time.time() - t0:.1f}s", flush=True)
        cond_model = load_denoiser(str(cpaths[0]), 0, device)
        srng = torch.Generator().manual_seed(state_seed)
        xs, ts, targets = [], [], []
        for t in range(1, T_STEPS + 1):
            x = GRID_LOW + (GRID_HIGH - GRID_LOW) * torch.rand(args.states_per_step, generator=srng, dtype=torch.float64)
            g = foresight_target(
                cond_model, cond_mix, uncond_mix, x.float().to(device), t,
                args.completions, args.chunk, seed=state_seed + t, sched=sched,
            )
            xs += x.tolist()
            ts += [t] * args.states_per_step
            targets += g.double().cpu().tolist()
        record = {
            "name": name,
            "cond": cond_mix.to_dict(),
            "uncond": uncond_mix.to_dict(),
            "cond_ckpt": os.path.relpath(str(cpaths[0]), str(ROOT)),
            "uncond_ckpt": os.path.relpath(str(cpaths[1]), str(ROOT)),
            "target_completions": args.completions,
            "states": {"x": xs, "t": ts},
            "target": targets,
        }
        pfile.write_text(json.dumps(record))
        print(f"[hidden-build] {name}: done in {time.time() - t0:.1f}s", flush=True)

    index = {
        "description": "Hidden test problems (eval/test/build_hidden.py).",
        "schedule": Schedule("cpu", torch.float64).as_lists(),
        "problems": names,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
