"""Build the public (visible) training problems of the 1D guidance benchmark.

Each problem = conditional/unconditional data mixtures + two frozen denoisers +
a set of evaluation states (x_t, t) + the optimal foresight guidance target at
those states.  Everything written here is public: methods may read it.

Problems:
  paper_mixture     the paper's 1D mixtures, denoisers trained with the paper's recipe
  paper_checkpoint  the paper's mixtures with the authors' released denoisers
                    (fetched by scripts/download.sh into data/downloads/paper_ckpt/)
  family_1..3       draws from the public paper-like family (data/toy_1d.py)

Resumable: finished checkpoints and problem files are skipped.

Usage: python data/build_train_problems.py [--completions 1000] [--states-per-step 160]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.toy_1d import (  # noqa: E402
    GRID_HIGH,
    GRID_LOW,
    PAPER_COND,
    PAPER_UNCOND,
    T_STEPS,
    Schedule,
    foresight_target,
    load_denoiser,
    sample_family_problem,
    train_denoiser,
)

OUT = ROOT / "data" / "toy_1d_train"
PAPER_CKPT_DIR = ROOT / "data" / "downloads" / "paper_ckpt"
FAMILY_SEED = 2025
STATE_SEED = 31337


def problem_specs() -> list[dict]:
    specs = [
        {"name": "paper_mixture", "cond": PAPER_COND, "uncond": PAPER_UNCOND, "seeds": (0, 1), "ckpt": None},
        {
            "name": "paper_checkpoint",
            "cond": PAPER_COND,
            "uncond": PAPER_UNCOND,
            "seeds": None,
            "ckpt": (str(PAPER_CKPT_DIR / "cond.pth"), str(PAPER_CKPT_DIR / "uncond.pth")),
        },
    ]
    rng = random.Random(FAMILY_SEED)
    for i in range(1, 4):
        cond, uncond = sample_family_problem(rng)
        specs.append({"name": f"family_{i}", "cond": cond, "uncond": uncond, "seeds": (100 + 2 * i, 101 + 2 * i), "ckpt": None})
    return specs


def rel(p: Path | str) -> str:
    return os.path.relpath(str(p), str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=125)
    ap.add_argument("--states-per-step", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sched = Schedule(device)
    names = []
    for pi, spec in enumerate(problem_specs()):
        name = spec["name"]
        names.append(name)
        pfile = out / f"{name}.json"
        if pfile.is_file():
            print(f"[train-build] {name}: exists, skipping", flush=True)
            continue
        t0 = time.time()
        if spec["ckpt"] is None:
            cpaths = (out / f"{name}_cond.pt", out / f"{name}_uncond.pt")
            for path, mix, label, seed in (
                (cpaths[0], spec["cond"], 0, spec["seeds"][0]),
                (cpaths[1], spec["uncond"], -1, spec["seeds"][1]),
            ):
                if not path.is_file():
                    net = train_denoiser(mix, label, seed, device, epochs=args.epochs)
                    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, path)
                    print(f"[train-build] trained {path.name} in {time.time() - t0:.1f}s", flush=True)
        else:
            cpaths = tuple(Path(p) for p in spec["ckpt"])
            for p in cpaths:
                if not p.is_file():
                    raise FileNotFoundError(f"{p} missing: run scripts/download.sh first")
        cond_model = load_denoiser(str(cpaths[0]), 0, device)

        srng = torch.Generator().manual_seed(STATE_SEED + 1000 * pi)
        xs, ts, targets = [], [], []
        for t in range(1, T_STEPS + 1):
            x = GRID_LOW + (GRID_HIGH - GRID_LOW) * torch.rand(args.states_per_step, generator=srng, dtype=torch.float64)
            g = foresight_target(
                cond_model, spec["cond"], spec["uncond"], x.float().to(device), t,
                args.completions, args.chunk, seed=STATE_SEED + 1000 * pi + t, sched=sched,
            )
            xs += x.tolist()
            ts += [t] * args.states_per_step
            targets += g.double().cpu().tolist()
        record = {
            "name": name,
            "cond": spec["cond"].to_dict(),
            "uncond": spec["uncond"].to_dict(),
            "cond_ckpt": rel(cpaths[0]),
            "uncond_ckpt": rel(cpaths[1]),
            "target_completions": args.completions,
            "states": {"x": xs, "t": ts},
            "target": targets,
        }
        pfile.write_text(json.dumps(record))
        print(f"[train-build] {name}: done in {time.time() - t0:.1f}s", flush=True)

    index = {
        "description": "Public training problems for the 1D guidance benchmark (see data/build_train_problems.py).",
        "schedule": Schedule("cpu", torch.float64).as_lists(),
        "problems": names,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
