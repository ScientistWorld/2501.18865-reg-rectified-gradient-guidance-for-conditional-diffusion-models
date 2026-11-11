"""Fit the single REG multiplier of the `reg_calibrated` variant on the PUBLIC training problems only.

Writes rules/reg_calibrated/constants.json and rules/reg_calibrated_per_t/constants.json.  Uses public data (data/toy_1d_train) and public code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.toy_1d import Schedule, load_denoiser  # noqa: E402

PDIR = ROOT / "data" / "toy_1d_train"


def main() -> None:
    index = json.loads((PDIR / "index.json").read_text())
    sched = Schedule("cpu", torch.float64)
    regs, targets = [], []
    for name in index["problems"]:
        rec = json.loads((PDIR / f"{name}.json").read_text())
        cond = load_denoiser(str(ROOT / rec["cond_ckpt"]), 0, "cpu", torch.float64)
        uncond = load_denoiser(str(ROOT / rec["uncond_ckpt"]), -1, "cpu", torch.float64)
        x = torch.tensor(rec["states"]["x"], dtype=torch.float64, requires_grad=True)
        t = torch.tensor(rec["states"]["t"], dtype=torch.long)
        with torch.enable_grad():
            e_c = cond(x, t)
            (d_c,) = torch.autograd.grad(e_c.sum(), x)
        with torch.no_grad():
            e_u = uncond(x.detach(), t)
        s = torch.sqrt(1.0 - sched.alpha_bars[t - 1])
        regs.append(((e_c.detach() - e_u) * (1.0 - s * d_c)).numpy())
        targets.append(np.asarray(rec["target"]))
    reg = np.concatenate(regs)
    tgt = np.concatenate(targets)
    tt = np.concatenate([np.asarray(json.loads((PDIR / f"{n}.json").read_text())["states"]["t"]) for n in index["problems"]])
    grid = np.round(np.arange(0.2, 2.0001, 0.005), 3)

    def mae(c, m):
        return float(np.mean(np.minimum(np.abs(c * reg[m] - tgt[m]), 4.0)))

    allm = np.ones_like(tgt, dtype=bool)
    maes = [mae(c, allm) for c in grid]
    best = float(grid[int(np.argmin(maes))])
    rules = Path(__file__).resolve().parent / "rules"
    (rules / "reg_calibrated" / "constants.json").write_text(json.dumps({"scale": best, "train_mae_at_scale": min(maes), "train_mae_at_1": mae(1.0, allm)}, indent=2))
    per_t = []
    for step in range(1, int(index["schedule"]["T"]) + 1):
        m = tt == step
        per_t.append(float(grid[int(np.argmin([mae(c, m) for c in grid]))]))
    (rules / "reg_calibrated_per_t" / "constants.json").write_text(json.dumps({"scales": per_t}, indent=2))
    print(f"[calibration] scalar {best} (train MAE {min(maes):.5f}; at 1.0: {mae(1.0, allm):.5f}); per-t {per_t}")


if __name__ == "__main__":
    main()
