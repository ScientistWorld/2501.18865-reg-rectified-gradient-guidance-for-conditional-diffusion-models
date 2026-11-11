"""Fit the single REG multiplier of the `reg_calibrated` variant on the PUBLIC training problems only.

Writes rules/reg_calibrated/constants.json.  Uses public data (data/toy_1d_train) and public code.
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
    grid = np.round(np.arange(0.5, 1.5001, 0.005), 3)
    maes = [float(np.mean(np.minimum(np.abs(c * reg - tgt), 4.0))) for c in grid]
    best = float(grid[int(np.argmin(maes))])
    out = Path(__file__).resolve().parent / "rules" / "reg_calibrated" / "constants.json"
    out.write_text(json.dumps({"scale": best, "train_mae_at_scale": min(maes), "train_mae_at_1": maes[int(np.argmin(np.abs(grid - 1.0)))]}, indent=2))
    print(f"[calibration] best scale {best} (train MAE {min(maes):.5f}; at 1.0: {maes[int(np.argmin(np.abs(grid - 1.0)))]:.5f})")


if __name__ == "__main__":
    main()
