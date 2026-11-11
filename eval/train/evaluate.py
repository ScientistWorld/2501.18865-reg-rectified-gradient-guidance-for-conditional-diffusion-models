"""Visible training-slice evaluator for the 1D guidance benchmark.

Scores every guidance rule found under the rules root (default
`outputs/guidance_rules/<method>/guidance.py`).  Each rule runs in its own
isolated subprocess (eval/train/rule_runner.py) and can reach the frozen
denoisers only through counted oracle queries answered here.  All scored
numbers are recomputed from the raw guidance values the rule returns:

  guidance_mae             mean |g - g*| over all evaluation states (per-state error clipped)
  win_rate                 fraction of states where the rule is closer to g* than vanilla CFG
                           (ties count 1/2); vanilla CFG is computed here, not by the method
  worst_timestep_win_rate  minimum over time steps of the per-step win rate
  worst_problem_mae        maximum over problems of the per-problem MAE
  oracle_cost_per_state    denoiser evaluation units per guided state (forward = 1, +1 for a derivative)

A rule fails (and gets the worst value of every metric) if it cannot be
imported, crashes, times out, returns a wrong shape or non-finite values,
exceeds the per-state oracle budget, or returns different values for the same
state when the states are batched differently (rules must be per-state).

This file never imports from method/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.toy_1d import load_denoiser  # noqa: E402

EXPERIMENT = "toy_1d_guidance_approximation"
SLICE = "train"
PROBLEM_DIR = ROOT / "data" / "toy_1d_train"
RUNNER = Path(__file__).resolve().parent / "rule_runner.py"

BUDGET_UNITS = 8
ERROR_CLIP = 4.0
TIE_TOL = 1e-9
CONSISTENCY_ABS = 1e-5
CONSISTENCY_REL = 1e-5
PASSES = ((1234, 256), (98765, 199))  # (shuffle seed, batch size) for the two batch compositions
MESSAGE_TIMEOUT_S = 600.0
RULE_TIMEOUT_S = 3600.0

WORST = {
    "guidance_mae": ERROR_CLIP,
    "win_rate": 0.0,
    "worst_timestep_win_rate": 0.0,
    "worst_problem_mae": ERROR_CLIP,
    "oracle_cost_per_state": float(BUDGET_UNITS),
}


class RuleFailure(Exception):
    pass


# --------------------------------------------------------------------------
# Problems and oracle
# --------------------------------------------------------------------------


class Problems:
    def __init__(self, problem_dir: Path, only: list[str] | None):
        index = json.loads((problem_dir / "index.json").read_text())
        names = index["problems"] if not only else [n for n in index["problems"] if n in only]
        if not names:
            raise ValueError(f"no problems selected from {problem_dir}")
        self.names = names
        self.schedule = index["schedule"]
        self.models = []
        pid, xs, ts, tg = [], [], [], []
        for i, name in enumerate(names):
            rec = json.loads((problem_dir / f"{name}.json").read_text())
            cond = load_denoiser(str(ROOT / rec["cond_ckpt"]), 0, "cpu", torch.float64)
            uncond = load_denoiser(str(ROOT / rec["uncond_ckpt"]), -1, "cpu", torch.float64)
            self.models.append((cond, uncond))
            n = len(rec["target"])
            pid += [i] * n
            xs += rec["states"]["x"]
            ts += rec["states"]["t"]
            tg += rec["target"]
        self.pid = np.asarray(pid, dtype=np.int64)
        self.x = np.asarray(xs, dtype=np.float64)
        self.t = np.asarray(ts, dtype=np.int64)
        self.target = np.asarray(tg, dtype=np.float64)
        if not np.all(np.isfinite(self.target)):
            raise ValueError("non-finite optimal target in problem files")
        self.T = int(self.schedule["T"])
        self.cfg = self.vanilla_cfg()

    def eval_models(self, pid: np.ndarray, kind: str, x: np.ndarray, t: np.ndarray, deriv: bool):
        eps = np.empty_like(x)
        deps = np.empty_like(x) if deriv else None
        for p in np.unique(pid):
            rows = np.nonzero(pid == p)[0]
            model = self.models[p][0 if kind == "cond" else 1]
            xt = torch.tensor(x[rows], dtype=torch.float64)
            tt = torch.tensor(t[rows], dtype=torch.long)
            if deriv:
                xt.requires_grad_(True)
                with torch.enable_grad():
                    out = model(xt, tt)
                    (grad,) = torch.autograd.grad(out.sum(), xt)
                deps[rows] = grad.detach().numpy()
                eps[rows] = out.detach().numpy()
            else:
                with torch.no_grad():
                    eps[rows] = model(xt, tt).numpy()
        return eps, deps

    def vanilla_cfg(self) -> np.ndarray:
        """Fixed comparison direction: classifier-free guidance at scale 1, eps_c - eps_u."""
        e_c, _ = self.eval_models(self.pid, "cond", self.x, self.t, False)
        e_u, _ = self.eval_models(self.pid, "uncond", self.x, self.t, False)
        return e_c - e_u


# --------------------------------------------------------------------------
# Running one rule
# --------------------------------------------------------------------------


def _send(proc: subprocess.Popen, obj: dict) -> None:
    try:
        proc.stdin.write((json.dumps(obj) + "\n").encode())
        proc.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        raise RuleFailure(f"rule process closed its input ({exc})")


def _recv(proc: subprocess.Popen, deadline: float) -> dict:
    timeout = min(MESSAGE_TIMEOUT_S, deadline - time.time())
    if timeout <= 0:
        raise RuleFailure("rule exceeded its time limit")
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise RuleFailure("rule timed out")
    line = proc.stdout.readline()
    if not line:
        raise RuleFailure(f"rule process exited (code {proc.poll()})")
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuleFailure(f"malformed protocol message ({exc})")
    if not isinstance(msg, dict):
        raise RuleFailure("malformed protocol message")
    return msg


def _as_vector(values, n: int, what: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise RuleFailure(f"{what} is not numeric")
    if arr.shape != (n,):
        raise RuleFailure(f"{what} has shape {arr.shape}, expected ({n},)")
    if not np.all(np.isfinite(arr)):
        raise RuleFailure(f"{what} contains non-finite values")
    return arr


def run_pass(rule_file: Path, probs: Problems, seed: int, batch_size: int, log) -> tuple[np.ndarray, np.ndarray]:
    n = probs.x.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    g = np.full(n, np.nan)
    units = np.zeros(n)
    proc = subprocess.Popen(
        [sys.executable, "-I", str(RUNNER), str(rule_file)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        cwd=str(rule_file.parent),
    )
    deadline = time.time() + RULE_TIMEOUT_S
    try:
        init = {k: probs.schedule[k] for k in ("betas", "alphas", "alpha_bars", "posterior_variance")}
        init["T"] = probs.T
        _send(proc, init)
        msg = _recv(proc, deadline)
        if msg.get("op") != "ready":
            raise RuleFailure(str(msg.get("msg", "rule did not start"))[-1500:])
        for start in range(0, n, batch_size):
            rows = order[start : start + batch_size]
            b = rows.shape[0]
            _send(proc, {"op": "batch", "x": probs.x[rows].tolist(), "t": probs.t[rows].tolist()})
            used = 0
            while True:
                msg = _recv(proc, deadline)
                op = msg.get("op")
                if op == "query":
                    kind = msg.get("model")
                    if kind not in ("cond", "uncond"):
                        raise RuleFailure(f"unknown model {kind!r}")
                    deriv = bool(msg.get("deriv", False))
                    used += 2 if deriv else 1
                    if used > BUDGET_UNITS:
                        raise RuleFailure(f"oracle budget exceeded ({used} > {BUDGET_UNITS} units per state)")
                    xq = _as_vector(msg.get("x"), b, "query x")
                    tq = _as_vector(msg.get("t"), b, "query t")
                    if not np.all(tq == np.round(tq)) or tq.min() < 1 or tq.max() > probs.T:
                        raise RuleFailure("query t outside 1..T")
                    eps, deps = probs.eval_models(probs.pid[rows], kind, xq, tq.astype(np.int64), deriv)
                    ans = {"op": "answer", "eps": eps.tolist()}
                    if deriv:
                        ans["deps"] = deps.tolist()
                    _send(proc, ans)
                elif op == "result":
                    g[rows] = _as_vector(msg.get("g"), b, "guidance output")
                    units[rows] = used
                    break
                elif op == "error":
                    raise RuleFailure("rule raised:\n" + str(msg.get("msg", ""))[-1500:])
                else:
                    raise RuleFailure(f"unexpected protocol op {op!r}")
        _send(proc, {"op": "exit"})
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return g, units


def score(g: np.ndarray, units: np.ndarray, probs: Problems) -> dict:
    err = np.minimum(np.abs(g - probs.target), ERROR_CLIP)
    err_cfg = np.minimum(np.abs(probs.cfg - probs.target), ERROR_CLIP)
    diff = err - err_cfg
    win = np.where(diff < -TIE_TOL, 1.0, np.where(diff > TIE_TOL, 0.0, 0.5))
    per_t = [float(win[probs.t == t].mean()) for t in np.unique(probs.t)]
    per_p = [float(err[probs.pid == p].mean()) for p in np.unique(probs.pid)]
    return {
        "guidance_mae": float(err.mean()),
        "win_rate": float(win.mean()),
        "worst_timestep_win_rate": float(min(per_t)),
        "worst_problem_mae": float(max(per_p)),
        "oracle_cost_per_state": float(units.mean()),
    }


def evaluate_rule(rule_dir: Path, probs: Problems, dump: Path | None = None) -> tuple[dict, str | None]:
    rule_file = rule_dir / "guidance.py"
    if not rule_file.is_file():
        return dict(WORST), "missing guidance.py"
    with tempfile.TemporaryFile() as log:
        try:
            outs = [run_pass(rule_file, probs, seed, bs, log) for seed, bs in PASSES]
            (g_a, u_a), (g_b, _) = outs
            if np.any(np.abs(g_a - g_b) > CONSISTENCY_ABS + CONSISTENCY_REL * np.abs(g_a)):
                worst = float(np.max(np.abs(g_a - g_b)))
                raise RuleFailure(f"output depends on batch composition (max diff {worst:.3g}); rules must be per-state")
            if dump is not None:
                np.save(dump / f"{rule_dir.name}.npy", g_a)
            return score(g_a, u_a, probs), None
        except Exception as exc:  # noqa: BLE001 - any failure while running the rule fails the rule
            if not isinstance(exc, RuleFailure):
                exc = RuleFailure(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - 1500))
            tail = log.read().decode(errors="replace")
            return dict(WORST), f"{exc}" + (f"\n[stderr tail]\n{tail}" if tail.strip() else "")


def discover(rules_root: Path) -> list[Path]:
    if not rules_root.is_dir():
        return []
    return sorted(p for p in rules_root.iterdir() if p.is_dir() and p.name != "__pycache__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-root", default=str(ROOT / "outputs" / "guidance_rules"))
    ap.add_argument("--scores", default=str(ROOT / "scoring" / "scores_train.json"))
    ap.add_argument("--problems", nargs="*", default=None, help="subset of problem names (default: all)")
    ap.add_argument("--slice-name", default=SLICE)
    args = ap.parse_args()

    out = Path(args.scores)
    if out.exists():
        out.unlink()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    rule_dirs = discover(Path(args.rules_root))
    if not rule_dirs:
        print(f"No guidance rules found under {args.rules_root}", file=sys.stderr)
        sys.exit(1)
    probs = Problems(PROBLEM_DIR, args.problems)
    dump = Path(os.environ["REG_GYM_DUMP_DIR"]) if os.environ.get("REG_GYM_DUMP_DIR") else None
    if dump is not None:  # analysis only: per-state outputs of passing rules
        dump.mkdir(parents=True, exist_ok=True)
        np.savez(dump / "meta.npz", pid=probs.pid, t=probs.t, target=probs.target, cfg=probs.cfg)
    results, failures = {}, {}
    for rd in rule_dirs:
        t0 = time.time()
        metrics, reason = evaluate_rule(rd, probs, dump)
        results[rd.name] = metrics
        if reason:
            failures[rd.name] = reason
        status = "FAILED: " + reason.splitlines()[0] if reason else "ok"
        print(f"[{args.slice_name}] {rd.name}: {status} ({time.time() - t0:.1f}s) {json.dumps(metrics)}", flush=True)
    scores = {
        "_metadata": {
            "slice": args.slice_name,
            "problems": probs.names,
            "num_states": int(probs.x.shape[0]),
            "oracle_budget_units_per_state": BUDGET_UNITS,
            "error_clip": ERROR_CLIP,
            "failures": failures,
        },
        "experiments": {EXPERIMENT: {"_metadata": {"slice": args.slice_name}, "results": results}},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
