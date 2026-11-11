"""Sandbox-side runner for a guidance rule (hidden test slice).

Launched by eval/test/evaluate.py as `python -I rule_runner.py <rule.py>`.
The rule never holds the denoisers: every denoiser evaluation is a counted
request sent to the evaluator process over a pipe.  The rule's own prints go to
stderr so they cannot corrupt the protocol.

Rule contract (file `guidance.py` in the method's rule directory):

    def guidance(x, t, query, schedule) -> array of shape (B,)

    x        float64 array (B,), noisy states x_t
    t        int64 array (B,), 1-indexed time steps (1..T)
    query    query(model, xq, tq, deriv=False)
               model: "cond" or "uncond"; xq: (B,) float array; tq: int or (B,) ints in 1..T
               returns eps (B,), or (eps, d eps / dx) when deriv=True
               cost: 1 unit per state, +1 when deriv=True; the per-state budget is fixed by the evaluator
    schedule dict with "T" and float64 arrays "betas", "alphas", "alpha_bars",
             "posterior_variance" (index t-1)

The returned g is added to the conditional prediction: eps_guided = eps_cond(x_t, t) + g.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback


def main() -> None:
    rule_path = os.path.abspath(sys.argv[1])
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_fd, 0)
    os.close(null_fd)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull)
    sys.stdout = sys.stderr

    def send(obj: dict) -> None:
        proto_out.write((json.dumps(obj) + "\n").encode())
        proto_out.flush()

    def recv() -> dict:
        line = proto_in.readline()
        if not line:
            raise SystemExit(1)
        return json.loads(line)

    import numpy as np

    init = recv()
    schedule_src = {k: np.asarray(init[k], dtype=np.float64) for k in ("betas", "alphas", "alpha_bars", "posterior_variance")}
    T = int(init["T"])

    try:
        sys.path.insert(0, os.path.dirname(rule_path))
        spec = importlib.util.spec_from_file_location("guidance_rule", rule_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rule = getattr(module, "guidance")
    except BaseException:  # noqa: BLE001 - any import failure fails the rule
        send({"op": "error", "msg": "import failed:\n" + traceback.format_exc()[-3000:]})
        return
    send({"op": "ready"})

    while True:
        msg = recv()
        op = msg.get("op")
        if op == "exit":
            return
        if op != "batch":
            send({"op": "error", "msg": f"unexpected op {op!r}"})
            return
        x = np.asarray(msg["x"], dtype=np.float64)
        t = np.asarray(msg["t"], dtype=np.int64)
        batch = x.shape[0]

        def query(model, xq, tq, deriv=False, _b=batch):
            xq = np.asarray(xq, dtype=np.float64).reshape(-1)
            if xq.shape != (_b,):
                raise ValueError(f"query x must have shape ({_b},), got {xq.shape}")
            tq = np.asarray(tq)
            if tq.ndim == 0:
                tq = np.full((_b,), tq)
            tq = tq.reshape(-1)
            if tq.shape != (_b,):
                raise ValueError(f"query t must be scalar or shape ({_b},)")
            if not np.all(np.mod(tq, 1) == 0):
                raise ValueError("query t must be integers")
            send({"op": "query", "model": str(model), "x": xq.tolist(), "t": [int(v) for v in tq], "deriv": bool(deriv)})
            ans = recv()
            if ans.get("op") != "answer":
                raise RuntimeError(f"oracle refused query: {ans.get('msg')}")
            eps = np.asarray(ans["eps"], dtype=np.float64)
            if deriv:
                return eps, np.asarray(ans["deps"], dtype=np.float64)
            return eps

        schedule = {k: v.copy() for k, v in schedule_src.items()}
        schedule["T"] = T
        try:
            g = rule(x.copy(), t.copy(), query, schedule)
            g = np.asarray(g, dtype=np.float64).reshape(-1)
            send({"op": "result", "g": g.tolist()})
        except BaseException:  # noqa: BLE001
            send({"op": "error", "msg": traceback.format_exc()[-3000:]})
            return


if __name__ == "__main__":
    main()
