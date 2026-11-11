"""Summarize the hardening check: metric tables, ranks, paired bootstrap, effect vs. problem spread.

Inputs (written by run_all.sh):
  results/cohort_scores_test.json, results/cohort_scores_train.json
  $DUMP/meta.npz + $DUMP/<method>.npy   per-state outputs on the hidden slice (deleted afterwards)
Outputs: results/summary.json, results/SUMMARY.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
EXP = "toy_1d_guidance_approximation"
REF = "reg_cfg"
HONEST = [
    "reg_cfg", "vanilla_cfg", "no_guidance", "reg_calibrated", "reg_exact_jacobian",
    "reg_lookahead1", "reg_lookahead2", "reg_flipped_correction",
]


def load_scores(path: Path) -> tuple[dict, dict]:
    d = json.loads(path.read_text())
    return d["experiments"][EXP]["results"], d.get("_metadata", {}).get("failures", {})


def ranks(results: dict, metrics_def: dict, cohort: list[str]) -> tuple[dict, dict]:
    names = list(metrics_def)
    sig = {}
    for m in names:
        vals = np.array([results[c][m] for c in cohort if c in results])
        s = float(vals.std())
        sig[m] = s if s > 0 else 1.0
    out = {}
    ref = results[REF]
    for meth, vals in results.items():
        total = 0.0
        for m in names:
            if m not in vals:
                continue
            d = vals[m] - ref[m]
            if not metrics_def[m]["higher_is_better"]:
                d = -d
            total += abs(metrics_def[m]["coefficient"]) * d / sig[m]
        out[meth] = total
    return out, sig


def bootstrap(meta: dict, g_var: np.ndarray, g_ref: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    tgt, cfg, pid = meta["target"], meta["cfg"], meta["pid"]
    e_v = np.minimum(np.abs(g_var - tgt), 4.0)
    e_r = np.minimum(np.abs(g_ref - tgt), 4.0)
    e_c = np.minimum(np.abs(cfg - tgt), 4.0)
    w = lambda e: np.where(e - e_c < -1e-9, 1.0, np.where(e - e_c > 1e-9, 0.0, 0.5))  # noqa: E731
    w_v, w_r = w(e_v), w(e_r)
    probs = np.unique(pid)
    idx = {p: np.nonzero(pid == p)[0] for p in probs}
    rng = np.random.default_rng(seed)
    d_mae, d_win = [], []
    for _ in range(n_boot):
        sel = np.concatenate([rng.choice(idx[p], idx[p].shape[0]) for p in rng.choice(probs, probs.shape[0])])
        d_mae.append(e_v[sel].mean() - e_r[sel].mean())
        d_win.append(w_v[sel].mean() - w_r[sel].mean())
    per_problem = [float(e_v[idx[p]].mean() - e_r[idx[p]].mean()) for p in probs]
    return {
        "delta_mae": float(e_v.mean() - e_r.mean()),
        "delta_mae_ci95": [float(np.percentile(d_mae, 2.5)), float(np.percentile(d_mae, 97.5))],
        "delta_win": float(w_v.mean() - w_r.mean()),
        "delta_win_ci95": [float(np.percentile(d_win, 2.5)), float(np.percentile(d_win, 97.5))],
        "per_problem_delta_mae": per_problem,
        "problems_improved": int(sum(v < 0 for v in per_problem)),
        "states_closer_than_reference": float(np.mean(e_v < e_r - 1e-9)),
    }


def fmt(v: float) -> str:
    return f"{v:.4f}"


def main() -> None:
    dump = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    ref_def = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"][EXP]["metrics"]
    test, test_fail = load_scores(RES / "cohort_scores_test.json")
    train, train_fail = load_scores(RES / "cohort_scores_train.json")
    cohort = [c for c in HONEST if c in test]
    r_test, sig_test = ranks(test, ref_def, cohort)
    r_train, _ = ranks(train, ref_def, [c for c in HONEST if c in train])
    summary = {
        "rank_cohort": cohort,
        "rank_sigma_test": sig_test,
        "test": test,
        "train": train,
        "test_rank": r_test,
        "train_rank": r_train,
        "test_failures": {k: v.splitlines()[0] for k, v in test_fail.items()},
        "train_failures": {k: v.splitlines()[0] for k, v in train_fail.items()},
    }
    extra = []
    if dump is not None and (dump / "meta.npz").is_file():
        meta = dict(np.load(dump / "meta.npz"))
        g_ref = np.load(dump / f"{REF}.npy")
        boots = {}
        for f in sorted(dump.glob("*.npy")):
            if f.stem != REF:
                boots[f.stem] = bootstrap(meta, np.load(f), g_ref)
        summary["paired_vs_reference_test"] = boots
        tgt, cfg, pid, t = meta["target"], meta["cfg"], meta["pid"], meta["t"]
        e_r = np.minimum(np.abs(g_ref - tgt), 4.0)
        e_c = np.minimum(np.abs(cfg - tgt), 4.0)
        e_0 = np.minimum(np.abs(tgt), 4.0)
        summary["reference_detail_test"] = {
            "mean_abs_target": float(np.abs(tgt).mean()),
            "frac_states_clipped_reference": float(np.mean(np.abs(g_ref - tgt) >= 4.0)),
            "per_problem_mae_reg": [float(e_r[pid == p].mean()) for p in np.unique(pid)],
            "per_problem_mae_cfg": [float(e_c[pid == p].mean()) for p in np.unique(pid)],
            "per_problem_mae_zero": [float(e_0[pid == p].mean()) for p in np.unique(pid)],
            "per_problem_win_reg_vs_cfg": [float(np.mean(e_r[pid == p] < e_c[pid == p])) for p in np.unique(pid)],
            "per_timestep_win_reg_vs_cfg": {int(s): float(np.mean(e_r[t == s] < e_c[t == s])) for s in np.unique(t)},
            "per_timestep_mae_reg": {int(s): float(e_r[t == s].mean()) for s in np.unique(t)},
            "per_timestep_mae_cfg": {int(s): float(e_c[t == s].mean()) for s in np.unique(t)},
        }
        d = summary["reference_detail_test"]
        extra.append("\n## Reference detail (hidden test)\n")
        extra.append(f"mean |g*| = {d['mean_abs_target']:.4f}; reference states clipped: {d['frac_states_clipped_reference']:.4%}\n")
        extra.append("| problem | MAE REG | MAE CFG | MAE zero | REG win vs CFG |\n|---|---:|---:|---:|---:|\n")
        for i, (a, b, c, w) in enumerate(zip(d["per_problem_mae_reg"], d["per_problem_mae_cfg"], d["per_problem_mae_zero"], d["per_problem_win_reg_vs_cfg"])):
            extra.append(f"| {i} | {fmt(a)} | {fmt(b)} | {fmt(c)} | {fmt(w)} |\n")
        extra.append("\n| t | MAE REG | MAE CFG | REG win vs CFG |\n|---:|---:|---:|---:|\n")
        for s in sorted(d["per_timestep_win_reg_vs_cfg"]):
            extra.append(f"| {s} | {fmt(d['per_timestep_mae_reg'][s])} | {fmt(d['per_timestep_mae_cfg'][s])} | {fmt(d['per_timestep_win_reg_vs_cfg'][s])} |\n")
        extra.append("\n## Paired comparison vs reference (hidden test, cluster bootstrap over problems and states)\n")
        extra.append("| method | dMAE | 95% CI | dWin | 95% CI | problems improved | states closer |\n|---|---:|---|---:|---|---:|---:|\n")
        for k, b in boots.items():
            extra.append(
                f"| {k} | {b['delta_mae']:+.4f} | [{b['delta_mae_ci95'][0]:+.4f}, {b['delta_mae_ci95'][1]:+.4f}] | "
                f"{b['delta_win']:+.4f} | [{b['delta_win_ci95'][0]:+.4f}, {b['delta_win_ci95'][1]:+.4f}] | "
                f"{b['problems_improved']}/{len(b['per_problem_delta_mae'])} | {b['states_closer_than_reference']:.3f} |\n"
            )
    metrics = list(ref_def)
    lines = ["# Hardening check summary\n\n", "## Hidden test (cohort run)\n\n"]
    lines.append("| method | " + " | ".join(metrics) + " | rank vs ref | status |\n")
    lines.append("|---|" + "---:|" * len(metrics) + "---:|---|\n")
    for meth in sorted(test, key=lambda m: -r_test[m]):
        status = "FAILED: " + summary["test_failures"][meth] if meth in summary["test_failures"] else "ok"
        lines.append(f"| {meth} | " + " | ".join(fmt(test[meth][m]) for m in metrics) + f" | {r_test[meth]:+.3f} | {status} |\n")
    lines.append(f"\nRank normalisation (population std over honest cohort {cohort}): " + ", ".join(f"{m}={v:.4f}" for m, v in sig_test.items()) + "\n")
    lines.append("\n## Visible train (cohort run)\n\n")
    lines.append("| method | " + " | ".join(metrics) + " | rank vs ref |\n")
    lines.append("|---|" + "---:|" * len(metrics) + "---:|\n")
    for meth in sorted(train, key=lambda m: -r_train[m]):
        lines.append(f"| {meth} | " + " | ".join(fmt(train[meth][m]) for m in metrics) + f" | {r_train[meth]:+.3f} |\n")
    lines += extra
    (RES / "SUMMARY.md").write_text("".join(lines))
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print("".join(lines))


if __name__ == "__main__":
    main()
