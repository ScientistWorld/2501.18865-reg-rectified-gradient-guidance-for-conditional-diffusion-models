"""Audit-only interpretation; never imported by any method or shortcut."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "redteam"
EXP = "toy_1d_guidance_approximation"


def read(path):
    return json.loads(path.read_text())


def results(path):
    return read(path)["experiments"][EXP]["results"]


def main():
    definition = read(HERE / "evidence/reference_definition.json")["experiments"][EXP]
    metric_defs = definition["metrics"]
    reference = results(HERE / "evidence/original_scores_test.json")["reg_cfg"]
    runs = {}
    for path in sorted(HERE.glob("*/run_status.json")):
        status = read(path)
        score_path = path.parent / "scores_test.json"
        output = results(score_path) if score_path.exists() else None
        runs[path.parent.name] = {"status": status, "results": output}

    honest_names = [name for name in runs if name.startswith(("reference_seed", "better_", "weaker_"))]
    honest = [runs[name]["results"]["reg_cfg"] for name in honest_names
              if runs[name]["status"]["test_returncode"] == 0]
    scales = {metric: statistics.pstdev([r[metric] for r in honest if metric in r])
              for metric in metric_defs}

    def rank(metrics, denominators):
        return definition["weight"] * sum(
            abs(spec["coefficient"]) * (1 if spec["higher_is_better"] else -1)
            * (metrics[metric] - reference[metric]) / (denominators[metric] or 1.0)
            for metric, spec in metric_defs.items()
            if metric in metrics and math.isfinite(metrics[metric]))

    pooled = [metrics for run in runs.values() if run["status"]["test_returncode"] == 0
              for metrics in (run["results"] or {}).values()]
    pooled_scales = {metric: statistics.pstdev([r[metric] for r in pooled if metric in r and math.isfinite(r[metric])])
                     for metric in metric_defs}
    for name, run in runs.items():
        if run["status"]["test_returncode"] != 0:
            run["audit_rank"] = None
            run["interpretation"] = "failed subprocess; evaluator wrote no worst-value score"
            continue
        metrics = (run["results"] or {}).get("reg_cfg", {})
        # Empty test results demonstrate the supplied rank's missing-value rule.
        run["audit_rank"] = rank(metrics, scales)
        run["pooled_audit_rank"] = rank(metrics, pooled_scales)
        deltas = {metric: (1 if spec["higher_is_better"] else -1) * (metrics[metric] - reference[metric])
                  for metric, spec in metric_defs.items() if metric in metrics}
        run["directed_deltas"] = deltas
        if not metrics:
            run["interpretation"] = "no candidate metrics: rank skips all values, tying reference at zero"
        elif all(x >= -1e-9 for x in deltas.values()) and any(x > 1e-9 for x in deltas.values()):
            run["interpretation"] = "Pareto improvement: outranks reference for any positive finite metric scales"
        elif all(abs(x) < 1e-6 for x in deltas.values()):
            run["interpretation"] = "tie within numerical precision"
        else:
            run["interpretation"] = "tradeoff: aggregate depends on cohort normalization"

    seed_metrics = [runs[f"reference_seed{seed}"]["results"]["reg_cfg"] for seed in range(3)]
    seed_ranks = [runs[f"reference_seed{seed}"]["audit_rank"] for seed in range(3)]
    spread = {metric: {"min": min(r[metric] for r in seed_metrics),
                       "max": max(r[metric] for r in seed_metrics),
                       "range": max(r[metric] for r in seed_metrics) - min(r[metric] for r in seed_metrics),
                       "std_population": statistics.pstdev(r[metric] for r in seed_metrics)}
              for metric in metric_defs}
    bounds = {metric: (0.0 if not spec["higher_is_better"] else 1.0)
              for metric, spec in metric_defs.items()}
    saturated = [metric for metric, value in reference.items() if metric in bounds and value == bounds[metric]]
    summary = {"reference_metrics": reference, "metric_definitions": metric_defs,
               "normalization": "population std across eleven genuine quality/reference variants only; no shortcuts",
               "honest_normalization_cohort": honest_names, "metric_scales": scales,
               "pooled_normalization": "all successful output method rows including impossible claims; illustrative only",
               "pooled_metric_scales": pooled_scales,
               "reference_rank": 0.0, "seed_ranks": seed_ranks,
               "seed_rank_range": max(seed_ranks) - min(seed_ranks),
               "seed_metric_spread": spread, "reference_at_best_bound": saturated,
               "runs": runs}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| Run | Test exit | MAE | Win rate | Rel. reduction | VJPs | Audit rank |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        metric = (run["results"] or {}).get("reg_cfg", {})
        fmt = lambda k: f"{metric[k]:.6f}" if k in metric else "missing"
        rank_text = "failed" if run["audit_rank"] is None else f"{run['audit_rank']:+.6f}"
        lines.append(f"| {name} | {run['status']['test_returncode']} | {fmt('guidance_mae')} | {fmt('win_rate')} | {fmt('relative_mae_reduction')} | {fmt('additional_vjp_per_guided_step')} | {rank_text} |")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
