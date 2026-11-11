# Evaluation

## What you submit

A guidance rule: a directory `outputs/guidance_rules/<method_name>/` containing `guidance.py` (plus any small files it reads from its own directory). Every directory under `outputs/guidance_rules/` is scored as a method, including directories without a valid rule, which get the worst scores. Nothing else you write is read: the evaluators never trust numbers, flags or claims produced by a method.

```python
def guidance(x, t, query, schedule):
    """
    x        float64 array (B,)  noisy states x_t
    t        int64 array (B,)    1-indexed time steps, 1..20 (t=1 is the last denoising step)
    query    query(model, xq, tq, deriv=False)
               model "cond" or "uncond"; xq float array (B,); tq int or int array (B,) in 1..20
               -> eps (B,)            or (eps, d eps/dx) (B,), (B,) when deriv=True
    schedule dict: "T" and float64 arrays "betas", "alphas", "alpha_bars",
             "posterior_variance" (index t-1)
    returns  g, float array (B,)    guided prediction = eps_cond(x_t, t) + g
    """
```

Rules for the rule:
- **The only access to the denoisers is `query`.** The rule runs in a separate process and does not hold the networks.
- **Cost:** each `query` call costs 1 unit per state, or 2 with `deriv=True`. The budget is **8 units per state**; exceeding it fails the method.
- **Per-state:** the output for a state must not depend on which other states share its batch. The evaluator scores every state twice, under two different random batchings; any difference beyond 1e-5 fails the method. Rules must be deterministic.
- **Failure:** a crash, timeout, wrong output shape, non-finite output, query with `t` outside 1..20, budget overrun or batching dependence all fail the method. A failed method receives the worst value of every metric.

The reference CFG direction at guidance scale 1 is `query("cond") - query("uncond")` (2 units).

## Metrics (experiment `toy_1d_guidance_approximation`)

All metrics are computed by the evaluator from the raw guidance values, against an optimal-target estimate it holds itself. That estimate uses 1,000 stochastic denoising completions per state.

| metric | direction | meaning |
|---|---|---|
| `guidance_mae` | lower | mean \|g - g*\| over all evaluation states; per-state error clipped at 4 |
| `win_rate` | higher | fraction of states where the rule is closer to g* than vanilla CFG (scale 1, computed by the evaluator); ties count 1/2 |
| `worst_timestep_win_rate` | higher | minimum over the 20 time steps of the per-step win rate |
| `worst_problem_mae` | lower | largest per-problem MAE (robustness across mixtures and denoisers) |
| `oracle_cost_per_state` | lower | mean denoiser units spent per state, as counted by the evaluator |

Ranking uses the coefficients in `scoring/reference.json`.

## Slices

- **Paper** (`bash scripts/evaluate.sh` → `scoring/scores.json`): the paper's 1D mixtures, with denoisers trained by the paper's recipe and with the authors' released denoisers.
- **Train / visible** (`bash scripts/evaluate_train.sh` → `scoring/scores_train.json`): the two paper problems plus three public paper-like mixtures. Denoisers, states and targets are in `data/toy_1d_train/`, so you can develop against them.
- **Test / hidden** (`bash scripts/evaluate_test.sh` → `scoring/scores_test.json`): hidden paper-like mixtures with their own denoisers and hidden evaluation states. It covers all 20 time steps, with x_t uniform in [-1, 2].

All three scripts take no arguments and score every rule on disk.

## Baselines

- `vanilla_cfg`: classifier-free guidance at scale 1 (2 units).
- `no_guidance`: g = 0 (0 units).

The comparison direction used by `win_rate` is fixed inside the evaluator and cannot be changed by a method.
