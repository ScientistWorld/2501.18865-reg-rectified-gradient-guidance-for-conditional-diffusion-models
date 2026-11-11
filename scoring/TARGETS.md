# Evaluation Targets

All targets belong to the single experiment `toy_1d_guidance_approximation`. Every metric is computed by the evaluator from the raw guidance values of each rule in `outputs/guidance_rules/`.

## Primary: guidance approximation vs. classifier-free guidance

The paper's Table 1 / Figure 1 claim: the rule is closer than vanilla CFG to the optimal foresight guidance.
- **Metric**: `win_rate` (higher is better; primary)
- **Metric**: `guidance_mae` (lower is better)
- **Run**: `bash scripts/evaluate_train.sh` (visible), `bash scripts/evaluate_test.sh` (hidden), `bash scripts/evaluate.sh` (paper problems only)

## Robustness: every noise level, every problem

Table 1 reports the win ratio per time step. The hidden slice uses mixtures and denoisers the rule has never seen.
- **Metric**: `worst_timestep_win_rate` (higher is better)
- **Metric**: `worst_problem_mae` (lower is better)
- **Run**: same scripts

## Constraint: no foresight, bounded overhead

The paper's method adds one derivative evaluation per guided step. The optimal target needs the whole future chain.
- **Metric**: `oracle_cost_per_state` (lower is better; denoiser units counted by the evaluator; hard cap of 8 units per state, exceeding it fails the method)
- **Run**: same scripts
