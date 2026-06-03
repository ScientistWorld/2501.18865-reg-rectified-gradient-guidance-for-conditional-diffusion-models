# Evaluation Targets

## Primary: toy_1d_guidance_approximation

Tests whether the candidate guidance direction is closer than standard guidance to the optimal guidance target on the fixed 1D synthetic benchmark.
- **Metric**: `win_rate` (higher is better)
- **Metric**: `guidance_mae` (lower is better)
- **Run**: `bash scripts/evaluate.sh outputs/toy_1d/results.json`

## Constraint: toy_1d_guidance_approximation

Tests whether the improvement is achieved without unbounded extra per-step computation.
- **Metric**: `additional_vjp_per_guided_step` (lower is better)
- **Run**: `bash scripts/evaluate.sh outputs/toy_1d/results.json`
