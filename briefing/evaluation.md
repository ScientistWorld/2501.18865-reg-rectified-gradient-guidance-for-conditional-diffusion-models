# Evaluation

## Metrics

- `guidance_mae`: mean absolute error between a candidate guidance direction and the optimal guidance direction on a fixed 1D grid. Lower is better.
- `win_rate`: fraction of grid/time-step cases where a candidate guidance direction has lower absolute error than the competing reference guidance direction. Higher is better.
- `relative_mae_reduction`: relative reduction in guidance error compared with the reference guidance direction. Higher is better.
- `additional_vjp_per_guided_step`: additional vector-Jacobian products required for each guided denoising step. Lower is better and captures the paper's overhead constraint in a machine-independent way.

## Baselines and Targets

The primary comparison is against the reference guidance direction on the same fixed 1D benchmark. The paper's low-dimensional experiments report that an improved guidance rule wins more often than the reference direction when compared to an optimal target; Table 1 reports win rates above 50% across all 2D time-step/class settings. The reproduction target is to show the same effect on the paper's 1D synthetic mixture setting.

## Evaluation Protocol

Produce an artifact at `outputs/toy_1d/results.json` containing per-time-step guidance-error records for the fixed synthetic Gaussian mixtures, grid `[-1, 2]`, and time steps 4, 8, 12, 16, and 20. Evaluation reads that artifact and writes `scoring/scores.json`. Future methods must use the same output contract so `scripts/evaluate.sh` can score them without importing their code.
