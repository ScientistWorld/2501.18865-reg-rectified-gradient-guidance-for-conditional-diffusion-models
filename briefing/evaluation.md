# Evaluation

## Metrics

- `guidance_mae`: mean absolute error between a candidate guidance direction and the optimal guidance direction on a fixed 1D grid. Lower is better.
- `win_rate`: fraction of grid/time-step cases where a candidate guidance direction has lower absolute error than the competing reference guidance direction. Higher is better.
- `relative_mae_reduction`: relative reduction in guidance error compared with the reference guidance direction. Higher is better.
- `additional_vjp_per_guided_step`: additional vector-Jacobian products required for each guided denoising step. Lower is better and captures the paper's overhead constraint in a machine-independent way.

## Baselines and Targets

The primary baseline is standard classifier-free guidance on the same trained 1D diffusion model. The paper's low-dimensional experiments report that the proposed enhanced guidance wins more often than standard guidance when compared to an optimal target; Table 1 reports win rates above 50% across all 2D time-step/class settings. The reproduction target is to show the same effect on the paper's 1D synthetic mixture setting.

## Evaluation Protocol

Run the method to train conditional and unconditional 1D denoisers on the fixed synthetic Gaussian mixtures, then evaluate guidance directions on the fixed grid `[-1, 2]` at time steps 4, 8, 12, 16, and 20. Evaluation reads `outputs/toy_1d/results.json` and writes `scoring/scores.json`. Future methods must use the same output contract so `scripts/evaluate.sh` can score them without importing their code.
