# Progress

## What Works

Implemented a self-contained 1D synthetic reproduction of REG for classifier-free guidance. The method trains conditional and unconditional epsilon-prediction DDPM denoisers on the paper's Gaussian mixture setup, computes vanilla CFG and REG guidance adjustments, and compares both with the deterministic optimal joint-scaling guidance target.

## Results

CPU job `026b8819-b0f` completed successfully and wrote `scoring/scores.json`.

| Method | Guidance MAE | Win Rate | Additional VJP / guided step |
|---|---:|---:|---:|
| Vanilla CFG | 2.4352 | 0.3076 | 0.0 |
| REG CFG | 2.1417 | 0.6924 | 1.0 |

REG reduced guidance MAE by 12.05% relative to vanilla CFG and won 69.24% of comparisons against the deterministic optimal guidance target. The job also ran `python validate.py --compare`, which passed all schema, separation, and comparison checks.

## Remaining

The reusable environment has reached `core_claim`. Future extensions would add image-generation experiments from the paper, more CFG weights, or additional low-dimensional settings to move beyond the minimum.

## Issues

The full ImageNet and COCO experiments require large pretrained diffusion models and many generated images. This reproduction targets the paper's low-dimensional synthetic claim because it implements the same REG computation and is feasible within the available budget.

## Deviations from Paper

The paper's 1D experiment evaluates the optimal expected reward by sampling 100 denoising-chain completions at each observed `x_t`. This reproduction uses the deterministic-sampler optimal target from the paper's derivation, `grad log R0(x0_hat)`, to keep evaluation affordable while preserving the REG computation in Eq. 21. The full image-generation tables are documented in `reference.json` as future zero-weight targets but are not part of the first submitted run.
