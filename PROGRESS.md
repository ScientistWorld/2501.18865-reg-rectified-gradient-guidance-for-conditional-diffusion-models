# Progress

## What Works

Implemented a self-contained 1D synthetic reproduction of REG for classifier-free guidance. The method trains conditional and unconditional epsilon-prediction DDPM denoisers on the paper's Gaussian mixture setup, computes vanilla CFG and REG guidance adjustments, and compares both with a deterministic joint-scaling guidance target obtained by differentiating through the remaining denoising chain.

## Results

Corrected CPU job `df24a0a7-242` completed successfully and wrote `scoring/scores.json` using the full deterministic denoising-chain target.

| Method | Guidance MAE | Win Rate | Additional VJP / guided step |
|---|---:|---:|---:|
| Vanilla CFG | 1.2693 | 0.3860 | 0.0 |
| REG CFG | 0.9484 | 0.6140 | 1.0 |

REG reduced guidance MAE by 25.28% relative to vanilla CFG and won 61.40% of comparisons against the deterministic joint-scaling guidance target. The job also ran `python validate.py --compare`, which passed all schema, separation, and comparison checks.

## Remaining

The reusable environment has reached `core_claim`. Future extensions would add 2D synthetic experiments, image-generation experiments from the paper, more CFG weights, or additional low-dimensional settings.

## Issues

The full ImageNet and COCO experiments require large pretrained diffusion models and many generated images. This reproduction targets the paper's low-dimensional synthetic claim because it implements the same REG computation and is feasible within the available budget.

## Deviations from Paper

The paper's 1D experiment evaluates the optimal expected reward by sampling 100 denoising-chain completions at each observed `x_t`. This reproduction uses the deterministic-sampler target from the paper's derivation and differentiates through the remaining denoising chain to keep evaluation affordable while preserving the REG computation in Eq. 21. The full ImageNet and COCO tables are not part of this minimum run because they require large pretrained generators and many generated images.
