# Progress

## What Works

Implemented a self-contained 1D synthetic reproduction of REG for classifier-free guidance. The method trains conditional and unconditional epsilon-prediction DDPM denoisers on the paper's Gaussian mixture setup, computes vanilla CFG and REG guidance adjustments, and compares both with a deterministic joint-scaling guidance target obtained by differentiating through the remaining denoising chain.

## Results

Earlier CPU job `026b8819-b0f` completed successfully and wrote `scoring/scores.json` using a one-step deterministic target.

| Method | Guidance MAE | Win Rate | Additional VJP / guided step |
|---|---:|---:|---:|
| Vanilla CFG | 2.4352 | 0.3076 | 0.0 |
| REG CFG | 2.1417 | 0.6924 | 1.0 |

REG reduced guidance MAE by 12.05% relative to vanilla CFG and won 69.24% of comparisons against the deterministic optimal guidance target. The job also ran `python validate.py --compare`, which passed all schema, separation, and comparison checks.

The method code has since been updated to use the full deterministic foresight chain for the target. A new job is being submitted to refresh these numbers before claiming `core_claim`.

## Remaining

The next step is to run the corrected method and evaluator in the compute container. If REG remains closer to the joint-scaling target than vanilla CFG, the reusable environment reaches `core_claim`. Future extensions would add 2D synthetic experiments, image-generation experiments from the paper, more CFG weights, or additional low-dimensional settings.

## Issues

The full ImageNet and COCO experiments require large pretrained diffusion models and many generated images. This reproduction targets the paper's low-dimensional synthetic claim because it implements the same REG computation and is feasible within the available budget.

## Deviations from Paper

The paper's 1D experiment evaluates the optimal expected reward by sampling 100 denoising-chain completions at each observed `x_t`. This reproduction uses the deterministic-sampler target from the paper's derivation and differentiates through the remaining denoising chain to keep evaluation affordable while preserving the REG computation in Eq. 21. The full ImageNet and COCO tables are not part of this minimum run because they require large pretrained generators and many generated images.
