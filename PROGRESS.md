# Progress

## What Works

Implemented a self-contained 1D synthetic reproduction of REG for classifier-free guidance. The method trains conditional and unconditional epsilon-prediction DDPM denoisers on the paper's Gaussian mixture setup, computes vanilla CFG and REG guidance adjustments, and compares both with a stochastic expected-reward guidance target obtained by differentiating through 100 sampled denoising-chain completions at each observed state.

The canonical evaluator now scores the complete metric family declared in `scoring/reference.json`, and `scripts/run.sh` regenerates the ignored method artifact before scoring it. This keeps the handoff path usable from a fresh clone.

## Results

CPU job `d504d843-623` completed successfully and wrote `scoring/scores.json` using the stochastic expected-reward target.

| Method | Guidance MAE | Win Rate | Additional VJP / guided step |
|---|---:|---:|---:|
| Vanilla CFG | 0.7692 | 0.1535 | 0.0 |
| REG CFG | 0.3720 | 0.8465 | 1.0 |

REG reduced guidance MAE by 51.64% relative to vanilla CFG and won 84.65% of comparisons against the stochastic expected-reward guidance target. The job also ran `python validate.py --compare`, which passed all schema, separation, and comparison checks.

On 2026-06-04, the canonical `scores.json` evaluator was repaired and locally re-run on the existing artifact. The repaired canonical scores include early/late step metrics and pass `python validate.py --compare`; a clean CPU regeneration job is pending to verify the updated `scripts/run.sh`.

## Remaining

The reusable environment has reached `core_claim`. Future extensions would add 2D synthetic experiments, image-generation experiments from the paper, more CFG weights, or additional low-dimensional settings.

## Issues

The full ImageNet and COCO experiments require large pretrained diffusion models and many generated images. This reproduction targets the paper's low-dimensional synthetic claim because it implements the same REG computation and is feasible within the available budget.

A packaging audit found that the prior `scripts/run.sh` assumed `outputs/toy_1d` already existed even though `outputs/` is gitignored, and that the canonical evaluator omitted some declared metrics. Both issues have been patched.

## Deviations from Paper

The minimum reproduction focuses on the paper's 1D Gaussian-mixture setting rather than the full ImageNet and COCO tables. The implemented evaluation now follows the paper's low-dimensional expected-reward protocol by sampling 100 denoising-chain completions per grid point. The larger image-generation experiments are omitted because they require large pretrained generators and many generated images beyond the current budget.
